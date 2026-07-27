"""Stage 3 per-patient adaptation — freeze backbone, train head on calibration data.

Protocol
--------
For each held-out test subject:
  1. Load epochs with the montage read from configs/dataset/*.yaml — the same
     config the training path uses — and assert it matches the montage recorded
     for the checkpoint.
  2. Apply the cross-subject preprocessor (fit on training data, loaded from
     the fold's preprocessor.pkl).
  3. Draw the calibration set with a stratified random split
     (train_test_split(train_size=N, stratify=y, random_state=seed)); the rest is
     the evaluation set. Note: the module docstring previously described a
     "first N epochs" temporal split, which the code has not done since bf23ef9.
  4. Clone the pre-trained model, freeze the backbone, keep the pretrained head.
  5. Train only the head on the calibration set.
  6. Report AUC vs calibration size, with the checkpoint and montage recorded in
     every output row.

Checkpoints are named explicitly. There is no best-of-N discovery; see
docs/AUDIT.md §0.4.

Usage
-----
    python scripts/adapt_stage3.py --paradigm mi --folds version_28
    python scripts/adapt_stage3.py --paradigm mi --folds version_27 version_28 version_29 version_30 --fold-mode average
    python scripts/adapt_stage3.py --paradigm p300
"""
from __future__ import annotations

import argparse
import copy
import glob
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Checkpoint discovery has moved to src/models/checkpoints.py, which requires folds
# to be named explicitly. The previous _find_best_checkpoint() ranked whatever was on
# disk by the val AUC in the filename; see docs/AUDIT.md §0.4 for why that made every
# MI result a function of the state of lightning_logs/ at run time.


# ---------------------------------------------------------------------------
# Model loading + head reset
# ---------------------------------------------------------------------------

def _load_model_frozen(
    ckpt_path: Path,
    paradigm: str,
    n_channels: int,
    n_times: int,
    reinit_head: bool = True,
) -> nn.Module:
    """Load checkpoint, freeze backbone. Re-initialises head only when reinit_head=True.

    reinit_head=False → pretrained head intact (baseline and warm-start adapted path).
    reinit_head=True  → fresh random head (cold-start ablation only).

    n_channels/n_times come from the dataset config, not from literals — the two
    hardcoded shape tables this function used to carry were a second place for the
    montage to drift out of sync with training (docs/AUDIT.md §0.1).
    """
    from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
    from src.training.lit_module import LitEEG

    backbone = BackboneEncoder(n_channels=n_channels, n_times=n_times)
    head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
    pretrained = EEGDecoder(backbone, head)

    class _Cfg:
        def get(self, k, d=None): return d

    lit = LitEEG.load_from_checkpoint(
        str(ckpt_path), model=pretrained, cfg=_Cfg(), map_location="cpu"
    )

    # Rebuild as stage3_adapted so train() override keeps backbone in eval
    from src.models.backbone import EEGDecoder as _EEGDecoder
    adapted = _EEGDecoder(
        copy.deepcopy(lit.model.backbone),
        copy.deepcopy(lit.model.head),
        stage="stage3_adapted",
    )
    if reinit_head:
        nn.init.xavier_uniform_(adapted.head.linear.weight)
        nn.init.zeros_(adapted.head.linear.bias)

    return adapted


# ---------------------------------------------------------------------------
# Head-only training
# ---------------------------------------------------------------------------

def _adapt_head(
    model: nn.Module,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> nn.Module:
    """Train only the decoder head on calibration data. Mutates and returns model."""
    from src.training.losses import weighted_bce

    model.train()
    # Only head parameters have requires_grad=True
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=1e-3)

    X_t = torch.from_numpy(X_calib).float()
    y_t = torch.from_numpy(y_calib).float()
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    n_pos = int(y_calib.sum())
    n_neg = len(y_calib) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)])

    for _ in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = weighted_bce(logits, yb, pos_weight=pw)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _epoch_auc(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    probs = torch.sigmoid(model(torch.from_numpy(X).float()).squeeze(-1)).numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs))


# ---------------------------------------------------------------------------
# EA reference scope, feature cache, and the per-condition runner
# ---------------------------------------------------------------------------

def _apply_ea(X: np.ndarray, fit_idx: np.ndarray | None) -> np.ndarray:
    """Fit a Euclidean Alignment reference and apply it to all of X.

    fit_idx=None  → fit on every epoch (transductive/offline EA over the whole
                    session). This is what the wrapper used to do internally and is
                    what --ea-ref session reproduces.
    fit_idx=array → fit on those epochs only (--ea-ref calibration), which is the
                    only variant achievable at deployment time (§11.1, §7.3).
    """
    from src.preprocessing.alignment import EuclideanAligner

    aligner = EuclideanAligner()
    aligner.fit(X if fit_idx is None else X[fit_idx])
    return aligner.transform(X)


@torch.no_grad()
def _features(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> torch.Tensor:
    """Frozen-backbone feature vectors for X. Deterministic: backbone is in eval."""
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).float()
        out.append(model.backbone(xb))
    return torch.cat(out, dim=0)


@torch.no_grad()
def _head_auc(head: nn.Module, feats: torch.Tensor, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    probs = torch.sigmoid(head(feats).squeeze(-1)).numpy()
    return float(roc_auc_score(y, probs))


def _adapt_head_on_features(
    head: nn.Module,
    feats: torch.Tensor,
    y: np.ndarray,
    n_epochs: int,
    lr: float,
    batch_size: int = 32,
    seed: int = 0,
) -> nn.Module:
    """Train the head on precomputed features. Equivalent to _adapt_head on epochs."""
    from src.training.losses import weighted_bce

    torch.manual_seed(seed)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)

    y_t = torch.from_numpy(np.asarray(y)).float()
    ds = TensorDataset(feats, y_t)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    n_pos = int(np.sum(y == 1))
    n_neg = len(y) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)])

    for _ in range(n_epochs):
        for fb, yb in loader:
            optimizer.zero_grad()
            loss = weighted_bce(head(fb).squeeze(-1), yb, pos_weight=pw)
            loss.backward()
            optimizer.step()

    head.eval()
    return head


def _run_one(
    *, model, pp, X_noea, y_all, order, source, arm, n_calib, seed, ea_ref,
    purge_k, feats_session, n_adapt_epochs, adapt_lr,
) -> dict | None:
    """One (arm, size, seed, ea_ref) condition. Returns a result row, or None if
    the condition cannot be constructed at all."""
    from src.datasets.splits import make_calibration_split

    n = len(y_all)
    if n_calib >= n:
        log.warning(
            "  arm=%s N=%d: subject has only %d epochs — condition does not exist.",
            arm, n_calib, n,
        )
        return None

    try:
        split = make_calibration_split(
            arm, y_all, n_calib, seed, order, source, purge_k=purge_k
        )
    except ValueError as e:
        log.warning("  arm=%s N=%d: cannot build split — %s", arm, n_calib, e)
        return None

    desc = split.describe(y_all, source)
    y_calib = np.asarray(y_all)[split.calib_idx]
    y_eval = np.asarray(y_all)[split.eval_idx]

    # Degenerate conditions are reported, not silently skipped: the row is emitted
    # with a status and NaN metrics so the table has no unexplained gaps.
    status = "ok"
    if len(split.eval_idx) == 0:
        status = "degenerate_empty_eval"
    elif len(np.unique(y_eval)) < 2:
        status = "degenerate_eval_single_class"
    elif len(np.unique(y_calib)) < 2:
        status = "degenerate_calib_single_class"

    if status != "ok":
        log.warning(
            "  arm=%s N=%d seed=%d: %s (n_eval=%d, eval_pos_rate=%.3f, "
            "eval_baseline_frac=%.3f) — reported, not skipped.",
            arm, n_calib, seed, status, desc["n_eval"], desc["eval_pos_rate"],
            desc["eval_baseline_frac"],
        )
        return {**desc, "baseline_auc": float("nan"), "adapted_auc": float("nan"),
                "delta_auc": float("nan"), "status": status}

    if ea_ref == "session":
        feats = feats_session
    else:
        X_ea = _apply_ea(X_noea, fit_idx=split.calib_idx)
        feats = _features(model, pp.transform(X_ea))

    f_calib = feats[split.calib_idx]
    f_eval = feats[split.eval_idx]

    # Baseline: pretrained head, same eval split as the adapted model so both
    # share the evaluation denominator.
    base_head = copy.deepcopy(model.head)
    base_head.eval()
    baseline_auc = _head_auc(base_head, f_eval, y_eval)

    # Adapted: warm-started from the pretrained head, trained on calibration only.
    adapted_head = copy.deepcopy(model.head)
    _adapt_head_on_features(
        adapted_head, f_calib, y_calib, n_epochs=n_adapt_epochs, lr=adapt_lr, seed=seed
    )
    adapted_auc = _head_auc(adapted_head, f_eval, y_eval)

    return {
        **desc,
        "baseline_auc": baseline_auc,
        "adapted_auc": adapted_auc,
        "delta_auc": adapted_auc - baseline_auc,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Main calibration-curve routine
# ---------------------------------------------------------------------------

def calibration_curve(
    paradigm: str,
    log_dir: str = "lightning_logs",
    calib_sizes: list[int] | None = None,
    n_adapt_epochs: int = 100,
    adapt_lr: float = 1e-3,
    seeds: list[int] | None = None,
    arms: list[str] | None = None,
    ea_refs: list[str] | None = None,
    purge_k: int = 5,
    folds: list[str] | None = None,
    fold_mode: str = "pinned",
    strict_montage: bool = False,
    out_path: str | None = None,
) -> list[dict]:
    from src.datasets.montage import assert_montage_matches, load_montage
    from src.datasets.registry import get_dataset
    from src.models.checkpoints import load_folds, loso_fold_for_subject, spec_from_config

    # Channel list, sfreq, band and window all come from the same config the training
    # path reads (configs/dataset/*.yaml). No second copy lives here.
    import dataclasses

    from src.datasets.splits import SPLIT_ARMS

    spec = spec_from_config(paradigm)
    n_times = int(round(spec.sfreq_target * (spec.epoch_window[1] - spec.epoch_window[0])))
    n_channels = len(spec.channels)
    log.info("Montage from config: %d channels %s", n_channels, list(spec.channels))

    seeds = list(seeds) if seeds else [42, 43, 44, 45, 46]
    arms = list(arms) if arms else list(SPLIT_ARMS)
    ea_refs = list(ea_refs) if ea_refs else ["session"]
    for a in arms:
        if a not in SPLIT_ARMS:
            raise ValueError(f"Unknown split arm {a!r}; expected one of {SPLIT_ARMS}")
    for e in ea_refs:
        if e not in ("session", "calibration"):
            raise ValueError(f"Unknown --ea-ref {e!r}")

    # EA is applied explicitly in _run_one so its reference scope is an experimental
    # variable rather than a fixed side effect of loading. The wrapper must therefore
    # not apply it: spec.euclidean_alignment is forced off here. --ea-ref session
    # reproduces the wrapper's previous behaviour exactly (fit over the whole
    # session), which tests/test_m9_ea_scope.py asserts.
    spec = dataclasses.replace(spec, euclidean_alignment=False)

    if calib_sizes is None:
        calib_sizes = [24, 48, 96, 240, 480] if paradigm == "p300" else [10, 20, 50, 100, 200]

    wrapper = get_dataset(spec)
    test_subjects = wrapper.subject_list[-10:]
    log.info("Test subjects for Stage 3: %s", test_subjects)

    # Folds are named explicitly; nothing is selected by ranking (docs/AUDIT.md §0.4).
    mi_folds = None
    if paradigm != "p300":
        mi_folds = load_folds(log_dir, folds or [], expect_channels=n_channels)
        if fold_mode == "pinned" and len(mi_folds) != 1:
            raise ValueError(
                f"fold_mode='pinned' needs exactly one --folds entry, got {len(mi_folds)}. "
                f"Use --fold-mode average to aggregate across folds."
            )

    results: list[dict] = []

    for subj in test_subjects:
        log.info("Subject %s — loading epochs…", subj)

        subj_folds = (
            [loso_fold_for_subject(log_dir, subj, strict=True)]
            if paradigm == "p300" else mi_folds
        )

        # Loaded once per subject with EA OFF; the EA reference is applied below so
        # its scope (session vs calibration) is an explicit experimental variable.
        X_noea, y_all, meta = wrapper.load_epochs([subj])
        order = np.asarray(meta["order"])
        source = np.asarray(meta["source"])
        n_epochs = len(y_all)
        log.info(
            "  subject %s: %d epochs (%d task, %d baseline), %d pos / %d neg",
            subj, n_epochs, int((source == "task").sum()), int((source == "baseline").sum()),
            int(np.sum(y_all == 1)), int(np.sum(y_all == 0)),
        )

        for ref in subj_folds:
            # The montage contract: what this checkpoint was trained on must equal
            # what we just loaded, name for name and position for position.
            montage = load_montage(ref.version_dir, spec, n_times, strict=strict_montage)
            assert_montage_matches(
                montage,
                meta["channels"],
                where=f"{ref.version} vs {paradigm} data for subject {subj}",
            )

            with open(ref.preprocessor_path, "rb") as f:
                pp = pickle.load(f)

            model = _load_model_frozen(
                ref.ckpt_path, paradigm, n_channels, n_times, reinit_head=False
            )
            model.eval()

            # --- feature cache -------------------------------------------------
            # The backbone is frozen and permanently in eval mode (EEGDecoder
            # stage="stage3_adapted"), so backbone(x) is a deterministic function of
            # x alone. Under --ea-ref session the input is identical for every arm,
            # size and seed, so features are computed once per (subject, fold) and
            # every head adaptation trains on the cached vectors. Under --ea-ref
            # calibration the EA reference depends on the split, so the input
            # changes and features must be recomputed per split.
            feats_session = None
            if "session" in ea_refs:
                X_sess = _apply_ea(X_noea, fit_idx=None)
                feats_session = _features(model, pp.transform(X_sess))

            for ea_ref in ea_refs:
                for arm in arms:
                    for n_calib in calib_sizes:
                        for sd in seeds:
                            row = _run_one(
                                model=model, pp=pp, X_noea=X_noea, y_all=y_all,
                                order=order, source=source, arm=arm, n_calib=n_calib,
                                seed=sd, ea_ref=ea_ref, purge_k=purge_k,
                                feats_session=feats_session,
                                n_adapt_epochs=n_adapt_epochs, adapt_lr=adapt_lr,
                            )
                            if row is None:
                                continue
                            row.update({
                                "subject": subj,
                                "ea_ref": ea_ref,
                                "seed": sd,
                                "calib_size": n_calib,
                                "fold": ref.version,
                                "fold_mode": fold_mode,
                                "checkpoint": str(ref.ckpt_path),
                                "checkpoint_val_auc": ref.val_auc,
                                "montage_source": montage.source,
                                "n_channels": n_channels,
                                "n_epochs_total": n_epochs,
                                "channels": "|".join(spec.channels),
                            })
                            results.append(row)
                            log.info(
                                "  subj %s | %s | %-20s | N=%3d | seed %d | ea=%s | "
                                "base=%.3f adapt=%.3f (Δ=%+.3f) n_eval=%d eval_base_frac=%.2f [%s]",
                                subj, ref.version, arm, n_calib, sd, ea_ref,
                                row["baseline_auc"], row["adapted_auc"], row["delta_auc"],
                                row["n_eval"], row["eval_baseline_frac"], row["status"],
                            )

    # Persist results so they can be re-read without re-running
    import csv
    csv_path = Path(out_path) if out_path else Path("experiments") / "stage3" / f"{paradigm}_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paradigm", "subject", "split", "calib_size", "seed", "ea_ref",
        "baseline_auc", "adapted_auc", "delta_auc",
        "n_calib", "n_eval", "calib_pos_rate", "eval_pos_rate",
        "calib_baseline_frac", "eval_baseline_frac", "n_purged", "status",
        "fold", "fold_mode", "checkpoint", "checkpoint_val_auc",
        "montage_source", "n_channels", "n_epochs_total", "channels",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({"paradigm": paradigm, **r})
    log.info("Results saved → %s", csv_path)

    try:
        import os
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "als-decode"),
            name=f"stage3_{paradigm}",
            job_type="stage3_adaptation",
            mode=os.environ.get("WANDB_MODE", "offline"),
            config={
                "paradigm": paradigm, "calib_sizes": calib_sizes, "seeds": seeds,
                "arms": arms, "ea_refs": ea_refs, "purge_k": purge_k,
                "folds": folds, "fold_mode": fold_mode,
            },
        )
        for r in results:
            wandb.log({"paradigm": paradigm, **r})
        wandb.finish()
    except ImportError:
        pass

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from src.datasets.splits import SPLIT_ARMS

    parser = argparse.ArgumentParser()
    parser.add_argument("--paradigm", choices=["mi", "p300"], required=True)
    parser.add_argument("--log-dir", default="lightning_logs")
    parser.add_argument("--calib-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--adapt-epochs", type=int, default=100)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument(
        "--split", nargs="+", dest="arms", default=None, choices=list(SPLIT_ARMS),
        help="Calibration split arm(s). Default: all five.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
        help="Every arm x size x subject runs once per seed. The seed controls the "
             "stratified draw and head-training stochasticity.",
    )
    parser.add_argument(
        "--ea-ref", nargs="+", dest="ea_refs", default=["session"],
        choices=["session", "calibration"],
        help="Euclidean Alignment reference scope. session: fit over the subject's "
             "whole session including evaluation epochs (transductive; reproduces "
             "prior behaviour). calibration: fit on calibration epochs only — the "
             "only variant achievable at deployment (design.md 11.1, 7.3).",
    )
    parser.add_argument(
        "--purge-k", type=int, default=5,
        help="purged_stratified: drop evaluation epochs within +/-k of any "
             "calibration epoch in chronological order.",
    )
    parser.add_argument(
        "--folds", nargs="+", default=None,
        help="Explicit MI fold version dirs, e.g. --folds version_28. Required for "
             "--paradigm mi; P300 resolves folds per subject from val_subjects.json.",
    )
    parser.add_argument(
        "--fold-mode", choices=["pinned", "average"], default="pinned",
        help="pinned: exactly one fold, recorded in every row. average: run the full "
             "adaptation separately against EACH named fold and write one row per "
             "fold; delta_auc is then averaged across folds per (subject, arm, size, "
             "seed) in scripts/analyze_stage3.py. This is NOT a prediction ensemble — "
             "no logits or probabilities are combined, and each fold's baseline and "
             "adapted models are evaluated entirely on their own. Best-of-N selection "
             "is not offered: it is selection optimism in the baseline "
             "(docs/AUDIT.md 0.4).",
    )
    parser.add_argument(
        "--strict-montage", action="store_true",
        help="Refuse checkpoints that carry no recorded montage.json instead of "
             "inferring the montage from the training config.",
    )
    parser.add_argument("--out", default=None, help="Override output CSV path.")
    args = parser.parse_args()

    results = calibration_curve(
        paradigm=args.paradigm,
        log_dir=args.log_dir,
        calib_sizes=args.calib_sizes,
        n_adapt_epochs=args.adapt_epochs,
        adapt_lr=args.adapt_lr,
        seeds=args.seeds,
        arms=args.arms,
        ea_refs=args.ea_refs,
        purge_k=args.purge_k,
        folds=args.folds,
        fold_mode=args.fold_mode,
        strict_montage=args.strict_montage,
        out_path=args.out,
    )

    ok = [r for r in results if r["status"] == "ok"]
    degenerate = [r for r in results if r["status"] != "ok"]

    print(f"\n{'='*100}")
    print(f"STAGE 3 ADAPTATION — {args.paradigm.upper()}  "
          f"({len(ok)} usable rows, {len(degenerate)} degenerate)")
    print(f"{'='*100}")
    import collections
    by = collections.defaultdict(list)
    for r in ok:
        by[(r["split"], r["ea_ref"], r["calib_size"])].append(r["delta_auc"])
    print(f"{'arm':<22} {'ea_ref':<12} {'N':>6} {'mean ΔAUC':>12} {'n':>5}")
    print("-" * 62)
    for k in sorted(by):
        v = np.array(by[k], dtype=float)
        print(f"{k[0]:<22} {k[1]:<12} {k[2]:>6} {np.nanmean(v):>+12.4f} {len(v):>5}")

    if degenerate:
        print(f"\nDegenerate conditions (reported, not dropped):")
        seen = collections.Counter(
            (r["split"], r["calib_size"], r["status"]) for r in degenerate
        )
        for (arm, size, st), cnt in sorted(seen.items()):
            print(f"    {arm:<22} N={size:<5} {st}  x{cnt}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
