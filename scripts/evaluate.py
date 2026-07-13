"""Stage 1 streaming evaluation — loads best fold checkpoints and reports TPR@FPR table.

Usage:
    python scripts/evaluate.py                          # uses lightning_logs/version_*/
    python scripts/evaluate.py --log-dir lightning_logs --fpr 0.01 0.05
"""
from __future__ import annotations

import argparse
import glob
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _find_best_checkpoints(log_dir: str) -> list[Path]:
    # Checkpoints are named best-epoch=NNN-val/roc_auc=X.ckpt — the metric
    # name contains a slash so they live one level deeper than the checkpoints dir.
    pattern = str(Path(log_dir) / "version_*" / "checkpoints" / "best-epoch=*" / "*.ckpt")
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No best checkpoints found under {log_dir}")
    # Skip the 3-epoch smoke run (version_0 has val AUC ~0.49)
    ckpts = [c for c in ckpts if "0.49" not in c]
    return [Path(c) for c in ckpts]


def _load_model(ckpt_path: Path) -> torch.nn.Module:
    from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
    from src.training.lit_module import LitEEG

    backbone = BackboneEncoder(n_channels=17, n_times=320)
    head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
    model = EEGDecoder(backbone, head)

    class _Cfg:
        def get(self, k, d=None):
            return d

    lit = LitEEG.load_from_checkpoint(
        str(ckpt_path), model=model, cfg=_Cfg(), map_location="cpu"
    )
    lit.eval()
    return lit.model


@torch.no_grad()
def _score_epochs(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    """Return sigmoid probabilities for (N, C, T) array."""
    x = torch.from_numpy(X).float()
    logits = model(x).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()


def _build_model_fn(model: torch.nn.Module):
    @torch.no_grad()
    def fn(window: np.ndarray) -> float:
        """window: (C, T) → scalar probability."""
        x = torch.from_numpy(window[np.newaxis]).float()
        return float(torch.sigmoid(model(x).squeeze()))
    return fn


def evaluate(
    log_dir: str = "lightning_logs",
    fpr_targets: list[float] | None = None,
    n_test_subjects: int = 10,
    debounce_k: int = 3,
    refractory_s: float = 5.0,
) -> dict:
    if fpr_targets is None:
        fpr_targets = [0.01, 0.05]

    from src.datasets.registry import get_dataset
    from src.datasets.registry import DatasetSpec
    from src.decoding.intent_switch import IntentSwitch
    from src.evaluation.metrics import asynchronous_metrics, bootstrap_ci, epoch_metrics, tpr_at_fixed_fpr
    from src.evaluation.streaming import simulate_from_epochs
    from src.preprocessing.pipeline import Preprocessor

    # ------------------------------------------------------------------ dataset
    spec = DatasetSpec(
        moabb_name="PhysionetMI",
        paradigm="mi",
        sfreq_target=160.0,
        channels=[
            "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
            "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
            "CP5", "CP3", "CP1",
        ],
        epoch_window=(0.0, 2.0),
        exclude_subjects=[88, 92, 100],
        band=(8.0, 30.0),
    )
    wrapper = get_dataset(spec)
    test_subjects = wrapper.subject_list[-n_test_subjects:]
    log.info("Test subjects: %s", test_subjects)

    X_test, y_test, meta = wrapper.load_epochs(test_subjects)
    log.info("Test set: %s epochs, %d control, %d idle",
             len(y_test), (y_test == 1).sum(), (y_test == 0).sum())

    # ------------------------------------------------------------------ checkpoints
    ckpts = _find_best_checkpoints(log_dir)
    log.info("Found %d checkpoint(s)", len(ckpts))

    fold_results: list[dict] = []

    for fold_idx, ckpt in enumerate(ckpts):
        log.info("Evaluating fold %d: %s", fold_idx, ckpt.name)
        model = _load_model(ckpt)

        # Load the preprocessor that was fit on training data for this fold.
        # PreprocessorCheckpoint saves it to <log_dir>/preprocessor.pkl.
        pp_path = ckpt.parent.parent / "preprocessor.pkl"
        if pp_path.exists():
            with open(pp_path, "rb") as f:
                pp = pickle.load(f)
            log.info("  Loaded preprocessor from %s", pp_path)
        else:
            log.warning(
                "  preprocessor.pkl not found at %s — refitting on test data "
                "(results will be approximate; re-run training to fix this)",
                pp_path,
            )
            cfg_pp = {"band": [8.0, 30.0], "keep_sfreq": 160.0,
                      "robust_scale": True, "euclidean_alignment": False}
            pp = Preprocessor(cfg_pp).fit(X_test)
        X_pp = pp.transform(X_test)

        # ---- epoch-level metrics ----------------------------------------
        probs = _score_epochs(model, X_pp)
        ep = epoch_metrics(y_test, probs)

        tpr_results: dict[str, float] = {}
        tau_results: dict[str, float] = {}
        for fpr in fpr_targets:
            tpr_val, tau = tpr_at_fixed_fpr(y_test, probs, target_fpr=fpr)
            key = f"tpr_at_fpr_{int(fpr*100):02d}pct"
            tpr_results[key] = tpr_val
            tau_results[f"tau_{int(fpr*100):02d}pct"] = tau

        # ---- streaming metrics (use τ @ 5% FPR) -------------------------
        tau_stream = tau_results.get("tau_05pct", 0.5)
        switch = IntentSwitch(
            tau=tau_stream,
            debounce_k=debounce_k,
            refractory_s=refractory_s,
        )
        model_fn = _build_model_fn(model)
        events, gt, idle_s = simulate_from_epochs(X_pp, y_test, model_fn, switch)
        async_m = asynchronous_metrics(
            [{"timestamp_s": e.timestamp_s} for e in events],
            gt,
            idle_duration_s=idle_s,
        )

        fold_results.append({
            "fold": fold_idx,
            "ckpt": ckpt.name,
            "roc_auc": ep["roc_auc"],
            **tpr_results,
            **tau_results,
            **{f"stream_{k}": v for k, v in async_m.items()},
        })

        log.info(
            "  Fold %d | AUC=%.3f | TPR@1%%FPR=%.3f | TPR@5%%FPR=%.3f | "
            "FA/min=%.2f | latency_med=%.2fs",
            fold_idx,
            ep["roc_auc"],
            tpr_results.get("tpr_at_fpr_01pct", float("nan")),
            tpr_results.get("tpr_at_fpr_05pct", float("nan")),
            async_m["false_activations_per_min"],
            async_m["median_latency_s"],
        )

    # ------------------------------------------------------------------ summary
    aucs = np.array([r["roc_auc"] for r in fold_results])
    tpr1 = np.array([r.get("tpr_at_fpr_01pct", np.nan) for r in fold_results])
    tpr5 = np.array([r.get("tpr_at_fpr_05pct", np.nan) for r in fold_results])

    ci_auc = bootstrap_ci(aucs)
    ci_tpr5 = bootstrap_ci(tpr5[~np.isnan(tpr5)])

    print("\n" + "=" * 60)
    print("STAGE 1 STREAMING EVALUATION — SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Mean':>8} {'95% CI':>20}")
    print("-" * 60)
    print(f"{'ROC-AUC (epoch)':<30} {aucs.mean():>8.3f} {f'[{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]':>20}")
    print(f"{'TPR @ 1% FPR':<30} {np.nanmean(tpr1):>8.3f}")
    print(f"{'TPR @ 5% FPR':<30} {np.nanmean(tpr5):>8.3f} {f'[{ci_tpr5[0]:.3f}, {ci_tpr5[1]:.3f}]':>20}")
    fa_vals = np.array([r["stream_false_activations_per_min"] for r in fold_results])
    lat_vals = np.array([r["stream_median_latency_s"] for r in fold_results])
    print(f"{'False activations/min':<30} {fa_vals.mean():>8.2f}")
    print(f"{'Median detection latency (s)':<30} {np.nanmean(lat_vals):>8.2f}")
    print("=" * 60)

    return {"fold_results": fold_results, "mean_auc": float(aucs.mean()),
            "mean_tpr_1pct": float(np.nanmean(tpr1)),
            "mean_tpr_5pct": float(np.nanmean(tpr5))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="lightning_logs")
    parser.add_argument("--fpr", nargs="+", type=float, default=[0.01, 0.05])
    parser.add_argument("--n-test-subjects", type=int, default=10)
    parser.add_argument("--debounce-k", type=int, default=3)
    parser.add_argument("--refractory-s", type=float, default=5.0)
    args = parser.parse_args()

    evaluate(
        log_dir=args.log_dir,
        fpr_targets=args.fpr,
        n_test_subjects=args.n_test_subjects,
        debounce_k=args.debounce_k,
        refractory_s=args.refractory_s,
    )


if __name__ == "__main__":
    main()
