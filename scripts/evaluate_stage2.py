"""Stage 2 epoch-level evaluation — loads best fold checkpoints and reports AUC table.

Usage:
    python scripts/evaluate_stage2.py
    python scripts/evaluate_stage2.py --log-dir lightning_logs
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


_STAGE2_N_CHANNELS = 8
_STAGE2_CKPT_KEY = "model.backbone.net.conv_spatial.parametrizations.weight.original"


def _is_stage2_ckpt(path: str) -> bool:
    """Return True if checkpoint has n_channels=8 (Stage 2 shape)."""
    try:
        sd = torch.load(path, map_location="cpu")["state_dict"]
        return sd[_STAGE2_CKPT_KEY].shape[2] == _STAGE2_N_CHANNELS
    except Exception:
        return False


def _find_best_checkpoints(log_dir: str) -> list[Path]:
    pattern = str(Path(log_dir) / "version_*" / "checkpoints" / "best-epoch=*" / "*.ckpt")
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {log_dir}")
    # Filter to Stage 2 (8-channel) checkpoints with preprocessor and val AUC > 0.55
    # (guards against aborted runs that stopped in the first few epochs)
    def _val_auc_from_path(p: str) -> float:
        try:
            return float(Path(p).stem.split("=")[-1])
        except Exception:
            return 0.0

    stage2 = [c for c in ckpts
               if _is_stage2_ckpt(c)
               and (Path(c).parents[2] / "preprocessor.pkl").exists()
               and _val_auc_from_path(c) > 0.55]
    if not stage2:
        raise FileNotFoundError("No Stage 2 checkpoints (n_channels=8, val AUC>0.55, with preprocessor.pkl) found.")
    return [Path(c) for c in stage2]


def _load_model(ckpt_path: Path) -> torch.nn.Module:
    from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
    from src.training.lit_module import LitEEG

    backbone = BackboneEncoder(n_channels=8, n_times=102)
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
    x = torch.from_numpy(X).float()
    logits = model(x).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()


def evaluate(log_dir: str = "lightning_logs", n_test_subjects: int = 2) -> dict:
    from moabb.datasets import BNCI2014_009 as _BNCI009
    from src.datasets.registry import DatasetSpec, get_dataset
    from src.evaluation.metrics import bootstrap_ci, epoch_metrics, tpr_at_fixed_fpr
    from src.evaluation.p300_char import character_accuracy
    from src.preprocessing.pipeline import Preprocessor

    spec = DatasetSpec(
        moabb_name="BNCI2014_009",
        paradigm="p300",
        channels=["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"],
        sfreq_target=128.0,
        exclude_subjects=[],
        band=(1.0, 24.0),
        epoch_window=(0.0, 0.8),
    )
    wrapper = get_dataset(spec)
    test_subjects = wrapper.subject_list[-n_test_subjects:]
    log.info("Test subjects: %s", test_subjects)

    X_test, y_test, _ = wrapper.load_epochs(test_subjects)
    log.info("Test set: %d epochs, %d target, %d non-target",
             len(y_test), (y_test == 1).sum(), (y_test == 0).sum())

    ds_moabb = _BNCI009()   # for raw stim-channel access in character_accuracy

    ckpts = _find_best_checkpoints(log_dir)
    log.info("Found %d Stage 2 checkpoint(s)", len(ckpts))

    fold_results: list[dict] = []

    for fold_idx, ckpt in enumerate(ckpts):
        log.info("Evaluating fold %d: %s", fold_idx, ckpt.name)
        model = _load_model(ckpt)

        pp_path = ckpt.parents[2] / "preprocessor.pkl"
        if pp_path.exists():
            with open(pp_path, "rb") as f:
                pp = pickle.load(f)
        else:
            cfg_pp = {"band": [1.0, 24.0], "keep_sfreq": 128.0,
                      "robust_scale": True, "euclidean_alignment": False}
            pp = Preprocessor(cfg_pp).fit(X_test)
            log.warning("  No preprocessor.pkl — refitting on test data (approximate)")

        X_pp = pp.transform(X_test)
        probs = _score_epochs(model, X_pp)
        ep = epoch_metrics(y_test, probs)

        tpr1, tau1 = tpr_at_fixed_fpr(y_test, probs, target_fpr=0.01)
        tpr5, tau5 = tpr_at_fixed_fpr(y_test, probs, target_fpr=0.05)

        log.info("  Computing character-level accuracy…")
        char_m = character_accuracy(model, pp, ds_moabb, spec, test_subjects, n_reps=4)

        fold_results.append({
            "fold": fold_idx,
            "roc_auc": ep["roc_auc"],
            "tpr_at_fpr_01pct": tpr1,
            "tpr_at_fpr_05pct": tpr5,
            **char_m,
        })

        log.info(
            "  Fold %d | AUC=%.3f | TPR@5%%FPR=%.3f | "
            "CharAcc(1rep)=%.3f CharAcc(4rep)=%.3f | trial_AUC=%.3f",
            fold_idx, ep["roc_auc"], tpr5,
            char_m.get("char_acc_1rep", float("nan")),
            char_m.get("char_acc_4rep", float("nan")),
            char_m.get("trial_auc", float("nan")),
        )

    aucs = np.array([r["roc_auc"] for r in fold_results])
    tpr1s = np.array([r["tpr_at_fpr_01pct"] for r in fold_results])
    tpr5s = np.array([r["tpr_at_fpr_05pct"] for r in fold_results])
    acc1 = np.array([r.get("char_acc_1rep", np.nan) for r in fold_results])
    acc2 = np.array([r.get("char_acc_2rep", np.nan) for r in fold_results])
    acc4 = np.array([r.get("char_acc_4rep", np.nan) for r in fold_results])
    tauc = np.array([r.get("trial_auc", np.nan) for r in fold_results])

    ci_auc = bootstrap_ci(aucs)
    ci_acc4 = bootstrap_ci(acc4[~np.isnan(acc4)])

    print("\n" + "=" * 60)
    print("STAGE 2 EVALUATION — BNCI2014_009 P300")
    print("=" * 60)
    print(f"{'Metric':<34} {'Mean':>8} {'95% CI':>16}")
    print("-" * 60)
    print(f"{'Epoch ROC-AUC':<34} {np.nanmean(aucs):>8.3f} "
          f"{f'[{ci_auc[0]:.3f},{ci_auc[1]:.3f}]':>16}")
    print(f"{'Within-trial AUC':<34} {np.nanmean(tauc):>8.3f}")
    print(f"{'TPR @ 1% FPR':<34} {np.nanmean(tpr1s):>8.3f}")
    print(f"{'TPR @ 5% FPR':<34} {np.nanmean(tpr5s):>8.3f}")
    print("-" * 60)
    print(f"{'Char accuracy (1 rep / 12 flash)':<34} {np.nanmean(acc1):>8.1%}")
    print(f"{'Char accuracy (2 rep / 24 flash)':<34} {np.nanmean(acc2):>8.1%}")
    print(f"{'Char accuracy (4 rep / 48 flash)':<34} {np.nanmean(acc4):>8.1%} "
          f"{f'[{ci_acc4[0]:.1%},{ci_acc4[1]:.1%}]':>16}")
    print(f"  Chance level: {1/36:.1%}  (36 characters)")
    print("=" * 60)

    return {
        "fold_results": fold_results,
        "mean_auc": float(np.nanmean(aucs)),
        "mean_char_acc_4rep": float(np.nanmean(acc4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="lightning_logs")
    parser.add_argument("--n-test-subjects", type=int, default=2)
    args = parser.parse_args()
    evaluate(log_dir=args.log_dir, n_test_subjects=args.n_test_subjects)


if __name__ == "__main__":
    main()
