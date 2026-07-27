"""Deterministic checkpoint resolution for Stage 3.

Replaces the filesystem scan that previously selected the MI checkpoint by
"highest val AUC in the filename" (docs/AUDIT.md §0.4). That scan made results a
function of the state of ``lightning_logs/`` at run time: any later training run
landing a higher-scoring checkpoint silently changed every Stage 3 number, and
nothing in the output recorded which checkpoint had been used.

Two problems are fixed here:

* **Pinning** — folds are named explicitly. Nothing is discovered by ranking.
* **Selection optimism** — taking the best of N folds and applying it to every
  held-out subject tunes the baseline on validation performance. Either pin one
  fold and record it, or evaluate every fold and aggregate across them.
"""
from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_STAGE2_KEY = "model.backbone.net.conv_spatial.parametrizations.weight.original"

_DATASET_CONFIG = {"mi": "physionet_mi.yaml", "p300": "bnci_009.yaml"}


@dataclass(frozen=True)
class CheckpointRef:
    """One pinned fold: its checkpoint, preprocessor, and recorded montage."""

    version: str
    ckpt_path: Path
    preprocessor_path: Path
    val_auc: float
    n_channels: int

    @property
    def version_dir(self) -> Path:
        return self.preprocessor_path.parent


def spec_from_config(paradigm: str, config_dir: Path | str | None = None):
    """Build the DatasetSpec from the same config file the training path reads.

    ``src/train.py`` builds its spec via ``build_spec_from_cfg(cfg.dataset)`` from
    ``configs/dataset/*.yaml``. Stage 3 must read the identical list, from the
    identical place — a second hardcoded copy is what produced the montage
    divergence in docs/AUDIT.md §0.1.
    """
    from omegaconf import OmegaConf

    from src.training.datamodule import build_spec_from_cfg

    if paradigm not in _DATASET_CONFIG:
        raise ValueError(f"Unknown paradigm {paradigm!r}")
    root = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "configs"
    path = root / "dataset" / _DATASET_CONFIG[paradigm]
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    return build_spec_from_cfg(OmegaConf.load(path))


def n_channels_of(ckpt_path: Path | str) -> int:
    """Channel count baked into a checkpoint's spatial convolution."""
    import torch

    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["state_dict"]
    return int(sd[_STAGE2_KEY].shape[2])


def _val_auc_from_path(path: Path | str) -> float:
    try:
        return float(Path(path).stem.split("=")[-1])
    except (ValueError, IndexError):
        return float("nan")


def load_fold(log_dir: Path | str, version: str) -> CheckpointRef:
    """Load one explicitly named fold, e.g. ``version_28``. Never ranks or guesses."""
    version_dir = Path(log_dir) / version
    if not version_dir.is_dir():
        raise FileNotFoundError(f"No such version directory: {version_dir}")

    ckpts = sorted(version_dir.glob("checkpoints/best-epoch=*/*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No best-epoch checkpoint in {version_dir}")
    if len(ckpts) > 1:
        raise ValueError(
            f"{version_dir} contains {len(ckpts)} best-epoch checkpoints; expected exactly "
            f"one so the choice is unambiguous: {[str(c) for c in ckpts]}"
        )
    ckpt = ckpts[0]

    pp = version_dir / "preprocessor.pkl"
    if not pp.exists():
        raise FileNotFoundError(
            f"No preprocessor.pkl beside {ckpt}. The Stage 3 transform must be the one "
            f"fitted on that fold's training subjects."
        )

    return CheckpointRef(
        version=version,
        ckpt_path=ckpt,
        preprocessor_path=pp,
        val_auc=_val_auc_from_path(ckpt),
        n_channels=n_channels_of(ckpt),
    )


def load_folds(log_dir: Path | str, versions: list[str], expect_channels: int | None = None) -> list[CheckpointRef]:
    """Load every named fold, asserting they agree on channel count."""
    if not versions:
        raise ValueError(
            "No fold versions given. Stage 3 requires the fold(s) to be named explicitly "
            "so the run is reproducible — see docs/AUDIT.md §0.4."
        )
    refs = [load_fold(log_dir, v) for v in versions]

    counts = {r.n_channels for r in refs}
    if len(counts) > 1:
        raise ValueError(
            f"Named folds disagree on channel count: "
            + ", ".join(f"{r.version}={r.n_channels}ch" for r in refs)
        )
    if expect_channels is not None and refs[0].n_channels != expect_channels:
        raise ValueError(
            f"Folds have {refs[0].n_channels} channels but the dataset config requests "
            f"{expect_channels}. The checkpoint and the config disagree on the montage size."
        )
    for r in refs:
        log.info("Pinned fold %s: %s (val AUC=%.4f, %dch)", r.version, r.ckpt_path, r.val_auc, r.n_channels)
    return refs


def loso_fold_for_subject(
    log_dir: Path | str,
    subject: int,
    n_channels: int | None = None,
    strict: bool = True,
) -> CheckpointRef:
    """Return the fold whose val_subjects.json contains *subject*.

    strict=True (the default, and what §11.2 requires) raises when no fold recorded
    that subject, rather than falling back to a checkpoint that may have trained on
    them. The old fallback to "best checkpoint overall" could silently score a
    subject against a model that had seen them.

    *n_channels* restricts the search to folds of the matching paradigm. Subject IDs
    are small integers in both PhysionetMI (1-109) and BNCI2014_009 (1-10), so an MI
    fold's val_subjects.json will collide with P300 subject numbers if the search is
    not filtered. Pass the paradigm's channel count to disambiguate.
    """
    import json

    metas = sorted(Path(log_dir).glob("version_*/val_subjects.json"))
    if not metas:
        raise FileNotFoundError(
            f"No val_subjects.json under {log_dir}; LOSO assignment cannot be verified. "
            f"Re-train with evaluation=p300_loso_full."
        )

    matches = []
    for meta in metas:
        if subject not in json.loads(meta.read_text()):
            continue
        if n_channels is not None:
            ckpts = sorted(meta.parent.glob("checkpoints/best-epoch=*/*.ckpt"))
            if not ckpts:
                continue
            try:
                if n_channels_of(ckpts[0]) != n_channels:
                    continue  # a fold from the other paradigm
            except Exception:
                continue
        matches.append(meta.parent.name)

    if not matches:
        raise FileNotFoundError(
            f"No fold held out subject {subject}. Folds present: "
            f"{[m.parent.name for m in metas]}. Every scored subject must have been in "
            f"some fold's validation set, or it was trained on."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Subject {subject} is in the validation set of multiple folds {matches}; "
            f"the LOSO assignment is ambiguous."
        )
    return load_fold(log_dir, matches[0])


def discover_versions(log_dir: Path | str, n_channels: int) -> list[str]:
    """List version dirs whose checkpoint has *n_channels*. For inspection only.

    Deliberately not used to pick a checkpoint — it exists so a human can see what
    is available and then name the folds explicitly on the command line.
    """
    out = []
    for c in sorted(glob.glob(str(Path(log_dir) / "version_*" / "checkpoints" / "best-epoch=*" / "*.ckpt"))):
        try:
            if n_channels_of(c) == n_channels:
                out.append(Path(c).parents[2].name)
        except Exception:
            continue
    return sorted(set(out), key=lambda v: int(v.split("_")[1]))
