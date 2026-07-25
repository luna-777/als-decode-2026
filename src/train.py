"""Hydra entrypoint — build datamodule + model + trainer and run training.

Usage:
    python -m src.train                           # Stage 1 defaults, fold 0
    python -m src.train fold=1                    # second GroupKFold fold
    python -m src.train training.max_epochs=5     # smoke run
    python -m src.train model=csp_lda            # classical comparator (TODO M4 CLI)
    python -m src.train seed=123 training.lr=5e-4
"""
from __future__ import annotations

import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.datasets.registry import get_dataset
    from src.datasets.splits import GroupKFoldSplitter
    from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
    from src.training.callbacks import PreprocessorCheckpoint, build_callbacks
    from src.training.datamodule import EEGDataModule, build_spec_from_cfg
    from src.training.lit_module import LitEEG
    from src.utils.seed import seed_everything

    seed_everything(int(cfg.seed))
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------ dataset
    spec = build_spec_from_cfg(cfg.dataset)
    wrapper = get_dataset(spec)
    all_subjects = wrapper.subject_list

    max_subjects = cfg.evaluation.get("max_subjects", None)
    if max_subjects is not None:
        all_subjects = all_subjects[: int(max_subjects)]
        log.info("Smoke mode: capped at %d subjects", len(all_subjects))

    # ------------------------------------------------------------------ splits
    held_out_n = int(cfg.evaluation.get("held_out_n_subjects", 10))
    if held_out_n == 0:
        # Full LOSO: all subjects participate in cross-validation, no final holdout.
        test_subjects = []
        dev_subjects = all_subjects
    else:
        test_subjects = all_subjects[-held_out_n:]
        dev_subjects = all_subjects[:-held_out_n]

    n_splits = int(cfg.evaluation.get("n_splits", 5))
    splitter = GroupKFoldSplitter(n_splits=n_splits)
    target_fold = int(cfg.get("fold", 0))

    log.info(
        "Dataset: %d total subjects | %d dev | %d held-out test",
        len(all_subjects),
        len(dev_subjects),
        len(test_subjects),
    )

    for fold_idx, (train_subjs, val_subjs) in enumerate(splitter.split(dev_subjects)):
        if fold_idx != target_fold:
            continue

        log.info(
            "Fold %d/%d — train %d subj, val %d subj, test %d subj",
            fold_idx,
            n_splits - 1,
            len(train_subjs),
            len(val_subjs),
            len(test_subjects),
        )

        # ------------------------------------------------------------------ data
        dm = EEGDataModule(cfg, wrapper, train_subjs, val_subjs, test_subjects)
        dm.setup()

        # ------------------------------------------------------------------ model
        n_ch = len(spec.channels)
        n_times = int(
            round(spec.sfreq_target * (spec.epoch_window[1] - spec.epoch_window[0]))
        )
        backbone = BackboneEncoder(n_channels=n_ch, n_times=n_times)
        head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
        model = EEGDecoder(backbone, head)

        lit = LitEEG(model, cfg, pos_weight=dm.pos_weight)
        log.info("EEGDecoder: feature_dim=%d", backbone.feature_dim)

        # ------------------------------------------------------------------ trainer
        max_epochs = int(cfg.training.get("max_epochs", 200))
        precision = "32"  # bf16-mixed requires hardware support; safe default

        callbacks = build_callbacks(cfg) + [PreprocessorCheckpoint(dm.preprocessor)]

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=1,
            deterministic=True,
            precision=precision,
        )

        trainer.fit(lit, dm)

        # Record which subjects were held out as validation so Stage 3 can
        # find the right fold checkpoint for each test subject.
        import json
        from pathlib import Path as _Path
        _log_dir = _Path(trainer.logger.log_dir)
        _log_dir.mkdir(parents=True, exist_ok=True)
        (_log_dir / "val_subjects.json").write_text(json.dumps(val_subjs))

        if test_subjects:
            trainer.test(lit, dm, ckpt_path="best")

        break  # one fold per invocation; use fold=N override for others


if __name__ == "__main__":
    main()
