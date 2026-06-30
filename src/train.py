"""Hydra entrypoint — build datamodule + model + trainer and run training.

Usage:
    python -m src.train                          # Stage 1 defaults
    python -m src.train model=csp_lda           # classical comparator
    python -m src.train evaluation=streaming    # streaming evaluation
    python -m src.train seed=123 training.lr=5e-4
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Implemented in Milestone 5.
    raise NotImplementedError(
        "Training entrypoint implemented in Milestone 5. "
        "Run 'bash scripts/run_stage1.sh' after M5 is complete."
    )


if __name__ == "__main__":
    main()
