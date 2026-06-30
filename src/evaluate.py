"""Evaluation entrypoint — load checkpoint(s), run an evaluation protocol, log results.

Usage:
    python -m src.evaluate checkpoint=results/stage1/run_xyz/best.ckpt evaluation=loso
    python -m src.evaluate checkpoint=... evaluation=streaming
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Implemented in Milestone 6.
    raise NotImplementedError("Evaluation entrypoint implemented in Milestone 6.")


if __name__ == "__main__":
    main()
