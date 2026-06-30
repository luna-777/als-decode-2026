"""W&B + console logging utilities."""
from __future__ import annotations

import logging
import os
from typing import Any


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a console logger with a compact timestamp format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)8s | %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def setup_wandb(cfg: Any, job_type: str = "train") -> None:
    """Initialize a W&B run from a resolved Hydra DictConfig.

    Set WANDB_MODE=offline to avoid network access.
    Logs resolved config + git SHA + seed per §8.1 reproducibility requirement.
    """
    try:
        import subprocess

        import wandb
        from omegaconf import OmegaConf

        git_sha = "unknown"
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            git_sha = result.stdout.strip()
        except Exception:
            pass

        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        assert isinstance(config_dict, dict)
        config_dict["_git_sha"] = git_sha

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "als-decode"),
            name=cfg.get("run_name", None),
            config=config_dict,
            job_type=job_type,
            mode=os.environ.get("WANDB_MODE", "online"),
        )
    except ImportError:
        get_logger(__name__).warning("wandb not installed; skipping W&B init")
