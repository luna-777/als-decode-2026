"""Montage contract — the channel list and order are part of a checkpoint's identity.

ADR-10: a Stage 2 checkpoint and the Stage 3 data loaded against it must agree on
the electrode montage *and its array order*. Two lists of equal length but
different order load without a shape error while every learned spatial filter is
applied to the wrong electrode; this must raise, never silently reindex.

The montage is written beside the Lightning version directory as ``montage.json``
at training time and asserted at load time.

Checkpoints trained before this module existed have no ``montage.json``. For those
the montage can only be *inferred* from the dataset config that the training path
read (``src/train.py`` -> ``build_spec_from_cfg`` -> ``configs/dataset/*.yaml``).
``load_montage`` reports which of the two cases applies so callers can record the
provenance, and ``strict=True`` refuses the inferred case.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MONTAGE_FILENAME = "montage.json"


@dataclass(frozen=True)
class Montage:
    """The channel contract recorded alongside a checkpoint."""

    channels: tuple[str, ...]
    sfreq: float
    n_times: int
    paradigm: str
    source: str  # "recorded" | "inferred_from_config"

    @property
    def n_channels(self) -> int:
        return len(self.channels)


def write_montage(version_dir: Path | str, spec, n_times: int) -> Path:
    """Persist *spec*'s montage next to a Lightning version directory."""
    version_dir = Path(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)
    path = version_dir / MONTAGE_FILENAME
    path.write_text(
        json.dumps(
            {
                "channels": list(spec.channels),
                "sfreq": float(spec.sfreq_target),
                "n_times": int(n_times),
                "paradigm": str(spec.paradigm),
            },
            indent=2,
        )
    )
    return path


def load_montage(version_dir: Path | str, spec, n_times: int, strict: bool = False) -> Montage:
    """Return the montage recorded for *version_dir*, or infer it from *spec*.

    strict=True raises when no montage.json exists instead of inferring.
    """
    version_dir = Path(version_dir)
    path = version_dir / MONTAGE_FILENAME

    if not path.exists():
        if strict:
            raise ValueError(
                f"No {MONTAGE_FILENAME} recorded for checkpoint directory {version_dir}. "
                f"This checkpoint predates the montage contract, so its electrode order "
                f"cannot be verified against the data being loaded. Re-train it, or drop "
                f"--strict-montage to proceed on the montage inferred from the training "
                f"config ({list(spec.channels)})."
            )
        log.warning(
            "No %s in %s — inferring montage from the dataset config. "
            "This is only sound because the config channel list is unambiguous for the "
            "period in which the checkpoint was trained; it is not verified.",
            MONTAGE_FILENAME,
            version_dir,
        )
        return Montage(
            channels=tuple(spec.channels),
            sfreq=float(spec.sfreq_target),
            n_times=int(n_times),
            paradigm=str(spec.paradigm),
            source="inferred_from_config",
        )

    d = json.loads(path.read_text())
    return Montage(
        channels=tuple(d["channels"]),
        sfreq=float(d["sfreq"]),
        n_times=int(d["n_times"]),
        paradigm=str(d["paradigm"]),
        source="recorded",
    )


def assert_montage_matches(montage: Montage, channels: list[str], where: str = "") -> None:
    """Raise ValueError unless *channels* equals the montage list exactly, in order.

    Equal length with different content or different order is the dangerous case —
    it loads without a shape error. Both lists appear in the message.
    """
    expected = list(montage.channels)
    got = list(channels)
    if expected == got:
        return

    ctx = f" ({where})" if where else ""
    lines = [
        f"Montage mismatch{ctx}: the checkpoint was trained on a different electrode "
        f"set or order than the data being loaded against it.",
        f"  checkpoint ({montage.source}, n={len(expected)}): {expected}",
        f"  data                    (n={len(got)}): {got}",
    ]

    if len(expected) == len(got):
        disagree = [i for i, (a, b) in enumerate(zip(expected, got)) if a != b]
        lines.append(
            f"  equal length ({len(expected)}) — this would NOT have raised a shape error. "
            f"{len(disagree)}/{len(expected)} positions disagree."
        )
        for i in disagree:
            lines.append(f"    idx {i:>2}: trained_as {expected[i]!r} -> loaded_as {got[i]!r}")
    only_ckpt = [c for c in expected if c not in got]
    only_data = [c for c in got if c not in expected]
    if only_ckpt:
        lines.append(f"  trained on but absent from data: {only_ckpt}")
    if only_data:
        lines.append(f"  present in data but never trained on: {only_data}")

    raise ValueError("\n".join(lines))
