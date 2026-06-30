"""Warm the MOABB cache for all Stage 1 datasets.

Downloads the PhysioNet EEGMMIDB dataset (~3 GB) into MOABB's default cache
directory (~/.mne/MNE-physionet-MI-data/) so subsequent training runs are
fully offline. Calls dataset.data_path() per subject, which fetches all 8
needed run files (baselines 1–2 + 6 imagined-movement runs).

Usage
-----
    conda run -n als-decode python scripts/download_data.py
    conda run -n als-decode python scripts/download_data.py --subjects 1 2 3
    conda run -n als-decode python scripts/download_data.py --exclude 88 92 100
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings

log = logging.getLogger(__name__)

_DEFAULT_EXCLUDE = [88, 92, 100]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-download PhysionetMI EEG data via MOABB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Specific subject IDs to download. Omit to download all non-excluded subjects.",
    )
    p.add_argument(
        "--exclude",
        nargs="+",
        type=int,
        default=_DEFAULT_EXCLUDE,
        metavar="ID",
        help="Subject IDs to skip.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from moabb.datasets import PhysionetMI

    ds = PhysionetMI(imagined=True, executed=False)
    all_subjects = [s for s in ds.subject_list if s not in args.exclude]
    subjects = args.subjects if args.subjects is not None else all_subjects

    log.info(
        "Downloading %d subject(s) — excluded: %s", len(subjects), args.exclude
    )
    n_fail = 0
    for idx, subj in enumerate(subjects, 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                paths = ds.data_path(subject=subj)
            log.info(
                "[%d/%d] Subject %03d — %d file(s) cached",
                idx, len(subjects), subj, len(paths),
            )
        except Exception as exc:
            log.warning("Subject %03d FAILED: %s", subj, exc)
            n_fail += 1

    if n_fail:
        log.error("%d subject(s) failed; re-run or check network.", n_fail)
        return 1
    log.info("All files cached successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
