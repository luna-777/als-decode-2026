"""Concatenate Stage 3 result CSVs into one tidy file.

The runner writes one CSV per invocation, and the EA-reference arms are run as
separate invocations (session over the full grid, calibration over the headline
sizes only). This merges them without touching any value, asserting that the
headers agree and that no (subject, split, calib_size, seed, ea_ref, fold,
purge_k) key is duplicated across inputs.

Usage
-----
    python scripts/merge_results.py --inputs a.csv b.csv --out merged.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

KEY = ("subject", "split", "calib_size", "seed", "ea_ref", "fold", "purge_k")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    header = None
    rows = []
    for p in args.inputs:
        with open(p, newline="") as fh:
            r = csv.DictReader(fh)
            if header is None:
                header = r.fieldnames
            elif r.fieldnames != header:
                raise SystemExit(
                    f"Header mismatch in {p}:\n  expected {header}\n  got      {r.fieldnames}"
                )
            n = 0
            for row in r:
                rows.append(row)
                n += 1
        log.info("%s: %d rows", p, n)

    dupes = [k for k, c in Counter(tuple(r.get(k, "") for k in KEY) for r in rows).items() if c > 1]
    if dupes:
        raise SystemExit(
            f"{len(dupes)} duplicated keys across inputs; refusing to merge. "
            f"First: {dupes[0]}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    log.info("merged %d rows → %s", len(rows), out)


if __name__ == "__main__":
    main()
