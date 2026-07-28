"""Round C.4 — ALS character-level accuracy and ITR, per patient.

An epoch AUC of 0.558 (§0.12) does not say whether the speller is unusable or
merely slow. A 6x6 matrix speller accumulates evidence over repetitions, so a weak
per-epoch discriminator can still select characters reliably given enough
repetitions — at a cost in bits per minute. This measures that directly.

Protocol
--------
BNCI2014_008 is 35 copy-spelled characters (7 five-letter words) at 10 repetitions
each, 12 flashes per repetition. Calibration uses the FIRST ``--calib-chars``
characters; the test set is always the LAST 20 characters, held fixed so the three
head variants and every calibration size are scored on identical trials. This is
also the deployable ordering — calibrate early, spell later — unlike the stratified
epoch draws used for the AUC grid.

Three heads are compared: the pretrained head unchanged, a head adapted on the
calibration characters, and the validation-gated head of Round C.3.

Information transfer rate uses Wolpaw's formula with M=36 alternatives, under an
explicitly stated timing model: T(k) = k x 12 flashes x 0.25 s SOA, with no
inter-character pause. That is an optimistic upper bound; a real speller adds 2-4 s
per selection, which lowers ITR roughly proportionally at small k.

Usage
-----
    python scripts/als_character.py --folds version_31 ... version_40
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

N_ALTERNATIVES = 36
SOA_S = 0.25
FLASHES_PER_REP = 12
N_REPS = 10
N_TEST_CHARS = 20


def wolpaw_bits(p: float, m: int = N_ALTERNATIVES) -> float:
    """Bits per selection at accuracy p among m alternatives."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return float(np.log2(m))
    return float(
        np.log2(m) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (m - 1))
    )


def itr_bits_per_min(p: float, k_reps: int) -> float:
    t = k_reps * FLASHES_PER_REP * SOA_S
    return wolpaw_bits(p) * 60.0 / t


def char_accuracy_by_reps(scores, codes, n_chars, row_codes, col_codes):
    """Accuracy using 1..N_REPS repetitions. scores/codes are in trial order."""
    from src.evaluation.p300_char import COL_CODES, ROW_CODES

    per_char = FLASHES_PER_REP * N_REPS
    acc = {k: [] for k in range(1, N_REPS + 1)}
    for c in range(n_chars):
        s0, s1 = c * per_char, (c + 1) * per_char
        cs, sc = codes[s0:s1], scores[s0:s1]
        tr, tc = row_codes[c], col_codes[c]
        for k in range(1, N_REPS + 1):
            end = k * FLASHES_PER_REP
            kc, ks = cs[:end], sc[:end]
            rmean = {code: ks[kc == code].mean() for code in set(kc.tolist())
                     if code in ROW_CODES}
            cmean = {code: ks[kc == code].mean() for code in set(kc.tolist())
                     if code in COL_CODES}
            if not rmean or not cmean:
                continue
            pr = max(rmean, key=rmean.get)
            pc = max(cmean, key=cmean.get)
            acc[k].append(1.0 if (pr == tr and pc == tc) else 0.0)
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--log-dir", default="lightning_logs")
    ap.add_argument("--calib-chars", nargs="+", type=int, default=[2, 5, 10, 15])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--adapt-epochs", type=int, default=100)
    ap.add_argument("--adapt-lr", type=float, default=1e-3)
    ap.add_argument("--val-floor", type=int, default=8)
    ap.add_argument("--out", default="experiments/roundc/als_character.csv")
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split

    from scripts.adapt_stage3 import (
        _adapt_head_on_features,
        _apply_ea,
        _features,
        _head_auc,
        _load_model_frozen,
    )
    from scripts.gated_adapt import _train_with_early_stop
    from src.datasets.montage import assert_montage_matches, load_montage
    from src.evaluation.p300_char import COL_CODES, ROW_CODES, _extract_coded_epochs
    from src.models.checkpoints import load_folds, spec_from_config

    spec = spec_from_config("als")
    contract_spec = spec_from_config("p300")
    n_times = int(round(spec.sfreq_target * (spec.epoch_window[1] - spec.epoch_window[0])))
    n_channels = len(spec.channels)
    refs = load_folds(args.log_dir, args.folds, expect_channels=n_channels)

    from moabb.datasets import BNCI2014_008
    ds = BNCI2014_008()
    patients = ds.subject_list

    rows = []
    for subj in patients:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_data = ds.get_data(subjects=[subj])
        sess = sorted(raw_data[subj])[0]
        run = sorted(raw_data[subj][sess])[0]
        X, y, codes = _extract_coded_epochs(
            raw_data[subj][sess][run], ch_names=list(spec.channels),
            sfreq_target=spec.sfreq_target, band=spec.band,
            tmin=spec.epoch_window[0], tmax=spec.epoch_window[1],
        )
        per_char = FLASHES_PER_REP * N_REPS
        n_chars = len(y) // per_char
        X, y, codes = X[: n_chars * per_char], y[: n_chars * per_char], codes[: n_chars * per_char]

        # true row/col code per character
        row_t, col_t = [], []
        for c in range(n_chars):
            s0, s1 = c * per_char, (c + 1) * per_char
            tgt = set(codes[s0:s1][y[s0:s1] == 1].tolist())
            r = [x for x in tgt if x in ROW_CODES]
            cc = [x for x in tgt if x in COL_CODES]
            row_t.append(r[0] if r else -1)
            col_t.append(cc[0] if cc else -1)
        row_t, col_t = np.array(row_t), np.array(col_t)

        test_lo = n_chars - N_TEST_CHARS
        log.info("patient %s: %d characters, %d epochs; test = chars %d-%d",
                 subj, n_chars, len(y), test_lo + 1, n_chars)

        X_ea = _apply_ea(X, fit_idx=None)   # session-scoped, as in the AUC grid

        for ref in refs:
            montage = load_montage(ref.version_dir, contract_spec, n_times)
            assert_montage_matches(montage, list(spec.channels),
                                   where=f"{ref.version} vs ALS patient {subj}")
            with open(ref.preprocessor_path, "rb") as f:
                pp = pickle.load(f)
            model = _load_model_frozen(ref.ckpt_path, "als", n_channels, n_times,
                                       reinit_head=False)
            model.eval()
            feats = _features(model, pp.transform(X_ea))

            test_slice = slice(test_lo * per_char, n_chars * per_char)
            f_test, y_test, c_test = feats[test_slice], y[test_slice], codes[test_slice]

            def emit(head, variant, C, sd, accepted="", sel_chars=5):
                """Score on the fixed test characters.

                The repetition count is an operating point that must be chosen
                without seeing the test set. It is selected on the first
                *sel_chars* characters (held-out from the test block either way),
                and the test ITR at that k is the honest number. The maximum over
                k on the test block is also recorded, explicitly labelled, because
                it is what a test-peeking analysis would report and it is
                materially higher.
                """
                with torch.no_grad():
                    sc = torch.sigmoid(head(f_test).squeeze(-1)).numpy()
                acc = char_accuracy_by_reps(sc, c_test, N_TEST_CHARS,
                                            row_t[test_lo:], col_t[test_lo:])
                auc = _head_auc(head, f_test, y_test)

                # operating point selected on calibration characters
                sel = slice(0, sel_chars * per_char)
                with torch.no_grad():
                    sc_sel = torch.sigmoid(head(feats[sel]).squeeze(-1)).numpy()
                acc_sel = char_accuracy_by_reps(sc_sel, codes[sel], sel_chars,
                                                row_t[:sel_chars], col_t[:sel_chars])
                itr_sel = {k: itr_bits_per_min(acc_sel[k], k) for k in acc_sel
                           if np.isfinite(acc_sel[k])}
                k_hon = max(itr_sel, key=itr_sel.get) if itr_sel else N_REPS

                itr_test = {k: itr_bits_per_min(acc[k], k) for k in acc
                            if np.isfinite(acc[k])}
                k_opt = max(itr_test, key=itr_test.get) if itr_test else None

                rows.append({
                    "patient": subj, "fold": ref.version, "variant": variant,
                    "calib_chars": C, "seed": sd, "accepted": accepted,
                    "epoch_auc": auc,
                    **{f"char_acc_{k}rep": acc[k] for k in range(1, N_REPS + 1)},
                    "acc_10rep": round(acc[N_REPS], 4) if np.isfinite(acc[N_REPS]) else "",
                    "itr_10rep": round(itr_bits_per_min(acc[N_REPS], N_REPS), 4)
                                 if np.isfinite(acc[N_REPS]) else "",
                    "k_selected_on_calib": k_hon,
                    "itr_at_selected_k": round(itr_test.get(k_hon, float("nan")), 4),
                    "acc_at_selected_k": round(acc.get(k_hon, float("nan")), 4),
                    "k_test_optimal_PEEKING": k_opt if k_opt else "",
                    "itr_test_max_PEEKING": round(itr_test[k_opt], 4) if k_opt else "",
                })

            base = copy.deepcopy(model.head); base.eval()
            emit(base, "unadapted", 0, 0)

            for C in args.calib_chars:
                if C >= test_lo:
                    continue
                cal = slice(0, C * per_char)
                f_cal, y_cal = feats[cal], y[cal]
                if len(np.unique(y_cal)) < 2:
                    continue
                for sd in args.seeds:
                    ad = copy.deepcopy(model.head)
                    _adapt_head_on_features(ad, f_cal, y_cal,
                                            n_epochs=args.adapt_epochs,
                                            lr=args.adapt_lr, seed=sd)
                    emit(ad, "adapted", C, sd)

                    n_val = int(round(len(y_cal) * 0.25))
                    if n_val < args.val_floor:
                        emit(copy.deepcopy(base), "gated", C, sd, accepted="ungateable")
                        continue
                    idx = np.arange(len(y_cal))
                    tr, va = train_test_split(idx, test_size=n_val,
                                              random_state=sd, stratify=y_cal)
                    if len(np.unique(y_cal[va])) < 2:
                        emit(copy.deepcopy(base), "gated", C, sd, accepted="ungateable")
                        continue
                    g = copy.deepcopy(model.head)
                    g, vauc, _ = _train_with_early_stop(
                        g, f_cal[tr], y_cal[tr], f_cal[va], y_cal[va],
                        args.adapt_epochs, args.adapt_lr, sd, patience=15)
                    bval = _head_auc(base, f_cal[va], y_cal[va])
                    ok = bool(np.isfinite(vauc) and vauc > bval)
                    emit(g if ok else copy.deepcopy(base), "gated", C, sd,
                         accepted=int(ok))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log.info("Wrote %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
