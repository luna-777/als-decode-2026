"""Character-level P300 accuracy for BNCI2014_009.

Flash stim channel codes (MOABB's encoding, offset by 2 from raw StimulusCode):
    Rows    3–8   (original StimulusCode 1–6, row 1–6)
    Columns 9–14  (original StimulusCode 7–12, column 1–6)

Evidence accumulation: average model score per flash code across repetitions,
then argmax over row codes → predicted row; argmax over col codes → predicted col.
Character correct iff both row and column predicted correctly.
"""
from __future__ import annotations

import warnings

import numpy as np

ROW_CODES: frozenset[int] = frozenset(range(3, 9))   # {3..8}
COL_CODES: frozenset[int] = frozenset(range(9, 15))  # {9..14}
N_STIMULI = 12   # flashes per repetition (6 rows + 6 columns)


def _extract_coded_epochs(
    raw,
    ch_names: list[str],
    sfreq_target: float,
    band: tuple[float, float],
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract bandpassed, resampled epochs with flash codes from a MOABB RawArray.

    Returns
    -------
    X : (N, C, T) float32
    y : (N,) int64 — 1 = Target, 0 = NonTarget
    codes : (N,) int — flash stimulus code for each epoch
    """
    import mne

    # Flash and target stim channels
    flash_data = raw.get_data(picks=["Flash stim"]).flatten()
    target_data = raw.get_data(picks=["Target stim"]).flatten()

    # Rising edges of Flash stim = stimulus onsets (at native sfreq)
    flash_diff = np.diff(np.concatenate([[0], flash_data.astype(int)]))
    event_samples_native = np.where(flash_diff > 0)[0]
    flash_codes = flash_data[event_samples_native].astype(int)
    target_flags = target_data[event_samples_native].astype(int)  # 1=NT, 2=T

    # Bandpass at native sfreq, then resample, then pick EEG channels
    native_sfreq = raw.info["sfreq"]
    raw_eeg = raw.copy().pick(picks=ch_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_eeg.filter(
            l_freq=band[0], h_freq=band[1],
            method="fir", phase="zero", fir_window="hamming", verbose=False,
        )
        raw_eeg.resample(sfreq_target, verbose=False)

    # Scale event sample indices to new sfreq
    ratio = sfreq_target / native_sfreq
    event_samples = np.round(event_samples_native * ratio).astype(int)

    # Build MNE events array: (sample, 0, flash_code)
    event_id = {str(c): c for c in np.unique(flash_codes)}
    events_mne = np.column_stack([
        event_samples,
        np.zeros(len(event_samples), dtype=int),
        flash_codes,
    ])

    tmax_adj = tmax - 1.0 / sfreq_target
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs = mne.Epochs(
            raw_eeg, events_mne, event_id=event_id,
            tmin=tmin, tmax=tmax_adj,
            baseline=None, preload=True, verbose=False,
        )

    X = epochs.get_data().astype(np.float32)   # (N, C, T)
    ep_codes = epochs.events[:, 2].astype(int)

    # Recover target flag per epoch (match by event sample)
    sample_to_flag = dict(zip(event_samples.tolist(), target_flags.tolist()))
    ep_flags = np.array([sample_to_flag.get(int(s), 1) for s in epochs.events[:, 0]])
    y = (ep_flags == 2).astype(np.int64)

    return X, y, ep_codes


def character_accuracy(
    model,
    preprocessor,
    ds_moabb,
    spec,
    subjects: list[int],
    n_reps: int = 4,
) -> dict[str, float]:
    """Compute character accuracy with 1..n_reps repetitions of evidence.

    For each character trial of (n_reps × N_STIMULI) epochs:
    1. Score with *model*.
    2. Average scores per flash code across k repetitions (k = 1..n_reps).
    3. Predicted character = (argmax row score, argmax col score).
    4. Correct iff predicted == true.

    Returns dict:
        char_acc_{k}rep  : accuracy using k repetitions
        trial_auc        : mean within-trial AUC (8 targets vs 40 non-targets)
    """
    import torch
    from sklearn.metrics import roc_auc_score

    trial_size = n_reps * N_STIMULI   # 48

    acc_by_reps: dict[int, list[float]] = {k: [] for k in range(1, n_reps + 1)}
    trial_aucs: list[float] = []

    for subj in subjects:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_data = ds_moabb.get_data(subjects=[subj])

        for sess_key in sorted(raw_data[subj].keys()):
            for run_key in sorted(raw_data[subj][sess_key].keys()):
                raw = raw_data[subj][sess_key][run_key]

                X, y, codes = _extract_coded_epochs(
                    raw,
                    ch_names=list(spec.channels),
                    sfreq_target=spec.sfreq_target,
                    band=spec.band,
                    tmin=spec.epoch_window[0],
                    tmax=spec.epoch_window[1],
                )

                X_pp = preprocessor.transform(X)

                with torch.no_grad():
                    x_t = torch.from_numpy(X_pp).float()
                    scores = torch.sigmoid(model(x_t).squeeze(-1)).cpu().numpy()

                n_trials = len(scores) // trial_size
                for trial_idx in range(n_trials):
                    s = trial_idx * trial_size
                    e = s + trial_size
                    t_codes = codes[s:e]
                    t_scores = scores[s:e]
                    t_labels = y[s:e]

                    # True target codes
                    target_code_set = set(t_codes[t_labels == 1].tolist())
                    rows_t = [c for c in target_code_set if c in ROW_CODES]
                    cols_t = [c for c in target_code_set if c in COL_CODES]
                    if not rows_t or not cols_t:
                        continue
                    true_row, true_col = rows_t[0], cols_t[0]

                    # Within-trial AUC (all n_reps reps)
                    if t_labels.sum() > 0 and (1 - t_labels).sum() > 0:
                        trial_aucs.append(float(roc_auc_score(t_labels, t_scores)))

                    # Evidence accumulation: k = 1..n_reps repetitions
                    for k in range(1, n_reps + 1):
                        end_k = k * N_STIMULI
                        sub_codes = t_codes[:end_k]
                        sub_scores = t_scores[:end_k]

                        code_score = {
                            c: float(sub_scores[sub_codes == c].mean())
                            for c in np.unique(sub_codes)
                        }
                        row_s = {c: code_score[c] for c in ROW_CODES if c in code_score}
                        col_s = {c: code_score[c] for c in COL_CODES if c in code_score}
                        if not row_s or not col_s:
                            continue

                        pred_row = max(row_s, key=row_s.get)  # type: ignore[arg-type]
                        pred_col = max(col_s, key=col_s.get)  # type: ignore[arg-type]
                        acc_by_reps[k].append(float(pred_row == true_row and pred_col == true_col))

    result = {f"char_acc_{k}rep": float(np.mean(v)) if v else float("nan")
              for k, v in acc_by_reps.items()}
    result["trial_auc"] = float(np.mean(trial_aucs)) if trial_aucs else float("nan")
    return result
