"""MoabbDatasetWrapper and BnciP300Wrapper — sole data-access classes. §8.3.

Never parse raw .edf/.mat files directly; all access goes through these classes via MOABB.
Responsible for channel selection, subject exclusion, and binary epoch labelling.
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import mne
import numpy as np

if TYPE_CHECKING:
    from src.datasets.registry import DatasetSpec

log = logging.getLogger(__name__)

# Semantic event names assigned by MOABB's PhysionetMI._get_single_subject_data.
_ALL_EVENT_ID: dict[str, int] = {
    "left_hand": 2,
    "right_hand": 3,
    "hands": 4,
    "feet": 5,
    "rest": 1,
}
_CONTROL_IDS: frozenset[int] = frozenset({2, 3, 4, 5})  # imagined movement = positive class
_BASELINE_RUN_NUMS: tuple[int, int] = (1, 2)  # eyes-open, eyes-closed


class MoabbDatasetWrapper:
    """Wraps PhysionetMI via MOABB; returns binary (control vs idle) epoch arrays.

    ADR-12: baseline runs 1–2 are not exposed by PhysionetMI.get_data(), so we call the
    private PhysionetMI._load_one_run() method for those runs only. All EDF reading still
    goes through MOABB's own internals — we never call mne.io.read_raw_edf ourselves.
    This private-API dependency is isolated here and documented in DECISIONS.md.
    """

    def __init__(self, spec: DatasetSpec) -> None:
        from moabb.datasets import PhysionetMI

        self.spec = spec
        self._moabb_ds = PhysionetMI(imagined=True, executed=False)
        self._subject_list: list[int] = [
            s for s in self._moabb_ds.subject_list if s not in spec.exclude_subjects
        ]

    @property
    def subject_list(self) -> list[int]:
        """All valid subject IDs after applying exclude_subjects."""
        return list(self._subject_list)

    def load_epochs(self, subjects: list[int]) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load and label epochs for *subjects*.

        Returns
        -------
        X : (N, C, T) float32 — epochs × channels × time-points
        y : (N,) int64 — 1 = control (imagined movement), 0 = idle (rest / baseline)
        metadata : dict with keys 'subjects', 'channels', 'sfreq', 'n_times'
        """
        ch_names = list(self.spec.channels)
        sfreq = self.spec.sfreq_target
        tmin = float(self.spec.epoch_window[0])
        # Half-open [tmin, tmax) so MNE yields exactly round(window_len * sfreq) samples.
        tmax = float(self.spec.epoch_window[1]) - 1.0 / sfreq
        n_times = int(round((tmax - tmin) * sfreq)) + 1  # 320 for 2 s @ 160 Hz

        all_X: list[np.ndarray] = []
        all_y: list[int] = []
        all_subj: list[int] = []

        for subj in subjects:
            log.info("Loading subject %d", subj)

            # 6 imagined-movement task runs via public MOABB API.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                subj_data = self._moabb_ds.get_data(subjects=[subj])

            for run_raw in subj_data[subj]["0"].values():
                _epoch_task_run(
                    run_raw, ch_names, tmin, tmax, subj, all_X, all_y, all_subj
                )

            # Eyes-open / eyes-closed baseline runs via private MOABB method (ADR-12).
            for run_num in _BASELINE_RUN_NUMS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_base = self._moabb_ds._load_one_run(subj, run_num)
                _epoch_baseline_run(raw_base, ch_names, n_times, subj, all_X, all_y, all_subj)

        X = np.stack(all_X, axis=0).astype(np.float32)
        y = np.array(all_y, dtype=np.int64)
        meta: dict = {
            "subjects": all_subj,
            "channels": ch_names,
            "sfreq": sfreq,
            "n_times": n_times,
        }
        return X, y, meta


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ---------------------------------------------------------------------------

def _pick_channels(raw: mne.io.BaseRaw, ch_names: list[str]) -> mne.io.BaseRaw:
    """Return a copy of *raw* with channels selected and reordered to *ch_names*."""
    return raw.copy().pick(picks=ch_names)


def _epoch_task_run(
    raw: mne.io.BaseRaw,
    ch_names: list[str],
    tmin: float,
    tmax: float,
    subj: int,
    all_X: list[np.ndarray],
    all_y: list[int],
    all_subj: list[int],
) -> None:
    """Epoch one annotated task run; append arrays and labels in-place."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_picked = _pick_channels(raw, ch_names)
        try:
            events, _ = mne.events_from_annotations(
                raw_picked, event_id=_ALL_EVENT_ID, verbose=False
            )
        except Exception:
            return
        if len(events) == 0:
            return
        # Filter event_id to only events present in this run to avoid MNE ValueError.
        present = set(int(e) for e in events[:, 2])
        event_id_run = {k: v for k, v in _ALL_EVENT_ID.items() if v in present}
        epochs = mne.Epochs(
            raw_picked,
            events,
            event_id=event_id_run,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        X = epochs.get_data().astype(np.float32)  # (n_ep, C, T)

    for i, ev_id in enumerate(epochs.events[:, 2]):
        all_X.append(X[i])
        all_y.append(1 if int(ev_id) in _CONTROL_IDS else 0)
        all_subj.append(subj)


def _epoch_baseline_run(
    raw: mne.io.BaseRaw,
    ch_names: list[str],
    n_times: int,
    subj: int,
    all_X: list[np.ndarray],
    all_y: list[int],
    all_subj: list[int],
) -> None:
    """Slice a continuous baseline run into non-overlapping idle epochs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_picked = _pick_channels(raw, ch_names)
        data = raw_picked.get_data().astype(np.float32)  # (C, total_T)
    n_epochs = data.shape[1] // n_times
    for i in range(n_epochs):
        all_X.append(data[:, i * n_times : (i + 1) * n_times])
        all_y.append(0)  # idle
        all_subj.append(subj)


# ---------------------------------------------------------------------------
# Stage 2 — BNCI2014_009 P300 wrapper
# ---------------------------------------------------------------------------

# Native channel order for BNCI2014_009 (all 8 electrodes, dataset-defined order).
# Documented in the BNCI Horizon 2020 dataset paper; MOABB returns them in this order.
_BNCI009_CHANNELS: list[str] = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]


class BnciP300Wrapper:
    """Wraps BNCI2014_009 via MOABB's P300 paradigm; returns binary epoch arrays.

    Labels: 1 = Target, 0 = NonTarget.
    All bandpass filtering and epoching are delegated to MOABB/MNE's P300 paradigm.
    """

    def __init__(self, spec: "DatasetSpec") -> None:
        from moabb.datasets import BNCI2014_009

        self.spec = spec
        self._moabb_ds = BNCI2014_009()
        self._subject_list: list[int] = [
            s for s in self._moabb_ds.subject_list if s not in spec.exclude_subjects
        ]

    @property
    def subject_list(self) -> list[int]:
        """All valid subject IDs after applying exclude_subjects."""
        return list(self._subject_list)

    def load_epochs(self, subjects: list[int]) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load P300 target/non-target epochs for *subjects*.

        Returns
        -------
        X : (N, C, T) float32 — epochs × channels × time-points
        y : (N,) int64 — 1 = Target, 0 = NonTarget
        metadata : dict with keys 'subjects', 'channels', 'sfreq', 'n_times'
        """
        from moabb.paradigms import P300

        fmin, fmax = float(self.spec.band[0]), float(self.spec.band[1])
        tmin = float(self.spec.epoch_window[0])
        # tmax adjusted by -1/sfreq so MNE yields exactly
        # round(window_len * sfreq) samples — same formula as train.py's n_times.
        tmax = float(self.spec.epoch_window[1]) - 1.0 / self.spec.sfreq_target

        paradigm = P300(
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            resample=self.spec.sfreq_target,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, y_str, meta_df = paradigm.get_data(
                dataset=self._moabb_ds,
                subjects=list(subjects),
            )

        # X: (n_epochs, n_channels, n_times) with channels in _BNCI009_CHANNELS order.
        # Select and reorder to match spec.channels.
        ch_to_idx = {ch: i for i, ch in enumerate(_BNCI009_CHANNELS)}
        ch_idx = [ch_to_idx[ch] for ch in self.spec.channels]
        X = X[:, ch_idx, :]

        y = (y_str == "Target").astype(np.int64)

        meta: dict = {
            "subjects": list(meta_df["subject"].values),
            "channels": list(self.spec.channels),
            "sfreq": self.spec.sfreq_target,
            "n_times": X.shape[-1],
        }
        return X.astype(np.float32), y, meta
