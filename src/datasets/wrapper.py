"""MoabbDatasetWrapper — sole data-access class. §8.3.

Never parse raw .edf/.mat files directly; all access goes through this class via MOABB.
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
