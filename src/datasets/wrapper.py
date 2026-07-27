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

        if self.spec.euclidean_alignment:
            from src.preprocessing.alignment import EuclideanAligner
            subj_arr = np.array(all_subj)
            for sid in np.unique(subj_arr):
                mask = subj_arr == sid
                X[mask] = EuclideanAligner().fit_transform(X[mask])
            log.info("Euclidean Alignment applied per subject (%d subjects)", len(np.unique(subj_arr)))

        meta: dict = {
            "subjects": all_subj,
            "channels": ch_names,
            "sfreq": sfreq,
            "n_times": n_times,
            # mne.Raw.pick() preserves the requested order, so the array axis order is
            # exactly ch_names. Recorded so the montage contract can be checked.
            "source_channels": ch_names,
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

# BNCI2014_009 returns 16 EEG channels, not 8. Verified against MOABB 1.5.0 for all
# 10 subjects (docs/AUDIT.md §0.2):
#   Fz Cz Pz Oz P3 P4 PO7 PO8 F3 F4 FCz C3 C4 CP3 CPz CP4
# The 8-channel ALS-compatible subset happens to occupy positions 0-7, so the previous
# hardcoded position map selected the right electrodes — by coincidence, not by
# construction. Channels are now resolved by name from what MOABB actually returns, so
# a MOABB reordering surfaces as an error rather than a silently wrong montage.
# (docs/design.md §4.2 lists a different 16-channel order; that documented order is
# wrong — see docs/AUDIT.md §0.2.)
_BNCI009_EXPECTED_16: list[str] = [
    "Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8",
    "F3", "F4", "FCz", "C3", "C4", "CP3", "CPz", "CP4",
]


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
            epochs, y_str, meta_df = paradigm.get_data(
                dataset=self._moabb_ds,
                subjects=list(subjects),
                return_epochs=True,
            )
            returned_channels = list(epochs.ch_names)
            X = epochs.get_data()

        # Resolve the requested montage by NAME against what MOABB actually returned.
        X = _select_channels_by_name(X, returned_channels, list(self.spec.channels))

        y = (y_str == "Target").astype(np.int64)
        X = X.astype(np.float32)

        if self.spec.euclidean_alignment:
            from src.preprocessing.alignment import EuclideanAligner
            subj_arr = meta_df["subject"].values
            for sid in np.unique(subj_arr):
                mask = subj_arr == sid
                X[mask] = EuclideanAligner().fit_transform(X[mask])
            log.info("Euclidean Alignment applied per subject (%d subjects)", len(np.unique(subj_arr)))

        meta: dict = {
            "subjects": list(meta_df["subject"].values),
            "channels": list(self.spec.channels),
            "sfreq": self.spec.sfreq_target,
            "n_times": X.shape[-1],
            "source_channels": returned_channels,
        }
        return X, y, meta


def _select_channels_by_name(
    X: np.ndarray, returned: list[str], requested: list[str]
) -> np.ndarray:
    """Select *requested* channels out of *X* by name, preserving requested order.

    Raises if any requested name is absent from what the dataset returned, so a
    montage change upstream surfaces as an error rather than a wrong slice.
    """
    missing = [c for c in requested if c not in returned]
    if missing:
        raise ValueError(
            f"Requested channels {missing} are not present in the data returned by "
            f"MOABB. Returned {len(returned)} channels: {returned}. "
            f"Requested {len(requested)}: {requested}."
        )
    dupes = [c for c in set(returned) if returned.count(c) > 1]
    if dupes:
        raise ValueError(f"Dataset returned duplicate channel names: {sorted(dupes)}")
    idx = [returned.index(c) for c in requested]
    return X[:, idx, :]
