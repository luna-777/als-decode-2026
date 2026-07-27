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

# MOABB keys the imagined task runs '0'..'5' but does NOT key them in acquisition
# order: PhysionetMI._get_single_subject_data emits hand_runs [4, 8, 12] as keys
# '0','1','2' and then feet_runs [6, 10, 14] as keys '3','4','5'. The EDF run number
# is the chronological quantity, so it is what we record and sort on.
_MOABB_KEY_TO_EDF_RUN: dict[str, int] = {
    "0": 4, "1": 8, "2": 12,   # hand runs
    "3": 6, "4": 10, "5": 14,  # feet runs
}


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
        metadata : dict with keys 'subjects', 'channels', 'sfreq', 'n_times',
            plus the per-epoch provenance arrays 'source', 'run' and 'order'.

        Provenance
        ----------
        source : "task" | "baseline" — task runs carry imagined-movement trials and
            their interleaved rest; baseline runs 1-2 are the continuous eyes-open /
            eyes-closed recordings sliced into idle epochs.
        run : EDF run number (1, 2 for baseline; 4, 6, 8, 10, 12, 14 for task).
        order : true chronological index within the subject's recording. Array order
            is NOT chronological — baseline runs are appended last but were acquired
            first, and MOABB keys the task runs out of acquisition order (see
            _MOABB_KEY_TO_EDF_RUN). This array is the only correct time axis.
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
        prov: list[dict] = []  # one dict per epoch: source, run, within_run

        for subj in subjects:
            log.info("Loading subject %d", subj)

            # 6 imagined-movement task runs via public MOABB API.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                subj_data = self._moabb_ds.get_data(subjects=[subj])

            # Appended in MOABB key order to preserve the historical array order;
            # chronology is carried by 'order', not by position.
            for run_key, run_raw in subj_data[subj]["0"].items():
                _epoch_task_run(
                    run_raw, ch_names, tmin, tmax, subj, all_X, all_y, all_subj,
                    prov, _MOABB_KEY_TO_EDF_RUN[str(run_key)],
                )

            # Eyes-open / eyes-closed baseline runs via private MOABB method (ADR-12).
            for run_num in _BASELINE_RUN_NUMS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_base = self._moabb_ds._load_one_run(subj, run_num)
                _epoch_baseline_run(
                    raw_base, ch_names, n_times, subj, all_X, all_y, all_subj,
                    prov, run_num,
                )

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
            "source": np.array([p["source"] for p in prov]),
            "run": np.array([p["run"] for p in prov], dtype=np.int64),
            "order": chronological_order(
                np.array(all_subj),
                np.array([p["run"] for p in prov], dtype=np.int64),
                np.array([p["within_run"] for p in prov], dtype=np.int64),
            ),
        }
        return X, y, meta


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ---------------------------------------------------------------------------

def chronological_order(
    subjects: np.ndarray, runs: np.ndarray, within_run: np.ndarray
) -> np.ndarray:
    """Per-epoch chronological index within each subject's recording.

    Ranks by (run number, position within run), independently per subject, so the
    result is 0..n_subject_epochs-1 for each subject regardless of array order.
    """
    order = np.empty(len(subjects), dtype=np.int64)
    for sid in np.unique(subjects):
        mask = np.flatnonzero(subjects == sid)
        # lexsort: last key is primary
        ranked = mask[np.lexsort((within_run[mask], runs[mask]))]
        order[ranked] = np.arange(len(ranked), dtype=np.int64)
    return order


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
    prov: list[dict],
    edf_run: int,
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
        prov.append({"source": "task", "run": edf_run, "within_run": i})


def _epoch_baseline_run(
    raw: mne.io.BaseRaw,
    ch_names: list[str],
    n_times: int,
    subj: int,
    all_X: list[np.ndarray],
    all_y: list[int],
    all_subj: list[int],
    prov: list[dict],
    edf_run: int,
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
        prov.append({"source": "baseline", "run": edf_run, "within_run": i})


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


# P300 datasets this wrapper can drive. Both are 6x6 Farwell-Donchin spellers at
# 256 Hz; BNCI2014_008 is the 8-patient ALS target and BNCI2014_009 the 10-subject
# healthy source. Channel resolution is by name in both cases, so the 8-channel ALS
# montage being a subset of the healthy montage is enforced rather than assumed.
_P300_DATASETS: dict[str, str] = {
    "BNCI2014_009": "BNCI2014_009",
    "BNCI2014_008": "BNCI2014_008",
}


class BnciP300Wrapper:
    """Wraps a BNCI P300 dataset via MOABB's P300 paradigm; returns binary epochs.

    Labels: 1 = Target, 0 = NonTarget.
    All bandpass filtering and epoching are delegated to MOABB/MNE's P300 paradigm.
    The concrete dataset comes from spec.moabb_name.
    """

    def __init__(self, spec: "DatasetSpec") -> None:
        import moabb.datasets as _mds

        name = getattr(spec, "moabb_name", "BNCI2014_009")
        if name not in _P300_DATASETS:
            raise ValueError(
                f"Unsupported P300 dataset {name!r}; expected one of "
                f"{sorted(_P300_DATASETS)}"
            )
        self.spec = spec
        self._moabb_ds = getattr(_mds, name)()
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

        # Per-epoch provenance. BNCI2014_009 has 3 sessions of 1 run each; MOABB
        # returns epochs grouped by session in acquisition order, so the position
        # within a (session, run) group is its chronological index there.
        subj_arr = np.asarray(meta_df["subject"].values)
        sess_arr = np.asarray(meta_df["session"].values).astype(str)
        run_arr = np.asarray(meta_df["run"].values).astype(str)

        # Rank sessions/runs by first appearance so 'order' follows acquisition even
        # if the labels are not numerically sortable.
        seq_key = np.empty(len(subj_arr), dtype=np.int64)
        within = np.empty(len(subj_arr), dtype=np.int64)
        for sid in np.unique(subj_arr):
            m = np.flatnonzero(subj_arr == sid)
            seen: dict[tuple[str, str], int] = {}
            counter: dict[tuple[str, str], int] = {}
            for i in m:
                key = (sess_arr[i], run_arr[i])
                if key not in seen:
                    seen[key] = len(seen)
                    counter[key] = 0
                seq_key[i] = seen[key]
                within[i] = counter[key]
                counter[key] += 1

        meta: dict = {
            "subjects": list(meta_df["subject"].values),
            "channels": list(self.spec.channels),
            "sfreq": self.spec.sfreq_target,
            "n_times": X.shape[-1],
            "source_channels": returned_channels,
            # Every P300 epoch is a flash in a task run; there is no baseline-run
            # sub-population as there is for PhysionetMI. Recorded for a uniform
            # split-protocol interface across paradigms.
            "source": np.array(["task"] * len(subj_arr)),
            "session": sess_arr,
            "run": run_arr,
            "order": chronological_order(subj_arr, seq_key, within),
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
