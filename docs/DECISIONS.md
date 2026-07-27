# Architectural Decision Records

Engineering decisions made during implementation. Each entry states the decision,
rationale, and rejected alternative. Sourced from Appendix F of design.md; entries
added here during implementation as new non-obvious choices arise.

---

## ADR-1 — Stage 3 reuses the Stage 2 backbone (does not replace it)

**Decision:** One shared `BackboneEncoder`; Stage 3 = backbone + adapters + few-shot calibration.

**Rationale:** Clinical narrative ("the patient system *is* the healthy system, adapted"), DRY code, and a free zero-shot ablation (Stage 2 weights applied cold = the lower bound).

**Rejected:** Training an independent ALS model — breaks the narrative, duplicates code, loses the clean lower bound.

---

## ADR-2 — EEGNet is the production backbone; Conformer/ATCNet are comparators

**Decision:** EEGNet everywhere in the production path.

**Rationale:** Few-shot ALS adaptation rewards small, freezable, adapter-friendly models. EEGNet is real-time on CPU. The Stage 3 low-data regime is precisely where compact architectures win. A few healthy-data accuracy points are not worth the transfer fragility of heavier models.

**Rejected:** Conformer/ATCNet as default — heavier, data-hungry, fragile under PEFT, higher latency.

---

## ADR-3 — Train Stage 2 on the 8-channel intersection by default

**Decision:** Default healthy training uses the 8 channels shared with the ALS dataset.

**Rationale:** Native channel compatibility for transfer; zero channel imputation needed in Stage 3.

**Rejected:** Training on 16 channels then projecting down — introduces a channel-mismatch failure mode; kept only as an upper-bound ablation.

---

## ADR-4 — Euclidean Alignment is the default alignment everywhere it helps

**Decision:** EA as the first-line domain-shift remedy across stages.

**Rationale:** Parameter-free, label-free on target, cheap, composes with any model, repeatedly shown to help with deep decoders (Junqueira et al., 2024).

**Rejected:** Skipping alignment (leaves easy gains on the table); jumping straight to adversarial DA (heavier, less stable, unnecessary as a default).

---

## ADR-5 — Geometric pipeline retained as a mandatory Stage 3 fallback

**Decision:** xDAWN-cov + Riemannian Alignment + tangent-space LDA runs for every patient alongside the deep path; per-patient winner is recorded.

**Rationale:** Geometric methods dominate the low-data regime that ALS imposes (MOABB reproducibility study). Some weak-P300 patients will be served better by them.

**Rejected:** Deep-only — brittle on the hardest patients.

---

## ADR-6 — MOABB is the sole data-access layer

**Decision:** No script parses raw `.edf`/`.mat` files. All data access through `MoabbDatasetWrapper`.

**Rationale:** Uniform API across all three datasets, cached downloads, standardized paradigms/evaluations, fewer bespoke bugs.

**Rejected:** Hand-rolled loaders — duplicate effort, subtle paradigm/label bugs.

---

## ADR-7 — Self-paced evaluation for Stage 1 (event stream, not shuffled epochs)

**Decision:** Score Stage 1 with a streaming simulator at a fixed idle FPR.

**Rationale:** A brain switch's real cost is false activations during open-ended idle; shuffled-epoch accuracy is misleading about deployed behavior.

**Rejected:** Reporting balanced accuracy on shuffled epochs — does not reflect the asynchronous operating point.

---

## ADR-8 — Within-patient calibration-then-test for Stage 3; never pool patients

**Decision:** Strict per-patient split matching the dataset's native calibration/test words.

**Rationale:** Pooling leaks patient identity and inflates results; the clinical question is per-patient usability.

**Rejected:** Pooled cross-validation — invalid for the clinical claim.

---

## ADR-9 — Parameter-efficient fine-tuning over full fine-tuning for adaptation

**Decision:** Freeze backbone; train bottleneck adapters + head on calibration data.

**Rationale:** Tens of calibration trials cannot safely fit a whole network. PEFT caps capacity and curbs overfitting on the tiny patient sets that ALS imposes.

**Rejected:** Full fine-tune as default — overfits; kept as an ablation with `last_block_unfrozen`.

---

## ADR-10 — Channel order is a hard checkpoint contract

**Decision:** Canonical channel order stored in every checkpoint bundle; mismatches raise, never silently reindex.

**Rationale:** Silent channel reindexing is the most damaging hidden bug in cross-dataset EEG transfer. Raising on mismatch makes the error immediate and obvious.

**Rejected:** Implicit reindexing — fails silently and corrupts spatial filters.

---

## ADR-11 — `src` as the top-level package (flat layout)

**Decision:** Code lives in `src/` directly (i.e., `src.datasets.registry`, not `als_decode.datasets.registry`).

**Rationale:** Matches the §9 repository layout exactly. The project is a research repo, not a redistributed library; flat layout is simpler.

**Rejected:** Nesting under `src/als_decode/` — adds an extra import level with no benefit for a single-project codebase.

---

## ADR-12 — Baseline runs accessed via PhysionetMI private method

**Decision:** `MoabbDatasetWrapper` calls `PhysionetMI._load_one_run(subject, run)` (a private, underscore-prefixed MOABB method) for the eyes-open/eyes-closed baseline runs (1, 2).

**Rationale:** `PhysionetMI.get_data()` returns only the 6 imagined-movement task runs; it never loads runs 1–2 into the returned dict. §5.2 explicitly requires baseline runs in the idle class ("it teaches the detector that quiet, non-task EEG is not control"). The two alternative paths were (a) call `mne.io.read_raw_edf()` directly on the cached EDF paths — explicitly forbidden by the hard constraint "never parse raw .edf files directly" — or (b) drop baseline runs entirely and use only T0 rest from task runs. The user selected this path over option (b) to preserve the §5.2 data construction spec. `_load_one_run` is MOABB's own EDF-reading helper (the actual `read_raw_edf` call is inside MOABB code, not ours), and the dependency is isolated entirely to `MoabbDatasetWrapper._epoch_baseline_run`, so a future MOABB API change requires touching one place.

**Rejected:** (a) direct `mne.io.read_raw_edf` on cached paths — violates ADR-6's constraint; (b) idle = T0 rest only — deviates from §5.2.

---

## ADR-13 — Stage 3 reads its montage from the training config, never a local copy

**Decision:** `scripts/adapt_stage3.py` builds its `DatasetSpec` via
`src.models.checkpoints.spec_from_config()`, which loads `configs/dataset/*.yaml`
through the same `build_spec_from_cfg` the training path uses. The hardcoded
17-channel MI list and the hardcoded `(n_channels, n_times)` shape tables are
deleted.

**Rationale:** The script's private copy of the channel list diverged from the
config on the very first commit that introduced it and never agreed with it
afterwards (docs/AUDIT.md §0.1). Both lists were length 17, so the checkpoint
loaded without a shape error while zero of the 17 array positions matched — every
learned spatial filter was applied to the wrong electrode. A second copy of a
value that must agree with the training path is a defect regardless of whether it
currently happens to match.

**Rejected:** Keeping the local list and adding a test that compares it to the
config — a test can be updated in the same commit that breaks the invariant. One
source removes the failure mode instead of detecting it.

---

## ADR-14 — The montage is recorded in the checkpoint bundle and asserted at load

**Decision:** `MontageCheckpoint` writes `montage.json` (channel names in array
order, sfreq, n_times, paradigm) beside each Lightning version directory.
`src/datasets/montage.py::assert_montage_matches` raises `ValueError` — listing
both lists and every disagreeing position — unless the loaded data matches
exactly, in order. Checkpoints predating this file report
`montage_source="inferred_from_config"`; `--strict-montage` refuses them.

**Rationale:** Implements ADR-10, which was specified but never enforced in code.
Equal-length-different-order is the dangerous case precisely because it produces
no shape error, so it must be checked explicitly rather than left to tensor
shapes. Recording provenance for legacy checkpoints keeps the inference visible in
the output rather than silent.

**Rejected:** Reindexing the loaded data to the checkpoint's order — ADR-10
forbids it; a montage disagreement means the experiment was misconfigured, and
silently repairing it hides that.

---

## ADR-15 — P300 channels are selected by name from what MOABB returns

**Decision:** `BnciP300Wrapper.load_epochs` calls
`P300.get_data(..., return_epochs=True)` and resolves `spec.channels` by name
against `epochs.ch_names`, raising if any requested electrode is absent.

**Rationale:** BNCI2014_009 returns **16** EEG channels, not 8 as the previous
comment claimed (docs/AUDIT.md §0.2). The old code built a position map from an
8-name list and sliced positions 0–7 of the 16. That selected the intended
electrodes only because MOABB happens to return the ALS-compatible subset first.
`docs/design.md` §4.2 documents a *different* 16-channel order; had the
documentation been correct, the slice would have taken the wrong montage and
every P300 result would have been invalid. Correctness must not rest on an
undocumented ordering coincidence.

**Rejected:** Asserting the 16-channel order matches a hardcoded expectation —
still couples the code to an ordering it does not control; by-name lookup does not
care about order at all.

---

## ADR-16 — Stage 3 folds are named explicitly; no best-of-N selection

**Decision:** `_find_best_checkpoint` is removed. `--folds` names the fold(s)
explicitly and `--fold-mode {pinned,average}` chooses between one recorded fold
and aggregation across folds. The LOSO lookup raises when no fold held a subject
out, instead of falling back to the best checkpoint. Every output row records
`fold`, `checkpoint`, `checkpoint_val_auc`, `montage_source`, `n_channels`, and
`channels`.

**Rationale:** Two separate problems. (a) *Reproducibility:* selection by "highest
val AUC found on disk" made every result a function of the state of
`lightning_logs/` at run time; a discarded run silently paired EA-transformed data
with a pre-EA checkpoint and nothing in the output revealed it (docs/AUDIT.md
§0.4). (b) *Selection optimism:* taking the best of N folds and applying it to
every held-out subject selects the backbone on validation performance and then
reports the baseline as if it were arbitrary, biasing ΔAUC downward by inflating
the baseline. Averaging across folds or pinning one fold are both defensible;
best-of-N is not.

**Rejected:** Keeping best-of-N with the choice merely logged — recording a biased
estimator does not debias it.

---

## ADR-17 — One independently computed p-value per alternative

**Decision:** `scripts/analyze_stage3.py` obtains `p_two_sided` and
`p_one_sided_greater` from separate `scipy.stats.wilcoxon` calls with explicit
`alternative=`. Neither is derived from the other. Multi-seed runs are collapsed to
a per-subject mean before testing.

**Rationale:** The previous ad-hoc table doubled a one-sided p to produce a
"two-sided" p that was already two-sided, publishing 0.0077 for a result whose
two-sided p is 0.0039, and dropped the sign of the n=10 mean (docs/AUDIT.md §0.5).
Deriving one p from another by a factor of two is only valid for symmetric
continuous nulls and is never necessary when the library computes both. Treating
seeds as independent samples would inflate n by the seed count.

**Rejected:** Reporting one-sided p as the headline — the direction of the effect
was not pre-registered, and a one-sided test is not justified post hoc.

---

## ADR-18 — Chronological order is recorded, never inferred from array position

**Decision:** Both wrappers return per-epoch `source`, `run`, and `order`.
`order` is computed by `chronological_order()` from (EDF run number, position
within run), per subject. Array position is never treated as time.

**Rationale:** The array is not in acquisition order, in two independent ways.
Baseline runs 1–2 are appended last but were recorded first. And MOABB does not key
the task runs chronologically — `PhysionetMI._get_single_subject_data` emits hand
runs [4, 8, 12] as keys `'0','1','2'` and feet runs [6, 10, 14] as `'3','4','5'`,
so even within the task block the array interleaves 4, 8, 12, 6, 10, 14. Any
"temporal" split that used array position would be measuring neither acquisition
order nor anything else meaningful. `_MOABB_KEY_TO_EDF_RUN` pins the mapping.

**Rejected:** Sorting the array into chronological order at load time — it would
silently change what `temporal_array` reproduces, and the historical array order is
itself an experimental arm.

---

## ADR-19 — `stratified_task_only` holds the evaluation set fixed

**Decision:** The task-only arm restricts only the calibration *draw* to
`source == "task"`. Its evaluation set is everything not drawn, baseline epochs
included, so it has the same evaluation denominator as `stratified`.

**Rationale:** The arm exists to separate two confounded explanations of the
stratified advantage: coverage of the baseline-idle sub-population versus temporal
spread within task runs. If the arm also shrank the evaluation set to task epochs,
any difference from `stratified` could be attributed to the changed evaluation
denominator and the arm would answer nothing. Holding evaluation fixed makes
calibration coverage the only difference.

**Rejected:** Restricting both sides — reads more "consistent" and destroys the
contrast the arm was built for.

---

## ADR-20 — EA reference scope is an explicit argument, not a property of loading

**Decision:** `load_epochs` no longer applies Euclidean Alignment on the Stage 3
path; the runner forces `spec.euclidean_alignment=False` and applies the reference
itself via `--ea-ref {session,calibration}`. `session` fits over all of a subject's
epochs (reproducing the previous in-wrapper behaviour exactly); `calibration` fits
on the calibration epochs only.

**Rationale:** Fitting EA over the whole session including evaluation epochs is
transductive. It is label-free, so it is not label leakage, but it contradicts
design.md §11.1 and §7.3 and is not achievable at deployment, where the evaluation
epochs have not happened yet. Making the scope an argument turns a hidden
assumption into a measured quantity. `session` remains the default so prior results
stay comparable. Training still applies EA inside the wrapper, which is correct —
a cross-subject model legitimately sees each training subject's whole session.

**Rejected:** Switching the default to `calibration` — it would silently change
every number relative to the archived runs, and the gap between the two scopes is
itself a result worth reporting.

---

## ADR-21 — Head adaptation trains on cached frozen-backbone features

**Decision:** Under `--ea-ref session`, backbone features are computed once per
(subject, fold) and every head adaptation trains on the cached vectors.
`--ea-ref calibration` recomputes features per split.

**Rationale:** `EEGDecoder(stage="stage3_adapted")` freezes the backbone and pins it
to eval mode, so `backbone(x)` is a deterministic function of `x` alone and
`forward(x) == head(backbone(x))`. Under session-scoped EA the input is identical
across every arm, size and seed, so recomputing it is pure waste — the Phase 2 grid
is 5 arms × 6 sizes × 5 seeds × 5 folds per subject. Under calibration-scoped EA
the reference depends on the split, so the input genuinely changes and the cache
would be wrong; that path recomputes. `scripts/verify_feature_cache.py` asserts the
equivalence on a real condition rather than assuming it, and reports bitwise
identity.

**Rejected:** Caching for `calibration` too, keyed on the split — correctness
depending on a cache key is exactly the failure mode this audit exists to remove.

---

## ADR-22 — Degenerate conditions are reported, never skipped

**Decision:** When a condition cannot produce an AUC — empty evaluation set,
single-class evaluation set, calibration pool too small — the runner emits a row
with a `status` string and NaN metrics instead of `continue`. `analyze_stage3.py`
excludes them from statistics and counts them in `n_degenerate`.

**Rationale:** The old code logged a warning and skipped, so the results table had
gaps with no recorded cause, and a reader could not distinguish "not run", "run and
failed", and "run and unremarkable". Three MI conditions are structurally
unmeasurable (docs/AUDIT.md §0.8) and that fact is a finding about the experimental
design, not an absence of data.

**Rejected:** Emitting AUC 0.5 for degenerate cases — fabricates a measurement.

---

## ADR-23 — LOSO fold lookup is filtered by paradigm

**Decision:** `loso_fold_for_subject` takes `n_channels` and only considers folds
whose checkpoint has that channel count.

**Rationale:** `lightning_logs/` holds folds from both paradigms, and subject IDs
overlap — PhysionetMI uses 1–109, BNCI2014_009 uses 1–10. When MI fold 4 was
trained, its `val_subjects.json` ([2, 7, 12, …]) collided with P300 subjects 2 and
7. Strict mode raised "ambiguous" rather than silently scoring those subjects
against a 17-channel MI backbone, which is the failure it was written to catch, but
the lookup should not have been paradigm-blind in the first place.

**Rejected:** Namespacing subject IDs per paradigm in `val_subjects.json` — would
require rewriting existing metadata files; filtering on a property already present
in the checkpoint is sufficient.
