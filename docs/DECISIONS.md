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

---

## ADR-24 — Montage override is an instrument, confined by a positive constraint

**Decision:** `--montage-override` is retained and promoted. Output from any run
with a bypassed montage assertion is confined to `experiments/degradation/` by a
positive check (`is_relative_to(degradation_dir)`), the effective montage and
permutation seed are recorded on every row, and `src/datasets/degradation.py`
provides the permutation instrument. `scripts/degradation_sweep.py` applies
degradation to data whose montage contract has already been asserted, rather than
bypassing the contract at all.

**Rationale:** Deleting the flag would have removed the only way to measure the
thing the audit found. The dose-response it produced (§0.10) is the result that
explains the pre-audit headline: apparent adaptation gain is a monotone function of
backbone damage, ρ = −1.000, and restoring the historical montage under the
original protocol restores +0.0863 against the published +0.0885. A destructive
capability with a recorded provenance trail and a confined output path is safer
than no capability, because the alternative is an unexplained result.

The sweep does not bypass ADR-14 at all: permuting the channel axis of a verified
load is exactly equivalent to loading a permuted list (channel selection is a pure
gather; EA commutes with permutation), so the assertion still runs and still
passes, and the degradation is an explicit transformation of verified data.

**Rejected:** Deleting the flag once the audit closed — it is the instrument, not
the residue. Bypassing the assertion inside the sweep — unnecessary, given the
equivalence, and it would have removed the guarantee that the *base* load is right.

---

## ADR-25 — Where the confound is structural, report the confound

**Decision:** `scripts/analyze_confound.py` reports collinearity diagnostics
(correlation, VIF with a properly specified auxiliary regression) alongside every
coefficient, aggregates fold and seed replicates to subject level before fitting,
and states "not separable" rather than interpreting coefficients when
|r| ≥ 0.95 or VIF ≥ 10.

**Rationale:** In PhysionetMI, calibration coverage of baseline runs and
calibration class ratio are related by an identity, not by a nuisance correlation:
baseline runs contain no positive-class epochs, so
`pos_rate = p_task × (1 − baseline_frac)`. With r = −0.997 to −1.000 the design
cannot attribute an effect to either regressor, and a cluster-robust fit that
appears to do so returns coefficients whose signs contradict the marginal
relationship. Reporting "driver unresolved" is the finding; reporting a
coefficient would be an artefact dressed as a mechanism.

**Rejected:** Dropping `calib_pos_rate` to obtain a clean coefficient on coverage —
that is the confound, not a solution to it. Reporting the cluster-robust
"significant" coefficients at N=100/150 without the sign-reversal diagnostic.

---

## ADR-26 — Operating points are selected on calibration data, never on the test block

**Decision:** `scripts/als_character.py` selects the repetition count that maximises
ITR on the *calibration* characters and reports test ITR at that k. The test-block
maximum is also emitted, under column names carrying an explicit `_PEEKING` suffix.

**Rationale:** With 20 test characters, character accuracy has a resolution of 0.05
and a standard error near 0.07, so the k that maximises test ITR is substantially a
noise draw. The test-peeked figure is ~3× the honest one (1.07 vs 0.36 bits/min) —
large enough to change the paper's conclusion about whether the speller is viable.
Emitting both, with the peeked one named so it cannot be quoted innocently, keeps
the comparison available without letting it be mistaken for a result.

**Rejected:** Reporting only the test maximum (standard in some BCI papers, and
optimistic by a factor of three here); reporting only accuracy at the full 10
repetitions (honest but discards the operating-point question the brief asked).

---

## ADR-27 — Degenerate gating conditions are "ungateable", not gated

**Decision:** When the 25% internal validation split would hold fewer than
`--val-floor` (8) trials, `scripts/gated_adapt.py` records the condition as
`ungateable` and deploys the pretrained head unchanged, rather than gating on the
available handful of trials.

**Rationale:** A gate fitted to 2–6 trials is a coin flip wearing a decision rule's
clothing, and its acceptance rate would be reported alongside genuine ones as if
comparable. Even at the sizes that clear the floor the observed acceptance rate is
48–89%, i.e. barely discriminating; extending that downward to N=10 would
manufacture a number with no content. Recording ungateable cells also makes visible
that gating is unavailable exactly where adaptation is most harmful — the smallest
calibration sets.

**Rejected:** Gating anyway with a warning; skipping the cells entirely, which
would leave the same unexplained gaps ADR-22 exists to prevent.

## ADR-28 — The band-pass config was never read; P300 and ALS were re-run, MI was proved unchanged

**Status:** accepted, 2026-08-05 · fixing commit `d90d18a` · report
[`docs/refilter_rerun.md`](refilter_rerun.md)

**Context:** Every P300 and ALS result in the repo up to commit `1934e62` was computed
under an effective ≈6.4–24 Hz passband, not the documented 1–24 Hz at 128 Hz. Two
independent faults compounded:

1. `Preprocessor.transform()` resolved the band via `self._get("band", [8.0, 30.0])`,
   but no preprocessing config defines a top-level `band` key — `mi_8_30.yaml` and
   `p300_1_24.yaml` both nest the values under `bandpass:`. The hardcoded fallback
   therefore applied to every paradigm and the whole `bandpass:` block was dead,
   `method` / `phase` / `fir_window` included.
2. The `version_31`…`version_40` P300 folds were launched against the default
   `configs/config.yaml`, whose `defaults:` list selects `preprocessing: mi_8_30`.
   `config_stage2.yaml` was correct but unused, so `keep_sfreq=160.0` was applied to
   128 Hz data and `amplitude_reject_uv=150.0` where 100.0 was intended.

An 8–30 Hz filter designed for 160 Hz applied to 128 Hz data realizes 8×128/160 = 6.4 to
30×128/160 = 24 Hz. Measured on a real epoch, output/input RMS was 0.006 at 2 Hz and
0.085 at 5 Hz — most of the P300's own frequency content was being removed.

**Decision:** Fix the resolution path only, and re-run Stage 2 and Stage 3 for P300 and
ALS. `_get_band_args()` now reads `bandpass.{low,high,method,phase,fir_window}`, with an
explicit top-level `band` key taking precedence so the ad-hoc `cfg_pp` dicts in
`scripts/evaluate.py` and `scripts/evaluate_stage2.py` keep working. No config file and
no default was changed: `config_stage2.yaml` was already correct, so the remedy for
fault 2 is an explicit run command, which keeps the fix visible at the call site.

**MI was verified unchanged rather than re-run.** `mi_8_30.yaml`'s `bandpass:` block is
8–30 Hz fir/zero/hamming, identical to the old fallback, so the fix is provably a no-op
for MI. A seeded `(32,17,320)` float32 array through `version_28/preprocessor.pkl` yields
output sha256 `e599954b933f8cd30104b3d5675a9aa764e2834cb9d30b11d8ddf80e3f8c552c` both
before and after, with `np.array_equal(before, after) == True` — exact equality, not
`allclose`. The MI rows of the regenerated `gated_summary.csv` are byte-identical to the
archived ones.

**Re-run:** ten BNCI2014_009 LOSO-10 folds (`version_42`…`version_51`) with
`preprocessing=p300_1_24 evaluation=p300_loso_full training=default seed=42`, then
`adapt_stage3.py` for p300 and als, `als_character.py`, `gated_adapt.py` for both, and
all reducers. `training=default` is pinned deliberately: `config_stage2.yaml` selects
`training: stage2` at lr 5e-4 whereas the archived runs used lr 1e-3, and pinning keeps
the preprocessing group the only variable that moved. All five row counts match the
archive exactly (1,750 / 10,000 / 1,520 / 500 / 4,000), so no grid changed.

**Archived:** the superseded P300/ALS CSVs and the two derived figures were **moved, not
deleted**, to `experiments/stage3/archive_prefilter_fix/` with a README recording the
effective passband, the date and the fixing commit — following the `archive_pre_audit/`
precedent. `git diff -M` confirms all eight CSVs are 100 %-similarity renames. Per-fold
checkpoint sha256s are recorded in `experiments/stage3/refilter_fold_manifest.csv`
because `lightning_logs/` is gitignored.

**Consequences:** Stage 2 mean val AUC 0.7764 → 0.8994 (10/10 subjects improved,
Wilcoxon p = 0.00195). P300 cohort baseline 0.716 → 0.901; adaptation remains negative
at every calibration size and still attenuates with N, at roughly a quarter of its former
magnitude. **ALS cohort baseline 0.557 → 0.711**, character accuracy 0.058 → 0.226
against a 0.028 chance floor, ITR 0.358 → 4.038 bits/min: the claim that the ALS cohort
sits near chance and thereby bounds what the study can claim no longer holds. ALS ΔAUC at
N=480 flips −0.0039 → +0.0067, but the bootstrap CI and the exact Wilcoxon p (0.0781,
n=8) disagree, so this is reported as "no longer detectably harmful", not as a gain.

**Rejected:** re-running MI as well, which would have burned compute to reproduce a
provably identical result and would have broken the read-only-on-MI constraint; changing
`configs/config.yaml`'s `preprocessing` default, which would have hidden the fault rather
than surfacing it at the call site; and bundling the FIR-filter-length fix (§F.1 of the
report — a 1–24 Hz filter needs 423 taps but a P300 epoch is 102 samples) into this
commit, which would have repeated the mistake `63ce683` made by mixing a correction with
an unrelated change.
