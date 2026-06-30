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
