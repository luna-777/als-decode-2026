# EEG-Based Communication System for Locked-In Syndrome / ALS Patients
## Research & Engineering Design Specification

**Document type:** Engineering design specification (implementation-ready)
**Intended consumer:** An implementing engineer (human or Claude Code) who will receive *only* this document.
**Status:** Specification — contains no model implementation code. Every design decision is fixed here. Where a value is a tunable hyperparameter, it is given a concrete default and a search range.
**Version:** 1.0

---

## Table of Contents

1. Executive Summary
2. Overall System Architecture
3. Literature Review
4. Dataset Review
5. Stage 1 Design — Communication-Intent Detection
6. Stage 2 Design — Healthy-Subject P300 Speller
7. Stage 3 Design — Transfer Learning & Adaptation to ALS
8. Software Architecture
9. Repository Layout
10. Training Pipelines
11. Evaluation Protocol
12. Ablation Studies
13. Future Extensions
14. References

---

## 1. Executive Summary

This document specifies a clinically-motivated, three-stage electroencephalography (EEG) brain–computer interface (BCI) for communication by people with Amyotrophic Lateral Sclerosis (ALS) and Locked-In Syndrome (LIS). The three stages are deliberately designed as **one continuous clinical system** rather than three disconnected experiments. Each stage produces an artifact that the next stage consumes:

- **Stage 1 — Intent Detection (the "brain switch").** A self-paced, asynchronous detector that continuously monitors idle EEG and fires when the user *intends to communicate*. Until Stage 1 fires, the speller is dormant. This stage is framed as a binary control-vs-idle detection problem and developed on the PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB). Its dominant design constraint is a **low false-activation rate during long idle periods**, not raw accuracy.

- **Stage 2 — Healthy-Subject P300 Speller.** Once Stage 1 fires, a P300-based speller decodes the intended characters. This stage is developed on large, clean, publicly available **healthy-subject** P300 datasets (primarily BNCI 2014-009) so that a strong, well-regularized decoder can be trained before any patient data is touched. The output is a stream of predicted characters that a downstream language model (future work) can correct into words and sentences.

- **Stage 3 — Transfer to ALS Patients.** The healthy Stage 2 decoder is *adapted*, not replaced, for ALS patients using domain adaptation and minimal patient calibration. The chosen patient dataset (BNCI 2014-008) shares the Stage 2 paradigm and a subset of its channel montage, making it an unusually clean transfer target. Stage 3 specifies a complete pipeline: Euclidean/Riemannian alignment → healthy pretraining → frozen encoder + lightweight adapters → few-shot patient fine-tuning → patient-specific calibration.

The engineering stack is standardized on PyTorch + PyTorch Lightning for deep models, MNE-Python + Braindecode for EEG handling, pyRiemann for geometric methods, MOABB for reproducible dataset access and benchmarking, Hydra for configuration, and Weights & Biases for experiment tracking. MOABB is central: it exposes every dataset named here (EEGMMIDB, BNCI 2014-008, BNCI 2014-009) through a uniform API, which removes most of the dataset-engineering risk from the project.

The central design philosophy is **clinical realism over benchmark maximization**. The system is evaluated subject-independently wherever possible; false-activation rate, calibration time, information transfer rate (ITR), and latency are treated as first-class metrics alongside accuracy. The reason ALS-specific results vary enormously across patients is well documented, so Stage 3 is built to degrade gracefully and to extract the maximum benefit from a handful of calibration trials.

---

## 2. Overall System Architecture

### 2.1 End-to-end runtime flow

```
                    ┌─────────────────────────────────────────┐
                    │            CONTINUOUS EEG STREAM          │
                    │   (acquisition @ device sampling rate)    │
                    └───────────────────┬───────────────────────┘
                                        │ sliding windows
                                        ▼
                    ┌─────────────────────────────────────────┐
   STAGE 1          │   INTENT DETECTOR ("brain switch")       │
   (asynchronous)   │   idle ──────────────► CONTROL?          │
                    └───────────────────┬───────────────────────┘
                                        │ fires once (debounced)
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │       COMMUNICATION SESSION BEGINS        │
                    │   speller matrix activated, flashing on   │
                    └───────────────────┬───────────────────────┘
                                        │ stimulus-locked epochs
                                        ▼
                    ┌─────────────────────────────────────────┐
   STAGE 2          │   P300 DECODER (healthy-pretrained)       │
   (synchronous)    │   epoch → P(target) → row/col scoring     │
                    └───────────────────┬───────────────────────┘
                                        │ argmax over matrix cells
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │            PREDICTED CHARACTER            │
                    └───────────────────┬───────────────────────┘
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │   LANGUAGE MODEL (FUTURE WORK)            │
                    │   character → word/sentence correction    │
                    └───────────────────┬───────────────────────┘
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │             COMPLETED SENTENCE            │
                    └─────────────────────────────────────────┘

   STAGE 3 is not a separate runtime block. It is the *adaptation procedure*
   that turns the healthy Stage 2 decoder into a patient-specific decoder.
   At runtime the patient simply runs Stage 1 + (adapted) Stage 2.
```

### 2.2 The "Stage 3 extends Stage 2" invariant

A hard architectural constraint binds the whole project: **the Stage 3 patient decoder must be the Stage 2 decoder plus an adaptation wrapper.** Concretely:

- Stage 2 trains a backbone encoder `f_θ` and a classification head `g_φ` on healthy data.
- Stage 3 *reuses* `f_θ` (optionally frozen), inserts parameter-efficient adapter modules, and fine-tunes on a small amount of ALS data. It never instantiates a fresh, unrelated architecture.

This invariant is enforced in code by a single `BackboneEncoder` class shared by both stages, and by a `model.stage` config flag (`stage2_healthy` | `stage3_adapted`) that toggles adapter insertion and the freezing schedule. The benefit is twofold: (1) it makes the clinical narrative defensible — the patient system is literally the healthy system, adapted — and (2) it makes the engineering DRY and the ablations clean (the "no-adaptation" ablation is just Stage 2 weights applied to ALS data with no further training).

### 2.3 Three repositories of truth

The system has exactly three "sources of truth" that must never be duplicated:

1. **Data access** is owned by MOABB + a thin project wrapper. No script ever parses raw `.edf`/`.mat` files directly.
2. **Preprocessing** is owned by a single `Preprocessor` pipeline keyed by Hydra config. Stage 1, Stage 2, and Stage 3 differ only by config, not by code paths.
3. **The backbone encoder** is owned by one class, as described in §2.2.

### 2.4 Data/control contracts between stages

| Producer | Artifact | Consumer | Contract |
|---|---|---|---|
| Stage 1 | `intent_event` (timestamp, confidence) | Session manager | Fires at most once per debounce window; carries a confidence used for thresholding |
| Stage 2 | `char_prediction` (char, posterior, n_sequences_used) | Language model / UI | Posterior over 36 matrix cells; supports early stopping when posterior is confident |
| Stage 2 (healthy) | `backbone_weights.ckpt` | Stage 3 | A checkpoint of `f_θ` + normalization statistics + channel order |
| Stage 3 | `patient_decoder.ckpt` | Runtime | Same interface as Stage 2 decoder; drop-in replacement |

---

## 3. Literature Review

This section reviews the methods relevant to each stage and ends with concrete recommendations. The guiding evidence is the recent large-scale reproducibility work on MOABB and the systematic studies of alignment-based transfer learning, both of which favor a small number of robust, well-understood methods over exotic architectures.

### 3.1 Decoding architectures (shared across stages)

**Compact convolutional networks.** EEGNet (Lawhern et al., 2018) is a compact CNN using a temporal convolution, a depthwise spatial convolution (acting as learned spatial filters), and a separable convolution. It is the de-facto baseline for motor imagery, P300, and SSVEP because it is small, regularizes well on limited data, and transfers across paradigms. ShallowConvNet and DeepConvNet (Schirrmeister et al., 2017) predate EEGNet; ShallowConvNet emulates the Filter-Bank Common Spatial Patterns (FBCSP) pipeline with a squaring nonlinearity and log-pooling and remains a strong motor-imagery baseline.

**Attention / transformer hybrids.** EEG Conformer (Song et al., 2023) combines a convolutional tokenizer with a transformer encoder and reports competitive within-subject motor-imagery accuracy (≈78.7% on BCI IV-2a, ≈84.6% on BCI IV-2b in the original work). ATCNet (Altaheri et al., 2022) couples an EEGNet-like convolutional front end with multi-head self-attention, a temporal convolutional network (TCN), and a sliding-window augmentation, reporting ≈82% on BCI IV-2a. More recent convolution-transformer hybrids (e.g. CTNet, 2024) report incremental gains. FBCNet (Mane et al., 2021) uses a filter-bank front end with variance-based temporal aggregation and is strong for oscillatory (motor-imagery) signals.

**Riemannian / geometric methods.** Covariance-based pipelines operate on symmetric positive-definite (SPD) matrices in their natural Riemannian geometry. For oscillatory signals: covariance → tangent-space projection → linear classifier, or Minimum Distance to Mean (MDM). For evoked potentials such as P300, a "super-trial" / xDAWN-covariance construction injects class-discriminant templates before the tangent-space projection (xDAWN+Riemannian). The large MOABB reproducibility study found Riemannian covariance methods to be the strongest overall family, *particularly when data are limited* — directly relevant to Stage 3's small ALS datasets.

**Takeaway for architecture selection.** Across stages the recommendation is the same shape: a **classical/geometric baseline** (CSP+LDA for oscillatory, xDAWN+Tangent-Space+LDA for P300), a **deep-learning baseline** (EEGNet), and **one SOTA option** (EEG Conformer or ATCNet). EEGNet is the recommended backbone for the production pipeline because it is the only architecture that is simultaneously competitive, tiny enough for real-time inference, and adapter-friendly for Stage 3.

### 3.2 P300 detection methods (Stage 2)

The classical state of the art for P300 spellers is **stepwise linear discriminant analysis (SWLDA)** on band-pass filtered, downsampled, channel-concatenated epochs, and **xDAWN spatial filtering + LDA** (Rivet et al., 2009). The modern strong baseline is **xDAWN-covariances + tangent space + LDA/logistic regression** (the canonical pyRiemann P300 pipeline). Deep learning for P300 is dominated by EEGNet, with convolutional and transformer variants offering small, dataset-dependent gains. Character-level (as opposed to epoch-level) decoding can be improved by **Bayesian accumulation of evidence across stimulus repetitions** — e.g. accumulating Riemannian class probabilities until a confidence threshold is reached, which enables dynamic stopping and higher ITR.

### 3.3 Asynchronous / self-paced detection (Stage 1)

Stage 1 is an **asynchronous (self-paced) BCI**, also called a *brain switch*: it must distinguish an intentional control state from an open-ended *idle* state that can contain arbitrary mental activity. The historically hard part is not detecting intent but **suppressing false positives during idle periods**; the field standardly reports true-positive rate (TPR) at a fixed, low false-positive rate (FPR), e.g. TPR @ FPR = 1–5%, rather than balanced accuracy. The idle state is multi-modal (it is "everything that is not control"), so detectors are tuned conservatively and use temporal debouncing (requiring k consecutive positive windows) to trade reaction time for specificity. Brain-switch modalities in the literature include motor-imagery ERD/ERS, post-movement beta rebound, and evoked potentials (c-VEP). For a dataset-driven design with public data, motor imagery vs rest is the natural choice.

### 3.4 Transfer learning & domain adaptation (Stage 3)

The cross-subject and healthy→patient shift in EEG is primarily a **covariate shift** in the marginal distribution of the signal (different head, electrodes, impedances, disease-related cortical changes). Recommended method families, in increasing complexity:

- **Data alignment (preprocessing-level).** *Euclidean Alignment* (EA; He & Wu, 2020) whitens each subject's trials by the inverse square root of their mean covariance, making subjects statistically more similar before any classifier sees them. It is parameter-free, cheap, label-free on the target, and composes with any downstream model — the systematic evaluation of EA with deep learning found it consistently helpful. *Riemannian Alignment* / Riemannian Procrustes Analysis (Zanini et al., 2018; Rodrigues et al., 2019) is the SPD-manifold analogue, recentering (and optionally rotating/scaling) covariance matrices to a common reference.
- **Statistical feature alignment.** *CORAL* (Sun et al., 2016) aligns second-order statistics of source and target features; *Transfer Component Analysis* (TCA) aligns marginal distributions in a shared subspace.
- **Deep domain adaptation.** *Domain-Adversarial Neural Networks* (DANN; Ganin et al., 2016) train an encoder to be domain-invariant via a gradient-reversal domain discriminator. *Test-time adaptation* methods (e.g. T-TIME, 2024) adapt online with no target labels.
- **Parameter-efficient fine-tuning (PEFT).** *Adapters* (Houlsby et al., 2019), *LoRA* (Hu et al., 2021), and prompt/prefix tuning insert a small number of trainable parameters into a frozen backbone. With only tens of calibration trials per ALS patient, PEFT is the right tool: it adapts without overfitting the whole network.

**Recommended Stage 3 strategy (one complete pipeline).** Euclidean Alignment on every subject → healthy EEGNet pretraining (Stage 2) → freeze the backbone → insert bottleneck adapters → few-shot fine-tune adapters + head on a small calibration set of the target ALS patient → optional Riemannian recentering for the calibration covariances. A pure-Riemannian alternative (xDAWN-cov + Riemannian Alignment + tangent-space LDA, retrained per patient) is specified as the strong non-deep baseline, because the MOABB evidence says geometric methods win in the low-data regime that ALS patients impose.

---

## 4. Dataset Review

All three datasets are available through MOABB, which is the *only* sanctioned access path. Raw downloads are cached by MOABB; the project never re-implements parsing.

### 4.1 Stage 1 dataset — PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)

- **MOABB name:** `PhysionetMI`. **Subjects:** 109 healthy volunteers. **Channels:** 64 (international 10-10 system). **Sampling rate:** 160 Hz. **Format:** EDF+ with an annotation channel.
- **Runs:** 14 per subject. Runs 1–2 are one-minute baselines (eyes open, eyes closed). The remaining 12 runs are three repetitions each of four task conditions: real fists L/R (runs 3,7,11), imagined fists L/R (4,8,12), real both-fists/both-feet (5,9,13), imagined both-fists/both-feet (6,10,14).
- **Annotations:** `T0` = rest; `T1` = left fist or both fists (run-dependent); `T2` = right fist or both feet (run-dependent). Each task run contains ~30 trials, each with a preparation phase followed by ~4 s of (imagined) movement.
- **Known data-quality caveat:** Subjects **88, 92, 100** (and frequently **104, 106**) are excluded in the literature because of inconsistent sampling rate / annotation timing. The project **must** exclude 88, 92, 100 by default and expose the exclusion list in config.
- **Why this dataset for Stage 1:** It is the largest open motor dataset with a clean rest condition, enabling a binary *control (motor imagery) vs idle (rest)* formulation and subject-independent (cross-subject) evaluation across ~106 subjects — exactly what a generalizing brain switch needs.

### 4.2 Stage 2 dataset — BNCI 2014-009 (healthy P300 speller)

- **MOABB name:** `BNCI2014_009`. **Subjects:** 10 healthy. **Channels:** 16 (Fz, FCz, Cz, CPz, Pz, Oz, F3, F4, C3, C4, CP3, CP4, P3, P4, PO7, PO8). **Sampling rate:** 256 Hz. **Matrix:** 6×6 (36 characters). The dataset contains both overt- and covert-attention paradigms; **use the overt-attention grid speller** (MOABB loads the grid speller by default).
- **Why primary for Stage 2:** It is clean, healthy, multi-session, P300-paradigm, and — critically — it is the **same experimental lineage and hardware family** as the Stage 3 ALS dataset (BNCI 2014-008), which makes the healthy→patient transfer in Stage 3 maximally controlled.
- **Secondary / cross-dataset generalization datasets:** `BNCI2014_008` (used as target, not source), `BNCI2015_003`, the Brain Invaders datasets (`bi2014a`, `bi2015a`), and the EPFL P300 dataset. The classic BCI Competition III dataset II (subjects A & B, 64 channels) is specified as an *external* sanity-check set only.

### 4.3 Stage 3 dataset — BNCI 2014-008 (ALS P300 speller) + the healthy→ALS pairing

- **MOABB name:** `BNCI2014_008`. **Subjects:** 8 patients with ALS. **Channels:** 8 (Fz, Cz, Pz, Oz, P3, P4, PO7, PO8). **Sampling rate:** 256 Hz. **Filtering (as acquired):** band-pass 0.1–30 Hz, right-earlobe reference, left-mastoid ground. **Paradigm:** standard 6×6 Farwell–Donchin matrix; rows/columns flash for 125 ms with a 125 ms inter-stimulus interval (stimulus-onset asynchrony 250 ms); 10 sequences (repetitions) per character. Patients copy-spelled 7 five-character words (35 characters); conventionally the first 3 words are calibration and the last 4 are test.
- **Why BNCI 2014-009 → BNCI 2014-008 is the recommended pairing:**
  1. **Identical paradigm.** Both are 6×6 P300 spellers with the same flashing logic and 256 Hz sampling — no resampling or paradigm mismatch.
  2. **Channel compatibility.** The ALS montage **{Fz, Cz, Pz, Oz, P3, P4, PO7, PO8} is a strict subset** of the healthy 16-channel montage. The transfer can therefore be run on the **8-channel intersection** with zero channel imputation — a rare and valuable property for healthy→patient transfer.
  3. **Same lab lineage and hardware family** (g.tec acquisition, Santa Lucia / BNCI Horizon 2020 repository), minimizing nuisance domain shift so that the *clinically meaningful* shift (healthy vs ALS cortex) is what Stage 3 actually models.
- **Expectation management.** ALS P300 quality is strongly subject-dependent: some BNCI 2014-008 patients yield near-ceiling character accuracy while others have weak, unstable P300 morphology. Stage 3 is evaluated per-patient and must report the full distribution, never just the mean.

### 4.4 Dataset summary table

| Stage | MOABB dataset | Role | Subjects | Channels | Fs (Hz) | Paradigm |
|---|---|---|---|---|---|---|
| 1 | `PhysionetMI` | dev + eval | 109 (exclude 88/92/100) | 64 | 160 | Motor imagery vs rest |
| 2 | `BNCI2014_009` | source (healthy) | 10 | 16 | 256 | 6×6 P300 speller |
| 3 | `BNCI2014_008` | target (ALS) | 8 | 8 | 256 | 6×6 P300 speller |

---

## 5. Stage 1 Design — Communication-Intent Detection ("brain switch")

### 5.1 Goal and formulation

Stage 1 is a **self-paced, asynchronous binary detector** that runs continuously on idle EEG and outputs a single `intent` event when the user wants to start communicating. It is formulated as **control (motor imagery) vs idle (rest)** binary classification over short sliding windows, with an explicit decision layer (thresholding + temporal debouncing) on top of the per-window classifier.

The dominant requirement is **specificity during idle**: a false activation drags the user into an unwanted spelling session, which is far more costly than a missed activation (the user simply re-attempts). Therefore Stage 1 is optimized at a fixed, low false-activation operating point, not for balanced accuracy.

### 5.2 Data construction from EEGMMIDB

- **Positive class (control):** epochs from the **imagined-movement** runs (4, 8, 12, 6, 10, 14), taken from the `T1`/`T2` annotated intervals. Optionally include real-movement runs as an augmentation/ablation, but the production positive class is *imagined* movement, because the patient cannot move.
- **Negative class (idle):** epochs from `T0` rest intervals across all task runs **plus** the eyes-open/eyes-closed baseline runs (1, 2). Including the baselines is important: it teaches the detector that quiet, non-task EEG is *not* control.
- **Channel selection:** default to the **sensorimotor subset** {FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4} (configurable). A 64-channel "all-channels" config is provided for ablation.
- **Sampling / filtering:** keep native 160 Hz. Band-pass **8–30 Hz** (mu + beta) with a zero-phase FIR filter for the oscillatory motor-imagery signature; provide a 4–40 Hz broadband config for deep models that prefer wider bands.
- **Epoch extraction:** window length **2.0 s**, stride **0.1 s** for the sliding-window training set (heavy overlap to teach time-invariance and to populate the idle class). Evaluation uses a streaming simulation (see §5.7).
- **Artifact handling:** per-channel robust z-scoring; reject epochs whose peak-to-peak amplitude exceeds a configurable threshold (default 150 µV) *only in training*; at inference nothing is rejected (the stream cannot be paused), but an amplitude gate can veto a positive decision.
- **Splits:** **subject-independent** is mandatory. Use a fixed `GroupKFold` over subjects (default 5 folds) plus a held-out final test set of subjects never seen during model selection. No epoch from a test subject may appear in train/val (no leakage via overlapping windows across the split boundary).

### 5.3 Models

| Tier | Model | Rationale |
|---|---|---|
| Classical baseline | CSP (6–8 components) + LDA, or Tangent-Space LR (pyRiemann) | Strong, fast, interpretable oscillatory baseline; the geometric variant is the low-data champion |
| Deep baseline | **EEGNet** (F1=8, D=2, F2=16, kernel length 64 @160 Hz) | Compact, real-time, the recommended production detector |
| SOTA option | **ATCNet** or **EEG Conformer** | Attention-based; included for the architecture ablation, not the default |

**Recommended production model:** EEGNet, for its size and latency. The geometric baseline is retained as a guaranteed-robust fallback and as the comparator in the architecture ablation.

### 5.4 Training specification (EEGNet)

- **Loss:** binary cross-entropy with class weighting (idle is over-represented after windowing). Provide a `focal_loss` config (γ=2.0) for the high-imbalance setting.
- **Optimizer:** AdamW, weight decay 1e-2.
- **LR:** 1e-3 with cosine annealing; 10-epoch linear warmup.
- **Batch size:** 64.
- **Max epochs:** 200 with **early stopping** on validation AUC (patience 20).
- **Augmentation:** time shifting (±200 ms), additive Gaussian noise (σ scaled to per-channel std), channel dropout (p=0.1), and mixup (α=0.2) as an ablation.
- **Regularization:** dropout 0.25–0.5 (EEGNet default 0.5 for within-subject, 0.25 for cross-subject); spatial-conv max-norm constraint 1.0 as in the original EEGNet.

### 5.5 Decision layer (the actual "switch")

The per-window probability `p_t` is converted to events by:
1. **Threshold** `τ` chosen on validation to fix the idle false-positive rate (default target **FPR = 1 per minute** of idle, equivalently ≈1–2% of windows).
2. **Debounce:** require `k` consecutive windows above `τ` (default k=3, i.e. ~0.3 s of persistence) before firing. This is the primary specificity knob.
3. **Refractory period:** after firing, suppress further firing for `R` seconds (default 5 s) so a single intention produces one event.

### 5.6 Metrics

- Primary: **TPR @ fixed FPR** (report at FPR = 1%, 5%, and "per-minute" idle rate), **false-activation rate per minute of idle**, and **detection latency** (time from intent onset to event).
- Secondary: ROC-AUC, balanced accuracy, precision/recall/F1 at the chosen operating point.
- **Expected performance (cross-subject):** ROC-AUC ≈ 0.75–0.85 with EEGNet; usable TPR (≈60–80%) at single-digit FPR. Within-subject calibration raises this substantially. These are planning figures, not guarantees; the idle state's diversity makes cross-subject FPR the hardest quantity to control.

### 5.7 Real-time deployment considerations

- **Streaming simulation harness:** the evaluation must replay test recordings sample-by-sample through a ring buffer, emit a window every stride, and score the *event stream* (not shuffled epochs). This is the only valid measure of a self-paced switch.
- **Latency budget:** window (2.0 s) + debounce (0.3 s) + inference (<10 ms on CPU for EEGNet) ⇒ ≈2.3 s worst-case reaction time. Document this; it is acceptable for a "wake up" gesture.
- **Compute:** EEGNet inference is <1 GFLOP per window; CPU-only real-time is feasible. The geometric pipeline is also real-time.
- **Per-user calibration hook:** Stage 1 exposes an optional short calibration that fine-tunes `τ` (and optionally the head) on a few minutes of the specific user's rest + attempted-MI data. ALS users will use this hook.

---

## 6. Stage 2 Design — Healthy-Subject P300 Speller

### 6.1 Goal

Decode intended characters from a 6×6 P300 speller using **healthy-subject** data, producing a backbone encoder and decision logic that Stage 3 will adapt to patients. Stage 2 must be strong *and* transfer-ready: the encoder must be the same object Stage 3 reuses.

### 6.2 Data pipeline (BNCI 2014-009)

- **Epoch window:** **0 to 800 ms** post-stimulus (the standard P300 window; MOABB's default P300 interval for these datasets is ~0–1.0 s — use 0–0.8 s by default, configurable to 1.0 s).
- **Stimulus alignment:** epochs are locked to each row/column flash onset. Each flash is labeled **target** (contains the attended cell) or **non-target**. The per-character target:non-target ratio is ≈1:5 (2 of 12 flashes per repetition are targets), so the epoch-level problem is **class-imbalanced** (~17% positive).
- **Filtering:** band-pass **1–24 Hz** (P300 is a low-frequency evoked component), zero-phase. Provide 0.1–30 Hz to match the ALS acquisition band for Stage 3 consistency.
- **Downsampling:** to **128 Hz** after filtering (reduces dimensionality without losing P300 information).
- **Channel set:** train Stage 2 on the **8-channel intersection** {Fz, Cz, Pz, Oz, P3, P4, PO7, PO8} by default, so the encoder is *natively compatible* with the ALS dataset. A 16-channel config is provided for an upper-bound ablation, but the transfer-ready default is 8 channels.
- **P300 averaging:** at **decision time** (not training), epochs for the same row/column across repetitions are averaged (or evidence-accumulated) to raise SNR. Training operates on single-trial epochs to maximize sample count.
- **Artifact handling:** robust per-channel scaling fit on training data only; amplitude-based epoch rejection in training; xDAWN spatial filtering for the geometric pipeline.
- **Cross-validation / splits:** report both **within-session** (the standard MOABB P300 evaluation) and **cross-subject** (leave-one-subject-out). Cross-subject is the number that matters for the "pretrain then transfer" story.

### 6.3 Models

| Tier | Model | Notes |
|---|---|---|
| Classical baseline | **xDAWN (nfilter=4) + Tangent Space + LDA** (pyRiemann) | The canonical strong P300 pipeline; also the low-data champion for Stage 3 |
| Linear reference | SWLDA on concatenated channels | Historical P300-speller standard; cheap sanity baseline |
| Deep baseline / **production backbone** | **EEGNet** (P300 config) | Shared backbone object reused by Stage 3 |
| SOTA option | **EEG Conformer** | Architecture ablation only |

**Recommended production decoder:** EEGNet backbone, because Stage 3 requires an adapter-friendly differentiable encoder. The xDAWN+Riemannian pipeline is kept as the mandatory comparator and as the non-deep Stage 3 fallback.

### 6.4 Training specification (EEGNet, P300)

- **Loss:** binary cross-entropy with positive-class weighting (≈5:1) to counter the target/non-target imbalance.
- **Optimizer / schedule:** AdamW (wd 1e-2), LR 1e-3, cosine annealing, 5-epoch warmup.
- **Batch size:** 128.
- **Epochs:** up to 300 with early stopping on validation ROC-AUC (patience 30).
- **Augmentation:** time jitter (±1 sample at 128 Hz), Gaussian noise, channel dropout (p=0.1). Avoid aggressive augmentation that distorts the evoked waveform.
- **Class-balanced sampling:** a `WeightedRandomSampler` to keep batches from collapsing onto non-targets.

### 6.5 From epoch scores to characters (decision logic)

For each character attempt:
1. Score every flash epoch → `P(target | epoch)`.
2. Accumulate scores per row (6) and per column (6) across repetitions. Default: **average log-odds** across repetitions.
3. Predict the row with the max accumulated score and the column with the max accumulated score; their intersection is the predicted character.
4. **Dynamic stopping (recommended):** stop accumulating once the gap between the top and second row/column scores exceeds a confidence threshold, or once a Bayesian accumulation of probabilities crosses a posterior threshold. This raises ITR by spending fewer repetitions on easy characters.

### 6.6 Metrics

- **Epoch level:** ROC-AUC (primary, robust to imbalance), balanced accuracy, precision/recall/F1.
- **Character level:** character accuracy as a function of #repetitions (the P300 "learning curve"); **Information Transfer Rate (ITR)** via the Wolpaw formula using the 36-symbol alphabet, decision accuracy, and time per selection; **latency** per character.
- **Expected performance:** within-session epoch ROC-AUC ≈ 0.85–0.92 on BNCI 2014-009; character accuracy approaching 100% when enough repetitions are averaged; the interesting comparison is ITR at matched accuracy across pipelines.

### 6.7 Real-time inference

- EEGNet epoch inference is sub-millisecond; the bottleneck is stimulus timing, not compute.
- The decoder must expose a streaming API that ingests `(epoch, row/col id)` tuples and returns a running posterior so the UI can implement dynamic stopping.
- The trained healthy backbone is exported as `backbone_weights.ckpt` together with channel order, filter settings, and normalization statistics — the exact contract Stage 3 consumes.

---

## 7. Stage 3 Design — Transfer Learning & Adaptation to ALS

### 7.1 Goal

Turn the healthy Stage 2 decoder into a **patient-specific** decoder using domain adaptation plus minimal calibration, **without** discarding the healthy model. Stage 3 = Stage 2 backbone + alignment + adapters + few-shot calibration.

### 7.2 Why the chosen pairing makes this tractable

As established in §4.3: BNCI 2014-009 (healthy, source) → BNCI 2014-008 (ALS, target) share paradigm, sampling rate, hardware lineage, and a **subset channel montage** (run everything on the 8 shared channels). The remaining shift is the clinically meaningful one (ALS cortex, weaker/variable P300), which is exactly what the adaptation must absorb.

### 7.3 Recommended complete strategy (single pipeline)

```
Healthy BNCI2014_009 (8-ch)              ALS BNCI2014_008 (8-ch, per patient)
        │                                         │
        ▼                                         ▼
 Euclidean Alignment (per subject)        Euclidean Alignment (per patient)
        │                                         │
        ▼                                         │
 EEGNet backbone pretraining (Stage 2)            │
        │                                         │
        ▼                                         │
 FREEZE backbone f_θ  ◄───────────────────────────┘ (load pretrained)
        │
        ▼
 Insert bottleneck ADAPTERS into f_θ  +  re-init head g_φ
        │
        ▼
 Few-shot fine-tune (adapters + head) on N calibration characters of THIS patient
        │
        ▼
 Patient-specific calibration of decision threshold / dynamic-stopping
        │
        ▼
 patient_decoder.ckpt  (drop-in replacement for Stage 2 decoder)
```

**Components:**
- **Euclidean Alignment (EA):** compute each subject's/patient's mean spatial covariance and whiten trials by its inverse square root. Label-free on the target; applied at both train and test. This is the single highest-value, lowest-cost step.
- **Frozen backbone + bottleneck adapters:** freeze `f_θ`; insert small adapter modules (down-project → nonlinearity → up-project, with a residual) after the convolutional blocks. Only adapters + the linear head train on patient data. This is the PEFT choice that prevents overfitting tens of calibration trials.
- **Few-shot calibration:** use the patient's first **3 calibration words (15 characters)** by default — matching the dataset's native calibration/test split — and report performance as a function of calibration size (see ablations).
- **Optional Riemannian recentering** of the calibration covariances for the geometric fallback path.

### 7.4 The mandatory non-deep alternative (strong baseline)

xDAWN-covariances + **Riemannian Alignment** (recenter target covariances to the source reference) + tangent-space LDA, refit per patient. Specified because the low-data ALS regime is precisely where geometric methods have been shown to win; this path needs no pretraining and is the safety net if the deep adapter path underperforms on weak-P300 patients.

### 7.5 Training pipeline specification

- **Healthy pretraining:** as in §6.4 (this *is* Stage 2).
- **Adapter fine-tuning:**
  - Trainable params: adapters + head only (backbone frozen). Provide a `last_block_unfrozen` config that additionally unfreezes the final conv block for the higher-data ablation.
  - **Optimizer:** AdamW, weight decay 1e-3.
  - **LR:** 3e-4 for adapters/head (lower than pretraining; small data). Cosine schedule, short warmup.
  - **Epochs:** ≤100 with early stopping on a *patient-held-out* calibration split (patience 15).
  - **Losses:** weighted BCE; optionally add a **CORAL** or **MMD** feature-alignment term (λ default 0.1) between healthy features and patient features as a domain-regularizer; provide a **DANN** config (gradient-reversal domain head) as an ablation.
  - **Regularization:** strong dropout (0.5) in adapters, early stopping, and small adapter bottleneck (default reduction factor 8) to cap capacity.
- **Calibration trials:** default **N = 15 characters** (3 words). Ablate N ∈ {3, 5, 10, 15, 25, 35} to produce the calibration-cost curve — a key clinical deliverable (how long must a patient train before the system is usable).
- **Evaluation protocol:** strictly **within-patient, calibration-then-test** (calibrate on the patient's calibration words, test on their held-out test words). Never pool patients into a single split. Report per-patient and the full distribution.

### 7.6 Metrics

Epoch ROC-AUC and character accuracy per patient; **calibration time** (minutes of patient data to reach a target accuracy); ITR; false-activation interplay with Stage 1; and the **adaptation gain** = adapted accuracy − zero-shot (healthy-model-applied-cold) accuracy. The zero-shot number is just the Stage 2 weights run on ALS data with no fine-tuning — the cleanest possible ablation, enabled by the §2.2 invariant.

### 7.7 Expected outcomes and honest caveats

Adaptation gains will be large for weak-P300 patients and small for already-strong patients (ceiling). Some patients may not reach reliable spelling regardless of method — this must be reported, not hidden. EA + few-shot adapters is expected to beat cold transfer and to be competitive with, or complementary to, the Riemannian-alignment baseline; the project should report whichever wins per patient and consider a simple per-patient model-selection rule.

---

## 8. Software Architecture

### 8.1 Principles

- **Config-driven, not code-driven.** Stages differ by Hydra config, not by forked code. A new experiment is a new YAML, not a new script.
- **One backbone, one preprocessor, one data layer** (the three sources of truth from §2.3).
- **Reproducibility by construction.** Every run is fully specified by its resolved config, a global seed, and pinned dependency versions; the resolved config and git SHA are logged to W&B and saved beside every checkpoint.
- **Leakage-proof by design.** Splitters operate on *groups* (subjects/patients), and normalization statistics are fit on training folds only, inside the pipeline, never globally.

### 8.2 Engineering stack (recommended and justified)

| Tool | Role | Why |
|---|---|---|
| **Python 3.11** | Language | Modern typing, broad EEG-ecosystem support |
| **PyTorch 2.x** | Deep learning | Standard; `torch.compile` for speed; ecosystem support |
| **PyTorch Lightning 2.x** | Training loop | Removes boilerplate; built-in checkpointing, early stopping, mixed precision, multi-GPU; clean separation of model/data/loop |
| **MNE-Python** | EEG I/O, filtering, epoching | The reference EEG library; everything below interoperates with it |
| **Braindecode** | EEG model zoo + dataset glue | Provides reference EEGNet, ShallowConvNet, DeepConvNet, EEGConformer, ATCNet implementations and MNE↔PyTorch bridges — avoids re-implementing architectures |
| **MOABB** | Dataset access + benchmarking | Uniform API to EEGMMIDB, BNCI 2014-008/009; standardized paradigms and evaluations; the sanctioned data layer |
| **pyRiemann** | Geometric methods | xDAWN, XdawnCovariances, TangentSpace, MDM, alignment — the classical baselines and the Riemannian Stage 3 path |
| **Hydra** | Configuration | Composable, override-from-CLI configs; structured config groups per stage/model/dataset |
| **Weights & Biases** | Experiment tracking | Metrics, configs, artifacts, sweep orchestration; offline mode supported |
| **NumPy / SciPy / scikit-learn** | Numerics, classical ML, metrics | Splitters, LDA/LR, ROC-AUC, calibration |
| **pytest** | Testing | Unit + integration tests (see §8.5) |

Pin all versions in `pyproject.toml`/`requirements.txt`. Braindecode and pyRiemann supply tested reference implementations, so the project should **prefer importing these over re-implementing** any standard architecture or geometric estimator.

### 8.3 Core abstractions (class specifications — interfaces only, no method bodies)

> These are interface contracts. The implementing engineer writes the bodies; signatures, responsibilities, and invariants are fixed here.

**`datasets/registry.py`**
- `class DatasetSpec` — dataclass: `moabb_name: str`, `paradigm: Literal["mi","p300"]`, `channels: list[str]`, `sfreq_target: float`, `exclude_subjects: list[int]`, `band: tuple[float,float]`, `epoch_window: tuple[float,float]`.
- `def get_dataset(spec: DatasetSpec) -> MoabbDatasetWrapper` — returns a wrapper exposing `get_data(subjects)` and `subject_list`.

**`datasets/wrapper.py`**
- `class MoabbDatasetWrapper` — wraps a MOABB dataset + paradigm. Methods: `load_epochs(subjects) -> Epochs/arrays`, `subject_list -> list[int]`. Responsible for channel selection to the configured intersection and for honoring `exclude_subjects`. **Single source of truth for data access.**

**`preprocessing/pipeline.py`**
- `class Preprocessor` — config-built, fit/transform interface. Steps: band-pass filter, resample, channel pick/reorder, robust scaling (fit on train only), optional EA/RA alignment, optional xDAWN. `fit(X_train)`, `transform(X)`. **Single source of truth for preprocessing.** Must be picklable and saved with each model.

**`models/backbone.py`**
- `class BackboneEncoder(nn.Module)` — the shared encoder (default EEGNet body). `forward(x) -> features`. Accepts `insert_adapters: bool` and a `freeze()` method. **Single source of truth for the encoder (the §2.2 invariant).**
- `class Adapter(nn.Module)` — bottleneck adapter (down/act/up + residual), reduction factor configurable.
- `class DecoderHead(nn.Module)` — linear (or small MLP) classification head.
- `class EEGDecoder(nn.Module)` — `BackboneEncoder` + `DecoderHead`; `stage` flag toggles adapters/freezing.

**`models/classical.py`**
- `def build_p300_riemann_pipeline(...) -> sklearn.Pipeline` — xDAWN(+cov) → TangentSpace → LDA/LR.
- `def build_mi_csp_pipeline(...) -> sklearn.Pipeline` — CSP → LDA, and a tangent-space variant.

**`training/lit_module.py`**
- `class LitEEG(pl.LightningModule)` — wraps `EEGDecoder`, loss, optimizer/scheduler, metrics. Hooks: `training_step`, `validation_step`, `test_step`, `configure_optimizers`. Reads everything from config.

**`training/datamodule.py`**
- `class EEGDataModule(pl.LightningDataModule)` — owns split logic (GroupKFold / LOSO / calibration-then-test), samplers (WeightedRandomSampler), and applies the `Preprocessor` per fold without leakage.

**`adaptation/align.py`**
- `def euclidean_alignment(X, ref=None) -> X_aligned, ref` and `def riemannian_alignment(covs, ref=None)`. Pure functions; ref returned for reuse on test.

**`adaptation/finetune.py`**
- `def adapt_to_patient(backbone_ckpt, patient_calib_data, cfg) -> patient_decoder` — implements the §7.3 pipeline: load, freeze, insert adapters, few-shot fit, calibrate threshold.

**`evaluation/metrics.py`**
- `def epoch_metrics(y, scores) -> dict` (ROC-AUC, balanced acc, P/R/F1).
- `def character_accuracy(...)`, `def itr_wolpaw(accuracy, n_classes, time_per_selection) -> float`.
- `def asynchronous_metrics(event_stream, ground_truth) -> dict` (TPR@FPR, false-activations/min, latency).

**`evaluation/streaming.py`**
- `class StreamSimulator` — replays a recording sample-by-sample, applies the sliding window + decision layer, emits events for Stage 1 scoring.

### 8.4 Configuration model (Hydra groups)

```
configs/
  config.yaml                # top-level defaults list
  stage/                     # stage1.yaml | stage2.yaml | stage3.yaml
  dataset/                   # physionet_mi.yaml | bnci2014_009.yaml | bnci2014_008.yaml
  preprocessing/             # mi_8_30.yaml | p300_1_24.yaml | ...
  model/                     # eegnet.yaml | shallowconvnet.yaml | atcnet.yaml | conformer.yaml | riemann_p300.yaml | csp_lda.yaml
  adaptation/                # none.yaml | ea.yaml | ea_adapters.yaml | ra_riemann.yaml | dann.yaml | coral.yaml
  training/                  # default.yaml (optimizer, scheduler, epochs, early_stop)
  evaluation/                # within_session.yaml | loso.yaml | calib_then_test.yaml | streaming.yaml
  experiment/                # composed end-to-end experiments (one file per paper figure)
```

A full run is, e.g.:
`python -m src.train experiment=stage3_ea_adapters dataset=bnci2014_008 model=eegnet adaptation=ea_adapters training.calib_chars=15`

### 8.5 Testing & quality gates

- **Unit tests:** preprocessing shape/no-NaN invariants; EA whitening (aligned mean covariance ≈ identity); adapter parameter-count and freeze correctness; ITR formula against known values.
- **Leakage tests:** assert no subject/group appears in two splits; assert scaler/EA reference is fit only on train indices.
- **Integration smoke test:** a tiny end-to-end run (2 subjects, 2 epochs) per stage in CI.
- **Determinism test:** same seed ⇒ same metrics within tolerance.

---

## 9. Repository Layout

```
eeg-lis-communication/
├── pyproject.toml                 # deps pinned; project metadata
├── README.md                      # quickstart, reproduce-the-paper commands
├── Makefile                       # make stage1 | stage2 | stage3 | test | lint
├── environment.yml                # conda env (mne, braindecode, moabb, pyriemann, ...)
│
├── configs/                       # Hydra config tree (see §8.4)
│   ├── config.yaml
│   ├── stage/ dataset/ preprocessing/ model/ adaptation/ training/ evaluation/ experiment/
│
├── src/
│   ├── __init__.py
│   ├── train.py                   # Hydra entrypoint: build datamodule+model+trainer, run
│   ├── evaluate.py                # load checkpoint(s), run an evaluation protocol, log
│   ├── adapt.py                   # Stage 3 entrypoint: healthy ckpt -> patient decoder
│   │
│   ├── datasets/
│   │   ├── registry.py            # DatasetSpec, get_dataset
│   │   ├── wrapper.py             # MoabbDatasetWrapper (sole data access)
│   │   └── splits.py              # GroupKFold, LOSO, calibration-then-test splitters
│   │
│   ├── preprocessing/
│   │   ├── pipeline.py            # Preprocessor (sole preprocessing)
│   │   ├── filters.py             # band-pass, notch, resample wrappers (MNE)
│   │   └── artifacts.py           # robust scaling, amplitude rejection/gating
│   │
│   ├── models/
│   │   ├── backbone.py            # BackboneEncoder, Adapter, DecoderHead, EEGDecoder
│   │   ├── zoo.py                 # thin wrappers around braindecode models for ablations
│   │   └── classical.py           # pyRiemann/sklearn pipelines (P300 + MI)
│   │
│   ├── training/
│   │   ├── lit_module.py          # LitEEG (LightningModule)
│   │   ├── datamodule.py          # EEGDataModule (LightningDataModule)
│   │   ├── losses.py              # weighted BCE, focal, CORAL/MMD, DANN reversal
│   │   └── callbacks.py           # early stopping, checkpoint, LR monitor, W&B
│   │
│   ├── adaptation/
│   │   ├── align.py               # euclidean_alignment, riemannian_alignment
│   │   ├── finetune.py            # adapt_to_patient (the §7.3 pipeline)
│   │   └── strategies.py          # registry: none | ea | ea_adapters | ra_riemann | dann | coral
│   │
│   ├── decoding/
│   │   ├── p300_decision.py       # epoch scores -> row/col accumulation -> character
│   │   ├── dynamic_stopping.py    # Bayesian/confidence early-stop logic
│   │   └── intent_switch.py       # Stage 1 decision layer (threshold/debounce/refractory)
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # epoch/character/ITR/asynchronous metrics
│   │   ├── streaming.py           # StreamSimulator
│   │   └── report.py              # per-subject tables, distribution plots, CIs
│   │
│   └── utils/
│       ├── seed.py                # global determinism
│       ├── logging.py             # W&B + console
│       └── io.py                  # checkpoint bundling (weights + preprocessor + meta)
│
├── scripts/
│   ├── download_data.py           # warm MOABB caches for all datasets
│   ├── run_stage1.sh              # reproduce Stage 1 results
│   ├── run_stage2.sh              # reproduce Stage 2 results
│   ├── run_stage3.sh              # reproduce Stage 3 results (per patient)
│   └── run_ablations.sh           # launch all ablation sweeps
│
├── experiments/                   # W&B sweep YAMLs, one per ablation in §12
│
├── results/                       # gitignored; metrics tables, figures, exported checkpoints
│
├── tests/
│   ├── test_preprocessing.py  test_alignment.py  test_adapters.py
│   ├── test_splits_no_leakage.py  test_metrics.py  test_smoke_end_to_end.py
│
└── docs/
    ├── DESIGN.md                  # this document
    ├── DATASETS.md                # per-dataset notes, caveats, citations
    ├── REPRODUCE.md               # exact commands per figure/table
    └── DECISIONS.md               # ADRs: log of every non-obvious engineering choice
```

**Module-by-module responsibilities** are fixed: `datasets/` only loads, `preprocessing/` only transforms, `models/` only defines networks, `training/` only orchestrates fitting, `adaptation/` only handles domain shift, `decoding/` only converts model outputs to user-facing predictions, `evaluation/` only scores. No module reaches across these boundaries.

---

## 10. Training Pipelines

### 10.1 Stage 1 pipeline (subject-independent MI-vs-rest)

1. `download_data.py` warms the `PhysionetMI` cache.
2. `EEGDataModule` builds GroupKFold-over-subjects splits (excluding 88/92/100) and a held-out test set of subjects.
3. `Preprocessor` (config `mi_8_30`): band-pass 8–30 Hz → keep 160 Hz → sensorimotor channels → robust scaling (train-fit) → sliding 2.0 s/0.1 s windows.
4. `LitEEG` trains EEGNet with weighted BCE, AdamW, cosine LR, early stop on val AUC.
5. `decoding/intent_switch.py` fits `τ` to the target idle FPR on validation; `StreamSimulator` evaluates the event stream on the test subjects.
6. Log TPR@FPR, false-activations/min, latency, AUC to W&B.

### 10.2 Stage 2 pipeline (healthy P300)

1. Warm `BNCI2014_009`.
2. Splits: within-session (MOABB default) **and** LOSO for the transfer-relevant number.
3. `Preprocessor` (config `p300_1_24`): band-pass 1–24 Hz → resample 128 Hz → 8-channel intersection → robust scaling → 0–0.8 s epochs, target/non-target labels.
4. Train (a) the xDAWN+TangentSpace+LDA pipeline and (b) the EEGNet backbone (weighted BCE, WeightedRandomSampler, early stop on AUC).
5. `decoding/p300_decision.py` + `dynamic_stopping.py` produce character accuracy vs repetitions and ITR.
6. Export `backbone_weights.ckpt` bundle (weights + `Preprocessor` + channel order + meta).

### 10.3 Stage 3 pipeline (per-ALS-patient adaptation)

For each of the 8 BNCI 2014-008 patients independently:
1. Load the healthy `backbone_weights.ckpt`.
2. `Preprocessor` reproduces the Stage 2 transform on the patient's 8 channels; apply **EA** using the patient's calibration covariances as reference.
3. Split: calibration words (default first 3) vs test words (last 4) — the dataset's native split; **never** mix patients.
4. `adapt_to_patient`: freeze backbone, insert adapters, few-shot fine-tune adapters+head (AdamW, LR 3e-4, early stop on a calibration-internal val split), optional CORAL/MMD regularizer.
5. Calibrate the dynamic-stopping threshold on calibration data.
6. Evaluate on the patient's test words; log per-patient epoch AUC, character accuracy, ITR, adaptation gain vs zero-shot.
7. Also run the Riemannian-alignment baseline path for the same patient and record the per-patient winner.

### 10.4 Cross-cutting requirements

- **Mixed precision** for deep training; `torch.compile` where stable.
- **Seeding** at process start; log the seed.
- **Checkpoint bundles** always include the fitted `Preprocessor` and channel order so inference is reproducible and leakage-free.
- **Sweeps** are W&B-orchestrated from `experiments/` YAMLs; no hand-editing of configs for ablations.

---

## 11. Evaluation Protocol

### 11.1 Anti-leakage rules (non-negotiable)

- **Group-aware splitting only.** Subjects (Stage 1, Stage 2-LOSO) and patients (Stage 3) are the grouping unit. No subject/patient spans splits.
- **No windowing leakage.** For Stage 1's overlapping windows, the train/val/test boundary is enforced at the subject level *before* windowing, so no two overlapping windows straddle the boundary.
- **Fit-on-train-only.** Scalers, EA/RA references, xDAWN filters, and dynamic-stopping thresholds are fit on training/calibration data only and reused unchanged at test.
- **Within-patient calibration-then-test for Stage 3.** Pooling patients into one CV split is forbidden — it leaks patient identity and inflates results.
- **k-fold caution.** Naïve k-fold over epochs (ignoring session/subject structure) is known to over-estimate EEG accuracy and is prohibited for reported numbers.

### 11.2 Evaluation settings per stage

| Stage | Primary setting | Also report |
|---|---|---|
| 1 | Cross-subject (LOSO/GroupKFold) + streaming event scoring | Within-subject after per-user calibration |
| 2 | Within-session (MOABB standard) | Cross-subject (LOSO) |
| 3 | Within-patient calibration-then-test | Zero-shot (cold transfer) as the lower bound |

### 11.3 Metric suite (computed for every reported run)

Accuracy, **Balanced Accuracy**, **ROC-AUC** (primary for imbalanced epoch tasks), Precision, Recall, **F1**, **Information Transfer Rate** (Wolpaw), **Calibration Time** (minutes of patient data to reach a target accuracy), **False-Activation Rate** (Stage 1), **Latency** (per decision), **Memory Usage**, **Model Size** (parameter count + on-disk bytes). Report **per-subject/per-patient distributions** with 95% confidence intervals (bootstrap), not just means — consistent with the MOABB benchmarking methodology.

### 11.4 Statistical reporting

- Bootstrap 95% CIs over subjects/patients for every headline metric.
- For pipeline comparisons, paired tests across subjects (e.g. Wilcoxon signed-rank) with multiple-comparison correction, mirroring the MOABB study's permutation-based approach.
- Always report the comparator pipelines (classical/geometric vs EEGNet vs SOTA) on identical splits.

---

## 12. Ablation Studies

Each ablation is a W&B sweep defined in `experiments/`. All use the §11 protocol.

1. **Stage 1 architecture:** CSP+LDA vs Tangent-Space LR vs EEGNet vs ATCNet vs EEG Conformer. Outcome: accuracy, FPR control, latency, model size. Hypothesis: EEGNet is the best accuracy/latency trade-off; geometric methods are competitive and robust.
2. **Stage 1 band / channels:** 8–30 Hz vs 4–40 Hz; sensorimotor subset vs 64 channels. Outcome: effect on cross-subject FPR.
3. **Stage 2 architecture:** xDAWN+Riemannian vs SWLDA vs EEGNet vs EEG Conformer. Outcome: epoch AUC, character accuracy vs repetitions, ITR.
4. **Stage 2 channels:** 8-channel intersection vs full 16 channels. Outcome: the cost of transfer-readiness (how much accuracy the 8-channel constraint sacrifices).
5. **Transfer-learning method (Stage 3):** none (zero-shot) vs EA-only vs EA+adapters vs full fine-tune vs CORAL vs DANN vs Riemannian-Alignment pipeline. Outcome: adaptation gain per patient.
6. **Calibration size (Stage 3):** N ∈ {3,5,10,15,25,35} characters → the **calibration-cost curve** (clinical headline: minimum training burden per patient).
7. **Cross-subject generalization (Stage 2):** within-session vs LOSO gap.
8. **Cross-session generalization:** train/test across sessions where datasets provide multiple sessions (BNCI 2014-009 is multi-session).
9. **Effect of alignment:** EA on vs off, holding everything else fixed, across stages — isolating the value of the single cheapest intervention.
10. **PEFT capacity:** adapter reduction factor ∈ {4,8,16} and `last_block_unfrozen` on/off — capacity vs overfitting on tiny patient data.

Each ablation must produce a table + a per-subject distribution plot and an entry in `docs/DECISIONS.md`.

---

## 13. Future Extensions

- **Language model decoding.** Insert an n-gram or neural LM (or a small LLM) between Stage 2 character posteriors and the final text, turning character-level posteriors into word/sentence beams — large ITR gains, and the natural place for context, word completion, and error correction. The Stage 2 streaming posterior API is already designed to feed it.
- **Online/continual adaptation.** Add test-time adaptation (e.g. entropy-minimization / T-TIME-style) so the patient decoder tracks within- and across-session drift without re-calibration.
- **Hybrid Stage 1.** Combine motor-imagery and evoked-potential brain switches (e.g. c-VEP) for lower false-activation idle performance.
- **Additional patient datasets.** Generalize Stage 3 beyond BNCI 2014-008 (e.g. other ALS/DOC P300 corpora) and report cross-dataset patient transfer.
- **Calibration-free target.** Push toward usable zero-calibration performance via large multi-dataset healthy pretraining + strong alignment, reducing patient burden to near zero.
- **Clinical deployment harness.** Real-time acquisition integration (LSL), an accessibility-first speller UI, caregiver setup tooling, and on-device (CPU/edge) inference — the path from this research system to a bedside device.

---

## 14. References

*(Representative key references grounding the design decisions above; the implementing engineer should consult these for exact hyperparameters and reported numbers.)*

1. Lawhern V. et al. (2018). *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.* Journal of Neural Engineering 15(5):056013.
2. Schirrmeister R. et al. (2017). *Deep learning with convolutional neural networks for EEG decoding and visualization.* Human Brain Mapping 38(11):5391–5420. (ShallowConvNet / DeepConvNet; Braindecode.)
3. Song Y. et al. (2023). *EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization.* IEEE TNSRE 31:710–719.
4. Altaheri H. et al. (2022). *Physics-informed attention temporal convolutional network for EEG-based motor imagery classification (ATCNet).* IEEE Transactions on Industrial Informatics.
5. Mane R. et al. (2021). *FBCNet: A Multi-view Convolutional Neural Network for Brain–Computer Interface.* arXiv:2104.01233.
6. Rivet B. et al. (2009). *xDAWN algorithm to enhance evoked potentials: application to brain–computer interface.* IEEE TBME 56(8):2035–2043.
7. Barachant A., Congedo M. et al. *pyRiemann*; Congedo M., Barachant A., Bhatia R. (2017). *Riemannian geometry for EEG-based brain–computer interfaces: a primer and a review.* Brain-Computer Interfaces 4(3).
8. Aristimunha B., Chevallier S. et al. (2024). *The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark.* (Riemannian methods strongest overall, especially with limited data.)
9. Eder M. et al. (2024). *Benchmarking BCI algorithms: Riemannian approaches vs convolutional neural networks.* Journal of Neural Engineering 21:044002.
10. Riccio A. et al. (2013). *Attention and P300-based BCI performance in people with ALS.* (BNCI 2014-008 / 008-2014 dataset; Frontiers in Human Neuroscience, doi:10.3389/fnhum.2013.00732.)
11. Aricò P. / Santa Lucia group. *BNCI 2014-009* P300 speller dataset (BNCI Horizon 2020 repository).
12. Schalk G. et al. (2004). *BCI2000: a general-purpose brain–computer interface system.* IEEE TBME. (EEGMMIDB / PhysioNet, Goldberger et al., 2000.)
13. He H., Wu D. (2020). *Transfer learning for brain–computer interfaces: a Euclidean space data alignment approach.* IEEE TBME 67(2):399–410.
14. Zanini P. et al. (2018). *Transfer learning: a Riemannian geometry framework with applications to BCIs.* IEEE TBME 65(5):1107–1116. (Riemannian Alignment; Rodrigues et al., 2019, Riemannian Procrustes Analysis.)
15. Sun B. et al. (2016). *Return of Frustratingly Easy Domain Adaptation (CORAL).* AAAI.
16. Ganin Y. et al. (2016). *Domain-Adversarial Training of Neural Networks (DANN).* JMLR.
17. Houlsby N. et al. (2019). *Parameter-Efficient Transfer Learning for NLP (Adapters).* ICML. Hu E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
18. Wu D., Xu Y., Lu B.-L. (2022). *Transfer learning for EEG-based BCIs: a review of progress made since 2016.* IEEE TCDS 14(1):4–19. Wu D., Jiang X., Peng R. (2022). *Transfer learning for motor-imagery BCIs: a tutorial.* Neural Networks 153:235–253.
19. Junqueira B., Aristimunha B. et al. (2024). *A systematic evaluation of Euclidean Alignment with deep learning for EEG decoding.* arXiv:2401.10746.
20. Barachant A. et al. (2022). *End-to-end P300 BCI using Bayesian accumulation of Riemannian probabilities.* arXiv:2203.07807.
21. Wolpaw J. et al. (2002). *Brain–computer interfaces for communication and control.* Clinical Neurophysiology (ITR definition).
22. Bashashati A. et al. (2007). *Towards development of a 3-state self-paced BCI.* Computational Intelligence and Neuroscience. (Asynchronous brain switch; TPR @ fixed FPR methodology.)

---

*End of specification.*

---

## Appendix A — Architecture Comparison Detail

This appendix expands the §3 recommendations with the explicit pros/cons that justify the three-tier selection (classical/geometric baseline, deep baseline, SOTA option) for each stage. It exists so the implementing engineer understands *why* a model is in or out, and can defend the choice.

### A.1 Candidate architectures (shared pool)

| Architecture | Family | Params (typical) | Strengths | Weaknesses | Verdict for this project |
|---|---|---|---|---|---|
| **CSP + LDA** | Classical (oscillatory) | ~hundreds | Fast, interpretable, strong for motor imagery; tiny data footprint | Oscillatory only (not evoked); sensitive to artifacts; needs band selection | **Stage 1 classical baseline** |
| **xDAWN + Tangent Space + LDA** | Geometric (evoked) | ~thousands | The strongest low-data P300 pipeline; robust; near-SOTA without deep learning | Not end-to-end differentiable; adapter-unfriendly | **Stage 2 classical baseline & Stage 3 non-deep fallback** |
| **SWLDA** | Linear (evoked) | linear | Historical P300-speller standard; trivially cheap | Weaker than geometric/deep; manual feature engineering | Stage 2 linear reference only |
| **EEGNet** | Compact CNN | ~2k–10k | Tiny, real-time, cross-paradigm, regularizes on small data, adapter-friendly | Slightly below heavy attention models on large within-subject data | **Production backbone, all stages** |
| **ShallowConvNet** | CNN (FBCSP-like) | ~40k | Strong motor-imagery baseline; emulates FBCSP | Larger; tuned for oscillatory, less for P300 | Stage 1 ablation |
| **DeepConvNet** | CNN | ~150k+ | Higher capacity | Overfits small EEG sets; needs more data | Ablation only |
| **FBCNet** | Filter-bank CNN | ~10k–20k | Excellent oscillatory SNR via filter bank + variance layer | Oscillatory-specialized; not for P300 | Stage 1 ablation |
| **ATCNet** | CNN + MHA + TCN | ~50k–120k | SOTA-class motor imagery; attention + temporal conv; built-in augmentation | Heavier; more hyperparameters; latency | **Stage 1 SOTA option** |
| **EEG Conformer** | Conv + Transformer | ~150k–800k | Strong within-subject MI & P300; global context | Data-hungry; heaviest; latency/memory cost | **Stage 1/2 SOTA option** |
| **Graph Neural Nets** | GNN | varies | Models electrode topology explicitly | Immature tooling; inconsistent gains; harder to deploy real-time | Out of scope (future) |

### A.2 Why EEGNet is the production backbone (not the SOTA models)

The decisive factor is **Stage 3**. The whole project hinges on adapting a healthy model to ALS patients with *tens* of calibration trials. That regime rewards (a) small parameter count, (b) clean separability into a freezable backbone + small head, and (c) adapter insertion points. EEGNet's depthwise/separable structure provides exactly these: the temporal + depthwise-spatial convolutions form a natural frozen feature extractor, and adapters slot cleanly after each block. EEG Conformer and ATCNet can win by a few accuracy points on large within-subject healthy data, but they are heavier, slower, more data-hungry (bad for ALS few-shot), and their attention stacks are more fragile under PEFT. They are therefore included as **ablation/SOTA comparators**, not as the production path. This is a deliberate, clinically-motivated trade of a small healthy-data accuracy margin for transfer robustness and real-time deployability.

### A.3 Why geometric methods are kept as first-class baselines

The largest open EEG benchmark to date found Riemannian covariance methods to be the strongest *overall* family and the most robust when data are scarce. ALS patient data is scarce by definition. A pipeline that ignored geometric methods would be leaving the most reliable low-data tool on the table. Hence: xDAWN+Tangent-Space+LDA is a mandatory Stage 2 comparator and the Stage 3 non-deep fallback, and a per-patient model-selection rule (deep-adapted vs Riemannian-aligned) is explicitly allowed, since the winner is patient-dependent.

### A.4 EEGNet reference configuration (all stages)

| Hyperparameter | Stage 1 (MI) | Stage 2/3 (P300) |
|---|---|---|
| Temporal filters `F1` | 8 | 8 |
| Depth multiplier `D` | 2 | 2 |
| Pointwise filters `F2` | 16 | 16 |
| Temporal kernel length | 64 (≈0.4 s @160 Hz) | 64 (≈0.5 s @128 Hz) |
| Pooling | avg, 4 then 8 | avg, 4 then 8 |
| Dropout | 0.25 (x-subject) / 0.5 (within) | 0.5 |
| Spatial-conv max-norm | 1.0 | 1.0 |
| Dense max-norm | 0.25 | 0.25 |
| Activation | ELU | ELU |
| Input channels | sensorimotor subset (~15) | 8 (intersection montage) |
| Input time samples | 320 (2.0 s @160 Hz) | ~102 (0.8 s @128 Hz) |

Use the Braindecode reference implementation rather than re-implementing.

---

## Appendix B — Consolidated Hyperparameter Reference

A single table the engineer can keep open while wiring `configs/`.

### B.1 Optimization (deep models)

| Setting | Stage 1 | Stage 2 | Stage 3 (adapt) |
|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW |
| Weight decay | 1e-2 | 1e-2 | 1e-3 |
| Base LR | 1e-3 | 1e-3 | 3e-4 |
| Schedule | cosine + 10ep warmup | cosine + 5ep warmup | cosine + short warmup |
| Batch size | 64 | 128 | 16–32 (small data) |
| Max epochs | 200 | 300 | 100 |
| Early-stop metric | val ROC-AUC | val ROC-AUC | calib-val ROC-AUC |
| Early-stop patience | 20 | 30 | 15 |
| Loss | weighted BCE (focal opt.) | weighted BCE (≈5:1) | weighted BCE (+CORAL/MMD opt.) |
| Mixed precision | yes | yes | yes |
| Trainable params | all | all | adapters + head (backbone frozen) |

### B.2 Decision-layer parameters

| Parameter | Default | Range | Stage |
|---|---|---|---|
| Window length | 2.0 s | 1.0–3.0 s | 1 |
| Window stride | 0.1 s | 0.05–0.25 s | 1 |
| Threshold `τ` | set to target idle FPR | — | 1 |
| Debounce `k` | 3 windows | 1–6 | 1 |
| Refractory `R` | 5 s | 2–10 s | 1 |
| Epoch window | 0–0.8 s | 0–1.0 s | 2/3 |
| Repetitions (max) | 10 | dataset-fixed | 2/3 |
| Dynamic-stop posterior | 0.95 | 0.9–0.99 | 2/3 |
| Calibration chars `N` | 15 | 3–35 | 3 |
| Adapter reduction | 8 | 4–16 | 3 |

### B.3 Geometric pipelines

| Pipeline | Components | Use |
|---|---|---|
| MI classical | CSP(n=6–8) → LDA; or Cov → TangentSpace → LogReg | Stage 1 baseline |
| P300 classical | Xdawn(nfilter=4) → XdawnCovariances → TangentSpace → LDA | Stage 2 baseline / Stage 3 fallback |
| P300 + alignment | + Riemannian Alignment (recenter target covs to source ref) | Stage 3 geometric path |

---

## Appendix C — Preprocessing Recipes (exact, per stage)

These recipes are the canonical `preprocessing/*.yaml` contents. The `Preprocessor` executes them in order. Steps fit on training data are marked **[fit-on-train]**.

### C.1 `mi_8_30.yaml` (Stage 1)

1. Pick channels: sensorimotor subset (configurable list).
2. Band-pass FIR, zero-phase, 8–30 Hz (Hamming, default MNE transition bands).
3. Keep native sampling 160 Hz.
4. Robust per-channel scaling **[fit-on-train]** (median/IQR).
5. Epoching: sliding window 2.0 s, stride 0.1 s; label by annotation (control vs idle).
6. Training-only amplitude rejection at 150 µV peak-to-peak.
7. (Optional) EA whitening **[fit-on-train]** for the cross-subject ablation.

### C.2 `p300_1_24.yaml` (Stage 2/3)

1. Pick channels: 8-channel intersection {Fz, Cz, Pz, Oz, P3, P4, PO7, PO8} (reorder to a fixed canonical order — this order is part of the checkpoint contract).
2. Band-pass FIR, zero-phase, 1–24 Hz (config alt: 0.1–30 Hz to match ALS acquisition exactly).
3. Resample to 128 Hz.
4. Epoching: 0–0.8 s post-flash; label target vs non-target.
5. Robust per-channel scaling **[fit-on-train]**.
6. (Stage 3) Euclidean Alignment **[fit-on-train]**, reference = patient calibration mean covariance.
7. (Geometric path) xDAWN spatial filtering **[fit-on-train]** instead of/after scaling.

### C.3 Channel-order contract

The 8-channel canonical order is fixed once and stored in every checkpoint bundle. Stage 2 trains in this order; Stage 3 loads the patient data, reorders to this exact sequence, and asserts equality before inference. Any channel-order mismatch must raise, never silently reindex — this is the most common and most damaging silent bug in cross-dataset EEG transfer.

---

## Appendix D — Reproducibility Checklist

- [ ] All datasets accessed only via MOABB; caches warmed by `download_data.py`.
- [ ] EEGMMIDB subjects 88/92/100 excluded by default.
- [ ] All splits are group-aware (subject/patient); leakage tests pass.
- [ ] Scalers / EA references / xDAWN / thresholds fit on train/calibration only.
- [ ] Stage 3 evaluated within-patient, calibration-then-test; patients never pooled.
- [ ] Every checkpoint bundles weights + fitted `Preprocessor` + canonical channel order + meta.
- [ ] Each headline metric reported with per-subject distribution + bootstrap 95% CI.
- [ ] Zero-shot (cold transfer) reported as the Stage 3 lower bound for every patient.
- [ ] Resolved Hydra config + git SHA + seed logged to W&B for every run.
- [ ] Comparator pipelines (geometric / EEGNet / SOTA) run on identical splits.

*End of appendices.*

---

## Appendix E — P300 Dataset Comparison (Stage 2 source selection)

The prompt requires an explicit comparison of candidate P300 corpora and a justified recommendation for the development (source) dataset. All candidates below are reachable through MOABB unless noted.

| Dataset | Population | Subjects (approx.) | Channels | Fs | Paradigm | Pros | Cons |
|---|---|---|---|---|---|---|---|
| **BNCI 2014-009** | Healthy | 10 | 16 | 256 Hz | 6×6 speller, overt + covert | Clean; multi-session; **same lineage/montage superset of the ALS target**; MOABB-native | Modest subject count |
| BNCI 2014-008 | ALS | 8 | 8 | 256 Hz | 6×6 speller | The clinical *target* | Patient data — not a source |
| BNCI 2015-003 | Healthy | ~10 | 8 | 256 Hz | 6×6 speller | 8-channel (matches ALS montage); MOABB-native | Fewer channels than 009; single-session |
| BCI Competition III — dataset II | Healthy | 2 (A, B) | 64 | 240 Hz | 6×6 speller | Historic benchmark; well-studied | Only 2 subjects; different Fs/montage; poor for cross-subject pretraining |
| EPFL P300 (Hoffmann et al.) | Healthy + disabled | ~8 | 32 | 2048→down | 6-image speller | Includes disabled users | Image (not 6×6 letter) paradigm; montage/format mismatch |
| Brain Invaders (bi2014a / bi2015a) | Healthy | dozens (large) | 16 | 512 Hz | P300 oddball / speller | **Large** subject pools; good for cross-subject pretraining | Different Fs/montage; gamified paradigm; extra harmonization work |

**Recommendation — BNCI 2014-009 as the primary source, for one decisive reason beyond cleanliness: maximal control of the Stage 3 transfer.** Because 2014-009 and the ALS target 2014-008 come from the same experimental lineage, share 256 Hz sampling, share the identical 6×6 speller paradigm, and because the ALS 8-channel montage is a strict subset of 2014-009's 16 channels, the healthy→ALS shift is reduced to the *clinically meaningful* component (ALS cortex / P300 morphology) with nuisance domain shift (hardware, paradigm, montage) held near zero. No other pairing achieves this. Choosing a larger but mismatched source (e.g. Brain Invaders) would inflate cross-subject pretraining size at the cost of injecting paradigm/montage shift into the very transfer step the project is trying to study cleanly.

**Secondary use.** BNCI 2015-003 (8-channel, healthy) is an excellent *additional* pretraining source precisely because it already matches the ALS montage; the project should support **multi-source healthy pretraining** (2014-009 ∪ 2015-003 on the 8-channel intersection) as a configuration and as an ablation, since more healthy data generally improves the pretrained backbone and the EA-based alignment. Brain Invaders datasets are reserved for a "scale-up pretraining" future extension where the added harmonization effort is justified by volume.

---

## Appendix F — Architectural Decision Records (ADRs)

A condensed log of the non-obvious decisions, mirroring `docs/DECISIONS.md`. Each ADR states the decision, the rationale, and the rejected alternative.

**ADR-1 — Stage 3 reuses the Stage 2 backbone (does not replace it).**
*Decision:* one shared `BackboneEncoder`; Stage 3 = backbone + adapters + few-shot.
*Rationale:* clinical narrative ("the patient system *is* the healthy system, adapted"), DRY code, and a free zero-shot ablation.
*Rejected:* training an independent ALS model — breaks the narrative, duplicates code, loses the clean lower bound.

**ADR-2 — EEGNet is the production backbone; Conformer/ATCNet are comparators.**
*Decision:* EEGNet everywhere in the production path.
*Rationale:* few-shot ALS adaptation rewards small, freezable, adapter-friendly models; EEGNet is real-time on CPU.
*Rejected:* Conformer/ATCNet as default — heavier, data-hungry, fragile under PEFT, higher latency.

**ADR-3 — Train Stage 2 on the 8-channel intersection by default.**
*Decision:* default healthy training uses the 8 channels shared with the ALS dataset.
*Rationale:* native channel compatibility for transfer; zero channel imputation in Stage 3.
*Rejected:* training on 16 channels then projecting down — introduces a channel-mismatch failure mode; kept only as an upper-bound ablation.

**ADR-4 — Euclidean Alignment is the default alignment everywhere it helps.**
*Decision:* EA as the first-line domain-shift remedy across stages.
*Rationale:* parameter-free, label-free on target, cheap, composes with any model, repeatedly shown to help with deep decoders.
*Rejected:* skipping alignment (leaves easy gains on the table); jumping straight to adversarial DA (heavier, less stable, unnecessary as a default).

**ADR-5 — Geometric pipeline retained as a mandatory Stage 3 fallback.**
*Decision:* xDAWN-cov + Riemannian Alignment + tangent-space LDA runs for every patient alongside the deep path; per-patient winner is recorded.
*Rationale:* geometric methods dominate the low-data regime that ALS imposes; some weak-P300 patients will be served better by it.
*Rejected:* deep-only — brittle on the hardest patients.

**ADR-6 — MOABB is the sole data-access layer.**
*Decision:* no script parses raw `.edf`/`.mat`.
*Rationale:* uniform API across all three datasets, cached downloads, standardized paradigms/evaluations, fewer bespoke bugs.
*Rejected:* hand-rolled loaders — duplicate effort, subtle paradigm/label bugs.

**ADR-7 — Self-paced evaluation for Stage 1 (event stream, not shuffled epochs).**
*Decision:* score Stage 1 with a streaming simulator at a fixed idle FPR.
*Rationale:* a brain switch's real cost is false activations during open-ended idle; shuffled-epoch accuracy is misleading.
*Rejected:* reporting balanced accuracy on shuffled epochs — does not reflect deployed behavior.

**ADR-8 — Within-patient calibration-then-test for Stage 3; never pool patients.**
*Decision:* strict per-patient split matching the dataset's native calibration/test words.
*Rationale:* pooling leaks patient identity and inflates results; the clinical question is per-patient usability.
*Rejected:* pooled cross-validation — invalid for the clinical claim.

**ADR-9 — Parameter-efficient fine-tuning over full fine-tuning for adaptation.**
*Decision:* freeze backbone, train bottleneck adapters + head on calibration data.
*Rationale:* tens of calibration trials cannot safely fit a whole network; PEFT caps capacity and curbs overfitting.
*Rejected:* full fine-tune as default — overfits tiny patient sets (kept as an ablation with `last_block_unfrozen`).

**ADR-10 — Channel order is a hard checkpoint contract.**
*Decision:* canonical 8-channel order stored in every bundle; mismatches raise.
*Rationale:* silent channel reindexing is the most damaging hidden bug in cross-dataset EEG transfer.
*Rejected:* implicit reindexing — fails silently and corrupts spatial filters.

---

## Appendix G — Glossary

- **P300 / P3b:** a positive evoked potential ~300 ms after a rare, attended stimulus; the signal the speller exploits.
- **Oddball paradigm:** rare targets among frequent non-targets; elicits the P300.
- **ERD/ERS:** event-related (de)synchronization of mu/beta rhythms during (imagined) movement; the Stage 1 signal.
- **Brain switch / asynchronous (self-paced) BCI:** a detector operating continuously, firing on intentional control while ignoring idle.
- **ITR (Information Transfer Rate):** bits/minute communicated, via the Wolpaw formula over the alphabet size, accuracy, and time per selection.
- **xDAWN:** spatial filtering that enhances evoked-potential SNR by maximizing the evoked-to-ongoing signal ratio.
- **Tangent space:** the Euclidean linearization of the SPD-covariance manifold at a reference point, where ordinary linear classifiers can be applied.
- **Euclidean Alignment (EA):** whitening each subject's trials by the inverse square root of their mean covariance to reduce inter-subject shift.
- **Riemannian Alignment (RA):** recentering (and optionally rotating/scaling) covariance matrices on the SPD manifold to a common reference.
- **Adapter / PEFT:** small trainable modules inserted into a frozen network so a few parameters can adapt it to a new domain.
- **LOSO:** leave-one-subject-out cross-validation (subject-independent evaluation).

*End of document.*