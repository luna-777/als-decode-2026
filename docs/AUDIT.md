# Stage 3 Pre-Publication Audit

**Date:** 2026-07-26
**Repo state audited:** `02af554` (clean tree, branch `main`)
**Environment:** conda env `als-decode` — moabb 1.5.0, mne 1.12.1, torch 2.12.1
**Status:** **PHASE 0 GATE FAILED (check 0.1).** Work stopped before Phase 1.

---

## Verdict

| Check | Result |
|---|---|
| 0.1 MI channel mismatch | **FAIL — real divergence.** All MI Stage 3 results are invalid. |
| 0.2 P300 channel indexing | **PASS on selection, FAIL on robustness.** Correct electrodes were selected, by coincidence. P300 numbers stand on this axis. |
| 0.3 EA provenance of temporal arm | **Hypothesis falsified.** The temporal arm *did* have EA. Not confounded with EA-vs-no-EA. |
| 0.4 (found during audit) Checkpoint pinning | **FAIL.** Stage 3 selects the MI checkpoint by a mutable filesystem scan. |
| 0.5 (found during audit) Statistics errors | **FAIL.** Sign error at n=10 and a mislabelled p-value column. |
| 0.6 Discriminating test on `version_28` | **PASS.** Config montage empirically confirmed; no retraining needed. |

**Bottom line for the paper's authors:** every MI number in
`experiments/stage3/mi_results*.csv` is dead and must be regenerated. The P300
numbers survive the channel checks but remain non-comparable across the two
arms for the reasons in §0.3.4. The central temporal-vs-stratified result is
**not** void for the reason the audit brief anticipated (EA), but it *is*
void because both arms were computed on a scrambled electrode montage.

---

## 0.1 — MI channel mismatch: CONFIRMED, and worse than the brief suggested

### What the training path actually used

Traced `src/train.py` → `EEGDataModule` → `DatasetSpec`:

- [src/train.py:35](../src/train.py#L35) — `spec = build_spec_from_cfg(cfg.dataset)`
- [src/training/datamodule.py:160-162](../src/training/datamodule.py#L160-L162) —
  `ch_config = "sensorimotor"`, `ch_list = channels_node[ch_config]`
- [configs/dataset/physionet_mi.yaml:11-29](../configs/dataset/physionet_mi.yaml#L11-L29) —
  the `sensorimotor` list

So **Stage 1/2 MI training used the config list**:

```
FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4
```

This matches [docs/design.md:205](design.md#L205), which specifies the
sensorimotor subset. The config list has **never been modified** since the
initial commit `de3810f` (2026-06-30); `git log --follow` on the file shows only
two touches, and the second (`f56f3ea`) added the `euclidean_alignment` key only.

### What Stage 3 actually used

[scripts/adapt_stage3.py:251-261](../scripts/adapt_stage3.py#L251-L261) hardcodes
a different list:

```
FC5, FC3, FC1, FCz, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1
```

This list was introduced in the **first** version of the file (`b4cd19f`,
2026-07-15) and has never been changed since — verified with
`git log -p --follow -- scripts/adapt_stage3.py`.

### Severity

Both lists have length 17, so `BackboneEncoder(n_channels=17)` loads the
Stage 2 checkpoint without a shape error. The divergence is silent, exactly as
ADR-10 warned.

- **Number of array positions where the two lists agree: 0 (zero).**
- 3 electrodes are in the Stage 3 list but were never trained on: `FC5, FC6, CP5`
- 3 electrodes were trained on but are absent at Stage 3: `CPz, CP2, CP4`
- The remaining 14 are present in both but at shifted indices

Position-by-position, `trained_as -> loaded_as`:

```
 0: FC3 -> FC5      6:  C3 -> FC6     12: CP3 -> C4
 1: FC1 -> FC3      7:  C1 -> C5      13: CP1 -> C6
 2: FCz -> FC1      8:  Cz -> C3      14: CPz -> CP5
 3: FC2 -> FCz      9:  C2 -> C1      15: CP2 -> CP3
 4: FC4 -> FC2     10:  C4 -> Cz      16: CP4 -> CP1
 5:  C5 -> FC4     11:  C6 -> C2
```

Every learned spatial filter weight was applied to the wrong electrode. This is
not a partial or near-miss misalignment — it is a total one.

### Did the lists diverge in the runs that produced `mi_results*.csv`?

**Yes, in every MI run ever performed.** The hardcoded list has been present
since `adapt_stage3.py` was created (2026-07-15) and the config list since
2026-06-30. There is no commit at which they agreed, and the reflog
(§0.3.1) shows no checkout that could have produced a different working tree.

> **All existing MI results are invalid.** This covers
> `experiments/stage3/mi_results.csv`, `experiments/stage3/mi_results_temporal.csv`,
> and `experiments/stage3/mi_wilcoxon.csv`, plus every MI row in the
> superseded committed versions at `9cebf15` and `fe9bb96`.

---

## 0.2 — P300 channel indexing: correct electrodes, by coincidence

### Ground truth from MOABB

Loaded every BNCI2014_009 subject through MOABB (`scripts/` probe, env `als-decode`):

```
RAW n_channels: 18   (16 EEG + 'Target stim' + 'Flash stim')
RAW ch_names:   ['Fz','Cz','Pz','Oz','P3','P4','PO7','PO8',
                 'F3','F4','FCz','C3','C4','CP3','CPz','CP4',
                 'Target stim','Flash stim']
sfreq: 256.0 Hz;  sessions: '0','1','2' (1 run each)
P300 paradigm get_data() X.shape: (1728, 16, 102)
class balance: 1440 NonTarget / 288 Target
```

**True EEG channel count: 16.** Confirmed identical order for all 10 subjects
(`match_intended=True` for subjects 1–10).

### Was the intended montage selected?

[src/datasets/wrapper.py:192-194](../src/datasets/wrapper.py#L192-L194) declares
`_BNCI009_CHANNELS` as 8 names with the comment *"all 8 electrodes,
dataset-defined order"*. [wrapper.py:252-254](../src/datasets/wrapper.py#L252-L254)
builds `ch_to_idx` from that 8-name list and slices `X[:, ch_idx, :]`, so it
selects positions **0–7 of the 16 returned channels**.

MOABB returns `Fz, Cz, Pz, Oz, P3, P4, PO7, PO8` as positions 0–7 — which is
**exactly** the intended montage, in the intended order.

> **The previous indexing did select the intended electrodes.** P300 results are
> not invalidated by this check.

### But the code is wrong, and the docs are wrong

Two defects remain, both latent rather than active:

1. The comment at [wrapper.py:192-193](../src/datasets/wrapper.py#L192-L193) is
   false: the dataset has 16 channels, not 8. The correctness of the slice is an
   accident of MOABB's ordering, not a property the code establishes.
2. [docs/design.md:170](design.md#L170) states the 16-channel order as
   `Fz, FCz, Cz, CPz, Pz, Oz, F3, F4, C3, C4, CP3, CP4, P3, P4, PO7, PO8`.
   **This is not the order MOABB returns.** Had the design doc's order been the
   real one, positions 0–7 would have been `Fz, FCz, Cz, CPz, Pz, Oz, F3, F4` —
   the wrong montage, and all P300 results would have been invalid. The code was
   saved by a discrepancy in the documentation.

Any MOABB version bump that reorders channels would silently corrupt every P300
result with no shape error. Selection must be by name with a presence assertion.

---

## 0.3 — EA provenance of the temporal arm: hypothesis falsified

### 0.3.1 Git forensics

`git reflog --date=iso` is **strictly linear** — 31 entries, all `commit:`, no
`checkout:`, `reset:`, `rebase:`, or `stash`. There is no point at which the
working tree was reverted to pre-EA code via git.

Relevant timeline:

| Time | Commit / event | Note |
|---|---|---|
| Jul 18 23:43 | wandb run `fqe8y6vb` (mi), `l3oz21am` (p300) | **pre-EA** |
| Jul 19 18:26:04 | `f56f3ea` implement Euclidean Alignment | msg: *"Requires retraining Stage 1 (4 folds) and Stage 2."* |
| Jul 19 18:27:45 | `9cebf15` "mi results" | 101 s after EA commit |
| Jul 19 18:27:55 | `2458e91` "P-300 results CSV" | 10 s later |
| Jul 19 18:37–20:15 | Stage 1 retrained → `lightning_logs/version_26..30` | the EA retraining |
| Jul 19 20:18 | wandb run `h3o3s5h5` (p300) | **post-EA** |
| Jul 19 20:20 | wandb run `hhbyr8xk` (mi) | **post-EA** |
| Jul 23 20:45 | wandb run `lyqv2x70` (mi, 50 rows incl. n=200) | **post-EA** |
| Jul 23 21:00:23 | `bf23ef9` split changed temporal → stratified | |
| Jul 23 21:04–21:05 | `*_temporal.csv` written to disk | LF endings, **no wandb run at this time** |
| Jul 23 21:08:29 | `7fd8a4d` commit temporal files | |
| Jul 24 23:24 | wandb run `63zf6qrx` (p300, LOSO-10, 10 subjects) | **post-EA** |

### 0.3.2 Matching every CSV to its generating run

The wandb offline datastores (`wandb/offline-run-*/*.wandb`) were decoded with
`wandb.sdk.internal.datastore` and every logged `(subject, calib_size,
baseline_auc, adapted_auc)` tuple compared to the CSVs to 1e-12:

| CSV | Generating wandb run | Run time | EA? |
|---|---|---|---|
| `mi_results.csv` (HEAD) | `lyqv2x70` | Jul 23 20:45 | **yes** |
| `mi_results_temporal.csv` (HEAD) | `hhbyr8xk` | Jul 19 20:20 | **yes** |
| `p300_results.csv` (HEAD) | `63zf6qrx` | Jul 24 23:24 | **yes** |
| `p300_results_temporal.csv` (HEAD) | `h3o3s5h5` | Jul 19 20:18 | **yes** |
| `mi_results.csv` @ `9cebf15` (superseded) | `fqe8y6vb` | Jul 18 23:43 | no |
| `p300_results.csv` @ `2458e91` (superseded) | `l3oz21am` | Jul 18 23:43 | no |

All four CSVs at HEAD are exact matches (every row, all four float fields).

> **The temporal arm had EA.** Both `*_temporal.csv` files at HEAD were produced
> by runs on 2026-07-19 at 20:18 and 20:20 — roughly two hours **after** EA was
> implemented at 18:26 the same day. The temporal-vs-stratified comparison is
> **not** confounded with EA-vs-no-EA.

The pre-EA runs do exist, but they are the *superseded* CSVs committed at
`9cebf15`/`2458e91` on Jul 19 at 18:27 — 101 seconds after the EA commit, far too
fast to be an EA run, and the EA commit message itself states retraining was
still required. Those files were overwritten on Jul 23 and are not part of the
current results.

### 0.3.3 Line-ending evidence — corroborates, with a correction

As the brief anticipated:

| File | Endings |
|---|---|
| `mi_results.csv` | CRLF |
| `p300_results.csv` | CRLF |
| `mi_results_temporal.csv` | **LF** |
| `p300_results_temporal.csv` | **LF** |
| `mi_wilcoxon.csv` | LF |
| `p300_sign_test.csv` | LF |

`csv.DictWriter` with `newline=""`
([adapt_stage3.py:340-344](../scripts/adapt_stage3.py#L340-L344)) emits CRLF, so
the CRLF files were written by that writer and the LF files were not. Confirmed
that the superseded Jul 19 committed versions were **also CRLF** — i.e. the
writer's behaviour is stable and LF genuinely indicates a different producer.

There is **no wandb run at Jul 23 21:04–21:05**, when the temporal files were
written, and `adapt_stage3.py` unconditionally calls `wandb.init`
([adapt_stage3.py:347-359](../scripts/adapt_stage3.py#L347-L359)). So the files
were not produced by the script at that moment.

**Correction to the brief's inference:** the LF endings show the *files* were
hand-transcribed on Jul 23, four minutes after the split was changed to
stratified — but the *numbers* in them trace exactly, to 1e-12, to the genuine
post-EA run `hhbyr8xk`/`h3o3s5h5` of Jul 19. The provenance is sloppy (results
reconstructed by hand from a wandb log rather than re-run), but the values are
authentic and post-EA. Line endings indicate a different *writer*, not different
*data*.

### 0.3.4 The temporal arm is nonetheless not comparable to the stratified arm

Three independent reasons, none of which is EA:

1. **Different electrode montage than the model was trained on** — §0.1. Applies
   to both MI arms equally, so the *comparison* is internally consistent, but
   both sit on scrambled spatial filters.
2. **MI: no `n=200` in the temporal arm.** `mi_results_temporal.csv` covers only
   `{10, 20, 50, 100}` (40 rows). `mi_results.csv` covers `{10, 20, 50, 100, 200}`
   (50 rows). The paper's headline claim — *"+0.089 at n=200, 9/10 subjects
   positive"* (`bf23ef9` commit message) — **has no temporal counterpart at all.**
   The comparison at the headline calibration size does not exist.
3. **P300: different LOSO regimes.** `p300_results_temporal.csv` has only
   subjects 9 and 10 (the 2-held-out regime,
   [configs/evaluation/p300_loso.yaml](../configs/evaluation/p300_loso.yaml));
   `p300_results.csv` has all 10 under LOSO-10
   ([p300_loso_full.yaml](../configs/evaluation/p300_loso_full.yaml)). Confirmed
   from the wandb subject lists.

---

## 0.4 — NEW FINDING: the MI checkpoint is not pinned

Not in the audit brief; found while explaining a numerical inconsistency.

[adapt_stage3.py:62-77](../scripts/adapt_stage3.py#L62-L77) `_find_best_checkpoint`
globs `lightning_logs/version_*/checkpoints/best-epoch=*/*.ckpt` and returns
whichever has the highest val AUC **in the filename**, filtered by channel count
and `val_auc > 0.55`. The result therefore depends on the state of the
filesystem at run time, not on anything recorded in the experiment.

Re-simulating that selection against checkpoint mtimes:

| MI Stage 3 run | Checkpoint it would have selected | val AUC |
|---|---|---|
| Jul 18 23:43 (pre-EA) | `version_12` (trained **pre-EA**) | 0.7899 |
| Jul 19 18:37 (post-EA, discarded) | `version_12` (trained **pre-EA**) | 0.7899 |
| Jul 19 20:20 → `mi_results_temporal.csv` | `version_28` (EA-trained) | 0.8016 |
| Jul 23 20:45 → `mi_results.csv` | `version_28` (EA-trained) | 0.8016 |

Two consequences:

- **Good news for the comparison:** both published MI arms used the *same*
  checkpoint, `version_28`. So the temporal-vs-stratified contrast is not
  confounded by checkpoint choice either.
- **Bad news generally:** the Jul 19 18:37 run applied EA-transformed Stage 3
  data to `version_12`, a checkpoint trained *without* EA. That run's mean
  baseline AUC differs from the pre-EA run on the same checkpoint by 0.080 and
  from the `version_28` run by 0.113. It was silently discarded, but nothing in
  the CSV, the config, or the commit log would have revealed the difference. Any
  future training run that lands a 17-channel checkpoint with val AUC > 0.8016
  will silently change all MI results.

The checkpoint path, its val AUC, and the channel list must be recorded in the
output CSV.

### Held-out subject verification

`wrapper.subject_list` = 106 subjects after excluding 88, 92, 100.
`subject_list[-10:]` = **[99, 101, 102, 103, 104, 105, 106, 107, 108, 109]**,
matching [src/train.py:51](../src/train.py#L51). With
`held_out_n_subjects: 10` ([configs/evaluation/loso.yaml](../configs/evaluation/loso.yaml)),
`dev_subjects = all_subjects[:-10]` and GroupKFold runs over dev only, so these
10 subjects appear in neither train nor val. **Exclusion is structurally
guaranteed for `version_28`.** Note this holds by construction, not by recorded
metadata — `version_26..30` predate `val_subjects.json`
([train.py:111-117](../src/train.py#L111-L117)), which only exists for
`version_31..40`.

---

## 0.5 — NEW FINDING: statistics errors in `mi_wilcoxon.csv`

Recomputed directly from `experiments/stage3/mi_results.csv`
(scipy `wilcoxon`, two-sided):

| n | published mean | **true mean** | sum | n_pos | W | published p₂ | **true p₂** |
|---|---|---|---|---|---|---|---|
| 10 | **+0.0008** | **−0.000759** | −0.007589 | 5 | 27.0 | 1.0000 | 1.0000 |
| 20 | +0.0104 | +0.010431 | +0.104314 | 5 | 19.0 | 0.4316 | 0.4316 |
| 50 | −0.0002 | −0.000237 | −0.002367 | 6 | 27.0 | 1.0000 | 1.0000 |
| 100 | +0.0230 | +0.022962 | +0.229616 | 8 | 14.0 | 0.1934 | 0.1934 |
| 200 | +0.0885 | +0.088518 | +0.885185 | 9 | 0.0 | **0.0077** | **0.0039** |

Two distinct errors:

1. **Sign error at n=10.** The true mean ΔAUC is **−0.000759**, published as
   **+0.0008**. The magnitude is right and the sign is dropped. It is a one-off:
   n=50 correctly carries its negative sign. The direction of the smallest
   calibration condition is reported backwards.
2. **Mislabelled p-value columns.** At n=200 the published `p_two_sided`
   (0.0077) is exactly 2× the published `p_one_sided` (0.0038), and the *true*
   two-sided p is 0.0039 — i.e. the column labelled `p_one_sided` holds the real
   two-sided p, and `p_two_sided` is that value doubled a second time. The same
   2× relation holds at every n. The headline significance claim is quoted
   against a p that is twice the correct value; it survives α=0.05 either way,
   but the reported number is wrong.

`p300_sign_test.csv` was spot-checked and its means agree with
`p300_results.csv`; it reports 0/10 improving at every size except n=480 (2/10).

---

## 0.6 — Discriminating test: is the config montage the one version_28 was trained on?

`version_28` predates the montage contract, so its electrode order cannot be read
back from the checkpoint — only *inferred* from the config the training path read.
The inference is documentary (the config has never been edited); this test makes it
empirical.

**Method.** `scripts/eval_heldout_mi.py` evaluates the pretrained model on the 10
held-out MI subjects with **no adaptation**, EA applied as currently implemented,
scoring all of each subject's epochs. Run twice against the *same* pinned
checkpoint and preprocessor, with the montage as the only difference:

```bash
python scripts/eval_heldout_mi.py --folds version_28
python scripts/eval_heldout_mi.py --folds version_28 \
    --montage-override FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1
```

**Result.**

| subject | wrong montage | config montage | Δ |
|---|---|---|---|
| 99 | 0.6171 | 0.7154 | +0.0983 |
| 101 | 0.6843 | 0.7171 | +0.0328 |
| 102 | 0.7508 | 0.9189 | +0.1681 |
| 103 | 0.7441 | 0.8715 | +0.1273 |
| 104 | 0.8229 | 0.8759 | +0.0530 |
| 105 | 0.7931 | 0.8005 | +0.0075 |
| 106 | 0.7623 | 0.8111 | +0.0488 |
| 107 | 0.5633 | 0.6605 | +0.0971 |
| 108 | 0.6941 | 0.6857 | −0.0084 |
| 109 | 0.6062 | 0.5680 | −0.0382 |
| **mean** | **0.7038** | **0.7625** | **+0.0586** |

Paired Wilcoxon W=6.0, **p=0.027** two-sided; 8/10 subjects improve; Cohen's
d_z=+0.914; bootstrap 95% CI on the mean gain **[+0.0225, +0.0977]**.

**Calibration of the comparison.** The wrong-montage control reproduces the
published figure — 0.7038 here versus 0.6949 in `mi_results.csv`. The small
residual is expected: the published number is scored on evaluation subsets after
calibration epochs are removed, whereas this test scores all epochs. That
agreement confirms the harness reproduces the pre-audit condition, so the +0.0586
is attributable to the montage and nothing else.

**Against the fold range.** The config-montage mean of 0.7625 falls inside the
quoted fold validation range 0.715–0.788, and at the lower edge of the EA-era
sweep's own range (0.7632–0.8016).

> **Verdict: the config montage is empirically confirmed.** Feeding correctly
> ordered electrodes moves held-out performance materially toward the range the
> checkpoint achieved on validation data, consistently across subjects.
> **Proceed against `version_28` without retraining Stage 1/2.**

### 0.6.1 Fold spread — and evidence for item 6

All four EA-era MI folds evaluated under the config montage
(`mi_heldout_montage_check_allfolds.csv`):

| fold | val AUC | held-out mean |
|---|---|---|
| version_27 | 0.7783 | 0.7749 |
| version_28 | **0.8016** | **0.7625** |
| version_29 | 0.7951 | 0.7736 |
| version_30 | 0.7632 | 0.7730 |

Fold-averaged mean **0.7710**; sd of the fold means 0.0057.

Note the ordering: `version_28` has the **highest validation AUC of the four and
the lowest held-out AUC**. Validation AUC does not rank folds usefully here
(n=4, so this is indicative rather than significant), which is a direct
demonstration of why best-of-N selection had to go — the selection criterion
carried no held-out signal. Fold-averaging is the better estimator and is what
`--fold-mode average` exists to produce.

Also worth recording: the EA-era sweep contains **four** MI folds
(`version_27..30`), not five. `configs/evaluation/loso.yaml` specifies
`n_splits: 5`, and the EA commit message says "Requires retraining Stage 1 (4
folds)". One fold of the 5-fold split was never retrained after EA. `version_26`
is the 8-channel Stage 2 P300 retrain, not the missing MI fold.

---

## Reproduction commands

```bash
PY=/Users/lunawang/miniconda3/envs/als-decode/bin/python

# 0.2 — true BNCI2014_009 channel count and order
$PY -c "
import warnings; warnings.simplefilter('ignore')
from moabb.datasets import BNCI2014_009
raw = BNCI2014_009().get_data(subjects=[1])[1]['0']['0']
print(len(raw.ch_names), raw.ch_names)"

# 0.3 — decode a wandb offline run and match against a CSV
# (see git history of this audit for /tmp/wread3.py and /tmp/cmp.py)

# 0.5 — recompute the Wilcoxon table
$PY -c "
import csv, collections, numpy as np
from scipy.stats import wilcoxon
rows=list(csv.DictReader(open('experiments/stage3/mi_results.csv', newline='')))
by=collections.defaultdict(list)
for r in rows: by[int(r['calib_size'])].append(float(r['delta_auc']))
for n in sorted(by):
    v=np.array(by[n]); W,p=wilcoxon(v)
    print(n, round(v.mean(),6), int((v>0).sum()), W, round(p,4))"
```

---

## Required remediation before any re-run

Blocking, in order:

1. **Single-source the MI channel list.** `adapt_stage3.py` must read
   `configs/dataset/physionet_mi.yaml` through the same `build_spec_from_cfg`
   path `train.py` uses; delete the hardcoded list at
   [adapt_stage3.py:251-261](../scripts/adapt_stage3.py#L251-L261).
2. **Make the montage part of the checkpoint contract.** Persist the channel
   list *and order* into the checkpoint bundle; assert equality at Stage 3 load
   time and raise `ValueError` with both lists on mismatch (ADR-10). Add the
   equal-length-different-order regression test.
3. **Select P300 channels by name** from what MOABB returns, asserting every
   requested name is present. Correct the false 8-channel comment at
   [wrapper.py:192-193](../src/datasets/wrapper.py#L192-L193) and the incorrect
   channel order at [design.md:170](design.md#L170).
4. **Pin the checkpoint.** Record checkpoint path, val AUC, and channel list in
   every output row.
5. **Retrain Stage 1/2 MI** if and only if step 2 shows `version_28`'s stored
   montage is unrecoverable — it predates any montage metadata, so the montage
   can only be inferred from the config at its training time (which is
   unambiguous: the config list has never changed).

Items 1–4 are cheap. **Phase 2's full re-run is worth doing only after they
land**, which is why this audit stops here.
