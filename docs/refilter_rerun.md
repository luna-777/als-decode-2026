# Re-filter re-run: P300 and ALS under the intended 1–24 Hz band

**Date:** 2026-08-05 · **Branch:** `fix/refilter-p300-als` · **Fixing commit:** `d90d18a`

Every P300 and ALS result previously in this repo was computed under an 8–30 Hz filter
designed for 160 Hz and applied to 128 Hz data — an effective passband of ≈ 6.4–24 Hz —
not the documented 1–24 Hz at 128 Hz. This document records the fix, the verification,
and the re-run. **MI is unaffected and was verified bit-identical rather than re-run.**

---

## The defect

Two independent faults compounded. Both were confirmed against artifacts before any
code was changed.

### Fault 1 — the `band` key was never read

`src/preprocessing/pipeline.py` resolved the passband as:

```python
band  = self._get("band", [8.0, 30.0])
sfreq = self._get("keep_sfreq", 160.0)
X_out = bandpass_filter(X, sfreq=float(sfreq), l_freq=float(band[0]), h_freq=float(band[1]))
```

Neither `configs/preprocessing/mi_8_30.yaml` nor `p300_1_24.yaml` defines a top-level
`band` key — both nest the values under `bandpass: {low:, high:, method:, phase:,
fir_window:}`. `_get` therefore always fell through to the hardcoded `[8.0, 30.0]`, for
every paradigm, and the whole `bandpass:` block was dead code including `method`,
`phase` and `fir_window`.

Confirmed by unpickling the as-trained preprocessors. All three carry identical keys and
no `band`:

| pickle | `cfg.keys()` | `'band' in cfg` | `keep_sfreq` | `amplitude_reject_uv` |
|---|---|---|---|---|
| `version_28` (MI) | `bandpass, keep_sfreq, window_len_s, window_stride_s, robust_scale, amplitude_reject_uv, euclidean_alignment` | **False** | 160.0 | 150.0 |
| `version_31` (P300) | *(same)* | **False** | 160.0 | 150.0 |
| `version_39` (P300) | *(same)* | **False** | 160.0 | 150.0 |

### Fault 2 — the P300 folds were trained under the MI preprocessing group

`outputs/2026-07-24/22-57-24/.hydra/overrides.yaml` records the launch of the runs that
produced `version_31`…`version_40`:

```yaml
- dataset=bnci_009
- stage=stage2
- evaluation=p300_loso_full
- fold=4
```

No `preprocessing` override and no `--config-name`, so composition fell to the default
`configs/config.yaml`, whose `defaults:` list selects `preprocessing: mi_8_30`. The
composed `preprocessing:` node in that run's `config.yaml` is verbatim `mi_8_30.yaml`:
`bandpass {low: 8.0, high: 30.0}`, `keep_sfreq: 160.0`, `amplitude_reject_uv: 150.0`.
`configs/config_stage2.yaml` existed and was correct; it simply was not used.

**Blast radius.** The composed node also carried MI's `window_len_s: 2.0` /
`window_stride_s: 0.1` instead of P300's `0.8 / 0.8`, but neither key is read anywhere
in `src/` or `scripts/` — `intent_switch.py` and `streaming.py` carry their own
independent defaults. So the consequences are confined to `keep_sfreq` and
`amplitude_reject_uv`.

---

## A. The diff

Cumulative against the pre-fix baseline `1934e62`:

| file | change | why it is not an experimental change |
|---|---|---|
| `src/preprocessing/pipeline.py` | +30 / −2 | The only source change. Adds `_get_band_args()`, which reads `bandpass.{low,high,method,phase,fir_window}` with an explicit top-level `band` key taking precedence and the previous 8–30/fir/zero/hamming values as fallback. `transform()` passes `**self._get_band_args()`. |
| `experiments/stage3/archive_prefilter_fix/**` | 8 CSVs + 6 figure files | **100 %-similarity renames** — `git diff -M --summary` reports `rename … (100%)` for all eight CSVs and every byte of the figures is preserved. No archived value was altered. Plus a new `README.md` marker. |
| `experiments/stage3/refilter_fold_manifest.csv` | +11 | New provenance record. `lightning_logs/` is gitignored, so per-fold checkpoint sha256s would otherwise be unrecoverable. |
| `.gitignore` | +4 / −1 | Ignores `lightning_logs_refilter/`, the regenerable symlink view described in §B.4. |

**The argument that nothing but the filter path moved:**

1. `git diff --name-only 1934e62..HEAD | grep -E '^(configs|scripts|src)/'` returns
   **exactly one path**: `src/preprocessing/pipeline.py`. No config file changed, no
   script changed, no model or training code changed.
2. Within that file, the change is confined to band-pass argument *resolution*. The
   scaler, the fit-on-train invariant, `fit()`, `fit_transform()`, and the ordering of
   filter-then-scale are untouched.
3. MI is byte-identical under the change — proved in §B.1, not asserted.
4. No default was altered to achieve the Stage 2 fix. `configs/config_stage2.yaml` was
   already correct; the defect was in how it was invoked, so the remedy is an explicit
   run command (§B.3). This was a deliberate choice: changing a default would have made
   the fix invisible at the call site.
5. **Deliberately not touched**, each deferred to its own commit: the inert
   `configs/model/eegnet.yaml` and `configs/training/stage3.yaml`; the
   `mixed_precision` / `precision="32"` conflict; the duplicate `euclidean_alignment`
   keys (`dataset` says `true`, `preprocessing` says `false`); the
   amplitude-rejection-after-scaling ordering; the six bootstrap implementations; and
   the FIR-filter-length problem in §F.1.

The one thing the fix does *not* address, and which the commit message says plainly, is
that `configs/config.yaml` still selects `preprocessing: mi_8_30` and would still
mis-compose a Stage 2 run launched the old way. Guarding that is a separate commit.

### Stale provenance pointers this creates

`version_31`…`version_40` are referenced as P300/ALS provenance **44 times in
`docs/paper_facts.md`** and **once in `docs/AUDIT.md`**. Those pointers now name
superseded checkpoints and should read `version_42`…`version_51`. They are not edited
here — the underlying numbers changed too, so those rows need regenerating rather than
find-and-replace.

---

## B. Verification

### B.1 Gate 1a — MI is bit-identical

`mi_8_30.yaml`'s `bandpass: {low: 8.0, high: 30.0, method: fir, phase: zero,
fir_window: hamming}` equals the old hardcoded fallback exactly, so the fix must be a
no-op for MI. Proved by running a fixed deterministic input through `version_28`'s
as-trained preprocessor before and after the change.

Input: seeded `(32, 17, 320)` float32, `np.random.default_rng(20260805)`, sha256
`e02a43a797e99cb55ff0ed20f0d481c7702088b9f767e21609aec3c3ec546d36`.

| | output sha256 |
|---|---|
| pre-fix | `e599954b933f8cd30104b3d5675a9aa764e2834cb9d30b11d8ddf80e3f8c552c` |
| post-fix | `e599954b933f8cd30104b3d5675a9aa764e2834cb9d30b11d8ddf80e3f8c552c` |

```
np.array_equal(before, after) = True
max abs diff                  = 0.0
byte-identical                = True
```

Exact equality, not `allclose`. A second check: re-running the *old* P300 config
(`version_31`'s cfg) through the fixed code reproduces its old response unchanged to
four decimals, confirming the fix changes only configs that actually carry a `bandpass:`
block distinct from the fallback.

Test suites `test_m2_preprocessing.py`, `test_smoke.py`, `test_m7_stage2.py`: **71
passed**. The two ad-hoc call sites that pass an explicit `band`
(`scripts/evaluate.py:140`, `scripts/evaluate_stage2.py:127`) keep working via the
precedence rule, and `Preprocessor(cfg={})` still resolves to the old defaults.

### B.2 Gate 1b — the filter moved for P300

Realized response measured by pushing pure sinusoids sampled at **128 Hz** through a
Stage-2 `Preprocessor`. Filter-only: `robust_scale` disabled, so the ratio is filter
gain and not the scaler's normalisation.

At the **actual 0.8 s / 102-sample P300 epoch length**:

| freq | old (≈6.4–24 Hz) | new (1–24 Hz) |
|---|---|---|
| 2 Hz | 0.0059 | **1.1247** |
| 5 Hz | 0.0853 | **0.9985** |
| 10 Hz | 0.9795 | **1.0728** |
| 20 Hz | 1.0061 | **1.2026** |
| 26 Hz | 0.7786 | **0.7815** |
| 30 Hz | 0.2292 | **0.6336** |

The old column reproduces the independently reported reference values
(0.006 / 0.085 / 0.980 / 1.006 at 2/5/10/20 Hz) exactly, which validates the harness.
2 Hz and 5 Hz now pass essentially unattenuated — the intended change.

The same measurement on an 8 s / 1024-sample signal, where the FIR filter is not
length-constrained, shows the response the config actually describes:

| freq | old, 8 s | new, 8 s |
|---|---|---|
| 2 Hz | 0.0024 | **0.9979** |
| 5 Hz | 0.0375 | **0.9982** |
| 10 Hz | 0.9983 | **1.0022** |
| 20 Hz | 0.9969 | **1.0115** |
| 26 Hz | 0.7737 | **0.7977** |
| 30 Hz | 0.0812 | **0.2170** |

The gap between the two tables is a real problem, independent of this fix and not
introduced by it. See §F.1.

### B.3 The Stage 2 run command

```
python -m src.train --config-name config_stage2 \
    dataset=bnci_009 stage=stage2 preprocessing=p300_1_24 \
    evaluation=p300_loso_full training=default seed=42 fold=N
```

for `N = 0…9`. Every group is pinned explicitly rather than inherited.

Two pins are load-bearing:

- **`evaluation=p300_loso_full`.** `config_stage2.yaml` defaults to `p300_loso`, which
  is 2 held-out subjects + 4 folds — *not* the LOSO-10 the archived runs used.
- **`training=default`.** `config_stage2.yaml` selects `training: stage2`, which
  differs from `training: default` in exactly one value: **`lr` 5.0e-4 vs 1.0e-3**.
  The archived runs went through `config.yaml` and so used 1.0e-3. Pinning `default`
  keeps the preprocessing group the only variable that moved, so the val-AUC change is
  attributable to the filter alone. Everything else in the two training groups —
  optimizer, weight_decay, cosine schedule, warmup 10, batch 64, max_epochs 200,
  early_stop patience 20 / min_delta 0.001, weighted_bce, mixed_precision — is
  identical. Whether Stage 2 *should* use 5e-4 is a separate question, deliberately not
  answered here.

The composed config was dry-run verified (`--cfg job --resolve`) before training.

### B.4 Per-fold preprocessor assertions — 10/10 PASS

| version | val subj | band | keep_sfreq | amplitude_reject_uv | FIR args | montage.json | montage_source |
|---|---|---|---|---|---|---|---|
| version_42 | 10 | [1.0, 24.0] | 128.0 | 100.0 | fir/zero/hamming | yes | recorded |
| version_43 | 9 | [1.0, 24.0] | 128.0 | 100.0 | fir/zero/hamming | yes | recorded |
| version_44 | 8 | [1.0, 24.0] | 128.0 | 100.0 | fir/zero/hamming | yes | recorded |
| version_45 | 7 | [1.0, 24.0] | 128.0 | 100.0 | fir/zero/hamming | yes | recorded |
| version_46 | 6 | [1.0, 24.0] | 128.0 | 100.0 | fir/zero/hamming | yes | recorded |
| version_47 | 5 | [1.0, 24.0] | 128.0 | 100.0 | fir/zero/hamming | yes | recorded |
| version_48 | 4 | [1.0, 24.0] | 128.0 | 100.0 | fir/zero/hamming | yes | recorded |
| version_49 | 3 | [1.0, 24.0] | 128.0 | 100.0 | fir/zero/hamming | yes | recorded |
| version_50 | 2 | [1.0, 24.0] | 128.0 | 100.0 | fir/zero/hamming | yes | recorded |
| version_51 | 1 | [1.0, 24.0] | 128.0 | 100.0 | fir/zero/hamming | yes | recorded |

The `band` column is the *resolved* value from `_get_band_args()`; the pickled cfg still
has no top-level `band` key, it now correctly reads `bandpass.low/high`. `method`,
`phase` and `fir_window` are read from the config for the first time — the resolved set
across all ten folds is the single value `{('fir', 'zero', 'hamming')}`.

`val_subjects.json` exists on all ten. `loso_fold_for_subject(..., strict=True)` with
`n_channels=8` resolves every one of the 10 subjects to exactly one fold.

**Provenance column change, flagged:** these folds post-date `MontageCheckpoint`, so all
ten wrote `montage.json` and `montage_source` is now **`recorded`** where the archived
runs recorded **`inferred_from_config`**. That is an improvement in verifiability, but it
is a column that changed for reasons unrelated to the filter, and it means the montage
contract is now actually *tested* for these folds rather than inferred from the config
it would be checked against.

**Fold resolution required a scoping decision.** `loso_fold_for_subject` raises when a
subject appears in more than one fold's `val_subjects.json`, and every new fold collides
with an archived one:

```
ValueError: Subject 1 is in the validation set of multiple folds
            ['version_39', 'version_51']; the LOSO assignment is ambiguous.
```

Resolved by building `lightning_logs_refilter/`, a directory of symlinks to only the ten
new folds, and passing `--log-dir lightning_logs_refilter` to every Stage 3 script.
`log_dir` is used only for fold resolution and never for output paths, so this changes
nothing else. `lightning_logs/` is left untouched, keeping the archived checkpoints and
the `outputs/` Hydra provenance trail in place.

### B.5 Checkpoints and seeds

Full per-fold record — checkpoint path, best epoch, val AUC, sha256, resolved filter
parameters, seed and every config group — is committed at
[`experiments/stage3/refilter_fold_manifest.csv`](../experiments/stage3/refilter_fold_manifest.csv).
All ten folds: `seed=42`, `lr=1.0e-3`, `preprocessing_group=p300_1_24`,
`training_group=default`, `evaluation_group=p300_loso_full`.

| version | best epoch | val AUC | checkpoint sha256 |
|---|---|---|---|
| version_42 | 25 | 0.9410 | `dd56b670ea4a65f727c565d592206327687310e36018bef684717abd0dce905b` |
| version_43 | 68 | 0.9614 | `3c284d4cd720c5405e40aa26c1c42e7bc861372bfbad49c593dfbc4e8d46e6fc` |
| version_44 | 29 | 0.8535 | `ea90996ea5d166908ae5f9f95f710749abbc3256f0ec0e508444e0370eec1b8c` |
| version_45 | 10 | 0.8716 | `282503a31bf9f47dcc6aec1ce6f086d57664570356f152d1d331b355ca2ba253` |
| version_46 | 26 | 0.8767 | `c1ed2657bec88ffb9b05c21fb1b7b260099a47e8cc6e0720ffa90eb8c43fab7b` |
| version_47 | 14 | 0.9429 | `f010e17f32ca02b3d723a7574c5764e80c4b386f06b1eeb3278932a2d0d5031d` |
| version_48 | 48 | 0.9091 | `a036cfafff88efda56c19f0d4594fc6e70db7369a6e5151668eba872d7b35e1f` |
| version_49 | 39 | 0.8621 | `02e9a1167e5555dbe73a37d7e5d79e724eb2b98c71db9a93614c6ac22b7b6384` |
| version_50 | 27 | 0.9188 | `ab4449d8c52c31be99f6b9fb2f3f25fb5639301bacd1515ab8c24065c96c1c46` |
| version_51 | 42 | 0.8566 | `aaf4dc096af8c7997a25d07574cdb5cfb74e7fc21d56e2755b88d1317d5003a3` |

### B.6 Stage 2 val AUC, old vs new

Compared **by subject**, not by fold index — see §F.2 for why the fold indices differ.

| subj | old fold | old val AUC | new fold | new val AUC | Δ |
|---|---|---|---|---|---|
| S1 | version_39 | 0.7208 | version_51 | 0.8566 | +0.1358 |
| S2 | version_34 | 0.7953 | version_50 | 0.9188 | +0.1235 |
| S3 | version_32 | 0.7743 | version_49 | 0.8621 | +0.0878 |
| S4 | version_38 | 0.8198 | version_48 | 0.9091 | +0.0893 |
| S5 | version_36 | 0.7494 | version_47 | 0.9429 | +0.1935 |
| S6 | version_37 | 0.6395 | version_46 | 0.8767 | +0.2372 |
| S7 | version_33 | 0.8294 | version_45 | 0.8716 | +0.0422 |
| S8 | version_35 | 0.7678 | version_44 | 0.8535 | +0.0857 |
| S9 | version_40 | 0.8162 | version_43 | 0.9614 | +0.1452 |
| S10 | version_31 | 0.8513 | version_42 | 0.9410 | +0.0897 |

**mean 0.7764 → 0.8994 (+0.1230). 10/10 subjects improved, 0 worsened.**
Wilcoxon signed-rank p = 0.00195 (the two-sided floor at n=10).
Range: old [0.6395, 0.8513], new [0.8535, 0.9614]. The weakest old fold gained the most.

`docs/design.md` §6.2 predicts within-session epoch AUC of 0.85–0.92 for BNCI2014_009.
7/10 new folds land inside that interval, 3 above, 0 below. **The filter was the
explanation for the depressed Stage 2 performance.**

One caveat on that comparison, since three different quantities are in play: §6.2's
0.85–0.92 is a *within-session* prediction, the table above is *leave-one-subject-out*
validation AUC, and the ~0.72 figure quoted elsewhere is the Stage 3 cohort *baseline*
AUC (§C). The table is internally consistent — same metric, same subjects, same
everything but the filter — but it is not literally the §6.2 quantity, so "now matches
the prediction" would overstate it.

---

## C. Old vs new, side by side

Every number in this section is produced by running the repo's own reducers —
`analyze_stage3.py`, `analyze_gated.py` — over the archived CSV and the new CSV in turn,
so both columns are computed identically. The headline arm is **`stratified`,
`ea_ref=session`**, which is the arm the previously reported figures came from
(confirmed: the reducer reproduces 0.7162 / −0.0157 / 0.5573 / −0.0039 exactly from the
archived files).

### C.0 Grid integrity

All five row counts match the archived runs exactly, so no grid moved:

| file | rows | expected |
|---|---|---|
| `experiments/stage3/p300_results.csv` | 1,750 | 1,750 ✅ |
| `experiments/stage3/als_results.csv` | 10,000 | 10,000 ✅ |
| `experiments/roundc/als_character.csv` | 1,520 | 1,520 ✅ |
| `experiments/roundc/p300_gated.csv` | 500 | 500 ✅ |
| `experiments/roundc/als_gated.csv` | 4,000 | 4,000 ✅ |

P300 is 1,250 `session` + 500 `calibration`, five arms, seeds 42–46, all `status=ok`.
ALS is 1,000 rows per fold × 10 folds, `fold_mode=average`, `session` only, all
`status=ok`. The MI rows in the regenerated `gated_summary.csv` are **byte-identical** to
the archived ones (`old == new` compares True), which independently confirms MI was not
re-run or re-derived.

### C.1 Headline table

| quantity | old (≈6.4–24 Hz) | new (1–24 Hz) |
|---|---|---|
| P300 mean baseline AUC (cohort) | 0.716 | **0.901** |
| P300 mean ΔAUC at N=480 | −0.0157 | **−0.0032** |
| ALS mean baseline AUC (cohort) | 0.557 | **0.711** |
| ALS mean ΔAUC at N=480 | −0.0039 | **+0.0067** |
| ALS character accuracy, pretrained | 0.058 @ 2.36 reps | **0.226 @ 2.90 reps** |
| ALS character accuracy, adapted | 0.054 @ 3.38 reps | **0.254 @ 3.17 reps** |
| ALS ITR, pretrained / adapted | 0.358 / 0.307 bits/min | **4.038 / 4.444 bits/min** |
| Stage 2 mean val AUC | 0.7764 | **0.8994** |

**Caveat on the character rows.** The "pretrained" figure averages all 80 `unadapted`
rows (that variant has only `calib_chars=0`, so it is unambiguous). The "adapted" figure
is the **`calib_chars=5`** condition specifically — that is what reproduces the
previously reported 0.054 / 3.38 / 0.307 exactly. So the original pairing compares an
all-rows pretrained mean against a single calibration size. Kept for comparability;
all three conditions are in C.5.

### C.2 P300 calibration curve — `stratified`, `session`, n=10

| N | old meanΔ | old 95% CI | old impr | old p | new meanΔ | new 95% CI | new impr | new p |
|---|---|---|---|---|---|---|---|---|
| 24 | −0.0748 | [−0.0969, −0.0536] | 0/10 | 0.0020 | **−0.0222** | [−0.0285, −0.0155] | 0/10 | 0.0020 |
| 48 | −0.0635 | [−0.0783, −0.0472] | 0/10 | 0.0020 | **−0.0190** | [−0.0257, −0.0129] | 0/10 | 0.0020 |
| 96 | −0.0471 | [−0.0613, −0.0334] | 0/10 | 0.0020 | **−0.0124** | [−0.0164, −0.0086] | 0/10 | 0.0020 |
| 240 | −0.0262 | [−0.0353, −0.0167] | 1/10 | 0.0039 | **−0.0060** | [−0.0090, −0.0030] | 1/10 | 0.0059 |
| 480 | −0.0157 | [−0.0225, −0.0087] | 1/10 | 0.0039 | **−0.0032** | [−0.0057, −0.0010] | 3/10 | 0.0371 |

`p` is the exact two-sided Wilcoxon signed-rank p; 0.0020 is the floor at n=10.
Every new CI still excludes zero. Cohen's dz stays large and negative
(−1.96, −1.69, −1.82, −1.15, −0.79 across N).

`ea_ref=calibration` (N=240/480 only) moves the same way: −0.0263 → −0.0060 and
−0.0157 → −0.0032.

### C.3 ALS calibration curve — `stratified`, `session`, n=8

| N | old meanΔ | old 95% CI | old impr | old p | new meanΔ | new 95% CI | new impr | new p |
|---|---|---|---|---|---|---|---|---|
| 24 | −0.0333 | [−0.0456, −0.0208] | 0/8 | 0.0078 | **−0.0521** | [−0.0704, −0.0362] | 0/8 | 0.0078 |
| 48 | −0.0342 | [−0.0442, −0.0245] | 0/8 | 0.0078 | **−0.0534** | [−0.0694, −0.0386] | 0/8 | 0.0078 |
| 96 | −0.0238 | [−0.0371, −0.0085] | 1/8 | 0.0391 | **−0.0286** | [−0.0359, −0.0206] | 0/8 | 0.0078 |
| 240 | −0.0128 | [−0.0258, +0.0031] | 2/8 | 0.2500 | **−0.0049** | [−0.0102, +0.0004] | 2/8 | 0.1953 |
| 480 | −0.0039 | [−0.0194, +0.0145] | 2/8 | 0.4609 | **+0.0067** | [+0.0012, +0.0131] | 6/8 | 0.0781 |

Note the sign flip at N=480 and the **conflict between the two statistics there**: the
bootstrap 95% CI excludes zero, but the exact Wilcoxon p is 0.0781, which does not clear
0.05 at n=8 (the floor is 0.0078). Discussed in §D.3.

### C.4 Per-subject and per-patient vectors (N=480, `stratified`, `session`)

**P300** — baseline AUC and ΔAUC per subject, mean over seeds:

| subj | base old | base new | Δbase | delta old | delta new |
|---|---|---|---|---|---|
| S1 | 0.6798 | 0.8594 | +0.1796 | −0.0139 | +0.0000 |
| S2 | 0.7425 | 0.9186 | +0.1761 | +0.0065 | +0.0000 |
| S3 | 0.7413 | 0.8574 | +0.1161 | −0.0123 | −0.0104 |
| S4 | 0.7236 | 0.9116 | +0.1879 | −0.0243 | −0.0026 |
| S5 | 0.7015 | 0.9447 | +0.2431 | −0.0115 | −0.0056 |
| S6 | 0.5851 | 0.8838 | +0.2988 | −0.0107 | +0.0012 |
| S7 | 0.7552 | 0.8719 | +0.1167 | −0.0148 | −0.0028 |
| S8 | 0.7025 | 0.8560 | +0.1534 | −0.0101 | −0.0094 |
| S9 | 0.7978 | 0.9612 | +0.1634 | −0.0327 | −0.0017 |
| S10 | 0.7440 | 0.9408 | +0.1969 | −0.0329 | −0.0005 |
| **mean** | **0.7173** | **0.9005** | **+0.1832** | **−0.0157** | **−0.0032** |

Baseline improved for **10/10** subjects. Subjects with Δ>0: 1/10 → 3/10.

**ALS** — per patient, mean over seeds and the 10 averaged folds:

| patient | base old | base new | Δbase | delta old | delta new |
|---|---|---|---|---|---|
| P1 | 0.5618 | 0.7190 | +0.1572 | −0.0220 | −0.0044 |
| P2 | 0.5762 | 0.6561 | +0.0799 | −0.0085 | +0.0151 |
| P3 | 0.5970 | 0.8023 | +0.2053 | −0.0365 | −0.0012 |
| P4 | 0.5690 | 0.6437 | +0.0747 | −0.0071 | +0.0015 |
| P5 | 0.5121 | 0.6660 | +0.1538 | +0.0180 | +0.0236 |
| P6 | 0.5610 | 0.6992 | +0.1382 | −0.0215 | +0.0055 |
| P7 | 0.5328 | 0.7251 | +0.1923 | +0.0482 | +0.0043 |
| P8 | 0.5439 | 0.7724 | +0.2285 | −0.0016 | +0.0092 |
| **mean** | **0.5567** | **0.7105** | **+0.1537** | **−0.0039** | **+0.0067** |

Baseline improved for **8/8** patients. Patients with Δ>0: 2/8 → 6/8. Every old baseline
was in [0.512, 0.597]; every new baseline is in [0.644, 0.802].

### C.5 ALS character-level metrics

Mean over 8 patients × 10 folds (× 3 seeds for the adapted/gated variants):

| variant | epoch AUC old → new | acc @ selected k, old → new | selected k, old → new | ITR (bits/min) old → new |
|---|---|---|---|---|
| pretrained (`unadapted`) | 0.5581 → **0.7185** | 0.0581 → **0.2256** | 2.362 → **2.900** | 0.3585 → **4.0380** |
| adapted (all `calib_chars`) | 0.5416 → **0.7089** | 0.0582 → **0.2497** | 3.342 → **2.987** | 0.3449 → **4.4528** |
| gated (all `calib_chars`) | 0.5521 → **0.7160** | 0.0597 → **0.2363** | 2.604 → **2.875** | 0.4325 → **4.2603** |

By calibration size, new (old in parentheses):

| variant | calib_chars | acc @ selected k | selected k | ITR |
|---|---|---|---|---|
| adapted | 2 | 0.2373 (0.0508) | 2.917 (3.771) | 4.3179 (0.3221) |
| adapted | 5 | 0.2544 (0.0544) | 3.167 (3.375) | 4.4437 (0.3071) |
| adapted | 10 | 0.2575 (0.0694) | 2.879 (2.879) | 4.5968 (0.4056) |
| gated | 2 | 0.2192 (0.0602) | 2.867 (2.562) | 3.9466 (0.3879) |
| gated | 5 | 0.2446 (0.0590) | 2.987 (2.729) | 4.3195 (0.4409) |
| gated | 10 | 0.2452 (0.0600) | 2.771 (2.521) | 4.5148 (0.4689) |

Chance character accuracy on a 6×6 BNCI2014_008 matrix is 1/36 ≈ 0.028. The old
pretrained 0.058 was ≈2× chance; the new 0.226 is ≈8× chance.

### C.6 Gated adaptation

Deployed ΔAUC after the accept/reject gate, `stratified` arm. N=24 is `ungateable` in
both generations (below the `--val-floor` of 8), which is why the ungateable counts are
100/500 for P300 and 800/4000 for ALS — identical to the archive.

| cohort | N | old acc. rate | old deployed Δ | old p | new acc. rate | new deployed Δ | new p |
|---|---|---|---|---|---|---|---|
| P300 | 48 | 0.38 | −0.00313 | 0.0234 | 0.20 | **−0.00067** | 0.0781 |
| P300 | 96 | 0.52 | −0.00484 | 0.0273 | 0.44 | **−0.00199** | 0.0059 |
| P300 | 240 | 0.48 | −0.00607 | 0.0098 | 0.64 | **−0.00189** | 0.0195 |
| P300 | 480 | 0.48 | −0.00521 | 0.0273 | 0.56 | **−0.00139** | 0.0488 |
| ALS | 48 | 0.3825 | −0.00133 | 0.1484 | 0.325 | **−0.00199** | 0.0078 |
| ALS | 96 | 0.66 | −0.00348 | 0.1953 | 0.6425 | **−0.00489** | 0.0078 |
| ALS | 240 | 0.755 | −0.00182 | 0.5469 | 0.70 | **−0.00219** | 0.0547 |
| ALS | 480 | 0.68 | +0.00046 | 0.5469 | 0.675 | **+0.00211** | 0.3828 |

Gating still converts a large negative raw Δ into a near-zero deployed Δ, and still
fails to turn adaptation into a net gain at any size for P300.

---

## D. Which conclusions changed

### D.1 The ALS cohort no longer sits near chance — this framing must change

**Changed, materially.** Every ALS metric moved out of the near-chance regime:

- Epoch-level baseline AUC **0.557 → 0.711** (all 8 patients improved; the old range
  [0.512, 0.597] and the new range [0.644, 0.802] do not overlap).
- Character-level pretrained accuracy **0.058 → 0.226**, against a 0.028 chance floor —
  from ≈2× chance to ≈8× chance.
- Pretrained ITR **0.358 → 4.038 bits/min**, an 11× increase.

The previous claim that the ALS cohort's near-chance transfer performance "bounds what
the study can claim" **no longer holds on these numbers**. A 0.711 AUC cohort is a weak
but genuine decoder, not a coin flip, and conclusions premised on the floor being near
chance need rewriting rather than rescaling. This is the single largest interpretive
change in the re-run.

Two honest qualifications. First, 0.711 AUC and 22.6% character accuracy are still poor
in absolute terms — far from a deployable speller — so "the ALS transfer target is hard"
survives; it is "indistinguishable from chance" that does not. Second, the character
numbers inherit the double-filtering caveat in §F.3.

### D.2 Healthy P300 adaptation is still negative everywhere and still attenuates

**Unchanged qualitatively; ~4× smaller quantitatively.** Under the corrected filter,
mean ΔAUC is negative at all five calibration sizes, every bootstrap CI still excludes
zero, the monotone attenuation with N is intact (−0.0222 → −0.0032 across N=24→480), and
0/10 subjects improve at N≤96 with the exact Wilcoxon p at its n=10 floor of 0.0020.

So both conclusions hold. What changes is magnitude: the effect is roughly a quarter of
its former size, and the count of subjects who benefit at the largest calibration size
rises from 1/10 to 3/10. Any statement of the form "adaptation costs ~1.6 AUC points at
N=480" becomes "~0.3 points".

### D.3 The small-sample-overfitting reading holds for P300, and needs a caveat for ALS

**Holds for P300, complicated for ALS.**

For P300 the reading is if anything cleaner: a much stronger backbone (val AUC 0.899,
baseline 0.901) leaves less headroom for a small calibration set to help, and the harm
shrinks and attenuates with N exactly as an overfitting account predicts.

For ALS two things cut against a simple restatement:

1. **Harm at small N got worse, not better**: −0.0333 → −0.0521 at N=24 and −0.0342 →
   −0.0534 at N=48, both at the n=8 Wilcoxon floor of 0.0078 with 0/8 patients
   improving. A stronger backbone apparently has *more* to lose when handed 24–48
   calibration epochs. That is consistent with an overfitting story — there is more
   good signal to damage — but it is the opposite sign of movement from P300's, and it
   is worth stating rather than smoothing over.
2. **The sign flips at N=480**: −0.0039 → **+0.0067**, with 6/8 patients improving where
   2/8 did before. **I would not report this as a positive adaptation effect.** The
   bootstrap 95% CI [+0.0012, +0.0131] excludes zero, but the exact two-sided Wilcoxon p
   is 0.0781 and does not clear 0.05 at n=8. The two statistics disagree, and with n=8
   the CI is the less trustworthy of the two here. The defensible claim is that ALS
   adaptation at the largest calibration size is **no longer detectably harmful**, not
   that it helps.

Also note the character-level result now favours adaptation (adapted ITR 4.45 vs
pretrained 4.04; accuracy 0.250 vs 0.226) where previously it slightly disfavoured it
(0.345 vs 0.358). That is a reversal, and it rests on the double-filtered path in §F.3,
so it should be treated as provisional until that is fixed.

### D.4 Conclusions that did not change

- Gating still cannot make adaptation net-positive for healthy P300 at any calibration
  size, and still works by suppressing a large negative rather than capturing a gain.
- N=24 remains `ungateable` in both cohorts, so gating is still unavailable exactly
  where adaptation is most harmful (ADR-27's rationale is untouched).
- The five split arms still rank the same way, and `stratified` /
  `stratified_task_only` remain numerically indistinguishable.
- MI is entirely unaffected: not re-run, verified bit-identical, and the MI rows of the
  regenerated `gated_summary.csv` are byte-identical to the archived ones.

---

## E. ADR

Recorded as ADR-28 in [`docs/DECISIONS.md`](DECISIONS.md).

---

## F. What could not be verified

### F.1 The FIR filter is longer than the epoch it filters

A 1–24 Hz zero-phase FIR at 128 Hz needs **423 taps**; a P300 epoch is **102 samples**.
MNE reports this and proceeds:

```
Filter length: 423 samples (3.305 s)
RuntimeWarning: filter_length (423) is longer than the signal (102),
                distortion is likely.
```

`src/preprocessing/filters.py:26-27` wraps the call in
`warnings.simplefilter("ignore")`, so this has been silent throughout the project's
history. The old 8–30 Hz filter had the same problem less severely (213 taps vs 102).

Consequence: the realized per-epoch response is not a clean 1–24 Hz passband. Several
ratios exceed 1.0 (1.12 at 2 Hz, 1.20 at 20 Hz) and the stopband above 24 Hz is far
weaker than the config implies (0.63 at 30 Hz, versus 0.22 on a long signal). **So "the
epochs were filtered at 1–24 Hz" is not literally true at the epoch level.**

Why the re-run is still sound: MOABB applies its own 1–24 Hz paradigm filter to the
*continuous* data before epoching, where filter length is not a constraint. The
epoch-level re-application is therefore largely redundant rather than load-bearing, and
the re-run is strictly closer to intent than the archived state. But this is a real
defect, it was not introduced by this fix, and it was deliberately **not** fixed in this
pass to keep the commit to one variable. It should be one commit together with the
`als_character.py` double-application below.

### F.2 The archived subject→fold assignment is not reproducible

The new folds assign subjects to fold indices in clean descending order (fold 0 → S10 …
fold 9 → S1). The archived runs used a scrambled assignment (fold 0 → S10, fold 1 → S3,
fold 2 → S7, fold 3 → S2, fold 4 → S8, fold 5 → S5, fold 6 → S6, fold 7 → S4,
fold 8 → S1, fold 9 → S9).

What was ruled out, by inspection rather than assumption:

- `GroupKFoldSplitter` is **byte-identical** between `7fd8a4d` (the commit live during
  the archived runs, 2026-07-23) and `HEAD`.
- The only later commit touching `src/datasets/splits.py` (`8a67c00`, 2026-07-27)
  changed the *calibration*-split functions, not this class.
- Installed scikit-learn is **1.9.0**, matching `requirements.lock.txt`, and the current
  environment reproduces the **new** mapping exactly.

**So the archived assignment cannot be reproduced under the pinned environment, and the
cause cannot be determined** — the scikit-learn version actually in use on 2026-07-24
was never recorded. This is stated rather than guessed.

**It affects no result.** With `n_splits == n_subjects == 10`, the LOSO-10 partition is
unique up to fold *ordering*: every fold trains on the other nine subjects and validates
on one, so which index a subject occupies changes no model's training data. Both
generations hold out all 10 subjects exactly once. All comparisons in this document are
keyed by subject, and Stage 3 resolves folds per subject from `val_subjects.json`, so
neither is affected.

### F.3 `als_character.py` still applies the band twice

`_extract_coded_epochs` filters correctly at 1–24 Hz and then passes the result through
`pp.transform`, which re-applies the band. Before the fix that was 1–24 Hz followed by
8–30 Hz — a genuine band conflict that removed the low-frequency P300 content twice
over. After the fix it is 1–24 Hz followed by 1–24 Hz.

**Assessment: acceptable for this pass, but it needs a separate fix.** Re-applying the
same band is close to idempotent in the passband interior; a second pass mainly steepens
the effective rolloff and compounds the §F.1 edge distortion rather than moving the
passband. So it no longer destroys signal the way the old mismatch did, and the ALS
character numbers below are interpretable. It remains a redundant filter whose
transition-band behaviour is not what the config describes. **Not changed unilaterally
in this pass**, per instruction.

### F.4 `fig6_als_per_patient` could not be regenerated

`fig2_delta_vs_size_p300.*` was regenerated from the new `p300_results.csv`.
`fig6_als_per_patient.*` was **not**, and cannot be by running `make_figures.py`:
`fig_als_per_patient` (line 420) and `fig_degradation` (line 361) are both defined
*after* the `if __name__ == "__main__": main()` block at line 355, and `main()` never
calls either. They are unreachable from the script's entry point, so whatever produced
the archived fig5 and fig6 was a separate invocation that is not recorded in the repo.

Consequence: `results/figures/` currently has no ALS per-patient figure. The archived
copy remains at `experiments/stage3/archive_prefilter_fix/figures/fig6_als_per_patient.*`
but **depicts the superseded numbers and must not be reused**. Regenerating it needs
either a driver added to `make_figures.py` or the original invocation recovered — not
done here, since adding a figure call is outside the scope of a filter fix.

### F.5 MI figure files were rewritten as a side effect

Running `make_figures.py` to regenerate the P300 figure also rewrote
`fig1_pipeline.*`, `fig2_delta_vs_size_mi.*`, `fig3_per_subject_N200_mi.*` and
`fig4_eval_composition_mi.*` — `main()` regenerates the MI figures unconditionally
before reaching the P300 branch. This was not intended and it brushes against the
read-only-on-MI constraint.

What can be said: every MI **data** artifact is unmodified — `git status` reports
`mi_results.csv`, `mi_summary.csv`, `mi_confound_regression.csv`, `mi_splithalf.csv`,
`mi_k0_17_gated.csv` and `mi_orthogonal_negatives.csv` all clean — so the figures were
redrawn by identical code from identical inputs and their content is equivalent.

What cannot be said: that the files are byte-identical. `results/` is gitignored, so the
previous bytes were never version-controlled and are now overwritten; PDF and SVG output
embeds a creation timestamp regardless. **The equivalence is inferred from unchanged
inputs, not verified against the old bytes.** `fig3_per_subject_N150_mi.*` and
`fig5_degradation_*.*` kept their original 2026-07-27 timestamps and were not touched.

### F.6 Not independently checked

- The archived numbers themselves were taken as given from the archived CSVs; they were
  not recomputed from the archived checkpoints.
- The 0.0022 discrepancy in the earlier reproduction attempt referenced in
  `docs/AUDIT.md` is untouched by this work and remains unresolvable.
