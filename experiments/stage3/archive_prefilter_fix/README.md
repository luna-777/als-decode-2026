# Pre-filter-fix P300 and ALS results — SUPERSEDED, retained for provenance only

Archived **2026-08-05**. Fixed by commit **`d90d18a`** ("Fix: Preprocessor never read
the bandpass config block"). Do not cite these numbers. See `docs/refilter_rerun.md`.

Every file here was produced under a band-pass filter that was **not** the documented
1–24 Hz at 128 Hz. Two independent faults compounded:

1. **`Preprocessor` never read the `bandpass:` config block.**
   `src/preprocessing/pipeline.py` resolved the passband via
   `self._get("band", [8.0, 30.0])`, but neither `configs/preprocessing/mi_8_30.yaml`
   nor `p300_1_24.yaml` defines a top-level `band` key — both nest the values under
   `bandpass:`. The hardcoded `[8.0, 30.0]` fallback therefore applied to every
   paradigm, and the whole `bandpass:` block was dead, `method` / `phase` /
   `fir_window` included.

2. **The P300 LOSO folds were trained under the MI preprocessing group.**
   `outputs/2026-07-24/22-57-24/.hydra/overrides.yaml` shows `version_31`…`version_40`
   were launched as `dataset=bnci_009 stage=stage2 evaluation=p300_loso_full fold=N`
   against the default `configs/config.yaml`, whose `defaults:` list selects
   `preprocessing: mi_8_30`. `configs/config_stage2.yaml` was correct but was never
   used. Consequence: `keep_sfreq=160.0` applied to 128 Hz data and
   `amplitude_reject_uv=150.0` where 100.0 was intended.

## Effective passband of these numbers

An 8–30 Hz filter designed for 160 Hz, applied to 128 Hz data, realizes
**≈ 6.4–24 Hz** (8 × 128/160 = 6.4, 30 × 128/160 = 24), composed with MOABB's own
1–24 Hz paradigm filter on the continuous data.

Measured response of the as-trained `version_31` preprocessor, sinusoids at 128 Hz,
filter-only (robust scaling disabled), at the true 0.8 s / 102-sample epoch length:

| 2 Hz | 5 Hz | 10 Hz | 20 Hz | 26 Hz | 30 Hz |
|---|---|---|---|---|---|
| 0.0059 | 0.0853 | 0.9795 | 1.0061 | 0.7786 | 0.2292 |

The low-frequency content that carries most of the P300 was very largely removed.

## Contents

| path | file | status |
|---|---|---|
| `stage3/` | `p300_results.csv`, `p300_summary.csv` | superseded — wrong passband |
| `stage3/` | `als_results.csv`, `als_summary.csv` | superseded — wrong passband |
| `roundc/` | `als_character.csv` | superseded — wrong passband |
| `roundc/` | `p300_gated.csv`, `als_gated.csv`, `gated_summary.csv` | superseded — wrong passband |
| `figures/` | `fig2_delta_vs_size_p300.{pdf,png,svg}` | superseded — derived from the above |
| `figures/` | `fig6_als_per_patient.{pdf,png,svg}` | superseded — derived from the above |

## MI is unaffected and is NOT archived here

`mi_8_30.yaml`'s `bandpass: {low: 8.0, high: 30.0}` happens to equal the old hardcoded
fallback exactly, so Fault 1 was a no-op for MI, and Fault 2 only ever mis-selected the
MI group *for P300 runs*. MI was verified bit-identical rather than re-run: a seeded
`(32, 17, 320)` float32 array through `lightning_logs/version_28/preprocessor.pkl`
yields output sha256
`e599954b933f8cd30104b3d5675a9aa764e2834cb9d30b11d8ddf80e3f8c552c` both before and
after the fix, with `np.array_equal(before, after) == True`.

Every MI artifact stays in its live location: `experiments/stage3/mi_results.csv`,
`mi_summary.csv`, `mi_confound_regression.csv`, `experiments/roundc/mi_*.csv`, and
`results/figures/fig2_delta_vs_size_mi.*`, `fig3_*_mi.*`, `fig4_*_mi.*`,
`fig5_degradation_*.*` (fig5 is MI — "apparent adaptation gain tracks backbone
degradation"). `fig1_pipeline.*` is a paradigm-neutral schematic and also stays.

## Superseded checkpoints

The `version_31`…`version_40` checkpoints under `lightning_logs/` are **not** moved —
`archive_pre_audit/` set no precedent for relocating checkpoints, and moving them would
break the `outputs/` Hydra provenance trail. They remain the artifacts these CSVs were
computed from. Their best-epoch val AUCs, for comparison against the re-run:

| subject | version | val AUC |
|---|---|---|
| S1 | version_39 | 0.7208 |
| S2 | version_34 | 0.7953 |
| S3 | version_32 | 0.7743 |
| S4 | version_38 | 0.8198 |
| S5 | version_36 | 0.7494 |
| S6 | version_37 | 0.6395 |
| S7 | version_33 | 0.8294 |
| S8 | version_35 | 0.7678 |
| S9 | version_40 | 0.8162 |
| S10 | version_31 | 0.8513 |

mean 0.7764, min 0.6395, max 0.8513, n = 10.

None of these folds wrote `montage.json` — they predate `MontageCheckpoint`, so their
provenance column is `montage_source = inferred_from_config`.
