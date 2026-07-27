# Pre-audit Stage 3 results — INVALID, retained for provenance only

These are the files as of commit `02af554`, before the Phase 0 audit. Do not cite
them. See `docs/AUDIT.md` for the full findings.

| file | status |
|---|---|
| `mi_results.csv` | **invalid** — wrong electrode montage (§0.1) |
| `mi_results_temporal.csv` | **invalid** — same montage defect; also has no n=200 rows |
| `mi_wilcoxon.csv` | **invalid** — derived from the above, plus a sign error at n=10 and mislabelled p columns (§0.5) |
| `p300_results.csv` | superseded — montage was correct (§0.2), but the run is not comparable to the temporal arm and predates the reproducibility fixes |
| `p300_results_temporal.csv` | superseded — only subjects 9 and 10, under the older 2-held-out regime |
| `p300_sign_test.csv` | superseded — derived from the above |

Every MI number here was produced with a 17-channel list that shares **zero array
positions** with the list the Stage 2 checkpoints were trained on. The correct
montage raises held-out AUC by +0.0586 (§0.6).
