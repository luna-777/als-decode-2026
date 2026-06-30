# ALS-Decode: EEG Brain–Computer Interface for Locked-In Syndrome

A three-stage EEG BCI for communication by people with ALS and Locked-In Syndrome.

- **Stage 1** (this repo, in progress): self-paced "brain switch" — detects communication intent (motor imagery) vs idle, on the PhysioNet EEGMMIDB dataset.
- **Stage 2** (future): P300-based speller on healthy subjects (BNCI 2014-009).
- **Stage 3** (future): transfer and adaptation to ALS patients (BNCI 2014-008).

See [docs/design.md](docs/design.md) for the full specification.

---

## Setup

Requires **conda** and an **Apple Silicon Mac** (arm64). CPU-only x86 also works; CUDA users should replace the `torch` pip line in `environment.yml` with the appropriate CUDA wheel.

```bash
conda env create -f environment.yml
conda activate als-decode
```

Verify:

```bash
make test    # should be green
make lint    # should pass
```

---

## Reproduce Stage 1

```bash
# 1. Warm the MOABB data cache (~3 GB, one-time download)
python scripts/download_data.py

# 2. Run the full Stage 1 LOSO evaluation
bash scripts/run_stage1.sh
```

Results are written to `results/stage1/` and logged to W&B (set `WANDB_MODE=offline` to skip the network).

---

## Development

| Command | What it does |
|---|---|
| `make test` | Run the full test suite |
| `make lint` | `ruff` check + format check |
| `make stage1` | Reproduce Stage 1 (alias for `run_stage1.sh`) |
| `make clean` | Remove build artifacts, Hydra outputs, W&B cache |

---

## Configuration

All experiments are config-driven via [Hydra](https://hydra.cc). The default config runs Stage 1 with EEGNet + LOSO evaluation:

```bash
python -m src.train                         # default: stage1 + EEGNet + LOSO
python -m src.train model=csp_lda          # classical comparator
python -m src.train evaluation=streaming   # streaming event evaluation
python -m src.train seed=123               # different seed
```

Config groups live in `configs/`. See [docs/design.md §8.4](docs/design.md) for the full Hydra schema.

---

## Repository layout (Stage 1 scope)

```
src/
  datasets/     — MOABB wrapper, DatasetSpec, group-aware splitters
  preprocessing/— Preprocessor (sole preprocessing class)
  models/       — BackboneEncoder (EEGNet), classical CSP+LDA
  training/     — LitEEG, EEGDataModule, losses, callbacks
  decoding/     — IntentSwitch (threshold / debounce / refractory)
  evaluation/   — StreamSimulator, asynchronous metrics, report
  utils/        — seed, logging, checkpoint I/O
configs/        — Hydra config tree
tests/          — unit + integration tests
scripts/        — data download, run scripts
docs/           — design spec, decisions log
```
