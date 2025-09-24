# Polyreactivity Predictor

Reliable, reproducible pipeline for estimating antibody polyreactivity with
lightweight sequence descriptors, optional protein language model embeddings,
and calibrated linear classifiers. The project ships a Python package, CLI,
Docker image, Gradio Space UI, and automated benchmarks on small public
antibody panels.

## Features

- Deterministic training with stratified cross-validation, probability
  calibration, and reproducible artifact exports (model + feature state).
- Multiple feature backends: physicochemical descriptors, ESM-style PLM
  embeddings, or their concatenation with caching.
- Dataset loaders for Boughter 2020 (train/validation) and Jain 2017,
  Shehata 2019, Harvey 2022 (external tests) with de-duplication logic.
- Comprehensive metrics (ROC/PR AUC, accuracy, F1, Brier), reliability/ROC/PR
  plots, and structured predictions for every evaluated sequence.
- Built-in bootstrap confidence intervals for every split plus optional
  Poisson / negative-binomial regression on ELISA flag counts to model graded reactivity.
- Command-line tools for training and inference, plus a Gradio Space for quick
  single/batch scoring.
- Unit/smoke tests covering feature extraction, model training, metrics, and
  artifact round-tripping.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ANARCI requires `hmmer` and `muscle`; they are installed automatically inside
`Dockerfile`, but for local development install via your package manager.

## Quickstart

### Training

Download or prepare CSV files with columns `id, heavy_seq[, light_seq], label`
for each dataset. Then run:

```bash
python -m polyreact.train \
  --config configs/default.yaml \
  --train data/processed/boughter_counts.csv \
  --eval data/processed/jain.csv data/processed/shehata_curated.csv data/processed/harvey.csv \
  --report-to artifacts/
```
Use `--bootstrap-samples` and `--bootstrap-alpha` to control confidence
interval reporting (set `--bootstrap-samples 0` to disable).

Outputs:

- `artifacts/model.joblib` — trained model, calibration, and feature pipeline.
- `artifacts/metrics.csv` — wide-form metrics (train CV mean/std, train full,
  external datasets) with *_ci_lower/_ci_upper columns from bootstrap resampling.
- `artifacts/preds.csv` — per-sequence predictions (`id, source, split, y_true,
  y_score, y_pred`).
- `artifacts/flag_regression_metrics.csv` — Poisson regression summary on ELISA flag counts
  alongside per-fold predictions under `flag_regression_*`.
- `artifacts/*.png` — ROC / PR / reliability plots for each split.

### Inference CLI

```bash
python -m polyreact.predict \
  --input inputs/to_score.csv \
  --output outputs/preds.csv \
  --weights artifacts/model.joblib
```

Optional overrides: `--backend`, `--plm-model`, `--paired`, `--device`,
`--cache-dir`, `--config`.

### Python API

```python
from polyreact import api

records = [{"id": "sample", "heavy_seq": "EVQLV..."}]
preds = api.predict_batch(records, weights="artifacts/model.joblib")
```

### Benchmarks & Replication

Run the end-to-end training/benchmark script (ships with small fixtures; replace
paths with full datasets when available):

```bash
python -m polyreact.benchmarks.run_benchmarks \
  --config configs/default.yaml \
  --train data/Boughter2020.csv \
  --eval data/Jain2017.csv data/Shehata2019.csv data/Harvey2022.csv \
  --report-dir artifacts/benchmark
```

For the replication run captured in this repo we reconstructed the Boughter
dataset from the public FASTA files + ELISA flag counts bundled with the
`AIMS_manuscripts` repository (see `scripts/rebuild_boughter_from_counts.py`).
Using the paper-matching configuration (mean-pooled `facebook/esm1v_t33_650M_UR90S_1`
embeddings + logistic regression with `C=0.1`) on the rebuilt CSV
(`data/processed/boughter_counts_rebuilt.csv`) yields the following metrics
(`artifacts/replication_v5/main/metrics.csv`):

- Train CV accuracy ≈0.65 (ROC-AUC 0.69, PR-AUC 0.71)
- Jain 2017 accuracy 0.56 (ROC-AUC 0.63, PR-AUC 0.53)
- Shehata 2019 curated (88 sequences) accuracy 0.66 (ROC-AUC 0.75, PR-AUC 0.80)
- Harvey 2022 accuracy 0.60 (ROC-AUC 0.68, PR-AUC 0.65)

Leave-one-family-out experiments mirror the paper’s cross-species findings
(`artifacts/replication_v5/lofo_metrics.csv`): influenza 0.58/0.62 (accuracy/ROC-AUC),
HIV 0.61/0.66, mouse IgA 0.65/0.67.

Because the full 398-entry Shehata panel sits behind a Cloudflare-protected
download we ship the curated 88-entry supplement as
`data/processed/shehata_curated.csv`. Supplying a local copy of
`mmc1.xlsx` to `scripts/rebuild_shehata_psr.py --output data/processed/shehata_full.csv`
adds the imbalanced evaluation column to the same pipeline.

Regenerate the paper-style metrics, tables, and plots with:

```bash
python -m polyreact.benchmarks.reproduce_paper \
  --config polyreactivity/configs/default.yaml \
  --train-data polyreactivity/data/processed/boughter_counts_rebuilt.csv \
  --full-data polyreactivity/data/processed/boughter_counts_rebuilt.csv \
  --shehata-curated polyreactivity/data/processed/shehata_curated.csv \
  --output-dir polyreactivity/artifacts/replication_v5 \
  --batch-size 4 \
  --bootstrap-samples 200 \
  --skip-descriptor-variants --skip-fragment-variants
```

Additional toggles:

- `--skip-lofo`, `--skip-flag-regression`, `--skip-descriptor-variants`,
  `--skip-fragment-variants` to prune expensive phases. Running with
  `--skip-descriptor-variants --skip-lofo --skip-flag-regression` and a fresh
  `--output-dir` (e.g. `artifacts/replication_v5_fragments`) materialises the
  fragment suite while reusing the main configuration.
- `--shehata` to point at `data/processed/shehata_full.csv` once the full PSR
  screen is rebuilt.

The script will:

- optionally rebuild the VH dataset and emit an audit JSON summarising
  translation failures, deduplication, and flag filtering
- train the lineage-aware CV model (default: all Boughter families) and export
  `metrics.csv`, `preds.csv`, `dataset_split_summary.csv`
- run leave-one-family-out evaluations unless skipped
- launch descriptor-only baselines and CDR fragment ablations, writing results
  under `descriptor_variants/` and `fragment_variants/` when enabled
- generate Matplotlib figures (`accuracy_overview.png`, `roc_overview.png`,
  `prob_vs_flags.png`)
- fit Poisson and negative-binomial regressions on ELISA flag counts and export
  `flag_regression_metrics.csv` / `flag_regression_preds.csv`

### Docker

Build and execute the CLI inside a container:

```bash
docker build -t polyreact:latest .
docker run --rm -v "$PWD":/workspace polyreact:latest \
  python -m polyreact.predict --help
```

Environment variables honoured in Docker/CLI:

- `POLYREACT_DEVICE` — force `cpu`, `cuda`, or `auto` (default `auto`).
- `HF_HOME` — Hugging Face cache location (default `~/.cache/huggingface`).
- `ALLOW_LARGE_DOWNLOADS` — gates PLM smoke tests (set to `1` to permit).

### Gradio Space

Launch the interactive Space locally:

```bash
python space/app.py
```

Upload or place a trained artifact at `artifacts/model.joblib` to enable
predictions. The UI supports single-sequence scoring and CSV batch uploads.

## Replication report

A full replication log (metrics, figures, plots, and LOFO analyses) is available
after running:

```bash
python -m polyreact.benchmarks.reproduce_paper \
  --config polyreactivity/configs/default.yaml \
  --train-data polyreactivity/data/processed/boughter_counts_rebuilt.csv \
  --shehata-curated polyreactivity/data/processed/shehata_curated.csv \
  --output-dir polyreactivity/artifacts/replication_v5 \
  --batch-size 4 \
  --bootstrap-samples 200 \
  --skip-descriptor-variants --skip-fragment-variants
```

The generated figures, CSVs, dispersion diagnostics, and LOFO summaries are
described in `REPLICATION_REPORT.md`.

## Datasets

Run the helper script to fetch publicly hosted sources and build the canonical
CSVs expected by the loaders. To regenerate the parsed Boughter counts used
for the replication benchmark, run:

```bash
python scripts/prepare_datasets.py --sample-per-class 4000 --harvey-per-class 500
python scripts/rebuild_boughter_from_counts.py
```

The script downloads the curated supplements released by Tessier *et al.*
(derived from Jain 2017 and Shehata 2019) and the Harvey 2022 FASTA archives
served by the Kruse Lab, then writes processed files under `data/processed/`.

| Dataset        | Usage                   | Source (public)                                               |
|----------------|-------------------------|---------------------------------------------------------------|
| Boughter 2020  | Primary train/val (CV)  | Tessier Lab supplemental datasets S1/S2 (GitHub)              |
| Jain 2017      | External test           | Tessier Lab supplemental dataset S8 (GitHub)                  |
| Shehata 2019   | External test           | Tessier Lab supplemental dataset S3 (GitHub)                  |
| Harvey 2022    | External test           | Kruse Lab nanobody FASTA archives                             |

Consult the original publications before redistributing the raw files.

> **Shehata label policy:** Supplementary sheet S3 provides 88 paired VH/VL
> antibodies with a qualitative High/Low PSR flag. We map `High→1`, `Low→0`
> (51 positive, 37 negative) by default. The raw 398-entry PSR table with
> quantitative scores is gated behind a Cloudflare challenge; download
> `mmc1.xlsx` manually from the Cell supplemental materials and run
> `python scripts/rebuild_shehata_psr.py` to regenerate
> `data/processed/shehata.csv` plus an audit JSON. This restores the 7/398
> prevalence reported in the manuscript while keeping the loader contracts.

> **Note:** The Boughter reconstruction step expects the
> [`AIMS_manuscripts`](https://github.com/ctboughter/AIMS_manuscripts) repository
> to be cloned into `data/AIMS_manuscripts/` so that the per-antigen ELISA count
> files referenced in the paper are available.

## Testing & Quality

- `make test` — unit and smoke tests (< 60 s, offline friendly).
- `make lint` — Ruff + Black checks.
- `make type` — mypy type checking (`ignore_missing_imports` enabled).

PLM-heavy tests run only when `ALLOW_LARGE_DOWNLOADS=1` **and** cached
embeddings exist.

## Configuration

`configs/default.yaml` contains default hyperparameters (feature backend,
calibration, CV, output paths). Override keys via custom YAML or CLI flags.

Key sections:

- `feature_backend`: choose `descriptors`, `plm`, or `concat`; configure PLM
  model name, pooling, cache directory.
- `descriptors`: toggle ANARCI numbering and selected physico-chemical features.
- `model`: linear head type (`logreg` or `linear_svm`), regularisation, class
  weights.
- `calibration`: probability calibration method (`isotonic`, `platt`, or null).

## FAQ

- **When should I prefer descriptors over PLM embeddings?**
  Choose descriptors when you need rapid iteration, minimal dependencies, or a
  more interpretable model on severely imbalanced assays (e.g. PSR panels).
  PLM embeddings (default: ESM-1v) deliver the best cross-panel accuracy and
  calibration but require GPU memory and longer runtimes. Switching backends is
  a one-flag change (`--backend descriptors|plm|concat`).

## Project Structure

```
polyreactivity/
  configs/                # Default YAML configs
  polyreact/              # Python package
    data_loaders/         # Dataset loaders + utilities
    features/             # Feature backends & pipeline manager
    models/               # Linear heads + calibration helpers
    benchmarks/           # Benchmark entrypoint & notebook stub
    utils/                # IO, logging, metrics, plots, seeds
    api.py                # Python prediction API
    predict.py            # CLI entrypoint
    train.py              # Training entrypoint
  space/                  # Gradio Space app & docs
  tests/                  # Unit + smoke tests + fixtures
```

## Gradio & Deployment Notes

- Use `POLYREACT_MODEL_PATH` and `POLYREACT_CONFIG_PATH` env vars to point the
  Space to different artifacts/configs.
- CPU inference is the default; PLM models require GPU for best throughput but
  run on CPU for small batches.

## License & Attribution

This project is released under the MIT License (see `LICENSE`). Cite original
assays when redistributing dataset-derived metrics: Boughter et al. 2020, Jain
et al. 2017, Shehata et al. 2019, Harvey et al. 2022.
