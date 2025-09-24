# Polyreactivity Predictor — Model Card

## Model Details
- **Model name:** Polyreactivity Predictor (logistic regression baseline)
- **Version:** 0.1.0
- **Authors:** Polyreact team (see project README)
- **License:** MIT
- **Architecture:** Descriptor or PLM embeddings (default: `facebook/esm1v_t33_650M_UR90S_1`
  pooled over residues) followed by calibrated linear classifiers (logistic
  regression or linear SVM).
- **Source repository:** https://example.com/polyreactivity (update with canonical URL)

## Intended Use
- **Primary purpose:** Estimate the probability that an antibody sequence is
  polyreactive based on heavy-chain (and optionally light-chain) sequences.
- **Intended users:** Antibody discovery scientists, computational biology
  practitioners, ML researchers exploring polyreactivity signals.
- **Out-of-scope uses:** Clinical decision making, therapeutic selection without
  laboratory validation, scoring sequences from organisms drastically different
  from the training data.

## Training Data
- **Primary dataset:** Boughter et al. 2020 heavy-chain sequences with binary
  polyreactivity annotations.
- **Pre-processing:** Standardised residue casing, optional ANARCI numbering for
  physico-chemical descriptors, de-duplication across splits to avoid leakage.

## Evaluation Data
- Jain et al. 2017 (IgGs, reactive vs non-reactive).
- Shehata et al. 2019 (binding assay, positive vs negative; curated 88-sequence
  supplement bundled with Boughter distribution — the 398-entry PSR screen can
  be rebuilt locally).
- Harvey et al. 2022 (binary polyreactivity flag).

Each dataset is loaded via `polyreact.data_loaders.*` helpers and shares the
canonical schema `id, heavy_seq, light_seq, label`.

## Metrics
- **Classification:** ROC-AUC, PR-AUC, Accuracy, F1, Brier score.
- **Calibration diagnostics:** Reliability plots (expected vs observed).

Metrics below are produced by `python -m polyreact.benchmarks.reproduce_paper`:

| Split                 | ROC-AUC | PR-AUC | Accuracy |   F1 | Brier |
|-----------------------|--------:|-------:|---------:|----:|------:|
| Train (CV mean)       | 0.693   | 0.706  | 0.647    |0.658| 0.225 |
| Jain 2017             | 0.629   | 0.528  | 0.563    |0.557| 0.243 |
| Shehata 2019 (curated)| 0.755   | 0.805  | 0.655    |0.762| 0.217 |
| Harvey 2022           | 0.685   | 0.654  | 0.595    |0.692| 0.248 |

Additional diagnostics:

- **Leave-one-family-out:** influenza 0.58/0.62, HIV 0.61/0.66, mouse IgA
  0.65/0.67 (accuracy / ROC-AUC) — see `artifacts/replication_v5/lofo_metrics.csv`.
- **Bootstrap intervals:** `metrics.csv` ships `*_ci_lower` / `*_ci_upper`
  columns (this run used 200 resamples).
- **Spearman correlation** between predicted probabilities and ELISA flag
  counts (Boughter): ≈0.61 (`artifacts/replication_v5/main/spearman_flags.json`).
- **Flag-count regression:** Poisson and negative-binomial diagnostics live in
  `artifacts/replication_v5/main/flag_regression_metrics.csv`
  (`alpha≈2.5`, Pearson dispersion ≈5.6 / 7.7).



## Ethical Considerations & Limitations
- Assays across datasets differ, so absolute scores may not transfer between
  experimental platforms.
- Heavy-chain only models may underperform for antibodies whose reactivity is
  strongly influenced by light chains or paired context.
- Polyreactivity predictions should be treated as screening hints; experimental
  validation remains essential.
- Model biases may reflect training data skew (species, expression system,
  laboratory assays). Consider retraining or reweighting for new cohorts.

## Environmental Impact
- Descriptor-only models incur minimal compute; PLM embeddings require loading
  HF models (e.g. ESM-2 35M) and may benefit from GPUs.

## How to Reproduce
1. Acquire datasets (respect licensing/redistribution terms).
2. Install dependencies: `pip install -r requirements.txt` (ensure `hmmer` and
   `muscle` are available for ANARCI).
3. Run the end-to-end reproduction script:
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
4. Metrics, LOFO analyses, ROC/PR plots, calibration figures, and flag-count
   regressions will populate under `artifacts/replication_v5/`.

## Contact
For questions or to report issues, open an issue in the repository or contact
polyreact@example.com (update with active maintainer).
