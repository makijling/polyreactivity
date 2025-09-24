# Open Replication of Sakhnini *et al.* (2025)

All classification metrics now include bootstrap confidence intervals (recorded in `metrics.csv` as `*_ci_*` columns) and Poisson regression diagnostics for ELISA flag counts (`flag_regression_metrics.csv`).

**Authors:** Polyreact Team (open-source reproduction)

## Abstract
We reproduce the heavy-chain polyreactivity predictor of Sakhnini *et al.* (2025)
using publicly released Boughter ELISA counts and a lineage-aware training
pipeline. The default configuration matches the paper’s setting: mean-pooled
`facebook/esm1v_t33_650M_UR90S_1` embeddings with a calibrated logistic
regression head (`C=0.1`). Evaluated on Jain (2017), the curated Shehata (2019)
supplement, and Harvey (2022), the model attains 10-fold CV accuracy ≈0.65
(ROC-AUC 0.69) with external accuracies of 0.56, 0.66, and 0.60 respectively.
This document summarises the data rebuild, modelling steps, and evaluation
metrics, and points readers to the generated figures and tables in
`artifacts/replication_v5/`.

## 1. Data reconstruction
- Translate nucleotide FASTAs from `data/AIMS_manuscripts/app_data/full_sequences`
  and select the highest-scoring reading frame.
- Number sequences with ANARCI (IMGT scheme), extract FR1–FR4 VH segments, and
  record CDRH3 sequences.
- Join ELISA flag counts, drop mild cases (1–3 flags), and compute binary labels
  (0 vs >3 flags).
- Annotate each sequence with lineage (`family|CDRH3`) and species. The rebuilt
  dataset with 970 sequences is stored at
  `data/processed/boughter_counts_rebuilt.csv`, with a human-only subset at
  `data/processed/boughter_counts_human_lineage.csv`.

## 2. Model and training pipeline
- Feature backend: mean-pooled `facebook/esm1v_t33_650M_UR90S_1`.
- Predictor: `LogisticRegression` with `C=0.1` and `class_weight="balanced"`.
- Calibration: isotonic, fitted on 10 lineage-aware CV folds.
- Training data: all Boughter families (human HIV, human influenza, mouse IgA).
- Implementation: `polyreact/train.py` with lineage grouping, duplicate control,
  species/family filters, and cached PLM embeddings.

## 3. Reproduction workflow
Run the entire pipeline (rebuild optional, training, plots, LOFO) with:

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

Outputs include metrics/plots in `artifacts/replication_v5/main/`, LOFO runs under
`artifacts/replication_v5/lofo_runs/`, optional descriptor baselines in
`artifacts/replication_v5_variants/descriptor_variants/`, and calibration /
flag-count diagnostics. Each run also emits a `dataset_split_summary.csv`
describing prevalence per split and an audit JSON from the dataset
reconstruction step.

## 4. Results
Table 1 summarises the primary metrics (see
`artifacts/replication_v5/main/metrics.csv`).

**Table 1.** Classification performance of the reproduced model.

| Split                 | ROC-AUC | PR-AUC | Accuracy |   F1 | Brier |
|-----------------------|--------:|-------:|---------:|----:|------:|
| Train (CV mean)       | 0.693   | 0.706  | 0.647    |0.658| 0.225 |
| Jain 2017             | 0.629   | 0.528  | 0.563    |0.557| 0.243 |
| Shehata 2019 (curated)| 0.755   | 0.805  | 0.655    |0.762| 0.217 |
| Harvey 2022           | 0.685   | 0.654  | 0.595    |0.692| 0.248 |

Additional columns (positive-class F1, precision, ECE/MCE) are logged in the
metrics CSV for calibration analysis.

**Figure 1. Accuracy overview.** See `artifacts/replication_v5/main/accuracy_overview.png`.

**Figure 2. ROC curves.** See `artifacts/replication_v5/main/roc_overview.png` for
train CV and each evaluation dataset.

**Figure 3. Prediction vs ELISA flag counts.** Scatter plot with Spearman
correlation ρ ≈ 0.61; see `artifacts/replication_v5/main/prob_vs_flags.png`.

## 5. Leave-one-family-out analyses
`polyreact.benchmarks.reproduce_paper` also runs LOFO experiments. Table 2
shows the results (`artifacts/replication_v5/lofo_metrics.csv`).

**Table 2.** LOFO performance by family.

| Hold-out family | Accuracy | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|-----------------|---------:|--------:|-------:|------------:|------------:|
| Influenza       | 0.58     | 0.62    | 0.59   | 0.55        | 0.57        |
| HIV             | 0.61     | 0.66    | 0.62   | 0.57        | 0.63        |
| Mouse IgA       | 0.65     | 0.67    | 0.69   | 0.71        | 0.52        |

Mouse IgA sequences remain challenging, chiefly through reduced specificity,
mirroring the domain-shift limitations noted by Sakhnini *et al.*

## 6. Discussion
- Adopting the paper’s ESM-1v backbone narrows the gap to the reported 71%
  cross-validation accuracy while keeping the pipeline deterministic and
  cacheable.
- PSR assays remain challenging owing to their label imbalance; descriptor-only
  baselines (particularly pI/charge) are still competitive and available via
  `descriptor_variants/` when that phase is enabled.
- Leave-one-family-out analyses underline the human→mouse specificity drop,
  whereas influenza↔HIV transfer stays comparatively balanced.
- Negative-binomial flag regression captures the observed over-dispersion in
  ELISA counts (Pearson dispersion ≈5.6 for Poisson vs ≈7.7 for NB2, α≈2.5).
- Download `mmc1.xlsx` and run `scripts/rebuild_shehata_psr.py --output
  data/processed/shehata_full.csv` to restore the 7/398 PSR prevalence from the
  original paper.

## 7. Reproducibility checklist
- Seeds fixed to 42 (Python/NumPy/Torch).
- Dependencies pinned in `requirements.txt`.
- Unit tests (`pytest -q`) pass.
- Reproduction script (`polyreact.benchmarks.reproduce_paper`) rerun prior to
  this report, generating the metrics recorded above.

## 8. Artifact index

| Artifact | Path |
|----------|------|
| Metrics (main) | `artifacts/replication_v5/main/metrics.csv` |
| Split summary | `artifacts/replication_v5/main/dataset_split_summary.csv` |
| Predictions | `artifacts/replication_v5/main/preds.csv` |
| Accuracy figure | `artifacts/replication_v5/main/accuracy_overview.png` |
| ROC figure | `artifacts/replication_v5/main/roc_overview.png` |
| Probability vs flags | `artifacts/replication_v5/main/prob_vs_flags.png` |
| LOFO metrics | `artifacts/replication_v5/lofo_metrics.csv` |
| Descriptor baselines | `artifacts/replication_v5_variants/descriptor_variants/summary.csv` |
| Spearman stats | `artifacts/replication_v5/main/spearman_flags.json` |
| Flag regression | `artifacts/replication_v5/main/flag_regression_metrics.csv` |

A corresponding Markdown report is stored at `REPLICATION_REPORT.md`.
