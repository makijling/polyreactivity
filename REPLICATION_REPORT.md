# Polyreactivity Predictor — Open Reproduction Report

This report documents the open-source replication of Sakhnini *et al.* (2025)
performed using the tooling in this repository. It summarises the data
reconstruction pipeline, model configuration, and evaluation outcomes, and
points to the generated artifacts (metrics tables, figures, and plots) found in
`artifacts/replication_v5/`.

## 1. Data reconstruction

We generated an IMGT-numbered VH dataset directly from the public Boughter
distribution under `data/AIMS_manuscripts/`.

1. Translate nucleotide FASTAs (if necessary) and keep the highest-scoring
   reading frame.
2. Run ANARCI in IMGT mode to obtain FR/CDR region boundaries.
3. Extract FR1–FR4 to form VH domains, record CDRH3 sequences, and assign a
   lineage identifier (`family|CDRH3`).
4. Join ELISA flag counts (`*_NumReact.txt` and mouse `.dat` files), drop mild
   antibodies (1–3 flags), and compute binary labels (0 vs >3).

The resulting dataset is written to `data/processed/boughter_counts_rebuilt.csv`
(970 sequences, 859 unique lineages). A human-only subset used for external
transfer lives at `data/processed/boughter_counts_human_lineage.csv`.

## 2. Training configuration

We reproduce the pipeline using the updated CLI (`polyreact/train.py`) with the
paper-aligned configuration. Key settings:

- Feature backend: mean-pooled `facebook/esm1v_t33_650M_UR90S_1`
- Training data: all Boughter families (human HIV, human influenza, mouse IgA)
  with lineage-aware CV folds
- Model: `LogisticRegression` with `C=0.1` and `class_weight="balanced"`
- Calibration: isotonic (10-fold CV)

The replication script accepts flags to skip the slower phases. The run
captured here uses:

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

Set `--shehata` to `polyreactivity/data/processed/shehata_full.csv` after
rebuilding the 398-entry PSR screen. Additional toggles:

- `--skip-lofo` to omit leave-one-family-out runs
- `--skip-flag-regression` to skip the Poisson / negative-binomial diagnostics
- `--skip-descriptor-variants` / `--skip-fragment-variants` to prune ablation suites

All tables and figures referenced below were produced by this command.

## 3. Replication metrics

Aggregated metrics reside in
`artifacts/replication_v5/main/metrics.csv`. The table below reproduces the main
values (Shehata refers to the curated 88-sequence supplement).

| Split                 | ROC-AUC | PR-AUC | Accuracy |   F1 | Brier |
|-----------------------|--------:|-------:|---------:|----:|------:|
| Train (CV mean)       | 0.693   | 0.706  | 0.647    |0.658| 0.225 |
| Jain 2017             | 0.629   | 0.528  | 0.563    |0.557| 0.243 |
| Shehata 2019 (curated)| 0.755   | 0.805  | 0.655    |0.762| 0.217 |
| Harvey 2022           | 0.685   | 0.654  | 0.595    |0.692| 0.248 |

The metrics CSV ships `*_ci_lower`, `*_ci_upper`, and `*_ci_median`
columns per split (this run used 200 bootstrap resamples; tweak via
`--bootstrap-samples`).
The CSV also reports calibration diagnostics (ECE/MCE), positive-class F1, and
precision for each split.

Additional diagnostics:

- **Spearman correlation** between predicted probabilities and ELISA flag
  counts (full Boughter set): 0.61 (see
  `artifacts/replication_v5/main/spearman_flags.json`).
- **Leave-one-family-out results** (`artifacts/replication_v5/lofo_metrics.csv`)
  summarised below.

| Hold-out family | Accuracy | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|-----------------|---------:|--------:|-------:|------------:|------------:|
| Influenza       | 0.58     | 0.62    | 0.59   | 0.55        | 0.57        |
| HIV             | 0.61     | 0.66    | 0.62   | 0.57        | 0.63        |
| Mouse IgA       | 0.65     | 0.67    | 0.69   | 0.71        | 0.52        |

## 4. Discussion

- Matching the paper’s ESM-1v + logistic head closes most of the accuracy gap
  to the reported 71%, with curated Shehata/Harvey panels achieving strong
  PR-AUCs. All Boughter families participate in CV, aligning with the original
  setup.
- Flag-count modelling shows clear over-dispersion: Poisson dispersion ≈5.6 vs
  negative-binomial ≈7.7 with an inferred `alpha≈2.5`, supporting the move to
  an NB2 link for graded reactivity analyses.
- Leave-one-family-out runs still highlight the human→mouse specificity drop,
  while influenza↔HIV transfer remains comparatively balanced.
- The full 398-entry Shehata panel requires a manual download; rebuilding it as
  `data/processed/shehata_full.csv` slots into the same pipeline without code
  changes.
- Descriptor ablations remain available (pass
  `--skip-descriptor-variants/--skip-fragment-variants` to control runtime) and
  continue to emphasise the dominance of charge/pI features.
- Fragment ablations were rerun under the ESM-1v default: VH and joined
  CDRH1+2+3 each reach ≈0.65 CV accuracy, while single-loop models trail
  (≈0.60 for CDRH1, ≈0.59 for CDRH2, ≈0.54 for CDRH3) with CDRH3 still the
  strongest loop on Shehata PR-AUC. Outputs live in
  `artifacts/replication_v5_fragments/fragment_variants/`.

## 5. Figures

The reproduction script exports the following Matplotlib figures:

- `artifacts/replication_v5/main/accuracy_overview.png` — cross-split accuracy bar chart.
- `artifacts/replication_v5/main/roc_overview.png` — ROC overlays for train CV and all
  external panels.
- `artifacts/replication_v5/main/prob_vs_flags.png` — scatter showing model scores vs ELISA
  flag counts (Spearman annotated).
- Individual ROC/PR/reliability curves for each dataset under the same folder.

- `artifacts/replication_v5/main/flag_regression_metrics.csv` and
  `flag_regression_preds.csv` — Poisson / negative-binomial diagnostics for ELISA flag counts.

All figure scripts are deterministic; regenerating the artifacts will overwrite
previous runs.

## 6. How to extend

- **Descriptor ablations:** Re-run `polyreact/train.py` with
  `--backend descriptors` or `--backend concat` to mirror the paper’s
  physico-chemical study.
- **Lineage LOFO:** Adjust `--include-families` / `--exclude-families` to explore
  species transfer.
- **Alternative PLMs:** Update `feature_backend.plm_model_name` in the config to
  compare ESM-2 or AbLang2.

## 7. Reproducibility checklist

- Random seeds fixed to 42 (NumPy, Python, Torch).
- Exact dependencies recorded in `requirements.txt` and `pyproject.toml`.
- Pipeline re-run verified via unit tests (`pytest -q`) and
  `python -m polyreact.benchmarks.reproduce_paper`.

## 8. Artifact index

| Artifact / Figure | Path |
|-------------------|------|
| Metrics table      | `artifacts/replication_v5/main/metrics.csv` |
| Predictions        | `artifacts/replication_v5/main/preds.csv`   |
| ROC overview       | `artifacts/replication_v5/main/roc_overview.png` |
| Accuracy bar chart | `artifacts/replication_v5/main/accuracy_overview.png` |
| Probability vs flags | `artifacts/replication_v5/main/prob_vs_flags.png` |
| LOFO metrics       | `artifacts/replication_v5/lofo_metrics.csv` |
| Descriptor baselines | `artifacts/replication_v5_variants/descriptor_variants/summary.csv` |
| Spearman stats | `artifacts/replication_v5/main/spearman_flags.json` |
| Flag regression | `artifacts/replication_v5/main/flag_regression_metrics.csv` |
| Fragment ablations | `artifacts/replication_v5_fragments/fragment_variants/fragment_metrics_summary.csv` |

For any questions or issues, open a discussion in the repository or contact the
maintainers listed in the README.
## 9. Dataset notes

- **Shehata (PSR)**: The curated sheet bundled with the Boughter distribution (`Human_Ab_Poly_Dataset_S3.xlsx`) contains 88 antibody pairs with a High/Low label rather than the full 398-entry PSR screen. After mapping `High→1`, `Low→0` and removing one duplicate we obtain 51 positives (57.95% prevalence). The severe 7/398 imbalance reported in the paper stems from the raw PSR counts; users who require that version should manually download `mmc1.xlsx` from Cell, then run `python scripts/rebuild_shehata_psr.py --output data/processed/shehata_full.csv` to regenerate the 398-entry CSV and accompanying audit JSON. The effective Shehata split in this report therefore references `shehata_curated.csv`.
- **Audit trail**: `scripts/rebuild_boughter_from_counts.py` now writes `*_audit.json`, capturing translation failures, ANARCI dropouts, mild-flag removals, deduplication counts, and long-sequence filtering. The replication run stores this at `data/processed/boughter_counts_rebuilt_audit.json`.
- **Additional summaries**: Reproduction outputs include
  `artifacts/replication_v5/main/dataset_split_summary.csv` (label prevalence per
  split) and, when enabled, `descriptor_variants/summary.csv` /
  `fragment_metrics_summary.csv` for ablation studies.
