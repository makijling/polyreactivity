# Polyreactivity Space

Interactive Gradio interface for scoring antibody sequences with the trained
polyreactivity model.

## Usage

1. Train a model (see project README) and ensure the resulting artifact is
   accessible at `artifacts/model.joblib`, or upload the file through the UI.
2. Launch locally:
   ```bash
   python space/app.py
   ```
3. Provide a heavy-chain sequence (optional light chain) and click **Predict**,
   or upload a CSV with columns `id, heavy_seq[, light_seq]` for batch scoring.

### Benchmark mode

- Include a binary `label` column to obtain accuracy, F1, ROC-AUC, PR-AUC, and
  Brier score against your ground truth.
- Include a `reactivity_count` column to compute Spearman correlation between
  predicted probabilities and graded ELISA flag counts.
- The app writes merged inputs + predictions to `polyreact_predictions.csv`
  for downstream analysis.

### Environment Variables

- `POLYREACT_MODEL_PATH` — default path to the trained model artifact.
- `POLYREACT_CONFIG_PATH` — default YAML configuration for inference overrides.

Both variables are optional; when unset, the app looks for
`artifacts/model.joblib` and `configs/default.yaml` relative to the project root.

## Deploying to Hugging Face Spaces

Automate deployment with the helper script once you have set
`HF_TOKEN` (or another environment variable of your choice) with a
Hugging Face write token:

```bash
export HF_TOKEN=hf_your_write_token
python space/deploy.py --space-id your-username/polyreactivity-space
```

Add `--private` if you prefer a private Space or use `--token-env` when the
token lives under a different variable name. The script uploads the package,
configuration, and Space assets — including the default
`artifacts/model.joblib` — so the interface is ready immediately after the
build completes.
