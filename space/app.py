"""Gradio Space for polyreactivity prediction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from polyreact.api import predict_batch

DEFAULT_MODEL_PATH = Path(os.environ.get("POLYREACT_MODEL_PATH", "artifacts/model.joblib")).resolve()
DEFAULT_CONFIG_PATH = Path(os.environ.get("POLYREACT_CONFIG_PATH", "configs/default.yaml")).resolve()


def _resolve_model_path(upload: Optional[gr.File]) -> Path:
    if upload is not None:
        return Path(upload.name)
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    raise FileNotFoundError(
        "Model artifact not found. Upload a trained model (.joblib) to run predictions."
    )


def _predict_single(
    heavy_seq: str,
    light_seq: str,
    use_paired: bool,
    backend: str,
    model_file: Optional[gr.File],
) -> tuple[str, float, int]:
    model_path = _resolve_model_path(model_file)
    heavy_seq = (heavy_seq or "").strip()
    light_seq = (light_seq or "").strip()
    if not heavy_seq:
        raise gr.Error("Please provide a heavy-chain amino acid sequence.")

    record = {
        "id": "sample",
        "heavy_seq": heavy_seq,
        "light_seq": light_seq,
    }
    progress = gr.Progress(track_tqdm=True)
    progress(0.05, "Loading model and embeddings (first run may download ESM-1v, please wait)…")
    preds = predict_batch(
        [record],
        weights=model_path,
        heavy_only=not use_paired,
        backend=backend or None,
        config=DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None,
    )
    progress(1.0, "Prediction complete")
    score = float(preds.iloc[0]["score"])
    pred = int(preds.iloc[0]["pred"])
    label = "Polyreactive" if pred == 1 else "Non-polyreactive"
    return label, score, pred


def _format_metric(value: float) -> float:
    return float(f"{value:.4f}")


def _compute_metrics(results: pd.DataFrame) -> tuple[pd.DataFrame, list[str], Optional[str]]:
    metrics_rows: list[dict[str, float]] = []
    warnings: list[str] = []
    spearman_text: Optional[str] = None

    if "label" in results.columns:
        label_series = results["label"].dropna()
        valid_labels = label_series.isin({0, 1}).all()
        if valid_labels and label_series.nunique() > 1:
            y_true = results.loc[label_series.index, "label"].astype(int)
            y_score = results.loc[label_series.index, "score"].astype(float)
            y_pred = results.loc[label_series.index, "pred"].astype(int)

            metrics_rows.append({"metric": "Accuracy", "value": _format_metric(accuracy_score(y_true, y_pred))})
            metrics_rows.append({"metric": "F1", "value": _format_metric(f1_score(y_true, y_pred))})
            try:
                roc = roc_auc_score(y_true, y_score)
                metrics_rows.append({"metric": "ROC-AUC", "value": _format_metric(roc)})
            except ValueError:
                warnings.append("ROC-AUC skipped (requires both positive and negative labels).")
            try:
                pr_auc = average_precision_score(y_true, y_score)
                metrics_rows.append({"metric": "PR-AUC", "value": _format_metric(pr_auc)})
            except ValueError:
                warnings.append("PR-AUC skipped (requires both positive and negative labels).")
            try:
                brier = brier_score_loss(y_true, y_score)
                metrics_rows.append({"metric": "Brier", "value": _format_metric(brier)})
            except ValueError:
                warnings.append("Brier score skipped (invalid probability values).")
        else:
            warnings.append("Label column found but must contain binary 0/1 values with both classes present.")

    if "reactivity_count" in results.columns:
        valid = results[["reactivity_count", "score"]].dropna()
        if len(valid) > 2 and valid["reactivity_count"].nunique() > 1:
            stat, pval = spearmanr(valid["reactivity_count"], valid["score"])
            if stat == stat:  # NaN check
                spearman_text = f"Spearman ρ = {stat:.4f} (p = {pval:.3g})"
        else:
            warnings.append("Flag-count Spearman skipped (need ≥3 non-identical counts).")

    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df, warnings, spearman_text


def _predict_batch(
    input_file: gr.File,
    use_paired: bool,
    backend: str,
    model_file: Optional[gr.File],
) -> tuple[gr.File, gr.DataFrame, gr.Textbox, gr.Markdown]:
    if input_file is None:
        raise gr.Error("Upload a CSV file with columns id, heavy_seq[, light_seq].")
    model_path = _resolve_model_path(model_file)
    input_path = Path(input_file.name)
    frame = pd.read_csv(input_path)
    required_cols = {"id", "heavy_seq"}
    if not required_cols.issubset(frame.columns):
        raise gr.Error("CSV must include at least 'id' and 'heavy_seq' columns.")

    records = frame.to_dict("records")
    progress = gr.Progress(track_tqdm=True)
    progress(0.05, "Loading model and embeddings (first run may download ESM-1v, please wait)…")
    preds = predict_batch(
        records,
        weights=model_path,
        heavy_only=not use_paired,
        backend=backend or None,
        config=DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None,
    )
    progress(1.0, "Batch prediction complete")
    merged = frame.merge(preds, on="id", how="left")
    output_path = input_path.parent / "polyreact_predictions.csv"
    merged.to_csv(output_path, index=False)

    metrics_df, warnings, spearman_text = _compute_metrics(merged)
    metrics_update = gr.update(value=metrics_df, visible=not metrics_df.empty)
    spearman_update = gr.update(value=spearman_text or "", visible=spearman_text is not None)
    notes_update = gr.update(
        value="\n".join(f"- {msg}" for msg in warnings) if warnings else "",
        visible=bool(warnings),
    )

    return (
        gr.update(value=str(output_path), visible=True),
        metrics_update,
        spearman_update,
        notes_update,
    )


def make_interface() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Polyreactivity Predictor

            Provide an antibody heavy chain (and optional light chain) to estimate
            polyreactivity probability. Upload a trained model artifact or place it
            at `artifacts/model.joblib`.
            """
        )

        with gr.Tab("Single Sequence"):
            with gr.Row():
                heavy_input = gr.Textbox(
                    label="Heavy chain sequence",
                    lines=6,
                    placeholder="Enter amino acid sequence",
                )
                light_input = gr.Textbox(
                    label="Light chain sequence (optional)",
                    lines=6,
                    placeholder="Enter amino acid sequence",
                )
            with gr.Row():
                use_paired = gr.Checkbox(label="Use paired evaluation", value=False)
                backend_input = gr.Dropdown(
                    label="Feature backend override",
                    choices=["", "descriptors", "plm", "concat"],
                    value="",
                )
                model_upload = gr.File(label="Model artifact (.joblib)", file_types=[".joblib"], file_count="single")

            run_button = gr.Button("Predict", variant="primary")
            result_label = gr.Textbox(label="Prediction", interactive=False)
            result_score = gr.Number(label="Probability", precision=4)
            result_class = gr.Number(label="Binary call (1=polyreactive)")

            run_button.click(
                _predict_single,
                inputs=[heavy_input, light_input, use_paired, backend_input, model_upload],
                outputs=[result_label, result_score, result_class],
            )

        with gr.Tab("Batch CSV"):
            batch_file = gr.File(label="Upload CSV", file_types=[".csv"], file_count="single")
            batch_paired = gr.Checkbox(label="Use paired evaluation", value=False)
            batch_backend = gr.Dropdown(
                label="Feature backend override",
                choices=["", "descriptors", "plm", "concat"],
                value="",
            )
            batch_model = gr.File(label="Model artifact (.joblib)", file_types=[".joblib"], file_count="single")
            batch_button = gr.Button("Run batch predictions", variant="primary")
            batch_output = gr.File(label="Download predictions", visible=False)
            batch_metrics = gr.Dataframe(label="Benchmark metrics", visible=False)
            batch_spearman = gr.Textbox(label="Flag-count Spearman", interactive=False, visible=False)
            batch_notes = gr.Markdown(visible=False)

            batch_button.click(
                _predict_batch,
                inputs=[batch_file, batch_paired, batch_backend, batch_model],
                outputs=[batch_output, batch_metrics, batch_spearman, batch_notes],
            )

        gr.Markdown(
            """
            **Notes**
            - Default configuration expects heavy-chain only evaluation.
            - Backend overrides should match how the model was trained to avoid
              feature mismatch.
            - CSV inputs should include `id`, `heavy_seq`, and optionally `light_seq`.
            - Add a binary `label` column to compute accuracy/F1/ROC-AUC/PR-AUC/Brier.
            - Include `reactivity_count` to report Spearman correlation with predicted probabilities.
            - Initial runs may spend a few minutes downloading the 650M-parameter ESM-1v model before predictions start.
            """
        )

    return demo


def main() -> None:
    demo = make_interface()
    demo.launch()


if __name__ == "__main__":
    main()
