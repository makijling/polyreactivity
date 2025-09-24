"""Gradio Space for polyreactivity prediction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

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
    preds = predict_batch(
        [record],
        weights=model_path,
        heavy_only=not use_paired,
        backend=backend or None,
        config=DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None,
    )
    score = float(preds.iloc[0]["score"])
    pred = int(preds.iloc[0]["pred"])
    label = "Polyreactive" if pred == 1 else "Non-polyreactive"
    return label, score, pred


def _predict_batch(
    input_file: gr.File,
    use_paired: bool,
    backend: str,
    model_file: Optional[gr.File],
) -> gr.File:
    if input_file is None:
        raise gr.Error("Upload a CSV file with columns id, heavy_seq[, light_seq].")
    model_path = _resolve_model_path(model_file)
    input_path = Path(input_file.name)
    frame = pd.read_csv(input_path)
    required_cols = {"id", "heavy_seq"}
    if not required_cols.issubset(frame.columns):
        raise gr.Error("CSV must include at least 'id' and 'heavy_seq' columns.")

    records = frame.to_dict("records")
    preds = predict_batch(
        records,
        weights=model_path,
        heavy_only=not use_paired,
        backend=backend or None,
        config=DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None,
    )
    output_path = input_path.parent / "polyreact_predictions.csv"
    preds.to_csv(output_path, index=False)
    return gr.File.update(value=str(output_path), visible=True)


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

            batch_button.click(
                _predict_batch,
                inputs=[batch_file, batch_paired, batch_backend, batch_model],
                outputs=[batch_output],
            )

        gr.Markdown(
            """
            **Notes**
            - Default configuration expects heavy-chain only evaluation.
            - Backend overrides should match how the model was trained to avoid
              feature mismatch.
            - CSV inputs should include `id`, `heavy_seq`, and optionally `light_seq`.
            """
        )

    return demo


def main() -> None:
    demo = make_interface()
    demo.launch()


if __name__ == "__main__":
    main()
