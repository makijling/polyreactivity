"""Public Python API for polyreactivity prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import copy

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config, load_config
from .features.pipeline import FeaturePipeline, FeaturePipelineState, build_feature_pipeline


def predict_batch(  # noqa: ANN003
    records: Iterable[dict],
    *,
    config: Config | str | Path | None = None,
    backend: str | None = None,
    plm_model: str | None = None,
    weights: str | Path | None = None,
    heavy_only: bool = True,
    batch_size: int = 8,
    device: str | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Predict polyreactivity scores for a batch of sequences."""

    records_list = list(records)
    if not records_list:
        return pd.DataFrame(columns=["id", "score", "pred"])

    artifact = _load_artifact(weights)

    if config is None:
        artifact_config = artifact.get("config")
        if isinstance(artifact_config, Config):
            config = copy.deepcopy(artifact_config)
        else:
            config = load_config("configs/default.yaml")
    elif isinstance(config, (str, Path)):
        config = load_config(config)
    else:
        config = copy.deepcopy(config)

    if backend:
        config.feature_backend.type = backend
    if plm_model:
        config.feature_backend.plm_model_name = plm_model
    if device:
        config.device = device
    if cache_dir:
        config.feature_backend.cache_dir = cache_dir

    pipeline = _restore_pipeline(config, artifact)
    trained_model = artifact["model"]

    frame = pd.DataFrame(records_list)
    if frame.empty:
        raise ValueError("Prediction requires at least one record.")
    if "id" not in frame.columns:
        frame["id"] = frame.get("sequence_id", range(len(frame))).astype(str)
    if "heavy_seq" in frame.columns:
        frame["heavy_seq"] = frame["heavy_seq"].fillna("").astype(str)
    else:
        heavy_series = frame.get("heavy")
        if heavy_series is None:
            heavy_series = pd.Series([""] * len(frame))
        frame["heavy_seq"] = heavy_series.fillna("").astype(str)

    if "light_seq" in frame.columns:
        frame["light_seq"] = frame["light_seq"].fillna("").astype(str)
    else:
        light_series = frame.get("light")
        if light_series is None:
            light_series = pd.Series([""] * len(frame))
        frame["light_seq"] = light_series.fillna("").astype(str)

    if heavy_only:
        frame["light_seq"] = ""
    if frame["heavy_seq"].str.len().eq(0).all():
        raise ValueError("No heavy chain sequences provided for prediction.")

    features = pipeline.transform(frame, heavy_only=heavy_only, batch_size=batch_size)
    scores = trained_model.predict_proba(features)
    preds = (scores >= 0.5).astype(int)

    return pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "score": scores,
            "pred": preds,
        }
    )


def _load_artifact(weights: str | Path | None) -> dict:
    if weights is None:
        msg = "Prediction requires a path to model weights"
        raise ValueError(msg)
    artifact = joblib.load(weights)
    if not isinstance(artifact, dict):
        msg = "Model artifact must be a dictionary"
        raise ValueError(msg)
    return artifact


def _restore_pipeline(config: Config, artifact: dict) -> FeaturePipeline:
    pipeline = build_feature_pipeline(config)
    state = artifact.get("feature_state")
    if isinstance(state, FeaturePipelineState):
        pipeline.load_state(state)
        if pipeline.backend.type in {"plm", "concat"} and pipeline._plm_scaler is None:
            pipeline._plm_scaler = StandardScaler()
        return pipeline

    msg = "Model artifact is missing feature pipeline state"
    raise ValueError(msg)
