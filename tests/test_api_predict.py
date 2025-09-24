from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from polyreact.api import predict_batch
from polyreact.config import Config
from polyreact.features.pipeline import build_feature_pipeline
from polyreact.models.linear import LinearModelConfig, train_linear_model


def test_predict_batch_with_saved_artifact(tmp_path: Path) -> None:
    config = Config()
    config.feature_backend.type = "descriptors"
    config.descriptors.use_anarci = False
    config.calibration.method = None
    config.device = "cpu"

    pipeline = build_feature_pipeline(config)

    train_df = pd.DataFrame(
        {
            "id": ["pos", "neg"],
            "heavy_seq": [
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISSYGSSTYYG",
                "DVVMTQSPASLSVTPGEKVTITCQASQDISNYLNWYQQKPGKAPKLLIYDT",
            ],
            "label": [1, 0],
        }
    )

    features = pipeline.fit_transform(train_df, heavy_only=True, batch_size=1)
    model = train_linear_model(
        features,
        train_df["label"].to_numpy(dtype=int),
        config=LinearModelConfig(head="logreg", class_weight=None, C=1.0),
        random_state=0,
    )

    artifact = {
        "config": config,
        "feature_state": pipeline.get_state(),
        "model": model,
    }
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(artifact, artifact_path)

    records = [
        {
            "id": "sample",
            "heavy_seq": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
        }
    ]

    preds = predict_batch(
        records,
        weights=artifact_path,
        heavy_only=True,
        batch_size=1,
    )

    assert list(preds.columns) == ["id", "score", "pred"]
    assert preds.shape[0] == 1
    assert preds.at[0, "id"] == "sample"
    assert np.issubdtype(preds["score"].dtype, np.floating)
