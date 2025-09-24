from __future__ import annotations

import pandas as pd

from polyreact.config import Config
from polyreact.features.pipeline import FeaturePipeline, build_feature_pipeline


def test_feature_pipeline_state_roundtrip_descriptors() -> None:
    config = Config()
    config.feature_backend.type = "descriptors"
    pipeline = build_feature_pipeline(config)
    seq_a = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISSYGSSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARRGYYYGMDV"
    seq_b = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARRGYYYGMDV"
    df = pd.DataFrame({"id": ["a", "b"], "heavy_seq": [seq_a, seq_b]})

    features_fit = pipeline.fit_transform(df, heavy_only=True)
    state = pipeline.get_state()

    new_pipeline = build_feature_pipeline(config)
    new_pipeline.load_state(state)
    features_transform = new_pipeline.transform(df, heavy_only=True)

    assert features_fit.shape == features_transform.shape
    assert (features_fit == features_transform).all()
