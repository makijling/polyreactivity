from __future__ import annotations

import numpy as np

from polyreact.models.linear import LinearModelConfig, train_linear_model


def test_logistic_regression_trains_on_toy_data() -> None:
    rng = np.random.default_rng(0)
    X_pos = rng.normal(loc=1.0, scale=0.1, size=(20, 3))
    X_neg = rng.normal(loc=-1.0, scale=0.1, size=(20, 3))
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(20), np.zeros(20)])

    config = LinearModelConfig(head="logreg", C=1.0, class_weight="balanced")
    trained = train_linear_model(X, y, config=config, random_state=42)
    preds = trained.estimator.predict(X)
    assert preds.mean() > 0.4
    assert preds[:20].all()
    assert not preds[20:].any()
