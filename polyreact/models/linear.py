"""Linear classification heads for polyreactivity prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


@dataclass(slots=True)
class LinearModelConfig:
    """Configuration options for linear heads."""

    head: str = "logreg"
    C: float = 1.0
    class_weight: Any = "balanced"
    max_iter: int = 1000


@dataclass(slots=True)
class TrainedModel:
    """Container for trained estimators and optional calibration."""

    estimator: Any
    calibrator: Any | None = None
    vectorizer_name: str = ""
    feature_meta: dict[str, Any] = field(default_factory=dict)
    metrics_cv: dict[str, float] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.calibrator is not None and hasattr(self.calibrator, "predict"):
            return self.calibrator.predict(X)
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrator is not None and hasattr(self.calibrator, "predict_proba"):
            probs = self.calibrator.predict_proba(X)
            return probs[:, 1]
        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X)
            return probs[:, 1]
        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X)
            return 1.0 / (1.0 + np.exp(-scores))
        msg = "Estimator does not support probability prediction"
        raise AttributeError(msg)


def build_estimator(
    *, config: LinearModelConfig, random_state: int | None = 42
) -> Any:
    """Construct an unfitted linear estimator based on configuration."""

    if config.head == "logreg":
        estimator = LogisticRegression(
            C=config.C,
            max_iter=config.max_iter,
            class_weight=config.class_weight,
            solver="liblinear",
            random_state=random_state,
        )
    elif config.head == "linear_svm":
        estimator = LinearSVC(
            C=config.C,
            class_weight=config.class_weight,
            max_iter=config.max_iter,
            random_state=random_state,
        )
    else:  # pragma: no cover - defensive branch
        msg = f"Unsupported head type: {config.head}"
        raise ValueError(msg)
    return estimator


def train_linear_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: LinearModelConfig,
    random_state: int | None = 42,
) -> TrainedModel:
    """Fit a linear classifier on the provided feature matrix."""

    estimator = build_estimator(config=config, random_state=random_state)
    if isinstance(estimator, LogisticRegression) and X.shape[0] >= 1000:
        estimator.set_params(solver="lbfgs")
    estimator.fit(X, y)
    return TrainedModel(estimator=estimator)
