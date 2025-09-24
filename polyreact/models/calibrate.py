"""Probability calibration helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def fit_calibrator(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "isotonic",
    cv: int | str | None = "prefit",
) -> CalibratedClassifierCV:
    """Fit a ``CalibratedClassifierCV`` on top of a pre-trained estimator."""

    calibrator = CalibratedClassifierCV(estimator, method=method, cv=cv)
    calibrator.fit(X, y)
    return calibrator

