"""Ordinal/count modeling utilities for flag regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomialResults
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - fallback for older sklearn
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


@dataclass(slots=True)
class PoissonModel:
    """Wrapper storing a fitted Poisson regression model."""

    estimator: PoissonRegressor

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


def fit_poisson_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1e-6,
    max_iter: int = 1000,
) -> PoissonModel:
    """Train a Poisson regression model on count targets."""

    model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    model.fit(X, y)
    return PoissonModel(estimator=model)


@dataclass(slots=True)
class NegativeBinomialModel:
    """Wrapper storing a fitted negative-binomial regression model."""

    result: NegativeBinomialResults

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_const = sm.add_constant(X, has_constant="add")
        return self.result.predict(X_const)

    @property
    def alpha(self) -> float:
        params = np.asarray(self.result.params, dtype=float)
        exog_dim = self.result.model.exog.shape[1]
        if params.size > exog_dim:
            # statsmodels stores log(alpha) as the final coefficient
            return float(np.exp(params[-1]))
        model_alpha = getattr(self.result.model, "alpha", None)
        if model_alpha is not None:
            return float(model_alpha)
        return float("nan")


def fit_negative_binomial_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 200,
) -> NegativeBinomialModel:
    """Train a negative binomial regression model (NB2)."""

    X_const = sm.add_constant(X, has_constant="add")
    model = sm.NegativeBinomial(y, X_const, loglike_method="nb2")
    result = model.fit(maxiter=max_iter, disp=False)
    return NegativeBinomialModel(result=result)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics for count predictions."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def pearson_dispersion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dof: int,
) -> float:
    """Compute Pearson dispersion (chi-square / dof)."""

    eps = 1e-8
    adjusted = np.maximum(y_pred, eps)
    resid = (y_true - y_pred) / np.sqrt(adjusted)
    denom = max(dof, 1)
    return float(np.sum(resid**2) / denom)
