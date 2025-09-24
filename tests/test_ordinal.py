from __future__ import annotations

import numpy as np

from polyreact.models.ordinal import (
    fit_negative_binomial_model,
    fit_poisson_model,
    pearson_dispersion,
    regression_metrics,
)


def test_fit_poisson_model_learns_simple_relationship() -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(200, 2))
    true_coef = np.array([0.4, -0.2])
    rates = np.exp(X @ true_coef + 0.1)
    y = rng.poisson(rates)

    model = fit_poisson_model(X, y)
    preds = model.predict(X)
    metrics = regression_metrics(y, preds)

    assert metrics["rmse"] < 1.3
    assert -0.5 < metrics["r2"] < 1.0


def test_fit_negative_binomial_model_handles_overdispersion() -> None:
    rng = np.random.default_rng(321)
    X = rng.normal(size=(300, 3))
    # Generate over-dispersed counts via a gamma-poisson mixture
    rates = np.exp(0.3 * X[:, 0] - 0.5 * X[:, 1] + 0.2)
    gamma_noise = rng.gamma(shape=2.5, scale=1.0, size=rates.shape)
    y = rng.poisson(rates * gamma_noise)

    model = fit_negative_binomial_model(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.all(np.isfinite(preds))
    assert model.alpha > 0.0


def test_pearson_dispersion_matches_variance() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = y_true.copy()
    dispersion = pearson_dispersion(y_true, y_pred, dof=1)
    assert dispersion == 0.0
