"""Metrics utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: Iterable[float],
    y_score: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics from scores and labels."""

    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)

    metrics: dict[str, float] = {}

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_score_arr))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred))
    metrics["f1"] = float(f1_score(y_true_arr, y_pred, zero_division=0))
    metrics["f1_positive"] = float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    metrics["f1_negative"] = float(f1_score(y_true_arr, y_pred, pos_label=0, zero_division=0))
    metrics["sensitivity"] = float(recall_score(y_true_arr, y_pred, zero_division=0))
    # Specificity is recall on the negative class
    metrics["specificity"] = float(
        recall_score(1 - y_true_arr, 1 - y_pred, zero_division=0)
    )
    metrics["precision"] = float(precision_score(y_true_arr, y_pred, zero_division=0))
    metrics["positive_rate"] = float(y_true_arr.mean()) if y_true_arr.size else float("nan")
    metrics["brier"] = float(brier_score_loss(y_true_arr, y_score_arr))
    ece, mce = _calibration_errors(y_true_arr, y_score_arr)
    metrics["ece"] = float(ece)
    metrics["mce"] = float(mce)
    return metrics


def _calibration_errors(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float]:
    if y_true.size == 0:
        return float("nan"), float("nan")

    # Clamp scores to [0, 1] to avoid binning issues when calibrators overshoot
    scores = np.clip(y_score, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(scores, bins[1:-1], right=True)

    total = y_true.size
    ece = 0.0
    mce = 0.0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if not np.any(mask):
            continue
        bin_scores = scores[mask]
        bin_true = y_true[mask]
        confidence = float(bin_scores.mean())
        accuracy = float(bin_true.mean())
        gap = abs(confidence - accuracy)
        weight = float(mask.sum()) / float(total)
        ece += weight * gap
        mce = max(mce, gap)

    return ece, mce


def bootstrap_metric_intervals(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    threshold: float = 0.5,
    random_state: int | None = 42,
) -> dict[str, dict[str, float]]:
    """Estimate bootstrap confidence intervals for core metrics.

    Parameters
    ----------
    y_true, y_score:
        Arrays of ground-truth labels and probability scores.
    n_bootstrap:
        Number of bootstrap resamples; set to ``0`` to disable.
    alpha:
        Two-sided confidence level (default ``0.05`` gives 95% CI).
    threshold:
        Decision threshold passed to :func:`compute_metrics`.
    random_state:
        Seed controlling the bootstrap sampler.
    """

    if n_bootstrap <= 0:
        return {}

    y_true_arr = np.asarray(y_true, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)
    n = y_true_arr.size
    if n == 0:
        return {}

    rng = np.random.default_rng(random_state)
    collected: dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        resampled_true = y_true_arr[indices]
        resampled_score = y_score_arr[indices]
        if np.unique(resampled_true).size < 2:
            continue
        metrics = compute_metrics(resampled_true, resampled_score, threshold=threshold)
        for metric_name, value in metrics.items():
            collected.setdefault(metric_name, []).append(value)

    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    summary: dict[str, dict[str, float]] = {}
    for metric_name, values in collected.items():
        arr = np.asarray(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            continue
        lower = float(np.nanquantile(valid, lower_q))
        upper = float(np.nanquantile(valid, upper_q))
        median = float(np.nanmedian(valid))
        summary[metric_name] = {
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_median": median,
        }

    return summary
