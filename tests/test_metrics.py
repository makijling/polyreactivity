from __future__ import annotations

import numpy as np
from sklearn import metrics as sk_metrics

from polyreact.utils.metrics import bootstrap_metric_intervals, compute_metrics


def test_compute_metrics_matches_sklearn() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    result = compute_metrics(y_true, y_score, threshold=0.5)

    assert result["roc_auc"] == sk_metrics.roc_auc_score(y_true, y_score)
    assert result["pr_auc"] == sk_metrics.average_precision_score(y_true, y_score)
    assert result["brier"] == sk_metrics.brier_score_loss(y_true, y_score)

    y_pred = (y_score >= 0.5).astype(int)
    assert result["accuracy"] == sk_metrics.accuracy_score(y_true, y_pred)
    assert result["f1"] == sk_metrics.f1_score(y_true, y_pred)
    assert result["f1_positive"] == sk_metrics.f1_score(y_true, y_pred, pos_label=1)
    assert result["f1_negative"] == sk_metrics.f1_score(y_true, y_pred, pos_label=0)
    assert result["sensitivity"] == sk_metrics.recall_score(y_true, y_pred)
    assert result["specificity"] == sk_metrics.recall_score(1 - y_true, 1 - y_pred)
    assert result["precision"] == sk_metrics.precision_score(y_true, y_pred)
    assert result["positive_rate"] == y_true.mean()
    assert 0.0 <= result["ece"] <= 1.0
    assert 0.0 <= result["mce"] <= 1.0


def test_bootstrap_metric_intervals_bounds() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.3, 0.2, 0.7, 0.8, 0.9])
    intervals = bootstrap_metric_intervals(
        y_true,
        y_score,
        n_bootstrap=100,
        random_state=123,
    )

    assert "roc_auc" in intervals
    roc_ci = intervals["roc_auc"]
    assert roc_ci["ci_lower"] <= roc_ci["ci_median"] <= roc_ci["ci_upper"]
    assert 0.0 <= roc_ci["ci_lower"] <= 1.0
