"""Shared evaluation metrics across all benchmark modules."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def psnr(y_true: np.ndarray, y_pred: np.ndarray, max_val: float = 1.0) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(max_val**2 / mse))


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece_val += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return float(ece_val)


def normalized_score(achieved: float, optimal: float, baseline: float = 0.0) -> float:
    """Normalize a score to [0, 1] relative to baseline and optimal."""
    if optimal == baseline:
        return 0.0
    raw = (achieved - baseline) / (optimal - baseline)
    return float(np.clip(raw, 0.0, 1.0))
