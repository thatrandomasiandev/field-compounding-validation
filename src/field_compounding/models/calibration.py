"""Calibration for telemetry-derived violation severity estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def clip_violation(values: np.ndarray) -> np.ndarray:
    """Clip values to the violation severity unit interval."""
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = clip_violation(y_true)
    y_pred = clip_violation(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = clip_violation(y_true)
    y_pred = clip_violation(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def violation_calibration_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error for regression-style violation predictions."""
    y_true = clip_violation(y_true)
    y_pred = clip_violation(y_pred)
    if len(y_true) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for idx in range(n_bins):
        low = bin_boundaries[idx]
        high = bin_boundaries[idx + 1]
        if idx == n_bins - 1:
            mask = (y_pred >= low) & (y_pred <= high)
        else:
            mask = (y_pred >= low) & (y_pred < high)
        if not np.any(mask):
            continue
        mean_pred = float(np.mean(y_pred[mask]))
        mean_true = float(np.mean(y_true[mask]))
        ece_val += mask.sum() / len(y_true) * abs(mean_pred - mean_true)
    return float(ece_val)


@dataclass(frozen=True)
class ViolationCalibrationReport:
    """Before/after calibration metrics for violation severity estimates."""

    mae_before: float
    mae_after: float
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    n_samples: int

    @property
    def mae_improvement(self) -> float:
        return self.mae_before - self.mae_after

    @property
    def ece_improvement(self) -> float:
        return self.ece_before - self.ece_after


class IsotonicViolationCalibrator:
    """Monotone post-hoc map from raw proxy scores to calibrated v_k."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> IsotonicViolationCalibrator:
        scores_arr = clip_violation(scores).reshape(-1)
        y_arr = clip_violation(y_true).reshape(-1)
        if scores_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("scores and y_true must have the same length")
        if scores_arr.shape[0] < 2:
            raise ValueError("need at least 2 samples to fit isotonic calibrator")
        self._iso.fit(scores_arr, y_arr)
        self._is_fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("call fit before predict")
        scores_arr = clip_violation(scores).reshape(-1)
        return clip_violation(self._iso.predict(scores_arr))


class PlattViolationCalibrator:
    """Platt scaling for scalar violation proxy scores via logistic link."""

    def __init__(self) -> None:
        self._model = LogisticRegression(max_iter=500)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> PlattViolationCalibrator:
        scores_arr = clip_violation(scores).reshape(-1, 1)
        y_arr = clip_violation(y_true).reshape(-1)
        if scores_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("scores and y_true must have the same length")
        if scores_arr.shape[0] < 2:
            raise ValueError("need at least 2 samples to fit Platt calibrator")
        labels = (y_arr >= 0.5).astype(int)
        if len(np.unique(labels)) < 2:
            labels = (y_arr >= np.median(y_arr)).astype(int)
        self._model.fit(scores_arr, labels)
        self._is_fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("call fit before predict")
        scores_arr = clip_violation(scores).reshape(-1, 1)
        probs = self._model.predict_proba(scores_arr)[:, 1]
        return clip_violation(probs)


def evaluate_violation_calibration(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
) -> ViolationCalibrationReport:
    """Compare raw and calibrated violation predictions."""
    y_arr = clip_violation(y_true)
    raw_arr = clip_violation(raw_scores)
    cal_arr = clip_violation(calibrated_scores)
    return ViolationCalibrationReport(
        mae_before=mean_absolute_error(y_arr, raw_arr),
        mae_after=mean_absolute_error(y_arr, cal_arr),
        brier_before=brier_score(y_arr, raw_arr),
        brier_after=brier_score(y_arr, cal_arr),
        ece_before=violation_calibration_ece(y_arr, raw_arr),
        ece_after=violation_calibration_ece(y_arr, cal_arr),
        n_samples=int(y_arr.shape[0]),
    )
