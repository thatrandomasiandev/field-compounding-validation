"""Telemetry-based violation severity estimators for partial observability."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.linear_model import Ridge

from field_compounding.models.calibration import IsotonicViolationCalibrator, clip_violation


def stack_telemetry_features(
    gnss_drift: np.ndarray,
    false_positive_rate: np.ndarray,
) -> np.ndarray:
    """Build (N, 3) telemetry features: log1p(gnss), fpr, interaction."""
    gnss = np.asarray(gnss_drift, dtype=np.float64).reshape(-1)
    fpr = np.asarray(false_positive_rate, dtype=np.float64).reshape(-1)
    if gnss.shape[0] != fpr.shape[0]:
        raise ValueError("gnss_drift and false_positive_rate must have the same length")
    return np.column_stack([np.log1p(np.maximum(gnss, 0.0)), fpr, np.log1p(np.maximum(gnss, 0.0)) * fpr])


def proxy_violation_severity(
    gnss_drift: np.ndarray,
    false_positive_rate: np.ndarray,
    *,
    gnss_scale_m: float = 8.0,
    fpr_weight: float = 0.55,
) -> np.ndarray:
    """Monotone hand-crafted map from GNSS drift and FPR to v_k in [0, 1]."""
    gnss = np.asarray(gnss_drift, dtype=np.float64)
    fpr = np.asarray(false_positive_rate, dtype=np.float64)
    gnss_term = 1.0 - np.exp(-np.maximum(gnss, 0.0) / gnss_scale_m)
    raw = fpr_weight * fpr + (1.0 - fpr_weight) * gnss_term
    return clip_violation(raw)


def extract_telemetry_from_traces(
    traces: Iterable[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract GNSS drift, FPR, and violation severity arrays from trace rows."""
    gnss_list: list[float] = []
    fpr_list: list[float] = []
    v_list: list[float] = []
    for row in traces:
        gnss_list.append(float(row["gnss_drift"]))
        fpr_list.append(float(row["false_positive_rate"]))
        v_list.append(float(row["violation_severity"]))
    return (
        np.asarray(gnss_list, dtype=np.float64),
        np.asarray(fpr_list, dtype=np.float64),
        clip_violation(np.asarray(v_list, dtype=np.float64)),
    )


class ViolationEstimator(ABC):
    """Interface for mapping observable telemetry to violation severity v_k."""

    @abstractmethod
    def fit(
        self,
        gnss_drift: np.ndarray,
        false_positive_rate: np.ndarray,
        violation_severity: np.ndarray,
    ) -> ViolationEstimator:
        ...

    @abstractmethod
    def predict(
        self,
        gnss_drift: np.ndarray,
        false_positive_rate: np.ndarray,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class TelemetryFitResult:
    """Training diagnostics for a fitted telemetry violation estimator."""

    train_mae: float
    train_rmse: float
    n_samples: int


class TelemetryViolationEstimator(ViolationEstimator):
    """Learned map from GNSS drift and false-positive rate to v_k."""

    def __init__(
        self,
        *,
        ridge_alpha: float = 10.0,
        calibrate: bool = True,
    ) -> None:
        self.ridge_alpha = ridge_alpha
        self.calibrate = calibrate
        self._regressor: Ridge | None = None
        self._calibrator: IsotonicViolationCalibrator | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._fit_result: TelemetryFitResult | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def fit_result(self) -> TelemetryFitResult | None:
        return self._fit_result

    def _normalize_features(self, features: np.ndarray, *, fit: bool) -> np.ndarray:
        if fit:
            self._feature_mean = features.mean(axis=0)
            self._feature_std = features.std(axis=0)
            self._feature_std[self._feature_std < 1e-8] = 1.0
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("feature normalizer is not initialized")
        return (features - self._feature_mean) / self._feature_std

    def predict_raw(
        self,
        gnss_drift: np.ndarray,
        false_positive_rate: np.ndarray,
    ) -> np.ndarray:
        if not self._is_fitted or self._regressor is None:
            raise RuntimeError("call fit before predict")
        features = self._normalize_features(
            stack_telemetry_features(gnss_drift, false_positive_rate),
            fit=False,
        )
        return clip_violation(self._regressor.predict(features))

    def fit(
        self,
        gnss_drift: np.ndarray,
        false_positive_rate: np.ndarray,
        violation_severity: np.ndarray,
    ) -> TelemetryViolationEstimator:
        y = clip_violation(violation_severity).reshape(-1)
        features = stack_telemetry_features(gnss_drift, false_positive_rate)
        if features.shape[0] < 2:
            raise ValueError("need at least 2 samples to fit")
        if features.shape[0] != y.shape[0]:
            raise ValueError("telemetry rows must match violation_severity length")

        norm_features = self._normalize_features(features, fit=True)
        self._regressor = Ridge(alpha=self.ridge_alpha)
        self._regressor.fit(norm_features, y)

        raw = clip_violation(self._regressor.predict(norm_features))
        if self.calibrate:
            self._calibrator = IsotonicViolationCalibrator()
            self._calibrator.fit(raw, y)
            preds = self._calibrator.predict(raw)
        else:
            self._calibrator = None
            preds = raw

        self._is_fitted = True
        resid = y - preds
        self._fit_result = TelemetryFitResult(
            train_mae=float(np.mean(np.abs(resid))),
            train_rmse=float(np.sqrt(np.mean(resid**2))),
            n_samples=int(y.shape[0]),
        )
        return self

    def fit_from_traces(
        self,
        traces: Iterable[Mapping[str, Any]],
    ) -> TelemetryViolationEstimator:
        gnss, fpr, v = extract_telemetry_from_traces(traces)
        return self.fit(gnss, fpr, v)

    def predict(
        self,
        gnss_drift: np.ndarray,
        false_positive_rate: np.ndarray,
    ) -> np.ndarray:
        raw = self.predict_raw(gnss_drift, false_positive_rate)
        if self._calibrator is not None:
            return self._calibrator.predict(raw)
        return raw
