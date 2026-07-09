"""Tests for telemetry violation estimators and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from field_compounding.models.calibration import (
    IsotonicViolationCalibrator,
    PlattViolationCalibrator,
    evaluate_violation_calibration,
    mean_absolute_error,
    violation_calibration_ece,
)
from field_compounding.models.violation_estimator import (
    TelemetryViolationEstimator,
    extract_telemetry_from_traces,
    proxy_violation_severity,
    stack_telemetry_features,
)


def _synthetic_telemetry(n: int = 240, seed: int = 42):
    rng = np.random.default_rng(seed)
    gnss = rng.exponential(2.5, size=n)
    fpr = rng.beta(1.5, 8.0, size=n)
    base = proxy_violation_severity(gnss, fpr)
    v = np.clip(base + rng.normal(0.0, 0.04, size=n), 0.0, 1.0)
    return gnss, fpr, v


def test_proxy_violation_in_unit_interval() -> None:
    gnss, fpr, _ = _synthetic_telemetry()
    v = proxy_violation_severity(gnss, fpr)
    assert np.all(v >= 0.0)
    assert np.all(v <= 1.0)


def test_proxy_violation_monotone_in_gnss() -> None:
    fpr = np.full(20, 0.1)
    gnss = np.linspace(0.0, 10.0, 20)
    v = proxy_violation_severity(gnss, fpr)
    assert np.all(np.diff(v) >= -1e-12)


def test_proxy_violation_monotone_in_fpr() -> None:
    gnss = np.full(20, 2.0)
    fpr = np.linspace(0.0, 0.5, 20)
    v = proxy_violation_severity(gnss, fpr)
    assert np.all(np.diff(v) >= -1e-12)


def test_stack_telemetry_features_shape() -> None:
    gnss = np.array([1.0, 2.0, 3.0])
    fpr = np.array([0.1, 0.2, 0.3])
    features = stack_telemetry_features(gnss, fpr)
    assert features.shape == (3, 3)


def test_stack_telemetry_mismatched_length_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        stack_telemetry_features(np.array([1.0, 2.0]), np.array([0.1]))


def test_extract_telemetry_from_traces() -> None:
    traces = [
        {"gnss_drift": 1.2, "false_positive_rate": 0.04, "violation_severity": 0.31},
        {"gnss_drift": 2.8, "false_positive_rate": 0.07, "violation_severity": 0.42},
    ]
    gnss, fpr, v = extract_telemetry_from_traces(traces)
    assert gnss.shape == (2,)
    assert fpr.shape == (2,)
    assert v.shape == (2,)
    assert v[0] == pytest.approx(0.31)


def test_telemetry_estimator_fit_predict_shape() -> None:
    gnss, fpr, v = _synthetic_telemetry()
    est = TelemetryViolationEstimator()
    est.fit(gnss, fpr, v)
    preds = est.predict(gnss[:10], fpr[:10])
    assert preds.shape == (10,)


def test_telemetry_estimator_outputs_clipped() -> None:
    gnss, fpr, v = _synthetic_telemetry()
    est = TelemetryViolationEstimator()
    est.fit(gnss, fpr, v)
    preds = est.predict(gnss, fpr)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_telemetry_estimator_not_fitted_raises() -> None:
    est = TelemetryViolationEstimator()
    with pytest.raises(RuntimeError, match="call fit before predict"):
        est.predict(np.array([1.0]), np.array([0.1]))


def test_telemetry_estimator_fit_requires_min_samples() -> None:
    est = TelemetryViolationEstimator()
    with pytest.raises(ValueError, match="at least 2 samples"):
        est.fit(np.array([1.0]), np.array([0.1]), np.array([0.2]))


def test_telemetry_estimator_improves_over_proxy() -> None:
    gnss, fpr, v = _synthetic_telemetry(n=400, seed=7)
    proxy = proxy_violation_severity(gnss, fpr)
    est = TelemetryViolationEstimator()
    est.fit(gnss, fpr, v)
    preds = est.predict(gnss, fpr)
    assert mean_absolute_error(v, preds) <= mean_absolute_error(v, proxy) + 0.02


def test_telemetry_estimator_deterministic() -> None:
    gnss, fpr, v = _synthetic_telemetry(seed=11)
    a = TelemetryViolationEstimator()
    b = TelemetryViolationEstimator()
    a.fit(gnss, fpr, v)
    b.fit(gnss, fpr, v)
    assert np.allclose(a.predict(gnss, fpr), b.predict(gnss, fpr))


def test_isotonic_calibrator_monotone() -> None:
    raw = np.linspace(0.05, 0.95, 50)
    y = np.clip(raw + 0.1, 0.0, 1.0)
    cal = IsotonicViolationCalibrator().fit(raw, y)
    out = cal.predict(raw)
    assert np.all(np.diff(out) >= -1e-12)


def test_isotonic_calibrator_clips_output() -> None:
    cal = IsotonicViolationCalibrator()
    raw = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.5, 1.0])
    cal.fit(raw, y)
    out = cal.predict(np.array([-0.2, 1.5]))
    assert out[0] >= 0.0
    assert out[1] <= 1.0


def test_platt_calibrator_fit_predict() -> None:
    raw = np.linspace(0.1, 0.9, 80)
    y = np.clip(raw * 0.8 + 0.1, 0.0, 1.0)
    cal = PlattViolationCalibrator().fit(raw, y)
    out = cal.predict(raw)
    assert out.shape == raw.shape
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_calibration_report_improves_ece() -> None:
    gnss, fpr, v = _synthetic_telemetry(n=300, seed=19)
    raw = proxy_violation_severity(gnss, fpr)
    cal = IsotonicViolationCalibrator().fit(raw, v)
    calibrated = cal.predict(raw)
    report = evaluate_violation_calibration(v, raw, calibrated)
    assert report.ece_after <= report.ece_before + 1e-9
    assert report.n_samples == 300
    assert report.mae_improvement >= -1e-9


def test_violation_calibration_ece_zero_for_perfect_predictions() -> None:
    y = np.linspace(0.1, 0.9, 20)
    assert violation_calibration_ece(y, y) == pytest.approx(0.0, abs=1e-12)
