"""Tests for partial observability theory and compound bounds (Theorem 2)."""

from __future__ import annotations

import numpy as np
import pytest

from field_compounding.theory.compound_bound import (
    compound_excess_risk,
    predict_compound_excess,
    validate_theorem_2,
)
from field_compounding.theory.partial_observability import (
    bound_inflation,
    coupling_inflation,
    linear_inflation,
    observability_ratio,
    rmse_v,
    violation_estimation_error,
)


@pytest.mark.parametrize(
    "v, psi, gamma, expected_min",
    [
        (np.array([0.0, 0.0]), np.array([1.0, 2.0]), np.zeros((2, 2)), 0.0),
        (np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.zeros((2, 2)), 1.0),
        (np.array([1.0, 0.0]), np.array([2.0, 3.0]), np.array([[0, 0.5], [0.5, 0]]), 2.0),
        (
            np.array([0.2, 0.3, 0.1]),
            np.array([1.0, 1.5, 0.5]),
            np.full((3, 3), 0.1) - np.diag([0.1, 0.1, 0.1]),
            0.0,
        ),
    ],
    ids=["zero_violations", "symmetric_linear", "single_module", "three_modules"],
)
def test_compound_excess_risk(
    v: np.ndarray,
    psi: np.ndarray,
    gamma: np.ndarray,
    expected_min: float,
) -> None:
    result = compound_excess_risk(v, psi, gamma)
    assert isinstance(result, float)
    assert result >= expected_min - 1e-9


def test_compound_excess_risk_coupling_term() -> None:
    v = np.array([1.0, 1.0])
    psi = np.zeros(2)
    gamma = np.array([[0.0, 0.4], [0.4, 0.0]])
    assert compound_excess_risk(v, psi, gamma) == pytest.approx(0.4)


def test_violation_estimation_error_sign() -> None:
    v_hat = np.array([0.3, 0.7])
    v_true = np.array([0.5, 0.5])
    delta = violation_estimation_error(v_hat, v_true)
    assert delta[0] == pytest.approx(-0.2)
    assert delta[1] == pytest.approx(0.2)


def test_rmse_v_zero_when_identical() -> None:
    v = np.array([0.1, 0.4, 0.9])
    assert rmse_v(v, v) == pytest.approx(0.0)


def test_observability_ratio_perfect_and_degraded() -> None:
    v = np.array([0.5, 0.5])
    assert observability_ratio(v, v) == pytest.approx(1.0)
    assert observability_ratio(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_linear_inflation_underestimation_positive() -> None:
    delta = np.array([-0.2, -0.1])
    psi = np.array([1.0, 2.0])
    assert linear_inflation(delta, psi) == pytest.approx(0.4)


def test_coupling_inflation_quadratic_gap() -> None:
    v_hat = np.array([0.4, 0.4])
    v_true = np.array([0.6, 0.6])
    gamma = np.array([[0.0, 0.5], [0.5, 0.0]])
    expected = 0.5 * (0.6 * 0.6 - 0.4 * 0.4)
    assert coupling_inflation(v_hat, v_true, gamma) == pytest.approx(expected)


def test_bound_inflation_decomposes() -> None:
    v_hat = np.array([0.3, 0.4])
    v_true = np.array([0.5, 0.6])
    psi = np.array([1.0, 1.0])
    gamma = np.array([[0.0, 0.2], [0.2, 0.0]])
    delta = violation_estimation_error(v_hat, v_true)
    total = bound_inflation(v_hat, v_true, psi, gamma)
    parts = linear_inflation(delta, psi) + coupling_inflation(v_hat, v_true, gamma)
    assert total == pytest.approx(parts)


def test_predict_compound_excess_returns_dict() -> None:
    gamma = np.array([[0.0, 0.3], [0.3, 0.0]])
    v_hat = np.array([0.4, 0.4])
    v_true = np.array([0.6, 0.5])
    out = predict_compound_excess(gamma, v_hat, v_true)
    assert isinstance(out, dict)
    assert out["prediction_error"] == pytest.approx(out["inflation"])


def test_predict_compound_excess_zero_error() -> None:
    gamma = np.array([[0.0, 0.2], [0.2, 0.0]])
    v = np.array([0.5, 0.5])
    out = predict_compound_excess(gamma, v, v)
    assert out["inflation"] == pytest.approx(0.0)
    assert out["bound_at_hat"] == pytest.approx(out["bound_at_true"])


@pytest.mark.parametrize(
    "observed, bound_hat, bound_true, holds",
    [
        (1.5, 1.0, 1.2, True),
        (1.1, 1.0, 1.2, True),
        (0.8, 1.0, 1.2, False),
    ],
)
def test_validate_theorem_2(
    observed: float,
    bound_hat: float,
    bound_true: float,
    holds: bool,
) -> None:
    result = validate_theorem_2(observed, bound_hat, bound_true)
    assert result["bound_at_hat"] == bound_hat
    assert result["bound_at_true"] == bound_true
    assert result["observed"] == observed
    assert result["inflation_gap"] == pytest.approx(bound_true - bound_hat)
    assert bool(result["theorem_holds"]) is holds


@pytest.mark.parametrize("n_modules", [2, 3, 4])
def test_predict_compound_excess_shape(n_modules: int) -> None:
    rng = np.random.default_rng(7)
    gamma = rng.uniform(0, 0.3, (n_modules, n_modules))
    gamma = 0.5 * (gamma + gamma.T)
    np.fill_diagonal(gamma, 0.0)
    v_hat = rng.uniform(0.1, 0.5, n_modules)
    v_true = rng.uniform(0.2, 0.8, n_modules)
    out = predict_compound_excess(gamma, v_hat, v_true)
    assert np.isfinite(out["relative_inflation"])
