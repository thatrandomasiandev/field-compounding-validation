"""Compound excess risk bounds under partial observability (Theorem 2)."""

from __future__ import annotations

import numpy as np

from field_compounding.theory.partial_observability import bound_inflation


def compound_excess_risk(
    violation_severities: np.ndarray,
    psi: np.ndarray,
    gamma_matrix: np.ndarray,
) -> float:
    """Compute compound excess risk lower bound (Theorem 1)."""
    v = np.asarray(violation_severities, dtype=np.float64)
    psi_arr = np.asarray(psi, dtype=np.float64)
    gamma = np.asarray(gamma_matrix, dtype=np.float64)

    linear = float(np.dot(v, psi_arr))

    coupling = 0.0
    k_count = len(v)
    for k in range(k_count):
        for l in range(k + 1, k_count):
            coupling += gamma[k, l] * v[k] * v[l]

    return linear + coupling


def predict_compound_excess(
    gamma_hat: np.ndarray,
    v_hat: np.ndarray,
    v_true: np.ndarray,
    psi_hat: np.ndarray | None = None,
) -> dict[str, float]:
    """Predict compound excess under partial observability (Theorem 2)."""
    v_hat_arr = np.asarray(v_hat, dtype=np.float64)
    v_true_arr = np.asarray(v_true, dtype=np.float64)
    gamma = np.asarray(gamma_hat, dtype=np.float64)

    if psi_hat is None:
        psi_arr = np.ones(len(v_hat_arr), dtype=np.float64)
    else:
        psi_arr = np.asarray(psi_hat, dtype=np.float64)

    bound_at_hat = compound_excess_risk(v_hat_arr, psi_arr, gamma)
    bound_at_true = compound_excess_risk(v_true_arr, psi_arr, gamma)
    inflation = bound_inflation(v_hat_arr, v_true_arr, psi_arr, gamma)

    denom = max(abs(bound_at_true), 1e-9)
    relative_inflation = inflation / denom

    return {
        "bound_at_hat": bound_at_hat,
        "bound_at_true": bound_at_true,
        "inflation": inflation,
        "relative_inflation": relative_inflation,
        "prediction_error": bound_at_true - bound_at_hat,
    }


def validate_theorem_2(
    observed_compound_loss: float,
    bound_at_hat: float,
    bound_at_true: float,
) -> dict[str, float]:
    """Check Theorem 2: observed loss vs hat-bound and true-bound."""
    slack_hat = observed_compound_loss - bound_at_hat
    slack_true = observed_compound_loss - bound_at_true
    inflation_gap = bound_at_true - bound_at_hat

    theorem_holds = slack_hat >= -1e-6 and inflation_gap >= -1e-6

    return {
        "bound_at_hat": bound_at_hat,
        "bound_at_true": bound_at_true,
        "observed": observed_compound_loss,
        "slack_hat": slack_hat,
        "slack_true": slack_true,
        "inflation_gap": inflation_gap,
        "theorem_holds": float(theorem_holds),
    }
