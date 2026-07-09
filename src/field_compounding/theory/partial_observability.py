"""Partial observability: violation estimation error and bound inflation."""

from __future__ import annotations

import numpy as np


def violation_estimation_error(v_hat: np.ndarray, v_true: np.ndarray) -> np.ndarray:
    """Per-module estimation error delta_k = v_hat_k - v_true_k."""
    return np.asarray(v_hat, dtype=np.float64) - np.asarray(v_true, dtype=np.float64)


def rmse_v(v_hat: np.ndarray, v_true: np.ndarray) -> float:
    """Root mean squared error between inferred and oracle severities."""
    delta = violation_estimation_error(v_hat, v_true)
    return float(np.sqrt(np.mean(delta**2)))


def observability_ratio(v_hat: np.ndarray, v_true: np.ndarray) -> float:
    """Fraction of oracle severity energy captured by telemetry estimates."""
    v_hat_arr = np.asarray(v_hat, dtype=np.float64)
    v_true_arr = np.asarray(v_true, dtype=np.float64)

    num = float(np.dot(v_hat_arr, v_true_arr))
    denom = float(np.linalg.norm(v_hat_arr) * np.linalg.norm(v_true_arr))
    if denom < 1e-12:
        return 1.0 if np.allclose(v_hat_arr, v_true_arr) else 0.0

    cos_sim = num / denom
    return float(np.clip(cos_sim, 0.0, 1.0))


def linear_inflation(delta: np.ndarray, psi: np.ndarray) -> float:
    """First-order inflation from per-module estimation error."""
    delta_arr = np.asarray(delta, dtype=np.float64)
    psi_arr = np.asarray(psi, dtype=np.float64)
    return float(-np.dot(psi_arr, delta_arr))


def coupling_inflation(
    v_hat: np.ndarray,
    v_true: np.ndarray,
    gamma_matrix: np.ndarray,
) -> float:
    """Quadratic inflation from coupling under mis-estimated severities."""
    v_hat_arr = np.asarray(v_hat, dtype=np.float64)
    v_true_arr = np.asarray(v_true, dtype=np.float64)
    gamma = np.asarray(gamma_matrix, dtype=np.float64)

    k_count = len(v_hat_arr)
    inflation = 0.0
    for k in range(k_count):
        for l in range(k + 1, k_count):
            true_term = v_true_arr[k] * v_true_arr[l]
            hat_term = v_hat_arr[k] * v_hat_arr[l]
            inflation += gamma[k, l] * (true_term - hat_term)

    return float(inflation)


def bound_inflation(
    v_hat: np.ndarray,
    v_true: np.ndarray,
    psi: np.ndarray,
    gamma_matrix: np.ndarray,
) -> float:
    """Total bound inflation from partial observability (Theorem 2)."""
    delta = violation_estimation_error(v_hat, v_true)
    return linear_inflation(delta, psi) + coupling_inflation(v_hat, v_true, gamma_matrix)
