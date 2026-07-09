"""Compound excess risk bounds and coupling estimation."""

from __future__ import annotations

import numpy as np

from field_compounding.theory.partial_observability import bound_inflation


def _align_by_severity(
    v_k: np.ndarray,
    v_l: np.ndarray,
    score_k: np.ndarray,
    score_l: np.ndarray,
    *,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pairs: list[tuple[int, int]] = []
    for i, vk in enumerate(v_k):
        for j, vl in enumerate(v_l):
            if abs(float(vk) - float(vl)) <= tol:
                pairs.append((i, j))
                break
    if len(pairs) < 3:
        return None
    v_a = np.array([v_k[i] for i, _ in pairs], dtype=np.float64)
    score_k_a = np.array([score_k[i] for i, _ in pairs], dtype=np.float64)
    score_l_a = np.array([score_l[j] for _, j in pairs], dtype=np.float64)
    return v_a, score_k_a, score_l_a


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(np.clip(r, -1.0, 1.0))


def estimate_gamma_kl(
    results_k: dict[str, np.ndarray],
    results_l: dict[str, np.ndarray],
    metric_key: str = "normalized_score",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Estimate pairwise coupling coefficient gamma_{k,l}."""
    rng = np.random.default_rng(seed)

    v_k = results_k["violation_severity"]
    v_l = results_l["violation_severity"]
    score_k = np.asarray(results_k.get(metric_key, results_k.get("normalized_score")))
    score_l = np.asarray(results_l[metric_key])

    aligned = _align_by_severity(v_k, v_l, score_k, score_l)
    if aligned is None:
        return 0.0, -1.0, 1.0
    v_a, score_k_a, score_l_a = aligned

    residual_k = score_k_a - np.polyval(np.polyfit(v_a, score_k_a, 1), v_a)
    residual_l = score_l_a - np.polyval(np.polyfit(v_a, score_l_a, 1), v_a)
    gamma_hat = _correlation(residual_k, residual_l)

    n = len(v_a)
    bootstraps = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bootstraps[b] = _correlation(residual_k[idx], residual_l[idx])

    ci_lower = float(np.percentile(bootstraps, 2.5))
    ci_upper = float(np.percentile(bootstraps, 97.5))
    return float(gamma_hat), ci_lower, ci_upper


def compound_excess_risk(
    violation_severities: np.ndarray,
    psi: np.ndarray,
    gamma_matrix: np.ndarray,
) -> float:
    """Compute compound excess risk lower bound."""
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


def estimate_psi_k(violation_severities: np.ndarray, scores: np.ndarray) -> float:
    """Estimate per-module sensitivity psi_k via OLS slope."""
    beta = np.polyfit(violation_severities, scores, deg=1)
    return float(beta[0])


def validate_theorem(
    observed_compound_loss: float,
    predicted_lower_bound: float,
) -> dict[str, float]:
    """Check whether observed compound loss exceeds the predicted bound."""
    slack = observed_compound_loss - predicted_lower_bound
    return {
        "bound": predicted_lower_bound,
        "observed": observed_compound_loss,
        "slack": slack,
        "theorem_holds": float(slack >= -1e-6),
    }


def predict_compound_excess(
    gamma_hat: np.ndarray,
    v_hat: np.ndarray,
    v_true: np.ndarray,
    psi_hat: np.ndarray | None = None,
) -> dict[str, float]:
    """Predict compound excess under estimated vs true violation severities."""
    v_hat_arr = np.asarray(v_hat, dtype=np.float64)
    v_true_arr = np.asarray(v_true, dtype=np.float64)
    gamma = np.asarray(gamma_hat, dtype=np.float64)
    psi_arr = np.ones(len(v_hat_arr), dtype=np.float64) if psi_hat is None else np.asarray(psi_hat)

    bound_at_hat = compound_excess_risk(v_hat_arr, psi_arr, gamma)
    bound_at_true = compound_excess_risk(v_true_arr, psi_arr, gamma)
    inflation = bound_inflation(v_hat_arr, v_true_arr, psi_arr, gamma)
    denom = max(abs(bound_at_true), 1e-9)

    return {
        "bound_at_hat": bound_at_hat,
        "bound_at_true": bound_at_true,
        "inflation": inflation,
        "relative_inflation": inflation / denom,
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
