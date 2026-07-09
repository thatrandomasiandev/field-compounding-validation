"""Statistical testing utilities: bootstrap CIs, permutation tests."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        v = float(values[0])
        return v, v

    rng = np.random.default_rng(seed)
    n = len(values)
    alpha = 1 - confidence

    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
    alternative: str = "two-sided",
) -> float:
    """Two-sample permutation test for difference in means."""
    rng = np.random.default_rng(seed)

    observed_diff = np.mean(group_a) - np.mean(group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_diff = np.mean(perm[:n_a]) - np.mean(perm[n_a:])

        if alternative == "two-sided":
            if abs(perm_diff) >= abs(observed_diff):
                count += 1
        elif alternative == "greater":
            if perm_diff >= observed_diff:
                count += 1
        elif alternative == "less":
            if perm_diff <= observed_diff:
                count += 1

    return (count + 1) / (n_permutations + 1)


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap test comparing two models."""
    rng = np.random.default_rng(seed)

    diffs = scores_a - scores_b
    observed_mean_diff = float(np.mean(diffs))
    n = len(diffs)
    centered = diffs - observed_mean_diff

    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(centered, size=n, replace=True)
        if abs(np.mean(sample)) >= abs(observed_mean_diff):
            count += 1

    p_value = (count + 1) / (n_bootstrap + 1)

    boot_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_diffs[i] = np.mean(sample)

    ci_width = float(np.percentile(boot_diffs, 97.5) - np.percentile(boot_diffs, 2.5))
    return float(p_value), observed_mean_diff, ci_width


def effect_size_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std < 1e-10:
        return 0.0

    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)
