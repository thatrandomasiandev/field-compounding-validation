"""Condition D: heterogeneous violations plus architectural decoupling (field)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from field_compounding.models.decoupling_stack import (
    MODULE_ID_TO_LOOP_NODE,
    DecouplingConfig,
    DecouplingStack,
)

CONDITION_C_VIOLATIONS: dict[int, float] = {
    3: 0.3,
    4: 0.2,
    5: 0.7,
    6: 0.4,
    7: 0.6,
    8: 0.3,
    9: 0.5,
    10: 0.4,
    11: 0.3,
    12: 0.5,
    13: 0.8,
    14: 0.2,
}

DEFAULT_MODULE_IDS: tuple[int, ...] = tuple(range(3, 15))


@dataclass(frozen=True)
class ConditionDSpec:
    description: str
    violation_severities: dict[int, float]
    mitigation_enabled: bool = True
    cmd_latency_ms: float = 50.0
    feature_dim: int = 32
    latent_dim: int = 8
    module_ids: tuple[int, ...] = DEFAULT_MODULE_IDS

    def __post_init__(self) -> None:
        for mod_id, severity in self.violation_severities.items():
            if mod_id not in self.module_ids:
                raise ValueError(f"module_id {mod_id} not in module_ids")
            if not 0.0 <= severity <= 1.0:
                raise ValueError(f"violation severity for module {mod_id} must be in [0, 1]")


def default_condition_d_spec(cmd_latency_ms: float = 50.0) -> ConditionDSpec:
    return ConditionDSpec(
        description=(
            "Condition D: heterogeneous field violations with architectural "
            "decoupling and latency attenuation"
        ),
        violation_severities=dict(CONDITION_C_VIOLATIONS),
        mitigation_enabled=True,
        cmd_latency_ms=cmd_latency_ms,
    )


def condition_c_spec() -> ConditionDSpec:
    return ConditionDSpec(
        description="Condition C: heterogeneous violations without mitigation",
        violation_severities=dict(CONDITION_C_VIOLATIONS),
        mitigation_enabled=False,
        cmd_latency_ms=0.0,
    )


def violation_vector(spec: ConditionDSpec) -> np.ndarray:
    return np.array(
        [float(spec.violation_severities[mod_id]) for mod_id in spec.module_ids],
        dtype=np.float64,
    )


def build_decoupling_stack(spec: ConditionDSpec) -> DecouplingStack:
    config = DecouplingConfig(
        feature_dim=spec.feature_dim,
        latent_dim=spec.latent_dim,
        cmd_latency_ms=spec.cmd_latency_ms,
    )
    return DecouplingStack(config)


def compound_excess_bound(
    violation_severities: np.ndarray,
    psi: np.ndarray,
    gamma_matrix: np.ndarray,
) -> float:
    v = np.asarray(violation_severities, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    gamma = np.asarray(gamma_matrix, dtype=np.float64)
    linear = float(np.dot(v, psi))
    coupling = 0.0
    k_count = len(v)
    for k in range(k_count):
        for l in range(k + 1, k_count):
            coupling += gamma[k, l] * v[k] * v[l]
    return linear + coupling


def predicted_bound_with_mitigation(
    spec: ConditionDSpec,
    psi: np.ndarray,
    gamma_matrix: np.ndarray,
    stack: DecouplingStack | None = None,
) -> dict[str, float]:
    v = violation_vector(spec)
    gamma = np.asarray(gamma_matrix, dtype=np.float64)
    baseline = compound_excess_bound(v, psi, gamma)

    if not spec.mitigation_enabled:
        return {
            "baseline_bound": baseline,
            "mitigated_bound": baseline,
            "gamma_scale": 1.0,
            "mitigation_delta": 0.0,
        }

    active_stack = stack or build_decoupling_stack(spec)
    gamma_mit = active_stack.attenuate_coupling_matrix(gamma, spec.cmd_latency_ms)
    mitigated = compound_excess_bound(v, psi, gamma_mit)
    scale = active_stack.effective_gamma_scale(spec.cmd_latency_ms)
    return {
        "baseline_bound": baseline,
        "mitigated_bound": mitigated,
        "gamma_scale": scale,
        "mitigation_delta": baseline - mitigated,
    }


def compare_conditions_c_d(
    psi: np.ndarray,
    gamma_matrix: np.ndarray,
    cmd_latency_ms: float = 50.0,
) -> dict[str, Any]:
    spec_c = condition_c_spec()
    spec_d = default_condition_d_spec(cmd_latency_ms=cmd_latency_ms)
    stack = build_decoupling_stack(spec_d)

    c_bound = compound_excess_bound(violation_vector(spec_c), psi, gamma_matrix)
    d_result = predicted_bound_with_mitigation(spec_d, psi, gamma_matrix, stack=stack)

    reduction_pct = 0.0
    if abs(c_bound) > 1e-12:
        reduction_pct = 100.0 * d_result["mitigation_delta"] / abs(c_bound)

    return {
        "condition_c_bound": c_bound,
        "condition_d_bound": d_result["mitigated_bound"],
        "gamma_scale": d_result["gamma_scale"],
        "bound_reduction": d_result["mitigation_delta"],
        "bound_reduction_pct": reduction_pct,
        "cmd_latency_ms": cmd_latency_ms,
    }


def build_compound_condition_entry(spec: ConditionDSpec | None = None) -> dict[str, Any]:
    active = spec or default_condition_d_spec()
    return {
        "D": {
            "description": active.description,
            "mitigation_enabled": active.mitigation_enabled,
            "cmd_latency_ms": active.cmd_latency_ms,
            "violation_severities": {str(k): v for k, v in active.violation_severities.items()},
            "loop_nodes": {
                str(mod_id): MODULE_ID_TO_LOOP_NODE[mod_id] for mod_id in active.module_ids
            },
        }
    }


def latency_sweep_bounds(
    psi: np.ndarray,
    gamma_matrix: np.ndarray,
    latency_grid_ms: np.ndarray | None = None,
) -> list[dict[str, float]]:
    grid = latency_grid_ms if latency_grid_ms is not None else np.array([0.0, 25.0, 50.0, 100.0, 200.0])
    rows: list[dict[str, float]] = []
    for latency in grid:
        spec = default_condition_d_spec(cmd_latency_ms=float(latency))
        stack = build_decoupling_stack(spec)
        result = predicted_bound_with_mitigation(spec, psi, gamma_matrix, stack=stack)
        rows.append(
            {
                "cmd_latency_ms": float(latency),
                "mitigated_bound": result["mitigated_bound"],
                "gamma_scale": result["gamma_scale"],
            }
        )
    return rows
