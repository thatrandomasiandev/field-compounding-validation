"""Transfer Module 11 coupling estimates to field traces via trace-density correction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

VALID_LOOP_NODES: frozenset[str] = frozenset(
    {
        "scene_repr",
        "visual_ssl",
        "sim_to_real",
        "visuomotor",
        "causal",
        "world_model",
        "equivariant",
        "scene_graph",
        "uncertainty",
        "neurosymbolic",
        "federated",
        "safety",
    }
)

MODULE_ID_TO_LOOP_NODE: dict[int, str] = {
    3: "scene_repr",
    4: "visual_ssl",
    5: "sim_to_real",
    6: "visuomotor",
    7: "causal",
    8: "world_model",
    9: "equivariant",
    10: "scene_graph",
    11: "uncertainty",
    12: "neurosymbolic",
    13: "federated",
    14: "safety",
}

LOOP_NODE_TO_MODULE_ID: dict[str, int] = {v: k for k, v in MODULE_ID_TO_LOOP_NODE.items()}

DEFAULT_LOOP_NODE_ORDER: tuple[str, ...] = tuple(MODULE_ID_TO_LOOP_NODE[i] for i in range(3, 15))

MODULE11_RESULTS_ENV = "MODULE11_RESULTS"


def module_id_to_loop_node(module_id: int) -> str:
    try:
        return MODULE_ID_TO_LOOP_NODE[module_id]
    except KeyError as exc:
        raise ValueError(f"unsupported module_id: {module_id}") from exc


def loop_node_to_module_id(loop_node: str) -> int:
    if loop_node not in VALID_LOOP_NODES:
        raise ValueError(f"unknown loop_node: {loop_node!r}")
    return LOOP_NODE_TO_MODULE_ID[loop_node]


def bundled_coupling_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "observatory" / "module11_coupling.json"


@dataclass(frozen=True)
class CouplingMatrix:
    module_ids: tuple[int, ...]
    loop_nodes: tuple[str, ...]
    gamma: np.ndarray
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    source: str = "bundled"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        gamma = np.asarray(self.gamma, dtype=np.float64)
        if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
            raise ValueError("gamma must be a square matrix")
        n = gamma.shape[0]
        if len(self.module_ids) != n or len(self.loop_nodes) != n:
            raise ValueError("module_ids and loop_nodes must match gamma shape")
        for node in self.loop_nodes:
            if node not in VALID_LOOP_NODES:
                raise ValueError(f"unknown loop_node in coupling matrix: {node!r}")
        object.__setattr__(self, "gamma", gamma)

    @property
    def size(self) -> int:
        return int(self.gamma.shape[0])

    def pair_gamma(self, loop_node_k: str, loop_node_l: str) -> float:
        i = self.loop_nodes.index(loop_node_k)
        j = self.loop_nodes.index(loop_node_l)
        return float(self.gamma[i, j])


def _parse_coupling_payload(payload: Mapping[str, Any], *, source: str) -> CouplingMatrix:
    if "coupling_matrix" in payload:
        payload = payload["coupling_matrix"]

    module_ids = tuple(int(x) for x in payload["module_ids"])
    loop_nodes = payload.get("loop_nodes")
    if loop_nodes is None:
        loop_nodes = tuple(module_id_to_loop_node(mid) for mid in module_ids)
    else:
        loop_nodes = tuple(str(x) for x in loop_nodes)

    gamma = np.asarray(payload["gamma"], dtype=np.float64)
    ci_lower = payload.get("ci_lower")
    ci_upper = payload.get("ci_upper")
    metadata = payload.get("metadata")

    return CouplingMatrix(
        module_ids=module_ids,
        loop_nodes=loop_nodes,
        gamma=gamma,
        ci_lower=None if ci_lower is None else np.asarray(ci_lower, dtype=np.float64),
        ci_upper=None if ci_upper is None else np.asarray(ci_upper, dtype=np.float64),
        source=source,
        metadata=metadata,
    )


def _resolve_module11_path(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser().resolve()

    env_value = os.environ.get(MODULE11_RESULTS_ENV)
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if candidate.is_dir():
            section = candidate / "section_15.json"
            if section.is_file():
                return section
            raise FileNotFoundError(
                f"{MODULE11_RESULTS_ENV}={candidate} is a directory without section_15.json"
            )
        return candidate

    bundled = bundled_coupling_path()
    if not bundled.is_file():
        raise FileNotFoundError(f"bundled coupling stub missing: {bundled}")
    return bundled


def load_module11_coupling(path: Path | None = None) -> CouplingMatrix:
    resolved = _resolve_module11_path(path)
    with open(resolved) as handle:
        payload = json.load(handle)
    return _parse_coupling_payload(payload, source=str(resolved))


def trace_density_by_node(
    trace_counts: Mapping[str, int],
    *,
    reference_rows: int = 200,
) -> dict[str, float]:
    if reference_rows <= 0:
        raise ValueError("reference_rows must be positive")

    densities: dict[str, float] = {}
    for node in DEFAULT_LOOP_NODE_ORDER:
        count = int(trace_counts.get(node, 0))
        densities[node] = float(np.clip(count / reference_rows, 0.0, 1.0))
    return densities


def field_correction_factor(density_k: float, density_l: float) -> float:
    dk = float(np.clip(density_k, 0.0, 1.0))
    dl = float(np.clip(density_l, 0.0, 1.0))
    return float(np.sqrt(dk * dl))


def correction_factor_matrix(
    loop_nodes: tuple[str, ...] | list[str],
    trace_densities: Mapping[str, float],
) -> np.ndarray:
    nodes = tuple(loop_nodes)
    k = len(nodes)
    factors = np.ones((k, k), dtype=np.float64)
    for i, node_i in enumerate(nodes):
        rho_i = float(trace_densities.get(node_i, 0.0))
        for j, node_j in enumerate(nodes):
            rho_j = float(trace_densities.get(node_j, 0.0))
            factors[i, j] = field_correction_factor(rho_i, rho_j)
    return factors


def apply_field_correction(
    gamma: np.ndarray,
    loop_nodes: tuple[str, ...] | list[str],
    trace_densities: Mapping[str, float],
) -> np.ndarray:
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    if gamma_arr.ndim != 2 or gamma_arr.shape[0] != gamma_arr.shape[1]:
        raise ValueError("gamma must be a square matrix")
    if len(loop_nodes) != gamma_arr.shape[0]:
        raise ValueError("loop_nodes length must match gamma shape")

    factors = correction_factor_matrix(loop_nodes, trace_densities)
    corrected = gamma_arr * factors
    np.fill_diagonal(corrected, 0.0)
    return corrected


def transfer_coupling_to_field(
    coupling: CouplingMatrix,
    trace_counts: Mapping[str, int],
    *,
    reference_rows: int = 200,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    densities = trace_density_by_node(trace_counts, reference_rows=reference_rows)
    factors = correction_factor_matrix(coupling.loop_nodes, densities)
    gamma_field = apply_field_correction(coupling.gamma, coupling.loop_nodes, densities)
    return gamma_field, densities, factors


def dominant_field_couplings(
    gamma_field: np.ndarray,
    loop_nodes: tuple[str, ...] | list[str],
    *,
    min_abs: float = 0.02,
) -> list[tuple[str, str, float]]:
    nodes = tuple(loop_nodes)
    pairs: list[tuple[str, str, float]] = []
    k = len(nodes)
    for i in range(k):
        for j in range(i + 1, k):
            value = float(gamma_field[i, j])
            if abs(value) >= min_abs:
                pairs.append((nodes[i], nodes[j], value))
    return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)


def estimate_full_coupling_matrix(
    all_results: dict[int, dict[str, np.ndarray]],
    metric_key: str = "normalized_score",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the full K x K coupling matrix from module sweep results."""
    from field_compounding.theory.compound_bound import estimate_gamma_kl

    module_ids = sorted(all_results.keys())
    k = len(module_ids)
    gamma = np.zeros((k, k))
    ci_lower = np.zeros((k, k))
    ci_upper = np.zeros((k, k))

    for i, k_id in enumerate(module_ids):
        for j, l_id in enumerate(module_ids):
            if i == j:
                continue
            g, cl, cu = estimate_gamma_kl(
                all_results[k_id],
                all_results[l_id],
                metric_key=metric_key,
                n_bootstrap=n_bootstrap,
                seed=seed + i * k + j,
            )
            gamma[i, j] = g
            ci_lower[i, j] = cl
            ci_upper[i, j] = cu

    gamma = (gamma + gamma.T) / 2
    ci_lower = np.minimum(ci_lower, ci_lower.T)
    ci_upper = np.maximum(ci_upper, ci_upper.T)
    return gamma, ci_lower, ci_upper
