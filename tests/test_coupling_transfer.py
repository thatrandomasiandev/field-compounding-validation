"""Tests for Module 11 coupling transfer and trace-density correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from field_compounding.theory.coupling_transfer import (
    VALID_LOOP_NODES,
    CouplingMatrix,
    apply_field_correction,
    bundled_coupling_path,
    correction_factor_matrix,
    dominant_field_couplings,
    field_correction_factor,
    load_module11_coupling,
    loop_node_to_module_id,
    module_id_to_loop_node,
    trace_density_by_node,
    transfer_coupling_to_field,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED = REPO_ROOT / "observatory" / "module11_coupling.json"


@pytest.fixture
def sample_coupling() -> CouplingMatrix:
    gamma = np.array(
        [
            [0.0, 0.1, -0.5],
            [0.1, 0.0, 0.3],
            [-0.5, 0.3, 0.0],
        ],
        dtype=np.float64,
    )
    return CouplingMatrix(
        module_ids=(3, 4, 5),
        loop_nodes=("scene_repr", "visual_ssl", "sim_to_real"),
        gamma=gamma,
        source="test",
    )


def test_bundled_coupling_json_exists_and_parses() -> None:
    assert BUNDLED.is_file()
    payload = json.loads(BUNDLED.read_text())
    assert payload["module_ids"] == list(range(3, 15))
    assert len(payload["loop_nodes"]) == 12
    assert len(payload["gamma"]) == 12


def test_load_module11_coupling_from_bundled_stub() -> None:
    coupling = load_module11_coupling(BUNDLED)
    assert coupling.size == 12
    assert coupling.gamma.shape == (12, 12)
    assert coupling.loop_nodes[0] == "scene_repr"
    assert coupling.loop_nodes[-1] == "safety"


def test_load_module11_coupling_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    section = {
        "coupling_matrix": {
            "module_ids": [3, 4],
            "loop_nodes": ["scene_repr", "visual_ssl"],
            "gamma": [[0.0, 0.25], [0.25, 0.0]],
        }
    }
    path = tmp_path / "section_15.json"
    path.write_text(json.dumps(section))
    monkeypatch.setenv("MODULE11_RESULTS", str(path))
    coupling = load_module11_coupling()
    assert coupling.size == 2
    assert coupling.gamma[0, 1] == pytest.approx(0.25)


def test_load_module11_coupling_from_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    section = {
        "coupling_matrix": {
            "module_ids": [14],
            "loop_nodes": ["safety"],
            "gamma": [[0.0]],
        }
    }
    (tmp_path / "section_15.json").write_text(json.dumps(section))
    monkeypatch.setenv("MODULE11_RESULTS", str(tmp_path))
    coupling = load_module11_coupling()
    assert coupling.loop_nodes == ("safety",)


@pytest.mark.parametrize(
    "module_id, loop_node",
    [(3, "scene_repr"), (8, "world_model"), (14, "safety")],
)
def test_module_loop_node_round_trip(module_id: int, loop_node: str) -> None:
    assert module_id_to_loop_node(module_id) == loop_node
    assert loop_node_to_module_id(loop_node) == module_id


def test_invalid_module_id_raises() -> None:
    with pytest.raises(ValueError, match="unsupported module_id"):
        module_id_to_loop_node(99)


def test_trace_density_clips_to_unit_interval() -> None:
    dense = trace_density_by_node({"scene_repr": 400}, reference_rows=200)
    sparse = trace_density_by_node({"scene_repr": 50}, reference_rows=200)
    missing = trace_density_by_node({}, reference_rows=200)
    assert dense["scene_repr"] == pytest.approx(1.0)
    assert sparse["scene_repr"] == pytest.approx(0.25)
    assert missing["safety"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "rho_k, rho_l, expected",
    [(1.0, 1.0, 1.0), (0.0, 1.0, 0.0), (0.25, 0.64, 0.4)],
)
def test_field_correction_factor(rho_k: float, rho_l: float, expected: float) -> None:
    assert field_correction_factor(rho_k, rho_l) == pytest.approx(expected)


def test_apply_field_correction_attenuates_sparse_nodes(sample_coupling: CouplingMatrix) -> None:
    densities = trace_density_by_node(
        {"scene_repr": 200, "visual_ssl": 200, "sim_to_real": 0},
        reference_rows=200,
    )
    corrected = apply_field_correction(sample_coupling.gamma, sample_coupling.loop_nodes, densities)
    assert corrected[0, 1] == pytest.approx(sample_coupling.gamma[0, 1])
    assert corrected[0, 2] == pytest.approx(0.0)
    assert corrected[2, 0] == pytest.approx(0.0)
    assert np.all(np.diag(corrected) == 0.0)


def test_transfer_coupling_to_field_returns_factors(sample_coupling: CouplingMatrix) -> None:
    counts = {node: 200 for node in sample_coupling.loop_nodes}
    gamma_field, densities, factors = transfer_coupling_to_field(
        sample_coupling, counts, reference_rows=200
    )
    assert gamma_field.shape == sample_coupling.gamma.shape
    assert factors.shape == sample_coupling.gamma.shape
    assert all(densities[node] == pytest.approx(1.0) for node in sample_coupling.loop_nodes)
    assert gamma_field == pytest.approx(sample_coupling.gamma)


def test_correction_factor_matrix_is_symmetric() -> None:
    densities = trace_density_by_node({"scene_repr": 100, "visual_ssl": 50}, reference_rows=200)
    factors = correction_factor_matrix(("scene_repr", "visual_ssl"), densities)
    assert factors.shape == (2, 2)
    assert factors[0, 1] == pytest.approx(factors[1, 0])


def test_dominant_field_couplings_filters_small_entries(sample_coupling: CouplingMatrix) -> None:
    ranked = dominant_field_couplings(sample_coupling.gamma, sample_coupling.loop_nodes, min_abs=0.2)
    labels = {(a, b) for a, b, _ in ranked}
    assert ("scene_repr", "sim_to_real") in labels
    assert all(abs(value) >= 0.2 for _, _, value in ranked)


def test_coupling_matrix_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must match gamma shape"):
        CouplingMatrix(
            module_ids=(3, 4),
            loop_nodes=("scene_repr", "visual_ssl", "sim_to_real"),
            gamma=np.zeros((3, 3)),
        )


def test_all_valid_loop_nodes_have_module_ids() -> None:
    assert len(VALID_LOOP_NODES) == 12
    for node in VALID_LOOP_NODES:
        assert loop_node_to_module_id(node) in range(3, 15)


def test_bundled_path_points_to_observatory_file() -> None:
    assert bundled_coupling_path().name == "module11_coupling.json"
    assert bundled_coupling_path().parent.name == "observatory"
