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
    module_ids = tuple(range(3, 15))
    loop_nodes = tuple(module_id_to_loop_node(mid) for mid in module_ids)
    gamma = np.full((12, 12), 0.05, dtype=np.float64)
    np.fill_diagonal(gamma, 0.0)
    return CouplingMatrix(module_ids=module_ids, loop_nodes=loop_nodes, gamma=gamma, source="test")


def test_bundled_coupling_json_exists_and_parses():
    assert BUNDLED.is_file()
    payload = json.loads(BUNDLED.read_text())
    assert "gamma" in payload
    assert len(payload["module_ids"]) == 12


def test_bundled_path_points_to_observatory_file():
    path = bundled_coupling_path()
    assert path.name == "module11_coupling.json"
    assert path.parent.name == "observatory"


def test_load_module11_coupling_from_bundled_stub(sample_coupling):
    coupling = load_module11_coupling()
    assert coupling.size == 12
    assert coupling.gamma.shape == (12, 12)
    assert coupling.loop_nodes[0] == "scene_repr"


def test_load_module11_coupling_from_env(tmp_path, monkeypatch, sample_coupling):
    section = {
        "coupling_matrix": {
            "module_ids": list(sample_coupling.module_ids),
            "loop_nodes": list(sample_coupling.loop_nodes),
            "gamma": sample_coupling.gamma.tolist(),
        }
    }
    out = tmp_path / "section_15.json"
    out.write_text(json.dumps(section))
    monkeypatch.setenv("MODULE11_RESULTS", str(tmp_path))
    coupling = load_module11_coupling()
    assert coupling.gamma.shape == (12, 12)
    assert coupling.loop_nodes == sample_coupling.loop_nodes


def test_load_module11_coupling_from_results_dir(tmp_path, sample_coupling):
    out = tmp_path / "coupling.json"
    out.write_text(
        json.dumps(
            {
                "module_ids": list(sample_coupling.module_ids),
                "loop_nodes": list(sample_coupling.loop_nodes),
                "gamma": sample_coupling.gamma.tolist(),
            }
        )
    )
    coupling = load_module11_coupling(out)
    assert coupling.source.endswith("coupling.json")


@pytest.mark.parametrize(
    "module_id, loop_node",
    [(3, "scene_repr"), (8, "world_model"), (14, "safety")],
)
def test_module_loop_node_round_trip(module_id, loop_node):
    assert module_id_to_loop_node(module_id) == loop_node
    assert loop_node_to_module_id(loop_node) == module_id


def test_invalid_module_id_raises():
    with pytest.raises(ValueError, match="unsupported module_id"):
        module_id_to_loop_node(99)


def test_trace_density_clips_to_unit_interval():
    sparse = trace_density_by_node({"scene_repr": 10, "safety": 0}, reference_rows=200)
    assert sparse["scene_repr"] == pytest.approx(0.05)
    assert sparse["safety"] == 0.0
    dense = trace_density_by_node({node: 400 for node in VALID_LOOP_NODES}, reference_rows=200)
    assert all(v == 1.0 for v in dense.values())


@pytest.mark.parametrize(
    "rho_k, rho_l, expected",
    [(0.0, 1.0, 0.0), (0.25, 0.64, 0.4), (1.0, 1.0, 1.0)],
)
def test_field_correction_factor(rho_k, rho_l, expected):
    assert field_correction_factor(rho_k, rho_l) == pytest.approx(expected)


def test_apply_field_correction_attenuates_sparse_nodes():
    coupling = load_module11_coupling()
    densities = trace_density_by_node({"scene_repr": 200, "safety": 20})
    corrected = apply_field_correction(coupling.gamma, coupling.loop_nodes, densities)
    assert np.allclose(np.diag(corrected), 0.0)
    dense_idx = coupling.loop_nodes.index("scene_repr")
    sparse_idx = coupling.loop_nodes.index("safety")
    ssl_idx = coupling.loop_nodes.index("visual_ssl")
    assert corrected[dense_idx, ssl_idx] == 0.0
    assert corrected[dense_idx, sparse_idx] < coupling.gamma[dense_idx, sparse_idx]


def test_transfer_coupling_to_field_returns_factors(sample_coupling):
    counts = {node: 200 if node == "scene_repr" else 20 for node in sample_coupling.loop_nodes}
    gamma_field, densities, factors = transfer_coupling_to_field(sample_coupling, counts)
    assert gamma_field.shape == sample_coupling.gamma.shape
    assert "scene_repr" in densities
    assert factors.shape == gamma_field.shape
    assert np.all(factors >= 0.0)


def test_correction_factor_matrix_is_symmetric(sample_coupling):
    densities = trace_density_by_node({node: 100 for node in sample_coupling.loop_nodes})
    factors = correction_factor_matrix(sample_coupling.loop_nodes, densities)
    assert factors.shape == (12, 12)
    assert np.allclose(factors, factors.T)


def test_dominant_field_couplings_filters_small_entries(sample_coupling):
    gamma = sample_coupling.gamma.copy()
    gamma[0, 1] = 0.5
    gamma[1, 0] = 0.5
    ranked = dominant_field_couplings(gamma, sample_coupling.loop_nodes, min_abs=0.02)
    labels = {(a, b) for a, b, _ in ranked}
    assert ("scene_repr", "visual_ssl") in labels
    assert all(abs(v) >= 0.02 for _, _, v in ranked)


def test_coupling_matrix_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match gamma shape"):
        CouplingMatrix(
            module_ids=(3, 4),
            loop_nodes=("scene_repr", "visual_ssl", "sim_to_real"),
            gamma=np.zeros((2, 2)),
        )


def test_all_valid_loop_nodes_have_module_ids():
    for node in VALID_LOOP_NODES:
        mid = loop_node_to_module_id(node)
        assert module_id_to_loop_node(mid) == node
