"""Tests for Condition D mitigation and decoupling stack (Agent 20)."""

from __future__ import annotations

import numpy as np
import pytest

from field_compounding.models.condition_d import (
    CONDITION_C_VIOLATIONS,
    ConditionDSpec,
    build_compound_condition_entry,
    compare_conditions_c_d,
    compound_excess_bound,
    condition_c_spec,
    default_condition_d_spec,
    latency_sweep_bounds,
    predicted_bound_with_mitigation,
    violation_vector,
)
from field_compounding.models.decoupling_stack import (
    LOOP_NODE_ORDER,
    MODULE_ID_TO_LOOP_NODE,
    DecouplingConfig,
    DecouplingStack,
)


@pytest.fixture
def toy_psi_gamma():
    psi = np.linspace(-0.2, 0.1, 12)
    gamma = np.full((12, 12), 0.05)
    np.fill_diagonal(gamma, 0.0)
    gamma[0, 1] = gamma[1, 0] = 0.3
    return psi, gamma


def test_module_id_loop_node_mapping():
    assert len(MODULE_ID_TO_LOOP_NODE) == 12
    assert MODULE_ID_TO_LOOP_NODE[3] == "scene_repr"
    assert MODULE_ID_TO_LOOP_NODE[14] == "safety"
    assert tuple(MODULE_ID_TO_LOOP_NODE.values()) == LOOP_NODE_ORDER


def test_condition_c_violations_profile():
    assert len(CONDITION_C_VIOLATIONS) == 12
    assert CONDITION_C_VIOLATIONS[5] == 0.7
    assert CONDITION_C_VIOLATIONS[13] == 0.8
    assert all(0.0 <= v <= 1.0 for v in CONDITION_C_VIOLATIONS.values())


def test_default_condition_d_spec():
    spec = default_condition_d_spec(cmd_latency_ms=75.0)
    assert spec.mitigation_enabled
    assert spec.cmd_latency_ms == 75.0
    assert spec.violation_severities == CONDITION_C_VIOLATIONS
    v = violation_vector(spec)
    assert v.shape == (12,)
    assert v[2] == pytest.approx(0.7)


def test_condition_d_spec_rejects_invalid_severity():
    with pytest.raises(ValueError, match="violation severity"):
        ConditionDSpec(
            description="bad",
            violation_severities={3: 1.5},
        )


def test_compound_excess_bound_quadratic_term():
    v = np.array([0.5, 0.5])
    psi = np.array([0.0, 0.0])
    gamma = np.array([[0.0, 0.4], [0.4, 0.0]])
    bound = compound_excess_bound(v, psi, gamma)
    assert bound == pytest.approx(0.4 * 0.5 * 0.5)


def test_decoupling_stack_forward_shape(rng):
    stack = DecouplingStack(DecouplingConfig(feature_dim=16, latent_dim=4))
    x = rng.normal(size=(8, 16))
    out = stack.forward(x, cmd_latency_ms=40.0)
    assert isinstance(out.features, np.ndarray)
    assert out.features.shape == (8, 4)
    assert 0.0 < out.gamma_scale <= 1.0


def test_effective_gamma_scale_decreases_with_latency():
    stack = DecouplingStack(DecouplingConfig(feature_dim=16, latent_dim=4))
    low = stack.effective_gamma_scale(cmd_latency_ms=0.0)
    high = stack.effective_gamma_scale(cmd_latency_ms=500.0)
    assert high < low


def test_attenuate_coupling_matrix_preserves_diagonal(toy_psi_gamma):
    _, gamma = toy_psi_gamma
    stack = DecouplingStack()
    mit = stack.attenuate_coupling_matrix(gamma, cmd_latency_ms=50.0)
    assert np.allclose(np.diag(mit), 0.0)
    assert np.max(np.abs(mit)) <= np.max(np.abs(gamma))


def test_mitigation_reduces_coupling_bound(toy_psi_gamma):
    psi, gamma = toy_psi_gamma
    spec = default_condition_d_spec()
    result = predicted_bound_with_mitigation(spec, psi, gamma)
    assert result["mitigation_delta"] >= 0.0
    assert result["mitigated_bound"] <= result["baseline_bound"]


def test_condition_c_has_no_mitigation_delta(toy_psi_gamma):
    psi, gamma = toy_psi_gamma
    spec = condition_c_spec()
    result = predicted_bound_with_mitigation(spec, psi, gamma)
    assert result["gamma_scale"] == 1.0
    assert result["mitigation_delta"] == 0.0


def test_compare_conditions_c_d(toy_psi_gamma):
    psi, gamma = toy_psi_gamma
    report = compare_conditions_c_d(psi, gamma, cmd_latency_ms=60.0)
    assert report["condition_d_bound"] <= report["condition_c_bound"]
    assert report["bound_reduction"] >= 0.0
    assert report["cmd_latency_ms"] == 60.0


def test_build_compound_condition_entry():
    entry = build_compound_condition_entry()
    assert "D" in entry
    block = entry["D"]
    assert block["mitigation_enabled"] is True
    assert "violation_severities" in block
    assert block["loop_nodes"]["14"] == "safety"


def test_latency_sweep_monotonic_gamma_scale(toy_psi_gamma):
    psi, gamma = toy_psi_gamma
    rows = latency_sweep_bounds(psi, gamma, latency_grid_ms=np.array([0.0, 100.0, 300.0]))
    scales = [row["gamma_scale"] for row in rows]
    assert scales[0] >= scales[1] >= scales[2]


def test_bottleneck_fit_and_forward(rng):
    stack = DecouplingStack(DecouplingConfig(feature_dim=12, latent_dim=3))
    x = rng.normal(size=(32, 12)).astype(np.float32)
    loss = stack.fit_bottleneck(x, epochs=40)
    assert loss >= 0.0
    out = stack.forward(x[:4])
    assert out.features.shape[0] == 4


def test_equivariant_project_preserves_batch(rng):
    stack = DecouplingStack(DecouplingConfig(feature_dim=20, latent_dim=5, equivariant_groups=5))
    x = rng.normal(size=(6, 20))
    y = stack.equivariant_project(x)
    assert y.shape == (6, 20)


def test_decoupling_config_validation():
    with pytest.raises(ValueError, match="latent_dim"):
        DecouplingStack(DecouplingConfig(feature_dim=8, latent_dim=8))
