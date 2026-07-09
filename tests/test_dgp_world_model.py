"""Tests for WorldModelFieldDGP (Module 8 field replay)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from field_compounding.data.base import BenchmarkData
from field_compounding.data.world_model_field_dgp import (
    WorldModelFieldDGP,
    inferred_violation_from_telemetry,
    load_urc_trace_rows,
    sigma_m_from_violation,
    summarize_world_model_traces,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_STUB = REPO_ROOT / "observatory/traces/world_model_synthetic.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/module_08.yaml"

EXPECTED_KEYS = {
    "states",
    "actions",
    "observations",
    "next_states",
    "next_states_clean",
}


def _fast_dgp(**kwargs) -> WorldModelFieldDGP:
    defaults = {"seed": 42, "n_trajectories": 8, "horizon": 6}
    defaults.update(kwargs)
    return WorldModelFieldDGP(**defaults)


def test_loop_node_and_name() -> None:
    dgp = _fast_dgp()
    assert dgp.loop_node == "world_model"
    assert dgp.name == "world_model_field_planar_arm"


def test_generate_returns_benchmark_data() -> None:
    data = _fast_dgp(trace_path=str(TRACE_STUB)).generate()
    assert isinstance(data, BenchmarkData)
    assert set(data.train.keys()) == EXPECTED_KEYS
    assert set(data.test.keys()) == EXPECTED_KEYS
    assert data.train["states"].shape == (6, 6, 6)
    assert data.test["states"].shape[0] == 2
    assert data.train["observations"].shape[-1] == 22


def test_deterministic_with_seed() -> None:
    kw = {"trace_path": str(TRACE_STUB), "violation_severity": 0.3}
    a = _fast_dgp(**kw).generate()
    b = _fast_dgp(**kw).generate()
    np.testing.assert_array_equal(a.train["states"], b.train["states"])
    np.testing.assert_array_equal(a.train["next_states"], b.train["next_states"])


def test_higher_sigma_m_widens_noisy_clean_gap() -> None:
    lo = _fast_dgp(sigma_m=0.01, trace_path=None).generate()
    hi = _fast_dgp(sigma_m=0.5, trace_path=None).generate()
    lo_gap = float(np.mean((lo.test["next_states"] - lo.test["next_states_clean"]) ** 2))
    hi_gap = float(np.mean((hi.test["next_states"] - hi.test["next_states_clean"]) ** 2))
    assert hi_gap > lo_gap
    assert hi.metadata["sigma_m"] > lo.metadata["sigma_m"]


def test_higher_latency_increases_inferred_violation() -> None:
    lo = _fast_dgp(cmd_latency_ms=20.0, battery_pct=95.0, gnss_drift_m=0.5, trace_path=None)
    hi = _fast_dgp(cmd_latency_ms=350.0, battery_pct=95.0, gnss_drift_m=0.5, trace_path=None)
    lo_meta = lo.generate().metadata
    hi_meta = hi.generate().metadata
    assert hi_meta["inferred_violation_severity"] > lo_meta["inferred_violation_severity"]
    assert hi_meta["latency_steps"] >= lo_meta["latency_steps"]


def test_lower_battery_reduces_torque_scale() -> None:
    hi_batt = _fast_dgp(battery_pct=95.0, trace_path=None).generate().metadata
    lo_batt = _fast_dgp(battery_pct=20.0, trace_path=None).generate().metadata
    assert hi_batt["torque_scale"] > lo_batt["torque_scale"]


def test_trace_stub_loads_world_model_stats() -> None:
    rows = load_urc_trace_rows(TRACE_STUB)
    stats = summarize_world_model_traces(rows)
    assert stats["row_count"] == 6.0
    assert stats["mean_cmd_latency_ms"] > 100.0
    assert stats["mean_gnss_drift_m"] > 3.0


def test_metadata_includes_field_provenance() -> None:
    data = _fast_dgp(trace_path=str(TRACE_STUB)).generate()
    meta = data.metadata
    assert meta["loop_node"] == "world_model"
    assert meta["field_domain"] == "urc_outdoor"
    assert meta["cmd_latency_ms"] > 0.0
    assert meta["trace_row_count"] == 6
    assert "inferred_violation_severity" in meta
    assert meta["causal_graph"].shape == (6, 9)


def test_inferred_violation_from_telemetry_bounds() -> None:
    lo = inferred_violation_from_telemetry(0.0, 0.0, 100.0)
    hi = inferred_violation_from_telemetry(8.0, 300.0, 15.0)
    assert lo == pytest.approx(0.0)
    assert 0.0 < hi <= 1.0
    assert hi > lo


def test_sigma_m_from_violation_monotone() -> None:
    assert sigma_m_from_violation(0.0, 0.0) < sigma_m_from_violation(1.0, 0.0)
    assert sigma_m_from_violation(0.5, 0.0) < sigma_m_from_violation(0.5, 1.0)


def test_invalid_violation_severity_raises() -> None:
    with pytest.raises(AssertionError):
        WorldModelFieldDGP(violation_severity=1.5)


def test_missing_trace_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        _fast_dgp(trace_path="observatory/traces/does_not_exist.jsonl")


def test_module_08_config_references_field_dgp() -> None:
    with open(CONFIG_PATH) as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["module_id"] == 8
    assert cfg["loop_node"] == "world_model"
    assert "WorldModelFieldDGP" in cfg["dgp"]["class"]
    assert cfg["dgp"]["params"]["trace_path"] == "observatory/traces/world_model_synthetic.jsonl"
    assert len(cfg["violation_levels"]) >= 5
