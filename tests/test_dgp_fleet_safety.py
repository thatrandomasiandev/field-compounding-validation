"""Tests for federated and safety field DGPs (Module 13–14)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from field_compounding.data.federated_field_dgp import (
    FederatedFieldDGP,
    LOOP_NODE as FED_LOOP,
    summarize_federated_traces,
)
from field_compounding.data.safety_field_dgp import (
    LOOP_NODE as SAFETY_LOOP,
    SafetyFieldDGP,
    summarize_safety_traces,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def _small_federated(**kwargs) -> FederatedFieldDGP:
    defaults = {"n_clients": 3, "n_total": 120, "d_features": 8, "n_classes": 4}
    defaults.update(kwargs)
    return FederatedFieldDGP(**defaults)


def _small_safety(**kwargs) -> SafetyFieldDGP:
    defaults = {
        "n_traj_train": 6,
        "n_traj_test": 3,
        "hj_grid_size": 8,
        "hj_value_steps": 4,
        "n_obstacles": 2,
    }
    defaults.update(kwargs)
    return SafetyFieldDGP(**defaults)


def test_federated_loop_node_and_name() -> None:
    dgp = _small_federated()
    assert dgp.loop_node == FED_LOOP == "federated"
    assert dgp.name == "federated_field_learning"


def test_federated_train_keys() -> None:
    data = _small_federated().generate()
    expected = {"client_X", "client_y", "client_sizes", "label_distributions", "client_latency_weight"}
    assert expected <= set(data.train.keys())


def test_federated_test_keys() -> None:
    data = _small_federated().generate()
    assert data.test["X"].shape[0] == data.test["y"].shape[0]


def test_federated_metadata_includes_inferred_violation() -> None:
    data = _small_federated(violation_severity=0.5).generate()
    assert "inferred_violation_severity" in data.metadata
    assert data.metadata["loop_node"] == "federated"


def test_federated_heterogeneity_increases_with_severity() -> None:
    lo = _small_federated(seed=0, violation_severity=0.1).generate()
    hi = _small_federated(seed=0, violation_severity=0.9).generate()
    assert hi.metadata["label_heterogeneity"] >= lo.metadata["label_heterogeneity"]


def test_federated_deterministic_same_seed() -> None:
    a = _small_federated(seed=7).generate()
    b = _small_federated(seed=7).generate()
    np.testing.assert_array_equal(a.train["client_X"], b.train["client_X"])


def test_federated_client_sizes_match_padding() -> None:
    data = _small_federated().generate()
    for k, size in enumerate(data.train["client_sizes"]):
        assert size > 0
        assert np.all(data.train["client_y"][k, size:] == -1)


def test_safety_loop_node_and_name() -> None:
    dgp = _small_safety()
    assert dgp.loop_node == SAFETY_LOOP == "safety"
    assert dgp.name == "safety_field"


def test_safety_train_keys() -> None:
    data = _small_safety().generate()
    expected = {"states", "actions", "rewards", "costs", "safety_labels"}
    assert expected <= set(data.train.keys())


def test_safety_trajectory_shapes() -> None:
    data = _small_safety().generate()
    assert data.train["states"].shape[0] == 6
    assert data.train["actions"].shape[1] == data.train["states"].shape[1] - 1


def test_safety_cost_increases_with_severity() -> None:
    lo = _small_safety(seed=0, violation_severity=0.05).generate()
    hi = _small_safety(seed=0, violation_severity=0.9).generate()
    assert float(hi.train["costs"].mean()) >= float(lo.train["costs"].mean())


def test_safety_model_error_scales_with_severity() -> None:
    lo = _small_safety(violation_severity=0.0).generate()
    hi = _small_safety(violation_severity=1.0).generate()
    assert hi.metadata["model_error"] > lo.metadata["model_error"]


def test_safety_deterministic_same_seed() -> None:
    a = _small_safety(seed=3).generate()
    b = _small_safety(seed=3).generate()
    np.testing.assert_array_equal(a.train["states"], b.train["states"])


def test_summarize_traces_filters_loop_node() -> None:
    rows = [
        {
            "loop_node": "federated",
            "gnss_drift": 5.0,
            "false_positive_rate": 0.15,
            "cmd_latency_ms": 80.0,
            "battery_pct": 70.0,
            "violation_severity": 0.6,
        },
        {
            "loop_node": "safety",
            "gnss_drift": 8.0,
            "false_positive_rate": 0.2,
            "cmd_latency_ms": 95.0,
            "battery_pct": 55.0,
            "violation_severity": 0.9,
        },
    ]
    fed = summarize_federated_traces(rows)
    assert fed["row_count"] == 1.0
    assert fed["mean_gnss_drift_m"] == pytest.approx(5.0)
    safety = summarize_safety_traces(rows)
    assert safety["mean_violation_severity"] == pytest.approx(0.9)


def test_federated_loads_trace_path(tmp_path: Path) -> None:
    trace = tmp_path / "urc.jsonl"
    row = {
        "timestamp": "2026-01-01T00:00:00Z",
        "loop_node": "federated",
        "violation_severity": 0.55,
        "recovery": False,
        "gnss_drift": 4.1,
        "false_positive_rate": 0.12,
        "cmd_latency_ms": 72.0,
        "battery_pct": 68.0,
    }
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")
    data = _small_federated(trace_path=str(trace)).generate()
    assert data.metadata["trace_row_count"] == 1
    assert data.metadata["gnss_drift_m"] == pytest.approx(4.1)


@pytest.mark.parametrize("config_name,module_id", [("module_13.yaml", 13), ("module_14.yaml", 14)])
def test_module_configs_parse(config_name: str, module_id: int) -> None:
    path = CONFIGS / config_name
    assert path.is_file(), f"missing config: {path}"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["module_id"] == module_id
    assert cfg["dgp"]["class"].startswith("field_compounding.data.")
    assert len(cfg["models"]) >= 2
    assert len(cfg["seeds"]) == 20
    assert len(cfg["violation_levels"]) == 5
    assert "field" in cfg and "trace_path" in cfg["field"]
