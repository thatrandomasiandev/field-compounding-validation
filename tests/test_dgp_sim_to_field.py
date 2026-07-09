"""Tests for sim-to-field domain adaptation DGP (Module 5)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from field_compounding.data.sim_to_field_dgp import (
    LOOP_NODE,
    SimToFieldDGP,
    _TraceRow,
    _save_traces,
    compute_trace_domain_shift,
    generate_synthetic_traces,
    resolve_sim_field_gap,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "module_05.yaml"


def _make_dgp(**overrides) -> SimToFieldDGP:
    params = {
        "seed": 42,
        "violation_severity": 0.25,
        "n_source": 80,
        "n_target": 40,
        "feature_dim": 8,
        "n_classes": 3,
    }
    params.update(overrides)
    return SimToFieldDGP(**params)


def test_loop_node_and_name() -> None:
    dgp = _make_dgp()
    assert dgp.loop_node == "sim_to_real"
    assert dgp.name == "sim_to_field_adaptation"


def test_generate_returns_expected_keys() -> None:
    data = _make_dgp().generate()
    assert set(data.train.keys()) == {"X_source", "y_source", "X_target", "y_target"}
    assert set(data.test.keys()) == {"X_source", "y_source", "X_target", "y_target"}


def test_train_test_split_sizes() -> None:
    data = _make_dgp(n_source=100, n_target=50).generate()
    assert data.train["X_source"].shape[0] == 80
    assert data.test["X_source"].shape[0] == 20
    assert data.train["X_target"].shape[0] == 40
    assert data.test["X_target"].shape[0] == 10


def test_feature_shapes_and_dtypes() -> None:
    data = _make_dgp(feature_dim=12, n_classes=4).generate()
    assert data.train["X_source"].shape[1] == 12
    assert data.train["X_target"].shape[1] == 12
    assert data.train["X_source"].dtype == np.float32
    assert data.train["y_source"].dtype == np.int64


def test_sim_field_gap_metadata_present() -> None:
    data = _make_dgp(violation_severity=0.5).generate()
    meta = data.metadata
    for key in (
        "sim_field_gap",
        "trace_domain_shift",
        "effective_domain_shift",
        "domain_shift_delta",
        "inferred_violation_severity",
    ):
        assert key in meta
    assert meta["sim_field_gap"] == pytest.approx(1.0)
    assert 0.0 <= meta["trace_domain_shift"] <= 1.0
    assert meta["effective_domain_shift"] >= meta["sim_field_gap"]


def test_zero_violation_minimal_shift() -> None:
    low = _make_dgp(violation_severity=0.0, trace_coupling=0.0).generate()
    high = _make_dgp(violation_severity=0.8, trace_coupling=0.0).generate()
    assert low.metadata["effective_domain_shift"] < high.metadata["effective_domain_shift"]
    assert low.metadata["domain_shift_delta"] < high.metadata["domain_shift_delta"]


def test_explicit_sim_field_gap_overrides_severity() -> None:
    data = _make_dgp(violation_severity=0.1, sim_field_gap=1.75, trace_coupling=0.0).generate()
    assert data.metadata["sim_field_gap"] == pytest.approx(1.75)
    assert data.metadata["effective_domain_shift"] == pytest.approx(1.75)


def test_trace_path_increases_effective_shift(tmp_path: Path) -> None:
    entries = generate_synthetic_traces(n_rows=120, seed=7)
    heavy = [
        replace(
            entry,
            gnss_drift=9.5,
            false_positive_rate=0.85,
            cmd_latency_ms=420.0,
        )
        if entry.loop_node == LOOP_NODE
        else entry
        for entry in entries
    ]
    trace_file = tmp_path / "heavy_shift.jsonl"
    _save_traces(heavy, trace_file)
    without = _make_dgp(trace_path=None, trace_coupling=0.0, violation_severity=0.2).generate()
    with_trace = _make_dgp(
        trace_path=str(trace_file),
        trace_coupling=1.0,
        violation_severity=0.2,
        seed=7,
    ).generate()
    assert with_trace.metadata["trace_domain_shift"] > without.metadata["trace_domain_shift"]
    assert (
        with_trace.metadata["effective_domain_shift"]
        > without.metadata["effective_domain_shift"]
    )


def test_deterministic_with_same_seed() -> None:
    a = _make_dgp(seed=99).generate()
    b = _make_dgp(seed=99).generate()
    np.testing.assert_array_equal(a.train["X_source"], b.train["X_source"])
    assert a.metadata["domain_shift_delta"] == pytest.approx(b.metadata["domain_shift_delta"])


def test_rejects_invalid_violation_severity() -> None:
    with pytest.raises(AssertionError):
        SimToFieldDGP(violation_severity=1.5)


def test_compute_trace_domain_shift_bounds() -> None:
    entries = generate_synthetic_traces(n_rows=60, seed=0)
    shift = compute_trace_domain_shift(entries)
    assert 0.0 <= shift <= 1.0


def test_resolve_sim_field_gap_coupling() -> None:
    explicit, effective = resolve_sim_field_gap(
        violation_severity=0.5,
        sim_field_gap=None,
        trace_domain_shift=0.5,
        trace_coupling=1.0,
    )
    assert explicit == pytest.approx(1.0)
    assert effective == pytest.approx(2.0)


def test_module_05_config_loads() -> None:
    with open(CONFIG_PATH) as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["module_id"] == 5
    assert cfg["loop_node"] == LOOP_NODE
    assert cfg["dgp"]["class"].endswith("SimToFieldDGP")
    assert "violation_levels" in cfg
    assert len(cfg["evaluation"]["seeds"]) == 20


def test_trace_row_from_dict_rejects_unknown_loop_node() -> None:
    with pytest.raises(ValueError, match="unknown loop_node"):
        _TraceRow.from_dict(
            {
                "loop_node": "invalid",
                "gnss_drift": 1.0,
                "false_positive_rate": 0.1,
                "cmd_latency_ms": 50.0,
            }
        )
