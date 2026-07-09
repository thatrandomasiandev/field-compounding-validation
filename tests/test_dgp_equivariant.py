"""Tests for EquivariantFieldDGP (Module 9 / loop node equivariant)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from field_compounding.data.base import BenchmarkData
from field_compounding.data.equivariant_field_dgp import (
    EquivariantFieldDGP,
    load_equivariant_trace_rows,
    summarize_equivariant_traces,
    symmetry_break_scale,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_09_CONFIG = PROJECT_ROOT / "configs" / "module_09.yaml"

EXPECTED_TRAIN_KEYS = {
    "positions", "next_positions", "features", "energies", "torques", "q", "dq",
    "gnss_drift", "false_positive_rate", "field_violation_proxy",
}


def _make_dgp(**kwargs):
    defaults = {"seed": 42, "violation_severity": 0.1, "n_trajectories": 12, "T": 8, "trace_blend": 0.0}
    defaults.update(kwargs)
    return EquivariantFieldDGP(
        seed=defaults.pop("seed"),
        violation_severity=defaults.pop("violation_severity"),
        n_trajectories=defaults.pop("n_trajectories"),
        T=defaults.pop("T"),
        trace_blend=defaults.pop("trace_blend"),
        **defaults,
    )


@pytest.fixture
def sample_trace(tmp_path: Path) -> Path:
    rows = [
        {"loop_node": "visual_ssl", "violation_severity": 0.2, "gnss_drift": 1.5, "false_positive_rate": 0.05},
        {"loop_node": "equivariant", "violation_severity": 0.39, "gnss_drift": 2.4, "false_positive_rate": 0.08},
        {"loop_node": "equivariant", "violation_severity": 0.55, "gnss_drift": 5.2, "false_positive_rate": 0.14},
    ]
    path = tmp_path / "urc_equivariant.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_loop_node_and_name():
    dgp = _make_dgp()
    assert dgp.loop_node == "equivariant"
    assert dgp.name == "equivariant_field_urc"


def test_generates_valid_benchmark_data():
    data = _make_dgp().generate()
    assert isinstance(data, BenchmarkData)
    assert EXPECTED_TRAIN_KEYS <= set(data.train.keys())
    assert data.train["features"].shape[-1] == 16


def test_deterministic_with_same_seed():
    kw = {"seed": 7, "violation_severity": 0.25, "n_trajectories": 10, "T": 6}
    a, b = _make_dgp(**kw).generate(), _make_dgp(**kw).generate()
    np.testing.assert_allclose(a.train["positions"], b.train["positions"])


def test_higher_severity_yields_more_violated_steps():
    lo = _make_dgp(seed=0, violation_severity=0.05, n_trajectories=20, T=10).generate()
    hi = _make_dgp(seed=0, violation_severity=0.95, n_trajectories=20, T=10).generate()
    assert hi.metadata["n_violated"] >= lo.metadata["n_violated"]


def test_higher_gnss_drift_increases_symmetry_break_scale():
    assert symmetry_break_scale(8.0, 0.05, 0.0) > symmetry_break_scale(1.0, 0.05, 0.0)


def test_trace_loader_prefers_equivariant_rows(sample_trace):
    rows = load_equivariant_trace_rows(sample_trace)
    assert len(rows) == 2
    assert all(r["loop_node"] == "equivariant" for r in rows)


def test_trace_stats_summarize_telemetry(sample_trace):
    stats = summarize_equivariant_traces(load_equivariant_trace_rows(sample_trace))
    assert stats["row_count"] == 2.0
    assert stats["mean_gnss_drift_m"] == pytest.approx(3.8)


def test_metadata_includes_field_provenance(sample_trace):
    meta = _make_dgp(trace_path=str(sample_trace), trace_blend=0.5).generate().metadata
    assert meta["loop_node"] == "equivariant"
    assert meta["trace_stats"]["row_count"] == 2.0
    assert "field_gap" in meta


@pytest.mark.parametrize("invalid_severity", [-0.01, 1.05])
def test_rejects_invalid_violation_severity(invalid_severity):
    with pytest.raises(AssertionError):
        EquivariantFieldDGP(seed=0, violation_severity=invalid_severity)


def test_missing_trace_path_raises():
    with pytest.raises(FileNotFoundError):
        _make_dgp(trace_path="observatory/traces/does_not_exist.jsonl").generate()


def test_module_09_config_references_equivariant_field_dgp():
    cfg = yaml.safe_load(MODULE_09_CONFIG.read_text(encoding="utf-8"))
    assert cfg["module_id"] == 9
    assert cfg["loop_node"] == "equivariant"
    assert cfg["dgp"]["class"].endswith("EquivariantFieldDGP")
    assert len(cfg["evaluation"]["seeds"]) == 20
