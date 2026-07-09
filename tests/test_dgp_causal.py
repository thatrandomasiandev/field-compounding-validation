"""Tests for the field-backed causal scene understanding DGP (Module 7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from field_compounding.data.base import BenchmarkData
from field_compounding.data.causal_field_dgp import CausalFieldDGP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_07_CONFIG = PROJECT_ROOT / "configs" / "module_07.yaml"

CAUSAL_TRAIN_KEYS = {
    "X",
    "T",
    "Y",
    "Z",
    "propensity",
    "cate_true",
    "gnss_drift",
    "false_positive_rate",
    "field_violation_proxy",
}


def _make_dgp(**kwargs) -> CausalFieldDGP:
    defaults = dict(seed=42, violation_severity=0.1, n_train=80, n_test=20, trace_path=None)
    defaults.update(kwargs)
    return CausalFieldDGP(**defaults)


def _assert_valid_benchmark(data: BenchmarkData) -> None:
    assert CAUSAL_TRAIN_KEYS.issubset(data.train.keys())
    assert CAUSAL_TRAIN_KEYS.issubset(data.test.keys())


def test_generates_valid_benchmark_data() -> None:
    _assert_valid_benchmark(_make_dgp().generate())


def test_loop_node_is_causal() -> None:
    dgp = _make_dgp()
    assert dgp.loop_node == "causal"
    assert dgp.generate().metadata["loop_node"] == "causal"


def test_name_is_nonempty() -> None:
    assert _make_dgp().name == "field_causal_scene"


def test_metadata_records_violation_severity() -> None:
    data = _make_dgp(violation_severity=0.65).generate()
    assert abs(data.metadata["violation_severity"] - 0.65) < 1e-9


def test_metadata_records_gamma_and_inferred_proxy() -> None:
    data = _make_dgp(violation_severity=0.7).generate()
    assert abs(data.metadata["gamma"] - 0.7) < 1e-9
    assert 0.0 <= data.metadata["inferred_violation_severity"] <= 1.0


def test_train_shapes_match_metadata() -> None:
    data = _make_dgp(n_train=64, n_test=16).generate()
    assert data.train["X"].shape == (64, data.metadata["obs_dim"])
    assert data.test["X"].shape[0] == data.metadata["n_test"]


@pytest.mark.parametrize("invalid_severity", [-0.05, 1.01])
def test_rejects_invalid_violation_severity(invalid_severity: float) -> None:
    with pytest.raises(AssertionError):
        CausalFieldDGP(seed=0, violation_severity=invalid_severity)


def test_propensity_more_extreme_at_high_gamma() -> None:
    lo = _make_dgp(seed=0, violation_severity=0.0, n_train=400).generate()
    hi = _make_dgp(seed=0, violation_severity=1.0, n_train=400).generate()
    assert float(np.std(hi.train["propensity"])) > float(np.std(lo.train["propensity"]))


def test_deterministic_with_same_seed() -> None:
    a = _make_dgp(seed=7, n_train=40, n_test=10).generate()
    b = _make_dgp(seed=7, n_train=40, n_test=10).generate()
    np.testing.assert_allclose(a.train["X"], b.train["X"])


def test_replays_external_trace_file(tmp_path: Path) -> None:
    trace_path = tmp_path / "causal.jsonl"
    rows = [
        {
            "timestamp": "2025-05-18T14:00:00Z",
            "loop_node": "causal",
            "violation_severity": 0.4,
            "recovery": True,
            "gnss_drift": 3.2,
            "false_positive_rate": 0.15,
        },
        {
            "timestamp": "2025-05-18T14:03:00Z",
            "loop_node": "causal",
            "violation_severity": 0.8,
            "recovery": False,
            "gnss_drift": 6.5,
            "false_positive_rate": 0.35,
        },
    ]
    with trace_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    data = _make_dgp(seed=3, trace_path=str(trace_path), n_train=30).generate()
    assert data.metadata["trace_rows_used"] == 2


def test_module_07_config_points_to_causal_field_dgp() -> None:
    with MODULE_07_CONFIG.open() as handle:
        config = yaml.safe_load(handle)
    assert config["module_id"] == 7
    assert config["dgp"]["class"].endswith("CausalFieldDGP")
