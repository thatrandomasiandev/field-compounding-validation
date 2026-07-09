"""Tests for neurosymbolic field DGP (Module 12 / Agent 17)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from field_compounding.data.base import BenchmarkData
from field_compounding.data.neurosymbolic_field_dgp import (
    LOOP_NODE,
    NeurosymbolicFieldDGP,
    effective_grounding_noise,
    infer_violation_from_telemetry,
    load_neurosymbolic_trace_rows,
    summarize_trace_telemetry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_STUB = REPO_ROOT / "observatory/traces/urc_neurosymbolic_stub.jsonl"
MODULE_CONFIG = REPO_ROOT / "configs/module_12.yaml"
EXPECTED_TRAIN_KEYS = {"object_positions", "facts", "atom_targets", "task_ids", "gnss_drift_m", "false_positive_rate"}


def _make_dgp(**overrides) -> NeurosymbolicFieldDGP:
    params = {"seed": 42, "violation_severity": 0.1, "trace_path": str(TRACE_STUB), "n_objects": 12, "n_tasks": 8, "n_atoms": 4, "n_formulas": 3}
    params.update(overrides)
    return NeurosymbolicFieldDGP(**params)


def _assert_valid_benchmark(data: BenchmarkData) -> None:
    assert isinstance(data, BenchmarkData)
    assert EXPECTED_TRAIN_KEYS.issubset(data.train.keys())
    assert EXPECTED_TRAIN_KEYS.issubset(data.test.keys())
    for split in (data.train, data.test):
        for arr in split.values():
            assert isinstance(arr, np.ndarray) and arr.size > 0


def test_generates_valid_benchmark_data() -> None:
    _assert_valid_benchmark(_make_dgp().generate())


def test_loop_node_is_neurosymbolic() -> None:
    dgp = _make_dgp()
    assert dgp.loop_node == LOOP_NODE
    assert _make_dgp().generate().metadata["loop_node"] == "neurosymbolic"


def test_name_is_nonempty_string() -> None:
    assert _make_dgp().name


def test_metadata_records_violation_and_grounding_noise() -> None:
    data = _make_dgp(violation_severity=0.35).generate()
    assert data.metadata["violation_severity"] == 0.35
    assert 0.0 <= float(data.metadata["grounding_noise_frac"]) <= 1.0


def test_seed_reproducibility() -> None:
    a = _make_dgp(seed=7).generate().train
    b = _make_dgp(seed=7).generate().train
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("invalid_severity", [-0.1, 1.01])
def test_rejects_invalid_violation_severity(invalid_severity: float) -> None:
    with pytest.raises(AssertionError):
        NeurosymbolicFieldDGP(seed=42, violation_severity=invalid_severity)


def test_higher_severity_changes_atom_targets() -> None:
    lo = _make_dgp(seed=0, violation_severity=0.05).generate().train["atom_targets"]
    hi = _make_dgp(seed=0, violation_severity=0.95).generate().train["atom_targets"]
    assert not np.array_equal(lo, hi)


def test_inferred_violation_severity_in_metadata() -> None:
    inferred = float(_make_dgp().generate().metadata["inferred_violation_severity"])
    assert 0.0 <= inferred <= 1.0


def test_trace_loader_filters_loop_node() -> None:
    rows = load_neurosymbolic_trace_rows(str(TRACE_STUB))
    assert len(rows) == 3
    assert all(row["loop_node"] == "neurosymbolic" for row in rows)


def test_telemetry_helpers_are_bounded() -> None:
    telemetry = summarize_trace_telemetry(load_neurosymbolic_trace_rows(str(TRACE_STUB)))
    inferred = infer_violation_from_telemetry(telemetry["mean_gnss_drift_m"], telemetry["mean_false_positive_rate"])
    noise = effective_grounding_noise(0.5, telemetry)
    assert 0.0 <= inferred <= 1.0
    assert 0.0 <= noise <= 1.0


def test_module_12_yaml_references_dgp() -> None:
    config = yaml.safe_load(MODULE_CONFIG.read_text(encoding="utf-8"))
    assert config["module_id"] == 12
    assert config["loop_node"] == "neurosymbolic"
    assert config["dgp"]["class"].endswith("NeurosymbolicFieldDGP")
    assert len(config["violation_levels"]) == 5
