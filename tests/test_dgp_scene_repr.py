"""Tests for SceneReprFieldDGP (Module 3 field replay)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from field_compounding.data.base import BenchmarkData
from field_compounding.data.scene_repr_field_dgp import (
    SceneReprFieldDGP,
    load_urc_trace_rows,
    summarize_scene_repr_traces,
    views_from_gnss_drift,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_STUB = REPO_ROOT / "observatory/traces/urc_scene_repr_stub.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/module_03.yaml"

EXPECTED_TRAIN_KEYS = {
    "images",
    "depths",
    "camera_poses",
    "object_positions",
    "object_features",
    "grasp_candidates",
    "grasp_labels",
}


def _fast_dgp(**kwargs) -> SceneReprFieldDGP:
    defaults = {"seed": 42, "n_scenes_train": 2, "n_scenes_test": 1, "resolution": 16}
    defaults.update(kwargs)
    return SceneReprFieldDGP(**defaults)


def test_loop_node_and_name() -> None:
    dgp = _fast_dgp()
    assert dgp.loop_node == "scene_repr"
    assert dgp.name == "scene_repr_field_urc"


def test_generate_returns_benchmark_data() -> None:
    data = _fast_dgp(trace_path=str(TRACE_STUB)).generate()
    assert isinstance(data, BenchmarkData)
    assert set(data.train.keys()) == EXPECTED_TRAIN_KEYS
    assert set(data.test.keys()) == EXPECTED_TRAIN_KEYS
    assert data.train["images"].ndim == 5
    assert data.test["images"].shape[0] == 1


def test_deterministic_with_seed() -> None:
    kw = {"trace_path": str(TRACE_STUB), "violation_severity": 0.2}
    a = _fast_dgp(**kw).generate()
    b = _fast_dgp(**kw).generate()
    np.testing.assert_array_equal(a.train["images"], b.train["images"])
    np.testing.assert_array_equal(a.train["depths"], b.train["depths"])


def test_higher_violation_yields_fewer_views() -> None:
    lo = _fast_dgp(violation_severity=0.0, gnss_drift_m=0.0, trace_path=None).generate()
    hi = _fast_dgp(violation_severity=0.9, gnss_drift_m=0.0, trace_path=None).generate()
    assert lo.metadata["n_views"] > hi.metadata["n_views"]
    assert lo.metadata["effective_violation_severity"] < hi.metadata[
        "effective_violation_severity"
    ]


def test_higher_gnss_drift_yields_fewer_views() -> None:
    lo = _fast_dgp(violation_severity=0.0, gnss_drift_m=1.0, trace_path=None).generate()
    hi = _fast_dgp(violation_severity=0.0, gnss_drift_m=7.5, trace_path=None).generate()
    assert lo.metadata["n_views"] > hi.metadata["n_views"]


def test_trace_stub_loads_scene_repr_stats() -> None:
    rows = load_urc_trace_rows(TRACE_STUB)
    stats = summarize_scene_repr_traces(rows)
    assert stats["row_count"] == 6.0
    assert stats["mean_gnss_drift_m"] > 3.0
    assert stats["mean_violation_severity"] > 0.0


def test_metadata_includes_field_provenance() -> None:
    data = _fast_dgp(trace_path=str(TRACE_STUB)).generate()
    meta = data.metadata
    assert meta["loop_node"] == "scene_repr"
    assert meta["field_domain"] == "urc_outdoor"
    assert meta["gnss_drift_m"] > 0.0
    assert meta["trace_row_count"] == 6
    assert "inferred_violation_severity" in meta
    assert meta["v_max"] == 32


def test_views_from_gnss_drift_bounds() -> None:
    n_lo, v_lo = views_from_gnss_drift(0.0, 0.0)
    n_hi, v_hi = views_from_gnss_drift(8.0, 0.0)
    assert n_lo == 32
    assert v_lo == pytest.approx(0.0)
    assert n_hi == 4
    assert v_hi == pytest.approx(0.875)
    n_mid, _ = views_from_gnss_drift(4.0, 0.5)
    assert 4 <= n_mid <= 32


def test_invalid_violation_severity_raises() -> None:
    with pytest.raises(AssertionError):
        SceneReprFieldDGP(violation_severity=1.5)


def test_missing_trace_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        _fast_dgp(trace_path="observatory/traces/does_not_exist.jsonl")


def test_module_03_config_references_field_dgp() -> None:
    with open(CONFIG_PATH) as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["module_id"] == 3
    assert cfg["loop_node"] == "scene_repr"
    assert "SceneReprFieldDGP" in cfg["dgp"]["class"]
    assert cfg["dgp"]["params"]["trace_path"] == "observatory/traces/urc_scene_repr_stub.jsonl"
    assert len(cfg["violation_levels"]) >= 5
