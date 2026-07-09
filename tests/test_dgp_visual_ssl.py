"""Tests for VisualSSLFieldDGP (Module 4 field replay)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from field_compounding.data.base import BenchmarkData
from field_compounding.data.visual_ssl_field_dgp import (
    VisualSSLFieldDGP,
    augment_sigma_from_fpr,
    feature_drift_scale,
    labeled_fraction_from_field_stress,
    load_visual_ssl_trace_rows,
    save_visual_ssl_trace,
    summarize_visual_ssl_traces,
    _generate_synthetic_visual_ssl_traces,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_04_CONFIG = PROJECT_ROOT / "configs" / "module_04.yaml"

EXPECTED_TRAIN_KEYS = {
    "features",
    "labels",
    "label_mask",
    "augmented_view1",
    "augmented_view2",
    "masked_inputs",
    "masks",
    "temporal_sequences",
    "student_features",
    "teacher_features",
    "point_clouds",
    "point_cloud_labels",
    "point_cloud_label_mask",
}


def _fast_dgp(**kwargs) -> VisualSSLFieldDGP:
    defaults = {
        "seed": 42,
        "n_samples": 80,
        "n_point_clouds": 20,
        "pts_per_cloud": 32,
        "temporal_length": 10,
    }
    defaults.update(kwargs)
    return VisualSSLFieldDGP(**defaults)


def test_loop_node_and_name() -> None:
    dgp = _fast_dgp()
    assert dgp.loop_node == "visual_ssl"
    assert dgp.name == "visual_ssl_field_urc"


def test_generate_returns_benchmark_data() -> None:
    data = _fast_dgp().generate()
    assert isinstance(data, BenchmarkData)
    assert set(data.train.keys()) == EXPECTED_TRAIN_KEYS
    assert set(data.test.keys()) == EXPECTED_TRAIN_KEYS
    assert data.train["features"].shape == (64, 10)
    assert data.test["features"].shape == (16, 10)


def test_deterministic_with_seed() -> None:
    kw = {"violation_severity": 0.2}
    a = _fast_dgp(**kw).generate()
    b = _fast_dgp(**kw).generate()
    np.testing.assert_array_equal(a.train["features"], b.train["features"])
    np.testing.assert_array_equal(a.train["label_mask"], b.train["label_mask"])


def test_higher_violation_yields_sparser_labels() -> None:
    lo = _fast_dgp(violation_severity=0.05).generate()
    hi = _fast_dgp(violation_severity=0.95).generate()
    lo_frac = float(lo.train["label_mask"].mean())
    hi_frac = float(hi.train["label_mask"].mean())
    assert lo_frac > hi_frac
    assert lo.metadata["n_labeled"] > hi.metadata["n_labeled"]


def test_higher_fpr_inflates_augment_sigma() -> None:
    lo_sigma = augment_sigma_from_fpr(0.1, 0.02, trace_coupling=1.0)
    hi_sigma = augment_sigma_from_fpr(0.1, 0.20, trace_coupling=1.0)
    assert hi_sigma > lo_sigma


def test_gnss_drift_increases_feature_drift_scale() -> None:
    lo = feature_drift_scale(0.5, trace_coupling=1.0)
    hi = feature_drift_scale(8.0, trace_coupling=1.0)
    assert hi > lo
    assert feature_drift_scale(0.0) == pytest.approx(0.0)


def test_labeled_fraction_from_field_stress_bounds() -> None:
    n_lo, v_lo = labeled_fraction_from_field_stress(0.0, 0.0, 0.0, n_labeled_max=1000)
    n_hi, v_hi = labeled_fraction_from_field_stress(1.0, 10.0, 1.0, n_labeled_max=1000)
    assert n_lo == 1000
    assert v_lo == pytest.approx(0.0)
    assert n_hi >= 50
    assert v_hi == pytest.approx(1.0)


def test_trace_replay_from_synthetic_file(tmp_path: Path) -> None:
    traces = _generate_synthetic_visual_ssl_traces(n_rows=24, seed=9)
    trace_path = tmp_path / "urc_visual_ssl.jsonl"
    save_visual_ssl_trace(traces, trace_path)

    rows = load_visual_ssl_trace_rows(trace_path)
    stats = summarize_visual_ssl_traces(rows)
    assert stats["row_count"] == 24.0

    data = _fast_dgp(trace_path=str(trace_path), seed=3).generate()
    assert data.metadata["trace_source"] == str(trace_path)
    assert data.metadata["trace_row_count"] == 24
    assert data.metadata["loop_node"] == "visual_ssl"
    assert "inferred_violation_severity" in data.metadata


@pytest.mark.parametrize("invalid_severity", [-0.01, 1.01])
def test_invalid_violation_severity_raises(invalid_severity: float) -> None:
    with pytest.raises(AssertionError):
        VisualSSLFieldDGP(violation_severity=invalid_severity)


def test_module_04_config_points_to_visual_ssl_field_dgp() -> None:
    with MODULE_04_CONFIG.open() as handle:
        config = yaml.safe_load(handle)
    assert config["module_id"] == 4
    assert config["loop_node"] == "visual_ssl"
    assert config["dgp"]["class"].endswith("VisualSSLFieldDGP")
    assert len(config["violation_levels"]) >= 5
    assert len(config["seeds"]) == 20
