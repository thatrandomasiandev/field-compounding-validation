"""Tests for UncertaintyFieldDGP (Module 11 / loop node uncertainty)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from field_compounding.data.base import BenchmarkData
from field_compounding.data.uncertainty_field_dgp import (
    LOOP_NODE,
    UncertaintyFieldDGP,
    _load_uncertainty_trace_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "module_11.yaml"
TRACE_PATH = REPO_ROOT / "observatory" / "traces" / "urc_uncertainty_stub.jsonl"

TRAIN_KEYS = {
    "X_regression", "Y_regression", "noise_levels", "X_classification", "labels",
    "labels_clean", "ood_mask", "X_calibration", "Y_calibration", "X_cls_calibration", "labels_calibration",
}
TEST_KEYS = {
    "X_regression", "Y_regression", "noise_levels", "X_classification", "labels",
    "labels_clean", "ood_mask", "X_val_regression", "Y_val_regression", "X_val_classification", "labels_val",
}


def _make_dgp(**kwargs) -> UncertaintyFieldDGP:
    defaults: dict[str, object] = {"seed": 42, "violation_severity": 0.1, "n_samples": 200}
    defaults.update(kwargs)
    return UncertaintyFieldDGP(**defaults)


def test_name_and_loop_node() -> None:
    dgp = _make_dgp()
    assert dgp.name == "uncertainty_field"
    assert dgp.loop_node == LOOP_NODE


def test_generates_benchmark_data() -> None:
    data = _make_dgp().generate()
    assert isinstance(data, BenchmarkData)
    assert data.train and data.test and data.metadata


def test_train_and_test_keys() -> None:
    data = _make_dgp().generate()
    assert TRAIN_KEYS <= set(data.train.keys())
    assert TEST_KEYS <= set(data.test.keys())


def test_reproducible_with_seed() -> None:
    a = _make_dgp(seed=7).generate()
    b = _make_dgp(seed=7).generate()
    assert np.allclose(a.train["Y_regression"], b.train["Y_regression"])
    assert np.array_equal(a.train["labels"], b.train["labels"])


def test_label_noise_increases_with_severity() -> None:
    lo = _make_dgp(seed=0, violation_severity=0.05, n_samples=300).generate()
    hi = _make_dgp(seed=0, violation_severity=0.9, n_samples=300).generate()
    lo_flip = float((lo.train["labels"] != lo.train["labels_clean"]).mean())
    hi_flip = float((hi.train["labels"] != hi.train["labels_clean"]).mean())
    assert hi_flip > lo_flip


def test_heteroscedastic_noise_grows_with_input_norm() -> None:
    data = _make_dgp(n_samples=400).generate()
    norms = np.linalg.norm(data.train["X_regression"], axis=-1)
    noise = data.train["noise_levels"]
    assert float(np.corrcoef(norms, noise)[0, 1]) > 0.5


def test_ood_mask_marks_holdout_classes() -> None:
    data = _make_dgp(n_classes=10, n_ood_classes=2, n_samples=500).generate()
    labels_clean = data.test["labels_clean"]
    ood_mask = data.test["ood_mask"]
    assert np.all(labels_clean[ood_mask] >= 8)
    assert np.all(labels_clean[~ood_mask] < 8)


def test_trace_coupling_inflates_noise_and_label_corruption() -> None:
    base_data = _make_dgp(seed=1, trace_path=None, violation_severity=0.2, n_samples=300).generate()
    trace_data = _make_dgp(seed=1, trace_path=str(TRACE_PATH), trace_coupling=1.0, violation_severity=0.2, n_samples=300).generate()
    assert trace_data.metadata["heteroscedastic_multiplier"] > 1.0
    assert trace_data.metadata["p_noise"] > base_data.metadata["p_noise"]
    assert trace_data.train["noise_levels"].mean() > base_data.train["noise_levels"].mean()


def test_trace_calibration_metadata() -> None:
    data = _make_dgp(trace_path=str(TRACE_PATH)).generate()
    assert data.metadata["source"] == "trace_calibrated"
    assert data.metadata["trace_rows_used"] == 5
    assert data.metadata["mean_gnss_drift_m"] > 0.0
    assert 0.0 <= data.metadata["inferred_violation_severity"] <= 1.0


def test_load_trace_stats_ignores_other_loop_nodes(tmp_path: Path) -> None:
    rows = [
        {"timestamp": "2025-01-01T00:00:00Z", "loop_node": "uncertainty", "violation_severity": 0.4, "recovery": False, "gnss_drift": 4.0, "false_positive_rate": 0.2},
        {"timestamp": "2025-01-01T00:01:00Z", "loop_node": "safety", "violation_severity": 0.9, "recovery": False, "gnss_drift": 99.0, "false_positive_rate": 0.9},
    ]
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    stats = _load_uncertainty_trace_stats(path)
    assert stats.row_count == 1
    assert stats.mean_gnss_drift_m == pytest.approx(4.0)


def test_module_11_config_references_uncertainty_dgp() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["module_id"] == 11
    assert cfg["loop_node"] == "uncertainty"
    assert cfg["dgp"]["class"].endswith("UncertaintyFieldDGP")
    assert len(cfg["seeds"]) == 20
    assert len(cfg["violation_levels"]) == 5
