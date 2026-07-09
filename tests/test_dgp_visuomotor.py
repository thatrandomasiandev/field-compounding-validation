"""Tests for the visuomotor field DGP (Module 12 / Agent 11)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from field_compounding.data.base import BenchmarkData
from field_compounding.data.visuomotor_field_dgp import LOOP_NODE, VisuomotorFieldDGP, _load_visuomotor_trace_stats
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "module_06.yaml"
TRACE_PATH = REPO_ROOT / "observatory/traces/visuomotor_synthetic.jsonl"
TRAIN_KEYS = {"observations", "actions", "rewards", "modes", "cmd_latency_ms"}

def _fast_dgp(**kwargs):
    defaults = {"seed": 42, "n_trajectories": 8, "horizon": 5, "n_modes": 2}
    defaults.update(kwargs)
    return VisuomotorFieldDGP(**defaults)

def test_name_and_loop_node():
    dgp = _fast_dgp()
    assert dgp.name == "visuomotor_field" and dgp.loop_node == "visuomotor"

def test_generates_benchmark_data():
    data = _fast_dgp().generate()
    assert isinstance(data, BenchmarkData) and data.train and data.test

def test_train_test_keys_and_shapes():
    dgp = _fast_dgp(); data = dgp.generate()
    assert TRAIN_KEYS <= set(data.train.keys())
    n_train = int(0.8 * dgp.n_trajectories * dgp.horizon)
    assert data.train["observations"].shape == (n_train, dgp.OBS_DIM)

def test_deterministic_with_seed():
    a = _fast_dgp(seed=7).generate(); b = _fast_dgp(seed=7).generate()
    assert np.array_equal(a.train["observations"], b.train["observations"])

def test_beta_from_violation_severity():
    lo = VisuomotorFieldDGP(seed=0, violation_severity=0.0, n_trajectories=4, horizon=4)
    hi = VisuomotorFieldDGP(seed=0, violation_severity=1.0, n_trajectories=4, horizon=4)
    assert lo.beta > hi.beta and lo.beta == pytest.approx(1.0) and hi.beta == pytest.approx(0.3)

def test_data_differs_across_beta():
    lo = VisuomotorFieldDGP(seed=0, beta=0.9, n_trajectories=12, horizon=6).generate()
    hi = VisuomotorFieldDGP(seed=0, beta=0.35, n_trajectories=12, horizon=6).generate()
    assert not np.array_equal(lo.train["modes"], hi.train["modes"]) or not np.array_equal(lo.train["rewards"], hi.train["rewards"])

def test_field_gap_increases_observation_noise():
    low = _fast_dgp(field_gap=0.0, violation_severity=0.0).generate()
    high = _fast_dgp(field_gap=1.0, violation_severity=0.8).generate()
    assert high.metadata["field_gap"] > low.metadata["field_gap"]
    assert high.metadata["obs_noise_scale"] > low.metadata["obs_noise_scale"]

def test_trace_calibration_metadata():
    data = _fast_dgp(trace_path=str(TRACE_PATH)).generate()
    assert data.metadata["source"] == "trace_calibrated" and data.metadata["trace_rows_used"] == 5

def test_trace_loader_aggregates_visuomotor_rows(tmp_path: Path):
    trace_file = tmp_path / "mixed.jsonl"
    rows = [{"loop_node": "scene_repr", "violation_severity": 0.2, "gnss_drift": 1.0, "false_positive_rate": 0.05}, {"loop_node": "visuomotor", "violation_severity": 0.4, "gnss_drift": 2.0, "false_positive_rate": 0.1, "cmd_latency_ms": 40.0}, {"loop_node": "visuomotor", "violation_severity": 0.6, "gnss_drift": 4.0, "false_positive_rate": 0.2}]
    trace_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    stats = _load_visuomotor_trace_stats(trace_file)
    assert stats.row_count == 2 and stats.mean_violation_severity == pytest.approx(0.5)

def test_module_06_config_structure():
    with open(CONFIG_PATH) as handle:
        config = yaml.safe_load(handle)
    assert config["module_id"] == 6 and config["loop_node"] == "visuomotor"
    assert len(config["models"]) >= 4 and len(config["seeds"]) >= 20
