"""Tests for SceneGraphFieldDGP (Module 10 / loop node scene_graph)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from field_compounding.data.base import BenchmarkData
from field_compounding.data.scene_graph_field_dgp import SceneGraphFieldDGP

EXPECTED_TRAIN_KEYS = {"adjacency_matrix", "node_features", "edge_labels", "positive_edges", "negative_edges", "temporal_snapshots"}

def _make_dgp(seed=42, violation_severity=0.1, trace_path=None, **kwargs):
    defaults = {"n_nodes": 24, "T": 4, "feature_dim": 16}
    defaults.update(kwargs)
    return SceneGraphFieldDGP(seed=seed, violation_severity=violation_severity, trace_path=trace_path, **defaults)

@pytest.fixture
def sample_trace(tmp_path: Path) -> Path:
    rows = [
        {"false_positive_rate": 0.05, "gnss_drift": 2.0, "loop_node": "visual_ssl", "violation_severity": 0.2},
        {"false_positive_rate": 0.20, "gnss_drift": 8.0, "loop_node": "scene_graph", "violation_severity": 0.75},
        {"false_positive_rate": 0.10, "gnss_drift": 4.0, "loop_node": "scene_graph", "violation_severity": 0.55},
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path

def test_loop_node_and_name():
    dgp = _make_dgp()
    assert dgp.loop_node == "scene_graph"
    assert dgp.name == "scene_graph_field"

def test_generates_expected_train_keys():
    data = _make_dgp().generate()
    assert isinstance(data, BenchmarkData)
    assert EXPECTED_TRAIN_KEYS <= set(data.train.keys())

def test_adjacency_matches_n_nodes():
    data = _make_dgp(n_nodes=30).generate()
    adj = data.train["adjacency_matrix"]
    assert adj.shape == (30, 30)
    assert data.metadata["n_nodes"] == 30
    assert np.allclose(adj, adj.T)

def test_deterministic_with_same_seed():
    a = _make_dgp(seed=7, violation_severity=0.2).generate()
    b = _make_dgp(seed=7, violation_severity=0.2).generate()
    assert np.array_equal(a.train["adjacency_matrix"], b.train["adjacency_matrix"])

def test_wl_collisions_increase_with_severity():
    lo = _make_dgp(seed=0, violation_severity=0.05, trace_blend=0.0, n_nodes=40).generate().metadata["n_wl_collisions"]
    hi = _make_dgp(seed=0, violation_severity=0.9, trace_blend=0.0, n_nodes=40).generate().metadata["n_wl_collisions"]
    assert hi >= lo

def test_spurious_edges_scale_with_trace_fpr(sample_trace: Path):
    low = sample_trace.parent / "low.jsonl"
    high = sample_trace.parent / "high.jsonl"
    low.write_text(json.dumps({"false_positive_rate": 0.01, "gnss_drift": 1.0, "loop_node": "scene_graph"}) + "\n")
    high.write_text(json.dumps({"false_positive_rate": 0.45, "gnss_drift": 1.0, "loop_node": "scene_graph"}) + "\n")
    lo = _make_dgp(seed=1, violation_severity=0.0, trace_path=str(low)).generate().metadata["n_spurious_edges"]
    hi = _make_dgp(seed=1, violation_severity=0.0, trace_path=str(high)).generate().metadata["n_spurious_edges"]
    assert hi >= lo

def test_trace_stats_loaded_from_jsonl(sample_trace: Path):
    stats = _make_dgp(trace_path=str(sample_trace)).generate().metadata["trace_stats"]
    assert stats["false_positive_rate"] == pytest.approx(0.15)
    assert stats["gnss_drift_m"] == pytest.approx(6.0)
    assert stats["n_trace_rows"] == 2.0

def test_inferred_violation_severity_in_metadata():
    data = _make_dgp(violation_severity=0.3).generate()
    assert 0.0 <= data.metadata["inferred_violation_severity"] <= 1.0
    assert "field_gap" in data.metadata

def test_module_10_config_references_dgp():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "module_10.yaml").read_text())
    assert cfg["module_id"] == 10
    assert cfg["loop_node"] == "scene_graph"
    assert cfg["dgp"]["class"].endswith("SceneGraphFieldDGP")
    assert len(cfg["seeds"]) == 20
    assert cfg["violation_levels"] == [0.0, 0.1, 0.2, 0.3, 0.5]
