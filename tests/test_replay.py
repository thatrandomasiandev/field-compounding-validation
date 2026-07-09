"""Tests for trace schema and replay engine."""

from __future__ import annotations

import pytest
import numpy as np

from field_compounding.ingest.replay import ReplayBatch, ReplaySession, subsample_indices
from field_compounding.ingest.schema import TraceRecord, load_traces, save_traces

LOOP_NODES = [
    "scene_repr",
    "visual_ssl",
    "sim_to_real",
    "visuomotor",
    "causal",
    "world_model",
    "equivariant",
    "scene_graph",
    "uncertainty",
    "neurosymbolic",
    "federated",
    "safety",
]


def _base_row(
    *,
    loop_node: str = "scene_repr",
    violation_severity: float = 0.25,
    idx: int = 0,
) -> dict:
    return {
        "timestamp": f"2025-06-01T12:{idx:02d}:00Z",
        "loop_node": loop_node,
        "violation_severity": violation_severity,
        "recovery": idx % 2 == 0,
        "gnss_drift": 1.0 + idx * 0.1,
        "false_positive_rate": 0.05 + idx * 0.01,
    }


@pytest.fixture
def sample_traces() -> list[TraceRecord]:
    rows = []
    for idx in range(24):
        rows.append(TraceRecord.from_dict(_base_row(loop_node=LOOP_NODES[idx % 12], idx=idx)))
    return rows


@pytest.fixture
def trace_jsonl(tmp_path, sample_traces):
    path = tmp_path / "traces.jsonl"
    save_traces(sample_traces, path)
    return path


def test_trace_record_from_dict_core_fields():
    record = TraceRecord.from_dict(_base_row())
    assert record.loop_node == "scene_repr"
    assert record.violation_severity == pytest.approx(0.25)
    assert record.recovery is True


def test_trace_record_supports_urc_and_roomba_extensions():
    row = _base_row()
    row.update(
        {
            "battery_pct": 72.5,
            "cmd_latency_ms": 18.0,
            "cliff_events": 2,
            "map_drift": 0.4,
            "custom_tag": "lab_run",
        }
    )
    record = TraceRecord.from_dict(row)
    assert record.battery_pct == pytest.approx(72.5)
    assert record.cmd_latency_ms == pytest.approx(18.0)
    assert record.cliff_events == 2
    assert record.map_drift == pytest.approx(0.4)
    assert record.extra == {"custom_tag": "lab_run"}


def test_trace_record_rejects_unknown_loop_node():
    row = _base_row(loop_node="not_a_node")
    with pytest.raises(ValueError, match="unknown loop_node"):
        TraceRecord.from_dict(row)


def test_trace_record_clips_severity_and_rates():
    row = _base_row(violation_severity=1.7)
    row["false_positive_rate"] = -0.2
    record = TraceRecord.from_dict(row)
    assert record.violation_severity == pytest.approx(1.0)
    assert record.false_positive_rate == pytest.approx(0.0)


def test_load_and_save_traces_roundtrip(trace_jsonl, sample_traces):
    loaded = load_traces(trace_jsonl)
    assert len(loaded) == len(sample_traces)
    assert loaded[0].loop_node == sample_traces[0].loop_node
    assert loaded[-1].gnss_drift == pytest.approx(sample_traces[-1].gnss_drift)


def test_load_traces_reports_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_traces(path)


def test_replay_session_sliding_windows(sample_traces):
    session = ReplaySession(sample_traces, window_size=4, stride=2, seed=7)
    batches = session.batches()
    assert len(batches) == 11
    assert all(isinstance(batch, ReplayBatch) for batch in batches)
    assert batches[0].size == 4
    assert batches[0].window_start == 0
    assert batches[1].window_start == 2


def test_replay_session_filters_loop_node(sample_traces):
    session = ReplaySession(sample_traces, window_size=3, stride=1, loop_node="scene_repr", seed=1)
    assert session.source_count == 2
    assert len(session.batches()) == 0


def test_replay_session_deterministic_subsample(sample_traces):
    kwargs = dict(window_size=8, stride=4, subsample_fraction=0.5, seed=99)
    first = ReplaySession(sample_traces, **kwargs)
    second = ReplaySession(sample_traces, **kwargs)
    assert first.entry_count == second.entry_count
    assert [entry.timestamp for entry in first.batches()[0].entries] == [
        entry.timestamp for entry in second.batches()[0].entries
    ]


def test_replay_session_subsample_differs_by_seed(sample_traces):
    first = ReplaySession(sample_traces, subsample_fraction=0.5, seed=1)
    second = ReplaySession(sample_traces, subsample_fraction=0.5, seed=2)
    first_ts = {entry.timestamp for entry in first.entries}
    second_ts = {entry.timestamp for entry in second.entries}
    assert first_ts != second_ts


def test_subsample_indices_preserves_order_and_minimum():
    indices = subsample_indices(20, fraction=0.25, seed=42)
    assert len(indices) == 5
    assert np.all(indices[:-1] <= indices[1:])


def test_replay_session_batch_to_arrays(sample_traces):
    session = ReplaySession(sample_traces, window_size=3, stride=1, seed=0)
    batch = session.batches()[0]
    arrays = session.batch_to_arrays(batch)
    assert arrays["violation_severity"].shape == (3,)
    assert arrays["recovery"].dtype == bool


def test_replay_session_from_path(trace_jsonl):
    session = ReplaySession.from_path(trace_jsonl, window_size=5, stride=5, seed=3)
    assert session.entry_count == 24
    assert len(session) == 4


def test_replay_session_max_batches(sample_traces):
    session = ReplaySession(sample_traces, window_size=4, stride=1, max_batches=2, seed=0)
    assert len(list(session)) == 2
