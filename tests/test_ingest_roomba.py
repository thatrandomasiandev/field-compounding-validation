"""Tests for indoor Roomba CV field trace ingest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from field_compounding.ingest.roomba import (
    REQUIRED_FIELDS,
    VALID_LOOP_NODES,
    RoombaLogEntry,
    aggregate_roomba_stats,
    build_roomba_appendix,
    default_trace_path,
    filter_by_loop_node,
    generate_synthetic_traces,
    load_roomba_log,
    save_roomba_log,
    write_roomba_appendix,
)

SYNTHETIC = Path(__file__).resolve().parents[1] / "observatory" / "traces" / "roomba_synthetic.jsonl"


def test_load_synthetic_trace() -> None:
    entries = load_roomba_log(SYNTHETIC)
    assert len(entries) >= 200
    assert all(isinstance(e, RoombaLogEntry) for e in entries)
    assert entries[0].loop_node in VALID_LOOP_NODES


def test_roundtrip_jsonl(tmp_path: Path) -> None:
    original = load_roomba_log(SYNTHETIC)
    out = tmp_path / "trace.jsonl"
    save_roomba_log(original, out)
    reloaded = load_roomba_log(out)
    assert len(reloaded) == len(original)
    assert reloaded[0].violation_severity == original[0].violation_severity
    assert reloaded[0].cliff_events == original[0].cliff_events
    assert reloaded[0].map_drift == original[0].map_drift


def test_aggregate_stats() -> None:
    entries = load_roomba_log(SYNTHETIC)
    stats = aggregate_roomba_stats(entries)
    assert stats["row_count"] == len(entries)
    assert stats["overall"]["mean_violation_severity"] > 0
    assert 0 <= stats["overall"]["recovery_rate"] <= 1
    assert stats["overall"]["mean_map_drift_m"] >= 0
    assert len(stats["appendix_table"]) == len(stats["by_loop_node"])
    for row in stats["appendix_table"]:
        assert row["events"] >= 1
        assert 0 <= row["mean_violation_severity"] <= 1


def test_build_appendix_payload() -> None:
    entries = load_roomba_log(SYNTHETIC)
    appendix = build_roomba_appendix(entries, source_path=SYNTHETIC)
    assert appendix["schema_version"] == "1.0.0"
    assert appendix["venue"] == "Roomba CV"
    assert appendix["environment"] == "indoor"
    assert appendix["source"].endswith("roomba_synthetic.jsonl")
    assert appendix["appendix_table"]


def test_invalid_loop_node_raises() -> None:
    with pytest.raises(ValueError, match="unknown loop_node"):
        RoombaLogEntry(
            timestamp="2025-06-12T12:00:00Z",
            loop_node="invalid_node",
            violation_severity=0.5,
            recovery=True,
            cliff_events=1,
            map_drift=0.2,
        )


def test_cliff_events_clipped_non_negative() -> None:
    entry = RoombaLogEntry(
        timestamp="2025-06-12T12:00:00Z",
        loop_node="scene_repr",
        violation_severity=0.4,
        recovery=True,
        cliff_events=-3,
        map_drift=0.1,
    )
    assert entry.cliff_events == 0


def test_map_drift_clipped_non_negative() -> None:
    entry = RoombaLogEntry(
        timestamp="2025-06-12T12:00:00Z",
        loop_node="visual_ssl",
        violation_severity=0.4,
        recovery=False,
        cliff_events=2,
        map_drift=-0.5,
    )
    assert entry.map_drift == 0.0


def test_violation_severity_clipped() -> None:
    entry = RoombaLogEntry(
        timestamp="2025-06-12T12:00:00Z",
        loop_node="safety",
        violation_severity=1.7,
        recovery=True,
        cliff_events=0,
        map_drift=0.0,
    )
    assert entry.violation_severity == 1.0


def test_empty_log_aggregate() -> None:
    stats = aggregate_roomba_stats([])
    assert stats["row_count"] == 0
    assert stats["overall"] == {}
    assert stats["appendix_table"] == []


def test_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_roomba_log(bad)


def test_missing_required_field_raises() -> None:
    row = {field: "x" for field in REQUIRED_FIELDS}
    del row["map_drift"]
    with pytest.raises(ValueError, match="missing required fields"):
        RoombaLogEntry.from_dict(row)


def test_filter_by_loop_node() -> None:
    entries = load_roomba_log(SYNTHETIC)
    filtered = filter_by_loop_node(entries, "scene_repr")
    assert filtered
    assert all(e.loop_node == "scene_repr" for e in filtered)


def test_generate_synthetic_is_deterministic() -> None:
    a = generate_synthetic_traces(32, seed=7)
    b = generate_synthetic_traces(32, seed=7)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]


def test_write_appendix_creates_json(tmp_path: Path) -> None:
    entries = generate_synthetic_traces(12, seed=0)
    out = tmp_path / "appendix.json"
    write_roomba_appendix(entries, out, source_path=SYNTHETIC)
    payload = json.loads(out.read_text())
    assert payload["row_count"] == 12
    assert payload["environment"] == "indoor"


def test_default_trace_path_points_to_bundle() -> None:
    path = default_trace_path()
    assert path.name == "roomba_synthetic.jsonl"
    assert path.exists()


def test_all_loop_nodes_present_in_synthetic() -> None:
    entries = load_roomba_log(SYNTHETIC)
    nodes = {e.loop_node for e in entries}
    assert nodes == set(VALID_LOOP_NODES)


def test_cliff_events_correlate_with_severity() -> None:
    entries = load_roomba_log(SYNTHETIC)
    severities = np.array([e.violation_severity for e in entries])
    cliffs = np.array([e.cliff_events for e in entries], dtype=np.float64)
    corr = np.corrcoef(severities, cliffs)[0, 1]
    assert corr > 0.2
