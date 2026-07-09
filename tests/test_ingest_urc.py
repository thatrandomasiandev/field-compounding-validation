"""Tests for URC outdoor field trace ingest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from field_compounding.ingest.urc import (
    FIELD_LOG_REQUIRED_KEYS,
    URCFieldLogEntry,
    URC_EXTENSION_KEYS,
    VALID_LOOP_NODES,
    aggregate_urc_stats,
    generate_synthetic_urc_traces,
    load_urc_trace,
    save_urc_trace,
)

SYNTHETIC = (
    Path(__file__).resolve().parents[1] / "observatory" / "traces" / "urc_synthetic.jsonl"
)


def test_load_synthetic_trace() -> None:
    entries = load_urc_trace(SYNTHETIC)
    assert len(entries) >= 200
    assert all(isinstance(e, URCFieldLogEntry) for e in entries)


def test_all_loop_nodes_represented() -> None:
    entries = load_urc_trace(SYNTHETIC)
    assert {e.loop_node for e in entries} == set(VALID_LOOP_NODES)


def test_roundtrip_jsonl(tmp_path: Path) -> None:
    original = load_urc_trace(SYNTHETIC)[:20]
    out = tmp_path / "trace.jsonl"
    save_urc_trace(original, out)
    reloaded = load_urc_trace(out)
    assert len(reloaded) == len(original)
    assert reloaded[0].violation_severity == original[0].violation_severity
    assert reloaded[0].battery_pct == original[0].battery_pct
    assert reloaded[0].cmd_latency_ms == original[0].cmd_latency_ms


def test_field_log_schema_keys_present() -> None:
    entries = load_urc_trace(SYNTHETIC)
    row = entries[0].to_dict()
    assert FIELD_LOG_REQUIRED_KEYS.issubset(row.keys())
    assert URC_EXTENSION_KEYS.issubset(row.keys())


def test_to_field_log_dict_strips_extensions() -> None:
    entry = load_urc_trace(SYNTHETIC)[0]
    base = entry.to_field_log_dict()
    assert set(base.keys()) == FIELD_LOG_REQUIRED_KEYS
    assert "battery_pct" not in base
    assert "cmd_latency_ms" not in base


def test_aggregate_stats() -> None:
    entries = load_urc_trace(SYNTHETIC)
    stats = aggregate_urc_stats(entries)
    assert stats["row_count"] == len(entries)
    assert stats["overall"]["mean_violation_severity"] > 0
    assert 0 <= stats["overall"]["recovery_rate"] <= 1
    assert 0 <= stats["overall"]["mean_battery_pct"] <= 100
    assert stats["overall"]["mean_cmd_latency_ms"] >= 0
    assert len(stats["appendix_table"]) == len(stats["by_loop_node"])


def test_invalid_loop_node_raises() -> None:
    with pytest.raises(ValueError, match="unknown loop_node"):
        URCFieldLogEntry(
            timestamp="2025-05-18T12:00:00Z",
            loop_node="invalid_node",
            violation_severity=0.5,
            recovery=True,
            gnss_drift=1.0,
            false_positive_rate=0.1,
            battery_pct=80.0,
            cmd_latency_ms=25.0,
        )


def test_missing_extension_raises() -> None:
    with pytest.raises(KeyError, match="missing URC extension keys"):
        URCFieldLogEntry.from_dict(
            {
                "timestamp": "2025-05-18T12:00:00Z",
                "loop_node": "safety",
                "violation_severity": 0.5,
                "recovery": True,
                "gnss_drift": 1.0,
                "false_positive_rate": 0.1,
            }
        )


def test_violation_severity_clipped() -> None:
    entry = URCFieldLogEntry(
        timestamp="2025-05-18T12:00:00Z",
        loop_node="safety",
        violation_severity=1.8,
        recovery=False,
        gnss_drift=-2.0,
        false_positive_rate=-0.2,
        battery_pct=150.0,
        cmd_latency_ms=-5.0,
    )
    assert entry.violation_severity == 1.0
    assert entry.gnss_drift == 0.0
    assert entry.false_positive_rate == 0.0
    assert entry.battery_pct == 100.0
    assert entry.cmd_latency_ms == 0.0


def test_generate_synthetic_is_deterministic() -> None:
    a = generate_synthetic_urc_traces(50, seed=7)
    b = generate_synthetic_urc_traces(50, seed=7)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]


def test_generate_synthetic_covers_all_nodes() -> None:
    entries = generate_synthetic_urc_traces(240, seed=42)
    assert len(entries) == 240
    assert {e.loop_node for e in entries} == set(VALID_LOOP_NODES)


def test_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_urc_trace(bad)


def test_jsonl_rows_are_sorted_keys() -> None:
    line = SYNTHETIC.read_text().splitlines()[0]
    obj = json.loads(line)
    assert list(obj.keys()) == sorted(obj.keys())
