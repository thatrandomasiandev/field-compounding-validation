"""Unified field trace schema for URC outdoor and Roomba indoor logs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

VALID_LOOP_NODES = frozenset(
    {
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
    }
)

CORE_FIELDS = frozenset(
    {
        "timestamp",
        "loop_node",
        "violation_severity",
        "recovery",
        "gnss_drift",
        "false_positive_rate",
    }
)

URC_OPTIONAL_FIELDS = frozenset({"battery_pct", "cmd_latency_ms"})
ROOMBA_OPTIONAL_FIELDS = frozenset({"cliff_events", "map_drift"})
KNOWN_OPTIONAL_FIELDS = URC_OPTIONAL_FIELDS | ROOMBA_OPTIONAL_FIELDS


@dataclass
class TraceRecord:
    """Single field trace row compatible with Module 11 ``FieldLogEntry``."""

    timestamp: str
    loop_node: str
    violation_severity: float
    recovery: bool
    gnss_drift: float
    false_positive_rate: float
    battery_pct: float | None = None
    cmd_latency_ms: float | None = None
    cliff_events: int | None = None
    map_drift: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.loop_node not in VALID_LOOP_NODES:
            raise ValueError(f"unknown loop_node: {self.loop_node!r}")
        self.violation_severity = float(np.clip(self.violation_severity, 0.0, 1.0))
        self.gnss_drift = max(0.0, float(self.gnss_drift))
        self.false_positive_rate = float(np.clip(self.false_positive_rate, 0.0, 1.0))
        if self.battery_pct is not None:
            self.battery_pct = float(np.clip(self.battery_pct, 0.0, 100.0))
        if self.cmd_latency_ms is not None:
            self.cmd_latency_ms = max(0.0, float(self.cmd_latency_ms))
        if self.cliff_events is not None:
            self.cliff_events = max(0, int(self.cliff_events))
        if self.map_drift is not None:
            self.map_drift = max(0.0, float(self.map_drift))

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> TraceRecord:
        missing = CORE_FIELDS - row.keys()
        if missing:
            raise ValueError(f"missing required fields: {sorted(missing)}")

        extra = {
            key: value
            for key, value in row.items()
            if key not in CORE_FIELDS and key not in KNOWN_OPTIONAL_FIELDS
        }

        return cls(
            timestamp=str(row["timestamp"]),
            loop_node=str(row["loop_node"]),
            violation_severity=float(row["violation_severity"]),
            recovery=bool(row["recovery"]),
            gnss_drift=float(row["gnss_drift"]),
            false_positive_rate=float(row["false_positive_rate"]),
            battery_pct=_optional_float(row, "battery_pct"),
            cmd_latency_ms=_optional_float(row, "cmd_latency_ms"),
            cliff_events=_optional_int(row, "cliff_events"),
            map_drift=_optional_float(row, "map_drift"),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {})
        for key in list(payload):
            if payload[key] is None:
                del payload[key]
        payload.update(extra)
        return payload


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    if key not in row or row[key] is None:
        return None
    return float(row[key])


def _optional_int(row: dict[str, Any], key: str) -> int | None:
    if key not in row or row[key] is None:
        return None
    return int(row[key])


def load_traces(path: str | Path) -> list[TraceRecord]:
    """Load trace rows from a JSONL file."""
    entries: list[TraceRecord] = []
    trace_path = Path(path)
    with trace_path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{trace_path}:{line_no}: invalid JSON") from exc
            entries.append(TraceRecord.from_dict(row))
    return entries


def save_traces(entries: Iterable[TraceRecord], path: str | Path) -> Path:
    """Write trace rows to JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
    return out
