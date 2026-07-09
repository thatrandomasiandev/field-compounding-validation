"""URC outdoor field trace ingest aligned with Module 11 ``field_log`` schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Shared with Module 11 ``cv_robotics.observatory.field_log``.
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

FIELD_LOG_REQUIRED_KEYS = frozenset(
    {
        "timestamp",
        "loop_node",
        "violation_severity",
        "recovery",
        "gnss_drift",
        "false_positive_rate",
    }
)

URC_EXTENSION_KEYS = frozenset({"battery_pct", "cmd_latency_ms"})


@dataclass
class URCFieldLogEntry:
    """Single URC outdoor trace row: Module 11 base fields plus rover telemetry."""

    timestamp: str
    loop_node: str
    violation_severity: float
    recovery: bool
    gnss_drift: float
    false_positive_rate: float
    battery_pct: float
    cmd_latency_ms: float

    def __post_init__(self) -> None:
        if self.loop_node not in VALID_LOOP_NODES:
            raise ValueError(f"unknown loop_node: {self.loop_node!r}")
        self.violation_severity = float(np.clip(self.violation_severity, 0.0, 1.0))
        self.gnss_drift = max(0.0, float(self.gnss_drift))
        self.false_positive_rate = float(np.clip(self.false_positive_rate, 0.0, 1.0))
        self.battery_pct = float(np.clip(self.battery_pct, 0.0, 100.0))
        self.cmd_latency_ms = max(0.0, float(self.cmd_latency_ms))

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> URCFieldLogEntry:
        missing = FIELD_LOG_REQUIRED_KEYS - row.keys()
        if missing:
            raise KeyError(f"missing required field_log keys: {sorted(missing)}")
        missing_ext = URC_EXTENSION_KEYS - row.keys()
        if missing_ext:
            raise KeyError(f"missing URC extension keys: {sorted(missing_ext)}")
        return cls(
            timestamp=str(row["timestamp"]),
            loop_node=str(row["loop_node"]),
            violation_severity=float(row["violation_severity"]),
            recovery=bool(row["recovery"]),
            gnss_drift=float(row["gnss_drift"]),
            false_positive_rate=float(row["false_positive_rate"]),
            battery_pct=float(row["battery_pct"]),
            cmd_latency_ms=float(row["cmd_latency_ms"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_field_log_dict(self) -> dict[str, Any]:
        """Return the Module 11-compatible subset (no URC extensions)."""
        payload = self.to_dict()
        for key in URC_EXTENSION_KEYS:
            payload.pop(key, None)
        return payload


def load_urc_trace(path: str | Path) -> list[URCFieldLogEntry]:
    """Load URC trace rows from a JSONL file."""
    entries: list[URCFieldLogEntry] = []
    source = Path(path)
    with open(source) as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSON") from exc
            entries.append(URCFieldLogEntry.from_dict(row))
    return entries


def save_urc_trace(entries: Iterable[URCFieldLogEntry], path: str | Path) -> Path:
    """Write URC trace rows to JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
    return out


def aggregate_urc_stats(entries: list[URCFieldLogEntry]) -> dict[str, Any]:
    """Aggregate per-loop-node stats including URC telemetry extensions."""
    if not entries:
        return {
            "row_count": 0,
            "overall": {},
            "by_loop_node": {},
            "appendix_table": [],
        }

    severities = np.array([e.violation_severity for e in entries], dtype=np.float64)
    gnss = np.array([e.gnss_drift for e in entries], dtype=np.float64)
    fpr = np.array([e.false_positive_rate for e in entries], dtype=np.float64)
    recoveries = np.array([e.recovery for e in entries], dtype=bool)
    battery = np.array([e.battery_pct for e in entries], dtype=np.float64)
    latency = np.array([e.cmd_latency_ms for e in entries], dtype=np.float64)

    overall = {
        "mean_violation_severity": float(np.mean(severities)),
        "max_violation_severity": float(np.max(severities)),
        "recovery_rate": float(np.mean(recoveries)),
        "mean_gnss_drift_m": float(np.mean(gnss)),
        "mean_false_positive_rate": float(np.mean(fpr)),
        "mean_battery_pct": float(np.mean(battery)),
        "mean_cmd_latency_ms": float(np.mean(latency)),
    }

    by_loop_node: dict[str, dict[str, Any]] = {}
    appendix_table: list[dict[str, Any]] = []

    for node in sorted({e.loop_node for e in entries}):
        node_entries = [e for e in entries if e.loop_node == node]
        node_sev = np.array([e.violation_severity for e in node_entries], dtype=np.float64)
        node_gnss = np.array([e.gnss_drift for e in node_entries], dtype=np.float64)
        node_fpr = np.array([e.false_positive_rate for e in node_entries], dtype=np.float64)
        node_rec = np.array([e.recovery for e in node_entries], dtype=bool)
        node_batt = np.array([e.battery_pct for e in node_entries], dtype=np.float64)
        node_lat = np.array([e.cmd_latency_ms for e in node_entries], dtype=np.float64)

        stats = {
            "events": len(node_entries),
            "mean_violation_severity": float(np.mean(node_sev)),
            "max_violation_severity": float(np.max(node_sev)),
            "recovery_rate": float(np.mean(node_rec)),
            "mean_gnss_drift_m": float(np.mean(node_gnss)),
            "mean_false_positive_rate": float(np.mean(node_fpr)),
            "mean_battery_pct": float(np.mean(node_batt)),
            "mean_cmd_latency_ms": float(np.mean(node_lat)),
        }
        by_loop_node[node] = stats
        appendix_table.append({"loop_node": node, **stats})

    return {
        "row_count": len(entries),
        "overall": overall,
        "by_loop_node": by_loop_node,
        "appendix_table": appendix_table,
    }


def generate_synthetic_urc_traces(
    n_rows: int = 240,
    *,
    seed: int = 42,
    start: datetime | None = None,
) -> list[URCFieldLogEntry]:
    """Generate deterministic synthetic URC outdoor traces for replay benchmarks."""
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")

    rng = np.random.default_rng(seed)
    nodes = sorted(VALID_LOOP_NODES)
    base_time = start or datetime(2025, 5, 18, 14, 0, 0, tzinfo=timezone.utc)

    entries: list[URCFieldLogEntry] = []
    for idx in range(n_rows):
        node = nodes[idx % len(nodes)]
        severity = float(rng.beta(2.0, 4.0))
        gnss = float(rng.exponential(2.5))
        fpr = float(rng.beta(1.5, 8.0))
        recovery = bool(severity < 0.55 or rng.random() > 0.35)
        battery = float(rng.uniform(18.0, 98.0))
        latency = float(rng.lognormal(mean=3.2, sigma=0.45))

        ts = base_time + timedelta(minutes=idx * 3, seconds=int(rng.integers(0, 45)))
        entries.append(
            URCFieldLogEntry(
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                loop_node=node,
                violation_severity=severity,
                recovery=recovery,
                gnss_drift=gnss,
                false_positive_rate=fpr,
                battery_pct=battery,
                cmd_latency_ms=latency,
            )
        )
    return entries
