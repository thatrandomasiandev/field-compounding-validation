"""Indoor Roomba CV field trace schema and appendix aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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

REQUIRED_FIELDS = frozenset(
    {
        "timestamp",
        "loop_node",
        "violation_severity",
        "recovery",
        "cliff_events",
        "map_drift",
    }
)

DEFAULT_SYNTHETIC_PATH = (
    Path(__file__).resolve().parents[3] / "observatory" / "traces" / "roomba_synthetic.jsonl"
)


@dataclass
class RoombaLogEntry:
    """Single indoor field trace row from a Roomba CV autonomy run."""

    timestamp: str
    loop_node: str
    violation_severity: float
    recovery: bool
    cliff_events: int
    map_drift: float

    def __post_init__(self) -> None:
        if self.loop_node not in VALID_LOOP_NODES:
            raise ValueError(f"unknown loop_node: {self.loop_node!r}")
        self.violation_severity = float(np.clip(self.violation_severity, 0.0, 1.0))
        self.cliff_events = max(0, int(self.cliff_events))
        self.map_drift = max(0.0, float(self.map_drift))

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> RoombaLogEntry:
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"missing required fields: {missing_list}")
        return cls(
            timestamp=str(row["timestamp"]),
            loop_node=str(row["loop_node"]),
            violation_severity=float(row["violation_severity"]),
            recovery=bool(row["recovery"]),
            cliff_events=int(row["cliff_events"]),
            map_drift=float(row["map_drift"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_roomba_log(path: str | Path) -> list[RoombaLogEntry]:
    """Load indoor Roomba trace rows from a JSONL file."""
    entries: list[RoombaLogEntry] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            entries.append(RoombaLogEntry.from_dict(row))
    return entries


def save_roomba_log(entries: Iterable[RoombaLogEntry], path: str | Path) -> Path:
    """Write indoor Roomba trace rows to JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
    return out


def filter_by_loop_node(
    entries: list[RoombaLogEntry],
    loop_node: str,
) -> list[RoombaLogEntry]:
    """Return rows for a single loop node."""
    if loop_node not in VALID_LOOP_NODES:
        raise ValueError(f"unknown loop_node: {loop_node!r}")
    return [entry for entry in entries if entry.loop_node == loop_node]


def aggregate_roomba_stats(entries: list[RoombaLogEntry]) -> dict[str, Any]:
    """Aggregate per-loop-node stats for an appendix table."""
    if not entries:
        return {
            "row_count": 0,
            "overall": {},
            "by_loop_node": {},
            "appendix_table": [],
        }

    severities = np.array([e.violation_severity for e in entries], dtype=np.float64)
    cliff = np.array([e.cliff_events for e in entries], dtype=np.float64)
    drift = np.array([e.map_drift for e in entries], dtype=np.float64)
    recoveries = np.array([e.recovery for e in entries], dtype=bool)

    overall = {
        "mean_violation_severity": float(np.mean(severities)),
        "max_violation_severity": float(np.max(severities)),
        "recovery_rate": float(np.mean(recoveries)),
        "mean_cliff_events": float(np.mean(cliff)),
        "max_cliff_events": float(np.max(cliff)),
        "mean_map_drift_m": float(np.mean(drift)),
        "max_map_drift_m": float(np.max(drift)),
    }

    by_loop_node: dict[str, dict[str, Any]] = {}
    appendix_table: list[dict[str, Any]] = []

    nodes = sorted({e.loop_node for e in entries})
    for node in nodes:
        node_entries = [e for e in entries if e.loop_node == node]
        node_sev = np.array([e.violation_severity for e in node_entries], dtype=np.float64)
        node_cliff = np.array([e.cliff_events for e in node_entries], dtype=np.float64)
        node_drift = np.array([e.map_drift for e in node_entries], dtype=np.float64)
        node_rec = np.array([e.recovery for e in node_entries], dtype=bool)

        stats = {
            "events": len(node_entries),
            "mean_violation_severity": float(np.mean(node_sev)),
            "max_violation_severity": float(np.max(node_sev)),
            "recovery_rate": float(np.mean(node_rec)),
            "mean_cliff_events": float(np.mean(node_cliff)),
            "max_cliff_events": float(np.max(node_cliff)),
            "mean_map_drift_m": float(np.mean(node_drift)),
            "max_map_drift_m": float(np.max(node_drift)),
        }
        by_loop_node[node] = stats
        appendix_table.append({"loop_node": node, **stats})

    return {
        "row_count": len(entries),
        "overall": overall,
        "by_loop_node": by_loop_node,
        "appendix_table": appendix_table,
    }


def build_roomba_appendix(
    entries: list[RoombaLogEntry],
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build appendix JSON payload for Observatory manifest attachment."""
    stats = aggregate_roomba_stats(entries)
    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path) if source_path is not None else None,
        "venue": "Roomba CV",
        "environment": "indoor",
        **stats,
    }


def write_roomba_appendix(
    entries: list[RoombaLogEntry],
    output_path: str | Path,
    *,
    source_path: str | Path | None = None,
) -> Path:
    """Write appendix JSON to disk."""
    payload = build_roomba_appendix(entries, source_path=source_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def generate_synthetic_traces(
    n_rows: int = 240,
    *,
    seed: int = 42,
) -> list[RoombaLogEntry]:
    """Generate deterministic synthetic indoor traces for benchmarks."""
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")

    rng = np.random.default_rng(seed)
    nodes = sorted(VALID_LOOP_NODES)
    base_time = datetime(2025, 6, 12, 9, 0, 0, tzinfo=timezone.utc)

    entries: list[RoombaLogEntry] = []
    for idx in range(n_rows):
        node = nodes[idx % len(nodes)]
        severity = float(rng.beta(2.0, 4.0))
        cliff = int(rng.poisson(0.35 + 2.5 * severity))
        drift = float(rng.gamma(shape=1.5 + 3.0 * severity, scale=0.08))
        recovery = bool(rng.random() > 0.22 * severity)
        ts = base_time.replace(minute=(base_time.minute + idx) % 60, second=idx % 60)
        entries.append(
            RoombaLogEntry(
                timestamp=ts.isoformat().replace("+00:00", "Z"),
                loop_node=node,
                violation_severity=severity,
                recovery=recovery,
                cliff_events=cliff,
                map_drift=drift,
            )
        )
    return entries


def default_trace_path() -> Path:
    """Return bundled synthetic Roomba trace path."""
    return DEFAULT_SYNTHETIC_PATH
