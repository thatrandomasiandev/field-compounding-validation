"""Index field trace rows by loop node and timestamp."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Mapping, Sequence

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


def parse_timestamp(value: str | datetime) -> datetime:
    """Parse ISO-8601 timestamps used in field trace JSONL."""
    if isinstance(value, datetime):
        dt = value
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class TraceRecord:
    """Single indexed field trace row."""

    loop_node: str
    timestamp: str
    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> TraceRecord:
        loop_node = str(row["loop_node"])
        if loop_node not in VALID_LOOP_NODES:
            raise ValueError(f"unknown loop_node: {loop_node!r}")
        return cls(
            loop_node=loop_node,
            timestamp=str(row["timestamp"]),
            payload=dict(row),
        )


class TraceIndex:
    """Sorted, queryable index over field trace rows."""

    def __init__(self, records: Sequence[TraceRecord]):
        self._records = sorted(records, key=lambda record: parse_timestamp(record.timestamp))
        self._by_node: dict[str, list[int]] = defaultdict(list)
        for idx, record in enumerate(self._records):
            self._by_node[record.loop_node].append(idx)

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> TraceIndex:
        return cls([TraceRecord.from_dict(row) for row in rows])

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[TraceRecord]:
        return iter(self._records)

    def at(self, index: int) -> TraceRecord:
        return self._records[index]

    def loop_nodes(self) -> frozenset[str]:
        return frozenset(self._by_node)

    def indices_for(self, loop_node: str) -> list[int]:
        if loop_node not in VALID_LOOP_NODES:
            raise ValueError(f"unknown loop_node: {loop_node!r}")
        return list(self._by_node.get(loop_node, []))

    def records_for(self, loop_node: str) -> list[TraceRecord]:
        return [self._records[i] for i in self.indices_for(loop_node)]

    def slice_time(
        self,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> TraceIndex:
        """Return a sub-index whose timestamps fall in ``[start, end)``."""
        start_dt = parse_timestamp(start) if start is not None else None
        end_dt = parse_timestamp(end) if end is not None else None
        kept: list[TraceRecord] = []
        for record in self._records:
            ts = parse_timestamp(record.timestamp)
            if start_dt is not None and ts < start_dt:
                continue
            if end_dt is not None and ts >= end_dt:
                continue
            kept.append(record)
        return TraceIndex(kept)
