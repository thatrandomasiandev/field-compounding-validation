"""Deterministic sliding-window replay of field traces for DGPs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from field_compounding.ingest.schema import TraceRecord, load_traces


@dataclass(frozen=True)
class ReplayBatch:
    """One sliding window of trace rows."""

    entries: tuple[TraceRecord, ...]
    window_start: int
    window_end: int
    batch_index: int

    @property
    def size(self) -> int:
        return len(self.entries)


def subsample_indices(n_rows: int, *, fraction: float, seed: int) -> np.ndarray:
    """Return a deterministic subset of row indices in ascending order."""
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    if n_rows == 0 or fraction >= 1.0:
        return np.arange(n_rows, dtype=np.int64)

    target = max(1, int(np.floor(n_rows * fraction)))
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n_rows, size=target, replace=False))
    return chosen.astype(np.int64)


class ReplaySession:
    """Replay field traces as sliding-window batches for trace-backed DGPs."""

    def __init__(
        self,
        entries: Sequence[TraceRecord],
        *,
        window_size: int = 16,
        stride: int = 1,
        seed: int = 42,
        loop_node: str | None = None,
        subsample_fraction: float = 1.0,
        max_batches: int | None = None,
    ):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        filtered = list(entries)
        if loop_node is not None:
            filtered = [entry for entry in filtered if entry.loop_node == loop_node]

        self._source_count = len(filtered)
        indices = subsample_indices(len(filtered), fraction=subsample_fraction, seed=seed)
        self._entries = tuple(filtered[i] for i in indices)
        self.window_size = window_size
        self.stride = stride
        self.seed = seed
        self.loop_node = loop_node
        self.subsample_fraction = subsample_fraction
        self.max_batches = max_batches

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        window_size: int = 16,
        stride: int = 1,
        seed: int = 42,
        loop_node: str | None = None,
        subsample_fraction: float = 1.0,
        max_batches: int | None = None,
    ) -> ReplaySession:
        return cls(
            load_traces(path),
            window_size=window_size,
            stride=stride,
            seed=seed,
            loop_node=loop_node,
            subsample_fraction=subsample_fraction,
            max_batches=max_batches,
        )

    @property
    def entries(self) -> tuple[TraceRecord, ...]:
        return self._entries

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def source_count(self) -> int:
        return self._source_count

    def batch_count(self) -> int:
        if len(self._entries) < self.window_size:
            return 0
        total = 1 + (len(self._entries) - self.window_size) // self.stride
        if self.max_batches is None:
            return total
        return min(total, self.max_batches)

    def __len__(self) -> int:
        return self.batch_count()

    def __iter__(self) -> Iterator[ReplayBatch]:
        yield from self.batches()

    def batches(self) -> list[ReplayBatch]:
        batches: list[ReplayBatch] = []
        if len(self._entries) < self.window_size:
            return batches

        batch_index = 0
        for start in range(0, len(self._entries) - self.window_size + 1, self.stride):
            end = start + self.window_size
            batches.append(
                ReplayBatch(
                    entries=self._entries[start:end],
                    window_start=start,
                    window_end=end,
                    batch_index=batch_index,
                )
            )
            batch_index += 1
            if self.max_batches is not None and batch_index >= self.max_batches:
                break
        return batches

    def batch_to_arrays(self, batch: ReplayBatch) -> dict[str, np.ndarray]:
        """Materialize numeric columns for DGP consumption."""
        return {
            "violation_severity": np.array(
                [entry.violation_severity for entry in batch.entries], dtype=np.float64
            ),
            "gnss_drift": np.array([entry.gnss_drift for entry in batch.entries], dtype=np.float64),
            "false_positive_rate": np.array(
                [entry.false_positive_rate for entry in batch.entries], dtype=np.float64
            ),
            "recovery": np.array([entry.recovery for entry in batch.entries], dtype=bool),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "source_count": self.source_count,
            "entry_count": self.entry_count,
            "window_size": self.window_size,
            "stride": self.stride,
            "seed": self.seed,
            "loop_node": self.loop_node,
            "subsample_fraction": self.subsample_fraction,
            "batch_count": self.batch_count(),
        }
