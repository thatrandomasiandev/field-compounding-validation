"""Cross-fitting utilities for sample-splitting estimators on field traces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from field_compounding.utils.trace_index import TraceIndex


class KFoldSplitter:
    """Deterministic K-fold splitter wrapping sklearn."""

    def __init__(self, n_splits: int = 3, seed: int = 42):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        self.n_splits = n_splits
        self.seed = seed

    def split(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) tuples for ``n`` samples."""
        if n < self.n_splits:
            raise ValueError(f"n={n} must be >= n_splits={self.n_splits}")
        indices = np.arange(n)
        for train_idx, test_idx in self.kf.split(indices):
            yield train_idx, test_idx


def split_trace_index(
    index: TraceIndex,
    *,
    n_splits: int = 3,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return K-fold train/test index pairs over a :class:`TraceIndex`."""
    splitter = KFoldSplitter(n_splits=n_splits, seed=seed)
    return list(splitter.split(len(index)))
