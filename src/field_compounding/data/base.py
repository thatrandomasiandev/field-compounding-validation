"""Abstract base for trace-backed and hybrid field DGPs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from field_compounding.utils.seed import set_seed


@dataclass
class BenchmarkData:
    """Container for generated or replayed benchmark data."""

    train: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    metadata: dict[str, Any]


class BaseFieldDGP(ABC):
    """Base data process for Module 12 benchmarks.

    Unlike Module 11 synthetic DGPs, field DGPs may:
    - replay rows from ``observatory/traces/*.jsonl``
    - blend replay with synthetic perturbations (sim–field gap)
    - expose ``inferred_violation_severity`` when ground-truth v_k is latent
    """

    def __init__(
        self,
        seed: int = 42,
        violation_severity: float = 0.0,
        trace_path: str | None = None,
    ):
        assert 0.0 <= violation_severity <= 1.0
        self.seed = seed
        self.violation_severity = violation_severity
        self.trace_path = trace_path

    def generate(self) -> BenchmarkData:
        set_seed(self.seed)
        return self._generate()

    @abstractmethod
    def _generate(self) -> BenchmarkData:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def loop_node(self) -> str:
        """One of the 12 loop nodes shared with Module 11 field_log schema."""
