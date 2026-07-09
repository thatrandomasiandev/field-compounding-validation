"""Trace-backed and hybrid field data generators."""

from field_compounding.data.base import BaseFieldDGP, BenchmarkData
from field_compounding.data.causal_field_dgp import CausalFieldDGP

__all__ = ["BaseFieldDGP", "BenchmarkData", "CausalFieldDGP"]
