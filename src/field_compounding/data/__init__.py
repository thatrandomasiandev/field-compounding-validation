"""Trace-backed and hybrid field data generating processes."""

from field_compounding.data.base import BaseFieldDGP, BenchmarkData
from field_compounding.data.world_model_field_dgp import WorldModelFieldDGP

__all__ = ["BaseFieldDGP", "BenchmarkData", "WorldModelFieldDGP"]
