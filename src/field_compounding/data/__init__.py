"""Trace-backed and hybrid field data generating processes."""

from field_compounding.data.base import BaseFieldDGP, BenchmarkData
from field_compounding.data.scene_repr_field_dgp import SceneReprFieldDGP

__all__ = ["BaseFieldDGP", "BenchmarkData", "SceneReprFieldDGP"]
