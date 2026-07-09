"""Trace-backed and hybrid field data generating processes."""

from field_compounding.data.base import BaseFieldDGP, BenchmarkData
from field_compounding.data.scene_graph_field_dgp import SceneGraphFieldDGP

__all__ = ["BaseFieldDGP", "BenchmarkData", "SceneGraphFieldDGP"]
