"""Field trace-backed data generating processes."""

from field_compounding.data.base import BaseFieldDGP, BenchmarkData
from field_compounding.data.neurosymbolic_field_dgp import NeurosymbolicFieldDGP

__all__ = ["BaseFieldDGP", "BenchmarkData", "NeurosymbolicFieldDGP"]
