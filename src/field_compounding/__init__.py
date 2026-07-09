"""Field validation of the Visual Compounding Problem (Module 12)."""

from field_compounding.data import BaseFieldDGP, BenchmarkData
from field_compounding.utils import get_device, set_seed

__version__ = "0.1.0"

__all__ = [
    "BaseFieldDGP",
    "BenchmarkData",
    "__version__",
    "get_device",
    "set_seed",
]
