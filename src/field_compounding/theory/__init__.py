"""Theory: partial observability bounds and compound excess under field traces."""

from field_compounding.theory.compound_bound import (
    compound_excess_risk,
    predict_compound_excess,
    validate_theorem_2,
)
from field_compounding.theory.partial_observability import (
    bound_inflation,
    coupling_inflation,
    linear_inflation,
    observability_ratio,
    rmse_v,
    violation_estimation_error,
)

__all__ = [
    "bound_inflation",
    "compound_excess_risk",
    "coupling_inflation",
    "linear_inflation",
    "observability_ratio",
    "predict_compound_excess",
    "rmse_v",
    "validate_theorem_2",
    "violation_estimation_error",
]
