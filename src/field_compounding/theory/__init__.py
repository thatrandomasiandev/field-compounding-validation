"""Theory: partial observability, compound bounds, and coupling transfer."""

from field_compounding.theory.compound_bound import (
    compound_excess_risk,
    predict_compound_excess,
    validate_theorem_2,
)
from field_compounding.theory.coupling_transfer import (
    CouplingMatrix,
    apply_field_correction,
    field_correction_factor,
    load_module11_coupling,
    loop_node_to_module_id,
    module_id_to_loop_node,
    trace_density_by_node,
    transfer_coupling_to_field,
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
    "CouplingMatrix",
    "apply_field_correction",
    "bound_inflation",
    "compound_excess_risk",
    "coupling_inflation",
    "field_correction_factor",
    "linear_inflation",
    "load_module11_coupling",
    "loop_node_to_module_id",
    "module_id_to_loop_node",
    "observability_ratio",
    "predict_compound_excess",
    "rmse_v",
    "trace_density_by_node",
    "transfer_coupling_to_field",
    "validate_theorem_2",
    "violation_estimation_error",
]
