"""Theory layer: compound bounds and field coupling transfer."""

from field_compounding.theory.compound_bound import (
    compound_excess_risk,
    estimate_gamma_kl,
    estimate_psi_k,
    predict_compound_excess,
    validate_theorem,
)
from field_compounding.theory.coupling_transfer import (
    VALID_LOOP_NODES,
    CouplingMatrix,
    apply_field_correction,
    bundled_coupling_path,
    correction_factor_matrix,
    dominant_field_couplings,
    estimate_full_coupling_matrix,
    field_correction_factor,
    load_module11_coupling,
    loop_node_to_module_id,
    module_id_to_loop_node,
    trace_density_by_node,
    transfer_coupling_to_field,
)
from field_compounding.theory.violation_severity import get_violation_severity, sweep_violations

__all__ = [
    "VALID_LOOP_NODES",
    "CouplingMatrix",
    "apply_field_correction",
    "bundled_coupling_path",
    "compound_excess_risk",
    "correction_factor_matrix",
    "dominant_field_couplings",
    "estimate_full_coupling_matrix",
    "estimate_gamma_kl",
    "estimate_psi_k",
    "field_correction_factor",
    "get_violation_severity",
    "load_module11_coupling",
    "loop_node_to_module_id",
    "module_id_to_loop_node",
    "predict_compound_excess",
    "sweep_violations",
    "trace_density_by_node",
    "transfer_coupling_to_field",
    "validate_theorem",
]
