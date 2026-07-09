"""Theory layer: coupling transfer from Module 11 to field traces."""

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

__all__ = [
    "CouplingMatrix",
    "apply_field_correction",
    "field_correction_factor",
    "load_module11_coupling",
    "loop_node_to_module_id",
    "module_id_to_loop_node",
    "trace_density_by_node",
    "transfer_coupling_to_field",
]
