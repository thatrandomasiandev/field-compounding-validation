"""Evaluation adapters for modules 3–8 (field DGPs)."""

from field_compounding.evaluation.metrics import normalized_score
from field_compounding.evaluation.module_adapters import (
    TrialContext,
    build_model,
    enrich_field_metrics,
    field_adjusted_baseline,
    resolve_violation_severity,
    run_trial,
)

__all__ = [
    "TrialContext",
    "build_model",
    "enrich_field_metrics",
    "field_adjusted_baseline",
    "normalized_score",
    "resolve_violation_severity",
    "run_trial",
]
