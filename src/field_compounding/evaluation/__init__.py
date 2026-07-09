"""Evaluation layer exports."""

from field_compounding.evaluation.compound_field_runner import (
    _extract_scores,
    run_compound_field_experiment,
)
from field_compounding.evaluation.metrics import normalized_score
from field_compounding.evaluation.module_adapters import (
    TrialContext,
    build_model,
    enrich_field_metrics,
    field_adjusted_baseline,
    resolve_violation_severity,
    run_trial,
)
from field_compounding.evaluation.runner import (
    aggregate_trials,
    load_config,
    run_module_benchmark,
    run_single_trial,
)
from field_compounding.evaluation.statistical_tests import bootstrap_ci

__all__ = [
    "TrialContext",
    "_extract_scores",
    "aggregate_trials",
    "bootstrap_ci",
    "build_model",
    "enrich_field_metrics",
    "field_adjusted_baseline",
    "load_config",
    "normalized_score",
    "resolve_violation_severity",
    "run_compound_field_experiment",
    "run_module_benchmark",
    "run_single_trial",
    "run_trial",
]
