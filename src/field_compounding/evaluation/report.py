"""Result reporting and JSON output utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Write results dict to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_serializer)


def load_results(path: str | Path) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def merge_results(result_files: list[Path]) -> dict[str, Any]:
    """Merge multiple result files into a single summary."""
    merged: dict[str, Any] = {"modules": {}, "summary": {}}

    for f in result_files:
        data = load_results(f)
        mod_id = data.get("module_id")
        if mod_id:
            merged["modules"][mod_id] = data

    all_models: set[str] = set()
    for mod_data in merged["modules"].values():
        all_models.update(mod_data.get("models", {}).keys())

    merged["summary"]["n_modules"] = len(merged["modules"])
    merged["summary"]["n_models"] = len(all_models)
    merged["summary"]["model_names"] = sorted(all_models)
    return merged


def format_table_row(
    model_name: str,
    metrics: dict[str, dict[str, float]],
) -> str:
    """Format a single row for LaTeX table."""
    cols = [model_name]
    for values in metrics.values():
        mean = values["mean"]
        std = values["std"]
        cols.append(f"{mean:.3f} $\\pm$ {std:.3f}")
    return " & ".join(cols) + " \\\\"


def validate_results_schema(data):
    return [f"missing top-level key '{k}'" for k in ("module_id","n_trials","n_successful","models","metadata") if k not in data]

def _serializer(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
