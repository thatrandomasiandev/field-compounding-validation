#!/usr/bin/env python3
"""Fill \\VAL{} placeholders in LaTeX paper with computed results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

METRIC_ALIASES: dict[str, str] = {
    "target_accuracy": "target_acc",
    "source_accuracy": "source_acc",
    "mse": "train_loss",
    "gap": "domain_gap",
    "utility_gap": "gap",
    "edge_f1": "auc",
    "safety_rate": "normalized_score",
    "test_accuracy": "test_accuracy",
    "private_accuracy": "private_accuracy",
}

REVERSE_ALIASES: dict[str, str] = {v: k for k, v in METRIC_ALIASES.items() if v != k}


def _register_metric(
    registry: dict[str, str],
    base_key: str,
    metric_name: str,
    metric_data: dict,
) -> None:
    mean = metric_data["mean"]
    std = metric_data["std"]
    registry[base_key + "_mean"] = f"{mean:.3f}"
    registry[base_key + "_std"] = f"{std:.3f}"
    registry[base_key] = f"${mean:.3f} \\pm {std:.3f}$"

    for alias, canonical in METRIC_ALIASES.items():
        if metric_name == canonical:
            alias_key = base_key.replace(f"_{canonical}_", f"_{alias}_", 1)
            registry[alias_key] = registry[base_key]
            registry[alias_key + "_mean"] = registry[base_key + "_mean"]
            registry[alias_key + "_std"] = registry[base_key + "_std"]
        elif metric_name == alias:
            canonical_key = base_key.replace(f"_{alias}_", f"_{canonical}_", 1)
            if canonical_key not in registry:
                registry[canonical_key] = registry[base_key]
                registry[canonical_key + "_mean"] = registry[base_key + "_mean"]
                registry[canonical_key + "_std"] = registry[base_key + "_std"]

    if metric_name in REVERSE_ALIASES:
        alt = REVERSE_ALIASES[metric_name]
        alt_key = base_key.replace(f"_{metric_name}_", f"_{alt}_", 1)
        if alt_key not in registry:
            registry[alt_key] = registry[base_key]
            registry[alt_key + "_mean"] = registry[base_key + "_mean"]
            registry[alt_key + "_std"] = registry[base_key + "_std"]


def build_registry(results_dir: Path) -> dict[str, str]:
    registry: dict[str, str] = {}
    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        mod_id = data.get("module_id")
        prefix = f"sec{mod_id:02d}" if mod_id else json_file.stem

        for model_name, model_data in data.get("models", {}).items():
            model_key = model_name.replace(" ", "").replace("-", "")
            for v_key, metrics in model_data.items():
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        base_key = f"{prefix}_{model_key}_{metric_name}_v{v_key}"
                        _register_metric(registry, base_key, metric_name, metric_data)

        for cond, modules in data.get("conditions", {}).items():
            for mod_key, mod_data in modules.items():
                trials = mod_data.get("trials", {})
                for metric_name, metric_data in trials.items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        base_key = f"compound_{cond}_{mod_key}_{metric_name}"
                        _register_metric(registry, base_key, metric_name, metric_data)

        if "theorem_validation" in data:
            for cond, val in data["theorem_validation"].items():
                for k, v in val.items():
                    if isinstance(v, (int, float)):
                        registry[f"compound_{cond}_{k}"] = f"{v:.4f}"
                if "bound" in val and "observed" in val:
                    slack = float(val["observed"]) - float(val["bound"])
                    registry[f"compound_{cond}_slack"] = f"{slack:.4f}"

        if data.get("coupling_matrix"):
            cm = data["coupling_matrix"]
            gamma = np.array(cm["gamma"])
            module_ids = cm.get("module_ids", [i + 3 for i in range(gamma.shape[0])])
            ci_low = np.array(cm["ci_lower"]) if "ci_lower" in cm else None
            ci_high = np.array(cm["ci_upper"]) if "ci_upper" in cm else None
            for i in range(gamma.shape[0]):
                for j in range(gamma.shape[1]):
                    if i == j:
                        continue
                    mi, mj = int(module_ids[i]), int(module_ids[j])
                    val = float(gamma[i, j])
                    registry[f"gamma_{mi}_{mj}"] = f"{val:.3f}"
                    registry[f"gamma_{mj}_{mi}"] = f"{val:.3f}"
                    if ci_low is not None and ci_high is not None:
                        lo, hi = float(ci_low[i, j]), float(ci_high[i, j])
                        registry[f"gamma_{mi}_{mj}_ci_low"] = f"{lo:.3f}"
                        registry[f"gamma_{mi}_{mj}_ci_high"] = f"{hi:.3f}"
                        registry[f"gamma_{mj}_{mi}_ci_low"] = f"{lo:.3f}"
                        registry[f"gamma_{mj}_{mi}_ci_high"] = f"{hi:.3f}"

    return registry


def resolve_key(key: str, registry: dict[str, str]) -> str | None:
    return registry.get(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill paper values from results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--paper-dir", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--missing-value", type=str, default="---")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    paper_dir = Path(args.paper_dir)
    registry = build_registry(results_dir)
    print(f"Loaded {len(registry)} values")

    total = 0
    missing: set[str] = set()
    for tex_file in paper_dir.rglob("*.tex"):
        content = tex_file.read_text()
        matches = re.findall(r"\\VAL\{([^}]+)\}", content)
        if not matches:
            continue
        new_content = content
        replacements = 0
        for key in matches:
            value = resolve_key(key, registry)
            if value is not None:
                new_content = new_content.replace(f"\\VAL{{{key}}}", value)
                replacements += 1
            else:
                missing.add(key)
                if not args.dry_run:
                    new_content = new_content.replace(f"\\VAL{{{key}}}", args.missing_value)
        total += replacements
        if not args.dry_run and replacements:
            tex_file.write_text(new_content)

    print(f"Replaced {total} placeholders")
    if missing:
        print(f"Missing {len(missing)} keys")


if __name__ == "__main__":
    main()
