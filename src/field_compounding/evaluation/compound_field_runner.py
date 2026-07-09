"""Compound field experiment runner with theorem validation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from field_compounding.evaluation.runner import (
    _json_serializer,
    aggregate_trials,
    load_config,
    run_single_trial,
)
from field_compounding.theory.compound_bound import (
    compound_excess_risk,
    estimate_psi_k,
    validate_theorem,
)
from field_compounding.theory.coupling_transfer import (
    apply_field_correction,
    estimate_full_coupling_matrix,
    load_module11_coupling,
    trace_density_by_node,
    transfer_coupling_to_field,
)
from field_compounding.theory.violation_severity import get_violation_severity


def _extract_scores(data: dict[str, Any]) -> list[float]:
    scores: list[float] = []
    if not data.get("models"):
        return scores
    first_model = next(iter(data["models"].values()))
    for v_key in sorted(first_model.keys(), key=lambda x: float(x)):
        trial_data = first_model[v_key]
        if "normalized_score" in trial_data:
            scores.append(trial_data["normalized_score"]["mean"])
        elif trial_data:
            first_metric = next(iter(trial_data.values()))
            if isinstance(first_metric, dict) and "mean" in first_metric:
                scores.append(first_metric["mean"])
    return scores


def _load_module_results(results_dir: Path, module_ids: list[int]) -> dict[int, dict[str, np.ndarray]]:
    all_results: dict[int, dict[str, np.ndarray]] = {}
    for mod_id in module_ids:
        result_file = results_dir / f"section_{mod_id:02d}.json"
        if not result_file.exists():
            continue
        with open(result_file) as f:
            data = json.load(f)
        v_sev = np.array(data.get("violation_severity", []), dtype=np.float64)
        scores = _extract_scores(data)
        if len(v_sev) > 0 and len(scores) == len(v_sev):
            all_results[mod_id] = {
                "violation_severity": v_sev,
                "normalized_score": np.array(scores, dtype=np.float64),
            }
    return all_results


def run_compound_field_experiment(
    config_path: str | Path,
    results_dir: str | Path,
    output_path: str | Path | None = None,
    fast: bool | None = None,
    trace_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    results_dir = Path(results_dir)
    module_ids = [int(m) for m in config["modules"]]
    seeds = [int(s) for s in config.get("seeds", [0])]
    conditions = config["conditions"]

    if fast is None:
        fast = os.environ.get("FIELD_COMPOUNDING_FAST", "0") == "1"
    if fast:
        seeds = seeds[:2]

    configs_root = Path(config_path).parent
    module_configs: dict[int, dict[str, Any]] = {}
    for mod_id in module_ids:
        matches = list(configs_root.glob(f"module_{mod_id:02d}_*.yaml"))
        if not matches:
            matches = list(configs_root.glob(f"module_{mod_id:02d}.yaml"))
        if matches:
            module_configs[mod_id] = load_config(matches[0])

    compound_results: dict[str, Any] = {
        "module_id": 15,
        "experiment": "compound_field",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": {},
        "coupling_matrix": None,
        "field_coupling_matrix": None,
        "theorem_validation": {},
        "fast_mode": fast,
    }

    for cond_name, cond_spec in conditions.items():
        print(f"\n--- Condition {cond_name}: {cond_spec.get('description', '')} ---")
        if "violation_severity" in cond_spec:
            v_dict = {mod_id: float(cond_spec["violation_severity"]) for mod_id in module_ids}
        else:
            v_dict = {int(k): float(v) for k, v in cond_spec["violation_severities"].items()}

        cond_results: dict[str, Any] = {}
        for mod_id in module_ids:
            if mod_id not in module_configs:
                continue
            mc = module_configs[mod_id]
            model_cfg = mc["models"][0]
            v_k = v_dict.get(mod_id, 0.0)
            trial_results: list[dict[str, float]] = []
            for seed in seeds:
                try:
                    trial_results.append(
                        run_single_trial(
                            module_id=mod_id,
                            model_name=model_cfg["name"],
                            model_class=model_cfg["class"],
                            model_params=model_cfg.get("params", {}),
                            dgp_class=mc["dgp"]["class"],
                            dgp_params=mc["dgp"].get("params", {}),
                            seed=int(seed),
                            violation_severity=v_k,
                            fast=fast,
                        )
                    )
                except NotImplementedError as exc:
                    print(f"    Module {mod_id}, seed {seed} skipped: {exc}")
                except Exception as exc:
                    print(f"    Module {mod_id}, seed {seed} failed: {exc}")
                    trial_results.append({})

            cond_results[str(mod_id)] = {
                "violation_severity": v_k,
                "trials": aggregate_trials(trial_results, mc["evaluation"]["metrics"]),
            }

        compound_results["conditions"][cond_name] = cond_results

    all_module_results = _load_module_results(results_dir, module_ids)
    if len(all_module_results) >= 2:
        available = [m for m in module_ids if m in all_module_results]
        sub_results = {m: all_module_results[m] for m in available}
        gamma, ci_low, ci_high = estimate_full_coupling_matrix(sub_results)
        compound_results["coupling_matrix"] = {
            "gamma": gamma.tolist(),
            "ci_lower": ci_low.tolist(),
            "ci_upper": ci_high.tolist(),
            "module_ids": available,
        }

        try:
            m11 = load_module11_coupling()
            counts = trace_counts or {node: 200 for node in m11.loop_nodes}
            gamma_field, densities, _ = transfer_coupling_to_field(m11, counts)
            compound_results["field_coupling_matrix"] = {
                "gamma": gamma_field.tolist(),
                "trace_densities": densities,
                "module_ids": list(m11.module_ids),
                "loop_nodes": list(m11.loop_nodes),
            }
            gamma_for_bound = gamma_field
        except FileNotFoundError:
            gamma_for_bound = gamma

        for cond_name, cond_data in compound_results["conditions"].items():
            active = [m for m in available if str(m) in cond_data]
            if not active:
                continue

            v_array = np.array([cond_data[str(m)]["violation_severity"] for m in active])
            psi = np.zeros(len(active))
            for i, mod_id in enumerate(active):
                r = all_module_results[mod_id]
                psi[i] = estimate_psi_k(r["violation_severity"], r["normalized_score"])

            idx_map = {mod_id: i for i, mod_id in enumerate(available)}
            active_idx = [idx_map[m] for m in active]
            sub_gamma = gamma_for_bound[np.ix_(active_idx, active_idx)]
            predicted_bound = compound_excess_risk(v_array, psi, sub_gamma)
            observed_scores = []
            for mod_id in active:
                key = str(mod_id)
                if key in cond_data and "normalized_score" in cond_data[key]["trials"]:
                    observed_scores.append(1.0 - cond_data[key]["trials"]["normalized_score"]["mean"])
            observed_loss = float(np.mean(observed_scores)) if observed_scores else 0.0
            compound_results["theorem_validation"][cond_name] = validate_theorem(
                observed_loss, predicted_bound
            )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(compound_results, f, indent=2, default=_json_serializer)
        print(f"\nSaved compound results: {output_path}")

    return compound_results
