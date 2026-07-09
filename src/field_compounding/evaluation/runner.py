"""Per-module benchmark runner."""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from field_compounding.data.base import BenchmarkData
from field_compounding.evaluation.module_adapters import TrialContext, build_model, run_trial
from field_compounding.evaluation.statistical_tests import bootstrap_ci
from field_compounding.theory.violation_severity import get_violation_severity
from field_compounding.utils.seed import set_seed



def resolve_config_path(module, configs_dir="configs"):
    configs_dir = Path(configs_dir); mod = str(module).zfill(2)
    exact = configs_dir / f"module_{mod}.yaml"
    if exact.is_file(): return exact
    m = sorted(configs_dir.glob(f"module_{mod}_*.yaml"))
    if m: return m[0]
    raise FileNotFoundError(mod)

def get_evaluation_seeds(config, n_trials=None):
    seeds = config.get("evaluation",{}).get("seeds") or config.get("seeds") or [0]
    seeds = [int(s) for s in seeds]
    return seeds[:n_trials] if n_trials and len(seeds)>=n_trials else (list(range(n_trials)) if n_trials else seeds)

def get_violation_values(config):
    if "violation_sweep" in config: return [float(v) for v in config["violation_sweep"]["values"]]
    if "violation_levels" in config: return [float(v) for v in config["violation_levels"]]
    return [0.0,1.0]


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def instantiate_dgp(class_path: str, seed: int, violation_severity: float, params: dict[str, Any]):
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    sig = inspect.signature(cls.__init__)
    allowed = {k for k in sig.parameters if k not in {"self", "seed", "violation_severity"}}
    filtered = {k: v for k, v in params.items() if k in allowed}
    return cls(seed=seed, violation_severity=violation_severity, **filtered)


def aggregate_trials(trial_results: list[dict[str, float]], metrics: list[str]) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, float]] = {}
    all_keys = set()
    for result in trial_results:
        all_keys.update(result.keys())

    for key in all_keys:
        values = [r[key] for r in trial_results if key in r and isinstance(r.get(key), (int, float))]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(arr)
        aggregated[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "n_successful": len(values),
        }
    return aggregated


def _json_serializer(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_single_trial(
    module_id: int,
    model_name: str,
    model_class: str,
    model_params: dict[str, Any],
    dgp_class: str,
    dgp_params: dict[str, Any],
    seed: int,
    violation_severity: float,
    fast: bool = False,
) -> dict[str, float]:
    set_seed(seed)
    dgp = instantiate_dgp(dgp_class, seed, violation_severity, dgp_params)
    data: BenchmarkData = dgp.generate()
    ctx = TrialContext(
        module_id=module_id,
        model_name=model_name,
        class_path=model_class,
        params=model_params,
        fast=fast,
    )
    model = build_model(ctx, data.metadata)
    return run_trial(model, ctx, data)


def run_module_benchmark(
    config_path: str | Path,
    output_path: str | Path | None = None,
    fast: bool | None = None,
    n_trials: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    module_id = int(config["module_id"])
    module_name = config["module_name"]
    dgp_class = config["dgp"]["class"]
    dgp_params = config["dgp"].get("params", {})
    models = config["models"]
    metrics = config["evaluation"]["metrics"]
    seeds = get_evaluation_seeds(config, n_trials)
    violation_values = get_violation_values(config)

    if fast is None:
        fast = os.environ.get("FIELD_COMPOUNDING_FAST", "0") == "1"
    if fast:
        seeds = seeds[:1]
        if len(violation_values) >= 2:
            violation_values = [violation_values[0], violation_values[-1]]
        else:
            violation_values = violation_values[:1]

    from field_compounding.utils.device import get_device

    device = get_device()
    trials_per_model = len(violation_values) * len(seeds)
    total_trials = trials_per_model * len(models)

    print(f"=== Module {module_id}: {module_name} ===")
    print(f"Models: {[m['name'] for m in models]} | Seeds: {len(seeds)} | Fast: {fast} | Device: {device}")
    print(f"Total trials: {total_trials}")

    results: dict[str, Any] = {
        "module_id": module_id,
        "module_name": module_name,
        "n_trials": total_trials,
        "n_successful": 0,
        "violation_knob": config.get("violation_sweep",{}).get("knob","violation_level"),
        "violation_values": violation_values,
        "models": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {"loop_node": config.get("loop_node"), "trace_path": dgp_params.get("trace_path"), "fast_mode": fast, "device": str(device), "config_path": str(Path(config_path).resolve())},
    }

    n_successful = 0
    global_trial = 0
    for model_cfg in models:
        model_name = model_cfg["name"]
        print(f"  Running {model_name}...")
        model_results: dict[str, Any] = {}
        model_trial = 0

        for v_val in violation_values:
            v_severity = get_violation_severity(module_id, float(v_val))
            trial_results: list[dict[str, float]] = []

            for seed in seeds:
                global_trial += 1
                model_trial += 1
                print(
                    f"    [{global_trial}/{total_trials}] {model_name} "
                    f"({model_trial}/{trials_per_model}) seed={seed} v={v_val}",
                    flush=True,
                )
                t0 = time.perf_counter()
                trial_timeout = float(os.environ.get("FIELD_COMPOUNDING_TRIAL_TIMEOUT", "0"))
                try:
                    if trial_timeout > 0:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            future = pool.submit(
                                run_single_trial,
                                module_id=module_id,
                                model_name=model_name,
                                model_class=model_cfg["class"],
                                model_params=model_cfg.get("params", {}),
                                dgp_class=dgp_class,
                                dgp_params=dgp_params,
                                seed=int(seed),
                                violation_severity=v_severity,
                                fast=fast,
                            )
                            metrics = future.result(timeout=trial_timeout)
                    else:
                        metrics = run_single_trial(
                            module_id=module_id,
                            model_name=model_name,
                            model_class=model_cfg["class"],
                            model_params=model_cfg.get("params", {}),
                            dgp_class=dgp_class,
                            dgp_params=dgp_params,
                            seed=int(seed),
                            violation_severity=v_severity,
                            fast=fast,
                        )
                    trial_results.append(metrics); n_successful += 1
                except concurrent.futures.TimeoutError:
                    raise RuntimeError(
                        f"Module {module_id} model={model_name} seed={seed} v={v_val} "
                        f"timed out after {trial_timeout:.0f}s"
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Module {module_id} model={model_name} seed={seed} v={v_val} failed"
                    ) from exc
                elapsed = time.perf_counter() - t0
                print(f"      done in {elapsed:.1f}s", flush=True)

            model_results[str(v_val)] = aggregate_trials(trial_results, metrics)

        results["models"][model_name] = model_results

    results["n_successful"] = n_successful
    results["violation_severity"] = [
        get_violation_severity(module_id, float(v)) for v in violation_values
    ]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_serializer)
        print(f"  Saved: {output_path}")
        _maybe_push_results(module_id, fast)

    return results


def _maybe_push_results(module_id: int, fast: bool) -> None:
    if os.environ.get("FIELD_COMPOUNDING_AUTO_PUSH", "0") != "1":
        return
    mode = "fast" if fast else "full"
    message = f"Module {module_id:02d} {mode} benchmark results."
    script = Path(__file__).resolve().parents[3] / "scripts" / "push_results_to_github.py"
    if not script.is_file():
        print(f"  [WARN] Auto-push skipped: missing {script}")
        return
    import subprocess

    print(f"  Pushing results to GitHub...")
    proc = subprocess.run(
        [sys.executable, str(script), message],
        cwd=script.parent.parent,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown error").strip()
        print(f"  [WARN] GitHub push failed (exit {proc.returncode}): {detail}")
