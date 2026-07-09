#!/usr/bin/env python3
"""Post-run result validation script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PRIMARY_METRIC: dict[int, str] = {
    3: "psnr",
    4: "linear_probe_accuracy",
    5: "target_accuracy",
    6: "mse",
    7: "pehe",
    8: "transition_mse",
    9: "rmse",
    10: "auc",
    11: "rmse",
    12: "accuracy",
    13: "test_accuracy",
    14: "safety_rate",
}


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _check_file(path: Path, min_trials: int, errors: list[str], warnings: list[str]) -> None:
    module_id = int(path.stem.split("_")[1])

    try:
        data = _load(path)
    except Exception as exc:
        errors.append(f"{path.name}: failed to load — {exc}")
        return

    if module_id == 15:
        if "conditions" not in data:
            errors.append(f"{path.name}: missing key 'conditions'")
        elif len(data.get("conditions", {})) < 1:
            errors.append(f"{path.name}: no compound conditions recorded")
        return

    for key in ("module_id", "module_name", "models", "violation_values"):
        if key not in data:
            errors.append(f"{path.name}: missing key '{key}'")
            return

    if data.get("fast_mode", False):
        warnings.append(f"{path.name}: produced in fast_mode=True — re-run without --fast")

    models = data["models"]
    primary = PRIMARY_METRIC.get(module_id)

    for model_name, v_results in models.items():
        primary_means: list[float] = []

        for v_key, cell in v_results.items():
            loc = f"{path.name} | {model_name} @ v={v_key}"

            if not cell:
                errors.append(f"{loc}: empty result cell")
                continue

            for metric, stats in cell.items():
                if not isinstance(stats, dict):
                    continue
                mean = stats.get("mean")
                if mean is not None and not np.isfinite(mean):
                    errors.append(f"{loc} | {metric}.mean={mean}: NaN or Inf")

                ns = stats.get("n_successful", 0)
                if ns < 1:
                    errors.append(f"{loc} | {metric}: n_successful={ns}")
                elif ns < min_trials:
                    warnings.append(f"{loc} | {metric}: n_successful={ns} < {min_trials}")

                lo = stats.get("ci_lower")
                hi = stats.get("ci_upper")
                if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
                    if lo > hi + 1e-9:
                        errors.append(f"{loc} | {metric}: ci_lower={lo:.4f} > ci_upper={hi:.4f}")
                    if mean is not None and np.isfinite(mean):
                        if mean < lo - 1e-6:
                            errors.append(f"{loc} | {metric}: mean={mean:.4f} below ci_lower={lo:.4f}")
                        if mean > hi + 1e-6:
                            errors.append(f"{loc} | {metric}: mean={mean:.4f} above ci_upper={hi:.4f}")

            if primary and primary in cell and isinstance(cell[primary], dict):
                m = cell[primary].get("mean")
                if m is not None and np.isfinite(m):
                    primary_means.append(m)

        if primary and len(primary_means) >= 2:
            spread = max(primary_means) - min(primary_means)
            if spread < 1e-6:
                errors.append(
                    f"{path.name} | {model_name}: primary metric '{primary}' is flat "
                    f"across all violation levels (spread={spread:.2e})"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark JSON results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    parser.add_argument("--min-trials", type=int, default=20)
    parser.add_argument("--section-ids", type=int, nargs="+", default=list(range(3, 16)))
    args = parser.parse_args(argv)

    errors: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []

    for sid in args.section_ids:
        path = args.results_dir / f"section_{sid:02d}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        _check_file(path, args.min_trials, errors, warnings)

    print(f"\n{'=' * 70}")
    print(f"Results validation — {args.results_dir}")
    print(f"{'=' * 70}")

    if missing:
        print(f"\n[MISSING] {len(missing)} file(s):")
        for m in missing:
            print(f"  • {m}")

    if warnings:
        print(f"\n[WARN] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"\n[FAIL] {len(errors)} error(s):")
        for e in errors:
            print(f"  [FAIL] {e}")
        print()
        return 1

    total_checked = len(args.section_ids) - len(missing)
    print(f"\n[PASS] {total_checked} file(s) validated — {len(warnings)} warning(s), 0 errors\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
