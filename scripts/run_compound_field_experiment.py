#!/usr/bin/env python3
"""Run the field compound experiment (Section 15 analogue)."""

from __future__ import annotations

import argparse
from pathlib import Path

from field_compounding.evaluation.compound_field_runner import run_compound_field_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run field compound experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "compound_field_experiment.yaml",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "section_15.json",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    run_compound_field_experiment(
        config_path=args.config,
        results_dir=args.results_dir,
        output_path=args.output,
        fast=args.fast or None,
    )


if __name__ == "__main__":
    main()
