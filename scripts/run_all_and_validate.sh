#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
pip install -e ".[dev]" -q
pytest -q
FIELD_COMPOUNDING_FAST="${FIELD_COMPOUNDING_FAST:-0}" python scripts/run_benchmark.py --module all --n-trials 20
python scripts/run_compound_field_experiment.py --n-trials 20
python scripts/validate_results.py --min-trials 20
echo "Module 12 validation gate passed."
