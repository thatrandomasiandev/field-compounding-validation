# Field Validation of Visual Compounding

**Sequel to Module 11:** validates the Visual Compounding Problem on real robot field traces (URC outdoor autonomy, Roomba CV indoor), under partial observability of violation severity, with Condition D mitigation benchmarks and a compound field experiment.

Joshua J. Terranova · USC CS · jjt_373@usc.edu

## Thesis

Module 11 proves and benchmarks visual compounding in controlled synthetic modules. Module 12 asks whether the same coupling structure \(\hat{\gamma}_{k,l}\), compound excess, and mitigation gains hold when \(v_k\) must be **inferred from telemetry** rather than injected, and when traces come from **field hardware** rather than DGPs alone.

## Parity targets (match Module 11)

| Artifact | Module 11 | Module 12 target |
|----------|-----------|------------------|
| Benchmark modules | 13 (sections 3–14) + compound (15) | 13 field-replay modules + field compound |
| Config YAMLs | 13 | 13 |
| pytest functions | 121+ | 121+ |
| `validate_results.py` | required, min 20 trials | required, min 20 trials |
| Paper | JMLR scaffold, 360 pp stretch | same structure, field-specific prose |
| Scripts | run_benchmark, compound, fill, validate | same pipeline on field traces |

## Package layout

```
src/field_compounding/
  data/          trace-backed DGPs (one per loop node)
  ingest/        URC, Roomba, synthetic replay loaders
  models/        violation estimators, mitigation (Condition D)
  theory/        partial observability bounds, coupling transfer
  evaluation/    runner, adapters, compound field experiment
  observatory/   manifest v2, field appendix aggregation
  utils/
configs/         module_03.yaml … module_14.yaml
scripts/         run_benchmark, validate, fill_paper_values
tests/           10+ test modules, 121+ test functions
paper/sections/  sec03 … sec15 + intro/conclusion
```

## Reproduction

```bash
pip install -e ".[dev]"
pytest -q
FIELD_COMPOUNDING_FAST=0 python scripts/run_benchmark.py --module all --n-trials 20
python scripts/run_compound_field_experiment.py --n-trials 20
python scripts/validate_results.py --min-trials 20
```

## Colab CLI (GPU from terminal)

Install [Google Colab CLI](https://github.com/googlecolab/google-colab-cli):

```bash
uv tool install git+https://github.com/googlecolab/google-colab-cli
```

From this repo:

```bash
chmod +x scripts/colab_cli.sh

# Smoke test on T4 (~minutes)
./scripts/colab_cli.sh smoke

# One module, full trials
./scripts/colab_cli.sh module 08

# Full pipeline (long): 20 trials, compound, validate, download results/
COLAB_GPU=A100 ./scripts/colab_cli.sh full

# Tear down VM when done
./scripts/colab_cli.sh stop
```

One-shot (provision → run → release):

```bash
colab run --gpu T4 scripts/colab_remote_run.py -- --fast --n-trials 1
```

## Agent orchestration

See `MODULE12_AGENT_PLAN.md` for the 24 parallel cloud-agent workstreams.
