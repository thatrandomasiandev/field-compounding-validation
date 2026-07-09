# Module 12 — 24 Cloud Agent Workstreams

**Repo:** `12-field-compounding-validation`  
**Package:** `field_compounding`  
**Branch pattern:** `agent/XX-<slug>` (one branch per agent; merge to `main` after review)  
**Parity:** Match Module 11 (`11-cv-robotics-unified`) in structure, test count (121+), configs (13), validate pipeline, paper sections.

## Shared contracts (all agents)

- Loop nodes: same 12 as Module 11 `field_log.py` (`scene_repr`, `visual_ssl`, …, `safety`).
- Violation severity `v_k ∈ [0,1]`; field traces may only expose proxies → use `models/violation_estimator.py` interface.
- Results JSON schema mirrors Module 11: `{module_id, n_trials, n_successful, models: {...}, metadata}`.
- Config YAML keys mirror Module 11: `module_id`, `dgp`, `models`, `seeds`, `violation_levels`.
- Import Module 11 coupling matrix from env `MODULE11_RESULTS` or bundled `observatory/module11_coupling.json`.
- Every agent adds tests in their domain; Agent 23 owns integration gate.

---

## Agent 01 — Foundation & CI

**Branch:** `agent/01-foundation`  
**Owns:** `pyproject.toml`, `README.md`, `.gitignore`, `src/field_compounding/**/__init__.py`, `scripts/run_all_and_validate.sh`, `.github/workflows/ci.yml`  
**Deliver:** pip-installable package, CI runs `pytest -q` + ruff.  
**Tests:** `tests/test_package_imports.py` (≥5 tests).

---

## Agent 02 — Utils & cross-cutting

**Branch:** `agent/02-utils`  
**Owns:** `utils/crossfit.py`, `utils/trace_index.py`, extend `seed.py`/`device.py`  
**Deliver:** trace indexing by loop_node/timestamp; K-fold for field splits.  
**Tests:** `tests/test_utils.py` (≥8 tests).

---

## Agent 03 — Theory: partial observability

**Branch:** `agent/03-theory-partial-obs`  
**Owns:** `theory/partial_observability.py`, `theory/compound_bound.py`  
**Deliver:** Theorem 2 (estimation error inflates compound excess); functions `predict_compound_excess(gamma_hat, v_hat, v_true)`.  
**Tests:** `tests/test_theory_partial_obs.py` (≥12 tests). Mirror `11/tests/test_theory.py` style.

---

## Agent 04 — Theory: coupling transfer

**Branch:** `agent/04-theory-coupling`  
**Owns:** `theory/coupling_transfer.py`, `observatory/module11_coupling.json` (stub + loader)  
**Deliver:** Load Module 11 \(\hat{\gamma}_{k,l}\), apply field correction factor from trace density.  
**Tests:** `tests/test_coupling_transfer.py` (≥10 tests).

---

## Agent 05 — Ingest: URC outdoor

**Branch:** `agent/05-ingest-urc`  
**Owns:** `ingest/urc.py`, `observatory/traces/urc_synthetic.jsonl` (≥200 rows)  
**Deliver:** Parser for Module 11 `FieldLogEntry` schema + URC extensions (`battery_pct`, `cmd_latency_ms`).  
**Tests:** `tests/test_ingest_urc.py` (≥10 tests).

---

## Agent 06 — Ingest: Roomba indoor

**Branch:** `agent/06-ingest-roomba`  
**Owns:** `ingest/roomba.py`, `observatory/traces/roomba_synthetic.jsonl`  
**Deliver:** Indoor schema (`cliff_events`, `map_drift`, `loop_node`).  
**Tests:** `tests/test_ingest_roomba.py` (≥10 tests).

---

## Agent 07 — Ingest: replay engine

**Branch:** `agent/07-ingest-replay`  
**Owns:** `ingest/replay.py`, `ingest/schema.py`  
**Deliver:** `ReplaySession` sliding-window batches for DGPs; deterministic subsampling by seed.  
**Tests:** `tests/test_replay.py` (≥8 tests).

---

## Agent 08 — Data module 3 (scene_repr)

**Branch:** `agent/08-dgp-scene-repr`  
**Owns:** `data/scene_repr_field_dgp.py`, `configs/module_03.yaml`  
**Deliver:** GNSS-drift-scaled view sparsity from URC traces.  
**Tests:** `tests/test_dgp_scene_repr.py` (≥8 tests).

---

## Agent 09 — Data module 4 (visual_ssl)

**Branch:** `agent/09-dgp-visual-ssl`  
**Owns:** `data/visual_ssl_field_dgp.py`, `configs/module_04.yaml`  
**Tests:** `tests/test_dgp_visual_ssl.py` (≥8 tests).

---

## Agent 10 — Data module 5 (sim_to_field)

**Branch:** `agent/10-dgp-sim-to-field`  
**Owns:** `data/sim_to_field_dgp.py`, `configs/module_05.yaml`  
**Deliver:** Explicit sim–field gap parameter tied to trace domain shift.  
**Tests:** `tests/test_dgp_sim_to_field.py` (≥8 tests).

---

## Agent 11 — Data module 6 (visuomotor)

**Branch:** `agent/11-dgp-visuomotor`  
**Owns:** `data/visuomotor_field_dgp.py`, `configs/module_06.yaml`  
**Tests:** `tests/test_dgp_visuomotor.py` (≥8 tests).

---

## Agent 12 — Data module 7 (causal)

**Branch:** `agent/12-dgp-causal`  
**Owns:** `data/causal_field_dgp.py`, `configs/module_07.yaml`  
**Tests:** `tests/test_dgp_causal.py` (≥8 tests).

---

## Agent 13 — Data module 8 (world_model)

**Branch:** `agent/13-dgp-world-model`  
**Owns:** `data/world_model_field_dgp.py`, `configs/module_08.yaml`  
**Tests:** `tests/test_dgp_world_model.py` (≥8 tests).

---

## Agent 14 — Data module 9 (equivariant)

**Branch:** `agent/14-dgp-equivariant`  
**Owns:** `data/equivariant_field_dgp.py`, `configs/module_09.yaml`  
**Tests:** `tests/test_dgp_equivariant.py` (≥8 tests).

---

## Agent 15 — Data module 10 (scene_graph)

**Branch:** `agent/15-dgp-scene-graph`  
**Owns:** `data/scene_graph_field_dgp.py`, `configs/module_10.yaml`  
**Tests:** `tests/test_dgp_scene_graph.py` (≥8 tests).

---

## Agent 16 — Data module 11 (uncertainty)

**Branch:** `agent/16-dgp-uncertainty`  
**Owns:** `data/uncertainty_field_dgp.py`, `configs/module_11.yaml`  
**Tests:** `tests/test_dgp_uncertainty.py` (≥8 tests).

---

## Agent 17 — Data module 12 (neurosymbolic)

**Branch:** `agent/17-dgp-neurosymbolic`  
**Owns:** `data/neurosymbolic_field_dgp.py`, `configs/module_12.yaml`  
**Tests:** `tests/test_dgp_neurosymbolic.py` (≥8 tests).

---

## Agent 18 — Data modules 13–14 (federated + safety)

**Branch:** `agent/18-dgp-fleet-safety`  
**Owns:** `data/federated_field_dgp.py`, `data/safety_field_dgp.py`, `configs/module_13.yaml`, `configs/module_14.yaml`  
**Tests:** `tests/test_dgp_fleet_safety.py` (≥12 tests).

---

## Agent 19 — Models: violation estimators

**Branch:** `agent/19-models-estimators`  
**Owns:** `models/violation_estimator.py`, `models/calibration.py`  
**Deliver:** `TelemetryViolationEstimator.fit/predict`; map GNSS drift, FPR → v_k.  
**Tests:** `tests/test_violation_estimator.py` (≥12 tests).

---

## Agent 20 — Models: Condition D mitigation

**Branch:** `agent/20-models-mitigation`  
**Owns:** `models/condition_d.py`, `models/decoupling_stack.py`  
**Deliver:** Architectural decoupling baseline from Module 11 §15, adapted to field latency.  
**Tests:** `tests/test_mitigation.py` (≥10 tests).

---

## Agent 21 — Evaluation core

**Branch:** `agent/21-eval-core`  
**Owns:** `evaluation/runner.py`, `evaluation/metrics.py`, `evaluation/statistical_tests.py`, `evaluation/report.py`, `scripts/run_benchmark.py`  
**Tests:** `tests/test_evaluation_core.py` (≥10 tests).

---

## Agent 22 — Evaluation adapters (modules 3–8)

**Branch:** `agent/22-eval-adapters-a`  
**Owns:** `evaluation/module_adapters.py` functions `_eval_module_03` … `_eval_module_08`  
**Deliver:** Port/adapt from Module 11 `module_adapters.py` to consume field DGP metadata.  
**Tests:** `tests/test_adapters_a.py` (≥12 tests).

---

## Agent 23 — Evaluation adapters (9–14) + compound

**Branch:** `agent/23-eval-adapters-b`  
**Owns:** remainder of `module_adapters.py`, `evaluation/compound_field_runner.py`, `scripts/run_compound_field_experiment.py`, `scripts/validate_results.py`, `scripts/fill_paper_values.py`  
**Tests:** `tests/test_adapters_b.py`, `tests/test_pipeline_integration.py` (≥15 tests).

---

## Agent 24 — Paper + observatory v2

**Branch:** `agent/24-paper-observatory`  
**Owns:** `paper/paper12_field_compounding.tex`, `paper/sections/sec03.tex` … `sec15.tex`, intro/conclusion/appendix, `scripts/generate_observatory_manifest.py`, `observatory/manifest_v2.json` schema  
**Deliver:** Full JMLR scaffold (~same section count as Module 11); `\input{}` all sections; placeholder `\todo{}` only where results pending.  
**Tests:** `tests/test_paper_scaffold.py` (≥5 tests: all `\input` paths exist).

---

## Merge order

1. Agents 01–07 (foundation, theory, ingest)  
2. Agents 08–18 (DGPs + configs)  
3. Agents 19–20 (models)  
4. Agents 21–23 (evaluation + scripts)  
5. Agent 24 (paper)  
6. Full pipeline: `scripts/run_all_and_validate.sh`

## Definition of done

```bash
pip install -e ".[dev]"
pytest -q   # ≥121 passed
FIELD_COMPOUNDING_FAST=0 python scripts/run_benchmark.py --module all --n-trials 20
python scripts/run_compound_field_experiment.py --n-trials 20
python scripts/validate_results.py --min-trials 20  # exit 0
```
