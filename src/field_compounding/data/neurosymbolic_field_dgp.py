"""Trace-backed DGP for neurosymbolic visual task planning (Module 12 / loop node 12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from field_compounding.data.base import BaseFieldDGP, BenchmarkData

LOOP_NODE = "neurosymbolic"
OBJECT_TYPES = ["cup", "plate", "tool", "obstacle"]
PREDICATE_ARITIES = {"near": 2, "above": 2, "holding": 1, "clear": 1}
ACTION_NAMES = ["pick", "place", "push", "stack", "unstack", "move_to", "rotate", "release"]
DEFAULT_TRACE_ROWS = [
    {"false_positive_rate": 0.10, "gnss_drift": 3.7, "loop_node": LOOP_NODE, "recovery": True, "timestamp": "2025-05-18T17:14:06Z", "violation_severity": 0.51},
    {"false_positive_rate": 0.12, "gnss_drift": 4.2, "loop_node": LOOP_NODE, "recovery": False, "timestamp": "2025-05-18T17:22:11Z", "violation_severity": 0.58},
    {"false_positive_rate": 0.08, "gnss_drift": 2.9, "loop_node": LOOP_NODE, "recovery": True, "timestamp": "2025-05-18T17:31:44Z", "violation_severity": 0.44},
]

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

def _resolve_trace_path(trace_path: str | None) -> Path | None:
    if trace_path is None:
        return None
    direct = Path(trace_path)
    if direct.is_file():
        return direct
    rooted = _repo_root() / trace_path
    return rooted if rooted.is_file() else None

def load_neurosymbolic_trace_rows(trace_path: str | None) -> list[dict[str, Any]]:
    resolved = _resolve_trace_path(trace_path)
    if resolved is None:
        return list(DEFAULT_TRACE_ROWS)
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("loop_node") == LOOP_NODE:
                rows.append(row)
    return rows or list(DEFAULT_TRACE_ROWS)

def summarize_trace_telemetry(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"mean_gnss_drift_m": 3.7, "mean_false_positive_rate": 0.10, "mean_trace_violation": 0.51}
    gnss = np.array([float(r.get("gnss_drift", 0.0)) for r in rows], dtype=np.float64)
    fpr = np.array([float(r.get("false_positive_rate", 0.0)) for r in rows], dtype=np.float64)
    trace_v = np.array([float(r.get("violation_severity", 0.0)) for r in rows], dtype=np.float64)
    return {"mean_gnss_drift_m": float(gnss.mean()), "mean_false_positive_rate": float(fpr.mean()), "mean_trace_violation": float(trace_v.mean())}

def infer_violation_from_telemetry(gnss_drift_m: float, false_positive_rate: float, *, gnss_scale: float = 8.0, fpr_scale: float = 0.20) -> float:
    gnss_term = np.clip(gnss_drift_m / gnss_scale, 0.0, 1.0)
    fpr_term = np.clip(false_positive_rate / fpr_scale, 0.0, 1.0)
    return float(np.clip(0.55 * gnss_term + 0.45 * fpr_term, 0.0, 1.0))

def effective_grounding_noise(violation_severity: float, telemetry: dict[str, float], *, gnss_drift_scale: float = 1.0, field_blend: float = 0.35) -> float:
    drift_term = np.clip(telemetry["mean_gnss_drift_m"] * gnss_drift_scale / 8.0, 0.0, 1.0)
    fpr_term = np.clip(telemetry["mean_false_positive_rate"] / 0.20, 0.0, 1.0)
    trace_term = 0.5 * drift_term + 0.5 * fpr_term
    return float(np.clip((1.0 - field_blend) * violation_severity + field_blend * trace_term, 0.0, 1.0))

def _ground_fact(predicate: str, arity: int, obj_ids: list[int], rng: np.random.Generator) -> tuple[str, tuple[int, ...]]:
    return predicate, tuple(rng.choice(obj_ids, size=arity, replace=False).tolist())

def _generate_facts(n_objects: int, rng: np.random.Generator, noise_frac: float = 0.0) -> list[tuple[str, tuple[int, ...]]]:
    obj_ids = list(range(n_objects))
    facts: list[tuple[str, tuple[int, ...]]] = []
    for i in range(n_objects):
        if rng.random() < 0.3:
            facts.append(("clear", (i,)))
    for i in range(n_objects):
        for j in range(n_objects):
            if i == j:
                continue
            if rng.random() < 0.05:
                facts.append(("near", (i, j)))
            if rng.random() < 0.03:
                facts.append(("above", (i, j)))
    for _ in range(int(len(facts) * noise_frac)):
        pred = rng.choice(list(PREDICATE_ARITIES.keys()))
        facts.append(_ground_fact(pred, PREDICATE_ARITIES[pred], obj_ids, rng))
    return facts

def _build_action_schemas() -> list[dict[str, Any]]:
    return [
        {"name": "pick", "params": 1, "preconditions": [("clear", (0,))], "add_effects": [("holding", (0,))], "del_effects": [("clear", (0,))]},
        {"name": "place", "params": 2, "preconditions": [("holding", (0,))], "add_effects": [("near", (0, 1)), ("clear", (0,))], "del_effects": [("holding", (0,))]},
        {"name": "push", "params": 1, "preconditions": [("clear", (0,))], "add_effects": [], "del_effects": [("clear", (0,))]},
        {"name": "stack", "params": 2, "preconditions": [("holding", (0,)), ("clear", (1,))], "add_effects": [("above", (0, 1))], "del_effects": [("holding", (0,)), ("clear", (1,))]},
        {"name": "unstack", "params": 2, "preconditions": [("above", (0, 1))], "add_effects": [("holding", (0,)), ("clear", (1,))], "del_effects": [("above", (0, 1))]},
        {"name": "move_to", "params": 2, "preconditions": [("holding", (0,))], "add_effects": [("near", (0, 1))], "del_effects": []},
        {"name": "rotate", "params": 1, "preconditions": [("holding", (0,))], "add_effects": [], "del_effects": []},
        {"name": "release", "params": 1, "preconditions": [("holding", (0,))], "add_effects": [("clear", (0,))], "del_effects": [("holding", (0,))]},
    ]

def _build_rules() -> list[dict[str, Any]]:
    return [
        {"name": "near_transitive", "head": ("near", ("X", "Z")), "body": [("near", ("X", "Y")), ("near", ("Y", "Z"))]},
        {"name": "above_transitive", "head": ("above", ("X", "Z")), "body": [("above", ("X", "Y")), ("above", ("Y", "Z"))]},
        {"name": "above_implies_near", "head": ("near", ("X", "Y")), "body": [("above", ("X", "Y"))]},
        {"name": "holding_not_clear", "head": ("holding", ("X",)), "body": [("holding", ("X",))]},
        {"name": "near_symmetric", "head": ("near", ("Y", "X")), "body": [("near", ("X", "Y"))]},
    ]

def _generate_tasks(n_tasks: int, n_objects: int, facts: list[tuple[str, tuple[int, ...]]], rng: np.random.Generator) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for _ in range(n_tasks):
        subset_size = int(rng.integers(3, min(10, n_objects)))
        subset = rng.choice(n_objects, size=subset_size, replace=False).tolist()
        init_state = [(p, args) for p, args in facts if all(a in subset for a in args)]
        goal_preds = []
        for _ in range(int(rng.integers(1, 4))):
            pred = rng.choice(list(PREDICATE_ARITIES.keys()))
            args = tuple(rng.choice(subset, size=PREDICATE_ARITIES[pred], replace=False).tolist())
            goal_preds.append((pred, args))
        tasks.append({"objects": subset, "initial_state": init_state, "goal_state": goal_preds})
    return tasks

def _generate_formulas(n_atoms: int, n_formulas: int, rng: np.random.Generator, atom_targets: np.ndarray | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    atom_targets = rng.uniform(0.0, 1.0, size=n_atoms).astype(np.float32) if atom_targets is None else atom_targets.astype(np.float32).copy()
    formulas: list[dict[str, Any]] = []
    for _ in range(n_formulas):
        op = rng.choice(["and", "or"])
        n_args = int(rng.integers(2, min(4, n_atoms) + 1))
        arg_indices = rng.choice(n_atoms, size=n_args, replace=False).tolist()
        negated = rng.random(size=n_args) < 0.3
        vals = atom_targets[arg_indices].copy()
        for k in range(n_args):
            if negated[k]:
                vals[k] = 1.0 - vals[k]
        target = float(np.prod(vals)) if op == "and" else float(1.0 - np.prod(1.0 - vals))
        formulas.append({"op": op, "atom_indices": arg_indices, "negated": negated.tolist(), "target": target})
    return atom_targets, formulas

def _apply_field_position_drift(positions: np.ndarray, gnss_drift_m: float, rng: np.random.Generator, *, gnss_drift_scale: float) -> np.ndarray:
    noise = rng.normal(0.0, 0.04 * gnss_drift_m * gnss_drift_scale, size=positions.shape).astype(np.float32)
    return np.clip(positions + noise, -1.0, 1.0)

def _apply_grounding_noise(atom_targets: np.ndarray, noise_frac: float, rng: np.random.Generator) -> np.ndarray:
    noisy = atom_targets.astype(np.float32).copy()
    noisy += rng.normal(0.0, 0.35 * noise_frac, size=noisy.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    flip_mask = rng.uniform(size=noisy.shape) < noise_frac
    noisy[flip_mask] = 1.0 - noisy[flip_mask]
    return noisy

class NeurosymbolicFieldDGP(BaseFieldDGP):
    def __init__(self, seed: int = 42, violation_severity: float = 0.0, trace_path: str | None = None, *, gnss_drift_scale: float = 1.0, field_blend: float = 0.35, n_objects: int = 80, n_tasks: int = 20, n_atoms: int = 8, n_formulas: int = 6) -> None:
        super().__init__(seed=seed, violation_severity=violation_severity, trace_path=trace_path)
        self.gnss_drift_scale = gnss_drift_scale
        self.field_blend = field_blend
        self.n_objects = n_objects
        self.n_tasks = n_tasks
        self.n_atoms = n_atoms
        self.n_formulas = n_formulas

    @property
    def name(self) -> str:
        return "neurosymbolic_field_task_planning"

    @property
    def loop_node(self) -> str:
        return LOOP_NODE

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        trace_rows = load_neurosymbolic_trace_rows(self.trace_path)
        telemetry = summarize_trace_telemetry(trace_rows)
        grounding_noise = effective_grounding_noise(self.violation_severity, telemetry, gnss_drift_scale=self.gnss_drift_scale, field_blend=self.field_blend)
        inferred_v = infer_violation_from_telemetry(telemetry["mean_gnss_drift_m"], telemetry["mean_false_positive_rate"])
        obj_types = rng.choice(OBJECT_TYPES, size=self.n_objects).tolist()
        obj_positions = _apply_field_position_drift(rng.uniform(-1.0, 1.0, size=(self.n_objects, 3)).astype(np.float32), telemetry["mean_gnss_drift_m"], rng, gnss_drift_scale=self.gnss_drift_scale)
        facts = _generate_facts(self.n_objects, rng, noise_frac=grounding_noise)
        pred_names = list(PREDICATE_ARITIES.keys())
        fact_array = np.full((len(facts), 3), -1, dtype=np.int32)
        for i, (pred, args) in enumerate(facts):
            fact_array[i, 0] = pred_names.index(pred)
            for j, arg in enumerate(args):
                fact_array[i, 1 + j] = arg
        actions = _build_action_schemas()
        rules = _build_rules()
        tasks = _generate_tasks(self.n_tasks, self.n_objects, facts, rng)
        atom_targets = _apply_grounding_noise(rng.uniform(0.0, 1.0, size=self.n_atoms).astype(np.float32), grounding_noise, rng)
        atom_targets, formulas = _generate_formulas(self.n_atoms, self.n_formulas, rng, atom_targets=atom_targets)
        split = max(1, int(0.7 * len(tasks)))
        train_task_ids = np.arange(split, dtype=np.int32)
        test_task_ids = np.arange(split, len(tasks), dtype=np.int32)
        n_train, n_test = len(train_task_ids), len(test_task_ids)
        train_dict = {"object_positions": obj_positions, "facts": fact_array, "atom_targets": atom_targets, "task_ids": train_task_ids, "gnss_drift_m": np.full(n_train, telemetry["mean_gnss_drift_m"], dtype=np.float32), "false_positive_rate": np.full(n_train, telemetry["mean_false_positive_rate"], dtype=np.float32)}
        test_dict = {"object_positions": obj_positions, "facts": fact_array, "atom_targets": atom_targets, "task_ids": test_task_ids, "gnss_drift_m": np.full(n_test, telemetry["mean_gnss_drift_m"], dtype=np.float32), "false_positive_rate": np.full(n_test, telemetry["mean_false_positive_rate"], dtype=np.float32)}
        metadata = {"loop_node": self.loop_node, "n_objects": self.n_objects, "object_types": obj_types, "predicate_names": pred_names, "action_names": ACTION_NAMES, "actions": actions, "rules": rules, "tasks": tasks, "formulas": formulas, "n_atoms": self.n_atoms, "n_formulas": self.n_formulas, "violation_severity": self.violation_severity, "grounding_noise_frac": grounding_noise, "inferred_violation_severity": inferred_v, "trace_row_count": len(trace_rows), "telemetry": telemetry, "gnss_drift_scale": self.gnss_drift_scale, "field_blend": self.field_blend, "seed": self.seed}
        return BenchmarkData(train=train_dict, test=test_dict, metadata=metadata)
