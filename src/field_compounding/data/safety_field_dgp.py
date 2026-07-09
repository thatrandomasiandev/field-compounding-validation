"""Field-backed DGP for formal visual safety (Module 14)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from field_compounding.data.base import BaseFieldDGP, BenchmarkData

DT: float = 0.1
A_MAX: float = 1.0
GOAL_BONUS: float = 10.0
GOAL_RADIUS: float = 0.05
TRAJ_STEPS: int = 100
N_TRAJ_TRAIN: int = 200
N_TRAJ_TEST: int = 50
DEFAULT_N_OBS: int = 3
HJ_GRID_SIZE: int = 50
HJ_VALUE_STEPS: int = 100
DEFAULT_P_CATASTROPHE: float = 0.05
LOOP_NODE: str = "safety"


def load_urc_trace_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def summarize_safety_traces(rows: list[dict[str, Any]]) -> dict[str, float]:
    safety_rows = [r for r in rows if r.get("loop_node") == LOOP_NODE]
    if not safety_rows:
        return {
            "row_count": 0.0,
            "mean_gnss_drift_m": 0.0,
            "mean_false_positive_rate": 0.0,
            "mean_cmd_latency_ms": 0.0,
            "mean_battery_pct": 100.0,
            "mean_violation_severity": 0.0,
        }
    gnss = np.array([float(r["gnss_drift"]) for r in safety_rows], dtype=np.float64)
    fpr = np.array([float(r["false_positive_rate"]) for r in safety_rows], dtype=np.float64)
    latency = np.array([float(r.get("cmd_latency_ms", 50.0)) for r in safety_rows], dtype=np.float64)
    battery = np.array([float(r.get("battery_pct", 100.0)) for r in safety_rows], dtype=np.float64)
    sev = np.array([float(r["violation_severity"]) for r in safety_rows], dtype=np.float64)
    return {
        "row_count": float(len(safety_rows)),
        "mean_gnss_drift_m": float(np.mean(gnss)),
        "mean_false_positive_rate": float(np.mean(fpr)),
        "mean_cmd_latency_ms": float(np.mean(latency)),
        "mean_battery_pct": float(np.mean(battery)),
        "mean_violation_severity": float(np.mean(sev)),
    }


def _step(state: np.ndarray, action: np.ndarray, dt: float = DT) -> np.ndarray:
    x, y, vx, vy = state
    ax, ay = action
    return np.array(
        [x + (vx + ax * dt) * dt, y + (vy + ay * dt) * dt, vx + ax * dt, vy + ay * dt],
        dtype=np.float32,
    )


def _clip_action(action: np.ndarray, a_max: float) -> np.ndarray:
    norm = np.linalg.norm(action)
    if norm > a_max:
        action = action * (a_max / norm)
    return action


def _collision(pos: np.ndarray, obs_positions: np.ndarray, obs_radii: np.ndarray) -> bool:
    dists = np.linalg.norm(pos - obs_positions, axis=-1) - obs_radii
    return float(np.min(dists)) < 0.0


def _rollout(
    start: np.ndarray,
    goal: np.ndarray,
    obs_positions: np.ndarray,
    obs_radii: np.ndarray,
    rng: np.random.Generator,
    *,
    n_steps: int = TRAJ_STEPS,
    p_catastrophe: float = 0.0,
    model_error: float = 0.0,
    a_max: float = A_MAX,
) -> dict[str, np.ndarray]:
    states = np.zeros((n_steps + 1, 4), dtype=np.float32)
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    rewards = np.zeros(n_steps, dtype=np.float32)
    costs = np.zeros(n_steps, dtype=np.float32)
    states[0] = start
    reached_goal = False

    for t in range(n_steps):
        pos = states[t, :2]
        vel = states[t, 2:]
        err = goal - pos
        action = 2.0 * err - 0.5 * vel + rng.standard_normal(2).astype(np.float32) * 0.1
        actions[t] = _clip_action(action, a_max)
        states[t + 1] = _step(states[t], actions[t])

        if model_error > 0.0 and rng.random() < min(1.0, model_error + 0.05):
            pos_next = states[t + 1, :2]
            nearest = int(np.argmin(np.linalg.norm(obs_positions - pos_next, axis=-1)))
            direction = obs_positions[nearest] - pos_next
            norm = float(np.linalg.norm(direction)) + 1e-8
            states[t + 1, :2] = (
                pos_next + (direction / norm) * (0.05 + model_error * 0.25)
            ).astype(np.float32)

        if p_catastrophe > 0 and rng.random() < p_catastrophe:
            phantom = states[t + 1, :2] + rng.uniform(-0.1, 0.1, size=2).astype(np.float32)
            obs_positions = np.vstack([obs_positions, phantom[None, :]])
            obs_radii = np.append(obs_radii, rng.uniform(0.02, 0.05))

        if _collision(states[t + 1, :2], obs_positions, obs_radii):
            costs[t] = 1.0

        dist_to_goal = float(np.linalg.norm(states[t + 1, :2] - goal))
        rewards[t] = -dist_to_goal
        if dist_to_goal < GOAL_RADIUS and not reached_goal:
            rewards[t] += GOAL_BONUS
            reached_goal = True

    return {"states": states, "actions": actions, "rewards": rewards, "costs": costs}


class SafetyFieldDGP(BaseFieldDGP):
    """Formal visual safety field DGP with trace-backed model error."""

    def __init__(
        self,
        seed: int = 42,
        violation_severity: float = 0.0,
        trace_path: str | None = None,
        n_obstacles: int = DEFAULT_N_OBS,
        n_traj_train: int = N_TRAJ_TRAIN,
        n_traj_test: int = N_TRAJ_TEST,
        p_catastrophe: float = DEFAULT_P_CATASTROPHE,
        hj_grid_size: int = HJ_GRID_SIZE,
        hj_value_steps: int = HJ_VALUE_STEPS,
    ) -> None:
        super().__init__(seed=seed, violation_severity=violation_severity, trace_path=trace_path)
        self.n_obstacles = n_obstacles
        self.n_traj_train = n_traj_train
        self.n_traj_test = n_traj_test
        self.p_catastrophe = p_catastrophe
        self.hj_grid_size = hj_grid_size
        self.hj_value_steps = hj_value_steps
        self._trace_stats = self._load_trace_stats()

    @property
    def name(self) -> str:
        return "safety_field"

    @property
    def loop_node(self) -> str:
        return LOOP_NODE

    def _load_trace_stats(self) -> dict[str, float]:
        if self.trace_path is None:
            return summarize_safety_traces([])
        path = Path(self.trace_path)
        if not path.is_file():
            raise FileNotFoundError(f"trace file not found: {path}")
        return summarize_safety_traces(load_urc_trace_rows(path))

    def _field_proxies(self) -> dict[str, float]:
        if self._trace_stats["row_count"] > 0:
            return {
                "gnss_drift_m": self._trace_stats["mean_gnss_drift_m"],
                "false_positive_rate": self._trace_stats["mean_false_positive_rate"],
                "cmd_latency_ms": self._trace_stats["mean_cmd_latency_ms"],
                "battery_pct": self._trace_stats["mean_battery_pct"],
                "inferred_violation_severity": self._trace_stats["mean_violation_severity"],
            }
        v = self.violation_severity
        return {
            "gnss_drift_m": 1.2 + 6.6 * v,
            "false_positive_rate": 0.04 + 0.14 * v,
            "cmd_latency_ms": 15.0 + 85.0 * v,
            "battery_pct": 100.0 - 35.0 * v,
            "inferred_violation_severity": v,
        }

    def _generate_split(
        self,
        rng: np.random.Generator,
        n_traj: int,
        goal: np.ndarray,
        obs_positions: np.ndarray,
        obs_radii: np.ndarray,
        *,
        model_error: float,
        p_catastrophe: float,
        a_max: float,
    ) -> dict[str, np.ndarray]:
        all_states: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_rewards: list[np.ndarray] = []
        all_costs: list[np.ndarray] = []
        for _ in range(n_traj):
            start_pos = rng.uniform(0.05, 0.95, size=2).astype(np.float32)
            while any(
                np.linalg.norm(start_pos - op) < r + 0.05
                for op, r in zip(obs_positions, obs_radii)
            ):
                start_pos = rng.uniform(0.05, 0.95, size=2).astype(np.float32)
            traj = _rollout(
                np.array([*start_pos, 0.0, 0.0], dtype=np.float32),
                goal,
                obs_positions.copy(),
                obs_radii.copy(),
                rng,
                p_catastrophe=p_catastrophe,
                model_error=model_error,
                a_max=a_max,
            )
            all_states.append(traj["states"])
            all_actions.append(traj["actions"])
            all_rewards.append(traj["rewards"])
            all_costs.append(traj["costs"])
        return {
            "states": np.stack(all_states),
            "actions": np.stack(all_actions),
            "rewards": np.stack(all_rewards),
            "costs": np.stack(all_costs),
        }

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        field = self._field_proxies()

        drift_factor = 1.0 + 0.12 * field["gnss_drift_m"]
        max_error = 0.15 * drift_factor
        model_error = self.violation_severity * max_error
        battery_scale = float(np.clip(field["battery_pct"] / 100.0, 0.5, 1.0))
        effective_a_max = A_MAX * battery_scale
        effective_p_catastrophe = min(
            0.25,
            self.p_catastrophe + field["false_positive_rate"] * 0.3,
        )

        obs_positions = rng.uniform(0.2, 0.8, size=(self.n_obstacles, 2)).astype(np.float32)
        obs_radii = rng.uniform(0.04, 0.10, size=self.n_obstacles).astype(np.float32)
        obs_positions_model = obs_positions + rng.uniform(
            -model_error, model_error, size=obs_positions.shape
        ).astype(np.float32)
        effective_radii = np.clip(
            obs_radii * (1.0 + 0.35 * self.violation_severity),
            0.02,
            0.25,
        )

        goal = rng.uniform(0.1, 0.9, size=2).astype(np.float32)
        while any(np.linalg.norm(goal - op) < r + 0.1 for op, r in zip(obs_positions, obs_radii)):
            goal = rng.uniform(0.1, 0.9, size=2).astype(np.float32)

        rollout_kw = {
            "model_error": model_error,
            "p_catastrophe": effective_p_catastrophe,
            "a_max": effective_a_max,
        }
        train_data = self._generate_split(
            rng, self.n_traj_train, goal, obs_positions, effective_radii, **rollout_kw
        )
        test_data = self._generate_split(
            rng, self.n_traj_test, goal, obs_positions, effective_radii, **rollout_kw
        )
        train_data["safety_labels"] = (train_data["costs"].sum(axis=-1) == 0).astype(np.float32)
        test_data["safety_labels"] = (test_data["costs"].sum(axis=-1) == 0).astype(np.float32)

        metadata: dict[str, Any] = {
            "loop_node": self.loop_node,
            "field_domain": "urc_outdoor",
            "obstacle_positions": obs_positions.tolist(),
            "obstacle_positions_model": obs_positions_model.tolist(),
            "obstacle_radii": obs_radii.tolist(),
            "goal": goal.tolist(),
            "dt": DT,
            "a_max": effective_a_max,
            "n_obstacles": self.n_obstacles,
            "p_catastrophe": effective_p_catastrophe,
            "hj_grid_size": self.hj_grid_size,
            "hj_value_steps": self.hj_value_steps,
            "violation_severity": self.violation_severity,
            "model_error": model_error,
            "battery_scale": battery_scale,
            "trace_row_count": int(self._trace_stats["row_count"]),
            "trace_mean_violation_severity": self._trace_stats["mean_violation_severity"],
            "inferred_violation_severity": field["inferred_violation_severity"],
            "gnss_drift_m": field["gnss_drift_m"],
            "false_positive_rate": field["false_positive_rate"],
            "cmd_latency_ms": field["cmd_latency_ms"],
            "battery_pct": field["battery_pct"],
        }
        return BenchmarkData(train=train_data, test=test_data, metadata=metadata)
