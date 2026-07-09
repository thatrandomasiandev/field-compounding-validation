"""Field DGP for Module 8 (visual world models) on URC outdoor traces.

Wraps the Module 11 planar three-link arm dynamics and injects field telemetry:
``cmd_latency_ms`` stale torques, ``battery_pct`` torque derating, and
``gnss_drift`` visual observation noise from ``world_model`` trace rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from field_compounding.data.base import BaseFieldDGP, BenchmarkData


class _FixedVisualEncoder(nn.Module):
    """Frozen random MLP simulating a camera feature encoder."""

    def __init__(self, state_dim: int, feat_dim: int, seed: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, feat_dim),
        )
        gen = torch.Generator().manual_seed(seed)
        for param in self.net.parameters():
            nn.init.normal_(param, 0.0, 0.3, generator=gen)
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_urc_trace_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            rows.append(row)
    return rows


def summarize_world_model_traces(rows: list[dict[str, Any]]) -> dict[str, float]:
    wm_rows = [r for r in rows if r.get("loop_node") == "world_model"]
    if not wm_rows:
        return {
            "row_count": 0.0,
            "mean_gnss_drift_m": 0.0,
            "mean_cmd_latency_ms": 0.0,
            "mean_battery_pct": 100.0,
            "mean_violation_severity": 0.0,
        }
    gnss = np.array([float(r["gnss_drift"]) for r in wm_rows], dtype=np.float64)
    latency = np.array(
        [float(r.get("cmd_latency_ms", 0.0)) for r in wm_rows],
        dtype=np.float64,
    )
    battery = np.array(
        [float(r.get("battery_pct", 100.0)) for r in wm_rows],
        dtype=np.float64,
    )
    sev = np.array([float(r["violation_severity"]) for r in wm_rows], dtype=np.float64)
    return {
        "row_count": float(len(wm_rows)),
        "mean_gnss_drift_m": float(np.mean(gnss)),
        "mean_cmd_latency_ms": float(np.mean(latency)),
        "mean_battery_pct": float(np.mean(battery)),
        "mean_violation_severity": float(np.mean(sev)),
    }


def inferred_violation_from_telemetry(
    gnss_drift_m: float,
    cmd_latency_ms: float,
    battery_pct: float,
    *,
    gnss_ref_m: float = 8.0,
    latency_ref_ms: float = 250.0,
) -> float:
    """Proxy violation severity from field telemetry when ground truth is latent."""
    gnss_term = min(1.0, max(0.0, gnss_drift_m / gnss_ref_m))
    latency_term = min(1.0, max(0.0, cmd_latency_ms / latency_ref_ms))
    battery_term = min(1.0, max(0.0, (100.0 - battery_pct) / 100.0))
    return float(np.clip(0.45 * gnss_term + 0.35 * latency_term + 0.20 * battery_term, 0.0, 1.0))


def sigma_m_from_violation(violation_severity: float, field_gap: float) -> float:
    base = max(0.01, min(0.5, violation_severity * 0.5))
    gap = max(0.0, min(1.0, field_gap))
    return float(np.clip(base + gap * 0.15, 0.01, 0.5))


class WorldModelFieldDGP(BaseFieldDGP):
    """URC trace-backed planar arm world model DGP."""

    N_LINKS: int = 3
    STATE_DIM: int = 6
    ACTION_DIM: int = 3
    VISUAL_FEAT_DIM: int = 16
    OBS_DIM: int = STATE_DIM + VISUAL_FEAT_DIM

    MASS: float = 1.0
    LENGTH: float = 1.0
    GRAVITY: float = 9.81
    LOOP_NODE: str = "world_model"
    GNSS_REF_M: float = 8.0
    LATENCY_REF_MS: float = 250.0

    def __init__(
        self,
        seed: int = 42,
        violation_severity: float = 0.0,
        trace_path: str | None = None,
        sigma_m: float | None = None,
        field_gap: float = 0.0,
        cmd_latency_ms: float | None = None,
        battery_pct: float | None = None,
        gnss_drift_m: float | None = None,
        n_trajectories: int = 500,
        horizon: int = 50,
        dt: float = 0.02,
    ):
        super().__init__(seed=seed, violation_severity=violation_severity, trace_path=trace_path)
        self.field_gap = float(np.clip(field_gap, 0.0, 1.0))
        self.cmd_latency_ms_override = cmd_latency_ms
        self.battery_pct_override = battery_pct
        self.gnss_drift_m_override = gnss_drift_m
        self.n_trajectories = n_trajectories
        self.horizon = horizon
        self.dt = dt
        self._trace_stats = self._load_trace_stats()

        if sigma_m is None:
            sigma_m = sigma_m_from_violation(violation_severity, self.field_gap)
        self.sigma_m = float(np.clip(sigma_m, 0.01, 0.5))
        self.violation_severity = float(
            np.clip((self.sigma_m - 0.01) / (0.5 - 0.01), 0.0, 1.0)
        )

    @property
    def name(self) -> str:
        return "world_model_field_planar_arm"

    @property
    def loop_node(self) -> str:
        return self.LOOP_NODE

    def _load_trace_stats(self) -> dict[str, float]:
        if self.trace_path is None:
            return summarize_world_model_traces([])
        path = Path(self.trace_path)
        if not path.is_file():
            raise FileNotFoundError(f"trace file not found: {path}")
        return summarize_world_model_traces(load_urc_trace_rows(path))

    def _resolved_telemetry(self) -> tuple[float, float, float]:
        gnss = (
            float(self.gnss_drift_m_override)
            if self.gnss_drift_m_override is not None
            else self._trace_stats["mean_gnss_drift_m"]
        )
        latency = (
            float(self.cmd_latency_ms_override)
            if self.cmd_latency_ms_override is not None
            else self._trace_stats["mean_cmd_latency_ms"]
        )
        battery = (
            float(self.battery_pct_override)
            if self.battery_pct_override is not None
            else self._trace_stats["mean_battery_pct"]
        )
        return max(0.0, gnss), max(0.0, latency), float(np.clip(battery, 0.0, 100.0))

    def _latency_steps(self, cmd_latency_ms: float) -> int:
        delay_s = cmd_latency_ms / 1000.0
        return max(0, int(round(delay_s / self.dt)))

    def _torque_scale(self, battery_pct: float) -> float:
        return float(np.clip(0.35 + 0.65 * (battery_pct / 100.0), 0.35, 1.0))

    def _step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
        rng: np.random.Generator,
        *,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        ml2 = self.MASS * self.LENGTH**2
        m_diag = np.full(self.N_LINKS, ml2)
        g_vec = self.MASS * self.GRAVITY * self.LENGTH * np.sin(q)
        noise = (
            rng.standard_normal(self.N_LINKS).astype(np.float64) * self.sigma_m
            + np.ones(self.N_LINKS, dtype=np.float64) * self.sigma_m * 1.5
            if add_noise
            else np.zeros(self.N_LINKS, dtype=np.float64)
        )
        ddq = (tau - g_vec + noise) / m_diag
        dq_new = dq + ddq * self.dt
        q_new = q + dq_new * self.dt
        q_new = ((q_new + np.pi) % (2 * np.pi)) - np.pi
        return q_new, dq_new

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        vis_encoder = _FixedVisualEncoder(self.STATE_DIM, self.VISUAL_FEAT_DIM, seed=self.seed)
        gnss_drift_m, cmd_latency_ms, battery_pct = self._resolved_telemetry()
        latency_steps = self._latency_steps(cmd_latency_ms)
        torque_scale = self._torque_scale(battery_pct)
        obs_noise_scale = 0.15 * self.sigma_m + (gnss_drift_m / self.GNSS_REF_M) * 0.08

        all_states = np.zeros(
            (self.n_trajectories, self.horizon, self.STATE_DIM),
            dtype=np.float32,
        )
        all_actions = np.zeros(
            (self.n_trajectories, self.horizon, self.ACTION_DIM),
            dtype=np.float32,
        )
        all_next_states = np.zeros_like(all_states)
        all_next_states_clean = np.zeros_like(all_states)
        all_obs = np.zeros(
            (self.n_trajectories, self.horizon, self.OBS_DIM),
            dtype=np.float32,
        )

        causal_graph = np.zeros((self.STATE_DIM, self.STATE_DIM + self.ACTION_DIM), dtype=np.float32)
        for i in range(self.N_LINKS):
            causal_graph[i, i] = 1.0
            causal_graph[i, self.N_LINKS + i] = 1.0
            causal_graph[self.N_LINKS + i, i] = 1.0
            causal_graph[self.N_LINKS + i, self.N_LINKS + i] = 1.0
            causal_graph[self.N_LINKS + i, self.STATE_DIM + i] = 1.0

        for traj_idx in range(self.n_trajectories):
            q = rng.uniform(-np.pi, np.pi, size=self.N_LINKS).astype(np.float64)
            dq = rng.standard_normal(self.N_LINKS).astype(np.float64) * 0.5
            action_buffer: list[np.ndarray] = []

            for t in range(self.horizon):
                state = np.concatenate([q, dq]).astype(np.float32)
                tau_raw = rng.standard_normal(self.ACTION_DIM).astype(np.float64) * 0.5
                tau = tau_raw * torque_scale
                action_buffer.append(tau.astype(np.float32))
                if len(action_buffer) > latency_steps + 1:
                    action_buffer.pop(0)
                applied_tau = action_buffer[0].astype(np.float64) if action_buffer else tau

                s_tensor = torch.from_numpy(state).unsqueeze(0)
                vis_feat = vis_encoder(s_tensor).squeeze(0).numpy()
                obs = np.concatenate([state, vis_feat]).astype(np.float32)
                obs += rng.normal(0.0, obs_noise_scale, size=self.OBS_DIM).astype(np.float32)

                all_states[traj_idx, t] = state
                all_actions[traj_idx, t] = tau.astype(np.float32)
                all_obs[traj_idx, t] = obs

                q, dq = self._step(q, dq, applied_tau, rng, add_noise=True)
                all_next_states[traj_idx, t] = np.concatenate([q, dq]).astype(np.float32)

                q_clean, dq_clean = self._step(
                    state[: self.N_LINKS].astype(np.float64),
                    state[self.N_LINKS :].astype(np.float64),
                    applied_tau,
                    rng,
                    add_noise=False,
                )
                all_next_states_clean[traj_idx, t] = np.concatenate([q_clean, dq_clean]).astype(
                    np.float32
                )

        n_train = int(0.8 * self.n_trajectories)
        train_data = {
            "states": all_states[:n_train],
            "actions": all_actions[:n_train],
            "observations": all_obs[:n_train],
            "next_states": all_next_states[:n_train],
            "next_states_clean": all_next_states_clean[:n_train],
        }
        test_data = {
            "states": all_states[n_train:],
            "actions": all_actions[n_train:],
            "observations": all_obs[n_train:],
            "next_states": all_next_states[n_train:],
            "next_states_clean": all_next_states_clean[n_train:],
        }

        inferred_v = inferred_violation_from_telemetry(
            gnss_drift_m,
            cmd_latency_ms,
            battery_pct,
            gnss_ref_m=self.GNSS_REF_M,
            latency_ref_ms=self.LATENCY_REF_MS,
        )
        effective_v = max(
            self.violation_severity,
            inferred_v if self._trace_stats["row_count"] > 0 else 0.0,
        )

        return BenchmarkData(
            train=train_data,
            test=test_data,
            metadata={
                "loop_node": self.loop_node,
                "field_domain": "urc_outdoor",
                "state_dim": self.STATE_DIM,
                "action_dim": self.ACTION_DIM,
                "obs_dim": self.OBS_DIM,
                "visual_feat_dim": self.VISUAL_FEAT_DIM,
                "n_links": self.N_LINKS,
                "sigma_m": self.sigma_m,
                "field_gap": self.field_gap,
                "dt": self.dt,
                "n_trajectories": self.n_trajectories,
                "horizon": self.horizon,
                "violation_severity": self.violation_severity,
                "effective_violation_severity": effective_v,
                "inferred_violation_severity": inferred_v,
                "gnss_drift_m": gnss_drift_m,
                "cmd_latency_ms": cmd_latency_ms,
                "battery_pct": battery_pct,
                "latency_steps": latency_steps,
                "torque_scale": torque_scale,
                "trace_row_count": int(self._trace_stats["row_count"]),
                "trace_mean_violation_severity": self._trace_stats["mean_violation_severity"],
                "causal_graph": causal_graph,
            },
        )
