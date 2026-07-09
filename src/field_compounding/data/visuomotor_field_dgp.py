"""Field-calibrated visuomotor policy DGP (Module 12 / loop node visuomotor)."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from field_compounding.data.base import BaseFieldDGP, BenchmarkData
LOOP_NODE = "visuomotor"

@dataclass(frozen=True)
class FieldTraceStats:
    row_count: int
    mean_violation_severity: float
    mean_gnss_drift_m: float
    mean_false_positive_rate: float
    mean_cmd_latency_ms: float

def _load_visuomotor_trace_stats(trace_path: str | Path) -> FieldTraceStats:
    path = Path(trace_path)
    if not path.is_file():
        raise FileNotFoundError(f"trace file not found: {path}")
    violations, gnss, fpr, latencies = [], [], [], []
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped: continue
            row = json.loads(stripped)
            if str(row.get("loop_node", "")) != LOOP_NODE: continue
            violations.append(float(np.clip(row["violation_severity"], 0.0, 1.0)))
            gnss.append(max(0.0, float(row["gnss_drift"])))
            fpr.append(float(np.clip(row["false_positive_rate"], 0.0, 1.0)))
            latencies.append(max(0.0, float(row["cmd_latency_ms"])) if "cmd_latency_ms" in row else 25.0 + float(row["gnss_drift"]) * 12.0)
    if not violations:
        raise ValueError(f"{path}: no rows with loop_node={LOOP_NODE!r}")
    return FieldTraceStats(len(violations), float(np.mean(violations)), float(np.mean(gnss)), float(np.mean(fpr)), float(np.mean(latencies)))

class _FixedMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, seed=0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out_dim))
        gen = torch.Generator().manual_seed(seed)
        for param in self.net.parameters():
            nn.init.normal_(param, 0.0, 0.3, generator=gen); param.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x): return self.net(x)

def _beta_from_violation(violation_severity, beta):
    coverage = max(0.3, 1.0 - violation_severity) if beta is None else max(0.3, min(1.0, beta))
    return coverage, max(0.0, min(1.0, 1.0 - (coverage - 0.3) / 0.7))

class VisuomotorFieldDGP(BaseFieldDGP):
    STATE_DIM, ACTION_DIM, SCENE_FEAT_DIM = 6, 2, 32
    OBS_DIM = STATE_DIM + SCENE_FEAT_DIM
    def __init__(self, seed=42, violation_severity=0.0, trace_path=None, beta=None, field_gap=0.0, n_modes=2, n_trajectories=500, horizon=50):
        coverage, vs = _beta_from_violation(violation_severity, beta)
        super().__init__(seed=seed, violation_severity=vs, trace_path=trace_path)
        self.beta, self.field_gap = coverage, float(np.clip(field_gap, 0.0, 1.0))
        self.n_modes, self.n_trajectories, self.horizon = int(n_modes), int(n_trajectories), int(horizon)
        self._trace_stats = _load_visuomotor_trace_stats(trace_path) if trace_path else None
    @property
    def name(self): return "visuomotor_field"
    @property
    def loop_node(self): return LOOP_NODE
    def _field_calibration(self):
        if self._trace_stats:
            s = self._trace_stats
            gap = float(np.clip(max(self.field_gap, 0.35 * s.mean_violation_severity + 0.05 * s.mean_gnss_drift_m), 0.0, 1.0))
            return gap, s.mean_gnss_drift_m, s.mean_cmd_latency_ms, float(np.clip(0.5 * self.violation_severity + 0.5 * s.mean_violation_severity, 0.0, 1.0))
        gap = float(np.clip(self.field_gap + 0.4 * self.violation_severity, 0.0, 1.0))
        return gap, 1.5 + 6.0 * gap, 20.0 + 80.0 * gap, self.violation_severity
    def _generate(self):
        rng = np.random.default_rng(self.seed)
        field_gap, gnss_drift_m, cmd_latency_ms, inferred_v = self._field_calibration()
        dynamics = _FixedMLP(self.STATE_DIM + self.ACTION_DIM, self.STATE_DIM, seed=self.seed)
        scene_proj = _FixedMLP(self.STATE_DIM, self.SCENE_FEAT_DIM, seed=self.seed + 999)
        expert_policies = [_FixedMLP(self.STATE_DIM, self.ACTION_DIM, seed=self.seed + 100 + m) for m in range(self.n_modes)]
        obs_noise_scale = 0.02 + 0.08 * field_gap + 0.01 * gnss_drift_m
        action_lag_scale = (cmd_latency_ms / 100.0) * field_gap
        init_range = np.pi * self.beta
        chunks = [[], [], [], [], []]
        for traj_idx in range(self.n_trajectories):
            mode_k = traj_idx % self.n_modes; policy = expert_policies[mode_k]
            state = rng.uniform(-init_range, init_range, size=self.STATE_DIM).astype(np.float32)
            prev_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
            traj_obs = np.zeros((self.horizon, self.OBS_DIM), dtype=np.float32)
            traj_actions = np.zeros((self.horizon, self.ACTION_DIM), dtype=np.float32)
            traj_rewards = np.zeros(self.horizon, dtype=np.float32)
            traj_latency = np.zeros(self.horizon, dtype=np.float32)
            for step in range(self.horizon):
                st = torch.from_numpy(state).unsqueeze(0)
                scene_feat = scene_proj(st).squeeze(0).numpy() + rng.normal(0.0, obs_noise_scale, size=(self.SCENE_FEAT_DIM,)).astype(np.float32)
                traj_obs[step] = np.concatenate([state, scene_feat])
                action = policy(st).squeeze(0).numpy()
                if action_lag_scale > 0.0:
                    action = (1.0 - action_lag_scale) * action + action_lag_scale * prev_action
                    action += rng.normal(0.0, 0.05 * field_gap, size=action.shape).astype(np.float32)
                traj_actions[step], traj_latency[step] = action, cmd_latency_ms * (1.0 + 0.1 * rng.standard_normal())
                prev_action = action.copy()
                state = dynamics(torch.from_numpy(np.concatenate([state, action])).unsqueeze(0)).squeeze(0).numpy()
                traj_rewards[step] = -float(np.sum(state[:3] ** 2))
            chunks[0].append(traj_obs); chunks[1].append(traj_actions); chunks[2].append(traj_rewards)
            chunks[3].append(np.full(self.horizon, mode_k, dtype=np.int64)); chunks[4].append(traj_latency)
        obs, actions, rewards, modes, latency = [np.concatenate(c, axis=0) for c in chunks]
        n_train = int(0.8 * self.n_trajectories * self.horizon)
        train = {"observations": obs[:n_train], "actions": actions[:n_train], "rewards": rewards[:n_train], "modes": modes[:n_train], "cmd_latency_ms": latency[:n_train]}
        test = {"observations": obs[n_train:], "actions": actions[n_train:], "rewards": rewards[n_train:], "modes": modes[n_train:], "cmd_latency_ms": latency[n_train:]}
        meta = {"loop_node": self.loop_node, "state_dim": self.STATE_DIM, "action_dim": self.ACTION_DIM, "obs_dim": self.OBS_DIM, "scene_feat_dim": self.SCENE_FEAT_DIM, "beta": self.beta, "field_gap": field_gap, "n_modes": self.n_modes, "n_trajectories": self.n_trajectories, "horizon": self.horizon, "violation_severity": self.violation_severity, "inferred_violation_severity": inferred_v, "mean_gnss_drift_m": gnss_drift_m, "mean_cmd_latency_ms": cmd_latency_ms, "obs_noise_scale": obs_noise_scale, "trace_path": self.trace_path, "trace_rows_used": self._trace_stats.row_count if self._trace_stats else 0, "source": "trace_calibrated" if self._trace_stats else "synthetic"}
        return BenchmarkData(train=train, test=test, metadata=meta)
