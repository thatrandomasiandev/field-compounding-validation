"""Field-backed equivariant dynamics DGP (Module 9 / loop node equivariant)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from field_compounding.data.base import BaseFieldDGP, BenchmarkData

LOOP_NODE = "equivariant"


def _random_so3(rng: np.random.Generator) -> np.ndarray:
    H = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _forward_kinematics(q: np.ndarray, link_length: float = 1.0) -> np.ndarray:
    shape = q.shape[:-1]
    positions = np.zeros((*shape, 3, 3), dtype=np.float64)
    cumulative_angle = np.zeros(shape, dtype=np.float64)
    for i in range(3):
        cumulative_angle = cumulative_angle + q[..., i]
        if i == 0:
            positions[..., i, 0] = link_length * np.cos(cumulative_angle)
            positions[..., i, 1] = link_length * np.sin(cumulative_angle)
        else:
            positions[..., i, 0] = positions[..., i - 1, 0] + link_length * np.cos(cumulative_angle)
            positions[..., i, 1] = positions[..., i - 1, 1] + link_length * np.sin(cumulative_angle)
    return positions


def _compute_energy(q, dq, m=1.0, l=1.0, g=9.81):
    return 0.5 * m * l**2 * np.sum(dq**2, axis=-1) - m * g * l * np.sum(np.cos(q), axis=-1)


def _euler_step(q, dq, tau, dt=0.02, m=1.0, l=1.0, g=9.81):
    g_vec = m * g * l * np.sin(q)
    ddq = (tau - g_vec) / (m * l**2)
    dq_new = dq + ddq * dt
    return q + dq_new * dt, dq_new


def load_equivariant_trace_rows(trace_path: str | Path) -> list[dict[str, Any]]:
    path = Path(trace_path)
    if not path.is_file():
        raise FileNotFoundError(trace_path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    equivariant_rows = [row for row in rows if row.get('loop_node') == LOOP_NODE]
    return equivariant_rows if equivariant_rows else rows


def summarize_equivariant_traces(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {'row_count': 0.0, 'mean_gnss_drift_m': 0.0, 'mean_false_positive_rate': 0.0, 'mean_violation_severity': 0.0}
    gnss = np.asarray([float(row.get('gnss_drift', 0.0)) for row in rows])
    fpr = np.asarray([float(row.get('false_positive_rate', 0.0)) for row in rows])
    viol = np.asarray([float(row.get('violation_severity', 0.0)) for row in rows])
    return {
        'row_count': float(len(rows)),
        'mean_gnss_drift_m': float(np.mean(gnss)),
        'mean_false_positive_rate': float(np.mean(fpr)),
        'mean_violation_severity': float(np.mean(np.clip(viol, 0.0, 1.0))),
    }


def symmetry_break_scale(gnss_drift_m, false_positive_rate, violation_severity, *, trace_blend=0.0, trace_stats=None) -> float:
    base = 0.55 * violation_severity + 0.30 * np.clip(gnss_drift_m / 10.0, 0, 1) + 0.15 * np.clip(false_positive_rate, 0, 1)
    if trace_stats and trace_blend > 0.0:
        trace_term = 0.5 * trace_stats.get('mean_violation_severity', 0.0) + 0.3 * np.clip(trace_stats.get('mean_gnss_drift_m', 0.0) / 10.0, 0, 1) + 0.2 * trace_stats.get('mean_false_positive_rate', 0.0)
        base = (1.0 - trace_blend) * base + trace_blend * trace_term
    return float(np.clip(base, 0.0, 1.0))


class EquivariantFieldDGP(BaseFieldDGP):
    def __init__(self, seed=42, violation_severity=0.0, trace_path=None, n_trajectories=500, T=50, dt=0.02, gnss_drift_m=2.4, false_positive_rate=0.08, trace_blend=0.35):
        super().__init__(seed=seed, violation_severity=violation_severity, trace_path=trace_path)
        assert n_trajectories >= 2 and T >= 1 and 0.0 <= trace_blend <= 1.0
        self.n_trajectories, self.T, self.dt = n_trajectories, T, dt
        self.gnss_drift_m = max(0.0, float(gnss_drift_m))
        self.false_positive_rate = float(np.clip(false_positive_rate, 0.0, 1.0))
        self.trace_blend = trace_blend

    @property
    def name(self) -> str:
        return 'equivariant_field_urc'

    @property
    def loop_node(self) -> str:
        return LOOP_NODE

    def _resolve_trace_stats(self):
        if self.trace_path is None:
            return {}, None
        stats = summarize_equivariant_traces(load_equivariant_trace_rows(self.trace_path))
        if stats['row_count'] > 0:
            self.gnss_drift_m = stats['mean_gnss_drift_m']
            self.false_positive_rate = stats['mean_false_positive_rate']
        return stats, str(self.trace_path)

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        trace_stats, trace_source = self._resolve_trace_stats()
        effective_severity = symmetry_break_scale(self.gnss_drift_m, self.false_positive_rate, self.violation_severity, trace_blend=self.trace_blend, trace_stats=trace_stats or None)
        n, T = self.n_trajectories, self.T
        q_all = np.zeros((n, T + 1, 3))
        dq_all = np.zeros((n, T + 1, 3))
        tau_all = np.zeros((n, T, 3))
        q_all[:, 0] = rng.uniform(-np.pi, np.pi, size=(n, 3))
        dq_all[:, 0] = rng.uniform(-2.0, 2.0, size=(n, 3))
        for t in range(T):
            tau = rng.uniform(-5.0, 5.0, size=(n, 3))
            tau_all[:, t] = tau
            q_all[:, t + 1], dq_all[:, t + 1] = _euler_step(q_all[:, t], dq_all[:, t], tau, dt=self.dt)
        positions = _forward_kinematics(q_all[:, :T])
        next_positions = _forward_kinematics(q_all[:, 1:T + 1])
        energies = _compute_energy(q_all[:, :T], dq_all[:, :T])
        features = np.zeros((n, T, 3, 16), dtype=np.float32)
        for i in range(3):
            angle_input = q_all[:, :T, i:i + 1]
            freqs = rng.standard_normal((1, 1, 8)).astype(np.float32) * 2.0
            phases = rng.uniform(0, 2 * np.pi, size=(1, 1, 8)).astype(np.float32)
            features[:, :, i, :] = np.concatenate([np.sin(angle_input @ freqs.reshape(1, 8) + phases.reshape(1, 8)), np.cos(angle_input @ freqs.reshape(1, 8) + phases.reshape(1, 8))], axis=-1)
        q_obs = q_all[:, :T].astype(np.float32).copy()
        dq_obs = dq_all[:, :T].astype(np.float32).copy()
        positions_obs = positions.astype(np.float32).copy()
        energies_obs = energies.astype(np.float32).copy()
        features_obs = features.copy()
        rotation_matrices = np.zeros((n, T, 3, 3))
        n_violated_steps = 0
        gnss_per_traj = rng.uniform(max(0.0, self.gnss_drift_m * 0.5), self.gnss_drift_m * 1.5 + 1e-6, size=n).astype(np.float32)
        fpr_per_traj = rng.uniform(0.0, min(1.0, self.false_positive_rate * 1.5 + 0.05), size=n).astype(np.float32)
        field_proxy = np.zeros((n, T), dtype=np.float32)
        for traj_idx in range(n):
            traj_severity = symmetry_break_scale(float(gnss_per_traj[traj_idx]), float(fpr_per_traj[traj_idx]), self.violation_severity, trace_blend=self.trace_blend, trace_stats=trace_stats or None)
            for t in range(T):
                step_p = float(np.clip(traj_severity + 0.1 * self.false_positive_rate, 0.0, 1.0))
                field_proxy[traj_idx, t] = step_p
                if rng.random() >= step_p:
                    continue
                n_violated_steps += 1
                R = _random_so3(rng)
                rotation_matrices[traj_idx, t] = R
                positions_obs[traj_idx, t] = (R @ positions_obs[traj_idx, t].T).T
                q_obs[traj_idx, t] += rng.normal(0, 1.2 * step_p, size=3).astype(np.float32)
                dq_obs[traj_idx, t] += rng.normal(0, 0.8 * step_p, size=3).astype(np.float32)
                energies_obs[traj_idx, t] += float(rng.normal(0, 2.5 * step_p))
                features_obs[traj_idx, t] += rng.normal(0, 0.8 * step_p, size=(3, 16)).astype(np.float32)
        split = int(0.8 * n)
        train_data = {'positions': positions_obs[:split], 'next_positions': next_positions[:split].astype(np.float32), 'features': features_obs[:split], 'energies': energies_obs[:split], 'torques': tau_all[:split].astype(np.float32), 'q': q_obs[:split], 'dq': dq_obs[:split], 'gnss_drift': np.repeat(gnss_per_traj[:split, None], T, axis=1), 'false_positive_rate': np.repeat(fpr_per_traj[:split, None], T, axis=1), 'field_violation_proxy': field_proxy[:split]}
        test_data = {'positions': positions_obs[split:], 'next_positions': next_positions[split:].astype(np.float32), 'features': features_obs[split:], 'energies': energies_obs[split:], 'torques': tau_all[split:].astype(np.float32), 'q': q_obs[split:], 'dq': dq_obs[split:], 'gnss_drift': np.repeat(gnss_per_traj[split:, None], T, axis=1), 'false_positive_rate': np.repeat(fpr_per_traj[split:, None], T, axis=1), 'field_violation_proxy': field_proxy[split:]}
        inferred_v = symmetry_break_scale(self.gnss_drift_m, self.false_positive_rate, self.violation_severity, trace_blend=1.0 if trace_stats else self.trace_blend, trace_stats=trace_stats or None)
        metadata = {'loop_node': LOOP_NODE, 'field_domain': 'urc_outdoor', 'n_trajectories': n, 'n_train': split, 'n_test': n - split, 'T': T, 'dt': self.dt, 'feature_dim': 16, 'violation_severity': self.violation_severity, 'effective_violation_severity': effective_severity, 'inferred_violation_severity': inferred_v, 'n_violated': n_violated_steps, 'gnss_drift_m': self.gnss_drift_m, 'false_positive_rate': self.false_positive_rate, 'trace_blend': self.trace_blend, 'trace_source': trace_source, 'trace_stats': trace_stats, 'rotation_matrices': rotation_matrices[:split].astype(np.float32), 'field_gap': {'gnss_drift_m': self.gnss_drift_m, 'false_positive_rate': self.false_positive_rate, 'symmetry_break_scale': effective_severity}}
        return BenchmarkData(train=train_data, test=test_data, metadata=metadata)
