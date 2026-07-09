"""Tests for evaluation adapters modules 9–14."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from field_compounding.data.base import BenchmarkData
from field_compounding.evaluation.module_adapters import (
    TrialContext,
    _field_noise,
    run_trial,
)


class MockEGNN:
    def train_step(self, positions, features, energies):
        self.last_energy = float(np.mean(energies))

    def evaluate(self, positions, features, energies):
        rmse = 0.05 + abs(float(np.mean(energies)) - getattr(self, "last_energy", 0.0))
        return {"rmse": rmse}


class MockGNN:
    def train_step(self, adjacency_matrix, node_features, positive_edges, negative_edges, temporal_snapshots=None):
        self.trained = True

    def evaluate(self, adjacency_matrix, node_features, positive_edges, negative_edges, temporal_snapshots=None):
        return {"auc": 0.85, "ap": 0.8, "hits_at_10": 0.7}

    def evaluate_wl_collisions(self, adjacency_matrix, node_features, pairs, temporal_snapshots=None):
        return 0.6


class MockSymbolicSolver:
    def evaluate(self, facts, rules, query):
        return SimpleNamespace(accuracy=0.9)


class MockFedAvg:
    def train(self, client_data, X_test, y_test):
        return SimpleNamespace(test_accuracy=0.82, gap=0.08)


class MockCBF:
    def rollout(self, start, actions, goal):
        return np.stack([start, start + 0.1])

    def evaluate(self, trajectories, goal):
        return SimpleNamespace(safety_rate=0.95)


def _bench(metadata: dict) -> BenchmarkData:
    rng = np.random.default_rng(0)
    n = 4
    return BenchmarkData(
        train={
            "positions": rng.normal(size=(n, 2, 3, 3)),
            "features": rng.normal(size=(n, 2, 3, 8)),
            "energies": rng.normal(size=(n, 2)),
            "adjacency_matrix": np.eye(5),
            "node_features": rng.normal(size=(5, 8)),
            "positive_edges": np.array([[0, 1], [1, 2]]),
            "negative_edges": np.array([[0, 3], [2, 4]]),
            "facts": np.array([[0, 1, 2, -1], [1, 2, 3, -1]]),
            "client_sizes": np.array([3, 3]),
            "client_X": rng.normal(size=(2, 5, 6)),
            "client_y": rng.integers(0, 2, size=(2, 5)),
            "X_regression": rng.normal(size=(16, 5)),
            "Y_regression": rng.normal(size=(16, 1)).reshape(-1),
            "q": rng.normal(size=(n, 4, 3)),
            "dq": rng.normal(size=(n, 4, 3)),
        },
        test={
            "positions": rng.normal(size=(n, 2, 3, 3)),
            "features": rng.normal(size=(n, 2, 3, 8)),
            "energies": rng.normal(size=(n, 2)),
            "adjacency_matrix": np.eye(5),
            "node_features": rng.normal(size=(5, 8)),
            "positive_edges": np.array([[0, 1], [1, 2]]),
            "negative_edges": np.array([[0, 3], [2, 4]]),
            "X": rng.normal(size=(8, 6)),
            "y": rng.integers(0, 2, size=8),
            "X_regression": rng.normal(size=(8, 5)),
            "Y_regression": rng.normal(size=(8, 1)).reshape(-1),
            "states": rng.normal(size=(n, 3, 4)),
            "actions": rng.normal(size=(n, 3, 2)),
            "q": rng.normal(size=(n, 4, 3)),
            "dq": rng.normal(size=(n, 4, 3)),
        },
        metadata=metadata,
    )


def test_field_noise_increases_with_low_trace_density():
    dense = {"field_trace_density": 0.95, "violation_severity": 0.1}
    sparse = {"field_trace_density": 0.05, "violation_severity": 0.1}
    assert _field_noise(sparse) > _field_noise(dense)


def test_eval_module_09_egnn():
    ctx = TrialContext(9, "EGNN", "cv_robotics.models.egnn.EGNN", {}, fast=True)
    data = _bench({"field_trace_density": 0.8, "violation_severity": 0.2})
    metrics = run_trial(MockEGNN(), ctx, data)
    assert "rmse" in metrics
    assert "normalized_score" in metrics


def test_eval_module_09_hnn_branch():
    ctx = TrialContext(9, "HNN", "cv_robotics.models.hamiltonian_nn.HamiltonianNN", {}, fast=True)

    class MockHNN:
        def evaluate_energy_drift(self, q, dq):
            return {"hnn_energy_drift": 0.08}

    data = _bench({"field_trace_density": 0.5, "violation_severity": 0.3})
    metrics = run_trial(MockHNN(), ctx, data)
    assert "energy_drift" in metrics
    assert metrics["rmse"] > 0.0


def test_eval_module_10_gnn():
    ctx = TrialContext(10, "GCN", "cv_robotics.models.gnn.GCN", {}, fast=True)
    data = _bench(
        {
            "field_trace_density": 0.9,
            "violation_severity": 0.1,
            "wl_collision_pairs": [(0, 1)],
        }
    )
    metrics = run_trial(MockGNN(), ctx, data)
    assert metrics["auc"] > 0.0
    assert "normalized_score" in metrics


def test_sparse_trace_lowers_module_10_auc():
    ctx = TrialContext(10, "GCN", "cv_robotics.models.gnn.GCN", {}, fast=True)
    dense = run_trial(MockGNN(), ctx, _bench({"field_trace_density": 1.0, "wl_collision_pairs": []}))
    sparse = run_trial(MockGNN(), ctx, _bench({"field_trace_density": 0.05, "wl_collision_pairs": []}))
    assert sparse["auc"] <= dense["auc"]


def test_eval_module_11_mc_dropout():
    ctx = TrialContext(11, "MCDropout", "cv_robotics.models.bayesian_nn.MCDropout", {}, fast=True)

    class MockMCD:
        def train_step(self, x, y):
            pass

        def predict(self, x):
            return {"rmse": 0.25}

    metrics = run_trial(MockMCD(), ctx, _bench({"field_trace_density": 0.7}))
    assert "rmse" in metrics
    assert "normalized_score" in metrics


def test_eval_module_12_symbolic():
    ctx = TrialContext(12, "SymbolicSolver", "cv_robotics.models.neurosymbolic.SymbolicSolver", {}, fast=True)
    data = _bench(
        {
            "field_trace_density": 0.6,
            "predicate_names": ["on", "near"],
            "rules": [("on", 2)],
        }
    )
    metrics = run_trial(MockSymbolicSolver(), ctx, data)
    assert metrics["accuracy"] > 0.0


def test_eval_module_13_fedavg():
    ctx = TrialContext(13, "FedAvg", "cv_robotics.models.federated.FedAvg", {}, fast=True)
    metrics = run_trial(MockFedAvg(), ctx, _bench({"field_trace_density": 0.75}))
    assert metrics["test_accuracy"] > 0.0


def test_eval_module_14_cbf():
    ctx = TrialContext(14, "CBF", "cv_robotics.models.safety.CBFSafetyFilter", {}, fast=True)
    data = _bench(
        {
            "field_trace_density": 0.85,
            "goal": [1.0, 0.0],
            "obstacle_positions": [[0.5, 0.5]],
            "obstacle_radii": [0.1],
        }
    )
    metrics = run_trial(MockCBF(), ctx, data)
    assert metrics["safety_rate"] > 0.0


def test_eval_module_03_requires_train_images():
    ctx = TrialContext(3, "NeRF", "cv_robotics.models.nerf.NeRF", {}, fast=True)
    with pytest.raises(KeyError, match="images"):
        run_trial(mock.Mock(), ctx, _bench({}))


def test_unknown_module_raises():
    ctx = TrialContext(99, "X", "x.Y", {}, fast=True)
    with pytest.raises(ValueError, match="unsupported module_id"):
        run_trial(mock.Mock(), ctx, _bench({}))
