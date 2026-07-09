"""Field-backed DGP for scene graph benchmark (Module 10 / loop node scene_graph)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from field_compounding.data.base import BaseFieldDGP, BenchmarkData

_DEFAULT_SCENE_GRAPH_TRACE = {
    "false_positive_rate": 0.15,
    "gnss_drift_m": 6.2,
    "violation_severity": 0.71,
}


def _barabasi_albert(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            adj[i, j] = adj[j, i] = 1.0
    degrees = adj.sum(axis=1)
    for new_node in range(m, n):
        probs = degrees[:new_node].copy()
        if probs.sum() == 0:
            probs = np.ones(new_node)
        probs /= probs.sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
        degrees = adj.sum(axis=1)
    return adj


def _generate_clustered_features(n: int, d: int, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    assignments = rng.integers(0, n_clusters, size=n)
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
    return centers[assignments] + 0.3 * rng.standard_normal((n, d)).astype(np.float32)


def _assign_edge_types(adj: np.ndarray, n_types: int, rng: np.random.Generator) -> np.ndarray:
    n = adj.shape[0]
    edge_labels = np.zeros((n, n), dtype=np.int32)
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    types = rng.integers(0, n_types, size=len(rows))
    for idx, (i, j) in enumerate(zip(rows, cols)):
        edge_labels[i, j] = types[idx] + 1
        edge_labels[j, i] = types[idx] + 1
    return edge_labels


def _temporal_snapshots(n: int, t_steps: int, rng: np.random.Generator) -> np.ndarray:
    snapshots = np.zeros((t_steps, n, n), dtype=np.float32)
    for t in range(t_steps):
        p_t = 0.05 + 0.02 * np.sin(2 * np.pi * t / t_steps)
        rand_mat = rng.random((n, n))
        upper = (rand_mat < p_t).astype(np.float32)
        upper = np.triu(upper, k=1)
        snapshots[t] = upper + upper.T
    return snapshots


def _inject_wl_collisions(features: np.ndarray, rho: float, rng: np.random.Generator):
    n = features.shape[0]
    n_collisions = int(rho * n) // 2
    if n_collisions == 0:
        return features, []
    features = features.copy()
    indices = rng.choice(n, size=2 * n_collisions, replace=False)
    pairs = []
    for k in range(n_collisions):
        i, j = indices[2 * k], indices[2 * k + 1]
        features[j] = features[i]
        pairs.append((int(i), int(j)))
    return features, pairs


def _load_scene_graph_trace_stats(trace_path: str | None) -> dict[str, float]:
    if trace_path is None:
        return dict(_DEFAULT_SCENE_GRAPH_TRACE)
    path = Path(trace_path)
    if not path.exists():
        return dict(_DEFAULT_SCENE_GRAPH_TRACE)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("loop_node") == "scene_graph":
                rows.append(row)
    if not rows:
        return dict(_DEFAULT_SCENE_GRAPH_TRACE)
    return {
        "false_positive_rate": float(np.mean([float(r.get("false_positive_rate", 0.0)) for r in rows])),
        "gnss_drift_m": float(np.mean([float(r.get("gnss_drift", r.get("gnss_drift_m", 0.0))) for r in rows])),
        "violation_severity": float(np.mean([float(r.get("violation_severity", 0.0)) for r in rows])),
        "n_trace_rows": float(len(rows)),
    }


def _inject_spurious_edges(adj: np.ndarray, fpr: float, rng: np.random.Generator):
    n = adj.shape[0]
    adj = adj.copy()
    max_non_edges = n * (n - 1) // 2 - int(np.triu(adj, k=1).sum())
    if max_non_edges <= 0 or fpr <= 0:
        return adj, 0
    n_spurious = int(fpr * max_non_edges)
    added = 0
    attempts = 0
    while added < n_spurious and attempts < n_spurious * 20:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        attempts += 1
        if i == j or adj[i, j] > 0:
            continue
        adj[i, j] = adj[j, i] = 1.0
        added += 1
    return adj, added


def _mislabel_relations(edge_labels, adj, drift_m, n_types, rng):
    edge_labels = edge_labels.copy()
    mislabel_prob = min(0.5, drift_m / 20.0)
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    n_flipped = 0
    for i, j in zip(rows, cols):
        if edge_labels[i, j] == 0:
            continue
        if rng.random() < mislabel_prob:
            new_type = int(rng.integers(1, n_types + 1))
            while new_type == edge_labels[i, j]:
                new_type = int(rng.integers(1, n_types + 1))
            edge_labels[i, j] = edge_labels[j, i] = new_type
            n_flipped += 1
    return edge_labels, n_flipped


class SceneGraphFieldDGP(BaseFieldDGP):
    def __init__(self, seed=42, violation_severity=0.0, trace_path=None, n_nodes=200, m=2, T=12, feature_dim=32, n_edge_types=4, trace_blend=0.35):
        super().__init__(seed=seed, violation_severity=violation_severity, trace_path=trace_path)
        self.n_nodes, self.m, self.T = n_nodes, m, T
        self.feature_dim, self.n_edge_types, self.trace_blend = feature_dim, n_edge_types, trace_blend

    @property
    def name(self):
        return "scene_graph_field"

    @property
    def loop_node(self):
        return "scene_graph"

    def _effective_wl_rho(self, trace_stats):
        blended = (1.0 - self.trace_blend) * self.violation_severity + self.trace_blend * trace_stats["false_positive_rate"]
        return float(np.clip(blended, 0.0, 1.0))

    def _inferred_violation_severity(self, rho, trace_stats, n_spurious, n_flipped):
        n = max(self.n_nodes, 1)
        spurious_rate = n_spurious / max(n * (n - 1) // 2, 1)
        mislabel_rate = n_flipped / max(int(np.triu(np.ones((n, n)), k=1).sum()), 1)
        proxy = 0.5 * rho + 0.3 * trace_stats["false_positive_rate"] + 0.2 * min(1.0, trace_stats["gnss_drift_m"] / 10.0)
        return float(np.clip(max(proxy, spurious_rate, mislabel_rate), 0.0, 1.0))

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        n = self.n_nodes
        trace_stats = _load_scene_graph_trace_stats(self.trace_path)
        rho = self._effective_wl_rho(trace_stats)
        adj = _barabasi_albert(n, self.m, rng)
        adj, n_spurious = _inject_spurious_edges(adj, trace_stats["false_positive_rate"], rng)
        features = _generate_clustered_features(n, self.feature_dim, 5, rng)
        edge_labels = _assign_edge_types(adj, self.n_edge_types, rng)
        edge_labels, n_flipped = _mislabel_relations(edge_labels, adj, trace_stats["gnss_drift_m"], self.n_edge_types, rng)
        temporal = _temporal_snapshots(n, self.T, rng)
        features, wl_pairs = _inject_wl_collisions(features, rho, rng)
        edges = np.array(np.where(np.triu(adj, k=1) > 0)).T
        perm = rng.permutation(len(edges))
        n_train = int(0.70 * len(edges))
        n_val = int(0.15 * len(edges))
        train_edges, val_edges, test_edges = edges[perm[:n_train]], edges[perm[n_train:n_train+n_val]], edges[perm[n_train+n_val:]]

        def _sample_negatives(n_neg):
            neg = []
            while len(neg) < n_neg:
                i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
                if i != j and adj[i, j] == 0:
                    neg.append((i, j))
            return np.array(neg, dtype=np.int64)

        train_neg = _sample_negatives(max(n_train, 1))
        val_neg = _sample_negatives(max(len(val_edges), 1))
        test_neg = _sample_negatives(max(len(test_edges), 1))
        inferred_v = self._inferred_violation_severity(rho, trace_stats, n_spurious, n_flipped)
        train_data = {"adjacency_matrix": adj, "node_features": features, "edge_labels": edge_labels, "positive_edges": train_edges.astype(np.int64), "negative_edges": train_neg, "temporal_snapshots": temporal}
        test_data = {**train_data, "positive_edges": test_edges.astype(np.int64), "negative_edges": test_neg, "val_positive_edges": val_edges.astype(np.int64), "val_negative_edges": val_neg}
        metadata = {"loop_node": self.loop_node, "n_nodes": n, "m": self.m, "T": self.T, "feature_dim": self.feature_dim, "n_edge_types": self.n_edge_types, "violation_severity": self.violation_severity, "effective_wl_rho": rho, "wl_collision_pairs": wl_pairs, "n_wl_collisions": len(wl_pairs), "n_spurious_edges": n_spurious, "n_mislabeled_relations": n_flipped, "trace_stats": trace_stats, "inferred_violation_severity": inferred_v, "field_gap": {"false_positive_rate": trace_stats["false_positive_rate"], "gnss_drift_m": trace_stats["gnss_drift_m"]}}
        return BenchmarkData(train=train_data, test=test_data, metadata=metadata)
