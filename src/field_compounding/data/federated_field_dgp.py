"""Field-backed DGP for federated visual learning across robot fleets (Module 13).

Extends Module 11 ``RobotFleetDGP`` with URC field trace telemetry: GNSS drift
scales per-client feature shift, command latency weights stale updates, and
false-positive rate injects label noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from field_compounding.data.base import BaseFieldDGP, BenchmarkData

DEFAULT_K_CLIENTS: int = 5
DEFAULT_N_CLASSES: int = 10
DEFAULT_D_FEATURES: int = 20
DEFAULT_N_TOTAL: int = 5000
DEFAULT_NOISE_SIGMA: float = 1.0
DEFAULT_CLIP_NORM: float = 1.0
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_D_UPDATE: int = 256
LOOP_NODE: str = "federated"


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


def summarize_federated_traces(rows: list[dict[str, Any]]) -> dict[str, float]:
    fed_rows = [r for r in rows if r.get("loop_node") == LOOP_NODE]
    if not fed_rows:
        return {
            "row_count": 0.0,
            "mean_gnss_drift_m": 0.0,
            "mean_false_positive_rate": 0.0,
            "mean_cmd_latency_ms": 0.0,
            "mean_battery_pct": 100.0,
            "mean_violation_severity": 0.0,
        }
    gnss = np.array([float(r["gnss_drift"]) for r in fed_rows], dtype=np.float64)
    fpr = np.array([float(r["false_positive_rate"]) for r in fed_rows], dtype=np.float64)
    latency = np.array([float(r.get("cmd_latency_ms", 50.0)) for r in fed_rows], dtype=np.float64)
    battery = np.array([float(r.get("battery_pct", 100.0)) for r in fed_rows], dtype=np.float64)
    sev = np.array([float(r["violation_severity"]) for r in fed_rows], dtype=np.float64)
    return {
        "row_count": float(len(fed_rows)),
        "mean_gnss_drift_m": float(np.mean(gnss)),
        "mean_false_positive_rate": float(np.mean(fpr)),
        "mean_cmd_latency_ms": float(np.mean(latency)),
        "mean_battery_pct": float(np.mean(battery)),
        "mean_violation_severity": float(np.mean(sev)),
    }


def _class_conditional_features(
    label: int,
    n_classes: int,
    d: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.zeros(d, dtype=np.float32)
    mean[label % d] = 2.0
    mean[(label * 3 + 1) % d] = 1.5
    cov_diag = 0.5 + 0.1 * (label / n_classes)
    return (rng.standard_normal(d) * cov_diag + mean).astype(np.float32)


def _dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n_classes = int(labels.max()) + 1
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)
        splits = np.split(class_idx, proportions[:-1])
        for k in range(n_clients):
            client_indices[k].extend(splits[k].tolist())
    return [np.array(idx, dtype=np.int64) for idx in client_indices]


def _label_entropy(distributions: np.ndarray) -> float:
    return float(np.var(distributions, axis=0).mean())


class FederatedFieldDGP(BaseFieldDGP):
    """Federated field learning DGP with trace-backed heterogeneity."""

    def __init__(
        self,
        seed: int = 42,
        violation_severity: float = 0.0,
        trace_path: str | None = None,
        n_clients: int = DEFAULT_K_CLIENTS,
        n_classes: int = DEFAULT_N_CLASSES,
        d_features: int = DEFAULT_D_FEATURES,
        n_total: int = DEFAULT_N_TOTAL,
        alpha: float | None = None,
        noise_sigma: float = DEFAULT_NOISE_SIGMA,
        clip_norm: float = DEFAULT_CLIP_NORM,
        batch_size: int = DEFAULT_BATCH_SIZE,
        d_update: int = DEFAULT_D_UPDATE,
    ) -> None:
        if alpha is None:
            alpha = max(0.1, 5.0 - 4.9 * float(violation_severity))
        severity = float(np.clip((5.0 - alpha) / (5.0 - 0.1), 0.0, 1.0))
        super().__init__(seed=seed, violation_severity=severity, trace_path=trace_path)
        self.n_clients = n_clients
        self.n_classes = n_classes
        self.d_features = d_features
        self.n_total = n_total
        self.alpha = float(alpha)
        self.noise_sigma = noise_sigma
        self.clip_norm = clip_norm
        self.batch_size = batch_size
        self.d_update = d_update
        self._trace_stats = self._load_trace_stats()

    @property
    def name(self) -> str:
        return "federated_field_learning"

    @property
    def loop_node(self) -> str:
        return LOOP_NODE

    def _load_trace_stats(self) -> dict[str, float]:
        if self.trace_path is None:
            return summarize_federated_traces([])
        path = Path(self.trace_path)
        if not path.is_file():
            raise FileNotFoundError(f"trace file not found: {path}")
        return summarize_federated_traces(load_urc_trace_rows(path))

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

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        field = self._field_proxies()

        drift_scale = 1.0 + 0.08 * field["gnss_drift_m"]
        label_flip_prob = min(0.35, field["false_positive_rate"] * 0.5)
        latency_weight = float(np.clip(field["cmd_latency_ms"] / 100.0, 0.1, 1.5))

        labels = rng.integers(0, self.n_classes, size=self.n_total).astype(np.int64)
        features = np.vstack([
            _class_conditional_features(int(y), self.n_classes, self.d_features, rng)
            for y in labels
        ])
        if label_flip_prob > 0.0:
            flip_mask = rng.random(self.n_total) < label_flip_prob
            n_flip = int(flip_mask.sum())
            if n_flip > 0:
                labels[flip_mask] = (
                    labels[flip_mask] + rng.integers(1, self.n_classes, size=n_flip)
                ) % self.n_classes

        client_idx = _dirichlet_partition(labels, self.n_clients, self.alpha, rng)
        train_X_parts: list[np.ndarray] = []
        train_y_parts: list[np.ndarray] = []
        test_X_parts: list[np.ndarray] = []
        test_y_parts: list[np.ndarray] = []
        label_distributions: list[np.ndarray] = []
        client_latency: list[float] = []

        for k, idx in enumerate(client_idx):
            client_drift = rng.normal(0.0, 0.15 * drift_scale, size=self.d_features).astype(
                np.float32
            )
            X_k = features[idx] + client_drift
            y_k = labels[idx]
            n_k = len(X_k)
            n_test = max(1, int(0.2 * n_k))
            perm = rng.permutation(n_k)
            test_X_parts.append(X_k[perm[:n_test]])
            test_y_parts.append(y_k[perm[:n_test]])
            train_X_parts.append(X_k[perm[n_test:]])
            train_y_parts.append(y_k[perm[n_test:]])
            dist = np.bincount(y_k, minlength=self.n_classes).astype(np.float32)
            dist /= max(dist.sum(), 1.0)
            label_distributions.append(dist)
            client_latency.append(latency_weight * (1.0 + 0.1 * k))

        max_train = max(len(x) for x in train_X_parts)
        padded_X = np.zeros((self.n_clients, max_train, self.d_features), dtype=np.float32)
        padded_y = np.full((self.n_clients, max_train), -1, dtype=np.int64)
        client_sizes = np.zeros(self.n_clients, dtype=np.int32)
        for k in range(self.n_clients):
            n_k = len(train_X_parts[k])
            padded_X[k, :n_k] = train_X_parts[k]
            padded_y[k, :n_k] = train_y_parts[k]
            client_sizes[k] = n_k

        label_dist_array = np.stack(label_distributions, axis=0)
        metadata: dict[str, Any] = {
            "loop_node": self.loop_node,
            "field_domain": "urc_outdoor",
            "n_clients": self.n_clients,
            "n_classes": self.n_classes,
            "d_features": self.d_features,
            "n_total": self.n_total,
            "alpha": self.alpha,
            "noise_sigma": self.noise_sigma,
            "clip_norm": self.clip_norm,
            "batch_size": self.batch_size,
            "d_update": self.d_update,
            "violation_severity": self.violation_severity,
            "label_heterogeneity": _label_entropy(label_dist_array),
            "label_flip_prob": label_flip_prob,
            "mlp_architecture": [self.d_features, 128, 64, self.n_classes],
            "trace_row_count": int(self._trace_stats["row_count"]),
            "trace_mean_violation_severity": self._trace_stats["mean_violation_severity"],
            "inferred_violation_severity": field["inferred_violation_severity"],
            "gnss_drift_m": field["gnss_drift_m"],
            "false_positive_rate": field["false_positive_rate"],
            "cmd_latency_ms": field["cmd_latency_ms"],
            "battery_pct": field["battery_pct"],
        }
        return BenchmarkData(
            train={
                "client_X": padded_X,
                "client_y": padded_y,
                "client_sizes": client_sizes,
                "label_distributions": label_dist_array,
                "client_latency_weight": np.array(client_latency, dtype=np.float32),
            },
            test={
                "X": np.concatenate(test_X_parts, axis=0),
                "y": np.concatenate(test_y_parts, axis=0),
            },
            metadata=metadata,
        )
