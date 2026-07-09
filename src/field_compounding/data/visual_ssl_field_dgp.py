"""Field-backed visual self-supervised learning DGP (Module 4).

Extends Module 11 ``VisualSSLDGP`` with URC trace replay. Field ``gnss_drift``
induces embedding drift; ``false_positive_rate`` inflates contrastive augment
noise and sparsifies supervision; ``violation_severity`` reduces labeled samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from field_compounding.data.base import BaseFieldDGP, BenchmarkData

FIELD_LOG_REQUIRED_KEYS = frozenset(
    {
        "timestamp",
        "loop_node",
        "violation_severity",
        "recovery",
        "gnss_drift",
        "false_positive_rate",
    }
)

VALID_LOOP_NODES = frozenset(
    {
        "scene_repr",
        "visual_ssl",
        "sim_to_real",
        "visuomotor",
        "causal",
        "world_model",
        "equivariant",
        "scene_graph",
        "uncertainty",
        "neurosymbolic",
        "federated",
        "safety",
    }
)


@dataclass
class _TraceRow:
    timestamp: str
    loop_node: str
    violation_severity: float
    recovery: bool
    gnss_drift: float
    false_positive_rate: float

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> _TraceRow:
        missing = FIELD_LOG_REQUIRED_KEYS - row.keys()
        if missing:
            raise KeyError(f"missing required field_log keys: {sorted(missing)}")
        loop_node = str(row["loop_node"])
        if loop_node not in VALID_LOOP_NODES:
            raise ValueError(f"unknown loop_node: {loop_node!r}")
        return cls(
            timestamp=str(row["timestamp"]),
            loop_node=loop_node,
            violation_severity=float(np.clip(float(row["violation_severity"]), 0.0, 1.0)),
            recovery=bool(row["recovery"]),
            gnss_drift=max(0.0, float(row["gnss_drift"])),
            false_positive_rate=float(np.clip(float(row["false_positive_rate"]), 0.0, 1.0)),
        )


def load_visual_ssl_trace_rows(trace_path: str | Path) -> list[_TraceRow]:
    """Load ``loop_node == visual_ssl`` rows from a URC JSONL trace."""
    path = Path(trace_path)
    if not path.is_file():
        raise FileNotFoundError(f"trace file not found: {path}")

    rows: list[_TraceRow] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            rows.append(_TraceRow.from_dict(payload))

    ssl_rows = [row for row in rows if row.loop_node == "visual_ssl"]
    return ssl_rows if ssl_rows else rows


def summarize_visual_ssl_traces(rows: list[_TraceRow]) -> dict[str, float]:
    """Aggregate telemetry from visual-SSL field trace rows."""
    if not rows:
        return {
            "row_count": 0.0,
            "mean_gnss_drift_m": 0.0,
            "mean_false_positive_rate": 0.0,
            "mean_violation_severity": 0.0,
        }

    gnss = np.array([row.gnss_drift for row in rows], dtype=np.float64)
    fpr = np.array([row.false_positive_rate for row in rows], dtype=np.float64)
    severity = np.array([row.violation_severity for row in rows], dtype=np.float64)

    return {
        "row_count": float(len(rows)),
        "mean_gnss_drift_m": float(np.mean(gnss)),
        "mean_false_positive_rate": float(np.mean(fpr)),
        "mean_violation_severity": float(np.mean(severity)),
    }


def labeled_fraction_from_field_stress(
    violation_severity: float,
    gnss_drift_m: float,
    false_positive_rate: float,
    *,
    n_labeled_max: int = 3000,
) -> tuple[int, float]:
    """Map field stress to labeled count and effective violation in ``[0, 1]``."""
    gnss_term = float(np.clip(gnss_drift_m / 10.0, 0.0, 1.0))
    effective_v = float(
        np.clip(
            0.5 * violation_severity + 0.3 * gnss_term + 0.2 * false_positive_rate,
            0.0,
            1.0,
        )
    )
    min_labeled = max(1, int(0.05 * n_labeled_max))
    n_labeled = int(n_labeled_max * (1.0 - effective_v))
    return max(min_labeled, n_labeled), effective_v


def augment_sigma_from_fpr(
    base_sigma: float,
    false_positive_rate: float,
    *,
    trace_coupling: float = 1.0,
) -> float:
    """Inflate contrastive augment noise when detector FPR is high."""
    coupling = float(np.clip(trace_coupling, 0.0, 1.0))
    return base_sigma * (1.0 + coupling * false_positive_rate * 4.0)


def feature_drift_scale(gnss_drift_m: float, *, trace_coupling: float = 1.0) -> float:
    """Scale embedding drift from GNSS localization error (meters)."""
    coupling = float(np.clip(trace_coupling, 0.0, 1.0))
    return coupling * float(np.clip(gnss_drift_m / 10.0, 0.0, 1.0))


def _generate_synthetic_visual_ssl_traces(n_rows: int, seed: int) -> list[_TraceRow]:
    """Deterministic synthetic URC rows biased to the visual_ssl loop node."""
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")

    rng = np.random.default_rng(seed)
    base_time = datetime(2025, 5, 18, 14, 0, 0, tzinfo=timezone.utc)
    rows: list[_TraceRow] = []

    for idx in range(n_rows):
        severity = float(rng.beta(2.2, 3.8))
        gnss = float(rng.exponential(2.8))
        fpr = float(rng.beta(1.8, 9.0))
        recovery = bool(severity < 0.55 or rng.random() > 0.35)
        ts = base_time + timedelta(minutes=idx * 3, seconds=int(rng.integers(0, 45)))
        rows.append(
            _TraceRow(
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                loop_node="visual_ssl",
                violation_severity=severity,
                recovery=recovery,
                gnss_drift=gnss,
                false_positive_rate=fpr,
            )
        )
    return rows


def _load_visual_ssl_rows(
    trace_path: str | None,
    seed: int,
    min_rows: int,
) -> list[_TraceRow]:
    if trace_path is not None:
        return load_visual_ssl_trace_rows(trace_path)
    return _generate_synthetic_visual_ssl_traces(max(min_rows, 120), seed=seed)


def save_visual_ssl_trace(rows: list[_TraceRow], path: str | Path) -> Path:
    """Write trace rows to JSONL (test helper)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            payload = {
                "timestamp": row.timestamp,
                "loop_node": row.loop_node,
                "violation_severity": row.violation_severity,
                "recovery": row.recovery,
                "gnss_drift": row.gnss_drift,
                "false_positive_rate": row.false_positive_rate,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return out


class VisualSSLFieldDGP(BaseFieldDGP):
    """Visual SSL DGP with optional URC field trace conditioning."""

    def __init__(
        self,
        seed: int = 42,
        violation_severity: float = 0.0,
        trace_path: str | None = None,
        n_samples: int = 3000,
        n_clusters: int = 8,
        feature_dim: int = 10,
        cluster_spacing: float = 4.0,
        cluster_sigma: float = 1.0,
        augment_sigma: float = 0.1,
        mae_mask_ratio: float = 0.5,
        n_point_clouds: int = 1000,
        pts_per_cloud: int = 512,
        point_noise_sigma: float = 0.01,
        temporal_length: int = 20,
        ema_momentum: float = 0.996,
        train_ratio: float = 0.8,
        trace_coupling: float = 1.0,
    ) -> None:
        super().__init__(seed=seed, violation_severity=violation_severity, trace_path=trace_path)
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.cluster_spacing = cluster_spacing
        self.cluster_sigma = cluster_sigma
        self.augment_sigma = augment_sigma
        self.mae_mask_ratio = mae_mask_ratio
        self.n_point_clouds = n_point_clouds
        self.pts_per_cloud = pts_per_cloud
        self.point_noise_sigma = point_noise_sigma
        self.temporal_length = temporal_length
        self.ema_momentum = ema_momentum
        self.train_ratio = train_ratio
        self.trace_coupling = float(np.clip(trace_coupling, 0.0, 1.0))

    @property
    def name(self) -> str:
        return "visual_ssl_field_urc"

    @property
    def loop_node(self) -> str:
        return "visual_ssl"

    def _trace_stress(self, trace_rows: list[_TraceRow]) -> dict[str, float]:
        stats = summarize_visual_ssl_traces(trace_rows)
        return {
            "mean_gnss_drift_m": stats["mean_gnss_drift_m"],
            "mean_false_positive_rate": stats["mean_false_positive_rate"],
            "mean_trace_violation_severity": stats["mean_violation_severity"],
            "n_trace_rows": stats["row_count"],
        }

    def _generate_cluster_centers(self, rng: np.random.Generator) -> np.ndarray:
        centers = np.zeros((self.n_clusters, self.feature_dim))
        for i in range(self.n_clusters):
            centers[i] = rng.standard_normal(self.feature_dim)
            centers[i] = (
                centers[i] / np.linalg.norm(centers[i]) * self.cluster_spacing * (i + 1) / self.n_clusters
            )
        return centers

    def _generate_features_and_labels(
        self,
        rng: np.random.Generator,
        n: int,
        centers: np.ndarray,
        drift_scale: float,
        trace_rows: list[_TraceRow],
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, self.n_clusters, size=n)
        features = np.zeros((n, self.feature_dim), dtype=np.float32)
        for i in range(n):
            row = trace_rows[i % len(trace_rows)]
            sample_drift = feature_drift_scale(row.gnss_drift, trace_coupling=self.trace_coupling)
            drift = rng.normal(0.0, self.cluster_sigma + drift_scale + sample_drift, self.feature_dim)
            features[i] = centers[labels[i]] + drift
        return features, labels

    def _generate_augmented_pairs(
        self,
        features: np.ndarray,
        rng: np.random.Generator,
        augment_sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        view1 = features + rng.normal(0, augment_sigma, features.shape).astype(np.float32)
        view2 = features + rng.normal(0, augment_sigma, features.shape).astype(np.float32)
        return view1, view2

    def _generate_masked_inputs(
        self,
        features: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        n, d = features.shape
        masks = rng.random((n, d)) < self.mae_mask_ratio
        masked = features.copy()
        masked[masks] = 0.0
        return masked.astype(np.float32), masks.astype(np.float32)

    def _generate_point_clouds(
        self,
        rng: np.random.Generator,
        trace_rows: list[_TraceRow],
        effective_v: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        clouds = np.zeros((self.n_point_clouds, self.pts_per_cloud, 3), dtype=np.float32)
        labels = np.zeros(self.n_point_clouds, dtype=np.int64)
        n_shapes = 4
        v = float(effective_v)
        noise_scale = self.point_noise_sigma * (1.0 + 5.0 * v)
        shrink = 1.0 - 0.45 * v
        label_flip_p = 0.15 + 0.55 * v

        for i in range(self.n_point_clouds):
            row = trace_rows[i % len(trace_rows)]
            fpr_boost = 1.0 + self.trace_coupling * row.false_positive_rate
            shape_type = i % n_shapes
            labels[i] = shape_type
            if rng.random() < label_flip_p * fpr_boost:
                labels[i] = int(rng.integers(0, n_shapes))

            if shape_type == 0:
                phi = rng.uniform(0, 2 * np.pi, self.pts_per_cloud)
                cos_theta = rng.uniform(-1, 1, self.pts_per_cloud)
                theta = np.arccos(cos_theta)
                clouds[i, :, 0] = np.sin(theta) * np.cos(phi)
                clouds[i, :, 1] = np.sin(theta) * np.sin(phi)
                clouds[i, :, 2] = cos_theta
            elif shape_type == 1:
                face = rng.integers(0, 6, self.pts_per_cloud)
                coords = rng.uniform(-1, 1, (self.pts_per_cloud, 2))
                for j in range(self.pts_per_cloud):
                    f = face[j]
                    axis = f // 2
                    sign = 2 * (f % 2) - 1
                    other_axes = [a for a in range(3) if a != axis]
                    clouds[i, j, axis] = sign
                    clouds[i, j, other_axes[0]] = coords[j, 0]
                    clouds[i, j, other_axes[1]] = coords[j, 1]
            elif shape_type == 2:
                theta = rng.uniform(0, 2 * np.pi, self.pts_per_cloud)
                z = rng.uniform(-1, 1, self.pts_per_cloud)
                clouds[i, :, 0] = np.cos(theta)
                clouds[i, :, 1] = np.sin(theta)
                clouds[i, :, 2] = z
            else:
                t = rng.uniform(0, 1, self.pts_per_cloud)
                theta = rng.uniform(0, 2 * np.pi, self.pts_per_cloud)
                clouds[i, :, 0] = t * np.cos(theta)
                clouds[i, :, 1] = t * np.sin(theta)
                clouds[i, :, 2] = 1 - t

            gnss_noise = feature_drift_scale(row.gnss_drift, trace_coupling=self.trace_coupling)
            clouds[i] *= shrink
            clouds[i] += rng.normal(0, noise_scale * (1.0 + gnss_noise), (self.pts_per_cloud, 3))

        n_labeled = max(int(0.05 * self.n_point_clouds), int(self.n_point_clouds * (1.0 - v)))
        label_mask = np.zeros(self.n_point_clouds, dtype=np.float32)
        label_mask[:n_labeled] = 1.0
        rng.shuffle(label_mask)

        return clouds.astype(np.float32), labels, label_mask

    def _generate_temporal_sequences(
        self,
        features: np.ndarray,
        rng: np.random.Generator,
        drift_scale: float,
    ) -> np.ndarray:
        n = features.shape[0]
        n_sequences = n // self.temporal_length
        sequences = np.zeros((n_sequences, self.temporal_length, self.feature_dim), dtype=np.float32)
        for i in range(n_sequences):
            base = features[i * self.temporal_length]
            for t in range(self.temporal_length):
                drift = rng.normal(0, (0.05 + drift_scale) * (t + 1), self.feature_dim)
                sequences[i, t] = base + drift
        return sequences

    def _generate_dino_pairs(
        self,
        features: np.ndarray,
        rng: np.random.Generator,
        augment_sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        student_features = features + rng.normal(0, augment_sigma * 2, features.shape).astype(np.float32)
        teacher_features = (
            self.ema_momentum * features + (1 - self.ema_momentum) * student_features
        ).astype(np.float32)
        return student_features, teacher_features

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        trace_rows = _load_visual_ssl_rows(self.trace_path, seed=self.seed, min_rows=self.n_samples)
        trace_stats = self._trace_stress(trace_rows)

        mean_gnss = trace_stats["mean_gnss_drift_m"]
        mean_fpr = trace_stats["mean_false_positive_rate"]
        drift_scale = feature_drift_scale(mean_gnss, trace_coupling=self.trace_coupling)
        augment_sigma = augment_sigma_from_fpr(
            self.augment_sigma,
            mean_fpr,
            trace_coupling=self.trace_coupling,
        )

        n_labeled, effective_v = labeled_fraction_from_field_stress(
            self.violation_severity,
            mean_gnss,
            mean_fpr,
            n_labeled_max=self.n_samples,
        )

        centers = self._generate_cluster_centers(rng)
        features, labels = self._generate_features_and_labels(
            rng,
            self.n_samples,
            centers,
            drift_scale,
            trace_rows,
        )

        view1, view2 = self._generate_augmented_pairs(features, rng, augment_sigma)
        masked_inputs, masks = self._generate_masked_inputs(features, rng)
        temporal_sequences = self._generate_temporal_sequences(features, rng, drift_scale)
        student_features, teacher_features = self._generate_dino_pairs(features, rng, augment_sigma)
        point_clouds, pc_labels, pc_label_mask = self._generate_point_clouds(
            rng,
            trace_rows,
            effective_v,
        )

        label_mask = np.zeros(self.n_samples, dtype=np.float32)
        label_mask[:n_labeled] = 1.0
        rng.shuffle(label_mask)

        n_train = int(self.n_samples * self.train_ratio)
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, self.n_samples)
        n_pc_train = int(self.n_point_clouds * self.train_ratio)

        train_data: dict[str, np.ndarray] = {
            "features": features[train_idx],
            "labels": labels[train_idx].astype(np.int64),
            "label_mask": label_mask[train_idx],
            "augmented_view1": view1[train_idx],
            "augmented_view2": view2[train_idx],
            "masked_inputs": masked_inputs[train_idx],
            "masks": masks[train_idx],
            "temporal_sequences": temporal_sequences[: n_train // self.temporal_length],
            "student_features": student_features[train_idx],
            "teacher_features": teacher_features[train_idx],
            "point_clouds": point_clouds[:n_pc_train],
            "point_cloud_labels": pc_labels[:n_pc_train],
            "point_cloud_label_mask": pc_label_mask[:n_pc_train],
        }

        test_data: dict[str, np.ndarray] = {
            "features": features[test_idx],
            "labels": labels[test_idx].astype(np.int64),
            "label_mask": label_mask[test_idx],
            "augmented_view1": view1[test_idx],
            "augmented_view2": view2[test_idx],
            "masked_inputs": masked_inputs[test_idx],
            "masks": masks[test_idx],
            "temporal_sequences": temporal_sequences[n_train // self.temporal_length :],
            "student_features": student_features[test_idx],
            "teacher_features": teacher_features[test_idx],
            "point_clouds": point_clouds[n_pc_train:],
            "point_cloud_labels": pc_labels[n_pc_train:],
            "point_cloud_label_mask": pc_label_mask[n_pc_train:],
        }

        inferred_violation = float(
            np.clip(
                0.5 * trace_stats["mean_trace_violation_severity"] + 0.5 * mean_fpr,
                0.0,
                1.0,
            )
        )

        trace_source = str(Path(self.trace_path)) if self.trace_path else "synthetic"

        metadata: dict[str, Any] = {
            "loop_node": self.loop_node,
            "field_domain": "urc_outdoor",
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "feature_dim": self.feature_dim,
            "n_labeled": n_labeled,
            "n_point_clouds": self.n_point_clouds,
            "n_point_cloud_labeled": int(pc_label_mask.sum()),
            "pts_per_cloud": self.pts_per_cloud,
            "temporal_length": self.temporal_length,
            "violation_severity": self.violation_severity,
            "effective_violation_severity": effective_v,
            "augment_sigma": augment_sigma,
            "feature_drift_scale": drift_scale,
            "trace_coupling": self.trace_coupling,
            "trace_path": self.trace_path,
            "trace_source": trace_source,
            "trace_row_count": int(trace_stats["n_trace_rows"]),
            "gnss_drift_m": mean_gnss,
            "false_positive_rate": mean_fpr,
            "inferred_violation_severity": inferred_violation,
            "cluster_centers": centers,
            "field_telemetry": trace_stats,
        }

        return BenchmarkData(train=train_data, test=test_data, metadata=metadata)
