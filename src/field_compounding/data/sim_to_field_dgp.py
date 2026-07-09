"""Sim-to-field domain adaptation DGP for Module 5 (field validation)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from field_compounding.data.base import BaseFieldDGP, BenchmarkData

LOOP_NODE = "sim_to_real"
_GNSS_DRIFT_SCALE_M = 10.0

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
    loop_node: str
    gnss_drift: float
    false_positive_rate: float
    cmd_latency_ms: float

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "_TraceRow":
        loop_node = str(row["loop_node"])
        if loop_node not in VALID_LOOP_NODES:
            raise ValueError(f"unknown loop_node: {loop_node!r}")
        latency = row.get("cmd_latency_ms")
        return cls(
            loop_node=loop_node,
            gnss_drift=max(0.0, float(row["gnss_drift"])),
            false_positive_rate=float(np.clip(float(row["false_positive_rate"]), 0.0, 1.0)),
            cmd_latency_ms=max(0.0, float(latency if latency is not None else 120.0)),
        )


def _load_traces(path: str | Path) -> list[_TraceRow]:
    entries: list[_TraceRow] = []
    trace_path = Path(path)
    with trace_path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{trace_path}:{line_no}: invalid JSON") from exc
            entries.append(_TraceRow.from_dict(row))
    return entries


def _save_traces(entries: list[_TraceRow], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for entry in entries:
            payload = {
                "timestamp": "2025-05-18T12:00:00Z",
                "loop_node": entry.loop_node,
                "violation_severity": 0.0,
                "recovery": True,
                "gnss_drift": entry.gnss_drift,
                "false_positive_rate": entry.false_positive_rate,
                "cmd_latency_ms": entry.cmd_latency_ms,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return out


def compute_trace_domain_shift(entries: list[_TraceRow]) -> float:
    node_rows = [e for e in entries if e.loop_node == LOOP_NODE]
    rows = node_rows if node_rows else entries
    if not rows:
        return 0.0
    gnss = np.array([e.gnss_drift for e in rows], dtype=np.float64)
    fpr = np.array([e.false_positive_rate for e in rows], dtype=np.float64)
    latencies = np.array([e.cmd_latency_ms for e in rows], dtype=np.float64)
    gnss_norm = float(np.clip(np.mean(gnss) / _GNSS_DRIFT_SCALE_M, 0.0, 1.0))
    fpr_mean = float(np.mean(fpr))
    latency_norm = float(np.clip(np.mean(latencies) / 500.0, 0.0, 1.0))
    return float(np.clip(0.45 * gnss_norm + 0.35 * fpr_mean + 0.20 * latency_norm, 0.0, 1.0))


def resolve_sim_field_gap(
    violation_severity: float,
    sim_field_gap: float | None,
    trace_domain_shift: float,
    *,
    trace_coupling: float = 1.0,
) -> tuple[float, float]:
    explicit = sim_field_gap if sim_field_gap is not None else violation_severity * 2.0
    coupling = float(np.clip(trace_coupling, 0.0, 2.0))
    effective = explicit + coupling * trace_domain_shift * 2.0
    return explicit, float(effective)


def generate_synthetic_traces(n_rows: int = 240, *, seed: int = 42) -> list[_TraceRow]:
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    rng = np.random.default_rng(seed)
    nodes = sorted(VALID_LOOP_NODES)
    entries: list[_TraceRow] = []
    for idx in range(n_rows):
        node = nodes[idx % len(nodes)]
        entries.append(
            _TraceRow(
                loop_node=node,
                gnss_drift=float(rng.exponential(2.5)),
                false_positive_rate=float(rng.beta(1.5, 8.0)),
                cmd_latency_ms=float(rng.lognormal(mean=3.2, sigma=0.45)),
            )
        )
    return entries


class SimToFieldDGP(BaseFieldDGP):
    def __init__(
        self,
        seed: int = 42,
        violation_severity: float = 0.0,
        trace_path: str | None = None,
        n_source: int = 1000,
        n_target: int = 500,
        feature_dim: int = 10,
        n_classes: int = 4,
        sim_field_gap: float | None = None,
        trace_coupling: float = 1.0,
        source_sigma: float = 1.0,
        target_sigma_scale: float = 1.2,
    ):
        super().__init__(seed=seed, violation_severity=violation_severity, trace_path=trace_path)
        self.n_source = n_source
        self.n_target = n_target
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.sim_field_gap = sim_field_gap
        self.trace_coupling = trace_coupling
        self.source_sigma = source_sigma
        self.target_sigma_scale = target_sigma_scale + 0.8 * violation_severity

    @property
    def name(self) -> str:
        return "sim_to_field_adaptation"

    @property
    def loop_node(self) -> str:
        return LOOP_NODE

    def _load_trace_entries(self) -> list[_TraceRow]:
        if self.trace_path is not None:
            path = Path(self.trace_path)
            if path.is_file():
                return _load_traces(path)
        return generate_synthetic_traces(n_rows=240, seed=self.seed)

    def _generate_class_centers(self, rng: np.random.Generator) -> np.ndarray:
        centers = np.zeros((self.n_classes, self.feature_dim), dtype=np.float32)
        for i in range(self.n_classes):
            direction = rng.standard_normal(self.feature_dim)
            direction /= np.linalg.norm(direction) + 1e-8
            centers[i] = direction * 3.0 * (i + 1) / self.n_classes
        return centers

    def _generate_domain_data(
        self,
        rng: np.random.Generator,
        n: int,
        class_centers: np.ndarray,
        mu_shift: np.ndarray,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, self.n_classes, size=n)
        features = np.zeros((n, self.feature_dim), dtype=np.float32)
        for i in range(n):
            features[i] = (
                class_centers[labels[i]]
                + mu_shift
                + rng.normal(0.0, sigma, self.feature_dim).astype(np.float32)
            )
        return features, labels.astype(np.int64)

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        trace_entries = self._load_trace_entries()
        trace_domain_shift = compute_trace_domain_shift(trace_entries)
        explicit_gap, effective_shift = resolve_sim_field_gap(
            self.violation_severity,
            self.sim_field_gap,
            trace_domain_shift,
            trace_coupling=self.trace_coupling,
        )
        class_centers = self._generate_class_centers(rng)
        mu_source = np.zeros(self.feature_dim, dtype=np.float32)
        shift_direction = rng.standard_normal(self.feature_dim).astype(np.float32)
        shift_direction /= np.linalg.norm(shift_direction) + 1e-8
        mu_target = shift_direction * effective_shift
        target_sigma = self.source_sigma * self.target_sigma_scale
        X_source, y_source = self._generate_domain_data(
            rng, self.n_source, class_centers, mu_source, self.source_sigma
        )
        X_target, y_target = self._generate_domain_data(
            rng, self.n_target, class_centers, mu_target, target_sigma
        )
        n_source_train = int(0.8 * self.n_source)
        n_target_train = int(0.8 * self.n_target)
        node_rows = [e for e in trace_entries if e.loop_node == LOOP_NODE]
        telemetry_rows = node_rows if node_rows else trace_entries
        metadata: dict[str, Any] = {
            "loop_node": self.loop_node,
            "n_source": self.n_source,
            "n_target": self.n_target,
            "feature_dim": self.feature_dim,
            "n_classes": self.n_classes,
            "sim_field_gap": explicit_gap,
            "trace_domain_shift": trace_domain_shift,
            "trace_coupling": self.trace_coupling,
            "effective_domain_shift": effective_shift,
            "domain_shift_delta": float(np.linalg.norm(mu_target - mu_source)),
            "shift_direction": shift_direction,
            "mu_source": mu_source,
            "mu_target": mu_target,
            "source_sigma": self.source_sigma,
            "target_sigma": target_sigma,
            "violation_severity": self.violation_severity,
            "inferred_violation_severity": float(np.clip(effective_shift / 2.0, 0.0, 1.0)),
            "trace_row_count": len(trace_entries),
            "sim_to_real_trace_rows": len(node_rows),
            "gnss_drift": np.array([e.gnss_drift for e in telemetry_rows], dtype=np.float32),
            "false_positive_rate": np.array(
                [e.false_positive_rate for e in telemetry_rows], dtype=np.float32
            ),
            "cmd_latency_ms": np.array([e.cmd_latency_ms for e in telemetry_rows], dtype=np.float32),
            "trace_path": self.trace_path,
        }
        return BenchmarkData(
            train={
                "X_source": X_source[:n_source_train],
                "y_source": y_source[:n_source_train],
                "X_target": X_target[:n_target_train],
                "y_target": y_target[:n_target_train],
            },
            test={
                "X_source": X_source[n_source_train:],
                "y_source": y_source[n_source_train:],
                "X_target": X_target[n_target_train:],
                "y_target": y_target[n_target_train:],
            },
            metadata=metadata,
        )
