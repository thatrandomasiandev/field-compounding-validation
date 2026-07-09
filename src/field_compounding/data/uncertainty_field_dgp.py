"""Field-backed DGP for uncertainty quantification (Module 11 / loop node ``uncertainty``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from field_compounding.data.base import BaseFieldDGP, BenchmarkData

LOOP_NODE = "uncertainty"
GNSS_DRIFT_REF_M = 10.0


@dataclass(frozen=True)
class UncertaintyTraceStats:
    row_count: int
    mean_violation_severity: float
    mean_gnss_drift_m: float
    mean_false_positive_rate: float


def _load_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def _load_uncertainty_trace_stats(trace_path: str | Path) -> UncertaintyTraceStats:
    path = Path(trace_path)
    if not path.is_file():
        raise FileNotFoundError(f"trace file not found: {path}")
    uncertainty_rows = [row for row in _load_trace_rows(path) if row.get("loop_node") == LOOP_NODE]
    if not uncertainty_rows:
        raise ValueError(f"{path}: no rows with loop_node={LOOP_NODE!r}")
    gnss = np.array([float(row["gnss_drift"]) for row in uncertainty_rows], dtype=np.float64)
    fpr = np.array([float(np.clip(float(row["false_positive_rate"]), 0.0, 1.0)) for row in uncertainty_rows], dtype=np.float64)
    severity = np.array([float(np.clip(float(row["violation_severity"]), 0.0, 1.0)) for row in uncertainty_rows], dtype=np.float64)
    return UncertaintyTraceStats(
        row_count=len(uncertainty_rows),
        mean_violation_severity=float(np.mean(severity)),
        mean_gnss_drift_m=float(np.mean(np.maximum(gnss, 0.0))),
        mean_false_positive_rate=float(np.mean(fpr)),
    )


def _inferred_violation_proxy(stats: UncertaintyTraceStats) -> float:
    gnss_norm = float(np.clip(stats.mean_gnss_drift_m / GNSS_DRIFT_REF_M, 0.0, 1.0))
    return float(np.clip(0.45 * stats.mean_violation_severity + 0.35 * stats.mean_false_positive_rate + 0.20 * gnss_norm, 0.0, 1.0))


class UncertaintyFieldDGP(BaseFieldDGP):
    def __init__(
        self,
        seed: int = 42,
        violation_severity: float = 0.0,
        trace_path: str | None = None,
        n_samples: int = 3000,
        input_dim: int = 5,
        sigma_h: float = 1.0,
        n_classes: int = 10,
        n_ood_classes: int = 2,
        trace_coupling: float = 1.0,
        field_gap: float = 0.0,
    ) -> None:
        super().__init__(seed=seed, violation_severity=violation_severity, trace_path=trace_path)
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.sigma_h = sigma_h
        self.n_classes = n_classes
        self.n_ood_classes = n_ood_classes
        self.trace_coupling = float(np.clip(trace_coupling, 0.0, 1.0))
        self.field_gap = float(np.clip(field_gap, 0.0, 1.0))
        self._trace_stats = self._resolve_trace_stats()

    @property
    def name(self) -> str:
        return "uncertainty_field"

    @property
    def loop_node(self) -> str:
        return LOOP_NODE

    def _resolve_trace_stats(self) -> UncertaintyTraceStats | None:
        if self.trace_path is None:
            return None
        return _load_uncertainty_trace_stats(self.trace_path)

    def _noise_modifiers(self) -> tuple[float, float, float]:
        base_p_noise = self.violation_severity * 0.3
        hetero_mult = 1.0 + self.field_gap * 0.5
        extra_label_noise = self.field_gap * 0.1
        if self._trace_stats is None:
            return hetero_mult, min(1.0, base_p_noise + extra_label_noise), self.field_gap
        coupling = self.trace_coupling
        gnss_term = coupling * self._trace_stats.mean_gnss_drift_m / GNSS_DRIFT_REF_M
        fpr_term = coupling * self._trace_stats.mean_false_positive_rate
        hetero_mult = 1.0 + gnss_term + self.field_gap * 0.5
        p_noise = min(1.0, base_p_noise + fpr_term * 0.5 + extra_label_noise)
        effective_gap = float(np.clip(self.field_gap + coupling * gnss_term * 0.5, 0.0, 1.0))
        return hetero_mult, p_noise, effective_gap

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        n, d = self.n_samples, self.input_dim
        hetero_mult, p_noise, effective_gap = self._noise_modifiers()
        X_reg = rng.uniform(-3.0, 3.0, size=(n, d)).astype(np.float32)
        norms = np.linalg.norm(X_reg, axis=-1)
        noise_std = (self.sigma_h * hetero_mult * np.sqrt(1.0 + norms)).astype(np.float32)
        Y_reg = (np.sin(norms) + rng.standard_normal(n) * noise_std).astype(np.float32)
        n_id_classes = self.n_classes - self.n_ood_classes
        centers = rng.standard_normal((self.n_classes, d)).astype(np.float32) * 3.0
        samples_per_class = n // self.n_classes
        X_cls_list, labels_list = [], []
        for class_idx in range(self.n_classes):
            X_cls_list.append(centers[class_idx] + 0.5 * rng.standard_normal((samples_per_class, d)).astype(np.float32))
            labels_list.append(np.full(samples_per_class, class_idx, dtype=np.int64))
        X_cls = np.concatenate(X_cls_list, axis=0)
        labels_clean = np.concatenate(labels_list, axis=0)
        labels = labels_clean.copy()
        if p_noise > 0.0:
            flip_mask = rng.random(len(labels)) < p_noise
            labels[flip_mask] = rng.integers(0, self.n_classes, size=len(labels))[flip_mask]
        ood_mask = labels_clean >= n_id_classes
        perm_reg = rng.permutation(n)
        n_train, n_val, n_cal = int(0.70 * n), int(0.15 * n), int(0.10 * n)
        train_idx, val_idx = perm_reg[:n_train], perm_reg[n_train : n_train + n_val]
        test_idx, cal_idx = perm_reg[n_train + n_val :], perm_reg[:n_cal]
        n_cls = len(X_cls)
        perm_cls = rng.permutation(n_cls)
        n_cls_train, n_cls_val = int(0.70 * n_cls), int(0.15 * n_cls)
        cls_train_idx = perm_cls[:n_cls_train]
        cls_val_idx = perm_cls[n_cls_train : n_cls_train + n_cls_val]
        cls_test_idx = perm_cls[n_cls_train + n_cls_val :]
        cls_cal_idx = cls_train_idx[: int(0.10 * n_cls)]
        train_data = {
            "X_regression": X_reg[train_idx], "Y_regression": Y_reg[train_idx], "noise_levels": noise_std[train_idx],
            "X_classification": X_cls[cls_train_idx], "labels": labels[cls_train_idx], "labels_clean": labels_clean[cls_train_idx],
            "ood_mask": ood_mask[cls_train_idx], "X_calibration": X_reg[cal_idx], "Y_calibration": Y_reg[cal_idx],
            "X_cls_calibration": X_cls[cls_cal_idx], "labels_calibration": labels[cls_cal_idx],
        }
        test_data = {
            "X_regression": X_reg[test_idx], "Y_regression": Y_reg[test_idx], "noise_levels": noise_std[test_idx],
            "X_classification": X_cls[cls_test_idx], "labels": labels[cls_test_idx], "labels_clean": labels_clean[cls_test_idx],
            "ood_mask": ood_mask[cls_test_idx], "X_val_regression": X_reg[val_idx], "Y_val_regression": Y_reg[val_idx],
            "X_val_classification": X_cls[cls_val_idx], "labels_val": labels[cls_val_idx],
        }
        metadata: dict[str, Any] = {
            "loop_node": self.loop_node, "n_samples": n, "input_dim": d, "sigma_h": self.sigma_h,
            "heteroscedastic_multiplier": hetero_mult, "p_noise": p_noise, "violation_severity": self.violation_severity,
            "n_classes": self.n_classes, "n_ood_classes": self.n_ood_classes, "n_id_classes": n_id_classes,
            "calibration_split_size": n_cal, "trace_coupling": self.trace_coupling,
            "field_gap": {"configured": self.field_gap, "effective": effective_gap,
                          "false_positive_rate": self._trace_stats.mean_false_positive_rate if self._trace_stats else 0.0},
            "trace_path": self.trace_path, "source": "trace_calibrated" if self._trace_stats else "synthetic",
        }
        if self._trace_stats is not None:
            metadata["trace_stats"] = {
                "n_trace_rows": float(self._trace_stats.row_count),
                "mean_violation_severity": self._trace_stats.mean_violation_severity,
                "gnss_drift_m": self._trace_stats.mean_gnss_drift_m,
                "false_positive_rate": self._trace_stats.mean_false_positive_rate,
            }
            metadata["trace_rows_used"] = self._trace_stats.row_count
            metadata["mean_gnss_drift_m"] = self._trace_stats.mean_gnss_drift_m
            metadata["inferred_violation_severity"] = _inferred_violation_proxy(self._trace_stats)
        else:
            metadata["inferred_violation_severity"] = float(self.violation_severity)
        return BenchmarkData(train=train_data, test=test_data, metadata=metadata)
