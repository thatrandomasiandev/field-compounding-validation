"""Field-backed causal scene understanding DGP (Module 7 / loop node causal)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from field_compounding.data.base import BaseFieldDGP, BenchmarkData

LOOP_NODE = "causal"


@dataclass(frozen=True)
class CausalTraceRow:
    gnss_drift: float
    false_positive_rate: float
    violation_severity: float


class _FixedNonlinearMap(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, seed: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        gen = torch.Generator().manual_seed(seed)
        for param in self.net.parameters():
            nn.init.normal_(param, 0.0, 0.5, generator=gen)
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_causal_trace_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            rows.append(row)
    return rows


def _rows_to_causal_telemetry(rows: list[dict[str, Any]]) -> list[CausalTraceRow]:
    causal_rows = [row for row in rows if row.get("loop_node") == LOOP_NODE]
    source = causal_rows if causal_rows else rows
    return [
        CausalTraceRow(
            gnss_drift=max(0.0, float(row["gnss_drift"])),
            false_positive_rate=float(np.clip(float(row["false_positive_rate"]), 0.0, 1.0)),
            violation_severity=float(np.clip(float(row["violation_severity"]), 0.0, 1.0)),
        )
        for row in source
    ]


def generate_synthetic_causal_traces(n_rows: int = 120, *, seed: int = 42) -> list[CausalTraceRow]:
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    rng = np.random.default_rng(seed)
    return [
        CausalTraceRow(
            gnss_drift=float(rng.exponential(2.5)),
            false_positive_rate=float(rng.beta(1.5, 8.0)),
            violation_severity=float(rng.beta(2.0, 4.0)),
        )
        for _ in range(n_rows)
    ]


def _inferred_violation_proxy(gnss_drift: np.ndarray, false_positive_rate: np.ndarray) -> np.ndarray:
    gnss_norm = np.clip(gnss_drift / 10.0, 0.0, 1.0)
    return np.clip(0.55 * gnss_norm + 0.45 * false_positive_rate, 0.0, 1.0)


def _resolve_trace_rows(trace_path: str | None, seed: int, min_rows: int) -> list[CausalTraceRow]:
    if trace_path is not None:
        path = Path(trace_path)
        if not path.is_file():
            raise FileNotFoundError(f"trace file not found: {path}")
        return _rows_to_causal_telemetry(load_causal_trace_rows(path))
    return generate_synthetic_causal_traces(n_rows=max(min_rows, 120), seed=seed)


class CausalFieldDGP(BaseFieldDGP):
    LATENT_DIM: int = 5
    OBS_DIM: int = 12

    def __init__(
        self,
        seed: int = 42,
        violation_severity: float = 0.0,
        trace_path: str | None = None,
        gamma: float = 0.0,
        n_train: int = 2000,
        n_test: int = 500,
    ):
        effective_gamma = violation_severity if gamma == 0.0 else gamma
        super().__init__(seed=seed, violation_severity=effective_gamma, trace_path=trace_path)
        self.gamma = effective_gamma
        self.n_train = n_train
        self.n_test = n_test

    @property
    def name(self) -> str:
        return "field_causal_scene"

    @property
    def loop_node(self) -> str:
        return LOOP_NODE

    def _generate_split(
        self,
        rng: np.random.Generator,
        n: int,
        g_map: _FixedNonlinearMap,
        trace_rows: list[CausalTraceRow],
    ) -> dict[str, np.ndarray]:
        if not trace_rows:
            raise ValueError("causal trace replay requires at least one trace row")

        gnss = np.empty(n, dtype=np.float32)
        fpr = np.empty(n, dtype=np.float32)
        for idx in range(n):
            row = trace_rows[idx % len(trace_rows)]
            gnss[idx] = row.gnss_drift
            fpr[idx] = row.false_positive_rate

        field_proxy = _inferred_violation_proxy(gnss, fpr).astype(np.float32)
        z_bg = rng.standard_normal(n).astype(np.float32) * 0.25 + 0.9 * (field_proxy - 0.5)
        z_shape = rng.standard_normal(n).astype(np.float32)
        z_color = rng.standard_normal(n).astype(np.float32)
        z_light = rng.standard_normal(n).astype(np.float32)
        z_pose = rng.standard_normal(n).astype(np.float32)
        Z = np.stack([z_shape, z_color, z_light, z_pose, z_bg], axis=1)

        logit = self.gamma * (0.65 * z_bg + 0.35 * field_proxy)
        logit += rng.standard_normal(n).astype(np.float32) * 0.3
        propensity = 1.0 / (1.0 + np.exp(-logit))
        treatment = (rng.uniform(size=n) < propensity).astype(np.float32)

        X = g_map(torch.from_numpy(Z)).numpy()
        field_noise = (gnss / 10.0)[:, None] * rng.standard_normal((n, self.OBS_DIM)).astype(np.float32)
        X = X + field_noise * 0.15

        treatment_effect = 0.5 * z_shape + 0.3 * z_color
        cate_true = treatment_effect.copy()
        outcome_base = 1.0 * z_shape**2 + 0.5 * z_color
        outcome = outcome_base + treatment * treatment_effect + rng.standard_normal(n).astype(np.float32) * 0.2

        return {
            "X": X.astype(np.float32),
            "T": treatment,
            "Y": outcome.astype(np.float32),
            "Z": Z.astype(np.float32),
            "propensity": propensity.astype(np.float32),
            "cate_true": cate_true.astype(np.float32),
            "gnss_drift": gnss,
            "false_positive_rate": fpr,
            "field_violation_proxy": field_proxy,
        }

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        trace_rows = _resolve_trace_rows(self.trace_path, seed=self.seed, min_rows=self.n_train + self.n_test)
        g_map = _FixedNonlinearMap(self.LATENT_DIM, self.OBS_DIM, seed=self.seed)
        train = self._generate_split(rng, self.n_train, g_map, trace_rows)
        test = self._generate_split(rng, self.n_test, g_map, trace_rows)
        inferred_violation = float(
            np.mean(np.concatenate([train["field_violation_proxy"], test["field_violation_proxy"]]))
        )
        trace_source = str(Path(self.trace_path)) if self.trace_path else "synthetic"
        return BenchmarkData(
            train=train,
            test=test,
            metadata={
                "loop_node": self.loop_node,
                "latent_dim": self.LATENT_DIM,
                "obs_dim": self.OBS_DIM,
                "gamma": self.gamma,
                "n_train": self.n_train,
                "n_test": self.n_test,
                "violation_severity": self.violation_severity,
                "inferred_violation_severity": inferred_violation,
                "trace_path": self.trace_path,
                "trace_source": trace_source,
                "trace_rows_used": len(trace_rows),
                "causal_parents_Y": ["Z_shape", "Z_color", "T"],
                "confounder": "Z_bg",
                "field_confounders": ["gnss_drift", "false_positive_rate"],
            },
        )
