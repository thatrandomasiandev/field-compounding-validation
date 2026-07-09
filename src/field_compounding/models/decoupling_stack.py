"""Architectural decoupling stack for Condition D (Module 12)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from field_compounding.utils.device import get_device

LOOP_NODE_ORDER: tuple[str, ...] = (
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
)

MODULE_ID_TO_LOOP_NODE: dict[int, str] = {
    module_id: loop_node for module_id, loop_node in zip(range(3, 15), LOOP_NODE_ORDER, strict=True)
}


@dataclass(frozen=True)
class DecouplingConfig:
    feature_dim: int = 32
    latent_dim: int = 8
    equivariant_groups: int = 4
    conformal_alpha: float = 0.1
    cmd_latency_ms: float = 50.0
    latency_tau_ms: float = 120.0
    coupling_shrink_floor: float = 0.35


@dataclass
class StackOutput:
    features: np.ndarray
    gamma_scale: float
    latency_penalty: float
    layer_scales: dict[str, float] = field(default_factory=dict)


class _Bottleneck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, in_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)


class DecouplingStack:
    """Three-layer mitigation stack with field-latency attenuation."""

    def __init__(self, config: DecouplingConfig | None = None) -> None:
        self.config = config or DecouplingConfig()
        if self.config.latent_dim >= self.config.feature_dim:
            raise ValueError("latent_dim must be smaller than feature_dim")
        if self.config.equivariant_groups < 1:
            raise ValueError("equivariant_groups must be >= 1")

        self.device = get_device()
        self._bottleneck = _Bottleneck(self.config.feature_dim, self.config.latent_dim).to(self.device)
        self._residual_scale = torch.ones(self.config.latent_dim, device=self.device)
        self._latency_buffer: np.ndarray | None = None

    def equivariant_project(self, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        batch, dim = x.shape
        groups = self.config.equivariant_groups
        target = self.config.feature_dim
        remainder = dim % groups
        if remainder != 0:
            x = F.pad(x, (0, groups - remainder))
            dim = x.shape[1]
        grouped = x.view(batch, groups, -1)
        chunk = dim // groups
        means = grouped.mean(dim=2, keepdim=True)
        tiled = means.expand(-1, -1, chunk).reshape(batch, dim)
        out = tiled[:, :target]
        if out.shape[1] < target:
            out = F.pad(out, (0, target - out.shape[1]))
        return out.detach().cpu().numpy()

    def causal_bottleneck(self, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        self._bottleneck.eval()
        with torch.no_grad():
            z, recon = self._bottleneck(x)
            per_dim = (x - recon).pow(2).mean(dim=0)
            self._residual_scale = 1.0 / (1.0 + per_dim[: self.config.latent_dim])
            z_np = z.detach().cpu().numpy()
        return z_np

    def conformal_gate(self, latent: np.ndarray) -> np.ndarray:
        z = torch.from_numpy(np.asarray(latent, dtype=np.float32)).to(self.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        scale = self._residual_scale[: z.shape[1]]
        gated = z * scale.unsqueeze(0)
        return gated.detach().cpu().numpy()

    def latency_penalty(self, cmd_latency_ms: float | None = None) -> float:
        latency = self.config.cmd_latency_ms if cmd_latency_ms is None else float(cmd_latency_ms)
        latency = max(0.0, latency)
        tau = max(self.config.latency_tau_ms, 1e-6)
        raw = float(np.exp(-latency / tau))
        floor = self.config.coupling_shrink_floor
        return floor + (1.0 - floor) * raw

    def layer_gamma_scales(self, cmd_latency_ms: float | None = None) -> dict[str, float]:
        latency = self.latency_penalty(cmd_latency_ms)
        return {
            "equivariant": 0.82,
            "causal_bottleneck": 0.70,
            "conformal": 0.88,
            "field_latency": latency,
        }

    def effective_gamma_scale(self, cmd_latency_ms: float | None = None) -> float:
        scales = self.layer_gamma_scales(cmd_latency_ms)
        combined = float(np.prod(list(scales.values())))
        return float(np.clip(combined, self.config.coupling_shrink_floor, 1.0))

    def attenuate_coupling_matrix(
        self,
        gamma: np.ndarray,
        cmd_latency_ms: float | None = None,
    ) -> np.ndarray:
        gamma = np.asarray(gamma, dtype=np.float64)
        scale = self.effective_gamma_scale(cmd_latency_ms)
        out = gamma.copy()
        mask = ~np.eye(gamma.shape[0], dtype=bool)
        out[mask] *= scale
        return out

    def latency_smooth(self, features: np.ndarray, cmd_latency_ms: float | None = None) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        alpha = self.latency_penalty(cmd_latency_ms)
        if self._latency_buffer is None or self._latency_buffer.shape != x.shape:
            self._latency_buffer = x.copy()
        else:
            self._latency_buffer = alpha * x + (1.0 - alpha) * self._latency_buffer
        return self._latency_buffer.copy()

    def forward(
        self,
        features: np.ndarray,
        cmd_latency_ms: float | None = None,
    ) -> StackOutput:
        smoothed = self.latency_smooth(features, cmd_latency_ms)
        equi = self.equivariant_project(smoothed)
        latent = self.causal_bottleneck(equi)
        gated = self.conformal_gate(latent)
        scales = self.layer_gamma_scales(cmd_latency_ms)
        return StackOutput(
            features=gated,
            gamma_scale=self.effective_gamma_scale(cmd_latency_ms),
            latency_penalty=scales["field_latency"],
            layer_scales=scales,
        )

    def fit_bottleneck(self, features: np.ndarray, epochs: int = 30, lr: float = 1e-2) -> float:
        x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        opt = torch.optim.Adam(self._bottleneck.parameters(), lr=lr)
        self._bottleneck.train()
        last_loss = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            _, recon = self._bottleneck(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        return last_loss
