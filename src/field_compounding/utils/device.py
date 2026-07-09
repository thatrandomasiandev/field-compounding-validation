"""Device selection for training and evaluation."""

from __future__ import annotations

import os

import torch

_VALID_DEVICES = frozenset({"cpu", "cuda", "mps"})


def get_device() -> torch.device:
    """Return compute device, respecting ``FIELD_COMPOUNDING_DEVICE`` override."""
    override = os.environ.get("FIELD_COMPOUNDING_DEVICE", "").lower().strip()
    if override:
        if override not in _VALID_DEVICES:
            raise ValueError(f"unsupported FIELD_COMPOUNDING_DEVICE: {override!r}")
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
