"""Device selection for training and evaluation."""

from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    override = os.environ.get("FIELD_COMPOUNDING_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
