"""Deterministic seeding for benchmarks and tests."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible benchmarks."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def trial_seed(base_seed: int, trial: int) -> int:
    """Derive a deterministic per-trial seed from a base benchmark seed."""
    if trial < 0:
        raise ValueError("trial must be non-negative")
    return int(base_seed) * 1_000_003 + int(trial)
