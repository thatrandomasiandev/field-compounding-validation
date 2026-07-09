"""Shared pytest fixtures."""

import os

import numpy as np
import pytest

from field_compounding.utils.seed import set_seed

os.environ.setdefault("FIELD_COMPOUNDING_DEVICE", "cpu")


@pytest.fixture(autouse=True)
def deterministic():
    set_seed(42)
    yield


@pytest.fixture
def rng():
    return np.random.default_rng(42)
