"""Smoke test that core package imports work."""

import field_compounding
from field_compounding.data.base import BaseFieldDGP, BenchmarkData
from field_compounding.utils.seed import set_seed


def test_version():
    assert field_compounding.__version__ == "0.1.0"


def test_benchmark_data_container():
    data = BenchmarkData(train={"x": []}, test={"x": []}, metadata={})
    assert "x" in data.train


def test_set_seed_idempotent():
    set_seed(0)
    set_seed(0)


def test_base_field_dgp_is_abstract():
    assert hasattr(BaseFieldDGP, "generate")
