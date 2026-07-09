"""Smoke tests that core package and submodule imports work."""

import importlib

import field_compounding
from field_compounding import BaseFieldDGP, BenchmarkData, get_device, set_seed
from field_compounding.data import BaseFieldDGP as DataBaseFieldDGP
from field_compounding.data import BenchmarkData as DataBenchmarkData
from field_compounding.utils import get_device as utils_get_device
from field_compounding.utils import set_seed as utils_set_seed


def test_version():
    assert field_compounding.__version__ == "0.1.0"


def test_top_level_public_api():
    assert set(field_compounding.__all__) == {
        "BaseFieldDGP",
        "BenchmarkData",
        "__version__",
        "get_device",
        "set_seed",
    }


def test_benchmark_data_container():
    data = BenchmarkData(train={"x": []}, test={"x": []}, metadata={})
    assert "x" in data.train
    assert data.metadata == {}


def test_set_seed_idempotent():
    set_seed(0)
    set_seed(0)


def test_base_field_dgp_is_abstract():
    assert hasattr(BaseFieldDGP, "generate")
    assert hasattr(BaseFieldDGP, "_generate")
    assert hasattr(BaseFieldDGP, "loop_node")


def test_get_device_respects_cpu_override():
    device = get_device()
    assert device.type == "cpu"


def test_subpackage_reexports_match_top_level():
    assert DataBaseFieldDGP is BaseFieldDGP
    assert DataBenchmarkData is BenchmarkData
    assert utils_get_device is get_device
    assert utils_set_seed is set_seed


def test_all_subpackages_importable():
    subpackages = [
        "field_compounding.data",
        "field_compounding.evaluation",
        "field_compounding.ingest",
        "field_compounding.models",
        "field_compounding.observatory",
        "field_compounding.theory",
        "field_compounding.utils",
    ]
    for name in subpackages:
        mod = importlib.import_module(name)
        assert hasattr(mod, "__all__")
