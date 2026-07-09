"""Tests for cross-cutting utils: seed, device, crossfit, trace_index."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from field_compounding.utils.crossfit import KFoldSplitter, split_trace_index
from field_compounding.utils.device import get_device
from field_compounding.utils.seed import set_seed, trial_seed
from field_compounding.utils.trace_index import TraceIndex, TraceRecord, parse_timestamp


def _sample_rows() -> list[dict]:
    return [
        {
            "timestamp": "2025-05-18T14:02:11Z",
            "loop_node": "scene_repr",
            "violation_severity": 0.31,
            "recovery": True,
            "gnss_drift": 1.2,
            "false_positive_rate": 0.04,
        },
        {
            "timestamp": "2025-05-18T14:18:44Z",
            "loop_node": "visual_ssl",
            "violation_severity": 0.42,
            "recovery": True,
            "gnss_drift": 2.8,
            "false_positive_rate": 0.07,
        },
        {
            "timestamp": "2025-05-18T14:35:02Z",
            "loop_node": "sim_to_real",
            "violation_severity": 0.58,
            "recovery": False,
            "gnss_drift": 4.6,
            "false_positive_rate": 0.11,
        },
        {
            "timestamp": "2025-05-18T15:01:27Z",
            "loop_node": "visuomotor",
            "violation_severity": 0.47,
            "recovery": True,
            "gnss_drift": 3.1,
            "false_positive_rate": 0.09,
        },
        {
            "timestamp": "2025-05-18T15:22:55Z",
            "loop_node": "causal",
            "violation_severity": 0.28,
            "recovery": True,
            "gnss_drift": 1.9,
            "false_positive_rate": 0.06,
        },
        {
            "timestamp": "2025-05-18T15:48:10Z",
            "loop_node": "world_model",
            "violation_severity": 0.63,
            "recovery": False,
            "gnss_drift": 5.4,
            "false_positive_rate": 0.13,
        },
    ]


def test_kfold_splitter_covers_all_indices() -> None:
    splitter = KFoldSplitter(n_splits=3, seed=7)
    folds = list(splitter.split(9))
    assert len(folds) == 3
    covered = np.sort(np.concatenate([test for _, test in folds]))
    assert np.array_equal(covered, np.arange(9))


def test_kfold_splitter_no_train_test_overlap() -> None:
    splitter = KFoldSplitter(n_splits=3, seed=7)
    for train_idx, test_idx in splitter.split(12):
        assert len(set(train_idx) & set(test_idx)) == 0


def test_kfold_splitter_deterministic() -> None:
    a = list(KFoldSplitter(n_splits=3, seed=99).split(12))
    b = list(KFoldSplitter(n_splits=3, seed=99).split(12))
    for (train_a, test_a), (train_b, test_b) in zip(a, b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)


def test_trace_index_from_rows_sorted() -> None:
    rows = _sample_rows()
    shuffled = [rows[2], rows[0], rows[4], rows[1], rows[3], rows[5]]
    index = TraceIndex.from_rows(shuffled)
    timestamps = [record.timestamp for record in index]
    assert timestamps == sorted(timestamps, key=parse_timestamp)


def test_trace_index_records_for_loop_node() -> None:
    index = TraceIndex.from_rows(_sample_rows())
    scene = index.records_for("scene_repr")
    assert len(scene) == 1
    assert scene[0].loop_node == "scene_repr"
    assert index.indices_for("scene_repr") == [0]


def test_trace_index_slice_time_half_open() -> None:
    index = TraceIndex.from_rows(_sample_rows())
    sliced = index.slice_time("2025-05-18T14:18:44Z", "2025-05-18T15:22:55Z")
    nodes = {record.loop_node for record in sliced}
    assert nodes == {"visual_ssl", "sim_to_real", "visuomotor"}


def test_trace_index_invalid_loop_node_raises() -> None:
    with pytest.raises(ValueError, match="unknown loop_node"):
        TraceRecord.from_dict(
            {
                "timestamp": "2025-05-18T14:02:11Z",
                "loop_node": "invalid_node",
                "violation_severity": 0.1,
            }
        )


def test_set_seed_reproducible_numpy() -> None:
    set_seed(123)
    first = np.random.rand(4)
    set_seed(123)
    second = np.random.rand(4)
    assert np.array_equal(first, second)


def test_trial_seed_unique_per_trial() -> None:
    base = 42
    seeds = {trial_seed(base, trial) for trial in range(20)}
    assert len(seeds) == 20


def test_get_device_respects_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIELD_COMPOUNDING_DEVICE", "cpu")
    assert get_device() == torch.device("cpu")


def test_get_device_invalid_override_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIELD_COMPOUNDING_DEVICE", "tpu")
    with pytest.raises(ValueError, match="unsupported FIELD_COMPOUNDING_DEVICE"):
        get_device()


def test_split_trace_index_returns_folds() -> None:
    index = TraceIndex.from_rows(_sample_rows())
    folds = split_trace_index(index, n_splits=3, seed=11)
    assert len(folds) == 3
    covered = np.sort(np.concatenate([test for _, test in folds]))
    assert np.array_equal(covered, np.arange(len(index)))
