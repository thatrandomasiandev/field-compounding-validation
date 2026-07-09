"""Tests for evaluation adapters modules 3–8 (Agent 22)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from field_compounding.data.base import BenchmarkData
from field_compounding.evaluation.metrics import normalized_score
from field_compounding.evaluation.module_adapters import (
    TrialContext,
    _add_normalized,
    _eval_module_03,
    _eval_module_04,
    _eval_module_05,
    _eval_module_06,
    _eval_module_07,
    _eval_module_08,
    _feature_field_query_labels,
    _flatten_wm,
    _scene_teacher_feature_maps,
    enrich_field_metrics,
    field_adjusted_baseline,
    resolve_violation_severity,
    run_trial,
)


class TestNormalizedScore:
    def test_higher_is_better(self) -> None:
        assert normalized_score(0.8, optimal=1.0, baseline=0.0) == pytest.approx(0.8)

    def test_add_normalized_pehe_lower_is_better(self) -> None:
        metrics = _add_normalized({"pehe": 0.1}, "pehe", optimal=0.05, baseline=1.0)
        assert "normalized_score" in metrics
        assert metrics["normalized_score"] > 0.5


class TestFieldMetadata:
    def test_resolve_violation_severity_ground_truth(self) -> None:
        assert resolve_violation_severity({"violation_severity": 0.4}) == pytest.approx(0.4)

    def test_resolve_violation_severity_inferred_fallback(self) -> None:
        assert resolve_violation_severity({"inferred_violation_severity": 0.7}) == pytest.approx(0.7)

    def test_resolve_violation_severity_defaults_to_zero(self) -> None:
        assert resolve_violation_severity({}) == 0.0

    def test_field_adjusted_baseline_shifts_with_violation(self) -> None:
        low = field_adjusted_baseline(0.25, 0.95, {"violation_severity": 0.0})
        high = field_adjusted_baseline(0.25, 0.95, {"violation_severity": 1.0})
        assert high > low

    def test_enrich_field_metrics_adds_provenance(self) -> None:
        meta = {
            "loop_node": "scene_repr",
            "trace_density": 0.8,
            "mean_gnss_drift_m": 1.2,
            "violation_severity": 0.3,
            "trace_source": "urc",
        }
        out = enrich_field_metrics({"psnr": 20.0}, meta)
        assert out["field_violation_severity"] == pytest.approx(0.3)
        assert out["field_trace_density"] == pytest.approx(0.8)
        assert out["field_mean_gnss_drift_m"] == pytest.approx(1.2)
        assert out["field_loop_node"] == "scene_repr"
        assert out["field_trace_source"] == "urc"


class TestHelpers:
    def test_flatten_wm_shapes(self) -> None:
        states = np.zeros((2, 3, 4))
        actions = np.zeros((2, 3, 2))
        next_states = np.zeros((2, 3, 4))
        s, a, ns = _flatten_wm(states, actions, next_states)
        assert s.shape == (6, 4)
        assert a.shape == (6, 2)
        assert ns.shape == (6, 4)

    def test_scene_teacher_feature_maps_1d(self) -> None:
        feats = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        maps = _scene_teacher_feature_maps(feats, n_views=2, height=4, width=4)
        assert maps.shape == (2, 4, 4, 3)
        assert np.allclose(maps[0, 0, 0], feats)

    def test_scene_teacher_feature_maps_2d_mean(self) -> None:
        feats = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
        maps = _scene_teacher_feature_maps(feats, n_views=1, height=2, width=2)
        assert np.allclose(maps[0, 0, 0], [2.0, 0.0])

    def test_feature_field_query_labels(self) -> None:
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[0, :3, 3] = [0.0, 0.0, 0.0]
        poses[1, :3, 3] = [10.0, 0.0, 0.0]
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        assert _feature_field_query_labels(poses, positions).tolist() == [0, 1]


def _ctx(module_id: int, model_name: str = "mock", class_path: str = "mock.Model") -> TrialContext:
    return TrialContext(module_id=module_id, model_name=model_name, class_path=class_path, params={}, fast=True)


class TestEvalModule03:
    def test_nerf_returns_psnr_and_normalized_score(self) -> None:
        model = MagicMock()
        model.evaluate_psnr.return_value = 25.0
        images = np.random.rand(4, 8, 8, 3).astype(np.float32)
        poses = np.tile(np.eye(4), (4, 1, 1)).astype(np.float32)
        data = BenchmarkData(
            train={"images": [images], "camera_poses": [poses]},
            test={"images": [images], "camera_poses": [poses]},
            metadata={"mean_gnss_drift_m": 0.5, "violation_severity": 0.1},
        )
        metrics = _eval_module_03(model, _ctx(3, class_path="pkg.NeRF"), data)
        model.fit.assert_called_once()
        assert metrics["psnr"] == pytest.approx(25.0)
        assert "normalized_score" in metrics
        assert metrics["gnss_drift_m"] == pytest.approx(0.5)


class TestEvalModule04:
    def test_ssl_linear_probe_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "field_compounding.evaluation.module_adapters.evaluate_ssl",
            lambda *a, **k: {"linear_probe_accuracy": 0.7, "silhouette_score": 0.4},
        )
        n, d = 16, 8
        feats = np.random.randn(n, d).astype(np.float32)
        labels = np.random.randint(0, 4, size=n)
        data = BenchmarkData(
            train={"augmented_view1": feats, "augmented_view2": feats, "features": feats, "labels": labels},
            test={"features": feats, "labels": labels},
            metadata={"trace_density": 0.6, "n_clusters": 4},
        )
        metrics = _eval_module_04(MagicMock(), _ctx(4, model_name="SimCLR"), data)
        assert metrics["linear_probe_accuracy"] == pytest.approx(0.7)
        assert "normalized_score" in metrics


class TestEvalModule05:
    def test_sim_to_field_target_accuracy(self) -> None:
        model = MagicMock()
        model.evaluate.return_value = {"source_acc": 0.9, "target_acc": 0.6, "domain_gap": 0.3, "n_trainable_params": 1000}
        n = 20
        x = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=n)
        data = BenchmarkData(
            train={"X_source": x, "y_source": y, "X_target": x, "y_target": y},
            test={"X_source": x, "y_source": y, "X_target": x, "y_target": y},
            metadata={"sim_field_gap": 0.4, "violation_severity": 0.2},
        )
        metrics = _eval_module_05(model, _ctx(5, model_name="LoRA"), data)
        assert metrics["target_accuracy"] == pytest.approx(0.6)
        assert "normalized_score" in metrics


class TestEvalModule06:
    def test_visuomotor_action_mse_primary(self) -> None:
        @dataclass
        class EvalOut:
            action_mse: float
            train_loss: float

        model = MagicMock()
        model.evaluate.return_value = EvalOut(action_mse=0.12, train_loss=0.0)
        obs = np.random.randn(10, 4).astype(np.float32)
        act = np.random.randn(10, 2).astype(np.float32)
        data = BenchmarkData(
            train={"observations": obs, "actions": act},
            test={"observations": obs, "actions": act},
            metadata={"cmd_latency_ms": 42.0},
        )
        metrics = _eval_module_06(model, _ctx(6, model_name="BC"), data)
        assert metrics["mse"] == pytest.approx(0.12)
        assert "normalized_score" in metrics


class TestEvalModule07:
    def test_cate_pehe_metrics(self) -> None:
        model = MagicMock()
        model.evaluate.return_value = SimpleNamespace(pehe=0.08, ate_error=0.02)
        n = 30
        x = np.random.randn(n, 3).astype(np.float32)
        t = np.random.randint(0, 2, size=n)
        y = np.random.randn(n).astype(np.float32)
        cate = np.random.randn(n).astype(np.float32)
        data = BenchmarkData(
            train={"X": x, "T": t, "Y": y},
            test={"X": x, "T": t, "Y": y, "cate_true": cate},
            metadata={"violation_severity": 0.15},
        )
        metrics = _eval_module_07(model, _ctx(7, model_name="DR-Learner"), data)
        assert metrics["pehe"] == pytest.approx(0.08)
        assert "normalized_score" in metrics


class TestEvalModule08:
    def test_world_model_transition_mse(self) -> None:
        model = MagicMock()
        model.evaluate.return_value = SimpleNamespace(transition_mse=0.03, nll=1.2)
        n, t, sd, ad = 4, 5, 6, 3
        states = np.random.randn(n, t, sd).astype(np.float32)
        actions = np.random.randn(n, t, ad).astype(np.float32)
        next_states = np.random.randn(n, t, sd).astype(np.float32)
        data = BenchmarkData(
            train={"states": states, "actions": actions, "next_states": next_states},
            test={"states": states, "actions": actions, "next_states": next_states},
            metadata={"model_error": 0.05, "violation_severity": 0.5},
        )
        metrics = _eval_module_08(model, _ctx(8, model_name="EnsembleWM", class_path="pkg.EnsembleWorldModel"), data)
        assert metrics["transition_mse"] == pytest.approx(0.03)
        assert "normalized_score" in metrics


class TestRunTrial:
    def test_run_trial_enriches_field_metadata(self) -> None:
        model = MagicMock()
        model.evaluate.return_value = SimpleNamespace(transition_mse=0.02)
        n, t, sd, ad = 2, 3, 4, 2
        states = np.zeros((n, t, sd), dtype=np.float32)
        actions = np.zeros((n, t, ad), dtype=np.float32)
        next_states = np.zeros((n, t, sd), dtype=np.float32)
        data = BenchmarkData(
            train={"states": states, "actions": actions, "next_states": next_states},
            test={"states": states, "actions": actions, "next_states": next_states},
            metadata={"violation_severity": 0.25, "loop_node": "world_model"},
        )
        out = run_trial(model, _ctx(8), data)
        assert out["field_violation_severity"] == pytest.approx(0.25)
        assert out["field_loop_node"] == "world_model"

    def test_run_trial_unsupported_module_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported module_id=99"):
            run_trial(MagicMock(), _ctx(99), BenchmarkData({}, {}, {}))
