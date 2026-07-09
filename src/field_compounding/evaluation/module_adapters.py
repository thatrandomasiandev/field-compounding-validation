"""Module-specific training and evaluation adapters for field DGPs (modules 3–8).

Agent 22 owns ``_eval_module_03`` … ``_eval_module_08``. Agent 23 extends this file
with modules 9–14 and compound evaluation.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from field_compounding.data.base import BenchmarkData
from field_compounding.evaluation.metrics import normalized_score

try:
    from cv_robotics.models.ssl import evaluate_ssl
except ImportError:  # pragma: no cover
    evaluate_ssl = None  # type: ignore[misc, assignment]


@dataclass
class TrialContext:
    module_id: int
    model_name: str
    class_path: str
    params: dict[str, Any]
    fast: bool = False


FIELD_METADATA_KEYS: tuple[str, ...] = (
    "loop_node",
    "trace_density",
    "field_trace_density",
    "mean_gnss_drift_m",
    "inferred_violation_severity",
    "violation_severity",
    "trace_source",
)


def resolve_violation_severity(metadata: dict[str, Any]) -> float:
    """Return effective violation severity for field traces."""
    if "violation_severity" in metadata:
        return float(np.clip(metadata["violation_severity"], 0.0, 1.0))
    if "inferred_violation_severity" in metadata:
        return float(np.clip(metadata["inferred_violation_severity"], 0.0, 1.0))
    return 0.0


def enrich_field_metrics(metrics: dict[str, float], metadata: dict[str, Any]) -> dict[str, float]:
    """Attach trace provenance and effective violation severity to trial metrics."""
    out = dict(metrics)
    out["field_violation_severity"] = resolve_violation_severity(metadata)
    for key in FIELD_METADATA_KEYS:
        if key in metadata and isinstance(metadata[key], (int, float, str)):
            out[f"field_{key}"] = (
                float(metadata[key]) if isinstance(metadata[key], (int, float)) else metadata[key]
            )
    return out


def field_adjusted_baseline(baseline: float, optimal: float, metadata: dict[str, Any]) -> float:
    """Shift normalization baseline toward optimal under high inferred violation."""
    v = resolve_violation_severity(metadata)
    return baseline + v * (optimal - baseline) * 0.25


def build_model(ctx: TrialContext, metadata: dict[str, Any]) -> Any:
    """Instantiate a model from config, delegating to Module 11 when available."""
    try:
        from cv_robotics.evaluation import module_adapters as m11

        m11_ctx = m11.TrialContext(
            ctx.module_id, ctx.model_name, ctx.class_path, ctx.params, ctx.fast
        )
        return m11.build_model(m11_ctx, metadata)
    except ImportError:
        module_path, class_name = ctx.class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        params = dict(ctx.params)
        if ctx.fast:
            for key in ("epochs", "n_epochs", "n_rounds"):
                if key in params and isinstance(params[key], int):
                    params[key] = max(2, min(params[key], 5))
        allowed = {k for k in inspect.signature(cls.__init__).parameters if k != "self"}
        return cls(**{k: v for k, v in params.items() if k in allowed})


def run_trial(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    """Train and evaluate a model on field benchmark data (modules 3–8)."""
    handlers = {
        3: _eval_module_03,
        4: _eval_module_04,
        5: _eval_module_05,
        6: _eval_module_06,
        7: _eval_module_07,
        8: _eval_module_08,
    }
    if ctx.module_id not in handlers:
        raise NotImplementedError(
            f"module_id={ctx.module_id} adapters are owned by Agent 23 (modules 9–14)"
        )
    metrics = handlers[ctx.module_id](model, ctx, data)
    return enrich_field_metrics(metrics, data.metadata)


def _add_normalized(
    metrics: dict[str, float],
    primary: str,
    optimal: float,
    baseline: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, float]:
    if metadata is not None:
        baseline = field_adjusted_baseline(baseline, optimal, metadata)
    if primary in metrics:
        if primary in {"mse", "pehe", "ece", "energy_drift", "rmse"}:
            metrics["normalized_score"] = normalized_score(-metrics[primary], -optimal, -baseline)
        else:
            metrics["normalized_score"] = normalized_score(metrics[primary], optimal, baseline)
    return metrics


def _flatten_wm(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, t, sd = states.shape
    _, _, ad = actions.shape
    return states.reshape(n * t, sd), actions.reshape(n * t, ad), next_states.reshape(n * t, sd)


def _scene_teacher_feature_maps(
    object_features: np.ndarray,
    n_views: int,
    height: int,
    width: int,
) -> np.ndarray:
    teacher_vec = object_features if object_features.ndim == 1 else object_features.mean(axis=0)
    feature_dim = teacher_vec.shape[-1]
    return np.broadcast_to(
        teacher_vec.reshape(1, 1, 1, feature_dim),
        (n_views, height, width, feature_dim),
    ).copy().astype(np.float32)


def _feature_field_query_labels(
    camera_poses: np.ndarray,
    object_positions: np.ndarray,
) -> np.ndarray:
    cam_pos = camera_poses[:, :3, 3]
    dist = np.linalg.norm(object_positions[None, :, :] - cam_pos[:, None, :], axis=-1)
    return np.argmin(dist, axis=1)


def _feature_field_retrieval_accuracy(
    model: Any,
    test_images: np.ndarray,
    test_poses: np.ndarray,
    object_features: np.ndarray,
    object_positions: np.ndarray,
) -> float:
    height, width = test_images.shape[1], test_images.shape[2]
    feature_dim = object_features.shape[-1]
    rendered = []
    for view_idx in range(test_poses.shape[0]):
        feat_map = model.render_features(test_poses[view_idx], height, width)
        rendered.append(feat_map.detach().cpu().numpy().reshape(-1, feature_dim).mean(axis=0))
    query_features = np.stack(rendered)
    gallery_features = object_features[None, :] if object_features.ndim == 1 else object_features
    gallery_labels = np.arange(gallery_features.shape[0])
    query_labels = _feature_field_query_labels(test_poses, object_positions)
    metrics = model.evaluate_retrieval(
        query_features, gallery_features, gallery_labels, query_labels
    )
    retrieval_k = model.config.retrieval_k
    return float(metrics.get(f"accuracy_at_{retrieval_k}", metrics["accuracy_at_1"]))


def _eval_module_03(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test, meta = data.train, data.test, data.metadata
    images, poses = train["images"][0], train["camera_poses"][0]
    test_images, test_poses = test["images"][0], test["camera_poses"][0]
    gnss_drift = float(meta.get("mean_gnss_drift_m", 0.0))

    if ctx.class_path.endswith("NeRF"):
        model.fit(images, poses)
        psnr_val = model.evaluate_psnr(test_images, test_poses)
        metrics = {"psnr": psnr_val, "rmse": float(10 ** (-psnr_val / 20)), "gnss_drift_m": gnss_drift}
        return _add_normalized(metrics, "psnr", 35.0, 15.0, metadata=meta)

    if ctx.class_path.endswith("GaussianSplatting"):
        model.fit(images, poses)
        metrics = model.evaluate(test_images, test_poses)
        metrics["gnss_drift_m"] = gnss_drift
        return _add_normalized(metrics, "psnr", 35.0, 15.0, metadata=meta)

    train_objects = train["object_features"][0]
    v, h, w = images.shape[0], images.shape[1], images.shape[2]
    teacher_maps = _scene_teacher_feature_maps(train_objects, v, h, w)
    model.gs.fit(images, poses)
    model.fit(teacher_maps, poses, images=images)
    feature_accuracy = _feature_field_retrieval_accuracy(
        model,
        test_images,
        test_poses,
        test["object_features"][0],
        test["object_positions"][0],
    )
    metrics = {"feature_accuracy": feature_accuracy, "gnss_drift_m": gnss_drift}
    return _add_normalized(metrics, "feature_accuracy", 0.95, 0.25, metadata=meta)


def _eval_module_04(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    if evaluate_ssl is None:
        raise ImportError(
            "cv_robotics is required for SSL evaluation (pip install -e ../11-cv-robotics-unified)"
        )

    train, test, meta = data.train, data.test, data.metadata
    name = ctx.model_name
    trace_density = float(meta.get("trace_density", meta.get("field_trace_density", 1.0)))

    if name == "PointMAE":
        model.fit(train["point_clouds"])
        train_reps = model.encode(train["point_clouds"])
        test_reps = model.encode(test["point_clouds"])
        train_labels, test_labels = train["point_cloud_labels"], test["point_cloud_labels"]
        label_mask = train.get("point_cloud_label_mask")
    elif name == "MAE":
        model.fit(train["features"], train["masked_inputs"], train["masks"])
        train_reps = model.encode(train["features"])
        test_reps = model.encode(test["features"])
        train_labels, test_labels = train["labels"], test["labels"]
        label_mask = train.get("label_mask")
    elif name == "DINOv2":
        model.fit(train["student_features"], train["teacher_features"])
        train_reps = model.encode(train["features"])
        test_reps = model.encode(test["features"])
        train_labels, test_labels = train["labels"], test["labels"]
        label_mask = train.get("label_mask")
    else:
        model.fit(train["augmented_view1"], train["augmented_view2"])
        train_reps = model.encode(train["features"])
        test_reps = model.encode(test["features"])
        train_labels, test_labels = train["labels"], test["labels"]
        label_mask = train.get("label_mask")

    metrics = evaluate_ssl(
        train_reps,
        train_labels,
        test_reps,
        test_labels,
        n_clusters=int(meta.get("n_clusters", 8)),
        label_mask=label_mask,
    )
    metrics["silhouette"] = metrics.pop("silhouette_score", 0.0)
    metrics["trace_density"] = trace_density
    return _add_normalized(metrics, "linear_probe_accuracy", 0.95, 0.15, metadata=meta)


def _eval_module_05(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test, meta = data.train, data.test, data.metadata
    sim_field_gap = float(meta.get("sim_field_gap", meta.get("domain_shift", 0.0)))

    model.fit(train["X_source"], train["y_source"], X_target=train["X_target"])
    raw = model.evaluate(test["X_source"], test["y_source"], test["X_target"], test["y_target"])
    metrics = {
        "source_accuracy": raw.get("source_acc", 0.0),
        "target_accuracy": raw.get("target_acc", 0.0),
        "domain_gap": raw.get("domain_gap", 0.0),
        "n_trainable_params": raw.get("n_trainable_params", 0.0),
        "sim_field_gap": sim_field_gap,
    }
    return _add_normalized(metrics, "target_accuracy", 0.95, 0.25, metadata=meta)


def _train_visuomotor(model: Any, train: dict[str, np.ndarray]) -> None:
    if hasattr(model, "teacher") and hasattr(model.teacher, "train"):
        model.teacher.train(train["observations"], train["actions"])
    train_sig = inspect.signature(model.train)
    if "rewards" in train_sig.parameters:
        model.train(train["observations"], train["actions"], train["rewards"])
    else:
        model.train(train["observations"], train["actions"])


def _eval_module_06(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test, meta = data.train, data.test, data.metadata
    cmd_latency_ms = float(meta.get("cmd_latency_ms", meta.get("mean_cmd_latency_ms", 0.0)))

    _train_visuomotor(model, train)
    modes_true = test.get("modes", np.zeros(len(test["actions"]), dtype=np.int64))
    eval_sig = inspect.signature(model.evaluate)
    if "modes_true" in eval_sig.parameters:
        metrics_obj = model.evaluate(test["observations"], test["actions"], modes_true)
    else:
        metrics_obj = model.evaluate(test["observations"], test["actions"])
    metrics = {k: float(v) for k, v in metrics_obj.__dict__.items()}
    metrics["cmd_latency_ms"] = cmd_latency_ms
    if "action_mse" in metrics:
        metrics["mse"] = metrics["action_mse"]
    elif "train_loss" in metrics and metrics["train_loss"] > 0:
        metrics["mse"] = metrics["train_loss"]
    elif "regret" in metrics:
        metrics["mse"] = metrics["regret"]
    if "normalized_score" in metrics:
        return metrics
    return _add_normalized(metrics, "mse", 0.05, 1.0, metadata=meta)


def _eval_module_07(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test, meta = data.train, data.test, data.metadata

    if ctx.class_path.endswith("CausalBottleneck"):
        model.train(train["X"])
        metrics_obj = model.evaluate(test["X"], test["T"], test["Y"], test["cate_true"])
        metrics = {k: float(v) for k, v in metrics_obj.__dict__.items()}
        return _add_normalized(metrics, "pehe_bn", 0.05, 1.0, metadata=meta)

    model.fit(train["X"], train["T"], train["Y"])
    cate_metrics = model.evaluate(test["X"], test["cate_true"])
    if hasattr(cate_metrics, "pehe"):
        metrics = {"pehe": float(cate_metrics.pehe), "ate_error": float(cate_metrics.ate_error)}
    else:
        metrics = {"pehe": float(cate_metrics["pehe"]), "ate_error": float(cate_metrics["ate_error"])}
    return _add_normalized(metrics, "pehe", 0.05, 1.0, metadata=meta)


def _eval_module_08(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test, meta = data.train, data.test, data.metadata
    s, a, ns = _flatten_wm(train["states"], train["actions"], train["next_states"])
    eval_next = test.get("next_states_clean", test["next_states"])
    ts, ta, tns = _flatten_wm(test["states"], test["actions"], eval_next)
    model_error = float(meta.get("model_error", resolve_violation_severity(meta) * 0.1))

    if ctx.class_path.endswith("RSSM"):
        model.train(train["observations"], train["actions"])
        metrics_obj = model.evaluate(test["observations"], test["actions"])
    elif ctx.class_path.endswith("CausalWorldModel"):
        model.train(s, a, ns)
        out = model.evaluate(ts, ta, tns, true_causal_graph=meta.get("causal_graph"))
        out["model_error"] = model_error
        return _add_normalized(out, "transition_mse", 0.01, 1.0, metadata=meta)
    else:
        model.train(s, a, ns)
        metrics_obj = model.evaluate(ts, ta, tns)

    out = {k: float(v) for k, v in metrics_obj.__dict__.items()}
    out["model_error"] = model_error
    return _add_normalized(out, "transition_mse", 0.01, 1.0, metadata=meta)


_eval_module_3 = _eval_module_03
_eval_module_4 = _eval_module_04
_eval_module_5 = _eval_module_05
_eval_module_6 = _eval_module_06
_eval_module_7 = _eval_module_07
_eval_module_8 = _eval_module_08
