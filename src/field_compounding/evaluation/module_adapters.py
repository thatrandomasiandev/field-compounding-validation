"""Module-specific training and evaluation adapters."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from field_compounding.data.base import BenchmarkData
from field_compounding.evaluation.metrics import normalized_score
try:
    from cv_robotics.models.ssl import SSLConfig, evaluate_ssl
except ImportError:
    SSLConfig = None
    evaluate_ssl = None


@dataclass
class TrialContext:
    module_id: int
    model_name: str
    class_path: str
    params: dict[str, Any]
    fast: bool = False



FIELD_METADATA_KEYS: tuple[str, ...] = (
    "loop_node", "trace_density", "field_trace_density", "mean_gnss_drift_m",
    "inferred_violation_severity", "violation_severity", "trace_source",
)

def resolve_violation_severity(metadata: dict[str, Any]) -> float:
    if "violation_severity" in metadata:
        return float(np.clip(metadata["violation_severity"], 0.0, 1.0))
    if "inferred_violation_severity" in metadata:
        return float(np.clip(metadata["inferred_violation_severity"], 0.0, 1.0))
    return 0.0

def enrich_field_metrics(metrics: dict[str, float], metadata: dict[str, Any]) -> dict[str, float]:
    out = dict(metrics)
    out["field_violation_severity"] = resolve_violation_severity(metadata)
    for key in FIELD_METADATA_KEYS:
        if key in metadata and isinstance(metadata[key], (int, float, str)):
            out[f"field_{key}"] = float(metadata[key]) if isinstance(metadata[key], (int, float)) else metadata[key]
    return out

def field_adjusted_baseline(baseline: float, optimal: float, metadata: dict[str, Any]) -> float:
    v = resolve_violation_severity(metadata)
    return baseline + v * (optimal - baseline) * 0.25

CLASS_ALIASES: dict[str, str] = {
    "cv_robotics.models.gnn.GCN": "cv_robotics.models.gnn.GNNTrainer",
    "cv_robotics.models.gnn.RollingGCN": "cv_robotics.models.gnn.GNNTrainer",
    "cv_robotics.models.gnn.GIN": "cv_robotics.models.gnn.GNNTrainer",
    "cv_robotics.models.egnn.EGNN": "cv_robotics.models.egnn.EGNNTrainer",
    "cv_robotics.models.hamiltonian_nn.HamiltonianNN": "cv_robotics.models.hamiltonian_nn.HNNTrainer",
    "cv_robotics.models.neural_ode.NeuralODE": "cv_robotics.models.neural_ode.NeuralODETrainer",
    "cv_robotics.models.pinn.PINN": "cv_robotics.models.pinn.PINNTrainer",
    "cv_robotics.models.equivariant_flow.EquivariantFlow": "cv_robotics.models.equivariant_flow.EquivariantFlowMatching",
    "cv_robotics.models.world_model.RSSM": "cv_robotics.models.world_model.RSSMTrainer",
    "cv_robotics.models.feature_fields.FeatureField": "cv_robotics.models.feature_fields.FeatureFieldDistillation",
}


def build_model(ctx: TrialContext, metadata: dict[str, Any]) -> Any:
    """Instantiate a model from config."""
    class_path = CLASS_ALIASES.get(ctx.class_path, ctx.class_path)
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    params = dict(ctx.params)

    if ctx.fast:
        for key in ("epochs", "n_epochs", "n_rounds", "n_steps", "n_value_iters"):
            if key in params and isinstance(params[key], int):
                params[key] = max(2, min(params[key], 5))

    if class_name in {"SimCLR", "VICReg", "MAE", "DINOv2", "PointMAE"}:
        cfg = SSLConfig(
            input_dim=int(metadata.get("feature_dim", 10)),
            lr=float(params.pop("lr", 1e-3)),
            n_epochs=int(params.pop("epochs", params.pop("n_epochs", 100))),
            temperature=float(params.pop("temperature", 0.1)),
            projection_dim=int(params.pop("proj_dim", 32)),
            vicreg_lambda=float(params.pop("lambda_var", 25.0)),
            vicreg_mu=float(params.pop("lambda_inv", 25.0)),
            vicreg_nu=float(params.pop("lambda_cov", 1.0)),
            mae_mask_ratio=float(params.pop("mask_ratio", 0.5)),
            dino_momentum=float(params.pop("ema_momentum", 0.996)),
            pointmae_mask_ratio=float(params.pop("mask_ratio", 0.6)),
        )
        return cls(config=cfg)

    if class_name == "NeRF":
        from cv_robotics.models.nerf import NeRF, NeRFConfig

        return NeRF(
            NeRFConfig(
                hidden_dim=int(params.pop("hidden_dim", 64)),
                n_samples=int(params.pop("n_samples", 64)),
                lr=float(params.pop("lr", 1e-3)),
                n_epochs=int(params.pop("epochs", 100)),
            )
        )

    if class_name == "GaussianSplatting":
        from cv_robotics.models.gaussian_splatting import GaussianSplatting, GaussianSplattingConfig

        return GaussianSplatting(
            GaussianSplattingConfig(
                n_init_gaussians=int(params.pop("n_gaussians", params.pop("n_init_gaussians", 200))),
                n_epochs=int(params.pop("epochs", 100)),
            )
        )

    if class_name == "FeatureFieldDistillation":
        from cv_robotics.models.feature_fields import FeatureFieldConfig, FeatureFieldDistillation
        from cv_robotics.models.gaussian_splatting import GaussianSplatting, GaussianSplattingConfig

        gs = GaussianSplatting(
            GaussianSplattingConfig(
                n_init_gaussians=100 if ctx.fast else 200,
                n_epochs=3 if ctx.fast else int(params.pop("epochs", 50)),
            )
        )
        cfg = FeatureFieldConfig(feature_dim=int(params.pop("feature_dim", 32)))
        return FeatureFieldDistillation(gs, cfg)

    if class_name in {"LinearProbe", "LoRA", "DoRA", "VisualPromptTuning", "DANN"}:
        from cv_robotics.models.peft import PEFTConfig

        cfg = PEFTConfig(
            input_dim=int(metadata.get("feature_dim", 10)),
            n_classes=int(metadata.get("n_classes", 4)),
            lr=float(params.pop("lr", 1e-3)),
            n_epochs=int(params.pop("epochs", 50)),
            lora_rank=int(params.pop("rank", 4)),
            lora_alpha=float(params.pop("alpha", 8.0)),
            prompt_length=int(params.pop("n_prompts", params.pop("prompt_length", 8))),
            dann_lambda_max=float(params.pop("lambda_domain", params.pop("dann_lambda_max", 1.0))),
        )
        return cls(config=cfg, pretrained_weights=metadata.get("backbone_weights"))

    if class_name == "GNNTrainer":
        model_type = {"GCN": "gcn", "RollingGCN": "rolling_gcn", "GIN": "gin"}[ctx.model_name]
        return GNNTrainer(
            model_type=model_type,
            input_dim=int(metadata.get("feature_dim", 32)),
            hidden_dim=int(params.pop("hidden_dim", 64)),
            lr=float(params.pop("lr", 1e-3)),
        )

    if class_name in {"SLearner", "TLearner", "XLearner", "DRLearner", "RLearner"}:
        return cls(n_estimators=int(params.pop("n_estimators", 100)))

    if class_name in {"BehavioralCloning", "CQL", "IQL"}:
        kwargs = {
            "obs_dim": int(metadata.get("obs_dim", 38)),
            "action_dim": int(metadata.get("action_dim", 2)),
            "n_epochs": int(params.pop("epochs", 100)),
            "lr": float(params.pop("lr", 1e-3)),
        }
        if class_name == "CQL":
            kwargs["alpha"] = float(params.pop("alpha", 1.0))
        if class_name == "IQL":
            kwargs["tau"] = float(params.pop("expectile", params.pop("tau", 0.7)))
        return cls(**kwargs)

    if class_name in {"DiffusionPolicy", "FlowMatchingPolicy", "OneStepFlowPolicy"}:
        kwargs = {
            "obs_dim": int(metadata.get("obs_dim", 38)),
            "action_dim": int(metadata.get("action_dim", 2)),
            "n_epochs": int(params.pop("epochs", 100)),
            "lr": float(params.pop("lr", 1e-3)),
        }
        if class_name == "DiffusionPolicy":
            kwargs["n_diffusion_steps"] = int(params.pop("n_diffusion_steps", 20 if ctx.fast else 100))
        if class_name == "FlowMatchingPolicy":
            kwargs["n_euler_steps"] = int(params.pop("n_euler_steps", 5 if ctx.fast else 10))
        if class_name == "OneStepFlowPolicy":
            teacher = FlowMatchingPolicy(
                obs_dim=kwargs["obs_dim"],
                action_dim=kwargs["action_dim"],
                n_euler_steps=int(params.pop("n_euler_steps", 5 if ctx.fast else 10)),
                n_epochs=max(2, kwargs["n_epochs"] // 2),
                lr=kwargs["lr"],
            )
            return cls(teacher=teacher, **kwargs)
        return cls(**kwargs)

    if class_name == "EnsembleWorldModel":
        return cls(
            state_dim=int(metadata.get("state_dim", 6)),
            action_dim=int(metadata.get("action_dim", 3)),
            n_ensemble=int(params.pop("n_members", 3 if ctx.fast else 5)),
            hidden=int(params.pop("hidden_dim", 128)),
            n_epochs=int(params.pop("epochs", 100)),
            lr=float(params.pop("lr", 1e-3)),
        )

    if class_name == "RSSMTrainer":
        return cls(
            obs_dim=int(metadata.get("obs_dim", 22)),
            action_dim=int(metadata.get("action_dim", 3)),
            latent_dim=int(params.pop("latent_dim", 32)),
            gru_hidden=int(params.pop("gru_hidden", 64)),
            n_epochs=int(params.pop("epochs", 100)),
            lr=float(params.pop("lr", 1e-3)),
        )

    if class_name == "CausalWorldModel":
        return cls(
            state_dim=int(metadata.get("state_dim", 6)),
            action_dim=int(metadata.get("action_dim", 3)),
            l1_weight=float(params.pop("sparsity_lambda", 0.01)),
            n_epochs=int(params.pop("epochs", 100)),
            lr=float(params.pop("lr", 1e-3)),
        )

    if class_name == "EGNNTrainer":
        return cls(
            input_dim=int(metadata.get("feature_dim", 16)),
            hidden_dim=int(params.pop("hidden_dim", 64)),
            n_layers=int(params.pop("n_layers", 4)),
            lr=float(params.pop("lr", 1e-3)),
        )

    if class_name == "HNNTrainer":
        return cls(hidden_dim=int(params.pop("hidden_dim", 64)), lr=float(params.pop("lr", 1e-3)))

    if class_name == "NeuralODETrainer":
        return cls(state_dim=6, hidden_dim=int(params.pop("hidden_dim", 128)), lr=float(params.pop("lr", 1e-3)))

    if class_name == "PINNTrainer":
        return cls(
            hidden_dim=int(params.pop("hidden_dim", 128)),
            lambda_r=float(params.pop("lambda_physics", 1.0)),
            lr=float(params.pop("lr", 1e-3)),
        )

    if class_name == "EquivariantFlowMatching":
        feat_dim = int(metadata.get("feature_dim", 16))
        return cls(
            n_nodes=3,
            feature_dim=feat_dim,
            hidden_dim=int(params.pop("hidden_dim", 64)),
            n_layers=int(params.pop("n_layers", 4)),
            lr=float(params.pop("lr", 1e-3)),
        )

    if class_name == "MCDropout":
        return cls(
            input_dim=int(metadata.get("input_dim", 5)),
            dropout_p=float(params.pop("dropout_p", 0.1)),
            T=int(params.pop("n_forward_passes", 10 if ctx.fast else 50)),
            lr=float(params.pop("lr", 1e-3)),
        )

    if class_name == "DeepEnsemble":
        return cls(
            input_dim=int(metadata.get("input_dim", 5)),
            M=int(params.pop("n_members", 3 if ctx.fast else 5)),
            lr=float(params.pop("lr", 1e-3)),
        )

    if class_name == "FedAvg":
        return cls(
            d_in=int(metadata.get("d_features", 20)),
            n_classes=int(metadata.get("n_classes", 10)),
            local_epochs=int(params.pop("local_epochs", 5)),
            lr=float(params.pop("lr", 0.01)),
            n_rounds=int(params.pop("n_rounds", 30)),
        )

    if class_name == "FedProx":
        return cls(
            d_in=int(metadata.get("d_features", 20)),
            n_classes=int(metadata.get("n_classes", 10)),
            local_epochs=int(params.pop("local_epochs", 5)),
            lr=float(params.pop("lr", 0.01)),
            n_rounds=int(params.pop("n_rounds", 30)),
            mu=float(params.pop("mu", 0.01)),
        )

    if class_name == "DPSGD":
        return cls(
            d_in=int(metadata.get("d_features", 20)),
            n_classes=int(metadata.get("n_classes", 10)),
            n_epochs=int(params.pop("n_rounds", 30)),
            lr=float(params.pop("lr", 0.01)),
            clip_norm=float(params.pop("clip_norm", 1.0)),
            noise_sigma=float(params.pop("noise_sigma", 1.0)),
        )

    if class_name == "SecureAggregation":
        return cls(
            n_clients=int(metadata.get("n_clients", 5)),
            d_update=int(params.pop("d_update", 256)),
        )

    if class_name == "SymbolicSolver":
        return cls(entities=list(range(int(metadata.get("n_objects", 80)))))

    if class_name == "BeamSearchSynthesizer":
        return cls(
            beam_width=int(params.pop("beam_width", 100)),
            max_depth=int(params.pop("max_depth", 5)),
        )

    if class_name == "DifferentiableReasoner":
        return cls(
            n_atoms=int(metadata.get("n_atoms", 8)),
            semantics=str(params.pop("semantics", "product")),
            lr=float(params.pop("lr", 0.01)),
            n_steps=int(params.pop("n_steps", 500)),
        )

    if class_name == "NeuralTheoremProver":
        return cls(
            n_entities=int(metadata.get("n_objects", 80)),
            embed_dim=int(params.pop("embed_dim", 32)),
            tau=float(params.pop("tau", 1.0)),
            lr=float(params.pop("lr", 1e-3)),
            n_steps=int(params.pop("epochs", 100)),
        )

    if class_name == "CBFSafetyFilter":
        obs_pos = np.array(metadata["obstacle_positions"], dtype=np.float64)
        obs_rad = np.array(metadata["obstacle_radii"], dtype=np.float64)
        return cls(obs_pos, obs_rad, alpha=float(params.pop("alpha", 1.0)))

    if class_name == "HJReachability":
        return cls(
            grid_size=int(params.pop("grid_size", 50)),
            n_steps=int(params.pop("n_value_iters", 100)),
            dt=float(params.pop("dt", 0.1)),
        )

    if class_name == "CausalBottleneck":
        return cls(
            in_dim=int(metadata.get("obs_dim", 10)),
            latent_dim=int(params.pop("latent_dim", 5)),
            lr=float(params.pop("lr", 1e-3)),
            n_epochs=int(params.pop("epochs", 100)),
        )

    return _filter_init(cls, params)


def _filter_init(cls: type, params: dict[str, Any]) -> Any:
    sig = inspect.signature(cls.__init__)
    allowed = {k for k in sig.parameters if k != "self"}
    filtered = {k: v for k, v in params.items() if k in allowed}
    return cls(**filtered)


def run_trial(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    handlers = {3: _eval_module_03, 4: _eval_module_04, 5: _eval_module_05, 6: _eval_module_06, 7: _eval_module_07, 8: _eval_module_08}
    if ctx.module_id not in handlers:
        raise NotImplementedError("module_id=%s adapters are owned by Agent 23 (modules 9–14)" % ctx.module_id)
    metrics = handlers[ctx.module_id](model, ctx, data)
    return enrich_field_metrics(metrics, data.metadata)


def _add_normalized(metrics: dict[str, float], primary: str, optimal: float, baseline: float = 0.0, metadata: dict[str, Any] | None = None) -> dict[str, float]:
    if metadata is not None:
        baseline = field_adjusted_baseline(baseline, optimal, metadata)
    if primary in metrics:
        if primary == "mse" or primary == "pehe" or primary == "ece" or primary == "energy_drift":
            metrics["normalized_score"] = normalized_score(-metrics[primary], -optimal, -baseline)
        else:
            metrics["normalized_score"] = normalized_score(metrics[primary], optimal, baseline)
    return metrics


def _flatten_wm(states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, t, sd = states.shape
    _, _, ad = actions.shape
    s = states.reshape(n * t, sd)
    a = actions.reshape(n * t, ad)
    ns = next_states.reshape(n * t, sd)
    return s, a, ns


def _scene_teacher_feature_maps(
    object_features: np.ndarray,
    n_views: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Build (V, H, W, D) teacher maps from per-object scene embeddings."""
    if object_features.ndim == 1:
        teacher_vec = object_features
    else:
        teacher_vec = object_features.mean(axis=0)
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
        rendered.append(
            feat_map.detach().cpu().numpy().reshape(-1, feature_dim).mean(axis=0)
        )
    query_features = np.stack(rendered)
    gallery_features = object_features[None, :] if object_features.ndim == 1 else object_features
    gallery_labels = np.arange(gallery_features.shape[0])
    query_labels = _feature_field_query_labels(test_poses, object_positions)
    metrics = model.evaluate_retrieval(
        query_features,
        gallery_features,
        gallery_labels,
        query_labels,
    )
    retrieval_k = model.config.retrieval_k
    return float(metrics.get(f"accuracy_at_{retrieval_k}", metrics["accuracy_at_1"]))


def _eval_module_03(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test, meta = data.train, data.test, data.metadata
    gnss_drift = float(meta.get("mean_gnss_drift_m", 0.0))
    images, poses = train["images"][0], train["camera_poses"][0]
    test_images, test_poses = test["images"][0], test["camera_poses"][0]

    if ctx.class_path.endswith("NeRF"):
        model.fit(images, poses)
        psnr_val = model.evaluate_psnr(test_images, test_poses)
        metrics = {"psnr": psnr_val, "rmse": float(10 ** (-psnr_val / 20)), "gnss_drift_m": gnss_drift}
        return _add_normalized(metrics, "psnr", 35.0, 15.0, metadata=meta)

    if ctx.class_path.endswith("GaussianSplatting"):
        model.fit(images, poses)
        metrics = model.evaluate(test_images, test_poses)
        return _add_normalized(metrics, "psnr", 35.0, 15.0)

    train_objects = train["object_features"][0]
    v, h, w = images.shape[0], images.shape[1], images.shape[2]
    teacher_maps = _scene_teacher_feature_maps(train_objects, v, h, w)
    model.gs.fit(images, poses)
    model.fit(teacher_maps, poses, images=images)
    test_objects = test["object_features"][0]
    test_positions = test["object_positions"][0]
    feature_accuracy = _feature_field_retrieval_accuracy(
        model, test_images, test_poses, test_objects, test_positions
    )
    return _add_normalized({"feature_accuracy": feature_accuracy}, "feature_accuracy", 0.95, 0.25)


def _eval_module_04(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test = data.train, data.test
    name = ctx.model_name

    if name == "PointMAE":
        model.fit(train["point_clouds"])
        train_reps = model.encode(train["point_clouds"])
        test_reps = model.encode(test["point_clouds"])
        train_labels = train["point_cloud_labels"]
        test_labels = test["point_cloud_labels"]
        label_mask = train.get("point_cloud_label_mask")
    elif name == "MAE":
        model.fit(train["features"], train["masked_inputs"], train["masks"])
        train_reps = model.encode(train["features"])
        test_reps = model.encode(test["features"])
        train_labels = train["labels"]
        test_labels = test["labels"]
        label_mask = train.get("label_mask")
    elif name == "DINOv2":
        model.fit(train["student_features"], train["teacher_features"])
        train_reps = model.encode(train["features"])
        test_reps = model.encode(test["features"])
        train_labels = train["labels"]
        test_labels = test["labels"]
        label_mask = train.get("label_mask")
    else:
        model.fit(train["augmented_view1"], train["augmented_view2"])
        train_reps = model.encode(train["features"])
        test_reps = model.encode(test["features"])
        train_labels = train["labels"]
        test_labels = test["labels"]
        label_mask = train.get("label_mask")

    metrics = evaluate_ssl(
        train_reps,
        train_labels,
        test_reps,
        test_labels,
        n_clusters=int(data.metadata.get("n_clusters", 8)),
        label_mask=label_mask,
    )
    metrics["silhouette"] = metrics.pop("silhouette_score", 0.0)
    return _add_normalized(metrics, "linear_probe_accuracy", 0.95, 0.15)


def _eval_module_05(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test = data.train, data.test
    model.fit(
        train["X_source"],
        train["y_source"],
        X_target=train["X_target"],
    )
    metrics = model.evaluate(test["X_source"], test["y_source"], test["X_target"], test["y_target"])
    metrics = {
        "source_accuracy": metrics.get("source_acc", 0.0),
        "target_accuracy": metrics.get("target_acc", 0.0),
        "domain_gap": metrics.get("domain_gap", 0.0),
        "n_trainable_params": metrics.get("n_trainable_params", 0.0),
    }
    return _add_normalized(metrics, "target_accuracy", 0.95, 0.25)


def _train_visuomotor(model: Any, train: dict[str, np.ndarray]) -> None:
    if hasattr(model, "teacher") and hasattr(model.teacher, "train"):
        model.teacher.train(train["observations"], train["actions"])
    train_sig = inspect.signature(model.train)
    if "rewards" in train_sig.parameters:
        model.train(train["observations"], train["actions"], train["rewards"])
    else:
        model.train(train["observations"], train["actions"])


def _eval_module_06(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test = data.train, data.test
    _train_visuomotor(model, train)
    modes_true = test.get("modes", np.zeros(len(test["actions"]), dtype=np.int64))
    eval_sig = inspect.signature(model.evaluate)
    if "modes_true" in eval_sig.parameters:
        metrics_obj = model.evaluate(test["observations"], test["actions"], modes_true)
    else:
        metrics_obj = model.evaluate(test["observations"], test["actions"])
    metrics = {k: float(v) for k, v in metrics_obj.__dict__.items()}
    # Flow policies report train_loss=0; prefer rollout action_mse for the primary metric.
    if "action_mse" in metrics:
        metrics["mse"] = metrics["action_mse"]
    elif "train_loss" in metrics and metrics["train_loss"] > 0:
        metrics["mse"] = metrics["train_loss"]
    elif "regret" in metrics:
        metrics["mse"] = metrics["regret"]
    if "normalized_score" in metrics:
        return metrics
    return _add_normalized(metrics, "mse", 0.05, 1.0)


def _eval_module_07(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test, meta = data.train, data.test, data.metadata
    if ctx.class_path.endswith("CausalBottleneck"):
        model.train(train["X"])
        metrics_obj = model.evaluate(test["X"], test["T"], test["Y"], test["cate_true"])
        metrics = {k: float(v) for k, v in metrics_obj.__dict__.items()}
        return _add_normalized(metrics, "pehe_bn", 0.05, 1.0)

    model.fit(train["X"], train["T"], train["Y"])
    cate_metrics: CATEMetrics = model.evaluate(test["X"], test["cate_true"])
    metrics = {"pehe": cate_metrics.pehe, "ate_error": cate_metrics.ate_error}
    return _add_normalized(metrics, "pehe", 0.05, 1.0)


def _eval_module_08(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test, meta = data.train, data.test, data.metadata
    s, a, ns = _flatten_wm(train["states"], train["actions"], train["next_states"])
    eval_next = test.get("next_states_clean", test["next_states"])
    ts, ta, tns = _flatten_wm(test["states"], test["actions"], eval_next)

    if ctx.class_path.endswith("RSSM"):
        model.train(train["observations"], train["actions"])
        metrics_obj = model.evaluate(test["observations"], test["actions"])
    elif ctx.class_path.endswith("CausalWorldModel"):
        model.train(s, a, ns)
        out = model.evaluate(ts, ta, tns, true_causal_graph=meta.get("causal_graph"))
        return _add_normalized(out, "transition_mse", 0.01, 1.0)
    else:
        model.train(s, a, ns)
        metrics_obj = model.evaluate(ts, ta, tns)

    out = {k: float(v) for k, v in metrics_obj.__dict__.items()}
    return _add_normalized(out, "transition_mse", 0.01, 1.0)


def _eval_module_9(model: Any, ctx: TrialContext, data: BenchmarkData) -> dict[str, float]:
    train, test = data.train, data.test
    epochs = 3 if ctx.fast else 15
    name = ctx.model_name
    n_eval = min(8, len(test["q"]))

    if name == "EGNN":
        # Train on multiple timesteps so corruption severity affects the learned map.
        t_idx = np.linspace(0, train["positions"].shape[1] - 1, num=min(5, train["positions"].shape[1]), dtype=int)
        for _ in range(epochs):
            for t in t_idx:
                model.train_step(
                    train["positions"][:, t, :, :],
                    train["features"][:, t, :, :],
                    train["energies"][:, t],
                )
        metrics = model.evaluate(
            test["positions"][:, -1, :, :],
            test["features"][:, -1, :, :],
            test["energies"][:, -1],
        )
        return _add_normalized(metrics, "rmse", 0.05, 1.0)

    if name == "HNN":
        dq_dt, dp_dt = model._finite_difference_targets(train["q"], train["dq"])
        q_flat = train["q"][:, :-1].reshape(-1, 3)
        p_flat = train["dq"][:, :-1].reshape(-1, 3)
        dq_dt = dq_dt.reshape(-1, 3)
        dp_dt = dp_dt.reshape(-1, 3)
        for _ in range(epochs):
            model.train_step_hnn(q_flat, p_flat, dq_dt, dp_dt)
        test_dq, test_dp = model._finite_difference_targets(test["q"], test["dq"])
        q_eval = test["q"][:, :-1].reshape(-1, 3)
        p_eval = test["dq"][:, :-1].reshape(-1, 3)
        dq_eval = test_dq.reshape(-1, 3)
        dp_eval = test_dp.reshape(-1, 3)
        import torch

        model.hnn.eval()
        q_t = torch.from_numpy(q_eval).float().to(model.device).requires_grad_(True)
        p_t = torch.from_numpy(p_eval).float().to(model.device).requires_grad_(True)
        with torch.enable_grad():
            dq_pred, dp_pred = model.hnn.time_derivative(q_t, p_t)
        dq_err = (dq_pred.detach().cpu().numpy() - dq_eval) ** 2
        dp_err = (dp_pred.detach().cpu().numpy() - dp_eval) ** 2
        rmse = float(np.sqrt(np.mean(dq_err) + np.mean(dp_err)))
        drifts = [
            float(model.evaluate_energy_drift(test["q"][i, 0], test["dq"][i, 0])["hnn_energy_drift"])
            for i in range(n_eval)
        ]
        drift = float(np.mean(drifts))
        return _add_normalized({"energy_drift": drift, "rmse": rmse}, "rmse", 0.05, 1.0)

    if name == "NeuralODE":
        states = np.concatenate([train["q"], train["dq"]], axis=-1)[:, :-1, :]
        next_states = np.concatenate([train["q"], train["dq"]], axis=-1)[:, 1:, :]
        s_flat = states.reshape(-1, 6)
        ns_flat = next_states.reshape(-1, 6)
        for _ in range(epochs):
            model.train_step(s_flat, ns_flat)
        rmses = []
        for i in range(n_eval):
            x0 = np.concatenate([test["q"][i, 0], test["dq"][i, 0]])
            x_true = np.concatenate([test["q"][i], test["dq"][i]], axis=-1)
            rollout = model.evaluate_rollout(x0[None, :], x_true[None, :, :])
            rmses.append(float(rollout["node_rmse"]))
        rmse = float(np.mean(rmses))
        return _add_normalized({"rmse": rmse}, "rmse", 0.05, 1.0)

    if name == "PINN":
        q0 = train["q"][:, 0]
        dq0 = train["dq"][:, 0]
        tau = train["torques"][:, 0]
        t = np.full((len(q0), 1), 0.02, dtype=np.float32)
        q_true = train["q"][:, 1]
        for _ in range(epochs):
            model.train_step(q0, dq0, tau, t, q_true)
        t_eval = np.full((len(test["q"]), 1), 0.02, dtype=np.float32)
        metrics = model.evaluate(
            test["q"][:, 0],
            test["dq"][:, 0],
            test["torques"][:, 0],
            t_eval,
            test["q"][:, 1],
        )
        return _add_normalized({k: float(v) for k, v in metrics.items()}, "rmse", 0.05, 1.0)

    pos = train["positions"][:, -1, :, :]
    feat = train["features"][:, -1, :, :]
    for _ in range(epochs):
        for i in range(pos.shape[0]):
            model.train_step(pos[i], feat[i])
    errors: list[float] = []
    n_flow_eval = min(8, len(test["positions"]))
    for i in range(n_flow_eval):
        target = test["positions"][i, -1]
        generated = model.sample(test["features"][i, -1])
        if not np.all(np.isfinite(generated)):
            generated = np.nan_to_num(generated, nan=0.0, posinf=1e3, neginf=-1e3)
        if not np.all(np.isfinite(target)):
            target = np.nan_to_num(target, nan=0.0, posinf=1e3, neginf=-1e3)
        sq_err = float(np.mean((generated - target) ** 2))
        errors.append(sq_err if np.isfinite(sq_err) else 1e6)
    rmse = float(np.sqrt(np.mean(errors))) if errors else 1e6
    if not np.isfinite(rmse):
        rmse = 1e6
    consistency = float(model.evaluate_rotation_consistency(test["features"][0, -1]))
    return _add_normalized(
        {"rotation_consistency": consistency, "rmse": rmse},
        "rmse",
        0.05,
        1.0,
    )


_eval_module_3 = _eval_module_03
_eval_module_4 = _eval_module_04
_eval_module_5 = _eval_module_05
_eval_module_6 = _eval_module_06
_eval_module_7 = _eval_module_07
_eval_module_8 = _eval_module_08
