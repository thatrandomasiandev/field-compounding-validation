"""Field DGP for Module 3 (scene representation) on URC outdoor traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from field_compounding.data.base import BaseFieldDGP, BenchmarkData


def _fibonacci_hemisphere(n_views: int) -> np.ndarray:
    indices = np.arange(n_views, dtype=np.float64)
    phi = np.arccos(1 - (indices + 0.5) / n_views)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def _ray_sphere_intersect(
    origins: np.ndarray,
    directions: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    oc = origins - center[None, None, :]
    a = np.sum(directions**2, axis=-1)
    b = 2.0 * np.sum(oc * directions, axis=-1)
    c = np.sum(oc**2, axis=-1) - radius**2
    discriminant = b**2 - 4 * a * c
    mask = discriminant > 0
    t_hit = np.full_like(a, np.inf)
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
    t1 = (-b - sqrt_disc) / (2.0 * a + 1e-10)
    t2 = (-b + sqrt_disc) / (2.0 * a + 1e-10)
    t_hit = np.where(mask & (t1 > 0), t1, t_hit)
    t_hit = np.where(mask & (t1 <= 0) & (t2 > 0), t2, t_hit)
    valid = mask & (t_hit < np.inf) & (t_hit > 0)
    return t_hit, valid


def _render_scene(
    c2w: np.ndarray,
    objects: list[dict[str, Any]],
    resolution: int = 64,
    fov: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    h = w = resolution
    focal = 0.5 * w / np.tan(0.5 * np.radians(fov))
    u = np.arange(w, dtype=np.float64) - w / 2.0 + 0.5
    v = np.arange(h, dtype=np.float64) - h / 2.0 + 0.5
    uu, vv = np.meshgrid(u, v, indexing="xy")
    dirs_cam = np.stack([uu / focal, -vv / focal, -np.ones_like(uu)], axis=-1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    rot = c2w[:3, :3]
    dirs_world = np.einsum("ij,hwj->hwi", rot, dirs_cam)
    origins = np.broadcast_to(c2w[:3, 3], (h, w, 3)).copy()
    image = np.ones((h, w, 3), dtype=np.float64) * 0.2
    depth = np.full((h, w), np.inf, dtype=np.float64)
    for obj in objects:
        t_hit, valid = _ray_sphere_intersect(
            origins, dirs_world, obj["center"], obj["radius"]
        )
        closer = valid & (t_hit < depth)
        hit_pts = origins + dirs_world * t_hit[..., None]
        normals = (hit_pts - obj["center"][None, None, :]) / obj["radius"]
        light_dir = np.array([0.5, 0.5, 1.0])
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.maximum(np.sum(normals * light_dir, axis=-1), 0.1)
        color = obj["color"][None, None, :] * diffuse[..., None]
        for c_idx in range(3):
            image[:, :, c_idx] = np.where(closer, color[:, :, c_idx], image[:, :, c_idx])
        depth = np.where(closer, t_hit, depth)
    depth = np.where(np.isinf(depth), 0.0, depth)
    return np.clip(image, 0.0, 1.0).astype(np.float32), depth.astype(np.float32)


def load_urc_trace_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            rows.append(row)
    return rows


def summarize_scene_repr_traces(rows: list[dict[str, Any]]) -> dict[str, float]:
    scene_rows = [r for r in rows if r.get("loop_node") == "scene_repr"]
    if not scene_rows:
        return {
            "row_count": 0.0,
            "mean_gnss_drift_m": 0.0,
            "max_gnss_drift_m": 0.0,
            "mean_violation_severity": 0.0,
        }
    gnss = np.array([float(r["gnss_drift"]) for r in scene_rows], dtype=np.float64)
    sev = np.array([float(r["violation_severity"]) for r in scene_rows], dtype=np.float64)
    return {
        "row_count": float(len(scene_rows)),
        "mean_gnss_drift_m": float(np.mean(gnss)),
        "max_gnss_drift_m": float(np.max(gnss)),
        "mean_violation_severity": float(np.mean(sev)),
    }


def views_from_gnss_drift(
    gnss_drift_m: float,
    violation_severity: float,
    *,
    v_max: int = 32,
    gnss_drift_ref_m: float = 8.0,
    gnss_drift_scale: float = 1.0,
) -> tuple[int, float]:
    gnss_component = min(
        1.0,
        max(0.0, gnss_drift_m / gnss_drift_ref_m) * gnss_drift_scale,
    )
    v_raw = float(np.clip(max(violation_severity, gnss_component), 0.0, 1.0))
    n_views = max(4, min(int(v_max * (1.0 - v_raw)), v_max))
    return n_views, 1.0 - n_views / v_max


class SceneReprFieldDGP(BaseFieldDGP):
    """URC trace-backed scene representation DGP."""

    V_MAX: int = 32
    GNSS_DRIFT_REF_M: float = 8.0
    LOOP_NODE: str = "scene_repr"

    def __init__(
        self,
        seed: int = 42,
        violation_severity: float = 0.0,
        trace_path: str | None = None,
        gnss_drift_scale: float = 1.0,
        gnss_drift_m: float | None = None,
        n_objects: int = 5,
        n_views: int | None = None,
        resolution: int = 64,
        feature_dim: int = 32,
        n_grasp_candidates: int = 50,
        n_scenes_train: int = 20,
        n_scenes_test: int = 5,
    ):
        super().__init__(seed=seed, violation_severity=violation_severity, trace_path=trace_path)
        self.gnss_drift_scale = gnss_drift_scale
        self.gnss_drift_m_override = gnss_drift_m
        self.n_objects = n_objects
        self.n_views_override = n_views
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.n_grasp_candidates = n_grasp_candidates
        self.n_scenes_train = n_scenes_train
        self.n_scenes_test = n_scenes_test
        self._trace_stats = self._load_trace_stats()

    @property
    def name(self) -> str:
        return "scene_repr_field_urc"

    @property
    def loop_node(self) -> str:
        return self.LOOP_NODE

    def _load_trace_stats(self) -> dict[str, float]:
        if self.trace_path is None:
            return summarize_scene_repr_traces([])
        path = Path(self.trace_path)
        if not path.is_file():
            raise FileNotFoundError(f"trace file not found: {path}")
        return summarize_scene_repr_traces(load_urc_trace_rows(path))

    def _resolved_gnss_drift_m(self) -> float:
        if self.gnss_drift_m_override is not None:
            return max(0.0, float(self.gnss_drift_m_override))
        return self._trace_stats["mean_gnss_drift_m"]

    def _resolve_views(self) -> tuple[int, float, float]:
        if self.n_views_override is not None:
            n_views = max(4, min(int(self.n_views_override), self.V_MAX))
            gnss_component = min(
                1.0,
                self._resolved_gnss_drift_m() / self.GNSS_DRIFT_REF_M * self.gnss_drift_scale,
            )
            return n_views, 1.0 - n_views / self.V_MAX, gnss_component
        gnss_drift = self._resolved_gnss_drift_m()
        n_views, v_eff = views_from_gnss_drift(
            gnss_drift,
            self.violation_severity,
            v_max=self.V_MAX,
            gnss_drift_ref_m=self.GNSS_DRIFT_REF_M,
            gnss_drift_scale=self.gnss_drift_scale,
        )
        gnss_component = min(
            1.0,
            gnss_drift / self.GNSS_DRIFT_REF_M * self.gnss_drift_scale,
        )
        return n_views, v_eff, gnss_component

    def _perturb_camera_poses(
        self,
        cam_poses: np.ndarray,
        rng: np.random.Generator,
        gnss_drift_m: float,
    ) -> np.ndarray:
        sigma = (gnss_drift_m / self.GNSS_DRIFT_REF_M) * 0.08
        if sigma <= 0.0:
            return cam_poses
        noise = rng.normal(0.0, sigma, size=cam_poses[:, :3, 3].shape)
        perturbed = cam_poses.copy()
        perturbed[:, :3, 3] += noise.astype(np.float32)
        return perturbed

    def _add_depth_noise(
        self,
        depths: np.ndarray,
        rng: np.random.Generator,
        gnss_drift_m: float,
    ) -> np.ndarray:
        sigma = (gnss_drift_m / self.GNSS_DRIFT_REF_M) * 0.05
        if sigma <= 0.0:
            return depths
        noisy = depths + rng.normal(0.0, sigma, size=depths.shape).astype(np.float32)
        return np.clip(noisy, 0.0, None)

    def _compute_grasp_feasibility(
        self,
        candidates: np.ndarray,
        obj_positions: np.ndarray,
        obj_radii: list[float],
    ) -> np.ndarray:
        positions = candidates[:, :3]
        reachable = np.linalg.norm(positions, axis=-1) < 0.5
        not_colliding = np.ones(len(candidates), dtype=bool)
        for pos, radius in zip(obj_positions, obj_radii):
            dist = np.linalg.norm(positions - pos[None, :], axis=-1)
            not_colliding &= dist > radius * 1.2
        near_object = np.zeros(len(candidates), dtype=bool)
        for pos, radius in zip(obj_positions, obj_radii):
            dist = np.linalg.norm(positions - pos[None, :], axis=-1)
            near_object |= dist < radius * 3.0
        return (reachable & not_colliding & near_object).astype(np.float32)

    def _generate_scene(
        self,
        rng: np.random.Generator,
        n_views: int,
        gnss_drift_m: float,
    ) -> dict[str, Any]:
        objects: list[dict[str, Any]] = []
        for _ in range(self.n_objects):
            center = rng.uniform([-0.4, -0.4, 0.0], [0.4, 0.4, 0.4], size=3)
            radius = rng.uniform(0.05, 0.15)
            color = rng.uniform(0.2, 1.0, size=3)
            objects.append({"center": center, "radius": radius, "color": color})
        cam_positions = _fibonacci_hemisphere(n_views) * 2.0
        cam_poses = np.stack([_look_at(pos, np.zeros(3)) for pos in cam_positions])
        cam_poses = self._perturb_camera_poses(cam_poses, rng, gnss_drift_m)
        images = np.zeros((n_views, self.resolution, self.resolution, 3), dtype=np.float32)
        depths = np.zeros((n_views, self.resolution, self.resolution), dtype=np.float32)
        for v_idx in range(n_views):
            images[v_idx], depths[v_idx] = _render_scene(
                cam_poses[v_idx], objects, self.resolution
            )
        depths = self._add_depth_noise(depths, rng, gnss_drift_m)
        object_positions = np.array([o["center"] for o in objects], dtype=np.float32)
        object_features = rng.standard_normal((self.n_objects, self.feature_dim)).astype(np.float32)
        object_features /= np.linalg.norm(object_features, axis=-1, keepdims=True)
        grasp_candidates = np.zeros((self.n_grasp_candidates, 6), dtype=np.float32)
        grasp_candidates[:, :3] = rng.uniform(-0.6, 0.6, (self.n_grasp_candidates, 3))
        grasp_candidates[:, 3:] = rng.uniform(-np.pi, np.pi, (self.n_grasp_candidates, 3))
        grasp_labels = self._compute_grasp_feasibility(
            grasp_candidates, object_positions, [o["radius"] for o in objects]
        )
        return {
            "images": images,
            "depths": depths,
            "camera_poses": cam_poses.astype(np.float32),
            "object_positions": object_positions,
            "object_features": object_features,
            "grasp_candidates": grasp_candidates,
            "grasp_labels": grasp_labels,
        }

    def _generate(self) -> BenchmarkData:
        rng = np.random.default_rng(self.seed)
        n_views, v_eff, gnss_component = self._resolve_views()
        gnss_drift_m = self._resolved_gnss_drift_m()
        train_scenes = [
            self._generate_scene(rng, n_views, gnss_drift_m) for _ in range(self.n_scenes_train)
        ]
        test_scenes = [
            self._generate_scene(rng, n_views, gnss_drift_m) for _ in range(self.n_scenes_test)
        ]

        def stack_scenes(scenes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
            return {key: np.stack([s[key] for s in scenes]) for key in scenes[0]}

        inferred_v = gnss_component if self._trace_stats["row_count"] > 0 else v_eff
        return BenchmarkData(
            train=stack_scenes(train_scenes),
            test=stack_scenes(test_scenes),
            metadata={
                "loop_node": self.loop_node,
                "field_domain": "urc_outdoor",
                "n_objects": self.n_objects,
                "n_views": n_views,
                "resolution": self.resolution,
                "feature_dim": self.feature_dim,
                "n_grasp_candidates": self.n_grasp_candidates,
                "violation_severity": self.violation_severity,
                "effective_violation_severity": v_eff,
                "inferred_violation_severity": inferred_v,
                "gnss_drift_m": gnss_drift_m,
                "gnss_drift_scale": self.gnss_drift_scale,
                "gnss_drift_ref_m": self.GNSS_DRIFT_REF_M,
                "trace_row_count": int(self._trace_stats["row_count"]),
                "trace_mean_violation_severity": self._trace_stats["mean_violation_severity"],
                "v_max": self.V_MAX,
            },
        )
