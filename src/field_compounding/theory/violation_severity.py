"""Violation severity mapping v_k for each module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ViolationSpec:
    """Specification of what violation_severity means for a module."""

    module_id: int
    module_name: str
    knob_name: str
    knob_range: tuple[float, float]
    description: str
    mapping: Callable[[float], float]  # knob_value -> v_k in [0, 1]


VIOLATION_SPECS: dict[int, ViolationSpec] = {
    3: ViolationSpec(
        module_id=3,
        module_name="3D Scene Understanding",
        knob_name="n_views",
        knob_range=(4, 32),
        description="Fewer camera views degrade multi-view reconstruction",
        mapping=lambda n_views: 1.0 - n_views / 32.0,
    ),
    4: ViolationSpec(
        module_id=4,
        module_name="Visual Self-Supervised Learning",
        knob_name="n_labeled",
        knob_range=(0, 3000),
        description="Fewer downstream labels degrade linear probe quality",
        mapping=lambda n_labeled: 1.0 - n_labeled / 3000.0,
    ),
    5: ViolationSpec(
        module_id=5,
        module_name="Sim-to-Field Adaptation",
        knob_name="domain_shift_delta",
        knob_range=(0, 2.0),
        description="Larger sim-to-real gap degrades transfer performance",
        mapping=lambda delta: delta / 2.0,
    ),
    6: ViolationSpec(
        module_id=6,
        module_name="Visuomotor Control",
        knob_name="coverage_beta",
        knob_range=(0.3, 1.0),
        description="Lower offline data coverage degrades policy learning",
        mapping=lambda beta: 1.0 - beta,
    ),
    7: ViolationSpec(
        module_id=7,
        module_name="Causal Visual Understanding",
        knob_name="confounding_gamma",
        knob_range=(0, 1.0),
        description="Stronger confounding biases causal effect estimates",
        mapping=lambda gamma: gamma,
    ),
    8: ViolationSpec(
        module_id=8,
        module_name="Visual World Models",
        knob_name="model_noise_sigma",
        knob_range=(0.01, 0.5),
        description="Higher dynamics noise degrades model prediction accuracy",
        mapping=lambda sigma: sigma / 0.5,
    ),
    9: ViolationSpec(
        module_id=9,
        module_name="Equivariant Perception",
        knob_name="symmetry_violation_frac",
        knob_range=(0, 1.0),
        description="More symmetry-broken data degrades equivariant models",
        mapping=lambda frac: frac,
    ),
    10: ViolationSpec(
        module_id=10,
        module_name="Scene Graph Reasoning",
        knob_name="wl_collision_rho",
        knob_range=(0, 0.5),
        description="Higher 1-WL collision rate limits GNN expressiveness",
        mapping=lambda rho: rho / 0.5,
    ),
    11: ViolationSpec(
        module_id=11,
        module_name="Uncertainty Quantification",
        knob_name="label_noise_p",
        knob_range=(0, 0.3),
        description="More label noise degrades calibration and coverage",
        mapping=lambda p: p / 0.3,
    ),
    12: ViolationSpec(
        module_id=12,
        module_name="Neurosymbolic Planning",
        knob_name="grounding_noise_frac",
        knob_range=(0, 1.0),
        description="Noisier symbol grounding degrades reasoning accuracy",
        mapping=lambda frac: frac,
    ),
    13: ViolationSpec(
        module_id=13,
        module_name="Federated Learning",
        knob_name="dirichlet_alpha",
        knob_range=(0.1, 5.0),
        description="Lower alpha = more data heterogeneity across robots",
        mapping=lambda alpha: float(np.clip((5.0 - alpha) / (5.0 - 0.1), 0.0, 1.0)),
    ),
    14: ViolationSpec(
        module_id=14,
        module_name="Safety and Risk",
        knob_name="model_error",
        knob_range=(0, 1.0),
        description="Larger model error degrades safety guarantees",
        mapping=lambda err: err,
    ),
}


def get_violation_severity(module_id: int, knob_value: float) -> float:
    """Map a module's native knob value to violation severity v_k in [0,1]."""
    spec = VIOLATION_SPECS[module_id]
    v = spec.mapping(knob_value)
    return float(np.clip(v, 0.0, 1.0))


def sweep_violations(
    module_id: int, n_points: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a sweep of knob values and corresponding violation severities.

    Returns:
        (knob_values, violation_severities) each of shape (n_points,)
    """
    spec = VIOLATION_SPECS[module_id]
    knob_values = np.linspace(spec.knob_range[0], spec.knob_range[1], n_points)
    severities = np.array([spec.mapping(k) for k in knob_values])
    severities = np.clip(severities, 0.0, 1.0)
    return knob_values, severities
