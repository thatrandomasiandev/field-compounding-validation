"""Integration tests for validate_results, compound runner, and fill_paper_values."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from field_compounding.evaluation.compound_field_runner import (
    _extract_scores,
    run_compound_field_experiment,
)
from field_compounding.theory.compound_bound import compound_excess_risk, validate_theorem
from field_compounding.theory.coupling_transfer import apply_field_correction, load_module11_coupling

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _sample_section(module_id: int, score_means: list[float]) -> dict:
    violation_values = [0.0, 0.5, 1.0][: len(score_means)]
    models = {
        "ModelA": {
            str(v): {
                "rmse": {
                    "mean": 1.0 - s,
                    "std": 0.01,
                    "ci_lower": 0.0,
                    "ci_upper": 1.0,
                    "n_successful": 20,
                },
                "normalized_score": {
                    "mean": s,
                    "std": 0.01,
                    "ci_lower": 0.0,
                    "ci_upper": 1.0,
                    "n_successful": 20,
                },
            }
            for v, s in zip(violation_values, score_means, strict=True)
        }
    }
    return {
        "module_id": module_id,
        "module_name": f"Module {module_id}",
        "violation_values": violation_values,
        "violation_severity": violation_values,
        "models": models,
    }


def test_validate_results_passes(tmp_path):
    for mod_id, means in [(9, [0.8, 0.6, 0.4]), (10, [0.75, 0.55, 0.35])]:
        (tmp_path / f"section_{mod_id:02d}.json").write_text(json.dumps(_sample_section(mod_id, means)))
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_results.py"), "--results-dir", str(tmp_path), "--section-ids", "9", "10"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0


def test_validate_results_fails_nan(tmp_path):
    data = _sample_section(12, [0.7, 0.5, 0.3])
    data["models"]["ModelA"]["0.0"]["accuracy"] = {
        "mean": float("nan"),
        "std": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 1.0,
        "n_successful": 20,
    }
    data["models"]["ModelA"]["0.0"]["normalized_score"]["mean"] = float("nan")
    (tmp_path / "section_12.json").write_text(json.dumps(data))
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_results.py"), "--results-dir", str(tmp_path), "--section-ids", "12"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1


def test_validate_results_flat_metric(tmp_path):
    data = _sample_section(14, [0.9, 0.9, 0.9])
    data["models"]["ModelA"]["0.0"]["safety_rate"] = data["models"]["ModelA"]["0.0"]["normalized_score"]
    data["models"]["ModelA"]["0.5"]["safety_rate"] = data["models"]["ModelA"]["0.5"]["normalized_score"]
    data["models"]["ModelA"]["1.0"]["safety_rate"] = data["models"]["ModelA"]["1.0"]["normalized_score"]
    (tmp_path / "section_14.json").write_text(json.dumps(data))
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_results.py"), "--results-dir", str(tmp_path), "--section-ids", "14"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1


def test_fill_paper_values_registry(tmp_path):
    results = tmp_path / "results"
    paper = tmp_path / "paper"
    results.mkdir()
    paper.mkdir()
    (results / "section_09.json").write_text(json.dumps(_sample_section(9, [0.8, 0.6, 0.4])))
    (paper / "stub.tex").write_text(r"\VAL{sec09_ModelA_rmse_v0.0_mean}")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "fill_paper_values.py"),
            "--results-dir",
            str(results),
            "--paper-dir",
            str(paper),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Loaded" in proc.stdout


def test_load_module11_coupling():
    coupling = load_module11_coupling()
    assert coupling.gamma.shape[0] == coupling.gamma.shape[1]
    assert len(coupling.loop_nodes) == coupling.gamma.shape[0]


def test_apply_field_correction():
    coupling = load_module11_coupling()
    densities = {node: (1.0 if node == "scene_repr" else 0.1) for node in coupling.loop_nodes}
    corrected = apply_field_correction(coupling.gamma, coupling.loop_nodes, densities)
    assert corrected.max() >= coupling.gamma.max() * 0.0
    assert np.allclose(np.diag(corrected), 0.0)


def test_compound_excess_risk():
    v = np.array([0.2, 0.4, 0.6])
    psi = np.array([0.5, 0.5, 0.5])
    gamma = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, 0.15], [0.2, 0.15, 0.0]])
    bound = compound_excess_risk(v, psi, gamma)
    assert bound > float(np.dot(v, psi))


def test_validate_theorem():
    result = validate_theorem(observed_compound_loss=0.5, predicted_lower_bound=0.3)
    assert result["theorem_holds"] == 1.0
    assert result["slack"] == pytest.approx(0.2)


def test_extract_scores():
    data = _sample_section(9, [0.8, 0.6, 0.4])
    scores = _extract_scores(data)
    assert scores == pytest.approx([0.8, 0.6, 0.4])


def test_run_compound_field_experiment(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for mod_id, means in [(9, [0.8, 0.6, 0.4]), (10, [0.75, 0.55, 0.35])]:
        (results_dir / f"section_{mod_id:02d}.json").write_text(json.dumps(_sample_section(mod_id, means)))

    out = tmp_path / "section_15.json"
    config = REPO_ROOT / "configs" / "compound_field_experiment.yaml"
    assert config.is_file()

    result = run_compound_field_experiment(
        config_path=config,
        results_dir=results_dir,
        output_path=out,
        fast=True,
    )
    assert "conditions" in result
    assert result["fast_mode"] is True
    assert out.is_file()


def test_run_compound_script(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "section_09.json").write_text(json.dumps(_sample_section(9, [0.8, 0.6, 0.4])))
    (results_dir / "section_10.json").write_text(json.dumps(_sample_section(10, [0.75, 0.55, 0.35])))
    out = tmp_path / "section_15.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_compound_field_experiment.py"),
            "--results-dir",
            str(results_dir),
            "--output",
            str(out),
            "--fast",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert out.is_file()
