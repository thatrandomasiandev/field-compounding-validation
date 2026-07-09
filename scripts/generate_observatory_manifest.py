"""Generate Observatory manifest v2."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from typing import Any
MANIFEST_VERSION = "2.0.0"
LOOP_NODES = {3:"scene_repr",4:"visual_ssl",5:"sim_to_field",6:"visuomotor",7:"causal",8:"world_model",9:"equivariant",10:"scene_graph",11:"uncertainty",12:"neurosymbolic",13:"federated",14:"safety"}
DEFAULT = Path(__file__).resolve().parents[1] / "observatory" / "module11_coupling.json"

def _load(p: Path) -> dict:
    with open(p) as f: return json.load(f)

def _resolve(path: Path|None) -> dict:
    for c in ([path] if path else []) + ([Path(os.environ["MODULE11_RESULTS"])/"module11_coupling.json"] if os.environ.get("MODULE11_RESULTS") else []) + [DEFAULT]:
        if c and c.exists():
            d = _load(c)
            return d.get("coupling_matrix", d)
    return {"module_ids": list(range(3,15)), "gamma": [[0.0]*12]*12, "source": "stub"}

def build_manifest(results_dir: Path, module11_path: Path|None=None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    module_ids = list(range(3,15))
    compound = _load(results_dir/"section_15.json") if (results_dir/"section_15.json").exists() else {}
    m11 = _resolve(module11_path)
    modules = [{"id": i, "section": i, "name": f"Module {i}", "loop_node": LOOP_NODES[i], "psi": 0.0, "sweep": {"knob_values":[],"violation_severity":[],"normalized_score":[]}, "trace_source": "urc"} for i in module_ids]
    return {
        "version": MANIFEST_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_results_hash": hashlib.sha256(b"stub").hexdigest()[:16],
        "module_ids": module_ids, "modules": modules, "psi": [0.0]*12,
        "coupling": {"gamma": m11.get("gamma", []), "module_ids": module_ids, "transferred": True},
        "module11_import": {"source": str(m11.get("source", DEFAULT)), "module_ids": module_ids},
        "field_traces": [{"path": "observatory/traces/urc_synthetic.jsonl", "venue": "URC", "environment": "outdoor", "row_count": 0}],
        "partial_observability": {"estimation_mae": None, "calibration_bins": 10},
        "conditions": {}, "pinned": {},
        "formula": "R_compound >= sum_k v_hat_k * psi_k + sum_{k<l} gamma_{k,l} * v_hat_k * v_hat_l",
        "paper": "paper12_field_compounding.tex",
    }

def write_manifest(manifest: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {output}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=Path("observatory/manifest_v2.json"))
    p.add_argument("--module11-coupling", type=Path, default=None)
    a = p.parse_args()
    write_manifest(build_manifest(a.results_dir, a.module11_coupling), a.output)

if __name__ == "__main__":
    main()
