from __future__ import annotations
import importlib.util, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "paper12_field_compounding.tex"
SECTIONS = ROOT / "paper" / "sections"
SCRIPT = ROOT / "scripts" / "generate_observatory_manifest.py"
SCHEMA = ROOT / "observatory" / "manifest_v2.schema.json"

def _inputs():
    return re.findall(r"\\input\{([^}]+)\}", MAIN.read_text())

def test_main_tex_exists():
    assert MAIN.exists()

def test_all_input_paths_exist():
    missing = []
    for rel in _inputs():
        p = ROOT / "paper" / rel
        if p.suffix != ".tex": p = p.with_suffix(".tex")
        if not p.exists(): missing.append(rel)
    assert not missing

def test_section_files_sec03_through_sec15_exist():
    for i in range(3, 16):
        assert (SECTIONS / f"sec{i:02d}.tex").exists()

def test_intro_conclusion_appendix_exist():
    for n in ["abstract.tex","sec01_introduction.tex","sec02_theory.tex","sec16_related_work.tex","sec17_conclusion.tex","appendices.tex"]:
        assert (SECTIONS / n).exists()

def test_generate_observatory_manifest_runs():
    out = ROOT / "observatory" / "_test_manifest_v2.json"
    try:
        r = subprocess.run([sys.executable, str(SCRIPT), "--output", str(out)], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        d = json.loads(out.read_text())
        assert d["version"] == "2.0.0"
        assert "field_traces" in d
    finally:
        out.unlink(missing_ok=True)

def test_manifest_v2_schema_defines_required_keys():
    req = set(json.loads(SCHEMA.read_text())["required"])
    assert {"version","field_traces","module11_import","partial_observability","modules","coupling"}.issubset(req)

def test_manifest_builder_importable():
    spec = importlib.util.spec_from_file_location("gm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.build_manifest(ROOT / "results")
    assert m["paper"] == "paper12_field_compounding.tex"
