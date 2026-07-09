#!/usr/bin/env python3
"""Run Module 12 benchmarks on a Colab VM (for `colab exec -f` / `colab run`)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FIELD_REPO = "https://github.com/thatrandomasiandev/field-compounding-validation.git"
M11_REPO = "https://github.com/thatrandomasiandev/visual-robot-learning.git"
WORK = Path("/content")
FIELD_ROOT = WORK / "field-compounding-validation"
M11_ROOT = WORK / "visual-robot-learning"
MODULES = [f"{i:02d}" for i in range(3, 15)]


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, cwd=cwd, env=merged, check=True)


def git_clone_or_pull(dest: Path, url: str, *, branch: str | None = None) -> None:
    if dest.is_dir() and (dest / ".git").is_dir():
        run(["git", "fetch", "origin"], cwd=dest)
        if branch:
            run(["git", "checkout", branch], cwd=dest)
            run(["git", "pull", "origin", branch], cwd=dest)
        else:
            run(["git", "pull"], cwd=dest)
        return
    clone_cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    if branch:
        clone_cmd = ["git", "clone", "--branch", branch, "--single-branch", "--depth", "1", url, str(dest)]
    run(clone_cmd)


def setup_repos(field_branch: str) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    git_clone_or_pull(FIELD_ROOT, FIELD_REPO, branch=field_branch)
    git_clone_or_pull(M11_ROOT, M11_REPO)


def install_packages() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(M11_ROOT)])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{FIELD_ROOT}[dev]"])


def run_benchmarks(
    modules: list[str],
    *,
    n_trials: int,
    fast: bool,
    force: bool,
) -> None:
    env = {"FIELD_COMPOUNDING_DEVICE": os.environ.get("FIELD_COMPOUNDING_DEVICE", "cuda")}
    if fast:
        env["FIELD_COMPOUNDING_FAST"] = "1"
    elif "FIELD_COMPOUNDING_FAST" in env:
        del env["FIELD_COMPOUNDING_FAST"]

    for mod in modules:
        print(f"\n=== benchmark module {mod} ===", flush=True)
        cmd = [
            sys.executable,
            "scripts/run_benchmark.py",
            "--module",
            mod,
            "--n-trials",
            str(n_trials),
        ]
        if fast:
            cmd.append("--fast")
        out = FIELD_ROOT / "results" / f"section_{mod}.json"
        if out.exists() and not force:
            print(f"skip module {mod} ({out} exists)", flush=True)
            continue
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=FIELD_ROOT, env={**os.environ, **env}, check=True)


def run_compound(*, n_trials: int, fast: bool) -> None:
    cmd = [sys.executable, "scripts/run_compound_field_experiment.py", "--n-trials", str(n_trials)]
    if fast:
        cmd.append("--fast")
    run(cmd, cwd=FIELD_ROOT)


def run_validate(*, min_trials: int) -> None:
    run(
        [sys.executable, "scripts/validate_results.py", "--min-trials", str(min_trials)],
        cwd=FIELD_ROOT,
    )


def list_results() -> None:
    results = sorted((FIELD_ROOT / "results").glob("section_*.json"))
    print(f"\n{len(results)} result file(s) in {FIELD_ROOT / 'results'}:")
    for path in results:
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 12 Colab remote benchmark runner")
    parser.add_argument("--field-branch", default="integration")
    parser.add_argument("--skip-setup", action="store_true", help="Skip clone/pull and pip install")
    parser.add_argument("--modules", default="all", help="Comma list (03,08) or 'all'")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run modules even if JSON exists")
    parser.add_argument("--compound", action="store_true", help="Run section 15 compound experiment")
    parser.add_argument("--validate", action="store_true", help="Run validate_results.py")
    parser.add_argument("--min-trials", type=int, default=20, help="For --validate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = MODULES if args.modules.lower() == "all" else [m.strip().zfill(2) for m in args.modules.split(",")]

    if not args.skip_setup:
        print("=== setup repos ===", flush=True)
        setup_repos(args.field_branch)
        print("=== install packages ===", flush=True)
        install_packages()

    os.environ.setdefault("FIELD_COMPOUNDING_DEVICE", "cuda")

    print("=== benchmarks ===", flush=True)
    run_benchmarks(modules, n_trials=args.n_trials, fast=args.fast, force=args.force)

    if args.compound:
        print("\n=== compound experiment ===", flush=True)
        run_compound(n_trials=args.n_trials, fast=args.fast)

    if args.validate:
        print("\n=== validate ===", flush=True)
        run_validate(min_trials=args.min_trials)

    list_results()
    print("\nDone. Download results with:")
    print(f"  colab download -s <SESSION> {FIELD_ROOT}/results ./results")


if __name__ == "__main__":
    main()
