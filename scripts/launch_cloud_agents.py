#!/usr/bin/env python3
"""Launch 24 Module 12 cloud agents via Cursor Cloud Agents API.

Requires CURSOR_API_KEY from https://cursor.com/dashboard/api

Usage:
  export CURSOR_API_KEY=...
  python scripts/launch_cloud_agents.py          # all 24
  python scripts/launch_cloud_agents.py --from 3  # agents 03-24 only
  python scripts/launch_cloud_agents.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "https://github.com/thatrandomasiandev/field-compounding-validation"
API = "https://api.cursor.com/v1/agents"
STARTING_REF = "main"

AGENTS: list[dict[str, str]] = [
    {"id": "01", "branch": "agent/01-foundation", "title": "Foundation & CI", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 01. Branch agent/01-foundation. CI, exports, tests/test_package_imports.py (≥5). Push."},
    {"id": "02", "branch": "agent/02-utils", "title": "Utils", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 02. Branch agent/02-utils. crossfit.py, trace_index.py, tests/test_utils.py (≥8). Push."},
    {"id": "03", "branch": "agent/03-theory-partial-obs", "title": "Theory partial obs", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 03. Branch agent/03-theory-partial-obs. partial_observability.py, compound_bound.py, tests (≥12). Push."},
    {"id": "04", "branch": "agent/04-theory-coupling", "title": "Theory coupling", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 04. Branch agent/04-theory-coupling. coupling_transfer.py, module11_coupling.json, tests (≥10). Push."},
    {"id": "05", "branch": "agent/05-ingest-urc", "title": "Ingest URC", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 05. Branch agent/05-ingest-urc. urc.py, urc_synthetic.jsonl (≥200), tests (≥10). Push."},
    {"id": "06", "branch": "agent/06-ingest-roomba", "title": "Ingest Roomba", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 06. Branch agent/06-ingest-roomba. roomba.py, traces, tests (≥10). Push."},
    {"id": "07", "branch": "agent/07-ingest-replay", "title": "Ingest replay", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 07. Branch agent/07-ingest-replay. replay.py, schema.py, tests (≥8). Push."},
    {"id": "08", "branch": "agent/08-dgp-scene-repr", "title": "DGP scene_repr", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 08. Branch agent/08-dgp-scene-repr. scene_repr_field_dgp.py, module_03.yaml, tests (≥8). Push."},
    {"id": "09", "branch": "agent/09-dgp-visual-ssl", "title": "DGP visual_ssl", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 09. Branch agent/09-dgp-visual-ssl. visual_ssl_field_dgp.py, module_04.yaml, tests (≥8). Push."},
    {"id": "10", "branch": "agent/10-dgp-sim-to-field", "title": "DGP sim_to_field", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 10. Branch agent/10-dgp-sim-to-field. sim_to_field_dgp.py, module_05.yaml, tests (≥8). Push."},
    {"id": "11", "branch": "agent/11-dgp-visuomotor", "title": "DGP visuomotor", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 11. Branch agent/11-dgp-visuomotor. visuomotor_field_dgp.py, module_06.yaml, tests (≥8). Push."},
    {"id": "12", "branch": "agent/12-dgp-causal", "title": "DGP causal", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 12. Branch agent/12-dgp-causal. causal_field_dgp.py, module_07.yaml, tests (≥8). Push."},
    {"id": "13", "branch": "agent/13-dgp-world-model", "title": "DGP world_model", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 13. Branch agent/13-dgp-world-model. world_model_field_dgp.py, module_08.yaml, tests (≥8). Push."},
    {"id": "14", "branch": "agent/14-dgp-equivariant", "title": "DGP equivariant", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 14. Branch agent/14-dgp-equivariant. equivariant_field_dgp.py, module_09.yaml, tests (≥8). Push."},
    {"id": "15", "branch": "agent/15-dgp-scene-graph", "title": "DGP scene_graph", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 15. Branch agent/15-dgp-scene-graph. scene_graph_field_dgp.py, module_10.yaml, tests (≥8). Push."},
    {"id": "16", "branch": "agent/16-dgp-uncertainty", "title": "DGP uncertainty", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 16. Branch agent/16-dgp-uncertainty. uncertainty_field_dgp.py, module_11.yaml, tests (≥8). Push."},
    {"id": "17", "branch": "agent/17-dgp-neurosymbolic", "title": "DGP neurosymbolic", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 17. Branch agent/17-dgp-neurosymbolic. neurosymbolic_field_dgp.py, module_12.yaml, tests (≥8). Push."},
    {"id": "18", "branch": "agent/18-dgp-fleet-safety", "title": "DGP fleet+safety", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 18. Branch agent/18-dgp-fleet-safety. federated+safety DGPs, module_13/14 yaml, tests (≥12). Push."},
    {"id": "19", "branch": "agent/19-models-estimators", "title": "Violation estimators", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 19. Branch agent/19-models-estimators. violation_estimator.py, calibration.py, tests (≥12). Push."},
    {"id": "20", "branch": "agent/20-models-mitigation", "title": "Mitigation", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 20. Branch agent/20-models-mitigation. condition_d.py, decoupling_stack.py, tests (≥10). Push."},
    {"id": "21", "branch": "agent/21-eval-core", "title": "Eval core", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 21. Branch agent/21-eval-core. runner, metrics, run_benchmark.py, tests (≥10). Push."},
    {"id": "22", "branch": "agent/22-eval-adapters-a", "title": "Adapters 3-8", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 22. Branch agent/22-eval-adapters-a. _eval_module_03-08, tests (≥12). Push."},
    {"id": "23", "branch": "agent/23-eval-adapters-b", "title": "Adapters 9-14", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 23. Branch agent/23-eval-adapters-b. adapters 9-14, compound, validate, tests (≥15). Push."},
    {"id": "24", "branch": "agent/24-paper-observatory", "title": "Paper", "prompt": "Read MODULE12_AGENT_PLAN.md Agent 24. Branch agent/24-paper-observatory. paper12 tex, sections, observatory v2, tests (≥5). Push."},
]


def launch_agent(api_key: str, spec: dict[str, str], dry_run: bool) -> dict:
    body = {
        "name": f"M12 Agent {spec['id']}: {spec['title']}",
        "prompt": {"text": spec["prompt"]},
        "repos": [{"url": REPO, "startingRef": STARTING_REF}],
        "autoCreatePR": False,
    }
    if dry_run:
        return {"agent": spec["id"], "dry_run": True, "branch": spec["branch"]}
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    agent_id = payload.get("agent", {}).get("id") or payload.get("id")
    return {"agent": spec["id"], "cursor_agent_id": agent_id, "branch": spec["branch"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_id", type=int, default=1)
    parser.add_argument("--to", dest="to_id", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    selected = [a for a in AGENTS if args.from_id <= int(a["id"]) <= args.to_id]
    api_key = os.environ.get("CURSOR_API_KEY", "")
    if not args.dry_run and not api_key:
        print("Set CURSOR_API_KEY from https://cursor.com/dashboard/api", file=sys.stderr)
        return 1
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(launch_agent, api_key, s, args.dry_run): s for s in selected}
        for fut in as_completed(futs):
            results.append(fut.result())
            print(json.dumps(results[-1]))
    print("Monitor: https://cursor.com/agents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
