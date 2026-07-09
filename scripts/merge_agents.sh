#!/usr/bin/env bash
# Merge all agent branches into integration, resolving __init__.py conflicts with theirs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BRANCHES=(
  agent/04-theory-coupling
  agent/05-ingest-urc
  agent/06-ingest-roomba
  agent/07-ingest-replay
  agent/08-dgp-scene-repr
  agent/09-dgp-visual-ssl
  agent/10-dgp-sim-to-field
  agent/11-dgp-visuomotor
  agent/12-dgp-causal
  agent/13-dgp-world-model
  agent/14-dgp-equivariant
  agent/15-dgp-scene-graph
  agent/16-dgp-uncertainty
  agent/17-dgp-neurosymbolic
  agent/18-dgp-fleet-safety
  agent/19-models-estimators
  agent/20-models-mitigation
  agent/21-eval-core
  agent/22-eval-adapters-a
  agent/23-eval-adapters-b
  agent/24-paper-observatory
)

resolve_init_conflicts() {
  local incoming="$1"
  for f in $(git diff --name-only --diff-filter=U 2>/dev/null || true); do
    if [[ "$f" == *data/__init__.py* ]] || [[ "$f" == *module11_coupling.json* ]]; then
      git checkout --theirs -- "$f" 2>/dev/null || git checkout "$incoming" -- "$f"
      git add "$f"
    elif [[ "$f" == *__init__.py* ]]; then
      git checkout "$incoming" -- "$f" 2>/dev/null || git checkout --theirs -- "$f"
      git add "$f"
    fi
  done
}

for b in "${BRANCHES[@]}"; do
  echo ">>> merging origin/$b"
  if git merge "origin/$b" -m "Merge $b into integration." --no-edit; then
    continue
  fi
  conflicts=$(git diff --name-only --diff-filter=U || true)
  if [[ -z "$conflicts" ]]; then
    git commit --no-edit || true
    continue
  fi
  echo "  conflicts: $conflicts"
  resolve_init_conflicts "origin/$b"
  remaining=$(git diff --name-only --diff-filter=U || true)
  if [[ -n "$remaining" ]]; then
    echo "  MANUAL NEEDED: $remaining"
    exit 1
  fi
  git commit -m "Merge $b into integration (auto-resolved __init__)." --no-edit
done

echo "All agent branches merged."
