#!/usr/bin/env bash
# Drive Module 12 benchmarks on Colab GPU from your local terminal.
#
# Prereq: uv tool install git+https://github.com/googlecolab/google-colab-cli
#
# Examples:
#   ./scripts/colab_cli.sh smoke              # T4, 1 trial, --fast, modules 03-14
#   ./scripts/colab_cli.sh full               # T4, 20 trials, module-by-module
#   ./scripts/colab_cli.sh module 08          # single module
#   COLAB_GPU=A100 ./scripts/colab_cli.sh smoke
#   ./scripts/colab_cli.sh stop
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${COLAB_SESSION:-m12}"
GPU="${COLAB_GPU:-T4}"
REMOTE_RESULTS="/content/field-compounding-validation/results"
LOCAL_RESULTS="${ROOT}/results"

if ! command -v colab >/dev/null 2>&1; then
  echo "colab CLI not found. Install with:"
  echo "  uv tool install git+https://github.com/googlecolab/google-colab-cli"
  exit 1
fi

ensure_session() {
  if colab status -s "$SESSION" >/dev/null 2>&1; then
    echo "Using existing session: $SESSION"
    colab status -s "$SESSION"
  else
    echo "Starting new session: $SESSION (GPU=$GPU)"
    colab new -s "$SESSION" --gpu "$GPU"
  fi
}

download_results() {
  mkdir -p "$LOCAL_RESULTS"
  echo "Downloading $REMOTE_RESULTS -> $LOCAL_RESULTS"
  colab download -s "$SESSION" "$REMOTE_RESULTS" "$LOCAL_RESULTS"
}

run_remote() {
  local extra_args=("$@")
  colab exec -s "$SESSION" -f "$ROOT/scripts/colab_remote_run.py" -- "${extra_args[@]}"
}

case "${1:-smoke}" in
  smoke)
    ensure_session
    run_remote --fast --n-trials 1 --modules all
    download_results
  ;;
  full)
    ensure_session
    run_remote --n-trials 20 --modules all --compound --validate --min-trials 20
    download_results
  ;;
  module)
  shift
    mod="${1:?usage: colab_cli.sh module 08}"
    ensure_session
    run_remote --n-trials "${N_TRIALS:-20}" --modules "$(printf '%02d' "$mod")" ${FAST:+--fast}
    download_results
  ;;
  compound)
    ensure_session
    run_remote --skip-setup --compound --n-trials "${N_TRIALS:-20}" ${FAST:+--fast}
    download_results
  ;;
  validate)
    ensure_session
    run_remote --skip-setup --validate --min-trials "${MIN_TRIALS:-20}"
  ;;
  download)
    download_results
  ;;
  status)
    colab status -s "$SESSION"
  ;;
  url)
    colab url -s "$SESSION" --open
  ;;
  stop)
    colab stop -s "$SESSION"
  ;;
  one-shot)
    shift
    colab run --gpu "$GPU" "$ROOT/scripts/colab_remote_run.py" -- "$@"
  ;;
  *)
    cat <<EOF
Usage: ./scripts/colab_cli.sh <command>

Commands:
  smoke          Fast smoke (default): --fast --n-trials 1, all modules
  full           Production: 20 trials, compound + validate, download results
  module NN      Run one module (e.g. module 08). Set N_TRIALS, FAST=1 optional
  compound       Section 15 only (repos must already be on VM)
  validate       validate_results.py on remote
  download       Pull results/ from active session
  status         Show session hardware/status
  url            Open session in browser
  stop           Terminate session VM
  one-shot [args]  colab run --gpu \$GPU colab_remote_run.py [args]

Env:
  COLAB_SESSION  Session name (default: m12)
  COLAB_GPU      T4 | L4 | A100 | H100 (default: T4)
  N_TRIALS       For 'module' / 'compound' (default: 20)
  FAST=1         Pass --fast to module/compound
EOF
    exit 1
    ;;
esac
