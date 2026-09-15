#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/local-scratch/thesis-artifacts-wiki}"
DEFAULT_VENV_ROOT="${DEFAULT_VENV_ROOT:-$LOCAL_SCRATCH_ROOT/.venvs}"
APP_VENV="${APP_VENV:-$DEFAULT_VENV_ROOT/app}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/.artifacts/results/wiki_ner_http_exact_v1}"

if [[ ! -x "$APP_VENV/bin/python" ]]; then
  echo "App environment not found at $APP_VENV"
  exit 1
fi

"$APP_VENV/bin/python" "$ROOT_DIR/comparison-wrapper/analyze_http_exact_results.py" \
  --results-dir "$RESULTS_DIR" \
  "$@"
