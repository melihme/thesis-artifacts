#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/local-scratch/thesis-artifacts-wiki}"
DEFAULT_VENV_ROOT="${DEFAULT_VENV_ROOT:-$LOCAL_SCRATCH_ROOT/.venvs}"
APP_VENV="${APP_VENV:-$DEFAULT_VENV_ROOT/app}"
PYTHON_BIN="${PYTHON_BIN:-$APP_VENV/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "App environment not found at $APP_VENV"
  echo "Run bash $ROOT_DIR/install_colab_requirements.sh first."
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/comparison-wrapper/run_http_exact_matrix.py" \
  --config "$ROOT_DIR/comparison-wrapper/wiki_http_exact_matrix.json" \
  --preflight-only \
  "$@"
