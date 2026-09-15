#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/local-scratch/thesis-artifacts-wiki}"
DEFAULT_VENV_ROOT="${DEFAULT_VENV_ROOT:-$LOCAL_SCRATCH_ROOT/.venvs}"
APP_VENV="${APP_VENV:-$DEFAULT_VENV_ROOT/app}"
MATRIX_LOG_FILE="${MATRIX_LOG_FILE:-$ROOT_DIR/.runtime/wiki_model_matrix.log}"
export LOCAL_SCRATCH_ROOT DEFAULT_VENV_ROOT APP_VENV
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LOCAL_SCRATCH_ROOT/cache}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TMPDIR="${TMPDIR:-$LOCAL_SCRATCH_ROOT/tmp}"

mkdir -p "$DEFAULT_VENV_ROOT" "$TORCH_HOME" "$TMPDIR" "$(dirname "$MATRIX_LOG_FILE")"
exec > >(tee -a "$MATRIX_LOG_FILE") 2>&1

echo "Wiki model matrix log file: $MATRIX_LOG_FILE"
echo "Started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ ! -x "$APP_VENV/bin/python" ]]; then
  echo "App environment not found at $APP_VENV"
  echo "Run bash $ROOT_DIR/install_colab_requirements.sh first."
  exit 1
fi

"$APP_VENV/bin/python" "$ROOT_DIR/comparison-wrapper/run_http_exact_matrix.py" \
  --config "$ROOT_DIR/comparison-wrapper/wiki_http_exact_matrix.json" \
  --python "$APP_VENV/bin/python" \
  "$@"
