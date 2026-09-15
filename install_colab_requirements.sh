#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/local-scratch/thesis-artifacts-wiki}"
DEFAULT_VENV_ROOT="${DEFAULT_VENV_ROOT:-$LOCAL_SCRATCH_ROOT/.venvs}"

APP_VENV="${APP_VENV:-$DEFAULT_VENV_ROOT/app}"
VLLM_VENV="${VLLM_VENV:-$DEFAULT_VENV_ROOT/vllm}"
APP_REQUIREMENTS_FILE="${APP_REQUIREMENTS_FILE:-$ROOT_DIR/requirements-lock.txt}"
VLLM_REQUIREMENTS_FILE="${VLLM_REQUIREMENTS_FILE:-$ROOT_DIR/requirements-vllm-lock.txt}"
export LOCAL_SCRATCH_ROOT DEFAULT_VENV_ROOT APP_VENV VLLM_VENV
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$LOCAL_SCRATCH_ROOT/cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LOCAL_SCRATCH_ROOT/cache}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TMPDIR="${TMPDIR:-$LOCAL_SCRATCH_ROOT/tmp}"
mkdir -p "$DEFAULT_VENV_ROOT" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TMPDIR"

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$("$ROOT_DIR/ensure_python310.sh")"
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Expected Python 3.10, got {sys.version.split()[0]}")
PY

bootstrap_venv() {
  local venv_path="$1"
  local requirements_path="$2"

  mkdir -p "$(dirname "$venv_path")"
  "$PYTHON_BIN" -m venv "$venv_path"
  # shellcheck disable=SC1090
  source "$venv_path/bin/activate"
  python -m pip install --upgrade pip==24.2 setuptools==70.3.0 wheel==0.43.0
  python -m pip install -r "$requirements_path"
  python -m pip check
  deactivate
}

bootstrap_venv "$APP_VENV" "$APP_REQUIREMENTS_FILE"
bootstrap_venv "$VLLM_VENV" "$VLLM_REQUIREMENTS_FILE"

echo "Installed app environment at $APP_VENV"
echo "Installed vLLM environment at $VLLM_VENV"
echo "Python executable: $PYTHON_BIN"
