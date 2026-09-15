#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/local-scratch/thesis-artifacts-wiki}"
DEFAULT_VENV_ROOT="${DEFAULT_VENV_ROOT:-$LOCAL_SCRATCH_ROOT/.venvs}"
DATS_RUNTIME_ROOT="${DATS_RUNTIME_ROOT:-$LOCAL_SCRATCH_ROOT/runtime}"

VLLM_VENV="${VLLM_VENV:-$DEFAULT_VENV_ROOT/vllm}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
# FlashInfer's sampler JIT is incompatible with the CUDA compiler currently
# exposed by Colab. The native vLLM sampler avoids that optional JIT path.
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
REVISION="${REVISION:-}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"
PID_FILE="${PID_FILE:-$DATS_RUNTIME_ROOT/vllm.pid}"
LOG_FILE="${LOG_FILE:-$DATS_RUNTIME_ROOT/vllm.server.log}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LOCAL_SCRATCH_ROOT/cache}"
TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
TMPDIR="${TMPDIR:-$LOCAL_SCRATCH_ROOT/tmp}"

mkdir -p "$DEFAULT_VENV_ROOT" "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

if [[ ! -x "$VLLM_VENV/bin/vllm" ]]; then
  echo "vLLM binary not found at $VLLM_VENV/bin/vllm"
  echo "Run bash $ROOT_DIR/install_colab_requirements.sh first."
  exit 1
fi

if [[ "$VLLM_USE_FLASHINFER_SAMPLER" != "0" && "$VLLM_USE_FLASHINFER_SAMPLER" != "1" ]]; then
  echo "VLLM_USE_FLASHINFER_SAMPLER must be 0 or 1, got: $VLLM_USE_FLASHINFER_SAMPLER"
  exit 1
fi

models_response() {
  curl -fsS "http://$HOST:$PORT/v1/models" 2>/dev/null || true
}

served_model_is_available() {
  local response
  response="$(models_response)"
  [[ -n "$response" ]] || return 1
  RESPONSE="$response" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" "$VLLM_VENV/bin/python" - <<'PY'
import json
import os
import sys

try:
    payload = json.loads(os.environ["RESPONSE"])
except json.JSONDecodeError:
    sys.exit(1)

expected = os.environ["SERVED_MODEL_NAME"]
model_ids = {str(item.get("id", "")) for item in payload.get("data", [])}
sys.exit(0 if expected in model_ids else 1)
PY
}

describe_served_models() {
  local response
  response="$(models_response)"
  if [[ -z "$response" ]]; then
    echo "<no /v1/models response>"
    return
  fi
  RESPONSE="$response" "$VLLM_VENV/bin/python" - <<'PY'
import json
import os

try:
    payload = json.loads(os.environ["RESPONSE"])
except json.JSONDecodeError:
    print(os.environ["RESPONSE"])
else:
    ids = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
    print(", ".join(ids) if ids else "<no model ids>")
PY
}

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    if served_model_is_available; then
      echo "vLLM already running with pid $OLD_PID and serving $SERVED_MODEL_NAME"
      exit 0
    fi
    echo "vLLM pid file points to live pid $OLD_PID, but http://$HOST:$PORT/v1/models does not list $SERVED_MODEL_NAME."
    echo "Currently served model ids: $(describe_served_models)"
    echo "Stop the stale server with: PID_FILE=$PID_FILE $ROOT_DIR/stop_vllm_server.sh"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1 || [[ -n "$(models_response)" ]]; then
  if served_model_is_available; then
    echo "vLLM is already serving $SERVED_MODEL_NAME on http://$HOST:$PORT"
    exit 0
  fi
  echo "A vLLM-compatible server is already listening on http://$HOST:$PORT, but it is not serving $SERVED_MODEL_NAME."
  echo "Currently served model ids: $(describe_served_models)"
  echo "Stop that server before starting this model."
  exit 1
fi

export HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME
export TORCH_HOME
export TMPDIR
export LOCAL_SCRATCH_ROOT DEFAULT_VENV_ROOT DATS_RUNTIME_ROOT VLLM_VENV
export PATH="$VLLM_VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER

CMD=(
  "$VLLM_VENV/bin/vllm"
  serve
  "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ -n "$REVISION" ]]; then
  CMD+=(--revision "$REVISION")
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  CMD+=(--enable-prefix-caching)
else
  CMD+=(--no-enable-prefix-caching)
fi

echo "Starting vLLM server:"
printf '  %q' "${CMD[@]}"
printf '\n'
echo "vLLM log file: $LOG_FILE"
echo "vLLM version: $("$VLLM_VENV/bin/python" -c 'from importlib.metadata import version; print(version("vllm"))')"
echo "Prefix caching requested: $ENABLE_PREFIX_CACHING"
echo "FlashInfer sampler enabled: $VLLM_USE_FLASHINFER_SAMPLER"
echo "Model revision requested: ${REVISION:-<server default>}"

nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
SERVER_PID="$!"
echo "$SERVER_PID" >"$PID_FILE"

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if served_model_is_available; then
    echo "vLLM server is ready on http://$HOST:$PORT and serving $SERVED_MODEL_NAME"
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "vLLM server exited before becoming ready."
    tail -n 80 "$LOG_FILE" || true
    exit 1
  fi
  sleep 2
done

echo "vLLM server did not become ready within ${STARTUP_TIMEOUT_SECONDS}s."
echo "Currently served model ids: $(describe_served_models)"
tail -n 80 "$LOG_FILE" || true
exit 1
