#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/local-scratch/thesis-artifacts-wiki}"
DATS_RUNTIME_ROOT="${DATS_RUNTIME_ROOT:-$LOCAL_SCRATCH_ROOT/runtime}"
PID_FILE="${PID_FILE:-$DATS_RUNTIME_ROOT/vllm.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No vLLM pid file found."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 30); do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi

rm -f "$PID_FILE"
echo "Stopped vLLM server pid $PID"
