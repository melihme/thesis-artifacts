#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '%s\n' "$*" >&2
}

require_python310() {
  local candidate="$1"
  if [[ ! -x "$candidate" ]]; then
    return 1
  fi
  "$candidate" - <<'PY' >/dev/null
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
}

if ! command -v python3.10 >/dev/null 2>&1 || ! require_python310 "$(command -v python3.10)"; then
  log "python3.10 is not available in this Colab runtime."
  log "Install or enable python3.10 first, then rerun this script."
  exit 1
fi

PYTHON310="$(command -v python3.10)"

log "Installing Python 3.10 venv support for this Colab runtime."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y >/dev/null
apt-get install -y python3.10-venv >/dev/null

log "Setting python3 default to $PYTHON310."
update-alternatives --install /usr/bin/python3 python3 "$PYTHON310" 100 >/dev/null
update-alternatives --set python3 "$PYTHON310" >/dev/null

python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY

python3 -m venv --help >/dev/null
printf '%s\n' "$(command -v python3)"
