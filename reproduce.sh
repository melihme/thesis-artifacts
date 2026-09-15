#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-help}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' \
    "Usage: bash reproduce.sh <stage> [stage arguments]" \
    "" \
    "Local inputs are written only under .artifacts, which is ignored." \
    "" \
    "Stages:" \
    "  validate-domain-ids      Validate the domain-shift Twitter IDs." \
    "  validate-wiki-source     Verify WIKI_SOURCE_DIR by SHA-256." \
    "  validate-domain-sources  Verify both domain-shift source snapshots." \
    "  prepare-wiki       Reconstruct WikiNER from its official train/dev/test files." \
    "  prepare-domain     Reconstruct both sources used by the domain-shift experiment." \
    "  train-wiki         Train both Wiki encoders for seed 42." \
    "  train-domain       Train both encoders on both domains for seeds 42 through 46." \
    "  wiki-preflight     Validate the Wiki protocol, inputs, models, and 81-row matrix." \
    "  wiki-matrix        Run the complete Wiki HTTP comparison." \
    "  wiki-analysis      Compute Wiki bootstrap and latency analyses." \
    "  domain-shift       Run and aggregate all 70 domain-shift evaluations." \
    "  capture-env        Save the local software and hardware record under .artifacts." \
    "  verify-reference   Check the aggregate reference results." \
    "  smoke              Check code, configuration, reference results, and split IDs." \
    "  all                Run both complete experiments after setup and source acquisition." \
    "" \
    "Set WIKI_SOURCE_DIR for the Wiki experiment. Set both WIKI_SOURCE_DIR and" \
    "TWITTER_SOURCE_DIR for the domain-shift experiment."
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required for this stage." >&2
    exit 2
  fi
}

validate_domain_ids() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/validate_domain_shift_ids.py"
}

validate_wiki_source() {
  require_var WIKI_SOURCE_DIR
  "$PYTHON_BIN" "$ROOT_DIR/scripts/validate_source_data.py" \
    --wiki-source "$WIKI_SOURCE_DIR" \
    "$@"
}

validate_domain_sources() {
  require_var WIKI_SOURCE_DIR
  require_var TWITTER_SOURCE_DIR
  "$PYTHON_BIN" "$ROOT_DIR/scripts/validate_source_data.py" \
    --wiki-source "$WIKI_SOURCE_DIR" \
    --twitter-source "$TWITTER_SOURCE_DIR" \
    "$@"
}

prepare_wiki() {
  require_var WIKI_SOURCE_DIR
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_encoder_matrix.py" \
    --wiki-source "$WIKI_SOURCE_DIR" \
    --datasets wiki_ner \
    --prepare-only \
    "$@"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/verify_materialized_counts.py" wiki_ner
}

prepare_domain() {
  require_var WIKI_SOURCE_DIR
  require_var TWITTER_SOURCE_DIR
  validate_domain_ids
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_encoder_matrix.py" \
    --wiki-source "$WIKI_SOURCE_DIR" \
    --twitter-source "$TWITTER_SOURCE_DIR" \
    --datasets wiki_ner twitter_ner \
    --prepare-only \
    "$@"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/verify_materialized_counts.py" wiki_ner twitter_ner
}

train_wiki() {
  require_var WIKI_SOURCE_DIR
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_encoder_matrix.py" \
    --wiki-source "$WIKI_SOURCE_DIR" \
    --datasets wiki_ner \
    --seeds 42 \
    "$@"
}

train_domain() {
  require_var WIKI_SOURCE_DIR
  require_var TWITTER_SOURCE_DIR
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_encoder_matrix.py" \
    --wiki-source "$WIKI_SOURCE_DIR" \
    --twitter-source "$TWITTER_SOURCE_DIR" \
    --datasets wiki_ner twitter_ner \
    "$@"
}

run_domain_shift() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_domain_shift.py" "$@"
  "$PYTHON_BIN" "$ROOT_DIR/analysis/summarize_domain_shift.py"
}

smoke() {
  export PYTHONDONTWRITEBYTECODE=1
  validate_domain_ids
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m unittest discover -s "$ROOT_DIR/tests" -v
  "$PYTHON_BIN" -c 'import json, pathlib, sys; root=pathlib.Path(sys.argv[1]); [json.loads(path.read_text(encoding="utf-8")) for path in root.rglob("*.json")]' "$ROOT_DIR"
  for path in "$ROOT_DIR"/*.sh; do
    bash -n "$path"
  done
  "$PYTHON_BIN" "$ROOT_DIR/scripts/verify_reference_results.py"
  "$PYTHON_BIN" "$ROOT_DIR/analysis/verify_thesis_metrics.py"
}

case "$STAGE" in
  validate-domain-ids)
    validate_domain_ids
    ;;
  validate-wiki-source)
    validate_wiki_source "$@"
    ;;
  validate-domain-sources)
    validate_domain_sources "$@"
    ;;
  prepare-wiki)
    prepare_wiki "$@"
    ;;
  prepare-domain)
    prepare_domain "$@"
    ;;
  train-wiki)
    train_wiki "$@"
    ;;
  train-domain)
    train_domain "$@"
    ;;
  wiki-preflight)
    bash "$ROOT_DIR/run_wiki_preflight.sh" "$@"
    ;;
  wiki-matrix)
    bash "$ROOT_DIR/run_wiki_model_matrix.sh" "$@"
    ;;
  wiki-analysis)
    bash "$ROOT_DIR/run_wiki_analysis.sh" "$@"
    ;;
  domain-shift)
    run_domain_shift "$@"
    ;;
  capture-env)
    "$PYTHON_BIN" "$ROOT_DIR/scripts/capture_environment.py"
    ;;
  verify-reference)
    "$PYTHON_BIN" "$ROOT_DIR/scripts/verify_reference_results.py"
    "$PYTHON_BIN" "$ROOT_DIR/analysis/verify_thesis_metrics.py"
    ;;
  smoke)
    smoke
    ;;
  all)
    validate_domain_sources
    prepare_domain
    train_domain
    bash "$ROOT_DIR/run_wiki_preflight.sh"
    bash "$ROOT_DIR/run_wiki_model_matrix.sh"
    bash "$ROOT_DIR/run_wiki_analysis.sh"
    run_domain_shift
    "$PYTHON_BIN" "$ROOT_DIR/scripts/capture_environment.py"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    usage >&2
    exit 2
    ;;
esac
