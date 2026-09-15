#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ner_framework.data import ensure_materialized_dataset, load_manifest, load_materialized_dataset
from ner_framework.exact_span import (
    SCORING_PROTOCOL,
    aggregate_span_counts,
    bio_to_spans,
    canonical_key,
    classify_span_errors,
    enrich_spans,
    per_label_metrics,
    score_exact_spans,
)
from ner_framework.timing import LATENCY_PROTOCOL, latency_summary
from ner_framework.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BERT NER HTTP service on the strict Wiki protocol.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--device-profile", required=True)
    parser.add_argument("--dataset-name", default="wiki_ner")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--materialized-data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-test-count", type=int, default=1000)
    parser.add_argument("--warmup-count", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_inside_bundle(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT_DIR)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside {ROOT_DIR}: {resolved}") from exc
    return resolved


def get_json(url: str, timeout: float) -> Dict[str, object]:
    with urlopen(Request(url, method="GET"), timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return payload


def post_example(base_url: str, example: Dict[str, object], timeout: float) -> Dict[str, object]:
    payload = {
        "example_id": example["id"],
        "tokens": example["tokens"],
        "text": example["text"],
        "token_offsets": example["token_offsets"],
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/v1/ner",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.perf_counter_ns()
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read()
            http_status = int(response.status)
    except HTTPError as exc:
        response_body = exc.read()
        raise RuntimeError(f"BERT HTTP {exc.code}: {response_body.decode('utf-8', errors='replace')}") from exc
    except URLError as exc:
        raise RuntimeError(f"BERT HTTP transport error: {exc.reason}") from exc
    client_e2e_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    response_payload = json.loads(response_body.decode("utf-8"))
    if not isinstance(response_payload, dict):
        raise ValueError("BERT HTTP response must be a JSON object")
    response_payload["_client_e2e_ms"] = client_e2e_ms
    response_payload["_http_status"] = http_status
    return response_payload


def prediction_record(example: Dict[str, object], response: Dict[str, object]) -> Dict[str, object]:
    predicted_labels = response.get("predicted_labels")
    if not isinstance(predicted_labels, list) or len(predicted_labels) != len(example["tokens"]):
        raise ValueError(f"BERT response label count mismatch for {example['id']}")

    gold_spans = enrich_spans(example, bio_to_spans(example["labels"]))
    predicted_spans = enrich_spans(example, bio_to_spans(predicted_labels))
    response_spans = response.get("predicted_spans")
    if not isinstance(response_spans, list):
        raise ValueError(f"BERT response is missing predicted_spans for {example['id']}")
    if [canonical_key(span) for span in response_spans] != [canonical_key(span) for span in predicted_spans]:
        raise ValueError(f"BERT response BIO/span disagreement for {example['id']}")

    counts = score_exact_spans(gold_spans, predicted_spans)
    error_taxonomy = classify_span_errors(gold_spans, predicted_spans)
    server_timing = response.get("timing") if isinstance(response.get("timing"), dict) else {}
    server_total_ms = server_timing.get("server_total_ms") if isinstance(server_timing, dict) else None
    client_e2e_ms = float(response["_client_e2e_ms"])
    residual = client_e2e_ms - float(server_total_ms) if isinstance(server_total_ms, (int, float)) else None
    correct_tokens = sum(str(gold) == str(predicted) for gold, predicted in zip(example["labels"], predicted_labels))

    return {
        "example_id": example["id"],
        "tokens": example["tokens"],
        "gold_labels": example["labels"],
        "predicted_labels": predicted_labels,
        "gold_spans": gold_spans,
        "predicted_spans": predicted_spans,
        **counts,
        "error_taxonomy": error_taxonomy,
        "correct_tokens": correct_tokens,
        "token_count": len(example["tokens"]),
        "output_valid": True,
        "truncated": bool(response.get("truncated", False)),
        "request": {
            "http_status": response["_http_status"],
            "retry_count": 0,
            "client_e2e_ms": client_e2e_ms,
            "http_framework_residual_ms": residual,
            "residual_negative": bool(residual is not None and residual < 0),
        },
        "server_timing": server_timing,
        "scoring_protocol": SCORING_PROTOCOL,
        "latency_protocol": LATENCY_PROTOCOL,
    }


def failure_record(example: Dict[str, object], error: Exception, elapsed_ms: float) -> Dict[str, object]:
    gold_spans = enrich_spans(example, bio_to_spans(example["labels"]))
    counts = score_exact_spans(gold_spans, [])
    return {
        "example_id": example["id"],
        "tokens": example["tokens"],
        "gold_labels": example["labels"],
        "predicted_labels": ["O"] * len(example["tokens"]),
        "gold_spans": gold_spans,
        "predicted_spans": [],
        **counts,
        "error_taxonomy": classify_span_errors(gold_spans, []),
        "correct_tokens": sum(label == "O" for label in example["labels"]),
        "token_count": len(example["tokens"]),
        "output_valid": False,
        "truncated": False,
        "request": {
            "http_status": None,
            "retry_count": 0,
            "client_e2e_ms": elapsed_ms,
            "error_type": type(error).__name__,
            "error": str(error),
        },
        "server_timing": {},
        "scoring_protocol": SCORING_PROTOCOL,
        "latency_protocol": LATENCY_PROTOCOL,
    }


def build_summary(
    args: argparse.Namespace,
    records: Sequence[Dict[str, object]],
    predictions_path: Path,
    server_metadata: Dict[str, object],
    entity_types: Sequence[str],
) -> Dict[str, object]:
    scores = aggregate_span_counts(records)
    token_count = sum(int(record["token_count"]) for record in records)
    correct_tokens = sum(int(record["correct_tokens"]) for record in records)
    successful = [record for record in records if record.get("output_valid")]
    error_taxonomy = Counter(
        {key: sum(int(record.get("error_taxonomy", {}).get(key, 0)) for record in records)
         for key in {name for record in records for name in record.get("error_taxonomy", {})}}
    )
    total_client_seconds = sum(float(record["request"]["client_e2e_ms"]) for record in successful) / 1000.0
    latency = latency_summary(
        successful,
        {
            "client_e2e": ("request", "client_e2e_ms"),
            "http_framework_residual": ("request", "http_framework_residual_ms"),
            "server_total": ("server_timing", "server_total_ms"),
            "request_parsing": ("server_timing", "request_parsing_ms"),
            "tokenization": ("server_timing", "tokenization_ms"),
            "queue": ("server_timing", "queue_ms"),
            "model_execution": ("server_timing", "model_execution_ms"),
            "postprocessing": ("server_timing", "postprocessing_ms"),
            "serialization": ("server_timing", "serialization_ms"),
        },
    )
    return {
        "status": (
            "completed"
            if args.max_examples is None and len(records) == args.expected_test_count
            else "partial"
        ),
        "approach": "transformer_http",
        "model_slug": args.model_slug,
        "device_profile": args.device_profile,
        "split": args.split,
        "examples_evaluated": len(records),
        "expected_examples": args.max_examples or args.expected_test_count,
        **scores,
        "accuracy": correct_tokens / token_count if token_count else 0.0,
        "per_entity_type": per_label_metrics(records, entity_types),
        "error_taxonomy": dict(error_taxonomy),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "truncated_requests": sum(bool(record.get("truncated")) for record in records),
        "requests_per_second": len(successful) / total_client_seconds if total_client_seconds else None,
        "latency_population": "http_status_200_only",
        "failure_latency_policy": "excluded_from_percentiles_reported_separately",
        "latency": latency,
        "server_metadata": server_metadata,
        "predictions_path": str(predictions_path),
        "scoring_protocol": SCORING_PROTOCOL,
        "latency_protocol": LATENCY_PROTOCOL,
        "run_parameters": {
            "api_base_url": args.api_base_url,
            "device_profile": args.device_profile,
            "warmup_count": args.warmup_count,
            "timeout_seconds": args.timeout,
            "batch_size": 1,
            "concurrency": 1,
        },
    }


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")
    args.dataset_root = ensure_inside_bundle(args.dataset_root, "Dataset root")
    args.materialized_data_root = ensure_inside_bundle(args.materialized_data_root, "Materialized data root")
    args.output_dir = ensure_inside_bundle(args.output_dir, "Output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"

    if args.overwrite:
        for path in (predictions_path, summary_path):
            if path.exists():
                path.unlink()
    elif predictions_path.exists() and not args.resume:
        raise FileExistsError(f"Predictions already exist; pass --resume or --overwrite: {predictions_path}")

    dataset = ensure_materialized_dataset(
        dataset_name=args.dataset_name,
        materialized_data_root=args.materialized_data_root,
        source_root=args.dataset_root,
    )
    examples_by_split = load_materialized_dataset(dataset.dataset_root)
    manifest = load_manifest(dataset.dataset_root)
    if len(examples_by_split["test"]) != args.expected_test_count:
        raise ValueError(f"Expected {args.expected_test_count} Wiki test examples, found {len(examples_by_split['test'])}")

    server_metadata = get_json(f"{args.api_base_url.rstrip('/')}/metadata", args.timeout)
    if server_metadata.get("scoring_protocol") != SCORING_PROTOCOL:
        raise ValueError("BERT server scoring protocol mismatch")
    if server_metadata.get("latency_protocol") != LATENCY_PROTOCOL:
        raise ValueError("BERT server latency protocol mismatch")

    for warmup_example in examples_by_split["train"][: args.warmup_count]:
        post_example(args.api_base_url, warmup_example, args.timeout)

    evaluation_examples = list(examples_by_split["test"])
    if args.max_examples is not None:
        evaluation_examples = evaluation_examples[: args.max_examples]
    existing = {record["example_id"]: record for record in read_jsonl(predictions_path)} if predictions_path.exists() else {}
    if any(
        record.get("scoring_protocol") != SCORING_PROTOCOL or record.get("latency_protocol") != LATENCY_PROTOCOL
        for record in existing.values()
    ):
        raise ValueError("Existing BERT predictions use a different protocol and cannot be resumed")
    if any(record.get("device_profile") != args.device_profile for record in existing.values()):
        raise ValueError("Existing BERT predictions use a different device profile and cannot be resumed")

    records_by_id = dict(existing)
    consecutive_errors = 0
    stop_error: Exception | None = None
    for index, example in enumerate(evaluation_examples, start=1):
        if example["id"] in records_by_id:
            continue
        started_ns = time.perf_counter_ns()
        try:
            response = post_example(args.api_base_url, example, args.timeout)
            record = prediction_record(example, response)
            consecutive_errors = 0
        except Exception as exc:
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            record = failure_record(example, exc, elapsed_ms)
            consecutive_errors += 1
        record["device_profile"] = args.device_profile
        records_by_id[example["id"]] = record
        ordered = [records_by_id[item["id"]] for item in evaluation_examples if item["id"] in records_by_id]
        write_jsonl(predictions_path, ordered)
        if index % 25 == 0:
            print(f"BERT HTTP evaluated {index}/{len(evaluation_examples)}", flush=True)
        if consecutive_errors >= args.max_consecutive_errors:
            stop_error = RuntimeError(f"Stopped after {consecutive_errors} consecutive BERT HTTP errors")
            break

    records = [records_by_id[item["id"]] for item in evaluation_examples if item["id"] in records_by_id]
    summary = build_summary(args, records, predictions_path, server_metadata, manifest["entity_types"])
    write_json(summary_path, summary)
    print(f"Saved BERT HTTP summary to {summary_path}")
    if stop_error is not None:
        raise stop_error


if __name__ == "__main__":
    main()
