#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
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
    classify_span_errors,
    enrich_spans,
    per_label_metrics,
    score_exact_spans,
    spans_to_bio,
    validate_prediction_entities,
)
from ner_framework.timing import LATENCY_PROTOCOL, latency_summary
from ner_framework.utils import read_jsonl, write_json, write_jsonl
from ner_framework.wiki_protocol import select_wiki_prompt_examples, validate_wiki_protocol


PROMPT_COUNTS = {"zero-shot": 0, "one-shot": 1, "three-shot": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict token-span Wiki NER evaluation through a local OpenAI-compatible API.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--dataset-name", default="wiki_ner")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--materialized-data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--prompt-strategy", choices=sorted(PROMPT_COUNTS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-test-count", type=int, default=1000)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--api-max-retries", type=int, default=0)
    parser.add_argument("--warmup-count", type=int, default=20)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument("--disable-thinking", action="store_true")
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


def indexed_tokens(tokens: Sequence[object]) -> str:
    return "\n".join(f"[{index}] {token}" for index, token in enumerate(tokens))


def prompt_entities(example: Mapping[str, object]) -> List[Dict[str, object]]:
    return enrich_spans(example, bio_to_spans(example.get("labels", [])))


def build_system_instruction() -> str:
    return (
        "You perform Turkish named entity recognition. Return exactly one JSON object and no explanation. "
        "Entity boundaries must refer to the supplied zero-based token indices."
    )


def build_prompt(
    example: Mapping[str, object],
    demonstrations: Sequence[Mapping[str, object]],
    allowed_entity_types: Sequence[str],
) -> str:
    schema = (
        '{"entities":[{"start_token":0,"end_token":0,"text":"...","label":"PERSON"}]}'
    )
    parts = [
        "Task: identify every named entity in the indexed Turkish tokens.",
        "Allowed labels: " + ", ".join(allowed_entity_types),
        "Required JSON schema: " + schema,
        "Rules:",
        "- start_token and end_token are inclusive integers.",
        "- Use only the allowed labels, exactly as written.",
        "- text must exactly equal the source substring covered by the indices.",
        "- Do not return overlapping or duplicate entities.",
        '- If there is no entity, return {"entities":[]}.' ,
    ]

    for index, demonstration in enumerate(demonstrations, start=1):
        parts.extend(
            [
                f"Demonstration {index} tokens:",
                indexed_tokens(demonstration.get("tokens", [])),
                "Demonstration answer:",
                json.dumps({"entities": prompt_entities(demonstration)}, ensure_ascii=False, separators=(",", ":")),
            ]
        )

    parts.extend(
        [
            "Input tokens:",
            indexed_tokens(example.get("tokens", [])),
            "Answer:",
        ]
    )
    return "\n".join(parts)


def strip_v1_suffix(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    return trimmed[:-3] if trimmed.endswith("/v1") else trimmed


def snapshot_prometheus_metrics(base_url: str, destination: Path, timeout: float = 10.0) -> Dict[str, object]:
    url = f"{strip_v1_suffix(base_url)}/metrics"
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        destination.write_text(body, encoding="utf-8")
        return {"available": True, "url": url, "path": str(destination)}
    except Exception as exc:
        destination.write_text(f"# unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"available": False, "url": url, "path": str(destination), "error": str(exc)}


def _choice_text(payload: Mapping[str, object], streaming: bool) -> Tuple[str, Optional[str]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "", None
    choice = choices[0]
    finish_reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
    if streaming:
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content, finish_reason
        text = choice.get("text")
        return (text if isinstance(text, str) else ""), finish_reason
    message = choice.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"]), finish_reason
    text = choice.get("text")
    return (text if isinstance(text, str) else ""), finish_reason


def parse_stream_lines(lines: Iterable[bytes], request_started_ns: int) -> Dict[str, object]:
    pieces: List[str] = []
    usage: Dict[str, object] = {}
    server_metrics: Dict[str, object] = {}
    finish_reason: Optional[str] = None
    response_id: Optional[str] = None
    first_content_ns: Optional[int] = None

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("id"), str):
            response_id = str(payload["id"])
        piece, current_finish_reason = _choice_text(payload, streaming=True)
        if piece:
            if first_content_ns is None:
                first_content_ns = time.perf_counter_ns()
            pieces.append(piece)
        if current_finish_reason is not None:
            finish_reason = current_finish_reason
        if isinstance(payload.get("usage"), dict):
            usage = dict(payload["usage"])
        if isinstance(payload.get("metrics"), dict):
            server_metrics = dict(payload["metrics"])

    return {
        "response_text": "".join(pieces),
        "response_id": response_id,
        "finish_reason": finish_reason,
        "usage": usage,
        "server_metrics": server_metrics,
        "client_ttft_ms": (
            (first_content_ns - request_started_ns) / 1_000_000 if first_content_ns is not None else None
        ),
    }


def call_openai_compatible(
    args: argparse.Namespace,
    prompt: str,
) -> Dict[str, object]:
    endpoint = f"{args.api_base_url.rstrip('/')}/chat/completions"
    payload: Dict[str, object] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": build_system_instruction()},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "response_format": {"type": "json_object"},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}

    attempts = max(1, args.api_max_retries + 1)
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        request_started_ns = time.perf_counter_ns()
        try:
            request = Request(endpoint, data=body, headers=headers, method="POST")
            with urlopen(request, timeout=args.timeout) as response:
                http_status = int(response.status)
                response_headers = dict(response.headers.items())
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    result = parse_stream_lines(response, request_started_ns)
                else:
                    response_payload = json.loads(response.read().decode("utf-8"))
                    if not isinstance(response_payload, dict):
                        raise ValueError("OpenAI-compatible response must be a JSON object")
                    text, finish_reason = _choice_text(response_payload, streaming=False)
                    result = {
                        "response_text": text,
                        "response_id": response_payload.get("id"),
                        "finish_reason": finish_reason,
                        "usage": response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {},
                        "server_metrics": response_payload.get("metrics") if isinstance(response_payload.get("metrics"), dict) else {},
                        "client_ttft_ms": None,
                    }
            client_e2e_ms = (time.perf_counter_ns() - request_started_ns) / 1_000_000
            client_ttft_ms = result.get("client_ttft_ms")
            result.update(
                {
                    "http_status": http_status,
                    "response_headers": response_headers,
                    "retry_count": attempt - 1,
                    "client_e2e_ms": client_e2e_ms,
                    "client_generation_after_first_token_ms": (
                        client_e2e_ms - float(client_ttft_ms)
                        if isinstance(client_ttft_ms, (int, float))
                        else None
                    ),
                }
            )
            return result
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenAI-compatible HTTP {exc.code}: {body_text}")
        except URLError as exc:
            last_error = RuntimeError(f"OpenAI-compatible transport error: {exc.reason}")
        except Exception as exc:
            last_error = exc
        if attempt < attempts:
            time.sleep(min(2 ** (attempt - 1), 8))

    assert last_error is not None
    raise last_error


def extract_json_payload(text: str) -> Optional[object]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def parse_entities(payload: object) -> Tuple[List[object], List[str]]:
    if payload is None:
        return [], ["invalid_json"]
    if not isinstance(payload, dict):
        return [], ["root_not_object"]
    entities = payload.get("entities")
    if not isinstance(entities, list):
        return [], ["entities_not_list"]
    return list(entities), []


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def result_record(
    example: Dict[str, object],
    prompt: str,
    response: Dict[str, object],
    allowed_entity_types: Sequence[str],
) -> Dict[str, object]:
    payload = extract_json_payload(str(response.get("response_text", "")))
    raw_entities, parse_failures = parse_entities(payload)
    predicted_spans, rejected = validate_prediction_entities(example, raw_entities, allowed_entity_types)
    gold_spans = enrich_spans(example, bio_to_spans(example["labels"]))
    counts = score_exact_spans(gold_spans, predicted_spans, rejected_prediction_count=len(rejected))
    error_taxonomy = classify_span_errors(gold_spans, predicted_spans, len(rejected))
    predicted_labels = spans_to_bio(predicted_spans, len(example["tokens"]))
    server_metrics = response.get("server_metrics") if isinstance(response.get("server_metrics"), dict) else {}
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    client_e2e_ms = response.get("client_e2e_ms")
    client_output_tokens_per_second = (
        float(completion_tokens) / (float(client_e2e_ms) / 1000.0)
        if isinstance(completion_tokens, (int, float))
        and isinstance(client_e2e_ms, (int, float))
        and float(client_e2e_ms) > 0
        else None
    )

    return {
        "example_id": example["id"],
        "text": example["text"],
        "tokens": example["tokens"],
        "indexed_tokens": indexed_tokens(example["tokens"]),
        "gold_labels": example["labels"],
        "gold_spans": gold_spans,
        "raw_response": response.get("response_text", ""),
        "raw_payload": payload,
        "raw_entity_objects": raw_entities,
        "predicted_spans": predicted_spans,
        "predicted_labels": predicted_labels,
        "rejected_predictions": rejected,
        "parse_failures": parse_failures,
        **counts,
        "error_taxonomy": error_taxonomy,
        "correct_tokens": sum(str(gold) == str(predicted) for gold, predicted in zip(example["labels"], predicted_labels)),
        "token_count": len(example["tokens"]),
        "json_valid": payload is not None,
        "schema_valid": not parse_failures and not rejected,
        "output_valid": not parse_failures and not rejected,
        "truncated": response.get("finish_reason") == "length",
        "prompt_sha256": prompt_hash(prompt),
        "request": {
            "http_status": response.get("http_status"),
            "response_id": response.get("response_id"),
            "retry_count": response.get("retry_count", 0),
            "client_e2e_ms": response.get("client_e2e_ms"),
            "client_ttft_ms": response.get("client_ttft_ms"),
            "client_generation_after_first_token_ms": response.get("client_generation_after_first_token_ms"),
            "client_output_tokens_per_second": client_output_tokens_per_second,
            "finish_reason": response.get("finish_reason"),
        },
        "usage": usage,
        "server_metrics": server_metrics,
        "scoring_protocol": SCORING_PROTOCOL,
        "latency_protocol": LATENCY_PROTOCOL,
    }


def failure_record(example: Dict[str, object], prompt: str, error: Exception, elapsed_ms: float) -> Dict[str, object]:
    gold_spans = enrich_spans(example, bio_to_spans(example["labels"]))
    counts = score_exact_spans(gold_spans, [])
    predicted_labels = ["O"] * len(example["tokens"])
    return {
        "example_id": example["id"],
        "text": example["text"],
        "tokens": example["tokens"],
        "indexed_tokens": indexed_tokens(example["tokens"]),
        "gold_labels": example["labels"],
        "gold_spans": gold_spans,
        "raw_response": "",
        "raw_payload": None,
        "raw_entity_objects": [],
        "predicted_spans": [],
        "predicted_labels": predicted_labels,
        "rejected_predictions": [],
        "parse_failures": ["request_failed"],
        **counts,
        "error_taxonomy": classify_span_errors(gold_spans, []),
        "correct_tokens": sum(label == "O" for label in example["labels"]),
        "token_count": len(example["tokens"]),
        "json_valid": False,
        "schema_valid": False,
        "output_valid": False,
        "truncated": False,
        "prompt_sha256": prompt_hash(prompt),
        "request": {
            "http_status": None,
            "retry_count": 0,
            "client_e2e_ms": elapsed_ms,
            "client_ttft_ms": None,
            "finish_reason": None,
            "error_type": type(error).__name__,
            "error": str(error),
        },
        "usage": {},
        "server_metrics": {},
        "scoring_protocol": SCORING_PROTOCOL,
        "latency_protocol": LATENCY_PROTOCOL,
    }


def build_summary(
    args: argparse.Namespace,
    records: Sequence[Dict[str, object]],
    demonstrations: Sequence[Dict[str, object]],
    predictions_path: Path,
    metrics_snapshots: Dict[str, object],
    entity_types: Sequence[str],
) -> Dict[str, object]:
    scores = aggregate_span_counts(records)
    token_count = sum(int(record["token_count"]) for record in records)
    correct_tokens = sum(int(record["correct_tokens"]) for record in records)
    request_successes = [record for record in records if record.get("request", {}).get("http_status") == 200]
    total_client_seconds = sum(float(record["request"]["client_e2e_ms"]) for record in request_successes) / 1000.0
    completion_tokens = sum(
        int(record.get("usage", {}).get("completion_tokens", 0) or 0)
        for record in request_successes
        if isinstance(record.get("usage"), dict)
    )
    parse_failures = Counter(reason for record in records for reason in record.get("parse_failures", []))
    rejection_reasons = Counter(
        str(item.get("reason"))
        for record in records
        for item in record.get("rejected_predictions", [])
        if isinstance(item, dict)
    )
    finish_reasons = Counter(
        str(record.get("request", {}).get("finish_reason"))
        for record in records
        if record.get("request", {}).get("finish_reason") is not None
    )
    error_taxonomy = Counter(
        {key: sum(int(record.get("error_taxonomy", {}).get(key, 0)) for record in records)
         for key in {name for record in records for name in record.get("error_taxonomy", {})}}
    )
    latency = latency_summary(
        request_successes,
        {
            "client_e2e": ("request", "client_e2e_ms"),
            "client_ttft": ("request", "client_ttft_ms"),
            "client_generation_after_first_token": ("request", "client_generation_after_first_token_ms"),
            "client_output_tokens_per_second": ("request", "client_output_tokens_per_second"),
            "server_ttft": ("server_metrics", "time_to_first_token_ms"),
            "server_queue": ("server_metrics", "queue_time_ms"),
            "server_prefill": ("server_metrics", "prefill_time_ms"),
            "server_inference": ("server_metrics", "inference_time_ms"),
            "server_decode": ("server_metrics", "decode_time_ms"),
            "server_generation": ("server_metrics", "generation_time_ms"),
            "server_mean_itl": ("server_metrics", "mean_itl_ms"),
            "server_tokens_per_second": ("server_metrics", "tokens_per_second"),
        },
    )
    return {
        "status": (
            "completed"
            if args.max_examples is None and len(records) == args.expected_test_count
            else "partial"
        ),
        "approach": "llm_http",
        "model": args.model,
        "model_slug": args.model_slug,
        "model_revision": args.model_revision,
        "split": args.split,
        "prompt_strategy": args.prompt_strategy,
        "prompt_seed": None if args.prompt_strategy == "zero-shot" else args.seed,
        "few_shot_count": len(demonstrations),
        "few_shot_example_ids": [example["id"] for example in demonstrations],
        "examples_evaluated": len(records),
        "expected_examples": args.max_examples or args.expected_test_count,
        **scores,
        "accuracy": correct_tokens / token_count if token_count else 0.0,
        "per_entity_type": per_label_metrics(records, entity_types),
        "error_taxonomy": dict(error_taxonomy),
        "successful_requests": len(request_successes),
        "failed_requests": len(records) - len(request_successes),
        "valid_json_requests": sum(bool(record.get("json_valid")) for record in records),
        "schema_valid_requests": sum(bool(record.get("schema_valid")) for record in records),
        "truncated_requests": sum(bool(record.get("truncated")) for record in records),
        "parse_failures": dict(parse_failures),
        "rejected_prediction_reasons": dict(rejection_reasons),
        "finish_reasons": dict(finish_reasons),
        "requests_per_second": len(request_successes) / total_client_seconds if total_client_seconds else None,
        "completion_tokens_per_second": completion_tokens / total_client_seconds if total_client_seconds else None,
        "latency_population": "http_status_200_only",
        "failure_latency_policy": "excluded_from_percentiles_reported_separately",
        "latency": latency,
        "server_metrics_snapshots": metrics_snapshots,
        "predictions_path": str(predictions_path),
        "scoring_protocol": SCORING_PROTOCOL,
        "latency_protocol": LATENCY_PROTOCOL,
        "run_parameters": {
            "api_base_url": args.api_base_url,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timeout_seconds": args.timeout,
            "api_max_retries": args.api_max_retries,
            "warmup_count": args.warmup_count,
            "stream": True,
            "batch_size": 1,
            "concurrency": 1,
            "disable_thinking": args.disable_thinking,
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
    metadata_path = args.output_dir / "run_metadata.json"

    if args.overwrite:
        for path in (predictions_path, summary_path, metadata_path):
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

    demonstration_count = PROMPT_COUNTS[args.prompt_strategy]
    demonstrations = select_wiki_prompt_examples(examples_by_split, seed=args.seed, count=demonstration_count)
    protocol_validation = validate_wiki_protocol(
        examples_by_split,
        demonstrations,
        seed=args.seed,
        evaluation_split=args.split,
        expected_test_count=args.expected_test_count,
    )
    metadata = {
        "model": args.model,
        "model_slug": args.model_slug,
        "model_revision": args.model_revision,
        "prompt_strategy": args.prompt_strategy,
        "prompt_seed": None if args.prompt_strategy == "zero-shot" else args.seed,
        "demonstration_ids": [example["id"] for example in demonstrations],
        "protocol_validation": protocol_validation,
        "scoring_protocol": SCORING_PROTOCOL,
        "latency_protocol": LATENCY_PROTOCOL,
    }
    if metadata_path.exists() and args.resume:
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in (
            "model", "model_revision", "prompt_strategy", "prompt_seed", "demonstration_ids",
            "scoring_protocol", "latency_protocol",
        ):
            if prior.get(key) != metadata.get(key):
                raise ValueError(f"Cannot resume LLM output with mismatched {key}")
    write_json(metadata_path, metadata)

    warmup_examples = examples_by_split["train"][: args.warmup_count]
    for warmup_example in warmup_examples:
        warmup_prompt = build_prompt(warmup_example, demonstrations, manifest["entity_types"])
        call_openai_compatible(args, warmup_prompt)

    metrics_snapshots = {
        "before": snapshot_prometheus_metrics(args.api_base_url, args.output_dir / "server_metrics_before.prom")
    }
    evaluation_examples = list(examples_by_split["test"])
    if args.max_examples is not None:
        evaluation_examples = evaluation_examples[: args.max_examples]
    existing = {record["example_id"]: record for record in read_jsonl(predictions_path)} if predictions_path.exists() else {}
    if any(
        record.get("scoring_protocol") != SCORING_PROTOCOL or record.get("latency_protocol") != LATENCY_PROTOCOL
        for record in existing.values()
    ):
        raise ValueError("Existing LLM predictions use a different protocol and cannot be resumed")

    records_by_id = dict(existing)
    consecutive_errors = 0
    stop_error: Exception | None = None
    for index, example in enumerate(evaluation_examples, start=1):
        if example["id"] in records_by_id:
            continue
        prompt = build_prompt(example, demonstrations, manifest["entity_types"])
        started_ns = time.perf_counter_ns()
        try:
            response = call_openai_compatible(args, prompt)
            record = result_record(example, prompt, response, manifest["entity_types"])
            consecutive_errors = 0
        except Exception as exc:
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            record = failure_record(example, prompt, exc, elapsed_ms)
            consecutive_errors += 1
        records_by_id[example["id"]] = record
        ordered = [records_by_id[item["id"]] for item in evaluation_examples if item["id"] in records_by_id]
        if index % 10 == 0 or index == len(evaluation_examples):
            write_jsonl(predictions_path, ordered)
            print(f"LLM HTTP evaluated {index}/{len(evaluation_examples)}", flush=True)
        if consecutive_errors >= args.max_consecutive_errors:
            write_jsonl(predictions_path, ordered)
            stop_error = RuntimeError(f"Stopped after {consecutive_errors} consecutive LLM HTTP errors")
            break

    records = [records_by_id[item["id"]] for item in evaluation_examples if item["id"] in records_by_id]
    write_jsonl(predictions_path, records)
    metrics_snapshots["after"] = snapshot_prometheus_metrics(args.api_base_url, args.output_dir / "server_metrics_after.prom")
    summary = build_summary(
        args, records, demonstrations, predictions_path, metrics_snapshots, manifest["entity_types"]
    )
    write_json(summary_path, summary)
    print(f"Saved strict LLM HTTP summary to {summary_path}")
    if stop_error is not None:
        raise stop_error


if __name__ == "__main__":
    main()
