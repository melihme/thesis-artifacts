#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ner_framework.bert_inference import BertNerEngine
from ner_framework.data import ensure_materialized_dataset, load_materialized_dataset
from ner_framework.exact_span import SCORING_PROTOCOL
from ner_framework.timing import LATENCY_PROTOCOL, summarise_milliseconds
from ner_framework.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate direct versus HTTP BERT inference without producing thesis results.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--materialized-data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="wiki_ner")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT_DIR)
    return resolved


def post(base_url: str, example: Dict[str, object], timeout: float) -> Dict[str, object]:
    body = json.dumps(
        {
            "example_id": example["id"],
            "tokens": example["tokens"],
            "text": example["text"],
            "token_offsets": example["token_offsets"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/v1/ner",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.perf_counter_ns()
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    payload["client_e2e_ms"] = (time.perf_counter_ns() - started_ns) / 1_000_000
    return payload


def main() -> None:
    args = parse_args()
    args.model_dir = inside(args.model_dir)
    args.dataset_root = inside(args.dataset_root)
    args.materialized_data_root = inside(args.materialized_data_root)
    args.output_path = inside(args.output_path)
    dataset = ensure_materialized_dataset(args.dataset_name, args.materialized_data_root, args.dataset_root)
    examples = load_materialized_dataset(dataset.dataset_root)["test"][: args.sample_size]
    engine = BertNerEngine(args.model_dir, max_length=args.max_length)

    direct_ms = []
    http_ms = []
    server_ms = []
    mismatches = []
    records = []
    for example in examples:
        direct_started = time.perf_counter_ns()
        direct = engine.predict(example["id"], example["tokens"], example["text"], example["token_offsets"])
        direct_elapsed = (time.perf_counter_ns() - direct_started) / 1_000_000
        remote = post(args.api_base_url, example, args.timeout)
        match = direct["predicted_labels"] == remote.get("predicted_labels")
        if not match:
            mismatches.append(example["id"])
        direct_ms.append(direct_elapsed)
        http_ms.append(float(remote["client_e2e_ms"]))
        if isinstance(remote.get("timing", {}).get("server_total_ms"), (int, float)):
            server_ms.append(float(remote["timing"]["server_total_ms"]))
        records.append(
            {
                "example_id": example["id"],
                "predictions_identical": match,
                "direct_e2e_ms": direct_elapsed,
                "http_client_e2e_ms": remote["client_e2e_ms"],
                "http_server_total_ms": remote.get("timing", {}).get("server_total_ms"),
            }
        )

    if mismatches:
        raise RuntimeError(f"Direct/HTTP prediction mismatch for: {', '.join(mismatches[:5])}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_path,
        {
            "status": "calibration_only",
            "sample_size": len(records),
            "predictions_identical": True,
            "direct_e2e": summarise_milliseconds(direct_ms),
            "http_client_e2e": summarise_milliseconds(http_ms),
            "http_server_total": summarise_milliseconds(server_ms),
            "median_http_minus_direct_ms": statistics.median(
                [http_value - direct_value for http_value, direct_value in zip(http_ms, direct_ms)]
            ) if records else None,
            "records": records,
            "scoring_protocol": SCORING_PROTOCOL,
            "latency_protocol": LATENCY_PROTOCOL,
        },
    )
    print(f"Saved calibration record to {args.output_path}")


if __name__ == "__main__":
    main()
