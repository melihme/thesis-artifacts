#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ner_framework.bert_inference import BertNerEngine
from ner_framework.exact_span import SCORING_PROTOCOL
from ner_framework.timing import LATENCY_PROTOCOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one local BERT-style NER checkpoint over HTTP.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return parser.parse_args()


def ensure_inside_bundle(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT_DIR)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside {ROOT_DIR}: {resolved}") from exc
    return resolved


class NerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ColabRun2BertNER/1.0"

    @property
    def engine(self) -> BertNerEngine:
        return self.server.engine  # type: ignore[attr-defined]

    def log_message(self, format_string: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format_string % args))

    def _send(self, status: int, payload: Dict[str, object], request_started_ns: int) -> None:
        serialization_started = time.perf_counter_ns()
        provisional = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        serialization_ms = (time.perf_counter_ns() - serialization_started) / 1_000_000
        timing = payload.setdefault("timing", {})
        if isinstance(timing, dict):
            timing["serialization_ms"] = serialization_ms
            timing["server_total_ms"] = (time.perf_counter_ns() - request_started_ns) / 1_000_000
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        request_started_ns = time.perf_counter_ns()
        if self.path == "/health":
            self._send(
                200,
                {
                    "status": "ready",
                    "model_dir": str(self.engine.model_dir),
                    "pid": os.getpid(),
                    "scoring_protocol": SCORING_PROTOCOL,
                    "latency_protocol": LATENCY_PROTOCOL,
                },
                request_started_ns,
            )
            return
        if self.path == "/metadata":
            self._send(200, self.engine.metadata(), request_started_ns)
            return
        self._send(404, {"error": "not_found"}, request_started_ns)

    def do_POST(self) -> None:
        request_started_ns = time.perf_counter_ns()
        if self.path != "/v1/ner":
            self._send(404, {"error": "not_found"}, request_started_ns)
            return

        parsing_started = time.perf_counter_ns()
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            request_payload = json.loads(body.decode("utf-8"))
            if not isinstance(request_payload, dict):
                raise ValueError("request body must be a JSON object")
            parsing_ms = (time.perf_counter_ns() - parsing_started) / 1_000_000
            result = self.engine.predict(
                example_id=str(request_payload.get("example_id", "")),
                tokens=request_payload.get("tokens", []),
                text=str(request_payload.get("text", "")),
                token_offsets=request_payload.get("token_offsets", []),
            )
            timing = result.setdefault("timing", {})
            if isinstance(timing, dict):
                timing["request_parsing_ms"] = parsing_ms
                timing["queue_ms"] = 0.0
            result["scoring_protocol"] = SCORING_PROTOCOL
            result["latency_protocol"] = LATENCY_PROTOCOL
            self._send(200, result, request_started_ns)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": "invalid_request", "detail": str(exc)}, request_started_ns)
        except Exception as exc:
            self._send(500, {"error": "inference_error", "detail": str(exc)}, request_started_ns)


def main() -> None:
    args = parse_args()
    model_dir = ensure_inside_bundle(args.model_dir, "Model directory")
    engine = BertNerEngine(
        model_dir=model_dir,
        max_length=args.max_length,
        device_name=args.device,
        dtype_name=args.dtype,
    )
    server = HTTPServer((args.host, args.port), NerHandler)
    server.engine = engine  # type: ignore[attr-defined]
    print(json.dumps({"status": "ready", "url": f"http://{args.host}:{args.port}", **engine.metadata()}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
