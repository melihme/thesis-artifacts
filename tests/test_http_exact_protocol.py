from __future__ import annotations

import importlib.util
import argparse
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT_DIR / "llm-ner" / "evaluate_http_exact.py"
SPEC = importlib.util.spec_from_file_location("evaluate_http_exact", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HttpExactProtocolTests(unittest.TestCase):
    def test_prompt_contains_indices_and_strict_schema(self) -> None:
        example = {
            "tokens": ["Ankara", "güzel"],
            "labels": ["B-GPE", "O"],
            "text": "Ankara güzel",
            "token_offsets": [[0, 6], [7, 12]],
        }
        prompt = MODULE.build_prompt(example, [], ["GPE"])
        self.assertIn("[0] Ankara", prompt)
        self.assertIn("start_token", prompt)
        self.assertIn('{"entities":[]}', prompt)

    def test_parse_stream_retains_text_usage_finish_reason_and_metrics(self) -> None:
        lines = [
            b'data: {"id":"req-1","choices":[{"delta":{"content":"{\\"entities\\":"},"finish_reason":null}]}\n',
            b'data: {"id":"req-1","choices":[{"delta":{"content":"[]}"},"finish_reason":"stop"}]}\n',
            b'data: {"id":"req-1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3},"metrics":{"queue_time_ms":1.5}}\n',
            b'data: [DONE]\n',
        ]
        result = MODULE.parse_stream_lines(lines, time.perf_counter_ns())
        self.assertEqual(result["response_text"], '{"entities":[]}')
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"]["completion_tokens"], 3)
        self.assertEqual(result["server_metrics"]["queue_time_ms"], 1.5)
        self.assertIsNotNone(result["client_ttft_ms"])

    def test_malformed_json_yields_no_entities(self) -> None:
        payload = MODULE.extract_json_payload("not-json")
        entities, failures = MODULE.parse_entities(payload)
        self.assertEqual(entities, [])
        self.assertEqual(failures, ["invalid_json"])

    def test_wrapped_json_is_not_silently_repaired(self) -> None:
        self.assertIsNone(MODULE.extract_json_payload('```json\n{"entities":[]}\n```'))
        self.assertIsNone(MODULE.extract_json_payload('Answer: {"entities":[]}'))

    def test_streaming_http_call_end_to_end(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            request_payload = None

            def log_message(self, *_args):
                return

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                Handler.request_payload = json.loads(self.rfile.read(length))
                chunks = [
                    {"id": "request-1", "choices": [{"delta": {"content": '{"entities":'}, "finish_reason": None}]},
                    {"id": "request-1", "choices": [{"delta": {"content": "[]}"}, "finish_reason": "stop"}]},
                    {"id": "request-1", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}, "metrics": {"queue_time_ms": 0.2}},
                ]
                body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                encoded = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        try:
            server = HTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            self.skipTest("current sandbox forbids loopback socket binding")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            args = argparse.Namespace(
                api_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="fake-model",
                max_tokens=32,
                temperature=0.0,
                disable_thinking=True,
                api_key="token",
                api_max_retries=0,
                timeout=5.0,
            )
            result = MODULE.call_openai_compatible(args, "prompt")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(result["response_text"], '{"entities":[]}')
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"]["completion_tokens"], 3)
        self.assertTrue(Handler.request_payload["stream"])


if __name__ == "__main__":
    unittest.main()
