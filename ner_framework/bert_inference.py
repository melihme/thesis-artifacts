from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Sequence

from .exact_span import SCORING_PROTOCOL, bio_to_spans, enrich_spans
from .timing import LATENCY_PROTOCOL


class BertNerEngine:
    def __init__(
        self,
        model_dir: Path,
        max_length: int = 512,
        device_name: str = "auto",
        dtype_name: str = "auto",
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.model_dir = model_dir.resolve()
        self.max_length = int(max_length)
        self.device = self._resolve_device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir, local_files_only=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("BERT HTTP service requires a fast tokenizer for word-to-subword alignment.")

        requested_dtype = self._resolve_dtype(dtype_name)
        if requested_dtype is not None:
            self.model = self.model.to(dtype=requested_dtype)
        self.model = self.model.to(self.device)
        self.model.eval()

        raw_id2label = dict(getattr(self.model.config, "id2label", {}))
        self.id2label = {int(key): str(value) for key, value in raw_id2label.items()}
        if not self.id2label:
            raise ValueError("Checkpoint config does not define id2label.")

    def _resolve_device(self, device_name: str):
        torch = self.torch
        if device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested, but PyTorch cannot access a CUDA device.")
        return device

    def _resolve_dtype(self, dtype_name: str):
        if dtype_name == "auto":
            return None
        lookup = {
            "float32": self.torch.float32,
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
        }
        if dtype_name not in lookup:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        return lookup[dtype_name]

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(self.torch, "mps"):
            self.torch.mps.synchronize()

    def metadata(self) -> Dict[str, object]:
        config = self.model.config
        return {
            "model_dir": str(self.model_dir),
            "model_type": getattr(config, "model_type", None),
            "architecture": list(getattr(config, "architectures", []) or []),
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "max_length": self.max_length,
            "labels": [self.id2label[index] for index in sorted(self.id2label)],
            "torch_version": self.torch.__version__,
            "transformers_version": self.transformers_version,
            "cuda_version": self.torch.version.cuda,
            "gpu_name": self.torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "python_version": platform.python_version(),
            "cpu_model": platform.processor() or platform.machine(),
            "cpu_logical_count": os.cpu_count(),
            "torch_num_threads": self.torch.get_num_threads(),
            "torch_num_interop_threads": self.torch.get_num_interop_threads(),
            "scoring_protocol": SCORING_PROTOCOL,
            "latency_protocol": LATENCY_PROTOCOL,
        }

    def predict(self, example_id: str, tokens: Sequence[str], text: str, token_offsets: Sequence[Sequence[int]]) -> Dict[str, object]:
        if not tokens or any(not isinstance(token, str) for token in tokens):
            raise ValueError("tokens must be a non-empty list of strings")
        if len(token_offsets) != len(tokens):
            raise ValueError("token_offsets must contain one offset pair per token")

        tokenization_started = time.perf_counter_ns()
        encoded = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoded.word_ids(batch_index=0)
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        if getattr(self.model.config, "model_type", None) == "distilbert":
            model_inputs.pop("token_type_ids", None)
        tokenization_ms = (time.perf_counter_ns() - tokenization_started) / 1_000_000

        self._synchronize()
        model_started = time.perf_counter_ns()
        with self.torch.inference_mode():
            logits = self.model(**model_inputs).logits
        self._synchronize()
        model_ms = (time.perf_counter_ns() - model_started) / 1_000_000

        postprocess_started = time.perf_counter_ns()
        subtoken_predictions = logits.argmax(dim=-1)[0].detach().cpu().tolist()
        predicted_labels: List[str] = ["O"] * len(tokens)
        seen_word_ids: set[int] = set()
        for word_id, prediction_id in zip(word_ids, subtoken_predictions):
            if word_id is None or word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            predicted_labels[int(word_id)] = self.id2label[int(prediction_id)]

        example = {
            "id": example_id,
            "tokens": list(tokens),
            "text": text,
            "token_offsets": [list(offset) for offset in token_offsets],
        }
        predicted_spans = enrich_spans(example, bio_to_spans(predicted_labels))
        postprocess_ms = (time.perf_counter_ns() - postprocess_started) / 1_000_000

        return {
            "example_id": example_id,
            "predicted_labels": predicted_labels,
            "predicted_spans": predicted_spans,
            "truncated": len(seen_word_ids) < len(tokens),
            "covered_token_count": len(seen_word_ids),
            "timing": {
                "tokenization_ms": tokenization_ms,
                "model_execution_ms": model_ms,
                "postprocessing_ms": postprocess_ms,
            },
        }
