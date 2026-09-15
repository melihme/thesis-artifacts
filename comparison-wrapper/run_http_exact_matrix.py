#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ner_framework.data import ensure_materialized_dataset, load_materialized_dataset
from ner_framework.exact_span import SCORING_PROTOCOL
from ner_framework.timing import LATENCY_PROTOCOL
from ner_framework.utils import write_json
from ner_framework.wiki_protocol import select_wiki_prompt_examples, validate_wiki_protocol


DEFAULT_CONFIG = ROOT_DIR / "comparison-wrapper" / "wiki_http_exact_matrix.json"
DEFAULT_LOCAL_SCRATCH_ROOT = Path("/mnt/local-scratch/thesis-artifacts-wiki")
DEFAULT_VENV_ROOT = Path(
    os.environ.get("DEFAULT_VENV_ROOT")
    or DEFAULT_LOCAL_SCRATCH_ROOT / ".venvs"
)
DEFAULT_APP_VENV = Path(os.environ.get("APP_VENV") or DEFAULT_VENV_ROOT / "app")
DEFAULT_PYTHON = DEFAULT_APP_VENV / "bin" / "python"


def local_scratch_root(environment: Optional[Dict[str, str]] = None) -> Path:
    source = os.environ if environment is None else environment
    return Path(
        source.get("LOCAL_SCRATCH_ROOT") or DEFAULT_LOCAL_SCRATCH_ROOT
    ).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strict Wiki NER HTTP comparison matrix.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--transformers", nargs="*", default=None)
    parser.add_argument("--bert-devices", nargs="*", default=None)
    parser.add_argument("--strategies", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-transformers", action="store_true")
    parser.add_argument("--skip-llms", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def ensure_inside_root(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT_DIR)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside {ROOT_DIR}: {resolved}") from exc
    return resolved


def root_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    return ensure_inside_root(path if path.is_absolute() else ROOT_DIR / path, label)


def select(items: Sequence[Dict[str, Any]], requested: Optional[Sequence[str]], label: str) -> List[Dict[str, Any]]:
    if not requested:
        return list(items)
    by_slug = {str(item["slug"] if "slug" in item else item["name"]): item for item in items}
    missing = [value for value in requested if value not in by_slug]
    if missing:
        raise ValueError(f"Unknown {label}: {', '.join(missing)}")
    return [by_slug[value] for value in requested]


def validate_config(config: Dict[str, Any]) -> None:
    protocol = config.get("protocol", {})
    if protocol.get("scoring") != SCORING_PROTOCOL:
        raise ValueError(f"Config must use scoring protocol {SCORING_PROTOCOL}")
    if protocol.get("latency") != LATENCY_PROTOCOL:
        raise ValueError(f"Config must use latency protocol {LATENCY_PROTOCOL}")
    measurement = config.get("measurement", {})
    if int(measurement.get("concurrency", 0)) != 1 or int(measurement.get("batch_size", 0)) != 1:
        raise ValueError("Primary HTTP latency protocol requires concurrency=1 and batch_size=1")
    if int(measurement.get("automatic_retries", -1)) != 0:
        raise ValueError("Primary latency measurements must disable automatic retries")
    strategies = {item["name"]: item for item in config.get("strategies", [])}
    if set(strategies) != {"zero-shot", "one-shot", "three-shot"}:
        raise ValueError("Matrix must define exactly zero-shot, one-shot, and three-shot")
    if strategies["zero-shot"].get("seed_mode") != "fixed":
        raise ValueError("Zero-shot must use seed_mode=fixed")
    if any(strategies[name].get("seed_mode") != "demonstration" for name in ("one-shot", "three-shot")):
        raise ValueError("One-shot and three-shot must use demonstration seeds")
    device_profiles = config.get("bert_server", {}).get("device_profiles", [])
    if not isinstance(device_profiles, list) or not device_profiles:
        raise ValueError("bert_server.device_profiles must define CPU and GPU runs")
    device_slugs = [str(profile.get("slug", "")) for profile in device_profiles]
    if not all(device_slugs) or len(device_slugs) != len(set(device_slugs)):
        raise ValueError("BERT device profile slugs must be present and unique")
    device_types = [str(profile.get("device", "")).split(":", 1)[0] for profile in device_profiles]
    if device_types != ["cpu", "cuda"]:
        raise ValueError("BERT device profiles must be ordered as explicit cpu and cuda runs")
    dtypes = {str(profile.get("dtype", "")) for profile in device_profiles}
    if dtypes != {"float32"}:
        raise ValueError("CPU/GPU device-effect comparison must use float32 for both runs")
    if not isinstance(config.get("llm_server", {}).get("use_flashinfer_sampler"), bool):
        raise ValueError("llm_server.use_flashinfer_sampler must be an explicit boolean")


def strategy_seed_runs(
    strategies: Sequence[Dict[str, Any]],
    demonstration_seeds: Sequence[int],
    zero_shot_seed: int,
) -> List[Tuple[Dict[str, Any], int, str]]:
    runs: List[Tuple[Dict[str, Any], int, str]] = []
    for strategy in strategies:
        if strategy.get("seed_mode") == "fixed":
            runs.append((strategy, int(zero_shot_seed), "fixed"))
        else:
            for seed in demonstration_seeds:
                runs.append((strategy, int(seed), str(int(seed))))
    return runs


def dataset_paths(config: Dict[str, Any]) -> Dict[str, Path | str | int]:
    dataset = config["dataset"]
    return {
        "name": str(dataset["name"]),
        "source_root": root_path(dataset["source_root"], "Dataset source root"),
        "materialized_data_root": root_path(dataset["materialized_data_root"], "Materialized data root"),
        "expected_test_count": int(dataset["expected_test_count"]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_command_output(command: Sequence[str]) -> Optional[str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        return completed.stdout.strip() or completed.stderr.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def installed_python_package_version(venv_path: Path, package_name: str) -> Optional[str]:
    normalized_name = package_name.lower().replace("-", "_")
    metadata_paths = sorted(
        venv_path.glob(f"lib/python*/site-packages/{normalized_name}-*.dist-info/METADATA")
    )
    for metadata_path in metadata_paths:
        with metadata_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("Version:"):
                    return line.partition(":")[2].strip() or None
    return None


def server_version(payload: Dict[str, Any]) -> Optional[str]:
    value = payload.get("version")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def write_protocol_manifest(config_path: Path, config: Dict[str, Any], results_dir: Path) -> None:
    paths = dataset_paths(config)
    materialized_root = Path(paths["materialized_data_root"]) / str(paths["name"])
    dataset_hashes = {
        f"{split}.jsonl": sha256_file(materialized_root / f"{split}.jsonl")
        for split in ("train", "eval", "test")
    }
    transformer_artifacts: List[Dict[str, object]] = []
    for transformer in config["transformers"]:
        model_dir = root_path(transformer["model_dir"], "Transformer model directory")
        checksum_path = model_dir / "model.safetensors.sha256"
        configured_checksum = checksum_path.read_text(encoding="utf-8").strip() if checksum_path.exists() else None
        weights_path = model_dir / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"Restore transformer weights before preflight: {weights_path}")
        actual_checksum = sha256_file(weights_path)
        if configured_checksum and configured_checksum != actual_checksum:
            raise ValueError(f"Transformer checksum mismatch: {weights_path}")
        transformer_artifacts.append(
            {
                "slug": transformer["slug"],
                "model_dir": str(model_dir),
                "model_safetensors_sha256": actual_checksum,
                "config_sha256": sha256_file(model_dir / "config.json") if (model_dir / "config.json").exists() else None,
                "tokenizer_sha256": sha256_file(model_dir / "tokenizer.json") if (model_dir / "tokenizer.json").exists() else None,
            }
        )
    write_json(
        results_dir / "protocol_manifest.json",
        {
            "protocol": config["protocol"],
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "dataset": {
                **config["dataset"],
                "hashes": dataset_hashes,
            },
            "measurement": config["measurement"],
            "bert_server": config["bert_server"],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "environment": {
                "nvidia_smi": optional_command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,uuid,driver_version,memory.total",
                        "--format=csv,noheader",
                    ]
                ),
                "requirements_sha256": sha256_file(ROOT_DIR / "requirements.txt"),
                "requirements_vllm_sha256": sha256_file(ROOT_DIR / "requirements-vllm.txt"),
            },
            "models": config["models"],
            "transformers": config["transformers"],
            "transformer_artifacts": transformer_artifacts,
            "strategies": config["strategies"],
            "demonstration_seeds": config["comparison"]["demonstration_seeds"],
        },
    )


def run_preflight(config_path: Path, config: Dict[str, Any], results_dir: Path, seeds: Sequence[int]) -> None:
    paths = dataset_paths(config)
    materialized = ensure_materialized_dataset(
        dataset_name=str(paths["name"]),
        materialized_data_root=Path(paths["materialized_data_root"]),
        source_root=Path(paths["source_root"]),
    )
    examples = load_materialized_dataset(materialized.dataset_root)
    selections: Dict[str, object] = {"zero-shot": {"prompt_example_ids": []}, "demonstration_seeds": {}}
    for seed in seeds:
        one = select_wiki_prompt_examples(examples, seed=seed, count=1)
        three = select_wiki_prompt_examples(examples, seed=seed, count=3)
        validate_wiki_protocol(
            examples,
            three,
            seed=seed,
            expected_test_count=int(paths["expected_test_count"]),
        )
        if [item["id"] for item in one] != [three[0]["id"]]:
            raise ValueError(f"One-shot is not the A prefix of three-shot for seed {seed}")
        selections["demonstration_seeds"][str(seed)] = {
            "one-shot": [item["id"] for item in one],
            "three-shot": [item["id"] for item in three],
            "records": three,
        }
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "wiki_prompt_examples.json", selections)
    write_protocol_manifest(config_path, config, results_dir)
    print(f"Preflight passed: {len(examples['test'])} Wiki test examples, seeds={list(seeds)}", flush=True)


def runtime_env(config: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    scratch_root = local_scratch_root(env)
    hf_home = root_path(config["llm_server"].get("hf_home", ".cache/huggingface"), "Hugging Face cache")
    runtime_tmp = Path(env.get("TMPDIR") or scratch_root / "tmp").expanduser().resolve()
    xdg_cache_home = Path(env.get("XDG_CACHE_HOME") or scratch_root / "cache").expanduser().resolve()
    torch_home = Path(env.get("TORCH_HOME") or xdg_cache_home / "torch").expanduser().resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    xdg_cache_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "LOCAL_SCRATCH_ROOT": str(scratch_root),
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
            "HF_DATASETS_CACHE": str(hf_home / "datasets"),
            "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
            "XDG_CACHE_HOME": str(xdg_cache_home),
            "TORCH_HOME": str(torch_home),
            "TMPDIR": str(runtime_tmp),
        }
    )
    return env


def run_command(command: Sequence[str], log_path: Path, env: Dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=str(ROOT_DIR), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def read_log_tail(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return "<log missing>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def wait_for_json(url: str, timeout_seconds: float, process: Optional[subprocess.Popen[Any]] = None) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited before readiness with code {process.returncode}")
        try:
            with urlopen(Request(url, method="GET"), timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Server not ready at {url}: {last_error}")


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def summary_matches(path: Path, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        summary = load_json(path)
    except Exception:
        return False
    return (
        int(summary.get("examples_evaluated", 0)) == expected_count
        and summary.get("scoring_protocol") == SCORING_PROTOCOL
        and summary.get("latency_protocol") == LATENCY_PROTOCOL
    )


def transformer_summary_matches(
    path: Path,
    expected_count: int,
    device_profile: Dict[str, Any],
) -> bool:
    if not summary_matches(path, expected_count):
        return False
    summary = load_json(path)
    if summary.get("device_profile") != device_profile.get("slug"):
        return False
    metadata = summary.get("server_metadata", {})
    if not isinstance(metadata, dict):
        return False
    requested_device_type = str(device_profile.get("device", "")).split(":", 1)[0]
    actual_device_type = str(metadata.get("device", "")).split(":", 1)[0]
    if actual_device_type != requested_device_type:
        return False
    expected_dtype = str(device_profile.get("dtype", ""))
    if expected_dtype and not str(metadata.get("dtype", "")).endswith(expected_dtype):
        return False
    required_gpu_name = device_profile.get("required_gpu_name_substring")
    if required_gpu_name and str(required_gpu_name).lower() not in str(metadata.get("gpu_name", "")).lower():
        return False
    return True


def transformer_output_dir(results_dir: Path, slug: str, device_profile: str) -> Path:
    return results_dir / "transformers" / slug / device_profile


def llm_output_dir(results_dir: Path, slug: str, strategy: str, seed_label: str) -> Path:
    return results_dir / "llms" / slug / strategy.replace("-", "_") / f"seed_{seed_label}"


def run_transformer(
    args: argparse.Namespace,
    config: Dict[str, Any],
    transformer: Dict[str, Any],
    device_profile: Dict[str, Any],
    results_dir: Path,
) -> None:
    slug = str(transformer["slug"])
    profile_slug = str(device_profile["slug"])
    output_dir = transformer_output_dir(results_dir, slug, profile_slug)
    expected_count = args.max_examples or int(config["dataset"]["expected_test_count"])
    if transformer_summary_matches(output_dir / "summary.json", expected_count, device_profile) and not args.overwrite:
        print(f"[skip] transformer {slug}/{profile_slug}", flush=True)
        return

    server = dict(config["bert_server"])
    server.update(device_profile)
    server_log = results_dir / "logs" / f"{slug}.{profile_slug}.server.log"
    evaluator_log = results_dir / "logs" / f"{slug}.{profile_slug}.evaluation.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    model_dir = root_path(transformer["model_dir"], "Transformer model directory")
    command = [
        str(args.python),
        str(ROOT_DIR / "bert-training" / "serve_http.py"),
        "--model-dir", str(model_dir),
        "--host", str(server["host"]),
        "--port", str(server["port"]),
        "--max-length", str(server["max_length"]),
        "--device", str(server["device"]),
        "--dtype", str(server["dtype"]),
    ]
    env = runtime_env(config)
    with server_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=str(ROOT_DIR), env=env, stdout=handle, stderr=subprocess.STDOUT)
        try:
            metadata = wait_for_json(
                f"http://{server['host']}:{server['port']}/metadata",
                float(server["startup_timeout_seconds"]),
                process,
            )
            if Path(str(metadata.get("model_dir", ""))).resolve() != model_dir:
                raise ValueError(f"BERT server loaded an unexpected checkpoint for {slug}")
            requested_device_type = str(device_profile["device"]).split(":", 1)[0]
            actual_device_type = str(metadata.get("device", "")).split(":", 1)[0]
            if actual_device_type != requested_device_type:
                raise ValueError(
                    f"BERT device mismatch for {slug}/{profile_slug}: "
                    f"requested {requested_device_type}, got {metadata.get('device')}"
                )
            required_gpu_name = device_profile.get("required_gpu_name_substring")
            if required_gpu_name and str(required_gpu_name).lower() not in str(metadata.get("gpu_name", "")).lower():
                raise ValueError(
                    f"BERT GPU mismatch for {slug}/{profile_slug}: expected name containing "
                    f"{required_gpu_name!r}, got {metadata.get('gpu_name')!r}"
                )
            paths = dataset_paths(config)
            evaluator = [
                str(args.python),
                str(ROOT_DIR / "bert-training" / "evaluate_http.py"),
                "--api-base-url", f"http://{server['host']}:{server['port']}",
                "--model-slug", slug,
                "--device-profile", profile_slug,
                "--dataset-name", str(paths["name"]),
                "--dataset-root", str(paths["source_root"]),
                "--materialized-data-root", str(paths["materialized_data_root"]),
                "--output-dir", str(output_dir),
                "--expected-test-count", str(paths["expected_test_count"]),
                "--warmup-count", str(config["measurement"]["warmup_count"]),
                "--timeout", str(server["request_timeout_seconds"]),
            ]
            if args.max_examples is not None:
                evaluator.extend(["--max-examples", str(args.max_examples)])
            if args.overwrite:
                evaluator.append("--overwrite")
            elif not args.no_resume:
                evaluator.append("--resume")
            run_command(evaluator, evaluator_log, env)
        finally:
            stop_process(process)


def vllm_pid_file(slug: str, environment: Optional[Dict[str, str]] = None) -> Path:
    source = os.environ if environment is None else environment
    runtime_root = Path(
        source.get("DATS_RUNTIME_ROOT") or local_scratch_root(source) / "runtime"
    ).expanduser().resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root / f"vllm.{slug}.pid"


def vllm_env(config: Dict[str, Any], model: Dict[str, Any], results_dir: Path) -> Dict[str, str]:
    server = dict(config["llm_server"])
    server.update(model.get("server", {}))
    env = runtime_env(config)
    slug = str(model["slug"])
    default_venv_root = Path(env.get("DEFAULT_VENV_ROOT") or local_scratch_root(env) / ".venvs")
    vllm_venv = Path(env.get("VLLM_VENV") or default_venv_root / "vllm").expanduser().resolve()
    env.update(
        {
            "DEFAULT_VENV_ROOT": str(default_venv_root),
            "VLLM_VENV": str(vllm_venv),
            "MODEL": str(model["model"]),
            "SERVED_MODEL_NAME": str(model["model"]),
            "HOST": str(server["host"]),
            "PORT": str(server["port"]),
            "DTYPE": str(server["dtype"]),
            "MAX_MODEL_LEN": str(server["max_model_len"]),
            "MAX_NUM_SEQS": str(server["max_num_seqs"]),
            "GPU_MEMORY_UTILIZATION": str(server["gpu_memory_utilization"]),
            "TRUST_REMOTE_CODE": "1" if server.get("trust_remote_code") else "0",
            "ENABLE_PREFIX_CACHING": "1" if server.get("enable_prefix_caching") else "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "1" if server["use_flashinfer_sampler"] else "0",
            "REVISION": str(model.get("revision") or ""),
            "STARTUP_TIMEOUT_SECONDS": str(server["startup_timeout_seconds"]),
            "PID_FILE": str(vllm_pid_file(slug, env)),
            "LOG_FILE": str(results_dir / "logs" / f"{slug}.vllm.log"),
        }
    )
    return env


def cached_huggingface_revision(hf_home: Path, model_id: str, requested_revision: Optional[str]) -> Dict[str, object]:
    model_cache = hf_home / "hub" / ("models--" + model_id.replace("/", "--"))
    candidates = [requested_revision] if requested_revision else []
    if "main" not in candidates:
        candidates.append("main")
    resolved_commit: Optional[str] = None
    resolved_ref: Optional[str] = None
    for candidate in candidates:
        if not candidate:
            continue
        ref_path = model_cache / "refs" / candidate
        if ref_path.exists():
            resolved_commit = ref_path.read_text(encoding="utf-8").strip()
            resolved_ref = candidate
            break
        snapshot_path = model_cache / "snapshots" / candidate
        if snapshot_path.is_dir():
            resolved_commit = candidate
            resolved_ref = candidate
            break
    if resolved_commit is None:
        snapshots = sorted(path.name for path in (model_cache / "snapshots").glob("*") if path.is_dir())
        if len(snapshots) == 1:
            resolved_commit = snapshots[0]
            resolved_ref = "single_cached_snapshot"
    return {
        "model_id": model_id,
        "requested_revision": requested_revision,
        "resolved_commit": resolved_commit,
        "resolved_ref": resolved_ref,
        "cache_path": str(model_cache),
    }


def start_vllm(
    config: Dict[str, Any], model: Dict[str, Any], results_dir: Path
) -> Tuple[Dict[str, str], Dict[str, object]]:
    slug = str(model["slug"])
    prior_metadata_path = results_dir / "llms" / slug / "server_metadata.json"
    effective_model = dict(model)
    if prior_metadata_path.exists() and not effective_model.get("revision"):
        prior_metadata = load_json(prior_metadata_path)
        if prior_metadata.get("model") != model.get("model"):
            raise ValueError(f"Stored server metadata model mismatch for slug {slug}")
        prior_commit = prior_metadata.get("revision", {}).get("resolved_commit") if isinstance(prior_metadata.get("revision"), dict) else None
        if prior_commit:
            effective_model["revision"] = str(prior_commit)
    env = vllm_env(config, effective_model, results_dir)
    subprocess.run(["bash", str(ROOT_DIR / "start_vllm_server.sh")], cwd=str(ROOT_DIR), env=env, check=True)
    server = dict(config["llm_server"])
    server.update(model.get("server", {}))
    api_payload = wait_for_json(
        f"http://{server['host']}:{server['port']}/v1/models",
        30.0,
    )
    version_payload = wait_for_json(
        f"http://{server['host']}:{server['port']}/version",
        30.0,
    )
    revision = cached_huggingface_revision(Path(env["HF_HOME"]), str(model["model"]), effective_model.get("revision"))
    if not revision.get("resolved_commit"):
        stop_vllm(slug, env)
        raise RuntimeError(f"Could not resolve an immutable Hugging Face commit for {model['model']}")
    configured_vllm_venv = Path(env["VLLM_VENV"])
    vllm_version = server_version(version_payload) or installed_python_package_version(configured_vllm_venv, "vllm")
    if not vllm_version:
        stop_vllm(slug, env)
        raise RuntimeError("vLLM started but its installed version could not be recorded")
    metadata: Dict[str, object] = {
        "model_slug": slug,
        "model": model["model"],
        "revision": revision,
        "vllm_version": vllm_version,
        "served_version_response": version_payload,
        "vllm_environment": {
            "VLLM_USE_FLASHINFER_SAMPLER": env["VLLM_USE_FLASHINFER_SAMPLER"],
        },
        "served_models_response": api_payload,
        "server_configuration": server,
        "scoring_protocol": SCORING_PROTOCOL,
        "latency_protocol": LATENCY_PROTOCOL,
    }
    prior_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(prior_metadata_path, metadata)
    return env, metadata


def stop_vllm(slug: str, env: Dict[str, str]) -> None:
    stop_env = os.environ.copy()
    stop_env.update(env)
    stop_env["PID_FILE"] = str(vllm_pid_file(slug, env))
    subprocess.run(["bash", str(ROOT_DIR / "stop_vllm_server.sh")], cwd=str(ROOT_DIR), env=stop_env, check=False)


def run_llm_configuration(
    args: argparse.Namespace,
    config: Dict[str, Any],
    model: Dict[str, Any],
    strategy: Dict[str, Any],
    seed: int,
    seed_label: str,
    results_dir: Path,
    env: Dict[str, str],
    model_revision: str,
) -> None:
    slug = str(model["slug"])
    strategy_name = str(strategy["name"])
    output_dir = llm_output_dir(results_dir, slug, strategy_name, seed_label)
    expected_count = args.max_examples or int(config["dataset"]["expected_test_count"])
    if summary_matches(output_dir / "summary.json", expected_count) and not args.overwrite:
        print(f"[skip] llm {slug}/{strategy_name}/{seed_label}", flush=True)
        return

    paths = dataset_paths(config)
    server = dict(config["llm_server"])
    server.update(model.get("server", {}))
    command = [
        str(args.python),
        str(ROOT_DIR / "llm-ner" / "evaluate_http_exact.py"),
        "--model", str(model["model"]),
        "--model-slug", slug,
        "--model-revision", model_revision,
        "--api-base-url", f"http://{server['host']}:{server['port']}/v1",
        "--dataset-name", str(paths["name"]),
        "--dataset-root", str(paths["source_root"]),
        "--materialized-data-root", str(paths["materialized_data_root"]),
        "--prompt-strategy", strategy_name,
        "--seed", str(seed),
        "--output-dir", str(output_dir),
        "--expected-test-count", str(paths["expected_test_count"]),
        "--max-tokens", str(server["max_tokens"]),
        "--temperature", str(server["temperature"]),
        "--timeout", str(server["request_timeout_seconds"]),
        "--api-max-retries", str(config["measurement"]["automatic_retries"]),
        "--warmup-count", str(config["measurement"]["warmup_count"]),
    ]
    if server.get("disable_thinking"):
        command.append("--disable-thinking")
    if args.max_examples is not None:
        command.extend(["--max-examples", str(args.max_examples)])
    if args.overwrite:
        command.append("--overwrite")
    elif not args.no_resume:
        command.append("--resume")
    log_path = results_dir / "logs" / f"{slug}.{strategy_name}.{seed_label}.evaluation.log"
    run_command(command, log_path, env)


def expected_rows(
    transformers: Sequence[Dict[str, Any]],
    bert_devices: Sequence[Dict[str, Any]],
    models: Sequence[Dict[str, Any]],
    strategy_runs: Sequence[Tuple[Dict[str, Any], int, str]],
    results_dir: Path,
    expected_count: int,
    failures: Dict[str, str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    bert_devices_by_slug = {str(profile["slug"]): profile for profile in bert_devices}
    definitions: List[Tuple[str, str, str, str, str, Path]] = []
    for item in transformers:
        for device_profile in bert_devices:
            profile_slug = str(device_profile["slug"])
            definitions.append(
                (
                    "transformer_http",
                    str(item["slug"]),
                    "fine-tuned",
                    "fixed",
                    profile_slug,
                    transformer_output_dir(results_dir, str(item["slug"]), profile_slug),
                )
            )
    for model in models:
        for strategy, _, seed_label in strategy_runs:
            definitions.append(("llm_http", str(model["slug"]), str(strategy["name"]), seed_label, "gpu", llm_output_dir(results_dir, str(model["slug"]), str(strategy["name"]), seed_label)))

    for approach, slug, strategy, seed_label, device_profile, output_dir in definitions:
        summary_path = output_dir / "summary.json"
        key = f"{approach}/{slug}/{strategy}/{seed_label}/{device_profile}"
        summary = load_json(summary_path) if summary_path.exists() else {}
        count = int(summary.get("examples_evaluated", 0) or 0)
        protocol_match = summary.get("scoring_protocol") == SCORING_PROTOCOL and summary.get("latency_protocol") == LATENCY_PROTOCOL
        configuration_match = protocol_match
        if approach == "transformer_http" and summary:
            configuration_match = transformer_summary_matches(
                summary_path,
                expected_count,
                bert_devices_by_slug[device_profile],
            )
        if key in failures:
            status = "failed"
        elif count == expected_count and configuration_match and summary.get("status") == "completed":
            status = "completed"
        elif summary:
            status = "partial"
        else:
            status = "pending"
        client_latency = summary.get("latency", {}).get("client_e2e", {}) if isinstance(summary.get("latency"), dict) else {}
        server_metadata = summary.get("server_metadata", {}) if isinstance(summary.get("server_metadata"), dict) else {}
        rows.append(
            {
                "status": status,
                "approach": approach,
                "model_slug": slug,
                "device_profile": device_profile,
                "runtime_device": server_metadata.get("device", "cuda" if approach == "llm_http" else ""),
                "runtime_dtype": server_metadata.get("dtype", ""),
                "gpu_name": server_metadata.get("gpu_name", ""),
                "prompt_strategy": strategy,
                "seed": seed_label,
                "examples_evaluated": count or "",
                "precision": summary.get("precision", ""),
                "recall": summary.get("recall", ""),
                "f1": summary.get("f1", ""),
                "accuracy": summary.get("accuracy", ""),
                "tp": summary.get("tp", ""),
                "fp": summary.get("fp", ""),
                "fn": summary.get("fn", ""),
                "successful_requests": summary.get("successful_requests", ""),
                "failed_requests": summary.get("failed_requests", ""),
                "valid_json_requests": summary.get("valid_json_requests", ""),
                "schema_valid_requests": summary.get("schema_valid_requests", ""),
                "truncated_requests": summary.get("truncated_requests", ""),
                "parse_failures": json.dumps(summary.get("parse_failures", {}), sort_keys=True),
                "rejected_prediction_reasons": json.dumps(summary.get("rejected_prediction_reasons", {}), sort_keys=True),
                "finish_reasons": json.dumps(summary.get("finish_reasons", {}), sort_keys=True),
                "error_taxonomy": json.dumps(summary.get("error_taxonomy", {}), sort_keys=True),
                "latency_mean_ms": client_latency.get("mean_ms", "") if isinstance(client_latency, dict) else "",
                "latency_median_ms": client_latency.get("median_ms", "") if isinstance(client_latency, dict) else "",
                "latency_p90_ms": client_latency.get("p90_ms", "") if isinstance(client_latency, dict) else "",
                "latency_p95_ms": client_latency.get("p95_ms", "") if isinstance(client_latency, dict) else "",
                "latency_p99_ms": client_latency.get("p99_ms", "") if isinstance(client_latency, dict) else "",
                "requests_per_second": summary.get("requests_per_second", ""),
                "scoring_protocol": summary.get("scoring_protocol", ""),
                "latency_protocol": summary.get("latency_protocol", ""),
                "summary_path": str(summary_path),
                "predictions_path": str(output_dir / "predictions.jsonl"),
                "error": failures.get(key, ""),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latency_component_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    component_rows: List[Dict[str, object]] = []
    for row in rows:
        summary_path = Path(str(row["summary_path"]))
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        latency = summary.get("latency", {})
        if not isinstance(latency, dict):
            continue
        for component, values in sorted(latency.items()):
            if not isinstance(values, dict):
                continue
            component_rows.append(
                {
                    "status": row["status"],
                    "approach": row["approach"],
                    "model_slug": row["model_slug"],
                    "device_profile": row["device_profile"],
                    "prompt_strategy": row["prompt_strategy"],
                    "seed": row["seed"],
                    "component": component,
                    "count": values.get("count", ""),
                    "mean_ms": values.get("mean_ms", ""),
                    "median_ms": values.get("median_ms", ""),
                    "p90_ms": values.get("p90_ms", ""),
                    "p95_ms": values.get("p95_ms", ""),
                    "p99_ms": values.get("p99_ms", ""),
                    "min_ms": values.get("min_ms", ""),
                    "max_ms": values.get("max_ms", ""),
                }
            )
    return component_rows


def entity_type_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for row in rows:
        summary_path = Path(str(row["summary_path"]))
        if not summary_path.exists():
            continue
        per_entity_type = load_json(summary_path).get("per_entity_type", {})
        if not isinstance(per_entity_type, dict):
            continue
        for entity_type, metrics in sorted(per_entity_type.items()):
            if not isinstance(metrics, dict):
                continue
            output.append(
                {
                    "status": row["status"],
                    "approach": row["approach"],
                    "model_slug": row["model_slug"],
                    "device_profile": row["device_profile"],
                    "prompt_strategy": row["prompt_strategy"],
                    "seed": row["seed"],
                    "entity_type": entity_type,
                    "tp": metrics.get("tp", ""),
                    "fp": metrics.get("fp", ""),
                    "fn": metrics.get("fn", ""),
                    "precision": metrics.get("precision", ""),
                    "recall": metrics.get("recall", ""),
                    "f1": metrics.get("f1", ""),
                }
            )
    return output


def numeric_mean_std(rows: Sequence[Dict[str, object]], key: str) -> Tuple[object, object]:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return "", ""
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else ""


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["approach"]),
            str(row["model_slug"]),
            str(row.get("device_profile", "")),
            str(row["prompt_strategy"]),
        )
        grouped.setdefault(key, []).append(row)
    aggregates: List[Dict[str, object]] = []
    for (approach, model_slug, device_profile, strategy), group in sorted(grouped.items()):
        completed = [row for row in group if row["status"] == "completed"]
        aggregate: Dict[str, object] = {
            "status": "completed" if len(completed) == len(group) else "partial" if completed else "pending",
            "approach": approach,
            "model_slug": model_slug,
            "device_profile": device_profile,
            "prompt_strategy": strategy,
            "runs_expected": len(group),
            "runs_completed": len(completed),
        }
        for metric in ("precision", "recall", "f1", "accuracy", "latency_median_ms", "latency_p95_ms"):
            mean, std = numeric_mean_std(completed, metric)
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
        aggregates.append(aggregate)
    return aggregates


def latency_stat(row: Dict[str, object], component: str, statistic: str) -> object:
    summary_path = Path(str(row["summary_path"]))
    if not summary_path.exists():
        return ""
    latency = load_json(summary_path).get("latency", {})
    if not isinstance(latency, dict):
        return ""
    values = latency.get(component, {})
    return values.get(statistic, "") if isinstance(values, dict) else ""


def speedup(cpu_value: object, gpu_value: object) -> object:
    if cpu_value in (None, "") or gpu_value in (None, ""):
        return ""
    denominator = float(gpu_value)
    return float(cpu_value) / denominator if denominator > 0 else ""


def bert_device_speedup_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_model: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        if row.get("approach") != "transformer_http" or row.get("status") != "completed":
            continue
        by_model.setdefault(str(row["model_slug"]), {})[str(row["device_profile"])] = row

    output: List[Dict[str, object]] = []
    for model_slug, device_rows in sorted(by_model.items()):
        cpu = device_rows.get("cpu")
        gpu = device_rows.get("gpu")
        if cpu is None or gpu is None:
            continue
        cpu_model_median = latency_stat(cpu, "model_execution", "median_ms")
        gpu_model_median = latency_stat(gpu, "model_execution", "median_ms")
        cpu_model_p95 = latency_stat(cpu, "model_execution", "p95_ms")
        gpu_model_p95 = latency_stat(gpu, "model_execution", "p95_ms")
        output.append(
            {
                "model_slug": model_slug,
                "cpu_runtime_device": cpu.get("runtime_device", ""),
                "gpu_runtime_device": gpu.get("runtime_device", ""),
                "gpu_name": gpu.get("gpu_name", ""),
                "dtype": gpu.get("runtime_dtype", ""),
                "cpu_f1": cpu.get("f1", ""),
                "gpu_f1": gpu.get("f1", ""),
                "gpu_minus_cpu_f1": float(gpu["f1"]) - float(cpu["f1"]),
                "cpu_client_mean_ms": cpu.get("latency_mean_ms", ""),
                "gpu_client_mean_ms": gpu.get("latency_mean_ms", ""),
                "client_mean_speedup_cpu_over_gpu": speedup(cpu.get("latency_mean_ms"), gpu.get("latency_mean_ms")),
                "cpu_client_median_ms": cpu.get("latency_median_ms", ""),
                "gpu_client_median_ms": gpu.get("latency_median_ms", ""),
                "client_median_speedup_cpu_over_gpu": speedup(cpu.get("latency_median_ms"), gpu.get("latency_median_ms")),
                "cpu_client_p95_ms": cpu.get("latency_p95_ms", ""),
                "gpu_client_p95_ms": gpu.get("latency_p95_ms", ""),
                "client_p95_speedup_cpu_over_gpu": speedup(cpu.get("latency_p95_ms"), gpu.get("latency_p95_ms")),
                "cpu_model_execution_median_ms": cpu_model_median,
                "gpu_model_execution_median_ms": gpu_model_median,
                "model_execution_median_speedup_cpu_over_gpu": speedup(cpu_model_median, gpu_model_median),
                "cpu_model_execution_p95_ms": cpu_model_p95,
                "gpu_model_execution_p95_ms": gpu_model_p95,
                "model_execution_p95_speedup_cpu_over_gpu": speedup(cpu_model_p95, gpu_model_p95),
                "cpu_requests_per_second": cpu.get("requests_per_second", ""),
                "gpu_requests_per_second": gpu.get("requests_per_second", ""),
                "throughput_gain_gpu_over_cpu": speedup(gpu.get("requests_per_second"), cpu.get("requests_per_second")),
            }
        )
    return output


def format_value(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, rows: Sequence[Dict[str, object]], aggregates: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Wiki NER HTTP Exact-Span Matrix",
        "",
        f"Scoring protocol: `{SCORING_PROTOCOL}`. Latency protocol: `{LATENCY_PROTOCOL}`.",
        "",
        "## Quality",
        "",
        "| status | approach | model | device | strategy | seed | n | precision | recall | F1 | accuracy |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                [
                    str(row["status"]), str(row["approach"]), str(row["model_slug"]), str(row["device_profile"]),
                    str(row["prompt_strategy"]), str(row["seed"]), format_value(row["examples_evaluated"]),
                    format_value(row["precision"]), format_value(row["recall"]), format_value(row["f1"]),
                    format_value(row["accuracy"]),
                ]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "## Output and request reliability",
            "",
            "| status | model | device | strategy | seed | successful | failed | valid JSON | strict-schema valid | truncated |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| " + " | ".join(
                [
                    str(row["status"]), str(row["model_slug"]), str(row["device_profile"]),
                    str(row["prompt_strategy"]), str(row["seed"]),
                    format_value(row["successful_requests"]), format_value(row["failed_requests"]),
                    format_value(row["valid_json_requests"]), format_value(row["schema_valid_requests"]),
                    format_value(row["truncated_requests"]),
                ]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "## Primary localhost HTTP latency",
            "",
            "| status | model | device | strategy | seed | mean ms | median ms | p90 ms | p95 ms | p99 ms | requests/s |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| " + " | ".join(
                [
                    str(row["status"]), str(row["model_slug"]), str(row["device_profile"]),
                    str(row["prompt_strategy"]), str(row["seed"]),
                    format_value(row["latency_mean_ms"]), format_value(row["latency_median_ms"]),
                    format_value(row["latency_p90_ms"]), format_value(row["latency_p95_ms"]),
                    format_value(row["latency_p99_ms"]), format_value(row["requests_per_second"]),
                ]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "## Prompt-seed aggregates",
            "",
            "| status | approach | model | device | strategy | runs | F1 mean±sd | median latency mean±sd ms |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        f1 = format_value(row["f1_mean"])
        if row["f1_std"] != "":
            f1 += f"±{format_value(row['f1_std'])}"
        latency = format_value(row["latency_median_ms_mean"])
        if row["latency_median_ms_std"] != "":
            latency += f"±{format_value(row['latency_median_ms_std'])}"
        lines.append(
            f"| {row['status']} | {row['approach']} | {row['model_slug']} | {row['device_profile']} | "
            f"{row['prompt_strategy']} | "
            f"{row['runs_completed']}/{row['runs_expected']} | {f1} | {latency} |"
        )
    lines.extend(
        [
            "",
            "## BERT CPU/GPU device ablation",
            "",
            "Speedup is CPU latency divided by GPU latency; values above 1 mean GPU is faster.",
            "",
            "| model | GPU | dtype | F1 Δ (GPU−CPU) | CPU median ms | GPU median ms | E2E speedup | model-execution speedup | throughput gain |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bert_device_speedup_rows(rows):
        lines.append(
            "| " + " | ".join(
                [
                    str(row["model_slug"]),
                    str(row["gpu_name"]),
                    str(row["dtype"]),
                    format_value(row["gpu_minus_cpu_f1"]),
                    format_value(row["cpu_client_median_ms"]),
                    format_value(row["gpu_client_median_ms"]),
                    format_value(row["client_median_speedup_cpu_over_gpu"]),
                    format_value(row["model_execution_median_speedup_cpu_over_gpu"]),
                    format_value(row["throughput_gain_gpu_over_cpu"]),
                ]
            ) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(
    results_dir: Path,
    transformers: Sequence[Dict[str, Any]],
    bert_devices: Sequence[Dict[str, Any]],
    models: Sequence[Dict[str, Any]],
    strategy_runs: Sequence[Tuple[Dict[str, Any], int, str]],
    expected_count: int,
    failures: Dict[str, str],
) -> List[Dict[str, object]]:
    rows = expected_rows(transformers, bert_devices, models, strategy_runs, results_dir, expected_count, failures)
    aggregates = aggregate_rows(rows)
    write_csv(results_dir / "matrix_results.csv", rows)
    write_csv(results_dir / "matrix_aggregates.csv", aggregates)
    write_csv(results_dir / "latency_components.csv", latency_component_rows(rows))
    write_csv(results_dir / "bert_cpu_gpu_speedups.csv", bert_device_speedup_rows(rows))
    write_csv(results_dir / "entity_type_metrics.csv", entity_type_rows(rows))
    write_markdown(results_dir / "matrix_results.md", rows, aggregates)
    write_json(
        results_dir / "matrix_status.json",
        {
            "counts": dict(Counter(str(row["status"]) for row in rows)),
            "expected_rows": len(rows),
            "failures": failures,
            "scoring_protocol": SCORING_PROTOCOL,
            "latency_protocol": LATENCY_PROTOCOL,
        },
    )
    return rows


def main() -> None:
    args = parse_args()
    config_path = root_path(args.config, "Matrix config")
    config = load_json(config_path)
    validate_config(config)
    results_dir = root_path(args.results_dir or config["comparison"]["output_dir"], "Results directory")
    results_dir.mkdir(parents=True, exist_ok=True)

    transformer_configs = select(config["transformers"], args.transformers, "transformer")
    bert_device_configs = select(config["bert_server"]["device_profiles"], args.bert_devices, "BERT device")
    model_configs = select(config["models"], args.models, "LLM")
    strategy_configs = select(config["strategies"], args.strategies, "strategy")
    configured_seeds = [int(seed) for seed in config["comparison"]["demonstration_seeds"]]
    seeds = [int(seed) for seed in (args.seeds or configured_seeds)]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("At least one unique demonstration seed is required")
    strategy_runs = strategy_seed_runs(strategy_configs, seeds, int(config["comparison"]["zero_shot_seed"]))

    run_preflight(config_path, config, results_dir, seeds)
    expected_count = args.max_examples or int(config["dataset"]["expected_test_count"])
    failures: Dict[str, str] = {}
    if args.preflight_only:
        write_reports(results_dir, transformer_configs, bert_device_configs, model_configs, strategy_runs, expected_count, failures)
        return
    if args.report_only:
        write_reports(results_dir, transformer_configs, bert_device_configs, model_configs, strategy_runs, expected_count, failures)
        return

    if not args.skip_transformers:
        for transformer in transformer_configs:
            slug = str(transformer["slug"])
            for device_profile in bert_device_configs:
                profile_slug = str(device_profile["slug"])
                key = f"transformer_http/{slug}/fine-tuned/fixed/{profile_slug}"
                try:
                    run_transformer(args, config, transformer, device_profile, results_dir)
                except Exception as exc:
                    failures[key] = f"{type(exc).__name__}: {exc}"
                    print(f"[error] {key}: {failures[key]}", file=sys.stderr, flush=True)
                finally:
                    write_reports(
                        results_dir,
                        transformer_configs,
                        bert_device_configs,
                        model_configs,
                        strategy_runs,
                        expected_count,
                        failures,
                    )

    if not args.skip_llms:
        for model in model_configs:
            slug = str(model["slug"])
            needed = [
                (strategy, seed, seed_label)
                for strategy, seed, seed_label in strategy_runs
                if not summary_matches(
                    llm_output_dir(results_dir, slug, str(strategy["name"]), seed_label) / "summary.json",
                    expected_count,
                ) or args.overwrite
            ]
            if not needed:
                print(f"[skip] all LLM configurations complete for {slug}", flush=True)
                continue
            env: Optional[Dict[str, str]] = None
            try:
                env, server_metadata = start_vllm(config, model, results_dir)
                revision_payload = server_metadata.get("revision", {})
                model_revision = str(revision_payload.get("resolved_commit")) if isinstance(revision_payload, dict) else ""
                for strategy, seed, seed_label in needed:
                    key = f"llm_http/{slug}/{strategy['name']}/{seed_label}/gpu"
                    try:
                        run_llm_configuration(
                            args, config, model, strategy, seed, seed_label, results_dir, env, model_revision
                        )
                    except Exception as exc:
                        log_path = results_dir / "logs" / f"{slug}.{strategy['name']}.{seed_label}.evaluation.log"
                        failures[key] = f"{type(exc).__name__}: {exc}\n{read_log_tail(log_path)}"
                        print(f"[error] {key}: {failures[key]}", file=sys.stderr, flush=True)
                    finally:
                        write_reports(
                            results_dir,
                            transformer_configs,
                            bert_device_configs,
                            model_configs,
                            strategy_runs,
                            expected_count,
                            failures,
                        )
            except Exception as exc:
                for strategy, _, seed_label in needed:
                    key = f"llm_http/{slug}/{strategy['name']}/{seed_label}/gpu"
                    failures.setdefault(key, f"server_start_failed: {type(exc).__name__}: {exc}")
                print(f"[error] vLLM server {slug}: {exc}", file=sys.stderr, flush=True)
            finally:
                if env is not None:
                    stop_vllm(slug, env)
                write_reports(
                    results_dir,
                    transformer_configs,
                    bert_device_configs,
                    model_configs,
                    strategy_runs,
                    expected_count,
                    failures,
                )

    rows = write_reports(
        results_dir,
        transformer_configs,
        bert_device_configs,
        model_configs,
        strategy_runs,
        expected_count,
        failures,
    )
    incomplete = [
        row
        for row in rows
        if row["status"] != "completed"
        and not (args.skip_transformers and row["approach"] == "transformer_http")
        and not (args.skip_llms and row["approach"] == "llm_http")
    ]
    if incomplete and not args.allow_failures:
        raise RuntimeError(f"{len(incomplete)} matrix rows are incomplete; see {results_dir / 'matrix_status.json'}")


if __name__ == "__main__":
    main()
