from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT_DIR / "comparison-wrapper" / "run_http_exact_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_http_exact_matrix", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class HttpExactMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT_DIR / "comparison-wrapper" / "wiki_http_exact_matrix.json").read_text())

    def test_executable_matrix_has_81_rows(self) -> None:
        runs = RUNNER.strategy_seed_runs(
            self.config["strategies"],
            self.config["comparison"]["demonstration_seeds"],
            self.config["comparison"]["zero_shot_seed"],
        )
        self.assertEqual(len(runs), 7)
        bert_rows = len(self.config["transformers"]) * len(self.config["bert_server"]["device_profiles"])
        self.assertEqual(bert_rows + len(self.config["models"]) * len(runs), 81)

    def test_executable_matrix_retains_trendyol(self) -> None:
        slugs = {model["slug"] for model in self.config["models"]}
        self.assertIn("trendyol-llm-7b-chat-v4.1.0", slugs)

    def test_bert_device_profiles_isolate_hardware(self) -> None:
        profiles = self.config["bert_server"]["device_profiles"]
        self.assertEqual([profile["device"] for profile in profiles], ["cpu", "cuda"])
        self.assertEqual({profile["dtype"] for profile in profiles}, {"float32"})
        self.assertEqual(profiles[1]["required_gpu_name_substring"], "A100")

    def test_zero_shot_runs_once(self) -> None:
        runs = RUNNER.strategy_seed_runs(self.config["strategies"], [42, 1771, 2401], 42)
        zero = [run for run in runs if run[0]["name"] == "zero-shot"]
        self.assertEqual(zero, [(self.config["strategies"][0], 42, "fixed")])

    def test_config_protocol_is_valid(self) -> None:
        RUNNER.validate_config(self.config)

    def test_summary_match_rejects_stale_protocol(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "examples_evaluated": 1000,
                        "scoring_protocol": "legacy_overlap_bio_v1",
                        "latency_protocol": self.config["protocol"]["latency"],
                    }
                )
            )
            self.assertFalse(RUNNER.summary_matches(path, 1000))

    def test_transformer_summary_match_rejects_wrong_runtime_device(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "device_profile": "gpu",
                        "examples_evaluated": 1000,
                        "scoring_protocol": self.config["protocol"]["scoring"],
                        "latency_protocol": self.config["protocol"]["latency"],
                        "server_metadata": {
                            "device": "cpu",
                            "dtype": "torch.float32",
                            "gpu_name": None,
                        },
                    }
                )
            )
            gpu_profile = self.config["bert_server"]["device_profiles"][1]
            self.assertFalse(RUNNER.transformer_summary_matches(path, 1000, gpu_profile))

    def test_single_fixed_run_has_no_artificial_standard_deviation(self) -> None:
        aggregate = RUNNER.aggregate_rows(
            [
                {
                    "status": "completed",
                    "approach": "llm_http",
                    "model_slug": "model",
                    "device_profile": "gpu",
                    "prompt_strategy": "zero-shot",
                    "f1": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                    "accuracy": 0.5,
                    "latency_median_ms": 10.0,
                    "latency_p95_ms": 12.0,
                }
            ]
        )[0]
        self.assertEqual(aggregate["runs_completed"], 1)
        self.assertEqual(aggregate["f1_std"], "")

    def test_three_seed_aggregate_uses_sample_standard_deviation(self) -> None:
        rows = []
        for seed, f1 in ((42, 0.2), (1771, 0.4), (2401, 0.6)):
            rows.append(
                {
                    "status": "completed",
                    "approach": "llm_http",
                    "model_slug": "model",
                    "device_profile": "gpu",
                    "prompt_strategy": "one-shot",
                    "seed": str(seed),
                    "f1": f1,
                    "precision": f1,
                    "recall": f1,
                    "accuracy": f1,
                    "latency_median_ms": 10.0,
                    "latency_p95_ms": 12.0,
                }
            )
        aggregate = RUNNER.aggregate_rows(rows)[0]
        self.assertEqual(aggregate["runs_completed"], 3)
        self.assertTrue(math.isclose(aggregate["f1_mean"], 0.4))
        self.assertTrue(math.isclose(aggregate["f1_std"], 0.2))

    def test_missing_result_rows_remain_pending(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as directory:
            strategy = {"name": "one-shot", "few_shot_count": 1, "seed_mode": "demonstration"}
            runs = [(strategy, seed, str(seed)) for seed in (42, 1771, 2401)]
            rows = RUNNER.expected_rows(
                transformers=[],
                bert_devices=self.config["bert_server"]["device_profiles"],
                models=[{"slug": "model", "model": "example/model"}],
                strategy_runs=runs,
                results_dir=Path(directory),
                expected_count=1000,
                failures={},
            )
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["status"] == "pending" for row in rows))

    def test_output_guard_rejects_path_outside_bundle(self) -> None:
        with self.assertRaisesRegex(ValueError, "must stay inside"):
            RUNNER.ensure_inside_root(ROOT_DIR.parent / "outside", "Test output")

    def test_runtime_env_uses_scratch_and_honors_tmpdir_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "scratch"
            tmp_override = Path(directory) / "custom-tmp"
            with mock.patch.dict(
                RUNNER.os.environ,
                {"LOCAL_SCRATCH_ROOT": str(scratch), "TMPDIR": str(tmp_override)},
                clear=True,
            ):
                env = RUNNER.runtime_env(self.config)
            self.assertEqual(env["TMPDIR"], str(tmp_override.resolve()))
            self.assertEqual(env["XDG_CACHE_HOME"], str((scratch / "cache").resolve()))
            self.assertEqual(env["TORCH_HOME"], str((scratch / "cache" / "torch").resolve()))
            self.assertTrue(Path(env["HF_HOME"]).is_relative_to(ROOT_DIR))

    def test_package_version_probe_reads_dist_info_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory)
            dist_info = venv / "lib" / "python3.10" / "site-packages" / "vllm-0.25.0.dist-info"
            dist_info.mkdir(parents=True)
            (dist_info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: vllm\nVersion: 0.25.0\n",
                encoding="utf-8",
            )
            with mock.patch.object(RUNNER.subprocess, "run") as run:
                version = RUNNER.installed_python_package_version(venv, "vllm")
            self.assertEqual(version, "0.25.0")
            run.assert_not_called()

    def test_server_version_uses_vllm_version_endpoint_payload(self) -> None:
        self.assertEqual(RUNNER.server_version({"version": "0.25.0"}), "0.25.0")
        self.assertIsNone(RUNNER.server_version({}))

    def test_vllm_env_disables_flashinfer_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "scratch"
            model = self.config["models"][0]
            results = ROOT_DIR / "comparison-wrapper" / "results" / "test"
            with mock.patch.dict(
                RUNNER.os.environ,
                {"LOCAL_SCRATCH_ROOT": str(scratch)},
                clear=True,
            ):
                env = RUNNER.vllm_env(self.config, model, results)
            self.assertEqual(env["VLLM_USE_FLASHINFER_SAMPLER"], "0")
            self.assertEqual(env["VLLM_VENV"], str((scratch / ".venvs" / "vllm").resolve()))
            self.assertEqual(
                env["PID_FILE"],
                str((scratch / "runtime" / "vllm.qwen3-0.6b.pid").resolve()),
            )
            self.assertTrue(Path(env["LOG_FILE"]).is_relative_to(ROOT_DIR))

    def test_vllm_start_does_not_probe_cli_help_per_model(self) -> None:
        script = (ROOT_DIR / "start_vllm_server.sh").read_text(encoding="utf-8")
        self.assertNotIn("serve --help", script)
        self.assertIn("CMD+=(--no-enable-prefix-caching)", script)

    def test_transformer_output_directories_separate_cpu_and_gpu(self) -> None:
        results = ROOT_DIR / "comparison-wrapper" / "results" / "test"
        cpu = RUNNER.transformer_output_dir(results, "model", "cpu")
        gpu = RUNNER.transformer_output_dir(results, "model", "gpu")
        self.assertNotEqual(cpu, gpu)
        self.assertEqual(cpu.name, "cpu")
        self.assertEqual(gpu.name, "gpu")

    def test_bert_speedup_is_cpu_latency_divided_by_gpu_latency(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as directory:
            root = Path(directory)
            rows = []
            for profile, latency in (("cpu", 20.0), ("gpu", 5.0)):
                output = root / profile
                output.mkdir()
                summary_path = output / "summary.json"
                summary_path.write_text(
                    json.dumps({"latency": {"model_execution": {"median_ms": latency, "p95_ms": latency}}})
                )
                rows.append(
                    {
                        "status": "completed",
                        "approach": "transformer_http",
                        "model_slug": "model",
                        "device_profile": profile,
                        "runtime_device": "cpu" if profile == "cpu" else "cuda:0",
                        "runtime_dtype": "torch.float32",
                        "gpu_name": "A100" if profile == "gpu" else "",
                        "f1": 0.8,
                        "latency_mean_ms": latency,
                        "latency_median_ms": latency,
                        "latency_p95_ms": latency,
                        "requests_per_second": 1000.0 / latency,
                        "summary_path": str(summary_path),
                    }
                )
            speedup = RUNNER.bert_device_speedup_rows(rows)[0]
            self.assertEqual(speedup["client_median_speedup_cpu_over_gpu"], 4.0)
            self.assertEqual(speedup["model_execution_median_speedup_cpu_over_gpu"], 4.0)
            self.assertEqual(speedup["gpu_minus_cpu_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
