import json
import os
import tempfile
import unittest
import subprocess
import sys

from scripts.build_paper_v3_release_manifest import index_paths
from scripts.summarize_paper_v3_downstream import (
    flatten_paired_comparison, paired_bootstrap_accuracy,
)
from scripts.summarize_paper_v3_systems import collect as collect_systems


class PostMilestoneDryRunTests(unittest.TestCase):
    def test_direct_reporting_entrypoints_can_import_repo_modules(self):
        for name in (
            "summarize_moe_aggregation_frontier.py",
            "compare_heapr_downstream.py",
        ):
            completed = subprocess.run(
                [sys.executable, os.path.join("scripts", name), "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_paired_accuracy_uses_matched_examples(self):
        first = {"task": [1.0, 1.0, 0.0, 1.0]}
        second = {"task": [0.0, 0.0, 0.0, 0.0]}
        result = paired_bootstrap_accuracy(first, second, n_resamples=1000, seed=7)
        self.assertAlmostEqual(result["task"]["difference"], 0.75)
        self.assertGreaterEqual(result["task"]["ci95_lower"], 0.0)

    def test_flattened_paired_comparison_marks_direction(self):
        rows = flatten_paired_comparison("ellipsoid", "activation", "selector", {
            "piqa": {"difference": 0.02, "ci95_lower": 0.01,
                     "ci95_upper": 0.03, "n_examples": 100},
        })
        self.assertTrue(rows[0]["significant_95pct"])
        self.assertEqual(rows[0]["favored_label_if_significant"], "ellipsoid")

    def test_systems_summary_computes_real_baseline_improvements(self):
        with tempfile.TemporaryDirectory() as root:
            for label, storage, latency, throughput in (
                ("baseline_unpruned", 1000, 10.0, 100.0),
                ("target6", 900, 9.0, 110.0),
            ):
                directory = os.path.join(root, label)
                os.makedirs(directory)
                payload = {
                    "label": label, "dtype": "bfloat16",
                    "checkpoint_storage_bytes": storage,
                    "load_time_seconds": latency,
                    "successful_load": True,
                    "after_load_allocated_bytes_total": storage,
                    "peak_inference_allocated_bytes_total": storage,
                    "nvidia_smi": "gpu", "torch_version": "x",
                    "cuda_runtime_version": "x", "transformers_version": "x",
                    "inference_engine": "test",
                    "reduced_intermediate_dimensions_executed": True,
                    "runtime_moe_execution_evidence": {
                        "all_packed_moe_layers_executed": True,
                    },
                    "cases": [{
                        "batch_size": 1, "prompt_length_tokens": 128,
                        "prefill_latency_median_ms": latency,
                        "prefill_latency_stdev_ms": 0.1,
                        "prefill_tokens_per_second_median": throughput,
                        "decode_latency_per_token_median_ms": latency,
                        "decode_latency_per_token_stdev_ms": 0.1,
                        "decode_tokens_per_second_median": throughput,
                        "warmup_repetitions": 3, "timed_repetitions": 10,
                    }],
                }
                with open(os.path.join(directory, "systems.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump(payload, handle)
            rows, _ = collect_systems(root)
            target = next(row for row in rows if row["label"] == "target6")
            self.assertAlmostEqual(
                target["prefill_latency_reduction_vs_baseline_pct"], 10.0
            )
            self.assertAlmostEqual(
                target["prefill_throughput_gain_vs_baseline_pct"], 10.0
            )

    def test_release_index_hashes_without_editing_source(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "evidence.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("a,b\n1,2\n")
            with open(path, "rb") as handle:
                before = handle.read()
            rows = index_paths([root])
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["sha256"]), 64)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), before)

    def test_gpu_shell_entrypoints_expose_dry_run(self):
        for name in (
            "run_paper_v3_downstream.sh", "run_paper_v3_systems.sh",
            "run_heapr_matched_baseline.sh", "run_paper_v3_method_cost.sh",
        ):
            with open(os.path.join("scripts", name), encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn("DRY_RUN", text, name)
            self.assertIn("refusing", text, name)
        for name in ("run_paper_v3_downstream.sh", "run_paper_v3_systems.sh"):
            with open(os.path.join("scripts", name), encoding="utf-8") as handle:
                self.assertIn("TOKENIZER_AUDIT", handle.read(), name)

    def test_heapr_patch_changes_reporting_not_pruning_call(self):
        with tempfile.TemporaryDirectory() as root:
            main_path = os.path.join(root, "main.py")
            record = os.path.join(root, "patch.json")
            with open(main_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "import os\n"
                    "def main():\n"
                    "    model.eval()\n"
                    "    pruning_global(model, cali_data, config, args, logger)\n"
                    "    from lm_eval.utils import make_table\n"
                    "    results = lm_eval.simple_evaluate(\n"
                    "        hflm, tasks=tasks, batch_size=\"auto\", max_batch_size=256\n"
                    "    )\n"
                )
            subprocess.run([
                sys.executable, "scripts/patch_heapr_matched_reporting.py",
                "--repo-dir", root, "--patch-record", record,
            ], check=True, capture_output=True, text=True)
            with open(main_path, encoding="utf-8") as handle:
                patched = handle.read()
            self.assertEqual(
                patched.count("pruning_global(model, cali_data, config, args, logger)"),
                1,
            )
            self.assertIn("log_samples=True", patched)
            self.assertIn("parameters_before", patched)


if __name__ == "__main__":
    unittest.main()
