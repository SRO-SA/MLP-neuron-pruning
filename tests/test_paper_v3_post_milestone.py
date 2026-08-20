import json
import os
import tempfile
import unittest
import subprocess
import sys

from scripts.build_paper_v3_release_manifest import index_paths
from scripts.summarize_paper_v3_downstream import paired_bootstrap_accuracy


class PostMilestoneDryRunTests(unittest.TestCase):
    def test_paired_accuracy_uses_matched_examples(self):
        first = {"task": [1.0, 1.0, 0.0, 1.0]}
        second = {"task": [0.0, 0.0, 0.0, 0.0]}
        result = paired_bootstrap_accuracy(first, second, n_resamples=1000, seed=7)
        self.assertAlmostEqual(result["task"]["difference"], 0.75)
        self.assertGreaterEqual(result["task"]["ci95_lower"], 0.0)

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
