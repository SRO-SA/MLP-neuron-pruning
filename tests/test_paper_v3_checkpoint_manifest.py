import unittest
import subprocess
import sys
import csv
import json
import os
import tempfile
from unittest.mock import patch

from scripts.generate_paper_v3_checkpoint_manifest import (
    select_additional_target6_downnorm_spec, select_checkpoint_specs,
)
from src.experiment_provenance import file_sha256


class PaperV3CheckpointManifestTests(unittest.TestCase):
    def test_direct_script_entrypoint_can_import_src(self):
        completed = subprocess.run(
            [sys.executable, "scripts/generate_paper_v3_checkpoint_manifest.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @patch("scripts.generate_paper_v3_checkpoint_manifest.os.path.isfile", return_value=True)
    @patch("scripts.generate_paper_v3_checkpoint_manifest.file_sha256", return_value="hash")
    def test_selects_only_primary_frozen_plans(self, _hash, _isfile):
        rows = []
        for target, source in ((4, "frozen_2_4_6"), (6, "frozen_2_4_6"),
                               (8, "target8_rmsnorm_primary_n1024_v1")):
            rows.append({
                "requested_pct": str(target), "allocation_source": "rmsnorm_bound",
                "ranking_source": "rmsnorm_ellipsoid_bound",
                "expert_aggregation": "p95", "source_group": source,
                "pruning_plan_path": f"target{target}.json",
                "pruning_plan_sha256": "hash", "actual_pct": str(target + 0.1),
                "removed_layer_channels": str(target * 100),
                "removed_expert_neurons": str(target * 12800),
            })
        specs = select_checkpoint_specs(rows, "toy/model")
        self.assertEqual([spec["target_pct"] for spec in specs], [0.0, 4.0, 6.0, 8.0])
        self.assertEqual(specs[2]["ranking_source"], "rmsnorm_ellipsoid_bound")

    @patch("scripts.generate_paper_v3_checkpoint_manifest.os.path.isfile", return_value=True)
    @patch("scripts.generate_paper_v3_checkpoint_manifest.file_sha256", return_value="hash")
    def test_adds_target2_and_available_target6_comparators(self, _hash, _isfile):
        rows = []
        for target, source in ((2, "frozen_2_4_6"), (4, "frozen_2_4_6"),
                               (6, "frozen_2_4_6"),
                               (8, "target8_rmsnorm_primary_n1024_v1")):
            rows.append({
                "requested_pct": str(target), "allocation_source": "rmsnorm_bound",
                "ranking_source": "rmsnorm_ellipsoid_bound",
                "expert_aggregation": "p95", "source_group": source,
                "pruning_plan_path": f"target{target}.json",
                "pruning_plan_sha256": "hash", "actual_pct": str(target),
                "removed_layer_channels": str(target * 100),
                "removed_expert_neurons": str(target * 12800),
            })
        for ranking in ("rmsnorm_bound", "activation_score", "down_norm"):
            rows.append({
                "requested_pct": "6", "allocation_source": "rmsnorm_bound",
                "ranking_source": ranking, "expert_aggregation": "p95",
                "source_group": "frozen_2_4_6",
                "pruning_plan_path": f"target6_{ranking}.json",
                "pruning_plan_sha256": "hash", "actual_pct": "6.2",
                "removed_layer_channels": "2288",
                "removed_expert_neurons": "292864",
            })
        specs = select_checkpoint_specs(
            rows, "toy/model", include_target6_comparators=True,
            include_target6_downnorm_if_available=True,
            include_target2_primary=True,
        )
        labels = {spec["label"] for spec in specs}
        self.assertIn("rmsnorm_alloc__ellipsoid_rank__p95__target2", labels)
        self.assertIn("rmsnorm_alloc__downnorm_rank__p95__target6", labels)
        target2 = next(spec for spec in specs if spec.get("target_pct") == 2.0)
        self.assertTrue(target2["additional_operating_point"])

    def test_additional_downnorm_run_is_validated(self):
        with tempfile.TemporaryDirectory() as root:
            plan = os.path.join(root, "plan.json")
            with open(plan, "w", encoding="utf-8") as handle:
                json.dump({
                    "layers": [],
                    "allocation_ranking": {
                        "experiment_name": "rmsnorm_alloc__downnorm_rank",
                        "allocation_source": "rmsnorm_bound",
                        "ranking_source": "down_norm",
                        "ranking_aggregation_mode": "p95",
                    },
                }, handle)
            fields = (
                "requested_pct", "allocation_source", "ranking_source",
                "ranking_aggregation", "pruning_plan_path", "actual_pct",
                "layer_channels", "expert_neurons", "dataset",
            )
            summary = os.path.join(root, "allocation_ranking_summary.csv")
            with open(summary, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for dataset in ("wikitext2", "c4"):
                    writer.writerow({
                        "requested_pct": 6, "allocation_source": "rmsnorm_bound",
                        "ranking_source": "down_norm", "ranking_aggregation": "p95",
                        "pruning_plan_path": plan, "actual_pct": 6.2066,
                        "layer_channels": 2288, "expert_neurons": 292864,
                        "dataset": dataset,
                    })
            spec = select_additional_target6_downnorm_spec(root, "toy/model")
            self.assertEqual(spec["removed_layer_channels"], 2288)
            self.assertEqual(spec["ranking_source"], "down_norm")

    def test_additional_downnorm_run_recovers_omitted_summary_plan_path(self):
        with tempfile.TemporaryDirectory() as root:
            plan_dir = os.path.join(
                root, "rmsnorm_alloc__downnorm_rank", "pruning_plans"
            )
            os.makedirs(plan_dir)
            plan = os.path.join(plan_dir, "derived.json")
            with open(plan, "w", encoding="utf-8") as handle:
                json.dump({
                    "layers": [],
                    "allocation_ranking": {
                        "experiment_name": "rmsnorm_alloc__downnorm_rank",
                        "allocation_source": "rmsnorm_bound",
                        "ranking_source": "down_norm",
                        "ranking_aggregation_mode": "p95",
                    },
                }, handle)
            fields = (
                "requested_pct", "allocation_source", "ranking_source",
                "ranking_aggregation", "pruning_plan_path", "actual_pct",
                "layer_channels", "expert_neurons", "dataset",
            )
            summary = os.path.join(root, "allocation_ranking_summary.csv")
            with open(summary, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for dataset in ("wikitext2", "c4"):
                    writer.writerow({
                        "requested_pct": 6,
                        "allocation_source": "rmsnorm_bound",
                        "ranking_source": "down_norm",
                        "ranking_aggregation": "p95",
                        "pruning_plan_path": "",
                        "actual_pct": 6.2066,
                        "layer_channels": 2288,
                        "expert_neurons": 292864,
                        "dataset": dataset,
                    })
            spec = select_additional_target6_downnorm_spec(root, "toy/model")
            self.assertEqual(spec["plan_path"], plan)
            self.assertEqual(spec["plan_sha256"], file_sha256(plan))


if __name__ == "__main__":
    unittest.main()
