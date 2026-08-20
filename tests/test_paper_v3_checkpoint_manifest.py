import unittest
import subprocess
import sys
from unittest.mock import patch

from scripts.generate_paper_v3_checkpoint_manifest import select_checkpoint_specs


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


if __name__ == "__main__":
    unittest.main()
