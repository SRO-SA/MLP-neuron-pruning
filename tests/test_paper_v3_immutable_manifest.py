import json
import os
import tempfile
import unittest

from scripts.build_paper_v3_immutable_checkpoint_manifest import (
    PLAN_FILENAME, byte_units, file_category, index_checkpoint,
)
from src.experiment_provenance import file_sha256


class PaperV3ImmutableManifestTests(unittest.TestCase):
    def test_decimal_and_binary_units_are_distinct(self):
        units = byte_units(1_073_741_824)
        self.assertEqual(units["size_gib_binary"], 1.0)
        self.assertAlmostEqual(units["size_gb_decimal"], 1.073741824)

    def test_categories(self):
        self.assertEqual(file_category("model-1.safetensors"), "safetensors_shard")
        self.assertEqual(file_category("model.safetensors.index.json"), "safetensors_index")
        self.assertEqual(file_category("tokenizer.json"), "tokenizer")

    def test_index_revalidates_export_hashes_and_plan(self):
        with tempfile.TemporaryDirectory() as root:
            files = {
                "model-1.safetensors": b"weights",
                "model.safetensors.index.json": b"{}",
                "config.json": b"{}", "generation_config.json": b"{}",
                "tokenizer.json": b"{}", PLAN_FILENAME: b'{"layers": []}',
            }
            for name, content in files.items():
                with open(os.path.join(root, name), "wb") as handle:
                    handle.write(content)
            hashes = {name: file_sha256(os.path.join(root, name)) for name in files}
            verification = {
                "successful_reload": True, "exact_logits_after_reload": True,
                "no_hidden_original_width_padding": True,
                "checkpoint_file_sha256": hashes,
                "plan_sha256": hashes[PLAN_FILENAME],
                "parameters_reloaded": {"total": 10, "moe_experts": 6},
                "serialized_weight_bytes": 7,
                "checkpoint_payload_bytes_excluding_verification_manifest": 20,
                "removed_layer_channels": 16, "removed_expert_neurons": 32,
            }
            with open(os.path.join(root, "checkpoint_verification.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(verification, handle)
            rows, summary = index_checkpoint({
                "label": "target", "checkpoint_dir": root,
                "target_pct": 4.0, "actual_pct": 4.1,
                "plan_sha256": hashes[PLAN_FILENAME],
            })
            self.assertEqual(summary["plan_sha256"], hashes[PLAN_FILENAME])
            self.assertEqual(summary["safetensors_shards"], 1)
            self.assertEqual(len(rows), len(files) + 1)


if __name__ == "__main__":
    unittest.main()
