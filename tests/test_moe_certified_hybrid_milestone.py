from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.build_moe_certified_hybrid_frontier import main as build_frontier
from scripts.generate_moe_fixed_plan_eval_configs import resolve_rmsnorm_allocation_plan
from scripts.validate_target6_matched_plans import merge_compatible_protocols
from src.experiment_provenance import file_sha256


def target6_plan(selector: str, *, down: bool = False) -> dict:
    layers = []
    for layer_idx in range(48):
        count = 48 if layer_idx < 47 else 32
        start = 32 if down else 0
        selected = list(range(start, start + count))
        layers.append({
            "layer_idx": layer_idx, "prune_idx": selected,
            "old_intermediate": 768, "new_intermediate": 768 - count,
            "pruned_channels": count,
        })
    return {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "transformers_version": "test", "torch_version": "test",
        "target_pct": 6.0, "actual_pct": 6.2066, "selector": selector,
        "aggregation_mode": "p95", "pruning_mode": "packed_same_channel",
        "channel_alignment": 16, "max_layer_frac": 0.2,
        "num_layers": 48, "num_experts_per_layer": 128,
        "total_selected_layer_channels": 2288, "layers": layers,
    }


class CertifiedHybridMilestoneTests(unittest.TestCase):
    def test_fixed_plan_eval_uses_true_rmsnorm_allocation_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier_dir = root / "certification_frontier"
            frontier_dir.mkdir()
            ellipsoid_path = root / "ellipsoid.json"
            rmsnorm_path = root / "rmsnorm.json"
            ellipsoid_path.write_text(json.dumps(
                target6_plan("rmsnorm_ellipsoid_bound")
            ))
            rmsnorm_path.write_text(json.dumps(target6_plan("rmsnorm_bound")))
            validation_path = root / "matched_plan_validation.json"
            validation_path.write_text(json.dumps({
                "strict_gate_passed": True,
                "plans": {
                    "rmsnorm_bound": {
                        "path": str(rmsnorm_path),
                        "sha256": file_sha256(str(rmsnorm_path)),
                    }
                },
            }))
            frontier_path = frontier_dir / "hybrid_frontier.json"
            payload = {
                "source_plans": {
                    "ellipsoid": {
                        "path": str(ellipsoid_path),
                        "sha256": file_sha256(str(ellipsoid_path)),
                    }
                }
            }
            frontier_path.write_text(json.dumps(payload))

            resolved = resolve_rmsnorm_allocation_plan(payload, frontier_path)

            self.assertEqual(resolved, rmsnorm_path)
            self.assertNotEqual(resolved, ellipsoid_path)

    def test_legacy_missing_protocol_metadata_is_not_a_mismatch(self) -> None:
        merged, coverage = merge_compatible_protocols({
            "legacy": {
                "common": {"model": "model", "revision": ""},
                "datasets": {"c4": {"corpus": "abc", "tokens": 10}},
            },
            "new": {
                "common": {"model": "model", "revision": "revision-1"},
                "datasets": {"c4": {"corpus": "abc", "tokens": 10}},
            },
        })
        self.assertEqual(merged["common"]["revision"], "revision-1")
        self.assertEqual(coverage["common.revision"], ["new"])
        with self.assertRaisesRegex(ValueError, "conflicting populated"):
            merge_compatible_protocols({
                "left": {"corpus": "abc"}, "right": {"corpus": "xyz"},
            })

    def test_frontier_dry_math_writes_five_fixed_budget_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ellipsoid_path = root / "ellipsoid.json"
            down_path = root / "down.json"
            rmsnorm_path = root / "rmsnorm.json"
            ellipsoid_path.write_text(json.dumps(target6_plan("rmsnorm_ellipsoid_bound")))
            down_path.write_text(json.dumps(target6_plan("down_norm", down=True)))
            rmsnorm_path.write_text(json.dumps(target6_plan("rmsnorm_bound")))
            arrays = {}
            rng = np.random.default_rng(42)
            for layer in range(48):
                base = rng.uniform(1.0, 2.0, size=(3, 768)).astype(np.float32)
                down = np.tile(np.linspace(2.0, 0.1, 768), (3, 1)).astype(np.float32)
                arrays[f"layer_{layer}__ellipsoid"] = base
                arrays[f"layer_{layer}__down_norm"] = down
            bundle = root / "scores.npz"
            np.savez_compressed(bundle, **arrays)
            score_manifest = root / "scores.json"
            score_manifest.write_text("{}")
            validation = root / "validation.json"
            validation.write_text(json.dumps({
                "strict_gate_passed": True,
                "plans": {
                    "rmsnorm_bound": {
                        "path": str(rmsnorm_path),
                        "sha256": file_sha256(str(rmsnorm_path)),
                    }
                },
            }))
            output = root / "frontier"
            argv = [
                "build", "--ellipsoid-plan", str(ellipsoid_path),
                "--down-norm-plan", str(down_path), "--score-bundle", str(bundle),
                "--score-manifest", str(score_manifest),
                "--matched-validation", str(validation), "--output-dir", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                build_frontier()
            frontier = json.loads((output / "hybrid_frontier.json").read_text())
            self.assertEqual(len(frontier["plans"]), 5)
            for row in frontier["plans"]:
                plan = json.loads(Path(row["plan_path"]).read_text())
                self.assertEqual(plan["total_selected_layer_channels"], 2288)
                self.assertTrue(all(
                    len(layer["prune_idx"]) % 16 == 0 for layer in plan["layers"]
                ))
            self.assertEqual(
                json.loads((output / "set_level_certificates.json").read_text())
                ["pure_ellipsoid"]["inequality_violations"], 0,
            )


if __name__ == "__main__":
    unittest.main()
