from __future__ import annotations

import unittest

import numpy as np

from src.moe_set_certification import (
    certificate_for_plan,
    clone_plan_with_selection,
    matched_plan_validation,
    refine_with_certificate_slack,
    selected_by_layer,
)


def make_plan(selected, selector="rmsnorm_ellipsoid_bound"):
    layers = []
    for layer_idx, indices in enumerate(selected):
        old = 8
        layers.append({
            "layer_idx": layer_idx,
            "old_intermediate": old,
            "new_intermediate": old - len(indices),
            "pruned_channels": len(indices),
            "prune_idx": list(indices),
        })
    return {
        "model_id": "synthetic",
        "selector": selector,
        "target_pct": 25.0,
        "actual_pct": 25.0,
        "pruning_mode": "packed_same_channel",
        "aggregation_mode": "p95",
        "channel_alignment": 2,
        "max_layer_frac": 0.5,
        "num_experts_per_layer": 3,
        "total_selected_layer_channels": sum(len(x) for x in selected),
        "layers": layers,
    }


class SetCertificateTests(unittest.TestCase):
    def setUp(self):
        # Shape: [expert, channel].  Ellipsoid selection [0,1] is certified
        # cheaper; down-norm endpoint [2,3] has the lower practical objective.
        self.scores = {}
        for layer in range(2):
            ellipsoid = np.array([
                [1.0, 1.2, 1.3, 1.6, 5, 5, 5, 5],
                [1.1, 1.0, 1.5, 1.4, 5, 5, 5, 5],
                [0.9, 1.1, 1.4, 1.5, 5, 5, 5, 5],
            ], dtype=np.float32) * (layer + 1)
            down = np.array([
                [4.0, 3.0, 1.0, 0.5, 6, 6, 6, 6],
                [4.1, 3.1, 1.1, 0.6, 6, 6, 6, 6],
                [3.9, 2.9, 0.9, 0.4, 6, 6, 6, 6],
            ], dtype=np.float32)
            self.scores[layer] = {"ellipsoid": ellipsoid, "down_norm": down}
        self.ellipsoid = make_plan([[0, 1], [0, 1]])
        self.down = make_plan([[2, 3], [2, 3]], selector="down_norm")

    def test_strict_set_certificate_is_no_larger_than_channelwise_max(self):
        report = certificate_for_plan(self.ellipsoid, self.scores)
        self.assertEqual(report["inequality_violations"], 0)
        self.assertLessEqual(
            report["strict_global_unpropagated_certificate"],
            report["older_global_channelwise_max_certificate"],
        )
        self.assertEqual(len(report["layers"]), 2)
        self.assertEqual(len(report["expert_set_bounds"]), 6)

    def test_matched_validation_rejects_changed_allocation(self):
        report = matched_plan_validation(
            {"a": self.ellipsoid, "b": self.down},
            expected_total=4, expected_alignment=2,
        )
        self.assertTrue(report["validation_passed"])
        bad = make_plan([[0, 1, 2, 3], []], selector="down_norm")
        with self.assertRaisesRegex(ValueError, "per-layer allocation"):
            matched_plan_validation(
                {"a": self.ellipsoid, "b": bad},
                expected_total=4, expected_alignment=2,
            )

    def test_refinement_is_deterministic_fixed_budget_and_bounded(self):
        selected1, audit1 = refine_with_certificate_slack(
            self.ellipsoid, self.down, self.scores, 0.25, seed=42
        )
        selected2, audit2 = refine_with_certificate_slack(
            self.ellipsoid, self.down, self.scores, 0.25, seed=42
        )
        self.assertEqual(selected1, selected2)
        self.assertEqual(audit1["selection_sha256"], audit2["selection_sha256"])
        self.assertEqual({k: len(v) for k, v in selected1.items()}, {0: 2, 1: 2})
        self.assertLessEqual(
            audit1["final_strict_certificate"],
            audit1["strict_certificate_threshold"] + 1e-7,
        )
        self.assertLessEqual(
            audit1["final_down_norm_objective"],
            audit1["base_down_norm_objective"],
        )

    def test_clone_changes_only_identities_and_metadata(self):
        selection = {0: {2, 3}, 1: {2, 3}}
        clone = clone_plan_with_selection(
            self.ellipsoid, selection, plan_name="hybrid", metadata={"rho": 0.1}
        )
        self.assertEqual(selected_by_layer(clone), {0: (2, 3), 1: (2, 3)})
        self.assertEqual(clone["total_selected_layer_channels"], 4)
        self.assertEqual(clone["layers"][0]["new_intermediate"], 6)
        self.assertEqual(clone["certified_hybrid"], {"rho": 0.1})


if __name__ == "__main__":
    unittest.main()
