"""CPU-only tests for the RMSNorm ellipsoid selector and diagnostics."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.moe_pruning import (
    MoELayerInfo,
    PackedExpertView,
    _score_expert_moe,
    collect_moe_bound_comparison_scores,
    compute_rmsnorm_ellipsoid_scores_for_expert,
    get_expert_scores,
    get_moe_input_rmsnorm_weight,
    save_moe_bound_comparison_diagnostics,
    save_moe_bound_tightness_diagnostics,
)
from src.rmsnorm_geometry import (
    compute_observed_channel_contribution_max_from_weights,
    compute_rmsnorm_bound_triplet_from_weights,
    compute_rmsnorm_ellipsoid_bound_from_weights,
    compute_rmsnorm_sphere_bound_from_weights,
)


class ToyExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)


class ToyPackedExperts(nn.Module):
    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.stack([torch.cat([gate, up], dim=0)]))
        self.down_proj = nn.Parameter(down.unsqueeze(0))


class ToyRMSNorm(nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(gamma.clone())
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = hidden_states * torch.rsqrt(
            hidden_states.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return q * self.weight


class ToyLayer(nn.Module):
    def __init__(self, gamma: torch.Tensor) -> None:
        super().__init__()
        self.post_attention_layernorm = ToyRMSNorm(gamma)


def _copy_expert_weights(
    expert: ToyExpert,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> None:
    with torch.no_grad():
        expert.gate_proj.weight.copy_(gate)
        expert.up_proj.weight.copy_(up)
        expert.down_proj.weight.copy_(down)


class RMSNormEllipsoidMathTests(unittest.TestCase):
    def test_sampled_rmsnorm_contributions_are_bounded(self) -> None:
        torch.manual_seed(7)
        d_model, d_ff = 9, 6
        gate = torch.randn(d_ff, d_model)
        up = torch.randn(d_ff, d_model)
        down = torch.randn(d_model, d_ff)
        gamma = 0.25 + 1.75 * torch.rand(d_model)
        bound = compute_rmsnorm_ellipsoid_bound_from_weights(
            gate, up, down, gamma
        )

        h = torch.randn(20000, d_model)
        rmsnorm = ToyRMSNorm(gamma)
        r = rmsnorm(h)
        activations = F.silu(r @ gate.T) * (r @ up.T)
        actual = activations.abs() * down.norm(dim=0).unsqueeze(0)
        tolerance = 5e-5 * torch.maximum(torch.ones_like(bound), bound)
        self.assertTrue(torch.all(actual <= bound.unsqueeze(0) + tolerance))
        observed_max = compute_observed_channel_contribution_max_from_weights(
            gate, up, down, r
        )
        torch.testing.assert_close(observed_max, actual.amax(dim=0))
        self.assertTrue(torch.all(observed_max <= bound + tolerance))

    def test_zero_gate_up_and_down_channels_have_zero_score(self) -> None:
        torch.manual_seed(11)
        d_model, d_ff = 5, 3
        gate = torch.randn(d_ff, d_model)
        up = torch.randn(d_ff, d_model)
        down = torch.randn(d_model, d_ff)
        gamma = torch.rand(d_model) + 0.5
        gate[0].zero_()
        up[1].zero_()
        down[:, 2].zero_()
        scores = compute_rmsnorm_ellipsoid_bound_from_weights(
            gate, up, down, gamma
        )
        torch.testing.assert_close(scores, torch.zeros_like(scores))

    def test_parallel_antiparallel_and_orthogonal_weighted_vectors(self) -> None:
        d_model, d_ff = 4, 3
        gamma = torch.tensor([0.5, 1.0, 1.5, 2.0])
        weighted_gate = torch.tensor([
            [1.0, 2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        weighted_up = torch.tensor([
            [2.0, 4.0, 0.0, 0.0],       # parallel
            [-2.0, -4.0, 0.0, 0.0],     # antiparallel
            [0.0, 3.0, 0.0, 0.0],       # orthogonal
        ])
        gate = weighted_gate / gamma
        up = weighted_up / gamma
        down = torch.eye(d_model, d_ff)
        scores = compute_rmsnorm_ellipsoid_bound_from_weights(
            gate, up, down, gamma
        )
        expected = (
            float(d_model)
            / 2.0
            * (
                weighted_gate.norm(dim=1) * weighted_up.norm(dim=1)
                + (weighted_gate * weighted_up).sum(dim=1).abs()
            )
            * down.norm(dim=0)
        )
        torch.testing.assert_close(scores, expected)
        self.assertAlmostEqual(float(scores[0]), float(scores[1]), places=5)

    def test_ellipsoid_never_exceeds_valid_sphere_bound(self) -> None:
        torch.manual_seed(23)
        for _ in range(20):
            d_model, d_ff = 13, 7
            gate = torch.randn(d_ff, d_model)
            up = torch.randn(d_ff, d_model)
            down = torch.randn(d_model, d_ff)
            gamma = torch.randn(d_model) * 1.5
            ellipsoid = compute_rmsnorm_ellipsoid_bound_from_weights(
                gate, up, down, gamma
            )
            sphere = compute_rmsnorm_sphere_bound_from_weights(
                gate, up, down, gamma
            )
            tolerance = 5e-5 * torch.maximum(torch.ones_like(sphere), sphere)
            self.assertTrue(torch.all(ellipsoid <= sphere + tolerance))

    def test_triplet_matches_individual_bounds_and_legacy_formula(self) -> None:
        torch.manual_seed(29)
        d_model, d_ff = 7, 5
        gate = torch.randn(d_ff, d_model)
        up = torch.randn(d_ff, d_model)
        down = torch.randn(d_model, d_ff)
        gamma = torch.rand(d_model) + 0.1
        legacy, sphere, ellipsoid = compute_rmsnorm_bound_triplet_from_weights(
            gate, up, down, gamma
        )
        expected_legacy = (
            (gate.norm(dim=1) * up.norm(dim=1) + (gate * up).sum(dim=1).abs())
            / 2.0
            * down.norm(dim=0)
        )
        torch.testing.assert_close(legacy, expected_legacy, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            sphere,
            compute_rmsnorm_sphere_bound_from_weights(gate, up, down, gamma),
        )
        torch.testing.assert_close(
            ellipsoid,
            compute_rmsnorm_ellipsoid_bound_from_weights(gate, up, down, gamma),
        )

    def test_shape_and_missing_gamma_assertions(self) -> None:
        gate = torch.randn(3, 5)
        up = torch.randn(3, 5)
        down = torch.randn(5, 3)
        with self.assertRaises(AssertionError):
            compute_rmsnorm_ellipsoid_bound_from_weights(
                gate, up, down, torch.ones(4)
            )
        with self.assertRaises(AssertionError):
            compute_rmsnorm_ellipsoid_bound_from_weights(
                gate, up, down.T, torch.ones(5)
            )


class RMSNormEllipsoidIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.d_model, self.d_ff = 6, 4
        self.gate = torch.randn(self.d_ff, self.d_model)
        self.up = torch.randn(self.d_ff, self.d_model)
        self.down = torch.randn(self.d_model, self.d_ff)
        self.gamma = torch.rand(self.d_model) + 0.25

    def test_unpacked_and_packed_experts_match(self) -> None:
        unpacked = ToyExpert(self.d_model, self.d_ff)
        _copy_expert_weights(unpacked, self.gate, self.up, self.down)
        packed_container = ToyPackedExperts(self.gate, self.up, self.down)
        packed = PackedExpertView(packed_container, 0, self.d_ff)
        unpacked_scores = compute_rmsnorm_ellipsoid_scores_for_expert(
            unpacked, self.gamma
        )
        packed_scores = compute_rmsnorm_ellipsoid_scores_for_expert(
            packed, self.gamma
        )
        torch.testing.assert_close(unpacked_scores, packed_scores)

    def test_dispatcher_requires_gamma_and_preserves_legacy_selector(self) -> None:
        expert = ToyExpert(self.d_model, self.d_ff)
        _copy_expert_weights(expert, self.gate, self.up, self.down)
        with self.assertRaises(AssertionError):
            _score_expert_moe(expert, "rmsnorm_ellipsoid_bound")
        ellipsoid = _score_expert_moe(
            expert, "rmsnorm_ellipsoid_bound", rmsnorm_gamma=self.gamma
        )
        direct = compute_rmsnorm_ellipsoid_scores_for_expert(expert, self.gamma)
        torch.testing.assert_close(ellipsoid, direct)
        torch.testing.assert_close(
            _score_expert_moe(expert, "rmsnorm_bound"),
            get_expert_scores(expert),
            rtol=0.0,
            atol=0.0,
        )
        with self.assertRaises(ValueError):
            _score_expert_moe(expert, "not_a_selector")

    def test_layer_gamma_is_taken_from_post_attention_rmsnorm(self) -> None:
        layer = ToyLayer(self.gamma)
        info = MoELayerInfo(3, layer)
        torch.testing.assert_close(
            get_moe_input_rmsnorm_weight(info),
            layer.post_attention_layernorm.weight,
        )
        missing = MoELayerInfo(4, nn.Module())
        with self.assertRaises(AssertionError):
            get_moe_input_rmsnorm_weight(missing)

    def test_comparison_diagnostics_are_compact_and_valid(self) -> None:
        layer = ToyLayer(self.gamma)
        info = MoELayerInfo(0, layer)
        info.is_moe = True
        info.num_experts = 2
        info.expert_modules = []
        for scale in (1.0, 1.2):
            expert = ToyExpert(self.d_model, self.d_ff)
            _copy_expert_weights(
                expert, self.gate * scale, self.up / scale, self.down
            )
            info.expert_modules.append(expert)

        records = collect_moe_bound_comparison_scores([info], aggregation="p95")
        self.assertEqual(len(records), 1)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, json_path = save_moe_bound_comparison_diagnostics(
                records, tmp, "test", "toy/model", "p95"
            )
            with open(csv_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([row["scope"] for row in rows], ["layer", "global"])
            self.assertEqual(int(rows[-1]["ellipsoid_gt_sphere_count"]), 0)
            payload = json.loads(Path(json_path).read_text())
            self.assertEqual(payload["new_selector"], "rmsnorm_ellipsoid_bound")
            self.assertIn("bottom_2pct", payload["global"]["bottom_channels"])
            self.assertIn("cross_layer_scale", payload)
            self.assertIn(
                "ellipsoid_layer_median_cross_layer_max_to_min_positive",
                payload["cross_layer_scale"],
            )
            self.assertIn("does not by itself", payload["cross_layer_scale"]["interpretation"])

    def test_bound_tightness_saves_preaggregation_expert_scores(self) -> None:
        layer = ToyLayer(self.gamma)
        info = MoELayerInfo(0, layer)
        info.is_moe = True
        info.num_experts = 2
        info.expert_modules = []
        activations = {}
        for expert_idx, scale in enumerate((1.0, 1.1)):
            expert = ToyExpert(self.d_model, self.d_ff)
            _copy_expert_weights(
                expert, self.gate * scale, self.up / scale, self.down
            )
            info.expert_modules.append(expert)
            raw = torch.randn(32, self.d_model)
            activations[(0, expert_idx)] = layer.post_attention_layernorm(raw)
        with tempfile.TemporaryDirectory() as tmp:
            json_path, npz_path = save_moe_bound_tightness_diagnostics(
                [info], activations, {(0, -1): [0]},
                output_dir=tmp,
                timestamp="test",
                model_name="toy/model",
                aggregation="p95",
                save_expert_scores=True,
            )
            payload = json.loads(Path(json_path).read_text())
            self.assertEqual(
                payload["global"]["ellipsoid_numerical_violations"], 0
            )
            self.assertEqual(payload["global"]["sphere_numerical_violations"], 0)
            self.assertLessEqual(payload["global"]["ellipsoid_all"]["max"], 1.0001)
            self.assertEqual(payload["sampled_routed_inputs"], 16)
            self.assertEqual(
                payload["expert_channel_pairs_evaluated"], 2 * self.d_ff
            )
            self.assertEqual(
                payload["routed_input_channel_contributions_evaluated"],
                16 * self.d_ff,
            )
            self.assertIn("sphere_to_ellipsoid_bound", payload["global"])
            self.assertIn("ellipsoid_pruned", payload["global"])
            self.assertEqual(
                payload["tolerance_rule"],
                "observed <= bound * (1 + relative_tolerance) + absolute_tolerance",
            )
            archive = dict(__import__("numpy").load(npz_path))
            self.assertEqual(
                archive["layer_0_ellipsoid_bound"].shape,
                (2, self.d_ff),
            )
            self.assertEqual(archive["layer_0_pruned_mask"].tolist(), [True, False, False, False])

    def test_bound_tightness_fails_on_a_sampled_violation(self) -> None:
        layer = ToyLayer(self.gamma)
        info = MoELayerInfo(0, layer)
        info.is_moe = True
        info.num_experts = 1
        expert = ToyExpert(self.d_model, self.d_ff)
        gate = torch.ones_like(self.gate)
        up = torch.ones_like(self.up)
        down = torch.ones_like(self.down)
        _copy_expert_weights(expert, gate, up, down)
        info.expert_modules = [expert]
        # Deliberately bypass RMSNorm with an invalid large-norm input. The
        # audit must reject it rather than merely reporting the violation.
        invalid_routed = torch.full((1, self.d_model), 100.0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(AssertionError, "exceeded"):
                save_moe_bound_tightness_diagnostics(
                    [info], {(0, 0): invalid_routed}, {(0, -1): [0]},
                    output_dir=tmp, timestamp="violation", model_name="toy/model",
                    aggregation="p95", save_expert_scores=False,
                )


if __name__ == "__main__":
    unittest.main()
