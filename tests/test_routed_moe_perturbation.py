from __future__ import annotations

import unittest

import numpy as np
import torch

from src.routed_moe_perturbation import (
    compute_fixed_routing,
    expert_set_bounds,
    fixed_routed_moe_output,
    paired_bootstrap_mean_difference,
    route_conditioned_bounds,
    safe_ratio,
    spearman_correlation,
    violation_mask,
)


class PackedExperts(torch.nn.Module):
    def __init__(self, experts=3, hidden=4, intermediate=5):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.gate_up_proj = torch.nn.Parameter(torch.randn(
            experts, 2 * intermediate, hidden, generator=generator,
        ))
        self.down_proj = torch.nn.Parameter(torch.randn(
            experts, hidden, intermediate, generator=generator,
        ))

    def forward(self, hidden, expert_ids):
        outputs = []
        for slot in range(expert_ids.shape[1]):
            slot_outputs = []
            for token in range(hidden.shape[0]):
                expert = int(expert_ids[token, slot])
                gate_up = self.gate_up_proj[expert] @ hidden[token]
                gate, up = gate_up.chunk(2, dim=0)
                value = torch.nn.functional.silu(gate) * up
                slot_outputs.append(self.down_proj[expert] @ value)
            outputs.append(torch.stack(slot_outputs))
        return torch.stack(outputs, dim=1)


class PackedMoe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 3, bias=False)
        self.experts = PackedExperts()
        self.top_k = 2
        self.norm_topk_prob = True


class Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 5, bias=False)
        self.up_proj = torch.nn.Linear(4, 5, bias=False)
        self.down_proj = torch.nn.Linear(5, 4, bias=False)


class UnpackedMoe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 3, bias=False)
        self.experts = torch.nn.ModuleList([Expert() for _ in range(3)])
        self.top_k = 2
        self.norm_topk_prob = True


class RoutedMoePerturbationTests(unittest.TestCase):
    def test_packed_fixed_routing_matches_manual_convex_sum(self):
        torch.manual_seed(3)
        moe = PackedMoe()
        hidden = torch.randn(2, 3, 4)
        ids, weights = compute_fixed_routing(moe, hidden)
        output = fixed_routed_moe_output(moe, hidden, ids, weights)
        raw = moe.experts(hidden.reshape(-1, 4), ids)
        expected = (raw * weights.unsqueeze(-1)).sum(1).reshape_as(hidden)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(weights.sum(1), torch.ones(weights.shape[0]))

    def test_unpacked_fixed_route_is_independent_of_router_after_capture(self):
        torch.manual_seed(5)
        moe = UnpackedMoe()
        hidden = torch.randn(2, 2, 4)
        ids, weights = compute_fixed_routing(moe, hidden)
        first = fixed_routed_moe_output(moe, hidden, ids, weights)
        with torch.no_grad():
            moe.gate.weight.mul_(-100)
        second = fixed_routed_moe_output(moe, hidden, ids, weights)
        torch.testing.assert_close(first, second)

    def test_route_bound_is_no_larger_than_strict_and_detects_violation(self):
        scores = np.array([
            [1.0, 2.0, 8.0], [2.0, 1.5, 8.0], [0.5, 1.0, 8.0],
        ])
        sums = expert_set_bounds(scores, [0, 1])
        ids = np.array([[0, 2], [1, 2]])
        weights = np.array([[0.75, 0.25], [0.2, 0.8]])
        route = route_conditioned_bounds(sums, ids, weights)
        strict = sums.max()
        self.assertTrue(np.all(route <= strict + 1e-12))
        actual = route * 0.9
        self.assertFalse(violation_mask(actual, route).any())
        self.assertTrue(violation_mask(route * 1.01, route).all())
        self.assertTrue(np.all(safe_ratio(actual, route) < 1.0))

    def test_statistics_are_deterministic_and_paired(self):
        first = np.array([1.0, 2.0, 3.0, 4.0])
        second = np.array([0.5, 2.0, 2.5, 3.0])
        one = paired_bootstrap_mean_difference(first, second, resamples=2000, seed=42)
        two = paired_bootstrap_mean_difference(first, second, resamples=2000, seed=42)
        self.assertEqual(one, two)
        self.assertAlmostEqual(one["difference"], 0.5)
        self.assertAlmostEqual(spearman_correlation(first, second), 1.0)


if __name__ == "__main__":
    unittest.main()
