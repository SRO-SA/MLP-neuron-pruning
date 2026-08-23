import json
import os
import tempfile
import unittest

from scripts.audit_paper_v3_plan_nesting import audit_pair, selected_by_layer
from scripts.build_paper_v3_pareto_table import _dominates
from src.system_evidence import (
    shape_integers,
    validate_packed_moe_shapes,
    validate_unpacked_moe_shapes,
)
from src.paired_bootstrap import paired_signflip_nll_p_value
from src.statistical_audit import (
    apply_multiplicity_adjustments,
    benjamini_hochberg_adjust,
    holm_adjust,
    paired_signflip_statistics,
)


class StatisticalAuditTests(unittest.TestCase):
    def test_holm_and_bh_known_values(self):
        values = [0.01, 0.04, 0.03]
        self.assertEqual(holm_adjust(values), [0.03, 0.06, 0.06])
        adjusted = benjamini_hochberg_adjust(values)
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)

    def test_adjustments_stay_inside_declared_families(self):
        rows = [
            {"multiplicity_family": "primary", "paired_randomization_p_value": 0.01},
            {"multiplicity_family": "exploratory", "paired_randomization_p_value": 0.02},
            {"multiplicity_family": "exploratory", "paired_randomization_p_value": 0.04},
        ]
        apply_multiplicity_adjustments(rows)
        self.assertAlmostEqual(rows[0]["holm_adjusted_p_value"], 0.01)
        self.assertAlmostEqual(rows[1]["holm_adjusted_p_value"], 0.04)

    def test_task_stratified_paired_randomization(self):
        first = {"a": [1, 1, 1, 1], "b": [1, 1, 1, 1]}
        second = {"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]}
        result = paired_signflip_statistics(
            first, second, n_resamples=4000, seed=9
        )
        self.assertLess(result["macro_average"], 0.02)

    def test_plan_nesting_reports_replaced_channels(self):
        lower = {0: {1, 2}, 1: set()}
        upper = {0: {2, 3, 4}, 1: {5}}
        summary, layers = audit_pair(4, lower, 6, upper)
        self.assertFalse(summary["fully_nested"])
        self.assertEqual(summary["lower_not_in_upper_count"], 1)
        self.assertEqual(layers[0]["lower_not_in_upper_indices"], [1])

    def test_plan_parser_validates_declared_counts(self):
        plan = {
            "total_selected_layer_channels": 2,
            "layers": [{"layer_idx": 0, "pruned_channels": 2,
                        "prune_idx": [2, 4]}],
        }
        self.assertEqual(selected_by_layer(plan), {0: {2, 4}})

    def test_paired_nll_signflip_detects_consistent_improvement(self):
        p_value = paired_signflip_nll_p_value(
            [0.0] * 12, [1.0] * 12, [1] * 12,
            n_resamples=5000, seed=3,
        )
        self.assertLess(p_value, 0.01)

    def test_pareto_dominance_uses_all_declared_objectives(self):
        better = {
            "whole_model_parameter_reduction_pct": 5,
            "serialized_byte_reduction_pct": 5,
            "wikitext2_mean_dnll": 0.1, "c4_mean_dnll": 0.1,
            "macro_accuracy_loss_points": 1.0,
        }
        worse = {**better, "whole_model_parameter_reduction_pct": 4,
                 "macro_accuracy_loss_points": 2.0}
        self.assertTrue(_dominates(better, worse))
        tradeoff = {**better, "whole_model_parameter_reduction_pct": 7,
                    "macro_accuracy_loss_points": 3.0}
        self.assertFalse(_dominates(better, tradeoff))

    def test_profiler_shape_extraction_is_nested(self):
        self.assertEqual(shape_integers([[2, 768], [128, [704]]]),
                         {2, 128, 704, 768})

    def test_packed_moe_execution_shapes(self):
        layout = validate_packed_moe_shapes([128, 1536, 2048], [128, 2048, 768])
        self.assertEqual(layout["layout"], "packed")
        self.assertEqual(layout["expert_count"], 128)
        self.assertEqual(layout["intermediate_width"], 768)

    def test_unpacked_moe_execution_shapes(self):
        shapes = [
            {"gate": [752, 2048], "up": [752, 2048], "down": [2048, 752]}
            for _ in range(4)
        ]
        layout = validate_unpacked_moe_shapes(shapes)
        self.assertEqual(layout["layout"], "unpacked")
        self.assertEqual(layout["expert_count"], 4)
        self.assertEqual(layout["intermediate_width"], 752)

    def test_unpacked_moe_execution_shapes_reject_mismatch(self):
        with self.assertRaisesRegex(ValueError, "heterogeneous experts"):
            validate_unpacked_moe_shapes([
                {"gate": [752, 2048], "up": [752, 2048], "down": [2048, 752]},
                {"gate": [736, 2048], "up": [736, 2048], "down": [2048, 736]},
            ])


if __name__ == "__main__":
    unittest.main()
