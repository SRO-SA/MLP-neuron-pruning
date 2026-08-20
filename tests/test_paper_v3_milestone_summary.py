import unittest

from scripts.summarize_moe_paper_v3_milestone import build_milestone_rows


class PaperV3MilestoneSummaryTests(unittest.TestCase):
    def test_pivots_identical_dataset_protocol_into_one_experiment(self):
        common = {
            "source_group": "target8", "experiment_name": "rms__ellipsoid",
            "requested_pct": "8", "allocation_source": "rmsnorm_bound",
            "ranking_source": "rmsnorm_ellipsoid_bound",
            "expert_aggregation": "p95", "exact_total_layer_channels": "",
            "actual_pct": "8.1", "removed_layer_channels": "3000",
            "removed_expert_neurons": "384000",
            "expert_param_reduction_pct": "8.1",
            "total_model_param_reduction_pct": "7.8",
            "pruning_plan_sha256": "abc", "pruning_plan_path": "plan.json",
            "result_directory": "result", "bound": {
                "ellipsoid_bound_violations": 0,
            },
        }
        rows = []
        for dataset, sample, rel in (
            ("wikitext2", "wiki-sample", "1.0"),
            ("c4", "c4-sample", "1.5"),
        ):
            rows.append({
                **common, "dataset": dataset, "evaluation_sample_set_id": sample,
                "baseline_ppl": "12", "pruned_ppl": "12.2",
                "relative_ppl_pct": rel, "mean_nll_difference": "0.01",
                "nll_ci95_lower": "0.001", "nll_ci95_upper": "0.02",
                "tokens": "100",
            })
        result = build_milestone_rows(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["wikitext2_relative_ppl_pct"], "1.0")
        self.assertEqual(result[0]["c4_relative_ppl_pct"], "1.5")
        self.assertEqual(result[0]["ellipsoid_bound_violations"], 0)

    def test_rejects_mixed_evaluation_sample_sets(self):
        rows = []
        for experiment, sample in (("a", "sample-a"), ("b", "sample-b")):
            for dataset in ("wikitext2", "c4"):
                rows.append({
                    "source_group": experiment, "experiment_name": experiment,
                    "dataset": dataset,
                    "requested_pct": "4", "allocation_source": "rmsnorm_bound",
                    "ranking_source": "rmsnorm_bound", "expert_aggregation": "p95",
                    "exact_total_layer_channels": "", "actual_pct": "4",
                    "removed_layer_channels": "10", "removed_expert_neurons": "20",
                    "expert_param_reduction_pct": "4",
                    "total_model_param_reduction_pct": "4", "baseline_ppl": "12",
                    "pruned_ppl": "12", "relative_ppl_pct": "0",
                    "mean_nll_difference": "0", "nll_ci95_lower": "-1",
                    "nll_ci95_upper": "1", "tokens": "10",
                    "evaluation_sample_set_id": (
                        sample if dataset == "wikitext2" else "one-c4-sample"
                    ),
                    "pruning_plan_sha256": experiment,
                    "pruning_plan_path": f"{experiment}.json",
                    "result_directory": experiment, "bound": {},
                })
        with self.assertRaisesRegex(ValueError, "sample set"):
            build_milestone_rows(rows)


if __name__ == "__main__":
    unittest.main()
