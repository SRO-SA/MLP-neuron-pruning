import csv
import os
import tempfile
import unittest

from scripts.summarize_moe_allocation_ranking import (
    build_aggregation_selection_comparisons,
    build_paired_aggregation_comparisons,
    build_paired_allocation_comparisons,
)


def _write_documents(path: str, nll_sums: list[float]) -> None:
    fields = [
        "dataset", "corpus_sha256", "sample_index", "n_tokens",
        "baseline_nll_sum", "baseline_nll_mean",
        "pruned_nll_sum", "pruned_nll_mean",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, nll_sum in enumerate(nll_sums):
            writer.writerow({
                "dataset": "c4",
                "corpus_sha256": "same-corpus",
                "sample_index": index,
                "n_tokens": 2,
                "baseline_nll_sum": 4.0,
                "baseline_nll_mean": 2.0,
                "pruned_nll_sum": nll_sum,
                "pruned_nll_mean": nll_sum / 2.0,
            })


class AllocationComparisonSummaryTests(unittest.TestCase):
    def test_exact_budget_allocation_ci_is_paired_and_directional(self):
        with tempfile.TemporaryDirectory() as tempdir:
            rmsnorm_path = os.path.join(tempdir, "rmsnorm.csv")
            downnorm_path = os.path.join(tempdir, "downnorm.csv")
            _write_documents(rmsnorm_path, [3.0, 5.0, 7.0])
            _write_documents(downnorm_path, [4.0, 6.0, 8.0])
            common = {
                "dataset": "c4",
                "ranking_source": "rmsnorm_ellipsoid_bound",
                "ranking_aggregation": "p95",
                "layer_channels": "1536",
                "exact_total_layer_channels": "1536",
            }
            rows = [
                {
                    **common,
                    "experiment_name": "rms",
                    "allocation_source": "rmsnorm_bound",
                    "per_example_nll_path": rmsnorm_path,
                },
                {
                    **common,
                    "experiment_name": "down",
                    "allocation_source": "down_norm",
                    "per_example_nll_path": downnorm_path,
                },
            ]
            result = build_paired_allocation_comparisons(
                rows, bootstrap_resamples=1000
            )
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(
                result[0]["rmsnorm_minus_downnorm_mean_nll"], -0.5
            )
            self.assertLess(result[0]["ci95_upper"], 0.0)

    def test_aggregation_comparison_counts_replaced_ids(self):
        with tempfile.TemporaryDirectory() as tempdir:
            rows = []
            for aggregation, indices in (("p95", [0, 1]), ("max", [1, 2])):
                path = os.path.join(tempdir, f"{aggregation}.json")
                with open(path, "w", encoding="utf-8") as handle:
                    __import__("json").dump({
                        "layers": [{"layer_idx": 0, "prune_idx": indices}]
                    }, handle)
                rows.append({
                    "experiment_name": aggregation,
                    "pruning_plan_path": path,
                    "allocation_source": "rmsnorm_bound",
                    "ranking_source": "rmsnorm_ellipsoid_bound",
                    "ranking_aggregation": aggregation,
                    "exact_total_layer_channels": "2",
                    "layer_channels": "2",
                })
            result = build_aggregation_selection_comparisons(rows)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["global_overlap_count"], 1)
            self.assertEqual(result[0]["global_changed_ids_per_ranking"], 1)
            self.assertEqual(result[0]["global_symmetric_difference_count"], 2)

    def test_paired_aggregation_ci_is_document_paired(self):
        with tempfile.TemporaryDirectory() as tempdir:
            p95_path = os.path.join(tempdir, "p95.csv")
            max_path = os.path.join(tempdir, "max.csv")
            _write_documents(p95_path, [10.0, 20.0, 30.0])
            _write_documents(max_path, [11.0, 21.0, 31.0])
            common = {
                "allocation_source": "rmsnorm_bound",
                "dataset": "c4",
                "ranking_source": "rmsnorm_ellipsoid_bound",
                "layer_channels": "1536",
                "exact_total_layer_channels": "1536",
            }
            rows = [
                {
                    **common, "experiment_name": "p95",
                    "ranking_aggregation": "p95",
                    "relative_ppl_change_pct": "1.0",
                    "per_example_nll_path": p95_path,
                },
                {
                    **common, "experiment_name": "max",
                    "ranking_aggregation": "max",
                    "relative_ppl_change_pct": "1.2",
                    "per_example_nll_path": max_path,
                },
            ]
            result = build_paired_aggregation_comparisons(
                rows, bootstrap_resamples=100
            )
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0]["max_minus_p95_mean_nll"], 0.5)
            self.assertEqual(result[0]["exact_removed_layer_channels"], "1536")


if __name__ == "__main__":
    unittest.main()
