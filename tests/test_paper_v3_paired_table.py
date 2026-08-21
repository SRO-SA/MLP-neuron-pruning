import csv
import os
import tempfile
import unittest

from scripts.build_paper_v3_paired_table import (
    PRIMARY_REQUESTS, add_paired_inference, normalize_comparison,
)


class PaperV3PairedTableTests(unittest.TestCase):
    def test_primary_profile_is_only_ranking_at_4_6_8(self):
        self.assertEqual({request[0] for request in PRIMARY_REQUESTS}, {4, 6, 8})
        self.assertEqual({request[1] for request in PRIMARY_REQUESTS}, {"ranking"})
        self.assertEqual(len(PRIMARY_REQUESTS), 7)

    def test_ranking_direction_and_significance(self):
        row = normalize_comparison({
            "comparison_type": "ranking", "source_group": "target8_n1024",
            "source_run_dir": "results/target8", "dataset": "c4",
            "allocation_source": "rmsnorm_bound",
            "competitor_ranking": "activation_score",
            "ellipsoid_minus_competitor_mean_nll": "-0.02",
            "ci95_lower": "-0.03", "ci95_upper": "-0.01",
            "n_documents": "1024", "n_tokens": "1000",
            "bootstrap_resamples": "10000",
        })
        self.assertEqual(row["target_pct"], 8)
        self.assertTrue(row["significant_95pct"])
        self.assertEqual(row["favored_method_if_significant"], "rmsnorm_ellipsoid_bound")

    def test_aggregation_is_flipped_to_p95_minus_max(self):
        row = normalize_comparison({
            "comparison_type": "aggregation", "source_group": "target6_aggregation",
            "source_run_dir": "", "dataset": "wikitext2",
            "allocation_source": "rmsnorm_bound",
            "max_minus_p95_mean_nll": "0.03",
            "ci95_lower": "0.01", "ci95_upper": "0.05",
            "n_documents": "1024", "n_tokens": "1000",
            "bootstrap_resamples": "10000",
        })
        self.assertAlmostEqual(row["mean_dnll_first_minus_second"], -0.03)
        self.assertAlmostEqual(row["ci95_lower"], -0.05)
        self.assertAlmostEqual(row["ci95_upper"], -0.01)
        self.assertEqual(row["favored_method_if_significant"], "p95")

    def test_interval_crossing_zero_is_not_significant(self):
        row = normalize_comparison({
            "comparison_type": "allocation", "source_group": "target6_exact",
            "source_run_dir": "", "dataset": "c4",
            "exact_removed_layer_channels": "2256",
            "ranking_source": "rmsnorm_ellipsoid_bound",
            "rmsnorm_minus_downnorm_mean_nll": "0.001",
            "ci95_lower": "-0.01", "ci95_upper": "0.02",
            "n_documents": "1024", "n_tokens": "1000",
            "bootstrap_resamples": "10000",
        })
        self.assertFalse(row["significant_95pct"])
        self.assertEqual(row["favored_method_if_significant"], "")

    def test_raw_paired_audit_adds_adjusted_p_values_and_ids(self):
        with tempfile.TemporaryDirectory() as root:
            nll_paths = {}
            for experiment, values in (("ellipsoid", [1.0, 1.0, 1.0, 1.0]),
                                       ("activation", [2.0, 2.0, 2.0, 2.0])):
                path = os.path.join(root, f"{experiment}.csv")
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    fields = ("dataset", "corpus_sha256", "sample_index",
                              "n_tokens", "pruned_nll_sum")
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for index, value in enumerate(values):
                        writer.writerow({"dataset": "c4", "corpus_sha256": "abc",
                                         "sample_index": index, "n_tokens": 1,
                                         "pruned_nll_sum": value})
                nll_paths[experiment] = path
            summary = os.path.join(root, "allocation_ranking_summary.csv")
            with open(summary, "w", newline="", encoding="utf-8") as handle:
                fields = ("experiment_name", "dataset", "per_example_nll_path")
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for experiment in ("ellipsoid", "activation"):
                    writer.writerow({"experiment_name": experiment, "dataset": "c4",
                                     "per_example_nll_path": nll_paths[experiment]})
            source = {
                "comparison_type": "ranking", "source_group": "target4_test",
                "source_run_dir": root, "dataset": "c4",
                "allocation_source": "rmsnorm_bound",
                "competitor_ranking": "activation_score",
                "ellipsoid_experiment": "ellipsoid",
                "competitor_experiment": "activation",
                "ellipsoid_minus_competitor_mean_nll": "-1",
                "ci95_lower": "-1", "ci95_upper": "-1",
                "n_documents": "4", "n_tokens": "4",
                "bootstrap_resamples": "1000",
            }
            output = normalize_comparison(source)
            output.pop("request_key")
            identifiers = add_paired_inference(
                [output], [source], source_root=root,
                randomization_replicates=1000, randomization_seed=5,
            )
            self.assertEqual(output["comparison_scope"], "primary")
            self.assertIn("holm_adjusted_p_value", output)
            self.assertEqual(len(identifiers["c4"]), 4)


if __name__ == "__main__":
    unittest.main()
