import unittest

from scripts.build_paper_v3_paired_table import normalize_comparison


class PaperV3PairedTableTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
