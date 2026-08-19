import csv
import os
import tempfile
import unittest

from src.paired_bootstrap import (
    paired_bootstrap_nll_difference,
    write_paired_nll_csv,
)


class PairedBootstrapTests(unittest.TestCase):
    def test_constant_paired_difference_has_exact_interval(self):
        result = paired_bootstrap_nll_difference(
            [12.0, 24.0, 36.0],
            [10.0, 20.0, 30.0],
            [2, 4, 6],
            n_resamples=1000,
            seed=7,
        )
        self.assertAlmostEqual(result["mean_nll_difference"], 1.0)
        self.assertAlmostEqual(result["ci_lower"], 1.0)
        self.assertAlmostEqual(result["ci_upper"], 1.0)

    def test_bootstrap_is_deterministic(self):
        args = ([3.0, 8.0, 4.0], [2.0, 7.0, 5.0], [2, 3, 2])
        first = paired_bootstrap_nll_difference(*args, n_resamples=500, seed=11)
        second = paired_bootstrap_nll_difference(*args, n_resamples=500, seed=11)
        self.assertEqual(first, second)

    def test_writer_rejects_token_mismatch(self):
        baseline = [{"sample_index": 0, "n_tokens": 2, "nll_sum": 3.0}]
        pruned = [{"sample_index": 0, "n_tokens": 3, "nll_sum": 4.0}]
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaisesRegex(ValueError, "token mismatch"):
                write_paired_nll_csv(
                    os.path.join(tempdir, "paired.csv"),
                    dataset="c4",
                    corpus_sha256="abc",
                    baseline_examples=baseline,
                    pruned_examples=pruned,
                )

    def test_writer_saves_aligned_rows(self):
        examples = [
            {"sample_index": 0, "n_tokens": 2, "nll_sum": 3.0},
            {"sample_index": 1, "n_tokens": 4, "nll_sum": 5.0},
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "paired.csv")
            write_paired_nll_csv(
                path,
                dataset="wikitext2",
                corpus_sha256="abc",
                baseline_examples=examples,
                pruned_examples=examples,
            )
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["sample_index"], "1")
            self.assertEqual(rows[0]["corpus_sha256"], "abc")


if __name__ == "__main__":
    unittest.main()
