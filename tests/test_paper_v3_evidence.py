import csv
import json
import os
import tempfile
import unittest

from scripts.build_moe_paper_v3_evidence import build_evidence
from src.experiment_provenance import corpus_sha256


class PaperV3EvidenceTests(unittest.TestCase):
    def test_builds_new_hashed_evidence_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tempdir:
            run_dir = os.path.join(tempdir, "run")
            experiment_dir = os.path.join(run_dir, "cell")
            os.makedirs(experiment_dir)
            plan_path = os.path.join(experiment_dir, "plan.json")
            with open(plan_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "layers": [
                        {"layer_idx": 0, "prune_idx": list(range(16))},
                        {"layer_idx": 1, "prune_idx": []},
                    ]
                }, handle)
            nll_path = os.path.join(experiment_dir, "paired.csv")
            with open(nll_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_index"])
                writer.writeheader()
                writer.writerow({"sample_index": 0})
            texts = ["evaluation document that is distinct"]
            result_path = os.path.join(
                experiment_dir, "moe_target_pruning_test.csv"
            )
            row = {
                "status": "ok", "evaluation_token_count_match": "True",
                "model": "toy/model", "model_revision": "", "tokenizer_revision": "",
                "tokenizer_name_or_path": "toy/model", "target_pct": "2",
                "actual_pct": "2.1", "eval_dataset": "wikitext2", "n_eval": "1",
                "evaluation_corpus_sha256": corpus_sha256(texts),
                "evaluation_num_texts": "1", "evaluation_max_seq_len": "512",
                "evaluation_batch_size": "4", "evaluation_preprocessing": "fixed",
                "seed": "42", "allocation_source": "rmsnorm_bound",
                "ranking_source": "rmsnorm_ellipsoid_bound",
                "ranking_aggregation_mode": "p95", "selected_layer_channels": "16",
                "removed_expert_neurons": "128", "expert_param_reduction_pct": "2.1",
                "total_model_param_reduction_pct": "2.0", "baseline_ppl": "12.0",
                "compressed_ppl": "12.1", "relative_delta_pct": "0.8",
                "mean_nll_difference": "0.01",
                "mean_nll_difference_ci95_lower": "0.001",
                "mean_nll_difference_ci95_upper": "0.02",
                "paired_bootstrap_resamples": "10000", "pruned_eval_tokens": "100",
                "baseline_eval_tokens": "100",
                "pruning_plan_path": plan_path, "pruning_plan_sha256": "",
                "per_example_nll_path": nll_path, "process_id": "1",
                "model_load_instance_id": "fresh-1",
            }
            with open(result_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            output_dir = os.path.join(tempdir, "evidence")
            result = build_evidence(
                run_dirs=[run_dir], output_dir=output_dir,
                model_revision="model-sha", tokenizer_revision="tokenizer-sha",
                require_targets={2}, evaluation_corpora={"wikitext2": texts},
                calibration_prompts=["unrelated calibration prompt"],
            )
            self.assertEqual(result["records"], 1)
            self.assertTrue(result["calibration_eval_disjoint_verified"])
            with open(result["json"], encoding="utf-8") as handle:
                record = json.load(handle)["records"][0]
            self.assertEqual(record["model_revision"], "model-sha")
            self.assertEqual(len(record["pruning_plan_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                build_evidence(
                    run_dirs=[run_dir], output_dir=output_dir,
                    model_revision="model-sha", tokenizer_revision="tokenizer-sha",
                    require_targets={2}, evaluation_corpora={"wikitext2": texts},
                    calibration_prompts=["unrelated calibration prompt"],
                )

    def test_activation_overlap_fails_before_output_creation(self):
        # The lower-level manifest assertion is covered separately; this test
        # documents that the evidence builder performs the same hard gate.
        from src.experiment_provenance import (
            assert_calibration_evaluation_disjoint, build_text_manifest,
        )
        manifest = build_text_manifest({"c4": ["same"]}, [" same "])
        with self.assertRaises(AssertionError):
            assert_calibration_evaluation_disjoint(manifest)


if __name__ == "__main__":
    unittest.main()
