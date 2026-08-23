import json
import os
import subprocess
import sys
import tempfile
import unittest

from scripts.audit_paper_v3_tokenizers import (
    build_decision, compare_records, encode_record, validate_checkpoint_cohort,
)


class _ToyTokenizer:
    def __init__(self, offset=0):
        self.offset = offset

    def encode(self, text, add_special_tokens=True):
        values = [ord(char) + self.offset for char in text]
        return ([1 + self.offset] + values) if add_special_tokens else values

    def decode(self, ids, **_kwargs):
        content = ids[1:]
        return "".join(chr(value - self.offset) for value in content)


class PaperV3TokenizerAuditTests(unittest.TestCase):
    def _run_dry_run(self, labels):
        with tempfile.TemporaryDirectory() as root:
            specs = []
            for label in labels:
                checkpoint_dir = os.path.join(root, label)
                os.mkdir(checkpoint_dir)
                specs.append({"label": label, "checkpoint_dir": checkpoint_dir})
            manifest = os.path.join(root, "checkpoint_manifest.json")
            with open(manifest, "w", encoding="utf-8") as handle:
                json.dump(specs, handle)
            completed = subprocess.run(
                [
                    sys.executable, "scripts/audit_paper_v3_tokenizers.py",
                    "--checkpoint-manifest", manifest,
                    "--output-dir", os.path.join(root, "audit"),
                    "--samples-per-dataset", "100", "--dry-run",
                ],
                check=False, capture_output=True, text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout

    def test_dry_run_accepts_old_ellipsoid_cohort_without_loading_model(self):
        output = self._run_dry_run([
            "baseline_unpruned",
            "rmsnorm_alloc__ellipsoid_rank__p95__target4",
            "rmsnorm_alloc__ellipsoid_rank__p95__target6",
            "rmsnorm_alloc__ellipsoid_rank__p95__target8",
        ])
        self.assertIn("cohort=paper_v3_frozen", output)

    def test_dry_run_accepts_downnorm_curve_cohort_without_loading_model(self):
        output = self._run_dry_run([
            "baseline_unpruned",
            "rmsnorm_alloc__ellipsoid_rank__p95__target6",
            "rmsnorm_alloc__downnorm_rank__p95__target2",
            "rmsnorm_alloc__downnorm_rank__p95__target4",
            "rmsnorm_alloc__downnorm_rank__p95__target6",
            "rmsnorm_alloc__downnorm_rank__p95__target8",
        ])
        self.assertIn("cohort=pure_downnorm_curve", output)

    def test_incomplete_old_ellipsoid_cohort_is_still_rejected(self):
        specs = [{"label": label} for label in (
            "baseline_unpruned",
            "rmsnorm_alloc__ellipsoid_rank__p95__target4",
            "rmsnorm_alloc__ellipsoid_rank__p95__target6",
        )]
        with self.assertRaisesRegex(ValueError, "paper_v3_frozen"):
            validate_checkpoint_cohort(specs)

    def test_equal_records_have_no_mismatch(self):
        examples = [{"collection": "canary", "sample_index": 0,
                     "text_sha256": "hash", "text": "abc"}]
        record = encode_record(_ToyTokenizer(), "abc")
        rows, representatives = compare_records([record], [dict(record)], examples)
        self.assertEqual(rows[0]["token_id_mismatches"], 0)
        self.assertEqual(rows[0]["decoded_mismatches"], 0)
        self.assertEqual(representatives, [])

    def test_token_id_change_is_reported_with_representative(self):
        examples = [{"collection": "wikitext2", "sample_index": 7,
                     "text_sha256": "hash", "text": "abc"}]
        left = encode_record(_ToyTokenizer(), "abc")
        right = encode_record(_ToyTokenizer(offset=1), "abc")
        rows, representatives = compare_records([left], [right], examples)
        self.assertEqual(rows[0]["token_id_mismatches"], 1)
        self.assertEqual(rows[0]["decoded_mismatches"], 0)
        self.assertEqual(representatives[0]["sample_index"], 7)

    def test_export_only_regex_change_does_not_invalidate_prior_hub_ppl(self):
        rows = [{
            "relation": "current_vs_fixed", "left_source": "hub_original",
            "right_mode": "fixed", "collection": "wikitext2",
            "token_id_mismatches": 0, "decoded_mismatches": 0,
        }, {
            "relation": "current_vs_fixed", "left_source": "baseline_unpruned",
            "right_mode": "fixed", "collection": "wikitext2",
            "token_id_mismatches": 2, "decoded_mismatches": 0,
        }, {
            "relation": "original_vs_export", "left_source": "hub_original",
            "right_mode": "fixed", "collection": "wikitext2",
            "token_id_mismatches": 0, "decoded_mismatches": 0,
        }]
        decision = build_decision(rows)
        self.assertFalse(decision["previous_ppl_rerun_required"])
        self.assertTrue(decision["audit_passed_for_downstream"])
        self.assertTrue(decision["use_fix_mistral_regex_for_future_evaluation"])

    def test_false_positive_local_warning_preserves_current_qwen_mode(self):
        rows = []
        for collection in ("canary", "wikitext2", "c4"):
            rows.extend([{
                "relation": "current_vs_fixed",
                "left_source": "hub_original",
                "left_mode": "current", "right_mode": "fixed",
                "collection": collection,
                "token_id_mismatches": 0, "decoded_mismatches": 0,
            }, {
                "relation": "current_vs_fixed",
                "left_source": "baseline_unpruned",
                "left_mode": "current", "right_mode": "fixed",
                "collection": collection,
                "token_id_mismatches": 1, "decoded_mismatches": 1,
            }, {
                "relation": "original_vs_export",
                "left_source": "hub_original",
                "left_mode": "current", "right_source": "baseline_unpruned",
                "right_mode": "current", "collection": collection,
                "token_id_mismatches": 0, "decoded_mismatches": 0,
            }, {
                "relation": "original_vs_export",
                "left_source": "hub_original",
                "left_mode": "fixed", "right_source": "baseline_unpruned",
                "right_mode": "fixed", "collection": collection,
                "token_id_mismatches": 1, "decoded_mismatches": 1,
            }])
        decision = build_decision(rows)
        self.assertTrue(decision["audit_passed_for_downstream"])
        self.assertEqual(decision["selected_tokenizer_mode"], "current")
        self.assertFalse(decision["use_fix_mistral_regex_for_future_evaluation"])
        self.assertTrue(
            decision["local_mistral_warning_consistent_with_false_positive"]
        )
        self.assertFalse(decision["previous_ppl_rerun_required"])

    def test_neither_matching_mode_fails_closed(self):
        rows = [{
            "relation": "original_vs_export", "left_source": "hub_original",
            "left_mode": mode, "right_source": "baseline_unpruned",
            "right_mode": mode, "collection": "canary",
            "token_id_mismatches": 1, "decoded_mismatches": 0,
        } for mode in ("current", "fixed")]
        decision = build_decision(rows)
        self.assertFalse(decision["audit_passed_for_downstream"])
        self.assertIsNone(decision["selected_tokenizer_mode"])
        self.assertIsNone(
            decision["use_fix_mistral_regex_for_future_evaluation"]
        )


if __name__ == "__main__":
    unittest.main()
