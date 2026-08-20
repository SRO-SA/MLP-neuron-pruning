import json
import os
import tempfile
import unittest

from src.tokenizer_policy import resolve_tokenizer_policy


class TokenizerPolicyTests(unittest.TestCase):
    def test_requires_passed_audit_and_matches_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, "checkpoint")
            os.makedirs(checkpoint)
            audit_path = os.path.join(root, "audit.json")
            with open(audit_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "decision": {
                        "audit_passed_for_downstream": True,
                        "use_fix_mistral_regex_for_future_evaluation": True,
                        "previous_ppl_rerun_required": False,
                    },
                    "sources": [{
                        "label": "frozen", "source": checkpoint,
                        "tokenizer_files_combined_sha256": "files-hash",
                    }],
                }, handle)
            policy = resolve_tokenizer_policy(
                audit_path, checkpoint, label="frozen"
            )
            self.assertTrue(policy["fix_mistral_regex"])
            self.assertEqual(policy["tokenizer_files_combined_sha256"], "files-hash")

    def test_rejects_uncovered_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            audit_path = os.path.join(root, "audit.json")
            with open(audit_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "decision": {
                        "audit_passed_for_downstream": True,
                        "use_fix_mistral_regex_for_future_evaluation": True,
                    },
                    "sources": [],
                }, handle)
            with self.assertRaises(ValueError):
                resolve_tokenizer_policy(audit_path, root, label="missing")


if __name__ == "__main__":
    unittest.main()
