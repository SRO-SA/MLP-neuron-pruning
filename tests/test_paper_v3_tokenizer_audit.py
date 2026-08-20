import unittest

from scripts.audit_paper_v3_tokenizers import (
    build_decision, compare_records, encode_record,
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


if __name__ == "__main__":
    unittest.main()
