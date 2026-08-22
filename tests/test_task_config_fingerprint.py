import json
import os
import tempfile
import unittest

from scripts.summarize_paper_v3_downstream import load_run
from src.task_config_fingerprint import (
    FINGERPRINT_VERSION, stable_task_config, task_config_sha256,
)


class TaskConfigFingerprintTests(unittest.TestCase):
    def test_python_repr_addresses_do_not_change_fingerprint(self):
        first = {"metric": "<function acc at 0x7f012345abcd>"}
        second = {"metric": "<function acc at 0x7f0fedcba987>"}
        self.assertEqual(task_config_sha256(first), task_config_sha256(second))
        self.assertEqual(
            stable_task_config(first)["metric"],
            "<function acc at 0xADDR>",
        )

    def test_meaningful_hexadecimal_text_is_preserved(self):
        first = {"prompt": "answer token 0x7f012345abcd"}
        second = {"prompt": "answer token 0x7f0fedcba987"}
        self.assertNotEqual(task_config_sha256(first), task_config_sha256(second))

    def test_substantive_task_change_does_not_match(self):
        first = {"doc_to_text": "Question: {{question}}", "num_fewshot": 0}
        second = {"doc_to_text": "Q: {{question}}", "num_fewshot": 0}
        self.assertNotEqual(task_config_sha256(first), task_config_sha256(second))

    def test_mapping_order_does_not_change_fingerprint(self):
        first = {"task": "arc", "metric": {"acc": True, "norm": True}}
        second = {"metric": {"norm": True, "acc": True}, "task": "arc"}
        self.assertEqual(task_config_sha256(first), task_config_sha256(second))

    def test_live_callable_matches_json_persisted_representation(self):
        def accuracy(items):
            return sum(items) / len(items)

        runtime = {"metric": accuracy, "num_fewshot": 0}
        persisted = json.loads(json.dumps(runtime, default=str))
        self.assertEqual(
            task_config_sha256(runtime),
            task_config_sha256(persisted),
        )

    def test_round_trip_fix_uses_new_fingerprint_version(self):
        self.assertEqual(FINGERPRINT_VERSION, "stable-json-v2")

    def test_v1_result_is_recomputed_from_persisted_config(self):
        configs = {"piqa": {"metric": "<function acc at 0x1234abcd>"}}
        payload = {
            "configs": configs,
            "paper_v3_protocol": {
                "task_configs_fingerprint": "stable-json-v1",
                "task_configs_sha256": "v1-runtime-only-digest",
                "task_configs_process_raw_sha256": "raw-process-digest",
            },
            "results": {"piqa": {"acc_norm,none": 1.0}},
            "samples": {"piqa": [{
                "doc_id": 0,
                "doc_hash": "doc",
                "target_hash": "target",
                "acc_norm": 1.0,
            }]},
        }
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "result.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            loaded = load_run(path)
        self.assertEqual(
            loaded["protocol"]["task_configs_sha256"],
            task_config_sha256(configs),
        )
        self.assertEqual(
            loaded["protocol"]["task_configs_fingerprint"],
            "stable-json-v2",
        )
        self.assertEqual(
            loaded["protocol"]["task_configs_process_raw_sha256"],
            "raw-process-digest",
        )


if __name__ == "__main__":
    unittest.main()
