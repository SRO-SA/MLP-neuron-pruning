import unittest

from src.task_config_fingerprint import stable_task_config, task_config_sha256


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


if __name__ == "__main__":
    unittest.main()
