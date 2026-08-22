import json
import os
import tempfile
import unittest

from scripts.summarize_paper_v3_downstream import (
    load_run, load_tokenizer_audit_registry, validate_tokenizer_audit_coverage,
)
from src.experiment_provenance import file_sha256
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

    def test_multiple_tokenizer_audits_are_verified_per_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            registry_paths = []
            file_hash = "tokenizer-files"
            for index, label in enumerate(("baseline", "target2")):
                path = os.path.join(root, f"audit{index}.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump({
                        "decision": {
                            "audit_passed_for_downstream": True,
                            "selected_tokenizer_mode": "current",
                            "use_fix_mistral_regex_for_future_evaluation": False,
                        },
                        "sources": [{
                            "label": label,
                            "tokenizer_files_combined_sha256": file_hash,
                        }],
                    }, handle)
                registry_paths.append(path)
            registry = load_tokenizer_audit_registry(registry_paths)
            loaded = {}
            for label, path in zip(("baseline", "target2"), registry_paths):
                loaded[label] = {"protocol": {
                    "tokenizer_audit_sha256": file_sha256(path),
                    "selected_tokenizer_mode": "current",
                    "fix_mistral_regex": False,
                    "tokenizer_files_combined_sha256": file_hash,
                }}
            coverage = validate_tokenizer_audit_coverage(loaded, registry)
        self.assertEqual(set(coverage), {"baseline", "target2"})

    def test_tokenizer_audit_must_cover_recorded_checkpoint_label(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "audit.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({
                    "decision": {
                        "audit_passed_for_downstream": True,
                        "selected_tokenizer_mode": "current",
                        "use_fix_mistral_regex_for_future_evaluation": False,
                    },
                    "sources": [{
                        "label": "baseline",
                        "tokenizer_files_combined_sha256": "files",
                    }],
                }, handle)
            registry = load_tokenizer_audit_registry([path])
            loaded = {"target2": {"protocol": {
                "tokenizer_audit_sha256": file_sha256(path),
                "selected_tokenizer_mode": "current",
                "fix_mistral_regex": False,
                "tokenizer_files_combined_sha256": "files",
            }}}
            with self.assertRaisesRegex(ValueError, "does not cover checkpoint"):
                validate_tokenizer_audit_coverage(loaded, registry)


if __name__ == "__main__":
    unittest.main()
