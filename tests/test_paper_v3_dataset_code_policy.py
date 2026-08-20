import types
import unittest

from src.dataset_code_policy import configure_dataset_code_trust


class PaperV3DatasetCodePolicyTests(unittest.TestCase):
    def test_mathqa_fails_closed_without_explicit_authorization(self):
        with self.assertRaisesRegex(ValueError, "mathqa"):
            configure_dataset_code_trust(
                ["hellaswag", "mathqa"], allow=False,
                datasets_module=types.SimpleNamespace(),
            )

    def test_authorized_mathqa_enables_datasets_loader_and_records_scope(self):
        fake = types.SimpleNamespace(
            config=types.SimpleNamespace(HF_DATASETS_TRUST_REMOTE_CODE=False)
        )
        policy = configure_dataset_code_trust(
            ["hellaswag", "mathqa"], allow=True, datasets_module=fake,
        )
        self.assertTrue(fake.config.HF_DATASETS_TRUST_REMOTE_CODE)
        self.assertEqual(policy, {
            "trust_dataset_code": True,
            "dataset_code_tasks": ["mathqa"],
        })

    def test_tasks_without_loading_code_do_not_enable_global_trust(self):
        fake = types.SimpleNamespace(
            config=types.SimpleNamespace(HF_DATASETS_TRUST_REMOTE_CODE=False)
        )
        policy = configure_dataset_code_trust(
            ["hellaswag", "piqa"], allow=False, datasets_module=fake,
        )
        self.assertFalse(fake.config.HF_DATASETS_TRUST_REMOTE_CODE)
        self.assertEqual(policy, {
            "trust_dataset_code": False,
            "dataset_code_tasks": [],
        })


if __name__ == "__main__":
    unittest.main()
