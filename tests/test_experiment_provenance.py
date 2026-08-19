import os
import tempfile
import unittest

from src.experiment_provenance import (
    assert_calibration_evaluation_disjoint,
    build_text_manifest,
    file_sha256,
)


class ExperimentProvenanceTests(unittest.TestCase):
    def test_disjoint_manifest_has_stable_sample_identifiers(self):
        manifest = build_text_manifest(
            {"wikitext2": ["evaluation example one", "evaluation example two"]},
            ["calibration prompt"],
        )
        assert_calibration_evaluation_disjoint(manifest)
        self.assertTrue(manifest["disjointness"]["verified_disjoint"])
        sample = manifest["evaluation"]["wikitext2"]["samples"][0]
        self.assertTrue(sample["sample_id"].startswith("wikitext2:0:"))

    def test_normalized_overlap_fails(self):
        manifest = build_text_manifest(
            {"c4": ["same   text"]}, ["same text"]
        )
        with self.assertRaisesRegex(AssertionError, "overlap"):
            assert_calibration_evaluation_disjoint(manifest)

    def test_file_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "plan.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("one")
            first = file_sha256(path)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("two")
            self.assertNotEqual(first, file_sha256(path))


if __name__ == "__main__":
    unittest.main()
