import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts.server import validate_llava_readout_diagnostic as validator


class TerminalHelpersTests(unittest.TestCase):
    def test_manifest_parser_rejects_duplicates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "SHA256SUMS"
            path.write_text("a" * 64 + "  ./metadata.env\n")
            self.assertEqual(validator.inventory(path), {"metadata.env": "a" * 64})
            path.write_text("a" * 64 + "  metadata.env\n" + "b" * 64 + "  metadata.env\n")
            with self.assertRaises(AssertionError):
                validator.inventory(path)

    def test_independent_contrast_reference_keeps_constant_members(self):
        values = np.asarray([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]])
        np.testing.assert_allclose(validator.contrasts(values), [1.0, 8 / 3, 2.5, -2.0, 1.0, 0.0])

    def test_tie_aware_auc(self):
        scores = np.asarray([0.0, 0.0, 1.0, 1.0])
        labels = np.asarray([0, 1, 0, 1])
        counts = np.ones((1, 4), dtype=np.int64)
        np.testing.assert_allclose(validator.auc(scores, labels, counts), [0.5])

    def test_altered_registered_bootstrap_is_rejected(self):
        with patch.object(validator, "N", 4), patch.object(validator, "B", 3):
            expected = np.random.default_rng(20260917).integers(0, 4, size=(3, 4), dtype=np.int64)
            np.testing.assert_array_equal(validator.validate_bootstrap(expected), expected)
            changed = expected.copy()
            changed[0, 0] = (changed[0, 0] + 1) % 4
            with self.assertRaisesRegex(AssertionError, "bootstrap index identity"):
                validator.validate_bootstrap(changed)

    def test_independent_semantic_preflight_rejects_statement_drift(self):
        token_ids = {}
        for condition, encoding in zip(validator.CONDITIONS, validator.ENCODINGS, strict=True):
            first, second = validator.TOKEN_GROUPS[encoding]
            present, absent = (second, first) if encoding == "B" else (first, second)
            token_ids[condition] = {
                "raw_first": first,
                "raw_second": second,
                "present": present,
                "absent": absent,
                "candidate_ids": sorted(set(first + second)),
            }
        cases = []
        for identity in validator.expected_mapping_cases():
            sign = 2 * identity["expected_present"] - 1
            record = {**identity, "semantic_margin": float(sign)}
            if identity["encoding"] == "Y":
                record["lse_margin"] = float(sign)
            cases.append(record)
        pilot_seconds, elapsed_seconds = 1.0, 10.0
        projected = elapsed_seconds + 8406 * pilot_seconds / 512 + 600
        preflight = {
            "validation_row_ids": list(validator.VALIDATION_ROW_IDS),
            "module": "model.vision_tower.encoder.layers.22",
            "pooling": "mean over all 577 block-output tokens",
            "forwarded_no_cls_exact": True,
            "clean_repeat_exact": True,
            "token_ids": token_ids,
            "mapping_cases": cases,
            "throughput": {
                "pilot_equivalents": 512,
                "pilot_seconds": pilot_seconds,
                "elapsed_seconds": elapsed_seconds,
                "scientific_outcomes": 8406,
                "projected_seconds": projected,
                "limit_seconds": 3600,
                "passed": True,
            },
        }
        validator.validate_semantic_preflight(preflight)
        preflight["mapping_cases"][0]["statement"] = "Changed."
        with self.assertRaisesRegex(AssertionError, "semantic mapping identity"):
            validator.validate_semantic_preflight(preflight)


if __name__ == "__main__":
    unittest.main()
