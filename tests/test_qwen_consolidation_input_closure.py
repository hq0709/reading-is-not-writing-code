from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.qwen_consolidation_input_closure import (
    BOOTSTRAP_RESAMPLES,
    CLINICAL_CONTROLS,
    EVAL_PAIRS,
    N_RANDOM,
    REGISTERED_PAIRS_SHA256,
    derive_registered_pairs,
    summarize_input_closure,
)


class QwenConsolidationInputClosureTests(unittest.TestCase):
    @staticmethod
    def _manifest() -> tuple[list[dict[str, str]], list[str]]:
        rows: list[dict[str, str]] = []
        excluded: list[str] = []
        for index in range(EVAL_PAIRS + 8):
            patient = f"patient-{index:03d}"
            rows.extend(
                [
                    {
                        "row_id": f"{patient}-positive-a",
                        "patient_id": patient,
                        "split": "test",
                        "Consolidation": "1",
                    },
                    {
                        "row_id": f"{patient}-positive-b",
                        "patient_id": patient,
                        "split": "test",
                        "Consolidation": "1",
                    },
                    {
                        "row_id": f"{patient}-negative-a",
                        "patient_id": patient,
                        "split": "test",
                        "Consolidation": "0",
                    },
                ]
            )
        rows.extend(
            [
                {
                    "row_id": "train-positive",
                    "patient_id": "train-patient",
                    "split": "train",
                    "Consolidation": "1",
                },
                {
                    "row_id": "test-positive-only",
                    "patient_id": "positive-only",
                    "split": "test",
                    "Consolidation": "1",
                },
                {
                    "row_id": "excluded-positive",
                    "patient_id": "excluded-patient",
                    "split": "test",
                    "Consolidation": "1",
                },
                {
                    "row_id": "excluded-negative",
                    "patient_id": "excluded-patient",
                    "split": "test",
                    "Consolidation": "0",
                },
            ]
        )
        excluded.append("excluded-positive")
        return rows, excluded

    def test_registered_pairs_are_deterministic_fresh_and_within_patient(self) -> None:
        rows, excluded = self._manifest()
        selected, digest = derive_registered_pairs(rows, excluded)
        reversed_selected, reversed_digest = derive_registered_pairs(list(reversed(rows)), excluded)

        self.assertEqual(EVAL_PAIRS, len(selected))
        self.assertEqual(selected, reversed_selected)
        self.assertEqual(digest, reversed_digest)
        self.assertEqual(EVAL_PAIRS, len({item["patient_id"] for item in selected}))
        self.assertNotIn("excluded-patient", {item["patient_id"] for item in selected})
        by_id = {row["row_id"]: row for row in rows}
        for item in selected:
            self.assertEqual(item["patient_id"], by_id[item["positive_row_id"]]["patient_id"])
            self.assertEqual(item["patient_id"], by_id[item["negative_row_id"]]["patient_id"])
            self.assertEqual("1", by_id[item["positive_row_id"]]["Consolidation"])
            self.assertEqual("0", by_id[item["negative_row_id"]]["Consolidation"])

    def test_real_registered_pair_digest_is_pinned(self) -> None:
        manifest = Path("/home/qingchan/data/concept-flow/datasets/nih-chestxray14/manifest.csv")
        if not manifest.is_file():
            self.skipTest("server NIH manifest unavailable")
        from src.qwen_consolidation_input_closure import load_prior_row_ids

        prior = load_prior_row_ids(Path("/home/qingchan/data/concept-flow/runs"))
        import csv

        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        selected, digest = derive_registered_pairs(rows, prior)
        self.assertEqual(REGISTERED_PAIRS_SHA256, digest)
        self.assertEqual(EVAL_PAIRS, len(selected))

    @staticmethod
    def _closure_inputs(strong_unrelated: bool = False):
        n = EVAL_PAIRS
        displacement = np.linspace(0.35, 0.65, n)
        positive = np.linspace(0.72, 0.82, n)
        negative = np.linspace(0.18, 0.28, n)
        steered = {"concept": positive - 0.02}
        for index in range(N_RANDOM):
            steered[f"random{index}"] = negative + 0.01 + index * 1e-5
        steered["sham"] = negative + 0.02
        for index, concept in enumerate(CLINICAL_CONTROLS):
            steered[f"unrelated_{concept}"] = negative + 0.03 + 0.002 * index
        if strong_unrelated:
            steered["unrelated_Effusion"] = positive - 0.005
        return displacement, positive, negative, steered

    def test_concept_closure_clears_registered_controls(self) -> None:
        displacement, positive, negative, steered = self._closure_inputs()
        summary, arrays = summarize_input_closure(
            displacement,
            positive,
            negative,
            steered,
            availability_eligible=True,
            alpha=np.full(EVAL_PAIRS, 0.1),
        )

        self.assertTrue(summary["input_closure"])
        self.assertEqual("Nodule", summary["maximum_unrelated"]["concept"])
        self.assertGreater(summary["input_displacement"]["one_sided_lower_95"], 0.0)
        self.assertGreater(summary["clinical_familywise_margin"]["one_sided_lower_95"], 0.0)
        self.assertEqual(1 / (N_RANDOM + 1), summary["exact_random_rank_p"])
        self.assertEqual((BOOTSTRAP_RESAMPLES, EVAL_PAIRS), arrays["patient_draw_indices"].shape)

    def test_stronger_unrelated_direction_blocks_input_closure(self) -> None:
        displacement, positive, negative, steered = self._closure_inputs(strong_unrelated=True)
        summary, _ = summarize_input_closure(
            displacement,
            positive,
            negative,
            steered,
            availability_eligible=True,
            alpha=np.full(EVAL_PAIRS, 0.1),
        )

        self.assertFalse(summary["input_closure"])
        self.assertEqual("Effusion", summary["maximum_unrelated"]["concept"])
        self.assertLess(summary["clinical_familywise_margin"]["estimate"], 0.0)

    def test_unavailable_probe_blocks_input_closure(self) -> None:
        displacement, positive, negative, steered = self._closure_inputs()
        summary, _ = summarize_input_closure(
            displacement,
            positive,
            negative,
            steered,
            availability_eligible=False,
            alpha=np.full(EVAL_PAIRS, 0.1),
        )
        self.assertFalse(summary["input_closure"])

    def test_incomplete_or_nonfinite_grid_is_rejected(self) -> None:
        displacement, positive, negative, steered = self._closure_inputs()
        del steered["random0"]
        with self.assertRaisesRegex(ValueError, "direction grid"):
            summarize_input_closure(
                displacement,
                positive,
                negative,
                steered,
                availability_eligible=True,
                alpha=np.full(EVAL_PAIRS, 0.1),
            )

        displacement, positive, negative, steered = self._closure_inputs()
        steered["concept"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_input_closure(
                displacement,
                positive,
                negative,
                steered,
                availability_eligible=True,
                alpha=np.full(EVAL_PAIRS, 0.1),
            )

    def test_runner_fixes_registered_payload_and_artifacts(self) -> None:
        runner = Path("scripts/server/run_qwen_consolidation_input_closure_gate.sh")
        text = runner.read_text(encoding="utf-8")
        self.assertTrue(os.access(runner, os.X_OK))
        for token in (
            "20260903T000321Z-caaae3ef346d-qwen-full",
            "20260904T125710Z-a3bd883540eb-causal-ownership",
            "qwen_consolidation_input_closure.py",
            "validate-source",
            "registered-pairs.json",
            "probe",
            "directions.npz",
            "evaluate",
            "input-closure-bootstrap.npz",
            "input-closure-summary.json",
        ):
            self.assertIn(token, text)

if __name__ == "__main__":
    unittest.main()
