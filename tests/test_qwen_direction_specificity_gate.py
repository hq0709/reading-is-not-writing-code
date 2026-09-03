from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.qwen_direction_specificity_gate import (
    ALPHA,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EVAL_ROWS,
    REGISTERED_ROWS_SHA256,
    derive_registered_rows,
    summarize_direction_specificity,
)

ROOT = Path(__file__).resolve().parents[1]


class QwenDirectionSpecificityGateTests(unittest.TestCase):
    def test_registered_rows_are_exact_patient_disjoint_and_stable(self) -> None:
        manifest = Path(
            "/home/qingchan/data/concept-flow/datasets/nih-chestxray14/manifest.csv"
        )
        source_summary = Path(
            "/home/qingchan/data/concept-flow/runs/"
            "20260903T000321Z-caaae3ef346d-qwen-full/artifacts/intervention-summary.json"
        )
        if not manifest.is_file() or not source_summary.is_file():
            self.skipTest("server evidence is unavailable")
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        discovery = json.loads(source_summary.read_text(encoding="utf-8"))["eval_row_ids"]

        selected, digest = derive_registered_rows(rows, discovery)

        by_id = {row["row_id"]: row for row in rows}
        discovery_patients = {by_id[row_id]["patient_id"] for row_id in discovery}
        self.assertEqual(EVAL_ROWS, len(selected))
        self.assertEqual(EVAL_ROWS, len({row["patient_id"] for row in selected}))
        self.assertFalse({row["patient_id"] for row in selected} & discovery_patients)
        self.assertEqual(REGISTERED_ROWS_SHA256, digest)

    def test_primary_margin_places_the_fixed_max_inside_each_bootstrap(self) -> None:
        row_ids = [f"row-{index}" for index in range(EVAL_ROWS)]
        patient_ids = [f"patient-{index}" for index in range(EVAL_ROWS)]
        records: list[dict[str, str]] = []

        def add(direction: str, alpha: float, offset: float) -> None:
            for row_id, patient_id in zip(row_ids, patient_ids, strict=True):
                records.append(
                    {
                        "row_id": row_id,
                        "patient_id": patient_id,
                        "locus": "vis.last",
                        "direction": direction,
                        "alpha": str(alpha),
                        "p_yes": str(0.2 + offset),
                        "margin": "0.0",
                    }
                )

        add("concept", 0.0, 0.0)
        add("concept", ALPHA, 0.20)
        for index in range(20):
            add(f"random{index}", ALPHA, 0.04)
        add("sham", ALPHA, 0.03)
        for name in ("Atelectasis", "Pneumothorax", "Cardiomegaly", "Mass", "Nodule"):
            add(f"unrelated_{name}", ALPHA, 0.10)

        with TemporaryDirectory() as temp:
            bootstrap = Path(temp) / "bootstrap.npz"
            summary = summarize_direction_specificity(
                records, row_ids, patient_ids, bootstrap
            )

        self.assertEqual(BOOTSTRAP_RESAMPLES, summary["bootstrap_resamples"])
        self.assertEqual(BOOTSTRAP_SEED, summary["bootstrap_seed"])
        self.assertAlmostEqual(0.10, summary["primary"]["margin"])
        self.assertGreater(summary["primary"]["one_sided_lower_95"], 0.0)
        self.assertTrue(summary["replicated_random_sham_selectivity"])
        self.assertTrue(summary["direction_specific"])

    def test_direction_specificity_fails_when_one_unrelated_normal_is_larger(self) -> None:
        row_ids = [f"row-{index}" for index in range(EVAL_ROWS)]
        patient_ids = [f"patient-{index}" for index in range(EVAL_ROWS)]
        records: list[dict[str, str]] = []
        directions = {
            "concept": 0.20,
            **{f"random{index}": 0.04 for index in range(20)},
            "sham": 0.03,
            "unrelated_Atelectasis": 0.10,
            "unrelated_Pneumothorax": 0.10,
            "unrelated_Cardiomegaly": 0.10,
            "unrelated_Mass": 0.10,
            "unrelated_Nodule": 0.25,
        }
        for row_id, patient_id in zip(row_ids, patient_ids, strict=True):
            records.append(
                {
                    "row_id": row_id,
                    "patient_id": patient_id,
                    "locus": "vis.last",
                    "direction": "concept",
                    "alpha": "0.0",
                    "p_yes": "0.2",
                    "margin": "0.0",
                }
            )
            for direction, effect in directions.items():
                records.append(
                    {
                        "row_id": row_id,
                        "patient_id": patient_id,
                        "locus": "vis.last",
                        "direction": direction,
                        "alpha": str(ALPHA),
                        "p_yes": str(0.2 + effect),
                        "margin": "0.0",
                    }
                )
        with TemporaryDirectory() as temp:
            summary = summarize_direction_specificity(
                records, row_ids, patient_ids, Path(temp) / "bootstrap.npz"
            )
        self.assertAlmostEqual(-0.05, summary["primary"]["margin"])
        self.assertFalse(summary["direction_specific"])

    def test_runner_fixes_the_registered_grid_and_immutable_payload(self) -> None:
        runner = (ROOT / "scripts/server/run_qwen_direction_specificity_gate.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "20260903T000321Z-caaae3ef346d-qwen-full",
            "--arch qwen7b",
            "--concept Effusion",
            "--loci vis.last",
            "--alphas=0,0.25",
            "--control-alphas=0.25",
            "--n-random 20",
            "--n-eval 400",
            "--eval-split test",
            "--eval-row-ids-file",
            "--per-image-out",
            "Is there a pleural effusion in this chest radiograph? Answer yes or no.",
        ):
            self.assertIn(token, runner)


if __name__ == "__main__":
    unittest.main()
