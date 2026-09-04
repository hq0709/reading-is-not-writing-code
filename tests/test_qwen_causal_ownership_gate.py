from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.qwen_causal_ownership_gate import (
    ALPHA,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONCEPTS,
    EVAL_ROWS,
    N_RANDOM,
    PREVIOUS_COMMIT,
    PROMPTS,
    REGISTERED_ROWS_SHA256,
    _simultaneous_lower_bounds,
    _validate_previous_run_receipt,
    derive_registered_rows,
    summarize_ownership_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


class QwenCausalOwnershipGateTests(unittest.TestCase):
    def test_centered_max_t_bounds_are_scale_equivariant(self) -> None:
        estimates = np.asarray([1.0, 100.0])
        bootstrap = np.asarray(
            [
                [0.8, 80.0],
                [0.9, 90.0],
                [1.0, 100.0],
                [1.1, 110.0],
                [1.2, 120.0],
            ]
        )

        lower, critical = _simultaneous_lower_bounds(estimates, bootstrap)

        self.assertGreater(critical, 0.0)
        self.assertAlmostEqual(lower[0] * 100.0, lower[1], places=10)

    def test_registered_rows_exclude_both_prior_qwen_cohorts(self) -> None:
        manifest_path = Path(
            "/home/qingchan/data/concept-flow/datasets/nih-chestxray14/manifest.csv"
        )
        source_path = Path(
            "/home/qingchan/data/concept-flow/runs/"
            "20260903T000321Z-caaae3ef346d-qwen-full/artifacts/intervention-summary.json"
        )
        previous_path = Path(
            "/home/qingchan/data/concept-flow/runs/"
            "20260903T103508Z-456c81bad460-7a3edc4b/artifacts/registered-rows.json"
        )
        if not all(path.is_file() for path in (manifest_path, source_path, previous_path)):
            self.skipTest("server evidence is unavailable")
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            manifest = list(csv.DictReader(stream))
        discovery = json.loads(source_path.read_text(encoding="utf-8"))["eval_row_ids"]
        previous = json.loads(previous_path.read_text(encoding="utf-8"))["row_ids"]

        selected, digest = derive_registered_rows(manifest, discovery, previous)

        by_id = {row["row_id"]: row for row in manifest}
        excluded_patients = {by_id[row_id]["patient_id"] for row_id in discovery + previous}
        self.assertEqual(EVAL_ROWS, len(selected))
        self.assertEqual(EVAL_ROWS, len({row["patient_id"] for row in selected}))
        self.assertFalse({row["patient_id"] for row in selected} & excluded_patients)
        self.assertEqual(REGISTERED_ROWS_SHA256, digest)

    def test_registered_rows_are_deterministic_and_patient_disjoint_portably(self) -> None:
        manifest = [
            {
                "row_id": "discovery-row",
                "patient_id": "discovery-patient",
                "split": "test",
            },
            {
                "row_id": "previous-row",
                "patient_id": "previous-patient",
                "split": "test",
            },
        ]
        manifest.extend(
            {
                "row_id": f"eligible-{index:04d}",
                "patient_id": f"patient-{index:04d}",
                "split": "test",
            }
            for index in range(EVAL_ROWS + 5)
        )

        selected, digest = derive_registered_rows(manifest, ["discovery-row"], ["previous-row"])
        replay, replay_digest = derive_registered_rows(
            list(reversed(manifest)), ["discovery-row"], ["previous-row"]
        )

        self.assertEqual(EVAL_ROWS, len(selected))
        self.assertEqual(EVAL_ROWS, len({row["patient_id"] for row in selected}))
        self.assertNotIn("discovery-patient", {row["patient_id"] for row in selected})
        self.assertNotIn("previous-patient", {row["patient_id"] for row in selected})
        self.assertEqual([row["row_id"] for row in selected], [row["row_id"] for row in replay])
        self.assertEqual(digest, replay_digest)

    def test_diagonal_matrix_reports_six_owned_concepts(self) -> None:
        records, row_ids, patient_ids = self._records(alias_direction=None)
        with TemporaryDirectory() as temp:
            summary = summarize_ownership_matrix(
                records,
                row_ids,
                patient_ids,
                Path(temp) / "bootstrap.npz",
            )

        self.assertEqual(BOOTSTRAP_RESAMPLES, summary["bootstrap_resamples"])
        self.assertEqual(BOOTSTRAP_SEED, summary["bootstrap_seed"])
        self.assertEqual(6, summary["n_owned_concepts"])
        self.assertEqual(0, summary["n_off_diagonal_winners"])
        self.assertFalse(summary["shared_alias"]["detected"])
        self.assertIsNone(summary["shared_alias"]["direction"])
        self.assertEqual(set(CONCEPTS), set(summary["shared_alias"]["by_direction"]))
        self.assertFalse(
            any(item["qualified"] for item in summary["shared_alias"]["by_direction"].values())
        )
        for concept in CONCEPTS:
            self.assertGreater(summary["columns"][concept]["ownership_margin"], 0.10)
            self.assertTrue(summary["columns"][concept]["owned"])

    def test_shared_nodule_axis_is_selected_inside_the_bootstrap(self) -> None:
        records, row_ids, patient_ids = self._records(alias_direction="Nodule")
        with TemporaryDirectory() as temp:
            summary = summarize_ownership_matrix(
                records,
                row_ids,
                patient_ids,
                Path(temp) / "bootstrap.npz",
            )

        self.assertEqual("Nodule", summary["shared_alias"]["direction"])
        self.assertGreater(summary["shared_alias"]["relative_dominance_simultaneous_lower_95"], 0.0)
        self.assertGreater(
            summary["shared_alias"]["off_diagonal_effect_simultaneous_lower_95"], 0.0
        )
        self.assertTrue(summary["shared_alias"]["detected"])
        self.assertTrue(summary["shared_alias"]["by_direction"]["Nodule"]["qualified"])
        self.assertEqual(5, summary["n_off_diagonal_winners"])
        self.assertFalse(summary["columns"]["Effusion"]["owned"])
        self.assertEqual("Nodule", summary["columns"]["Effusion"]["winner"])

    def test_all_negative_relative_dominance_is_not_a_causal_alias(self) -> None:
        records, row_ids, patient_ids = self._records(alias_direction="Nodule")
        for question_records in records.values():
            for record in question_records:
                if float(record["alpha"]) == 0.0:
                    record["p_yes"] = "0.5"
                elif record["direction"] == "concept":
                    record["p_yes"] = "0.3"
                elif record["direction"] == "unrelated_Nodule":
                    record["p_yes"] = "0.45"
                elif record["direction"].startswith("unrelated_"):
                    record["p_yes"] = "0.35"
                else:
                    record["p_yes"] = "0.49"
        with TemporaryDirectory() as temp:
            summary = summarize_ownership_matrix(
                records, row_ids, patient_ids, Path(temp) / "bootstrap.npz"
            )

        self.assertGreater(summary["shared_alias"]["relative_dominance"], 0.0)
        self.assertLess(summary["shared_alias"]["off_diagonal_effect"], 0.0)
        self.assertFalse(summary["shared_alias"]["detected"])

    def test_one_stronger_random_direction_blocks_column_ownership(self) -> None:
        records, row_ids, patient_ids = self._records(alias_direction=None)
        for record in records["Effusion"]:
            if record["direction"] == "random0":
                record["p_yes"] = "0.45"
        with TemporaryDirectory() as temp:
            summary = summarize_ownership_matrix(
                records, row_ids, patient_ids, Path(temp) / "bootstrap.npz"
            )

        self.assertFalse(summary["columns"]["Effusion"]["generic_selectivity"])
        self.assertFalse(summary["columns"]["Effusion"]["owned"])

    def test_global_null_selects_neither_ownership_nor_alias(self) -> None:
        records, row_ids, patient_ids = self._records(alias_direction=None)
        for question_records in records.values():
            for record in question_records:
                if record["direction"] == "concept" and float(record["alpha"]) == 0.0:
                    continue
                if record["direction"] == "concept" or record["direction"].startswith("unrelated_"):
                    record["p_yes"] = "0.3"
        with TemporaryDirectory() as temp:
            summary = summarize_ownership_matrix(
                records, row_ids, patient_ids, Path(temp) / "bootstrap.npz"
            )

        self.assertEqual(0, summary["n_owned_concepts"])
        self.assertEqual(0, summary["n_off_diagonal_winners"])
        self.assertFalse(summary["shared_alias"]["detected"])

    def test_incomplete_grid_is_rejected(self) -> None:
        records, row_ids, patient_ids = self._records(alias_direction=None)
        records["Effusion"].pop()
        with TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, "grid"):
            summarize_ownership_matrix(
                records,
                row_ids,
                patient_ids,
                Path(temp) / "bootstrap.npz",
            )

    def test_runner_fixes_the_six_question_grid_and_fresh_cohort(self) -> None:
        runner = (ROOT / "scripts/server/run_qwen_causal_ownership_gate.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "20260903T000321Z-caaae3ef346d-qwen-full",
            "20260903T103508Z-456c81bad460-7a3edc4b",
            "--loci vis.last",
            "--alphas=0,0.25",
            "--control-alphas=0.25",
            "--n-random 119",
            "--n-eval 400",
            "--eval-row-ids-file",
            "Effusion|Atelectasis|Pneumothorax|Cardiomegaly|Mass|Nodule",
            "src.qwen_mechanism_route",
            "causal-ownership-summary.json",
            "mechanism-route.json",
        ):
            self.assertIn(token, runner)
        self.assertIn("printf '%s\\n' \"$concepts\"", runner)
        for prompt in PROMPTS.values():
            self.assertIn(prompt, runner)

    def test_previous_run_receipt_requires_terminal_status_and_pinned_hashes(self) -> None:
        with TemporaryDirectory() as temp:
            run = Path(temp)
            artifacts = run / "artifacts"
            artifacts.mkdir()
            files = {
                "./metadata.env": (
                    f"SOURCE_COMMIT={PREVIOUS_COMMIT}\n"
                    "COMMAND_STATUS=0\nDISPATCHER_STATUS=0\n"
                    "CLEANUP_STATUS=0\nABORT_SIGNAL=none\n"
                ),
                "./command_exit_status": "0\n",
                "./exit_status": "0\n",
                "./artifacts/registered-rows.json": "{}\n",
                "./artifacts/direction-specificity-summary.json": "{}\n",
            }
            expected = {}
            for relative, content in files.items():
                path = run / relative.removeprefix("./")
                path.write_text(content, encoding="utf-8")
                expected[relative] = hashlib.sha256(content.encode()).hexdigest()
            (run / "SHA256SUMS").write_text(
                "".join(f"{digest}  {relative}\n" for relative, digest in expected.items()),
                encoding="utf-8",
            )

            with patch("src.qwen_causal_ownership_gate.PREVIOUS_SHA256", expected):
                _validate_previous_run_receipt(run)
                (run / "exit_status").write_text("1\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "dispatcher failed"):
                    _validate_previous_run_receipt(run)

    @staticmethod
    def _records(
        alias_direction: str | None,
    ) -> tuple[dict[str, list[dict[str, str]]], list[str], list[str]]:
        row_ids = [f"row-{index}" for index in range(EVAL_ROWS)]
        patient_ids = [f"patient-{index}" for index in range(EVAL_ROWS)]
        records = {concept: [] for concept in CONCEPTS}
        for question in CONCEPTS:
            for patient_index, (row_id, patient_id) in enumerate(
                zip(row_ids, patient_ids, strict=True)
            ):
                records[question].append(
                    {
                        "row_id": row_id,
                        "patient_id": patient_id,
                        "locus": "vis.last",
                        "direction": "concept",
                        "alpha": "0.0",
                        "p_yes": "0.2",
                    }
                )
                direction_effects = {concept: 0.08 for concept in CONCEPTS}
                direction_effects[question] = 0.20
                if alias_direction is not None and question != alias_direction:
                    direction_effects[alias_direction] = 0.27
                for direction_index, (direction, effect) in enumerate(direction_effects.items()):
                    effect += 0.01 * (((patient_index + 3 * direction_index) % 9) - 4) / 4
                    stored_direction = (
                        "concept" if direction == question else f"unrelated_{direction}"
                    )
                    records[question].append(
                        {
                            "row_id": row_id,
                            "patient_id": patient_id,
                            "locus": "vis.last",
                            "direction": stored_direction,
                            "alpha": str(ALPHA),
                            "p_yes": str(0.2 + effect),
                        }
                    )
                for index in range(N_RANDOM):
                    records[question].append(
                        {
                            "row_id": row_id,
                            "patient_id": patient_id,
                            "locus": "vis.last",
                            "direction": f"random{index}",
                            "alpha": str(ALPHA),
                            "p_yes": "0.24",
                        }
                    )
                records[question].append(
                    {
                        "row_id": row_id,
                        "patient_id": patient_id,
                        "locus": "vis.last",
                        "direction": "sham",
                        "alpha": str(ALPHA),
                        "p_yes": "0.23",
                    }
                )
        return records, row_ids, patient_ids


if __name__ == "__main__":
    unittest.main()
