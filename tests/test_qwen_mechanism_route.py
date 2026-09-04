from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
CONCEPTS = (
    "Effusion",
    "Atelectasis",
    "Pneumothorax",
    "Cardiomegaly",
    "Mass",
    "Nodule",
)
INTERVENTION_COMMIT = "1234567890abcdef1234567890abcdef12345678"


def _summary() -> dict:
    effect_matrix = {
        direction: {question: 0.30 if direction == question else 0.10 for question in CONCEPTS}
        for direction in CONCEPTS
    }
    columns = {
        concept: {
            "winner": concept,
            "diagonal_effect": 0.30,
            "maximum_off_diagonal_effect": 0.10,
            "ownership_margin": 0.20,
            "ownership_descriptive_ci95": [0.15, 0.25],
            "ownership_simultaneous_lower_95": -0.01,
            "random_effect_max": 0.40,
            "exact_random_rank_p": 1.0,
            "absolute_sham_effect": 0.01,
            "generic_selectivity": False,
            "owned": False,
        }
        for concept in CONCEPTS
    }
    by_direction = {
        direction: {
            "relative_dominance": -0.20,
            "relative_dominance_simultaneous_lower_95": -0.25,
            "off_diagonal_effect": 0.10,
            "off_diagonal_effect_simultaneous_lower_95": -0.01,
            "qualified": False,
        }
        for direction in CONCEPTS
    }
    prompts = {
        "Effusion": "Is there a pleural effusion in this chest radiograph? Answer yes or no.",
        "Atelectasis": "Is there atelectasis in this chest radiograph? Answer yes or no.",
        "Pneumothorax": "Is there a pneumothorax in this chest radiograph? Answer yes or no.",
        "Cardiomegaly": "Is there cardiomegaly in this chest radiograph? Answer yes or no.",
        "Mass": "Is there a lung mass in this chest radiograph? Answer yes or no.",
        "Nodule": "Is there a lung nodule in this chest radiograph? Answer yes or no.",
    }
    question_evidence = {
        concept: {
            "per_image_sha256": "a" * 64,
            "meta_sha256": "b" * 64,
            "completion_marker_sha256": "c" * 64,
        }
        for concept in CONCEPTS
    }
    return {
        "arch": "qwen7b",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "locus": "vis.last",
        "module": "model.visual.blocks.31",
        "alpha_mode": "reltoken",
        "alpha": 0.25,
        "concepts": list(CONCEPTS),
        "n_eval": 400,
        "n_eval_patients": 400,
        "bootstrap_unit": "patient",
        "bootstrap_resamples": 5000,
        "bootstrap_seed": 20260904,
        "n_random": 119,
        "familywise_alpha": 0.05,
        "source_run_id": "20260903T000321Z-caaae3ef346d-qwen-full",
        "source_commit": "caaae3ef346d3ac01c76c53ba99b3b7237066639",
        "previous_run_id": "20260903T103508Z-456c81bad460-7a3edc4b",
        "intervention_commit": INTERVENTION_COMMIT,
        "registered_rows_sha256": (
            "e657a9bd1707f13695b02a341047af6f53981295b840376eb0ea5bbebfe2d312"
        ),
        "prompts": prompts,
        "evidence": {
            "source_reference_sha256": "d" * 64,
            "registered_rows_receipt_sha256": "e" * 64,
            "bootstrap_sha256": "f" * 64,
            "questions": question_evidence,
        },
        "random_control_rule": (
            "each diagonal must beat all 119 random directions; its minimum exact "
            "one-sided rank p-value is 1/120 and Bonferroni controls six columns at 0.05"
        ),
        "simultaneous_bound": (
            "studentized centered max-T bootstrap across all 30 fixed clinical ownership "
            "contrasts and, separately, the 12 shared-alias dominance and positive-effect "
            "statistics"
        ),
        "matrix_orientation": ("rows are steering directions; columns are clinical questions"),
        "effect_matrix": effect_matrix,
        "columns": columns,
        "n_owned_concepts": 0,
        "n_off_diagonal_winners": 0,
        "shared_alias": {
            "statistic": (
                "a fixed direction's mean off-diagonal excess over each question's named "
                "direction, selected under simultaneous dominance and positive-effect bounds"
            ),
            "direction": None,
            "candidate_direction": "Effusion",
            "relative_dominance": -0.20,
            "relative_dominance_simultaneous_lower_95": -0.25,
            "off_diagonal_effect": 0.10,
            "off_diagonal_effect_simultaneous_lower_95": -0.01,
            "global_random_effect_max": 0.12,
            "global_absolute_sham_effect_max": 0.02,
            "by_direction": by_direction,
            "detected": False,
        },
    }


def _refresh_column(summary: dict, question: str) -> None:
    effects = [summary["effect_matrix"][direction][question] for direction in CONCEPTS]
    winner_index = max(range(len(CONCEPTS)), key=effects.__getitem__)
    diagonal_index = CONCEPTS.index(question)
    off_diagonal = [value for index, value in enumerate(effects) if index != diagonal_index]
    column = summary["columns"][question]
    column["winner"] = CONCEPTS[winner_index]
    column["diagonal_effect"] = effects[diagonal_index]
    column["maximum_off_diagonal_effect"] = max(off_diagonal)
    column["ownership_margin"] = effects[diagonal_index] - max(off_diagonal)


class QwenMechanismRouteTests(unittest.TestCase):
    def _route_module(self):
        try:
            return importlib.import_module("src.qwen_mechanism_route")
        except ModuleNotFoundError:
            self.fail("src.qwen_mechanism_route must implement the route-lock")

    def test_shared_alias_has_priority_and_ties_use_fixed_question_order(self) -> None:
        summary = _summary()
        for question in CONCEPTS[:-1]:
            summary["effect_matrix"]["Nodule"][question] = 0.60
            _refresh_column(summary, question)
        summary["n_off_diagonal_winners"] = 5
        summary["columns"]["Nodule"].update(
            {
                "ownership_simultaneous_lower_95": 0.20,
                "random_effect_max": 0.05,
                "generic_selectivity": True,
                "owned": True,
            }
        )
        summary["n_owned_concepts"] = 1
        summary["shared_alias"].update(
            {
                "direction": "Nodule",
                "candidate_direction": "Nodule",
                "relative_dominance": 0.30,
                "relative_dominance_simultaneous_lower_95": 0.10,
                "off_diagonal_effect": 0.60,
                "off_diagonal_effect_simultaneous_lower_95": 0.20,
                "global_random_effect_max": 0.15,
                "global_absolute_sham_effect_max": 0.03,
                "detected": True,
            }
        )
        summary["shared_alias"]["by_direction"]["Nodule"].update(
            {
                "relative_dominance": 0.30,
                "relative_dominance_simultaneous_lower_95": 0.10,
                "off_diagonal_effect": 0.60,
                "off_diagonal_effect_simultaneous_lower_95": 0.20,
                "qualified": True,
            }
        )

        plan = self._route_module().build_route_plan(summary)

        self.assertEqual("shared_alias", plan["selected_route"])
        self.assertEqual("Nodule", plan["selection"]["direction"])
        self.assertEqual("Effusion", plan["selection"]["question"])
        self.assertEqual(0.60, plan["selection"]["effect"])
        self.assertEqual(
            [
                "output_jacobian_alignment",
                "answer_encoding_replacement",
                "projection_removal_intervention",
            ],
            plan["mechanism_family"],
        )
        self.assertEqual(
            "maximum off-diagonal effect, then fixed CONCEPTS order",
            plan["selection"]["deterministic_tie_break"],
        )
        self.assertFalse(plan["changes_causal_ownership_result"])

    def test_detected_alias_may_trail_an_unqualified_raw_score_candidate(self) -> None:
        summary = _summary()
        for question in CONCEPTS[1:]:
            summary["effect_matrix"]["Effusion"][question] = 0.70
        for question in CONCEPTS[:-1]:
            summary["effect_matrix"]["Nodule"][question] = 0.60
        for question in CONCEPTS:
            _refresh_column(summary, question)
        summary["n_off_diagonal_winners"] = 6
        summary["shared_alias"].update(
            {
                "direction": "Nodule",
                "candidate_direction": "Nodule",
                "relative_dominance": 0.30,
                "relative_dominance_simultaneous_lower_95": 0.10,
                "off_diagonal_effect": 0.60,
                "off_diagonal_effect_simultaneous_lower_95": 0.20,
                "global_random_effect_max": 0.15,
                "global_absolute_sham_effect_max": 0.03,
                "detected": True,
            }
        )
        summary["shared_alias"]["by_direction"]["Effusion"].update(
            {
                "relative_dominance": 0.40,
                "off_diagonal_effect": 0.70,
            }
        )
        summary["shared_alias"]["by_direction"]["Nodule"].update(
            {
                "relative_dominance": 0.30,
                "relative_dominance_simultaneous_lower_95": 0.10,
                "off_diagonal_effect": 0.60,
                "off_diagonal_effect_simultaneous_lower_95": 0.20,
                "qualified": True,
            }
        )

        plan = self._route_module().build_route_plan(summary)

        self.assertEqual("shared_alias", plan["selected_route"])
        self.assertEqual("Nodule", plan["selection"]["direction"])

    def test_owned_route_uses_strongest_simultaneous_lower_bound_then_fixed_order(self) -> None:
        summary = _summary()
        for concept in ("Effusion", "Atelectasis"):
            summary["columns"][concept].update(
                {
                    "ownership_simultaneous_lower_95": 0.18,
                    "random_effect_max": 0.05,
                    "generic_selectivity": True,
                    "owned": True,
                }
            )
        summary["n_owned_concepts"] = 2

        plan = self._route_module().build_route_plan(summary)

        self.assertEqual("diagonal_ownership", plan["selected_route"])
        self.assertEqual("Effusion", plan["selection"]["concept"])
        self.assertEqual(0.18, plan["selection"]["ownership_simultaneous_lower_95"])
        self.assertEqual(
            [
                "input_induced_displacement_closure",
                "same_read_write_factorization",
            ],
            plan["mechanism_family"],
        )
        self.assertEqual(
            "maximum ownership_simultaneous_lower_95, then fixed CONCEPTS order",
            plan["selection"]["deterministic_tie_break"],
        )

    def test_no_alias_or_owned_concept_selects_consolidation_fresh_cohort(self) -> None:
        plan = self._route_module().build_route_plan(_summary())

        self.assertEqual("consolidation_fallback", plan["selected_route"])
        self.assertEqual("Consolidation", plan["selection"]["concept"])
        self.assertEqual("fresh", plan["selection"]["cohort"])
        self.assertEqual(
            ["consolidation_fresh_cohort_input_closure"],
            plan["mechanism_family"],
        )
        self.assertEqual(
            "not applicable; fallback is fixed",
            plan["selection"]["deterministic_tie_break"],
        )
        self.assertFalse(plan["changes_causal_ownership_result"])

    def test_registered_gate_identity_mismatch_is_rejected(self) -> None:
        mutations = {
            "arch": ("arch", "other"),
            "model": ("model_id", "other/model"),
            "revision": ("model_revision", "mutable-main"),
            "locus": ("locus", "vis.mid"),
            "module": ("module", "model.visual.blocks.30"),
            "alpha mode": ("alpha_mode", "absolute"),
            "alpha": ("alpha", 0.20),
            "concept order": ("concepts", list(reversed(CONCEPTS))),
            "evaluation rows": ("n_eval", 399),
            "patient unit": ("bootstrap_unit", "image"),
            "bootstrap count": ("bootstrap_resamples", 4999),
            "bootstrap seed": ("bootstrap_seed", 1),
            "random count": ("n_random", 118),
            "familywise alpha": ("familywise_alpha", 0.10),
            "orientation": ("matrix_orientation", "columns are directions"),
            "integer registration type": ("n_eval", 400.0),
        }
        route = self._route_module().build_route_plan

        for name, (key, value) in mutations.items():
            with self.subTest(name=name):
                summary = _summary()
                summary[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    route(summary)

    def test_malformed_matrix_or_columns_fail_closed(self) -> None:
        mutations = {}

        missing_row = _summary()
        del missing_row["effect_matrix"]["Mass"]
        mutations["missing matrix row"] = missing_row

        extra_question = _summary()
        extra_question["effect_matrix"]["Mass"]["Other"] = 0.1
        mutations["extra matrix question"] = extra_question

        nonfinite = _summary()
        nonfinite["effect_matrix"]["Mass"]["Effusion"] = math.nan
        mutations["non-finite matrix effect"] = nonfinite

        missing_column = _summary()
        del missing_column["columns"]["Mass"]
        mutations["missing column"] = missing_column

        wrong_winner = _summary()
        wrong_winner["columns"]["Effusion"]["winner"] = "Mass"
        mutations["winner inconsistent with matrix"] = wrong_winner

        wrong_margin = _summary()
        wrong_margin["columns"]["Effusion"]["ownership_margin"] = 9.0
        mutations["margin inconsistent with matrix"] = wrong_margin

        wrong_owned = _summary()
        wrong_owned["columns"]["Effusion"]["owned"] = True
        wrong_owned["n_owned_concepts"] = 1
        mutations["owned inconsistent with gate rule"] = wrong_owned

        impossible_rank_p = _summary()
        impossible_rank_p["columns"]["Effusion"]["exact_random_rank_p"] = 0.123
        mutations["rank p outside registered 1/120 grid"] = impossible_rank_p

        route = self._route_module().build_route_plan
        for name, summary in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                route(summary)

    def test_inconsistent_counts_and_invalid_detected_alias_fail_closed(self) -> None:
        mutations = {}

        wrong_owned_count = _summary()
        wrong_owned_count["n_owned_concepts"] = 1
        mutations["owned count"] = wrong_owned_count

        wrong_winner_count = _summary()
        wrong_winner_count["n_off_diagonal_winners"] = 1
        mutations["off-diagonal winner count"] = wrong_winner_count

        null_detected_direction = _summary()
        null_detected_direction["shared_alias"]["detected"] = True
        mutations["detected alias without direction"] = null_detected_direction

        unknown_detected_direction = _summary()
        unknown_detected_direction["shared_alias"].update(
            {"detected": True, "direction": "Unknown", "candidate_direction": "Unknown"}
        )
        mutations["detected alias with unknown direction"] = unknown_detected_direction

        direction_when_not_detected = _summary()
        direction_when_not_detected["shared_alias"]["direction"] = "Nodule"
        mutations["undetected alias with direction"] = direction_when_not_detected

        route = self._route_module().build_route_plan
        for name, summary in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                route(summary)

    def test_false_negative_shared_alias_is_rejected(self) -> None:
        summary = _summary()
        for question in CONCEPTS[1:]:
            summary["effect_matrix"]["Effusion"][question] = 0.60
            _refresh_column(summary, question)
        summary["n_off_diagonal_winners"] = 5
        summary["shared_alias"].update(
            {
                "relative_dominance": 0.30,
                "relative_dominance_simultaneous_lower_95": 0.03,
                "off_diagonal_effect": 0.60,
                "off_diagonal_effect_simultaneous_lower_95": 0.02,
                "global_random_effect_max": 0.05,
                "global_absolute_sham_effect_max": 0.01,
            }
        )
        summary["shared_alias"]["by_direction"]["Effusion"].update(
            {
                "relative_dominance": 0.30,
                "relative_dominance_simultaneous_lower_95": 0.03,
                "off_diagonal_effect": 0.60,
                "off_diagonal_effect_simultaneous_lower_95": 0.02,
                "qualified": True,
            }
        )

        with self.assertRaisesRegex(ValueError, "detected"):
            self._route_module().build_route_plan(summary)

    def test_missing_or_mismatched_provenance_is_rejected(self) -> None:
        route = self._route_module().build_route_plan
        mutations = []

        missing_source = _summary()
        del missing_source["source_run_id"]
        mutations.append(missing_source)

        wrong_commit = _summary()
        wrong_commit["source_commit"] = "0" * 40
        mutations.append(wrong_commit)

        missing_receipt = _summary()
        del missing_receipt["evidence"]["bootstrap_sha256"]
        mutations.append(missing_receipt)

        malformed_question_receipt = _summary()
        malformed_question_receipt["evidence"]["questions"]["Effusion"]["per_image_sha256"] = (
            "not-a-sha256"
        )
        mutations.append(malformed_question_receipt)

        for summary in mutations:
            with self.subTest(summary=summary), self.assertRaises((TypeError, ValueError)):
                route(summary)

    def test_cli_writes_deterministic_json(self) -> None:
        summary = _summary()
        with TemporaryDirectory() as temp:
            temp_path = Path(temp)
            summary_path = temp_path / "summary.json"
            first_output = temp_path / "first.json"
            second_output = temp_path / "second.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "src.qwen_mechanism_route",
                "--summary",
                str(summary_path),
                "--output",
            ]
            environment = {**os.environ, "SOURCE_COMMIT": INTERVENTION_COMMIT}

            first = subprocess.run(
                [*command, str(first_output)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            second = subprocess.run(
                [*command, str(second_output)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, first.returncode, first.stderr)
            self.assertEqual(0, second.returncode, second.stderr)
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            self.assertTrue(first_output.read_bytes().endswith(b"\n"))
            self.assertEqual(
                self._route_module().build_route_plan(summary),
                json.loads(first_output.read_text(encoding="utf-8")),
            )


if __name__ == "__main__":
    unittest.main()
