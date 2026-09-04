"""Deterministically lock the post-6x6 Qwen mechanism experiment route."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .qwen_causal_ownership_gate import (
    ALPHA,
    ARCH,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONCEPTS,
    EVAL_ROWS,
    EXPECTED_LOCUS,
    LOCUS,
    MODEL_ID,
    MODEL_REVISION,
    N_RANDOM,
    PREVIOUS_RUN_ID,
    PROMPTS,
    REGISTERED_ROWS_SHA256,
    SOURCE_COMMIT,
    SOURCE_RUN_ID,
)

MATRIX_ORIENTATION = "rows are steering directions; columns are clinical questions"
RANDOM_CONTROL_RULE = (
    "each diagonal must beat all 119 random directions; its minimum exact one-sided "
    "rank p-value is 1/120 and Bonferroni controls six columns at 0.05"
)
SIMULTANEOUS_BOUND = (
    "studentized centered max-T bootstrap across all 30 fixed clinical ownership "
    "contrasts and, separately, the 12 shared-alias dominance and positive-effect "
    "statistics"
)
SHARED_ALIAS_STATISTIC = (
    "a fixed direction's mean off-diagonal excess over each question's named "
    "direction, selected under simultaneous dominance and positive-effect bounds"
)
PRIORITY_ORDER = (
    "shared_alias",
    "diagonal_ownership",
    "consolidation_fallback",
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be a finite number")
    return number


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    return value


def _same_number(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def _hex_digest(value: Any, path: str, length: int) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{path} must be a lowercase {length}-character hex digest")
    return value


def _validate_provenance(
    summary: Mapping[str, Any], expected_intervention_commit: str | None
) -> None:
    expected = {
        "source_run_id": SOURCE_RUN_ID,
        "source_commit": SOURCE_COMMIT,
        "previous_run_id": PREVIOUS_RUN_ID,
        "registered_rows_sha256": REGISTERED_ROWS_SHA256,
        "prompts": PROMPTS,
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise ValueError(
                f"{key} does not match the registered 6x6 provenance: "
                f"expected {expected_value!r}, got {summary.get(key)!r}"
            )

    intervention_commit = _hex_digest(summary.get("intervention_commit"), "intervention_commit", 40)
    if (
        expected_intervention_commit is not None
        and intervention_commit != expected_intervention_commit
    ):
        raise ValueError("intervention_commit does not match the immutable runtime commit")

    evidence = _mapping(summary.get("evidence"), "evidence")
    expected_evidence_keys = {
        "source_reference_sha256",
        "registered_rows_receipt_sha256",
        "bootstrap_sha256",
        "questions",
    }
    if set(evidence) != expected_evidence_keys:
        raise ValueError("evidence must contain the complete registered receipt set")
    for key in expected_evidence_keys - {"questions"}:
        _hex_digest(evidence[key], f"evidence.{key}", 64)

    questions = _mapping(evidence["questions"], "evidence.questions")
    if set(questions) != set(CONCEPTS):
        raise ValueError("evidence.questions must contain exactly the registered concepts")
    receipt_keys = {
        "per_image_sha256",
        "meta_sha256",
        "completion_marker_sha256",
    }
    for concept in CONCEPTS:
        receipt = _mapping(questions[concept], f"evidence.questions.{concept}")
        if set(receipt) != receipt_keys:
            raise ValueError(f"evidence.questions.{concept} must contain the complete receipt set")
        for key in receipt_keys:
            _hex_digest(receipt[key], f"evidence.questions.{concept}.{key}", 64)


def _validate_registered_identity(summary: Mapping[str, Any]) -> None:
    expected = {
        "arch": ARCH,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "locus": LOCUS,
        "module": EXPECTED_LOCUS,
        "alpha_mode": "reltoken",
        "alpha": ALPHA,
        "concepts": list(CONCEPTS),
        "n_eval": EVAL_ROWS,
        "n_eval_patients": EVAL_ROWS,
        "bootstrap_unit": "patient",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_random": N_RANDOM,
        "familywise_alpha": 0.05,
        "random_control_rule": RANDOM_CONTROL_RULE,
        "simultaneous_bound": SIMULTANEOUS_BOUND,
        "matrix_orientation": MATRIX_ORIENTATION,
    }
    integer_fields = {
        "n_eval",
        "n_eval_patients",
        "bootstrap_resamples",
        "bootstrap_seed",
        "n_random",
    }
    for key, expected_value in expected.items():
        if key in integer_fields and (
            isinstance(summary.get(key), bool) or not isinstance(summary.get(key), int)
        ):
            raise ValueError(f"{key} must be an integer fixed by the registered 6x6 gate")
        if summary.get(key) != expected_value:
            raise ValueError(
                f"{key} does not match the registered 6x6 gate: "
                f"expected {expected_value!r}, got {summary.get(key)!r}"
            )


def _validate_effect_matrix(summary: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    raw_matrix = _mapping(summary.get("effect_matrix"), "effect_matrix")
    if set(raw_matrix) != set(CONCEPTS):
        raise ValueError("effect_matrix must contain exactly the six registered directions")

    matrix: dict[str, dict[str, float]] = {}
    for direction in CONCEPTS:
        raw_row = _mapping(raw_matrix[direction], f"effect_matrix.{direction}")
        if set(raw_row) != set(CONCEPTS):
            raise ValueError(
                f"effect_matrix.{direction} must contain exactly the six registered questions"
            )
        matrix[direction] = {
            question: _finite_number(raw_row[question], f"effect_matrix.{direction}.{question}")
            for question in CONCEPTS
        }
    return matrix


def _validate_columns(
    summary: Mapping[str, Any], matrix: Mapping[str, Mapping[str, float]]
) -> dict[str, Mapping[str, Any]]:
    raw_columns = _mapping(summary.get("columns"), "columns")
    if set(raw_columns) != set(CONCEPTS):
        raise ValueError("columns must contain exactly the six registered concepts")

    columns: dict[str, Mapping[str, Any]] = {}
    for question in CONCEPTS:
        path = f"columns.{question}"
        column = _mapping(raw_columns[question], path)
        columns[question] = column

        effects = [matrix[direction][question] for direction in CONCEPTS]
        winner_index = max(range(len(CONCEPTS)), key=effects.__getitem__)
        expected_winner = CONCEPTS[winner_index]
        if column.get("winner") != expected_winner:
            raise ValueError(f"{path}.winner is inconsistent with effect_matrix")

        diagonal_index = CONCEPTS.index(question)
        diagonal = effects[diagonal_index]
        maximum_off_diagonal = max(
            effect for index, effect in enumerate(effects) if index != diagonal_index
        )
        derived_numbers = {
            "diagonal_effect": diagonal,
            "maximum_off_diagonal_effect": maximum_off_diagonal,
            "ownership_margin": diagonal - maximum_off_diagonal,
        }
        for key, expected in derived_numbers.items():
            actual = _finite_number(column.get(key), f"{path}.{key}")
            if not _same_number(actual, expected):
                raise ValueError(f"{path}.{key} is inconsistent with effect_matrix")

        interval = column.get("ownership_descriptive_ci95")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"{path}.ownership_descriptive_ci95 must have two endpoints")
        lower_ci = _finite_number(interval[0], f"{path}.ownership_descriptive_ci95[0]")
        upper_ci = _finite_number(interval[1], f"{path}.ownership_descriptive_ci95[1]")
        if lower_ci > upper_ci:
            raise ValueError(f"{path}.ownership_descriptive_ci95 is reversed")

        ownership_lower = _finite_number(
            column.get("ownership_simultaneous_lower_95"),
            f"{path}.ownership_simultaneous_lower_95",
        )
        random_max = _finite_number(column.get("random_effect_max"), f"{path}.random_effect_max")
        random_p = _finite_number(column.get("exact_random_rank_p"), f"{path}.exact_random_rank_p")
        rank_denominator = N_RANDOM + 1
        if not (
            random_p >= 1.0 / rank_denominator
            and random_p <= 1.0
            and _same_number(
                random_p * rank_denominator,
                round(random_p * rank_denominator),
            )
        ):
            raise ValueError(
                f"{path}.exact_random_rank_p must lie on the registered 1/{rank_denominator} grid"
            )
        absolute_sham = _finite_number(
            column.get("absolute_sham_effect"), f"{path}.absolute_sham_effect"
        )
        if absolute_sham < 0.0:
            raise ValueError(f"{path}.absolute_sham_effect must be non-negative")

        generic_selectivity = column.get("generic_selectivity")
        if not isinstance(generic_selectivity, bool):
            raise TypeError(f"{path}.generic_selectivity must be boolean")
        expected_selectivity = diagonal > 0.0 and diagonal > random_max and diagonal > absolute_sham
        if generic_selectivity != expected_selectivity:
            raise ValueError(f"{path}.generic_selectivity is inconsistent with gate rule")

        owned = column.get("owned")
        if not isinstance(owned, bool):
            raise TypeError(f"{path}.owned must be boolean")
        if owned != (generic_selectivity and ownership_lower > 0.0):
            raise ValueError(f"{path}.owned is inconsistent with gate rule")

    actual_owned = sum(bool(columns[concept]["owned"]) for concept in CONCEPTS)
    reported_owned = _integer(summary.get("n_owned_concepts"), "n_owned_concepts")
    if reported_owned != actual_owned:
        raise ValueError("n_owned_concepts is inconsistent with columns")

    actual_off_diagonal = sum(columns[question]["winner"] != question for question in CONCEPTS)
    reported_off_diagonal = _integer(
        summary.get("n_off_diagonal_winners"), "n_off_diagonal_winners"
    )
    if reported_off_diagonal != actual_off_diagonal:
        raise ValueError("n_off_diagonal_winners is inconsistent with columns")
    return columns


def _validate_shared_alias(
    summary: Mapping[str, Any], matrix: Mapping[str, Mapping[str, float]]
) -> Mapping[str, Any]:
    alias = _mapping(summary.get("shared_alias"), "shared_alias")
    if alias.get("statistic") != SHARED_ALIAS_STATISTIC:
        raise ValueError("shared_alias.statistic does not match the registered gate")

    shared_scores: list[float] = []
    shared_effects: list[float] = []
    for direction in CONCEPTS:
        off_diagonal_questions = [q for q in CONCEPTS if q != direction]
        shared_effects.append(
            sum(matrix[direction][q] for q in off_diagonal_questions) / len(off_diagonal_questions)
        )
        shared_scores.append(
            sum(matrix[direction][q] - matrix[q][q] for q in off_diagonal_questions)
            / len(off_diagonal_questions)
        )
    numeric_fields = (
        "relative_dominance",
        "relative_dominance_simultaneous_lower_95",
        "off_diagonal_effect",
        "off_diagonal_effect_simultaneous_lower_95",
        "global_random_effect_max",
        "global_absolute_sham_effect_max",
    )
    numbers = {key: _finite_number(alias.get(key), f"shared_alias.{key}") for key in numeric_fields}
    if numbers["global_absolute_sham_effect_max"] < 0.0:
        raise ValueError("shared_alias.global_absolute_sham_effect_max must be non-negative")

    raw_by_direction = _mapping(alias.get("by_direction"), "shared_alias.by_direction")
    if set(raw_by_direction) != set(CONCEPTS):
        raise ValueError("shared_alias.by_direction must contain all registered directions")
    qualifying: list[str] = []
    by_direction: dict[str, dict[str, Any]] = {}
    for index, direction in enumerate(CONCEPTS):
        path = f"shared_alias.by_direction.{direction}"
        item = _mapping(raw_by_direction[direction], path)
        expected_keys = {
            "relative_dominance",
            "relative_dominance_simultaneous_lower_95",
            "off_diagonal_effect",
            "off_diagonal_effect_simultaneous_lower_95",
            "qualified",
        }
        if set(item) != expected_keys:
            raise ValueError(f"{path} must contain the complete qualification statistics")
        item_numbers = {
            key: _finite_number(item.get(key), f"{path}.{key}")
            for key in expected_keys - {"qualified"}
        }
        if not _same_number(item_numbers["relative_dominance"], shared_scores[index]):
            raise ValueError(f"{path}.relative_dominance is inconsistent with effect_matrix")
        if not _same_number(item_numbers["off_diagonal_effect"], shared_effects[index]):
            raise ValueError(f"{path}.off_diagonal_effect is inconsistent with effect_matrix")
        if (
            item_numbers["relative_dominance_simultaneous_lower_95"]
            > item_numbers["relative_dominance"]
            or item_numbers["off_diagonal_effect_simultaneous_lower_95"]
            > item_numbers["off_diagonal_effect"]
        ):
            raise ValueError(f"{path} has a lower bound above its estimate")
        qualified = item.get("qualified")
        if not isinstance(qualified, bool):
            raise TypeError(f"{path}.qualified must be boolean")
        expected_qualified = bool(
            item_numbers["relative_dominance_simultaneous_lower_95"] > 0.0
            and item_numbers["off_diagonal_effect_simultaneous_lower_95"] > 0.0
            and item_numbers["off_diagonal_effect"] > numbers["global_random_effect_max"]
            and item_numbers["off_diagonal_effect"] > numbers["global_absolute_sham_effect_max"]
        )
        if qualified != expected_qualified:
            raise ValueError(f"{path}.qualified is inconsistent with gate thresholds")
        if qualified:
            qualifying.append(direction)
        by_direction[direction] = {**item_numbers, "qualified": qualified}

    detected = alias.get("detected")
    if not isinstance(detected, bool):
        raise TypeError("shared_alias.detected must be boolean")
    if detected != bool(qualifying):
        raise ValueError("shared_alias.detected is inconsistent with per-direction qualification")
    expected_candidate = (
        max(qualifying, key=lambda direction: shared_scores[CONCEPTS.index(direction)])
        if qualifying
        else CONCEPTS[max(range(len(CONCEPTS)), key=shared_scores.__getitem__)]
    )
    candidate = alias.get("candidate_direction")
    if candidate != expected_candidate:
        raise ValueError("shared_alias.candidate_direction is inconsistent with gate selection")
    candidate_numbers = by_direction[candidate]
    for key in (
        "relative_dominance",
        "relative_dominance_simultaneous_lower_95",
        "off_diagonal_effect",
        "off_diagonal_effect_simultaneous_lower_95",
    ):
        if not _same_number(numbers[key], candidate_numbers[key]):
            raise ValueError(f"shared_alias.{key} is inconsistent with candidate_direction")

    direction = alias.get("direction")
    if detected:
        if direction != expected_candidate:
            raise ValueError("detected shared_alias.direction must equal candidate_direction")
    elif direction is not None:
        raise ValueError("undetected shared_alias.direction must be null")
    return alias


def _source_gate_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arch": ARCH,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "locus": LOCUS,
        "module": EXPECTED_LOCUS,
        "alpha_mode": "reltoken",
        "alpha": ALPHA,
        "concepts": list(CONCEPTS),
        "source_run_id": summary["source_run_id"],
        "source_commit": summary["source_commit"],
        "previous_run_id": summary["previous_run_id"],
        "intervention_commit": summary["intervention_commit"],
        "registered_rows_sha256": summary["registered_rows_sha256"],
    }


def build_route_plan(
    summary: Mapping[str, Any], *, expected_intervention_commit: str | None = None
) -> dict[str, Any]:
    """Validate a registered ownership summary and choose exactly one mechanism route."""
    summary = _mapping(summary, "summary")
    _validate_registered_identity(summary)
    _validate_provenance(summary, expected_intervention_commit)
    matrix = _validate_effect_matrix(summary)
    columns = _validate_columns(summary, matrix)
    alias = _validate_shared_alias(summary, matrix)

    if alias["detected"]:
        direction = alias["direction"]
        off_diagonal_questions = [q for q in CONCEPTS if q != direction]
        question = max(
            off_diagonal_questions,
            key=lambda candidate: matrix[direction][candidate],
        )
        selected_route = "shared_alias"
        selection = {
            "basis": "shared_alias.detected=true",
            "direction": direction,
            "question": question,
            "effect": matrix[direction][question],
            "deterministic_tie_break": ("maximum off-diagonal effect, then fixed CONCEPTS order"),
        }
        mechanisms = [
            "output_jacobian_alignment",
            "answer_encoding_replacement",
            "projection_removal_intervention",
        ]
    else:
        owned = [concept for concept in CONCEPTS if columns[concept]["owned"]]
        if owned:
            concept = max(
                owned,
                key=lambda candidate: columns[candidate]["ownership_simultaneous_lower_95"],
            )
            selected_route = "diagonal_ownership"
            selection = {
                "basis": "shared_alias.detected=false and at least one owned concept",
                "concept": concept,
                "ownership_simultaneous_lower_95": columns[concept][
                    "ownership_simultaneous_lower_95"
                ],
                "deterministic_tie_break": (
                    "maximum ownership_simultaneous_lower_95, then fixed CONCEPTS order"
                ),
            }
            mechanisms = [
                "input_induced_displacement_closure",
                "same_read_write_factorization",
            ]
        else:
            selected_route = "consolidation_fallback"
            selection = {
                "basis": "shared_alias.detected=false and no owned concept",
                "concept": "Consolidation",
                "cohort": "fresh",
                "deterministic_tie_break": "not applicable; fallback is fixed",
            }
            mechanisms = ["consolidation_fresh_cohort_input_closure"]

    return {
        "route_plan_version": 1,
        "source_gate": _source_gate_identity(summary),
        "priority_order": list(PRIORITY_ORDER),
        "selected_route": selected_route,
        "selection": selection,
        "mechanism_family": mechanisms,
        "changes_causal_ownership_result": False,
    }


def write_route_plan(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_intervention_commit = os.environ.get("SOURCE_COMMIT")
    if expected_intervention_commit is None:
        raise ValueError("immutable runtime SOURCE_COMMIT is unavailable")
    _hex_digest(expected_intervention_commit, "SOURCE_COMMIT", 40)
    plan = build_route_plan(summary, expected_intervention_commit=expected_intervention_commit)
    serialized = (
        json.dumps(
            plan,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8", newline="\n")
    os.replace(temporary, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lock the mechanism experiment route from a registered 6x6 summary."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_route_plan(args.summary, args.output)


if __name__ == "__main__":
    main()
