"""Registered 6x6 clinical direction-by-question causal ownership gate for Qwen."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

try:
    from .first_gate import sha256, write_json
    from .qwen_direction_specificity_gate import (
        ALPHA,
        SOURCE_COMMIT,
        SOURCE_RUN_ID,
        _checksum_inventory,
        _metadata,
        _percentile,
        _row_ids_hash,
        _tagged_hash,
    )
    from .qwen_direction_specificity_gate import (
        validate_source as validate_primary_source,
    )
    from .qwen_effusion_gate import (
        ARCH,
        CONCEPT,
        EXPECTED_LOCUS,
        LOCUS,
        MODEL_ID,
        MODEL_REVISION,
        REGISTERED_PROMPT,
        UNRELATED_CONCEPTS,
    )
except ImportError:
    from first_gate import sha256, write_json
    from qwen_direction_specificity_gate import (
        ALPHA,
        SOURCE_COMMIT,
        SOURCE_RUN_ID,
        _checksum_inventory,
        _metadata,
        _percentile,
        _row_ids_hash,
        _tagged_hash,
    )
    from qwen_direction_specificity_gate import (
        validate_source as validate_primary_source,
    )
    from qwen_effusion_gate import (
        ARCH,
        CONCEPT,
        EXPECTED_LOCUS,
        LOCUS,
        MODEL_ID,
        MODEL_REVISION,
        REGISTERED_PROMPT,
        UNRELATED_CONCEPTS,
    )


CONCEPTS = (CONCEPT, *UNRELATED_CONCEPTS)
PROMPTS = {
    "Effusion": REGISTERED_PROMPT,
    "Atelectasis": "Is there atelectasis in this chest radiograph? Answer yes or no.",
    "Pneumothorax": "Is there a pneumothorax in this chest radiograph? Answer yes or no.",
    "Cardiomegaly": "Is there cardiomegaly in this chest radiograph? Answer yes or no.",
    "Mass": "Is there a lung mass in this chest radiograph? Answer yes or no.",
    "Nodule": "Is there a lung nodule in this chest radiograph? Answer yes or no.",
}
PREVIOUS_RUN_ID = "20260903T103508Z-456c81bad460-7a3edc4b"
PREVIOUS_COMMIT = "456c81bad460c986d632b1365c0017364b8f6c84"
PREVIOUS_SHA256 = {
    "./artifacts/direction-specificity-summary.json": (
        "da9593c9a0516ed5e37afb57262216eb888462460b83549f6cd5f559320db084"
    ),
    "./artifacts/registered-rows.json": (
        "f55839817554cdf8f6abbc43d9c72d7f021b0f02a86bd3007f31aed417c8e8df"
    ),
    "./command_exit_status": ("9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"),
    "./exit_status": ("9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"),
    "./metadata.env": ("5c8dca661c74859dc9177edb81b7cdbc8f6cad67130f1cf4df2607e87c235c5a"),
}
EVAL_ROWS = 400
N_RANDOM = 119
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_904
REGISTERED_ROWS_SHA256 = "e657a9bd1707f13695b02a341047af6f53981295b840376eb0ea5bbebfe2d312"


def _validate_previous_run_receipt(previous_run: Path) -> dict[str, str]:
    metadata = _metadata(previous_run / "metadata.env")
    expected = {
        "SOURCE_COMMIT": PREVIOUS_COMMIT,
        "COMMAND_STATUS": "0",
        "DISPATCHER_STATUS": "0",
        "CLEANUP_STATUS": "0",
        "ABORT_SIGNAL": "none",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("accepted direction-specificity run is not terminal-successful")
    if (previous_run / "command_exit_status").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("accepted direction-specificity command failed")
    if (previous_run / "exit_status").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("accepted direction-specificity dispatcher failed")
    inventory = _checksum_inventory(previous_run / "SHA256SUMS")
    for relative, digest in PREVIOUS_SHA256.items():
        if inventory.get(relative) != digest or not (previous_run / relative).is_file():
            raise ValueError(f"accepted direction-specificity receipt mismatch for {relative}")
    return metadata


def derive_registered_rows(
    manifest_rows: list[dict[str, str]],
    discovery_row_ids: list[str],
    previous_row_ids: list[str],
) -> tuple[list[dict[str, str]], str]:
    """Choose one row from 400 patients absent from both earlier Qwen cohorts."""
    by_id = {row["row_id"]: row for row in manifest_rows}
    excluded_ids = discovery_row_ids + previous_row_ids
    if len(by_id) != len(manifest_rows) or any(row_id not in by_id for row_id in excluded_ids):
        raise ValueError("manifest or prior cohort identity is incomplete")
    excluded_patients = {by_id[row_id]["patient_id"] for row_id in excluded_ids}
    candidates: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        if row["split"] == "test" and row["patient_id"] not in excluded_patients:
            candidates.setdefault(row["patient_id"], []).append(row)
    chosen = [
        min(
            rows,
            key=lambda row: (
                _tagged_hash("qwen-causal-ownership-row-v1", row["row_id"]),
                row["row_id"],
            ),
        )
        for rows in candidates.values()
    ]
    chosen.sort(
        key=lambda row: (
            _tagged_hash("qwen-causal-ownership-patient-v1", row["patient_id"]),
            row["patient_id"],
        )
    )
    selected = chosen[:EVAL_ROWS]
    if len(selected) != EVAL_ROWS or len({row["patient_id"] for row in selected}) != EVAL_ROWS:
        raise ValueError("insufficient unique unseen test patients")
    row_ids = [row["row_id"] for row in selected]
    return selected, _row_ids_hash(row_ids)


def validate_source(
    source_run: Path,
    previous_run: Path,
    manifest: Path,
    reference_out: Path,
    rows_out: Path,
) -> None:
    """Bind the matrix to accepted Qwen artifacts and a third patient-disjoint cohort."""
    prior_derived = rows_out.with_name("prior-registered-rows-derived.json")
    validate_primary_source(source_run, manifest, reference_out, prior_derived)

    previous_run = previous_run.resolve(strict=True)
    if previous_run.name != PREVIOUS_RUN_ID:
        raise ValueError("unexpected direction-specificity source run")
    _validate_previous_run_receipt(previous_run)
    previous_rows = json.loads(
        (previous_run / "artifacts/registered-rows.json").read_text(encoding="utf-8")
    )
    derived_rows = json.loads(prior_derived.read_text(encoding="utf-8"))
    if previous_rows.get("row_ids") != derived_rows.get("row_ids") or previous_rows.get(
        "row_ids_sha256"
    ) != derived_rows.get("row_ids_sha256"):
        raise ValueError("accepted direction-specificity cohort changed")
    previous_summary = json.loads(
        (previous_run / "artifacts/direction-specificity-summary.json").read_text(encoding="utf-8")
    )
    if (
        previous_summary.get("source_run_id") != SOURCE_RUN_ID
        or previous_summary.get("intervention_commit") != PREVIOUS_COMMIT
        or previous_summary.get("n_eval") != EVAL_ROWS
        or previous_summary.get("direction_specific") is not False
    ):
        raise ValueError("accepted direction-specificity result identity changed")

    manifest_rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    source_summary = json.loads(
        (source_run / "artifacts/intervention-summary.json").read_text(encoding="utf-8")
    )
    selected, digest = derive_registered_rows(
        manifest_rows, source_summary["eval_row_ids"], previous_rows["row_ids"]
    )
    if digest != REGISTERED_ROWS_SHA256:
        raise ValueError("prospectively registered ownership cohort changed")
    payload = {
        "selection": (
            "one hash-ranked row from each of 400 hash-ranked test patients absent from "
            "both prior Qwen intervention cohorts"
        ),
        "selection_version": "qwen-causal-ownership-v1",
        "source_run_id": SOURCE_RUN_ID,
        "previous_run_id": PREVIOUS_RUN_ID,
        "n_rows": EVAL_ROWS,
        "n_patients": EVAL_ROWS,
        "row_ids_sha256": digest,
        "row_ids": [row["row_id"] for row in selected],
        "patient_ids": [row["patient_id"] for row in selected],
    }
    write_json(rows_out, payload)
    reference = json.loads(reference_out.read_text(encoding="utf-8"))
    reference.update(
        {
            "previous_run_id": PREVIOUS_RUN_ID,
            "previous_commit": PREVIOUS_COMMIT,
            "ownership_rows_sha256": digest,
        }
    )
    write_json(reference_out, reference)


def _question_effects(
    question: str,
    records: list[dict[str, str]],
    row_ids: list[str],
    patient_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_stored = {
        "concept",
        "sham",
        *(f"random{index}" for index in range(N_RANDOM)),
        *(f"unrelated_{concept}" for concept in CONCEPTS if concept != question),
    }
    expected_conditions = {("concept", 0.0), *((name, ALPHA) for name in expected_stored)}
    row_position = {row_id: index for index, row_id in enumerate(row_ids)}
    patient_by_row = dict(zip(row_ids, patient_ids, strict=True))
    values = {
        condition: np.full(EVAL_ROWS, np.nan, dtype=np.float64) for condition in expected_conditions
    }
    seen: set[tuple[str, str, float]] = set()
    for record in records:
        row_id = record.get("row_id", "")
        direction = record.get("direction", "")
        alpha = float(record.get("alpha", "nan"))
        condition = (direction, alpha)
        key = (row_id, direction, alpha)
        if (
            row_id not in row_position
            or record.get("patient_id") != patient_by_row[row_id]
            or record.get("locus") != LOCUS
            or condition not in expected_conditions
            or key in seen
        ):
            raise ValueError(f"{question} per-image intervention grid identity mismatch")
        probability = float(record.get("p_yes", "nan"))
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{question} per-image intervention probability is invalid")
        values[condition][row_position[row_id]] = probability
        seen.add(key)
    if len(records) != EVAL_ROWS * len(expected_conditions) or any(
        np.isnan(array).any() for array in values.values()
    ):
        raise ValueError(f"{question} per-image intervention grid is incomplete")

    baseline = values[("concept", 0.0)]
    clinical = np.empty((len(CONCEPTS), EVAL_ROWS), dtype=np.float64)
    for index, concept in enumerate(CONCEPTS):
        stored = "concept" if concept == question else f"unrelated_{concept}"
        clinical[index] = values[(stored, ALPHA)] - baseline
    random = np.stack([values[(f"random{index}", ALPHA)] - baseline for index in range(N_RANDOM)])
    sham = values[("sham", ALPHA)] - baseline
    return clinical, random, sham


def _simultaneous_lower_bounds(
    estimates: np.ndarray, bootstrap: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return one-sided 95% centered max-T lower bounds."""
    estimates = np.asarray(estimates, dtype=np.float64)
    bootstrap = np.asarray(bootstrap, dtype=np.float64)
    if bootstrap.ndim != 2 or estimates.shape != (bootstrap.shape[1],):
        raise ValueError("simultaneous-bound shape mismatch")
    standard_error = bootstrap.std(axis=0, ddof=1)
    standardized = np.zeros_like(bootstrap)
    varying = standard_error > 1e-12
    standardized[:, varying] = (bootstrap[:, varying] - estimates[None, varying]) / standard_error[
        None, varying
    ]
    critical = _percentile(standardized.max(axis=1), 95.0) if varying.any() else 0.0
    return estimates - critical * standard_error, critical


def summarize_ownership_matrix(
    records_by_question: dict[str, list[dict[str, str]]],
    row_ids: list[str],
    patient_ids: list[str],
    bootstrap_out: Path,
) -> dict:
    """Estimate semantic ownership and a shared off-diagonal causal alias."""
    if set(records_by_question) != set(CONCEPTS):
        raise ValueError("question grid is incomplete")
    if len(row_ids) != EVAL_ROWS or len(set(row_ids)) != EVAL_ROWS:
        raise ValueError("registered evaluation row count mismatch")
    if len(patient_ids) != EVAL_ROWS or len(set(patient_ids)) != EVAL_ROWS:
        raise ValueError("registered evaluation patients are not unique")

    clinical = np.empty((len(CONCEPTS), len(CONCEPTS), EVAL_ROWS), dtype=np.float64)
    random = np.empty((len(CONCEPTS), N_RANDOM, EVAL_ROWS), dtype=np.float64)
    sham = np.empty((len(CONCEPTS), EVAL_ROWS), dtype=np.float64)
    for question_index, question in enumerate(CONCEPTS):
        c, r, s = _question_effects(question, records_by_question[question], row_ids, patient_ids)
        clinical[:, question_index] = c
        random[question_index] = r
        sham[question_index] = s

    matrix = clinical.mean(axis=2)
    diagonal = np.diag(matrix)
    ownership = np.empty(len(CONCEPTS), dtype=np.float64)
    winners: list[str] = []
    for question_index in range(len(CONCEPTS)):
        off_diagonal = matrix[:, question_index].copy()
        off_diagonal[question_index] = -np.inf
        winner_index = (
            question_index
            if diagonal[question_index] >= float(off_diagonal.max())
            else int(np.argmax(matrix[:, question_index]))
        )
        winners.append(CONCEPTS[winner_index])
        ownership[question_index] = diagonal[question_index] - float(off_diagonal.max())

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    matrix_bootstrap = np.empty(
        (BOOTSTRAP_RESAMPLES, len(CONCEPTS), len(CONCEPTS)), dtype=np.float64
    )
    ownership_bootstrap = np.empty((BOOTSTRAP_RESAMPLES, len(CONCEPTS)), dtype=np.float64)
    pairwise_bootstrap = np.full(
        (BOOTSTRAP_RESAMPLES, len(CONCEPTS), len(CONCEPTS)),
        np.nan,
        dtype=np.float64,
    )
    shared_score_bootstrap = np.empty((BOOTSTRAP_RESAMPLES, len(CONCEPTS)), dtype=np.float64)
    shared_effect_bootstrap = np.empty((BOOTSTRAP_RESAMPLES, len(CONCEPTS)), dtype=np.float64)
    for draw in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.integers(0, EVAL_ROWS, size=EVAL_ROWS)
        sampled_matrix = clinical[:, :, sampled].mean(axis=2)
        matrix_bootstrap[draw] = sampled_matrix
        sampled_diagonal = np.diag(sampled_matrix)
        for question_index in range(len(CONCEPTS)):
            off_diagonal = sampled_matrix[:, question_index].copy()
            off_diagonal[question_index] = -np.inf
            ownership_bootstrap[draw, question_index] = sampled_diagonal[question_index] - float(
                off_diagonal.max()
            )
            for direction_index in range(len(CONCEPTS)):
                if direction_index != question_index:
                    pairwise_bootstrap[draw, direction_index, question_index] = (
                        sampled_diagonal[question_index]
                        - sampled_matrix[direction_index, question_index]
                    )
        for direction_index in range(len(CONCEPTS)):
            questions = [index for index in range(len(CONCEPTS)) if index != direction_index]
            shared_score_bootstrap[draw, direction_index] = np.mean(
                sampled_matrix[direction_index, questions] - sampled_diagonal[questions]
            )
            shared_effect_bootstrap[draw, direction_index] = np.mean(
                sampled_matrix[direction_index, questions]
            )

    shared_scores = np.empty(len(CONCEPTS), dtype=np.float64)
    shared_effects = np.empty(len(CONCEPTS), dtype=np.float64)
    for direction_index in range(len(CONCEPTS)):
        questions = [index for index in range(len(CONCEPTS)) if index != direction_index]
        shared_scores[direction_index] = np.mean(
            matrix[direction_index, questions] - diagonal[questions]
        )
        shared_effects[direction_index] = np.mean(matrix[direction_index, questions])
    shared_direction_index = int(np.argmax(shared_scores))

    pairwise = np.full((len(CONCEPTS), len(CONCEPTS)), np.nan, dtype=np.float64)
    for direction_index in range(len(CONCEPTS)):
        for question_index in range(len(CONCEPTS)):
            if direction_index != question_index:
                pairwise[direction_index, question_index] = (
                    diagonal[question_index] - matrix[direction_index, question_index]
                )
    pairwise_valid = np.isfinite(pairwise)
    pairwise_lower, ownership_critical = _simultaneous_lower_bounds(
        pairwise[pairwise_valid], pairwise_bootstrap[:, pairwise_valid]
    )
    pairwise_simultaneous_lower = np.full_like(pairwise, np.nan)
    pairwise_simultaneous_lower[pairwise_valid] = pairwise_lower

    shared_estimates = np.concatenate((shared_scores, shared_effects))
    shared_bootstrap = np.concatenate((shared_score_bootstrap, shared_effect_bootstrap), axis=1)
    shared_lower, shared_critical = _simultaneous_lower_bounds(shared_estimates, shared_bootstrap)
    shared_score_lower = shared_lower[: len(CONCEPTS)]
    shared_effect_lower = shared_lower[len(CONCEPTS) :]

    random_means = random.mean(axis=2)
    random_alias_effects = np.empty((N_RANDOM, len(CONCEPTS)), dtype=np.float64)
    for direction_index in range(len(CONCEPTS)):
        questions = [index for index in range(len(CONCEPTS)) if index != direction_index]
        random_alias_effects[:, direction_index] = random_means[questions].mean(axis=0)
    global_random_alias_max = float(random_alias_effects.max())
    global_sham_max = float(np.abs(sham.mean(axis=1)).max())
    qualifying_aliases = [
        index
        for index in range(len(CONCEPTS))
        if shared_score_lower[index] > 0
        and shared_effect_lower[index] > 0
        and shared_effects[index] > global_random_alias_max
        and shared_effects[index] > global_sham_max
    ]
    if qualifying_aliases:
        shared_direction_index = max(qualifying_aliases, key=shared_scores.__getitem__)

    bootstrap_out.parent.mkdir(parents=True, exist_ok=True)
    temp = bootstrap_out.with_suffix(bootstrap_out.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(
            stream,
            row_id=np.asarray(row_ids),
            patient_id=np.asarray(patient_ids),
            concepts=np.asarray(CONCEPTS),
            matrix_bootstrap=matrix_bootstrap,
            ownership_margin_bootstrap=ownership_bootstrap,
            pairwise_ownership_bootstrap=pairwise_bootstrap,
            pairwise_ownership_simultaneous_lower_95=pairwise_simultaneous_lower,
            shared_score_bootstrap=shared_score_bootstrap,
            shared_effect_bootstrap=shared_effect_bootstrap,
            shared_score_simultaneous_lower_95=shared_score_lower,
            shared_effect_simultaneous_lower_95=shared_effect_lower,
            ownership_max_t_critical=np.asarray(ownership_critical),
            shared_max_t_critical=np.asarray(shared_critical),
        )
    os.replace(temp, bootstrap_out)

    columns = {}
    n_owned = 0
    for index, concept in enumerate(CONCEPTS):
        column_random_means = random_means[index]
        random_max = float(column_random_means.max())
        exact_random_p = float(
            (1 + np.count_nonzero(column_random_means >= diagonal[index])) / (N_RANDOM + 1)
        )
        generic_selectivity = bool(
            diagonal[index] > 0
            and diagonal[index] > random_max
            and diagonal[index] > abs(float(sham[index].mean()))
        )
        competitors = [direction for direction in range(len(CONCEPTS)) if direction != index]
        lower = float(pairwise_simultaneous_lower[competitors, index].min())
        owned = bool(generic_selectivity and lower > 0)
        n_owned += int(owned)
        columns[concept] = {
            "winner": winners[index],
            "diagonal_effect": float(diagonal[index]),
            "maximum_off_diagonal_effect": float(
                max(matrix[d, index] for d in range(len(CONCEPTS)) if d != index)
            ),
            "ownership_margin": float(ownership[index]),
            "ownership_descriptive_ci95": [
                _percentile(ownership_bootstrap[:, index], 2.5),
                _percentile(ownership_bootstrap[:, index], 97.5),
            ],
            "ownership_simultaneous_lower_95": lower,
            "random_effect_max": random_max,
            "exact_random_rank_p": exact_random_p,
            "absolute_sham_effect": abs(float(sham[index].mean())),
            "generic_selectivity": generic_selectivity,
            "owned": owned,
        }

    return {
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
        "random_control_rule": (
            "each diagonal must beat all 119 random directions; its minimum exact one-sided "
            "rank p-value is 1/120 and Bonferroni controls six columns at 0.05"
        ),
        "simultaneous_bound": (
            "studentized centered max-T bootstrap across all 30 fixed clinical ownership "
            "contrasts and, separately, the 12 shared-alias dominance and positive-effect "
            "statistics"
        ),
        "matrix_orientation": "rows are steering directions; columns are clinical questions",
        "effect_matrix": {
            direction: {question: float(matrix[d, q]) for q, question in enumerate(CONCEPTS)}
            for d, direction in enumerate(CONCEPTS)
        },
        "columns": columns,
        "n_owned_concepts": n_owned,
        "n_off_diagonal_winners": sum(
            winner != question for winner, question in zip(winners, CONCEPTS, strict=True)
        ),
        "shared_alias": {
            "statistic": (
                "a fixed direction's mean off-diagonal excess over each question's named "
                "direction, selected under simultaneous dominance and positive-effect bounds"
            ),
            "direction": (CONCEPTS[shared_direction_index] if qualifying_aliases else None),
            "candidate_direction": CONCEPTS[shared_direction_index],
            "relative_dominance": float(shared_scores[shared_direction_index]),
            "relative_dominance_simultaneous_lower_95": float(
                shared_score_lower[shared_direction_index]
            ),
            "off_diagonal_effect": float(shared_effects[shared_direction_index]),
            "off_diagonal_effect_simultaneous_lower_95": float(
                shared_effect_lower[shared_direction_index]
            ),
            "global_random_effect_max": global_random_alias_max,
            "global_absolute_sham_effect_max": global_sham_max,
            "by_direction": {
                direction: {
                    "relative_dominance": float(shared_scores[index]),
                    "relative_dominance_simultaneous_lower_95": float(shared_score_lower[index]),
                    "off_diagonal_effect": float(shared_effects[index]),
                    "off_diagonal_effect_simultaneous_lower_95": float(shared_effect_lower[index]),
                    "qualified": index in qualifying_aliases,
                }
                for index, direction in enumerate(CONCEPTS)
            },
            "detected": bool(qualifying_aliases),
        },
    }


def summarize_run(
    questions_root: Path,
    source_reference_path: Path,
    registered_rows_path: Path,
    bootstrap_out: Path,
    out: Path,
) -> None:
    source = json.loads(source_reference_path.read_text(encoding="utf-8"))
    registered = json.loads(registered_rows_path.read_text(encoding="utf-8"))
    records_by_question: dict[str, list[dict[str, str]]] = {}
    evidence: dict[str, dict[str, str]] = {}
    intervention_commit = os.environ.get("SOURCE_COMMIT", "unknown")
    if intervention_commit == "unknown":
        raise ValueError("immutable source commit is unavailable")
    for concept in CONCEPTS:
        directory = questions_root / concept.lower()
        per_image = directory / "per-image.csv"
        meta_path = directory / "meta.json"
        done_path = directory / "DONE"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_meta = {
            "arch": ARCH,
            "concept": concept,
            "alpha_mode": "reltoken",
            "alphas": [0.0, ALPHA],
            "control_alphas": [ALPHA],
            "n_random": N_RANDOM,
            "n_eval": EVAL_ROWS,
            "seed": 0,
            "eval_split": "test",
            "model_source": source["model_source"],
            "model_local_only": True,
            "prompt": PROMPTS[concept],
            "directions_sha256": source["directions_sha256"],
            "eval_row_ids_sha256": REGISTERED_ROWS_SHA256,
            "source_commit": intervention_commit,
        }
        if any(meta.get(key) != value for key, value in expected_meta.items()):
            raise ValueError(f"{concept} ownership intervention metadata mismatch")
        if meta.get("eval_row_ids") != registered.get("row_ids"):
            raise ValueError(f"{concept} ownership intervention used different rows")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("loci") != [LOCUS] or done.get("rows") != N_RANDOM + 8:
            raise ValueError(f"{concept} ownership intervention is incomplete")
        records_by_question[concept] = list(
            csv.DictReader(per_image.open(newline="", encoding="utf-8"))
        )
        evidence[concept] = {
            "per_image_sha256": sha256(per_image),
            "meta_sha256": sha256(meta_path),
            "completion_marker_sha256": sha256(done_path),
        }

    summary = summarize_ownership_matrix(
        records_by_question,
        registered["row_ids"],
        registered["patient_ids"],
        bootstrap_out,
    )
    summary.update(
        {
            "source_run_id": SOURCE_RUN_ID,
            "source_commit": SOURCE_COMMIT,
            "previous_run_id": PREVIOUS_RUN_ID,
            "intervention_commit": intervention_commit,
            "registered_rows_sha256": REGISTERED_ROWS_SHA256,
            "prompts": PROMPTS,
            "evidence": {
                "source_reference_sha256": sha256(source_reference_path),
                "registered_rows_receipt_sha256": sha256(registered_rows_path),
                "bootstrap_sha256": sha256(bootstrap_out),
                "questions": evidence,
            },
        }
    )
    write_json(out, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-source")
    validate.add_argument("--source-run", type=Path, required=True)
    validate.add_argument("--previous-run", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    validate.add_argument("--rows-out", type=Path, required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--questions-root", type=Path, required=True)
    summarize.add_argument("--source-reference", type=Path, required=True)
    summarize.add_argument("--registered-rows", type=Path, required=True)
    summarize.add_argument("--bootstrap", type=Path, required=True)
    summarize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-source":
        validate_source(
            args.source_run,
            args.previous_run,
            args.manifest,
            args.out,
            args.rows_out,
        )
    else:
        summarize_run(
            args.questions_root,
            args.source_reference,
            args.registered_rows,
            args.bootstrap,
            args.out,
        )


if __name__ == "__main__":
    main()
