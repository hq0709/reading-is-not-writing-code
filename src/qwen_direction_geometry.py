"""Post-hoc descriptive geometry for accepted Qwen direction artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

ARCH = "qwen7b"
CONCEPT = "Effusion"
LOCUS = "vis.last"
EXPECTED_LOCUS = "model.visual.blocks.31"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SOURCE_RUN_ID = "20260903T000321Z-caaae3ef346d-qwen-full"
SOURCE_COMMIT = "caaae3ef346d3ac01c76c53ba99b3b7237066639"
SPECIFICITY_RUN_ID = "20260903T103508Z-456c81bad460-7a3edc4b"
SPECIFICITY_COMMIT = "456c81bad460c986d632b1365c0017364b8f6c84"
CONCEPTS = (
    "Effusion",
    "Atelectasis",
    "Pneumothorax",
    "Cardiomegaly",
    "Mass",
    "Nodule",
)
ALPHA = 0.25
EVAL_ROWS = 400

N_RANDOM = 20
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_905
SIGMOID_ABS_TOLERANCE = 1e-7
DIRECTIONS_SHA256 = "b1c00134fbb06c2ce0e392ff6eea5d6275837877412e0686d888f6d336dcde91"
SPECIFICITY_SUMMARY_SHA256 = (
    "da9593c9a0516ed5e37afb57262216eb888462460b83549f6cd5f559320db084"
)
PER_IMAGE_SHA256 = "3300f4c36ef56ab5e61e91afc54c5cc696ccad965b840609d17a3bef87d26250"

_EFFECT_KEYS = (
    "concept",
    *(f"random{index}" for index in range(N_RANDOM)),
    "sham",
    *(f"unrelated_{concept}" for concept in CONCEPTS[1:]),
)
_DIRECTION_FIELDS = {
    "names",
    "vectors",
    "projection",
    "scale",
    "coefficients",
    "locus",
    "raw_dim",
    "projection_dim",
    "projection_seed",
    "C",
    "probe_seed",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _validated_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(CONCEPTS) or matrix.shape[1] == 0:
        raise ValueError(f"{name} matrix shape mismatch")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError(f"{name} vectors must be finite nonzero rows")
    return matrix


def _cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return unit @ unit.T


def _validated_effects(direction_effects: Mapping[str, float]) -> dict[str, float]:
    if set(direction_effects) != set(_EFFECT_KEYS):
        raise ValueError("direction effect keys mismatch")
    effects = {key: float(direction_effects[key]) for key in _EFFECT_KEYS}
    if not all(math.isfinite(value) for value in effects.values()):
        raise ValueError("direction effects must be finite")
    return effects


def summarize_direction_geometry(
    concepts: Sequence[str],
    raw_vectors: np.ndarray,
    coefficients: np.ndarray,
    direction_effects: Mapping[str, float],
) -> dict:
    """Summarize fixed direction geometry without inferential interpretation."""
    if tuple(concepts) != CONCEPTS:
        raise ValueError("concept identity mismatch")
    raw = _validated_matrix("raw vector", raw_vectors)
    coefficient = _validated_matrix("coefficient", coefficients)
    effects = _validated_effects(direction_effects)
    raw_cosines = _cosine_matrix(raw)
    coefficient_cosines = _cosine_matrix(coefficient)

    competitors = list(CONCEPTS[1:])
    competitor = max(competitors, key=lambda name: effects[f"unrelated_{name}"])
    if competitor != "Nodule":
        raise ValueError("accepted strongest clinical competitor is not pinned Nodule")
    target_effect = effects["concept"]
    competitor_effect = effects[f"unrelated_{competitor}"]
    competitor_index = CONCEPTS.index(competitor)
    return {
        "analysis": "post_hoc_descriptive",
        "concepts": list(CONCEPTS),
        "matrix_orientation": "rows and columns follow concepts",
        "raw_vector_cosine_matrix": raw_cosines.tolist(),
        "coefficient_cosine_matrix": coefficient_cosines.tolist(),
        "target": {"concept": CONCEPT, "effect": target_effect},
        "strongest_non_target_clinical_competitor": {
            "concept": competitor,
            "effect": competitor_effect,
            "selected_by": "maximum accepted probability effect over five fixed clinical alternatives",
            "selection_scope": len(competitors),
        },
        "competitor_minus_target_gap": competitor_effect - target_effect,
        "effusion_vs_competitor_cosines": {
            "raw_vector": float(raw_cosines[0, competitor_index]),
            "projected_coefficient": float(coefficient_cosines[0, competitor_index]),
        },
    }


def _load_directions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != _DIRECTION_FIELDS:
            raise ValueError("direction bundle fields mismatch")
        names = tuple(str(value) for value in bundle["names"])
        vectors = np.asarray(bundle["vectors"], dtype=np.float64)
        projection = np.asarray(bundle["projection"], dtype=np.float64)
        scale = np.asarray(bundle["scale"], dtype=np.float64)
        coefficients = np.asarray(bundle["coefficients"], dtype=np.float64)
        raw_dim = int(bundle["raw_dim"])
        if names != CONCEPTS or str(bundle["locus"]) != LOCUS:
            raise ValueError("direction bundle identity mismatch")
        if vectors.shape != (len(CONCEPTS), raw_dim):
            raise ValueError("raw vector matrix shape mismatch")
        if projection.shape != (raw_dim, 512) or scale.shape != (512,):
            raise ValueError("direction bundle projection or scale shape mismatch")
        if coefficients.shape != (len(CONCEPTS), 512):
            raise ValueError("coefficient matrix shape mismatch")
        if int(bundle["projection_dim"]) != 512 or int(bundle["projection_seed"]) != 0:
            raise ValueError("direction bundle projection identity mismatch")
        if float(bundle["C"]) != 1.0 or int(bundle["probe_seed"]) != 0:
            raise ValueError("direction bundle readout identity mismatch")
    if not np.all(np.isfinite(projection)) or not np.all(np.isfinite(scale)):
        raise ValueError("direction bundle projection artifacts must be finite")
    reconstructed = projection @ (coefficients / np.maximum(scale, 1e-8)).T
    reconstructed_norms = np.linalg.norm(reconstructed, axis=0, keepdims=True)
    if not np.all(np.isfinite(reconstructed_norms)) or np.any(reconstructed_norms == 0):
        raise ValueError("raw directions cannot be reconstructed")
    reconstructed = (reconstructed / reconstructed_norms).T
    vector_norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(vector_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("stored raw directions are not unit normalized")
    if not np.allclose(vectors, reconstructed, rtol=1e-5, atol=1e-6):
        raise ValueError("stored raw directions do not match reconstructed directions")
    return vectors, coefficients


def _validate_specificity_summary(summary: dict, per_image_hash: str) -> dict[str, float]:
    expected = {
        "arch": ARCH,
        "concept": CONCEPT,
        "locus": LOCUS,
        "module": EXPECTED_LOCUS,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_run_id": SOURCE_RUN_ID,
        "source_commit": SOURCE_COMMIT,
        "intervention_commit": SPECIFICITY_COMMIT,
        "n_eval": EVAL_ROWS,
        "n_eval_patients": EVAL_ROWS,
        "alpha": ALPHA,
        "alpha_mode": "reltoken",
        "direction_specific": False,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("specificity summary identity mismatch")
    evidence = summary.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("per_image_sha256") != per_image_hash:
        raise ValueError("specificity summary per-image identity mismatch")
    effects = summary.get("direction_effects")
    if not isinstance(effects, dict):
        raise ValueError("specificity summary direction effects are missing")
    return _validated_effects(effects)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _load_per_image(path: Path) -> tuple[list[str], dict[tuple[str, float], np.ndarray], dict[tuple[str, float], np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected_fields = {"row_id", "patient_id", "locus", "direction", "alpha", "p_yes", "margin"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected_fields:
            raise ValueError("per-image fields mismatch")
        expected_conditions = {("concept", 0.0), *((name, ALPHA) for name in _EFFECT_KEYS)}
        patient_by_row: dict[str, str] = {}
        conditions_by_row: dict[str, set[tuple[str, float]]] = {}
        row_order: list[str] = []
        values: dict[tuple[str, str, float], tuple[float, float]] = {}
        record_count = 0
        for record in reader:
            record_count += 1
            row_id = record.get("row_id", "")
            patient_id = record.get("patient_id", "")
            if not row_id or not patient_id:
                raise ValueError("per-image patient identity is incomplete")
            if row_id not in patient_by_row:
                patient_by_row[row_id] = patient_id
                conditions_by_row[row_id] = set()
                row_order.append(row_id)
            elif patient_by_row[row_id] != patient_id:
                raise ValueError("per-image patient identity mismatch")
            try:
                alpha = float(record.get("alpha", "nan"))
                probability = float(record.get("p_yes", "nan"))
                margin = float(record.get("margin", "nan"))
            except ValueError as error:
                raise ValueError("per-image numeric value is invalid") from error
            condition = (record.get("direction", ""), alpha)
            key = (row_id, *condition)
            if record.get("locus") != LOCUS or condition not in expected_conditions or key in values:
                raise ValueError("per-image grid identity mismatch")
            if not all(math.isfinite(value) for value in (probability, margin)) or not 0.0 <= probability <= 1.0:
                raise ValueError("per-image values must be finite probabilities and margins")
            if not math.isclose(
                probability,
                _sigmoid(margin),
                rel_tol=0.0,
                abs_tol=SIGMOID_ABS_TOLERANCE,
            ):
                raise ValueError("per-image p_yes does not equal sigmoid(margin)")
            values[key] = (probability, margin)
            conditions_by_row[row_id].add(condition)

    if record_count != EVAL_ROWS * len(expected_conditions):
        raise ValueError("per-image grid row count mismatch")

    if len(row_order) != EVAL_ROWS or len(set(patient_by_row.values())) != EVAL_ROWS:
        raise ValueError("per-image patient identity mismatch")
    if any(conditions != expected_conditions for conditions in conditions_by_row.values()):
        raise ValueError("per-image grid is incomplete")

    probabilities = {
        condition: np.asarray([values[(row_id, *condition)][0] for row_id in row_order])
        for condition in expected_conditions
    }
    margins = {
        condition: np.asarray([values[(row_id, *condition)][1] for row_id in row_order])
        for condition in expected_conditions
    }
    return row_order, probabilities, margins


def _finite_difference_summary(
    raw_cosines: list[list[float]],
    probabilities: dict[tuple[str, float], np.ndarray],
    margins: dict[tuple[str, float], np.ndarray],
    accepted_effects: Mapping[str, float],
) -> dict:
    baseline_probability = probabilities[("concept", 0.0)]
    baseline_margin = margins[("concept", 0.0)]
    probability_effects = {
        name: probabilities[(name, ALPHA)] - baseline_probability for name in _EFFECT_KEYS
    }
    margin_effects = {name: margins[(name, ALPHA)] - baseline_margin for name in _EFFECT_KEYS}
    probability_means = {name: float(values.mean()) for name, values in probability_effects.items()}
    if any(
        not math.isclose(probability_means[name], accepted_effects[name], rel_tol=1e-10, abs_tol=1e-12)
        for name in _EFFECT_KEYS
    ):
        raise ValueError("per-image probability effects do not match accepted summary")

    stored_by_concept = {CONCEPT: "concept", **{name: f"unrelated_{name}" for name in CONCEPTS[1:]}}
    clinical_margin = {
        concept: float(margin_effects[stored].mean())
        for concept, stored in stored_by_concept.items()
    }
    clinical_probability = {
        concept: probability_means[stored] for concept, stored in stored_by_concept.items()
    }
    controls = {
        name: float(margin_effects[name].mean())
        for name in (*[f"random{index}" for index in range(N_RANDOM)], "sham")
    }

    nodule = margin_effects["unrelated_Nodule"]
    effusion = margin_effects["concept"]
    paired_difference = nodule - effusion
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, EVAL_ROWS, size=(BOOTSTRAP_RESAMPLES, EVAL_ROWS))
    nodule_bootstrap = nodule[sampled].mean(axis=1)
    difference_bootstrap = paired_difference[sampled].mean(axis=1)
    nodule_ci = [float(value) for value in np.percentile(nodule_bootstrap, [2.5, 97.5])]
    difference_ci = [
        float(value) for value in np.percentile(difference_bootstrap, [2.5, 97.5])
    ]
    q_nodule = float(raw_cosines[0][CONCEPTS.index("Nodule")])
    lower_bounds_positive = nodule_ci[0] > 0.0 and difference_ci[0] > 0.0
    clears_controls = (
        clinical_margin["Nodule"]
        > max(controls[f"random{index}"] for index in range(N_RANDOM))
        and clinical_margin["Nodule"] > abs(controls["sham"])
    )
    return {
        "derivation": "exploratory-derived",
        "changes_causal_ownership_route": False,
        "raw_direction_cosines": {
            concept: float(raw_cosines[0][index])
            for index, concept in enumerate(CONCEPTS)
        },
        "clinical_mean_margin_effects": clinical_margin,
        "clinical_mean_probability_effects": clinical_probability,
        "margin_controls": controls,
        "bootstrap": {
            "unit": "patient",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "nodule_mean_margin_effect": {
                "estimate": clinical_margin["Nodule"],
                "descriptive_ci95": nodule_ci,
            },
            "nodule_minus_effusion_margin_effect": {
                "estimate": float(paired_difference.mean()),
                "descriptive_ci95": difference_ci,
            },
        },
        "exploratory_sign_inversion": {
            "selection": "post_hoc strongest of five fixed clinical alternatives",
            "q_nodule_lt_zero": q_nodule < 0.0,
            "both_bootstrap_lower_bounds_gt_zero": lower_bounds_positive,
            "nodule_exceeds_all_random_margin_means_and_absolute_sham": clears_controls,
            "observed": q_nodule < 0.0 and lower_bounds_positive and clears_controls,
        },
    }


def summarize_run(
    directions_path: Path,
    specificity_summary_path: Path,
    per_image_path: Path,
    out: Path,
) -> None:
    directions_hash = _sha256(directions_path)
    specificity_hash = _sha256(specificity_summary_path)
    per_image_hash = _sha256(per_image_path)
    if directions_hash != DIRECTIONS_SHA256:
        raise ValueError("accepted directions hash mismatch")
    if specificity_hash != SPECIFICITY_SUMMARY_SHA256:
        raise ValueError("accepted specificity summary hash mismatch")
    if per_image_hash != PER_IMAGE_SHA256:
        raise ValueError("accepted per-image hash mismatch")

    vectors, coefficients = _load_directions(directions_path)
    specificity = json.loads(specificity_summary_path.read_text(encoding="utf-8"))
    effects = _validate_specificity_summary(specificity, per_image_hash)
    _, probabilities, margins = _load_per_image(per_image_path)
    summary = summarize_direction_geometry(CONCEPTS, vectors, coefficients, effects)
    summary.update(
        _finite_difference_summary(
            summary["raw_vector_cosine_matrix"], probabilities, margins, effects
        )
    )
    summary.update(
        {
            "source_run_id": SOURCE_RUN_ID,
            "source_commit": SOURCE_COMMIT,
            "specificity_run_id": SPECIFICITY_RUN_ID,
            "specificity_commit": SPECIFICITY_COMMIT,
            "source_hashes": {
                "directions_sha256": directions_hash,
                "specificity_summary_sha256": specificity_hash,
                "per_image_sha256": per_image_hash,
            },
        }
    )
    _write_json(out, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--specificity-summary", type=Path, required=True)
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize_run(args.directions, args.specificity_summary, args.per_image, args.out)


if __name__ == "__main__":
    main()
