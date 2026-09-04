"""CPU exploratory probability, discrimination, and matrix diagnostics for Qwen ownership."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    from .qwen_causal_ownership_gate import ALPHA, CONCEPTS, EVAL_ROWS, LOCUS, N_RANDOM
    from .qwen_direction_specificity_gate import _metadata
except ImportError:
    from qwen_causal_ownership_gate import ALPHA, CONCEPTS, EVAL_ROWS, LOCUS, N_RANDOM
    from qwen_direction_specificity_gate import _metadata


OWNERSHIP_RUN_ID = "20260904T125710Z-a3bd883540eb-causal-ownership"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260906


def _check_numeric(probability: np.ndarray, margin: np.ndarray) -> None:
    if (not np.isfinite(probability).all() or not np.isfinite(margin).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError("probabilities and margins must be finite, with probabilities in [0, 1]")
    # The source writes float32 sigmoid outputs and float32 yes-minus-no margins.
    sigmoid = np.exp(-np.logaddexp(0.0, -margin))
    if not np.allclose(probability, sigmoid, atol=1e-7, rtol=1e-6):
        raise ValueError("p_yes is inconsistent with sigmoid(margin)")


def _question_values(path: Path, question: str, row_ids: list, patient_ids: list) -> np.ndarray:
    """Read the complete control grid, then retain baseline and ordered clinical outcomes."""
    conditions = [("concept", 0.0)] + [
        ("concept" if name == question else f"unrelated_{name}", ALPHA) for name in CONCEPTS
    ] + [(f"random{i}", ALPHA) for i in range(N_RANDOM)] + [("sham", ALPHA)]
    condition_index = {key: i for i, key in enumerate(conditions)}
    row_index = {key: i for i, key in enumerate(row_ids)}
    values = np.full((len(conditions), len(row_ids), 2), np.nan)
    seen = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"row_id", "patient_id", "locus", "direction", "alpha", "p_yes", "margin"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{question}: missing per-image columns")
        for record in reader:
            condition = (record["direction"], float(record["alpha"]))
            row = row_index.get(record["row_id"])
            key = (condition, row)
            if (condition not in condition_index or row is None or key in seen
                    or record["patient_id"] != patient_ids[row] or record["locus"] != LOCUS):
                raise ValueError(f"{question}: per-image grid identity mismatch")
            values[condition_index[condition], row] = float(record["p_yes"]), float(record["margin"])
            seen.add(key)
    if len(seen) != len(conditions) * len(row_ids):
        raise ValueError(f"{question}: incomplete per-image grid")
    _check_numeric(values[..., 0], values[..., 1])
    return values[:1 + len(CONCEPTS)]


def load_ownership_run(ownership_run: Path, manifest: Path) -> tuple:
    """Reuse accepted terminal receipts and align all outcomes and labels to registered rows."""
    metadata = _metadata(ownership_run / "metadata.env")
    expected = {"RUN_ID": OWNERSHIP_RUN_ID, "COMMAND_STATUS": "0", "DISPATCHER_STATUS": "0",
                "CLEANUP_STATUS": "0", "ABORT_SIGNAL": "none"}
    commit = metadata.get("SOURCE_COMMIT", "")
    if (any(metadata.get(key) != value for key, value in expected.items())
            or len(commit) != 40 or not commit.startswith("a3bd883540eb")):
        raise ValueError("accepted ownership terminal metadata mismatch")
    for name in ("command_exit_status", "exit_status"):
        if (ownership_run / name).read_text(encoding="utf-8").strip() != "0":
            raise ValueError("accepted ownership exit receipt failed")
    artifacts = ownership_run / "artifacts"
    registered = json.loads((artifacts / "registered-rows.json").read_text(encoding="utf-8"))
    row_ids, patient_ids = registered["row_ids"], registered["patient_ids"]
    if (len(row_ids) != EVAL_ROWS or len(set(row_ids)) != EVAL_ROWS
            or len(patient_ids) != EVAL_ROWS or len(set(patient_ids)) != EVAL_ROWS):
        raise ValueError("registered cohort must contain exactly 400 unique rows and patients")
    selected = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            row_id = row["row_id"]
            if row_id in row_ids:
                if row_id in selected:
                    raise ValueError("duplicate registered row in manifest")
                selected[row_id] = row
    if set(selected) != set(row_ids) or any(
        selected[r]["patient_id"] != p or selected[r]["split"] != "test"
        for r, p in zip(row_ids, patient_ids, strict=True)
    ):
        raise ValueError("manifest registered patient alignment or test split mismatch")
    labels = np.array([[float(selected[r][c]) for r in row_ids] for c in CONCEPTS])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("manifest labels must be binary")
    questions = []
    for concept in CONCEPTS:
        directory = artifacts / "questions" / concept.lower()
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        expected_meta = {"source_commit": commit, "concept": concept, "n_eval": EVAL_ROWS,
                         "n_random": N_RANDOM, "alpha_mode": "reltoken",
                         "alphas": [0.0, ALPHA], "control_alphas": [ALPHA], "eval_row_ids": row_ids}
        done = json.loads((directory / "DONE").read_text(encoding="utf-8"))
        if (any(meta.get(k) != v for k, v in expected_meta.items())
                or done.get("loci") != [LOCUS] or done.get("rows") != N_RANDOM + 8):
            raise ValueError(f"{concept}: question metadata or completion receipt mismatch")
        questions.append(_question_values(directory / "per-image.csv", concept, row_ids, patient_ids))
    values = np.stack(questions, axis=1)
    provenance = {"source_run_id": metadata["RUN_ID"], "source_commit": commit,
                  "terminal_receipts": ["metadata.env", "command_exit_status", "exit_status"],
                  "cohort_receipt": "artifacts/registered-rows.json"}
    return values[..., 0], values[..., 1], labels, provenance


def matrix_geometry(matrix: np.ndarray) -> dict:
    """Decompose mean-effect energy using the raw squared Frobenius norm as denominator."""
    matrix = np.asarray(matrix, dtype=float)
    denominator = float(np.square(matrix).sum())
    interaction = (matrix - matrix.mean(axis=0, keepdims=True)
                   - matrix.mean(axis=1, keepdims=True) + matrix.mean())
    return {
        "raw_energy": denominator,
        "raw_top_singular_value_energy_fraction": (
            float(np.linalg.svd(matrix, compute_uv=False)[0] ** 2 / denominator)
            if denominator else None),
        "two_way_centered_interaction_energy_fraction": (
            float(np.square(interaction).sum() / denominator) if denominator else None),
        "denominator": "sum of squared entries of the raw mean clinical effect matrix",
        "interaction": "M - column_means - row_means + grand_mean",
    }


def _bootstrap(effects: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint patient index draws, reduced to count weights in bounded chunks."""
    n = effects.shape[-1]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAP_RESAMPLES, *effects.shape[:-1]))
    label_gaps = np.full((BOOTSTRAP_RESAMPLES, *effects.shape[1:-1]), np.nan)
    for start in range(0, BOOTSTRAP_RESAMPLES, 100):
        stop = min(start + 100, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, n, size=(stop - start, n))
        weights = np.zeros((stop - start, n))
        np.add.at(weights, (np.arange(stop - start)[:, None], indices), 1)
        draws[start:stop] = (weights @ effects.reshape(-1, n).T / n).reshape(
            stop - start, *effects.shape[:-1])
        for q, y in enumerate(labels):
            positive, negative = weights @ y, weights @ (1 - y)
            valid = (positive > 0) & (negative > 0)
            label_gaps[start:stop, :, q][valid] = (
                (weights[valid] @ (effects[1, :, q] * y).T) / positive[valid, None]
                - (weights[valid] @ (effects[1, :, q] * (1 - y)).T) / negative[valid, None])
    return draws, label_gaps


def _ci(draws: np.ndarray) -> list | None:
    finite = np.asarray(draws)[np.isfinite(draws)]
    return np.percentile(finite, [2.5, 97.5]).tolist() if finite.size else None


def summarize_diagnostics(probability: np.ndarray, margin: np.ndarray, labels: np.ndarray) -> dict:
    """Summarize (baseline + six directions, six questions, patients); allow small cohorts."""
    probability, margin, labels = (np.asarray(x, dtype=float) for x in (probability, margin, labels))
    k = len(CONCEPTS)
    if (probability.ndim != 3 or probability.shape[:2] != (k + 1, k)
            or probability.shape[-1] == 0 or margin.shape != probability.shape
            or labels.shape != probability.shape[1:] or not np.isin(labels, [0, 1]).all()):
        raise ValueError("expected aligned baseline/clinical arrays and binary question labels")
    _check_numeric(probability, margin)
    effects = np.stack([x[1:] - x[0] for x in (probability, margin)])
    draws, label_gaps = _bootstrap(effects, labels)
    result = {
        "analysis": "exploratory descriptive diagnostics; existing gates retain their decisions",
        "limitations": (
            "Label-conditioned shifts can identify possible broad answer bias but do not establish "
            "a mechanism. Matrix energy summaries are exploratory, not proof of a shared channel. "
            "Intervals are pointwise percentile summaries, not simultaneous gate bounds."),
        "concepts": list(CONCEPTS), "n_patients": labels.shape[-1], "alpha": ALPHA,
        "matrix_orientation": "rows: clinical steering directions; columns: questions",
        "bootstrap": {"unit": "patient", "resamples": BOOTSTRAP_RESAMPLES,
                      "seed": BOOTSTRAP_SEED, "joint_indices_across_questions": True},
        "definitions": {"ownership_margin": "named effect minus maximum other clinical effect",
                        "rank": "1 + number of strictly larger clinical effects; ties share rank",
                        "gap_differences": "named effect minus each clinical direction effect",
                        "correct_label_margin_effect": "mean((2 * label - 1) * margin_shift)",
                        "label_gap_change": "positive mean margin shift minus negative mean shift",
                        "auroc": "ranking by stored logit margin; null for constant labels"},
    }
    for scale, name in enumerate(("probability", "logit_margin")):
        matrix, sampled = effects[scale].mean(axis=-1), draws[:, scale]
        diagonal = np.diag(matrix)
        sampled_diagonal = np.diagonal(sampled, axis1=1, axis2=2)
        columns = {}
        for q, question in enumerate(CONCEPTS):
            others = [d for d in range(k) if d != q]
            gaps = diagonal[q] - matrix[:, q]
            gap_draws = sampled_diagonal[:, q, None] - sampled[:, :, q]
            columns[question] = {
                "raw_ownership_margin": float(diagonal[q] - matrix[others, q].max()),
                "ownership_ci95": _ci(sampled_diagonal[:, q] - sampled[:, others, q].max(axis=1)),
                "named_direction_rank": int(1 + np.count_nonzero(matrix[:, q] > diagonal[q])),
                "gap_differences": {c: {"mean": float(gaps[d]), "ci95": _ci(gap_draws[:, d])}
                                    for d, c in enumerate(CONCEPTS)},
            }
        result[name] = {"mean_clinical_effect_matrix": matrix.tolist(),
                        "mean_clinical_effect_ci95": np.percentile(
                            sampled, [2.5, 97.5], axis=0).transpose(1, 2, 0).tolist(),
                        "questions": columns, "geometry": matrix_geometry(matrix)}
    performance = {}
    for q, question in enumerate(CONCEPTS):
        y = labels[q]
        conditions = {}
        for d, condition in enumerate(("baseline", *CONCEPTS)):
            shift = margin[d, q] - margin[0, q]
            subgroups = {}
            for label, name in ((1, "positive"), (0, "negative")):
                subgroup = shift[y == label]
                subgroups[name] = {"count": int(subgroup.size), "mean_margin_shift": (
                    float(subgroup.mean()) if subgroup.size else None)}
            both = np.unique(y).size == 2
            gap_draws = label_gaps[:, d - 1, q] if d else np.zeros(BOOTSTRAP_RESAMPLES)
            conditions[condition] = {
                "auroc": float(roc_auc_score(y, margin[d, q])) if both else None,
                "auroc_unavailable_reason": None if both else "constant labels",
                "brier_score": float(brier_score_loss(y, probability[d, q])),
                "correct_label_margin_effect": float(np.mean((2 * y - 1) * shift)),
                "label_subgroups": subgroups,
                "label_gap_change": (float(shift[y == 1].mean() - shift[y == 0].mean())
                                     if both else None),
                "label_gap_change_ci95": _ci(gap_draws) if both else None,
                "label_gap_bootstrap_valid_draws": int(np.isfinite(gap_draws).sum()) if both else 0,
            }
        performance[question] = conditions
    result["label_conditioned_performance"] = performance
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership-run", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    probability, margin, labels, provenance = load_ownership_run(args.ownership_run, args.manifest)
    result = summarize_diagnostics(probability, margin, labels)
    result["provenance"] = provenance
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
