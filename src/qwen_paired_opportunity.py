"""CPU selection and patient-bootstrap statistics for paired Qwen opportunity."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import product

import numpy as np

from scripts.audit_nih_paired_pool import DISEASES
from scripts.audit_paired_opportunity import patient_exclusions
from src.build_manifest import patient_split


EVAL_PAIRS = 100
SELECTION_SEED = 20260911
BOOTSTRAP_SEED = 20260912
BOOTSTRAP_RESAMPLES = 5000
PRIMARY_PROMPTS = {"Mass": ("show/A_present", "show/B_present"), "Consolidation": ("anchor",)}
PROMPTS = {
    "Mass": {
        "show/A_present": "Does this chest radiograph show a lung mass? "
        "Answer A if the finding is present and B if it is absent. Reply with A or B only.",
        "show/B_present": "Does this chest radiograph show a lung mass? "
        "Answer B if the finding is present and A if it is absent. Reply with A or B only.",
    },
    "Consolidation": {"anchor": "Is there consolidation in this chest radiograph? Answer yes or no."},
}


def _check_concept(concept):
    if concept not in PRIMARY_PROMPTS:
        raise ValueError("target must be Mass or Consolidation")


def validate_raw_row(row):
    """Return the labels and matching key for an audited NIH metadata row."""
    fields = ("Image Index", "Patient ID", "Finding Labels", "View Position",
              "Patient Sex", "Patient Age")
    if any(field not in row or row[field] is None or str(row[field]).strip() == ""
           for field in fields):
        raise ValueError("missing required raw metadata")
    if row["View Position"] not in ("AP", "PA") or row["Patient Sex"] not in ("M", "F"):
        raise ValueError("unsupported view or sex metadata")
    age_value = row["Patient Age"]
    try:
        age = int(age_value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Patient Age must be a finite nonnegative integer") from None
    if age < 0 or isinstance(age_value, (float, np.floating)) and age != age_value:
        raise ValueError("Patient Age must be a finite nonnegative integer")
    if not isinstance(row["Finding Labels"], str):
        raise ValueError("Finding Labels must contain NIH labels")
    labels = frozenset(row["Finding Labels"].split("|"))
    if not labels <= set(DISEASES) | {"No Finding"}:
        raise ValueError("invalid raw disease label")
    if not isinstance(row["Patient ID"], str) or not isinstance(row["Image Index"], str):
        raise ValueError("raw patient and image IDs must be strings")
    return labels, (row["View Position"], row["Patient Sex"], age)


def select_pairs(raw_rows, manifest_rows, available_pngs, receipts, concept, *, n_pairs=EVAL_PAIRS):
    """Select one uniform eligible pair per sampled patient, before any scores.

    The accepted raw auditor establishes the full CSV schema upstream. ``n_pairs``
    permits small synthetic tests; the production cohort contains EVAL_PAIRS.
    Returns the selected pair records and their prospective receipt.
    """
    _check_concept(concept)
    if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs <= 0:
        raise ValueError("n_pairs must be a positive integer")
    excluded, provenance = patient_exclusions(manifest_rows, receipts)
    manifest_patients = {r["patient_id"] for r in manifest_rows}
    groups = defaultdict(lambda: ([], []))
    by_image = {}
    splits = {}
    labels_by_image = {}
    for row in raw_rows:
        labels, nuisance = validate_raw_row(row)
        image, patient = row["Image Index"], row["Patient ID"]
        if image in by_image:
            raise ValueError("raw image IDs must be unique")
        by_image[image] = row
        labels_by_image[image] = labels
        if patient not in splits:
            splits[patient] = patient_split(patient)
        if (image not in available_pngs or patient in manifest_patients
                or patient in excluded or splits[patient] != "test"):
            continue
        key = (patient, *nuisance, labels - {concept, "No Finding"})
        groups[key][0 if concept in labels else 1].append(image)
    by_patient = defaultdict(set)
    for (patient, *_), (positive, negative) in groups.items():
        by_patient[patient].update(product(positive, negative))
    by_patient = {p: sorted(pairs) for p, pairs in by_patient.items() if pairs}
    if len(by_patient) < n_pairs:
        raise ValueError(f"fewer than {n_pairs} unique eligible test patients")
    rng = np.random.default_rng(SELECTION_SEED)
    patients = rng.permutation(sorted(by_patient))[:n_pairs].tolist()
    selected = []
    for patient in patients:
        candidates = by_patient[patient]
        positive, negative = candidates[rng.integers(len(candidates))]
        pair = {"patient_id": patient, "split": "test"}
        for side, image in (("positive", positive), ("negative", negative)):
            pair.update({side + "_image_index": image, side + "_row_id": image[:-4],
                         side + "_labels": sorted(labels_by_image[image]),
                         side + "_no_finding": "No Finding" in labels_by_image[image],
                         side + "_metadata": deepcopy(by_image[image])})
        selected.append(pair)
    receipt = {
        "concept": concept, "selection_seed": SELECTION_SEED,
        "selection": "sorted patients, NumPy permutation, sorted pair tuples, next integer draw",
        "eligible_patients": len(by_patient), "n_pairs": n_pairs,
        "excluded_patients": len(excluded), "original_manifest_patients": len(manifest_patients),
        "exclusion_sources": provenance, "patient_ids": patients,
        "pairs": deepcopy(selected),
    }
    return selected, receipt


def _sigmoid(values):
    exp = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exp), exp / (1 + exp))


def summarize_opportunity(positive_margins, negative_margins, concept, *,
                          negative_no_finding=None, return_arrays=False):
    """Summarize clinically oriented margins in fixed prompt and patient order.

    Both arrays have shape (n_primary_prompts, 100). Each patient's outcome is
    retained with equal weight. Optional No Finding flags follow that patient
    order. ``return_arrays`` returns (summary, arrays), including the joint draws.
    """
    _check_concept(concept)
    cells = PRIMARY_PROMPTS[concept]
    positive, negative = (np.asarray(a, dtype=np.float64)
                          for a in (positive_margins, negative_margins))
    expected_shape = (len(cells), EVAL_PAIRS)
    if any(a.shape != expected_shape or not np.isfinite(a).all() for a in (positive, negative)):
        raise ValueError(f"complete finite margin arrays must have shape {expected_shape}")
    flags = None
    if negative_no_finding is not None:
        flags = np.asarray(negative_no_finding)
        if flags.shape != (EVAL_PAIRS,) or flags.dtype.kind not in "biuf" or not np.isin(flags, [0, 1]).all():
            raise ValueError("negative No Finding metadata must be 100 binary values")
        flags = flags.astype(bool)
    with np.errstate(over="ignore", invalid="ignore"):
        gap = positive - negative
    if not np.isfinite(gap).all():
        raise ValueError("paired margin gaps must be finite")
    pos_probability, neg_probability = _sigmoid(positive), _sigmoid(negative)
    probability_gap = pos_probability - neg_probability
    ranking = (positive > negative).astype(float) + .5 * (positive == negative)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, EVAL_PAIRS, size=(BOOTSTRAP_RESAMPLES, EVAL_PAIRS))
    bootstrap = gap[:, indices].mean(axis=2)
    if not np.isfinite(bootstrap).all():
        raise ValueError("bootstrap margin means must be finite")
    results = {}
    for i, cell in enumerate(cells):
        lower = float(np.percentile(bootstrap[i], 5))
        result = {
            "prompt": PROMPTS[concept][cell], "mean_margin_gap": float(gap[i].mean()),
            "margin_gap_lower_95": lower,
            "margin_gap_ci95": np.percentile(bootstrap[i], [2.5, 97.5]).tolist(),
            "mean_probability_gap": float(probability_gap[i].mean()),
            "mean_pair_ranking": float(ranking[i].mean()),
            "ranking_endpoint": "within-patient concordance",
            "opportunity_available": lower > 0,
            "per_patient": {"positive_margin": positive[i].tolist(),
                            "negative_margin": negative[i].tolist(),
                            "positive_probability": pos_probability[i].tolist(),
                            "negative_probability": neg_probability[i].tolist(),
                            "margin_gap": gap[i].tolist(),
                            "probability_gap": probability_gap[i].tolist(),
                            "pair_ranking": ranking[i].tolist()},
        }
        if flags is not None:
            result["negative_no_finding_strata"] = {
                str(status).lower(): {"patients": int(np.sum(flags == status)),
                    "mean_margin_gap": float(gap[i, flags == status].mean())
                    if np.any(flags == status) else None}
                for status in (True, False)}
        results[cell] = result
    summary = {"concept": concept, "n_pairs": EVAL_PAIRS, "primary_prompts": list(cells),
               "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
               "prompts": results,
               "opportunity_available": all(r["opportunity_available"] for r in results.values())}
    if return_arrays:
        return summary, {"bootstrap_indices": indices, "bootstrap_margin_gap": bootstrap,
                         "positive_margin": positive, "negative_margin": negative,
                         "margin_gap": gap, "probability_gap": probability_gap,
                         "pair_ranking": ranking}
    return summary
