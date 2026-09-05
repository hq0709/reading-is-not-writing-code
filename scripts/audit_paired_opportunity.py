"""Count fresh within-patient image pairs from the NIH manifest and cohort receipts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path


DISEASES = (
    "Effusion", "Atelectasis", "Pneumothorax", "Consolidation", "Cardiomegaly",
    "Edema", "Infiltration", "Mass", "Nodule",
)
CLINICAL_FIELDS = ("no_finding", *DISEASES)
MATCH_FIELDS = ("view_AP", "sex_M", "age")
COHORTS = (
    ("initial", "20260903T000321Z-caaae3ef346d-qwen-full",
     "intervention-summary.json", "eval_row_ids"),
    ("specificity", "20260903T103508Z-456c81bad460-7a3edc4b",
     "registered-rows.json", "row_ids"),
    ("ownership", "20260904T125710Z-a3bd883540eb-causal-ownership",
     "registered-rows.json", "row_ids"),
    ("closure", "20260904T162317Z-8a55a4c2f6b6-consolidation-closure",
     "registered-pairs.json", "pairs"),
    ("active", "20260905T015725Z-1c9820d7b576-mass-prompts",
     "registered-rows.json", "row_ids"),
)


def patient_exclusions(rows: list[dict], receipts: dict) -> tuple[set[str], list[dict]]:
    """Resolve the fixed receipt row keys to a patient union and per-source counts."""
    by_id = {row["row_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("manifest row IDs must be unique")
    excluded = set()
    provenance = []
    for name, run, filename, key in COHORTS:
        receipt = receipts[name]
        row_ids = (
            [pair[field] for pair in receipt[key]
             for field in ("positive_row_id", "negative_row_id")]
            if key == "pairs" else receipt[key]
        )
        source_rows = [by_id[row_id] for row_id in row_ids]
        if any(row["split"] != "test" for row in source_rows):
            raise ValueError(f"{name} contains a non-test row")
        patients = {row["patient_id"] for row in source_rows}
        if "patient_ids" in receipt and patients != set(map(str, receipt["patient_ids"])):
            raise ValueError(f"{name} patient IDs disagree with manifest")
        if key == "pairs" and any(
            by_id[pair[field]]["patient_id"] != str(pair["patient_id"])
            for pair in receipt[key]
            for field in ("positive_row_id", "negative_row_id")
        ):
            raise ValueError("closure pair patient IDs disagree with manifest")
        provenance.append({
            "name": name,
            "receipt": f"{run}/artifacts/{filename}",
            "extraction_key": key,
            "rows": len(row_ids),
            "unique_rows": len(set(row_ids)),
            "patients": len(patients),
            "overlap_with_previous_patients": len(patients & excluded),
        })
        excluded.update(patients)
    active = receipts["active"]
    calibration_ids = active["calibration_row_ids"]
    if any(row_id not in by_id for row_id in calibration_ids):
        raise ValueError("active calibration row is absent from manifest")
    calibration_rows = [by_id[row_id] for row_id in calibration_ids]
    if any(row["split"] != "test" for row in calibration_rows):
        raise ValueError("active calibration contains a non-test row")
    calibration_patients = {row["patient_id"] for row in calibration_rows}
    if "calibration_patient_ids" in active and calibration_patients != set(
        map(str, active["calibration_patient_ids"])
    ):
        raise ValueError("active calibration patient IDs disagree with manifest")
    uncovered = calibration_patients - excluded
    if uncovered:
        raise ValueError("active calibration patients are outside the exclusion union")
    provenance[-1]["calibration_coverage"] = {
        "rows": len(calibration_ids),
        "patients": len(calibration_patients),
        "covered_patients": len(calibration_patients & excluded),
        "uncovered_patients": len(uncovered),
    }
    return excluded, provenance


def remaining_test_rows(rows: list[dict], excluded: set[str]) -> list[dict]:
    """Exclude every image belonging to a previously exposed patient."""
    return [row for row in rows
            if row["split"] == "test" and row["patient_id"] not in excluded]


def _control_groups(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in ("patient_id", *MATCH_FIELDS, *CLINICAL_FIELDS))
        groups[key].append(row)
    return groups


def _control_counts(groups: list[list[dict]]) -> dict:
    return {
        "patients": len({group[0]["patient_id"] for group in groups if len(group) > 1}),
        "unordered_pairs": sum(len(group) * (len(group) - 1) // 2 for group in groups),
        "max_image_disjoint_pairs": sum(len(group) // 2 for group in groups),
    }


def unchanged_control_counts(rows: list[dict]) -> dict:
    """Count unordered pairs matching demographics, view and all ten label fields."""
    groups = list(_control_groups(rows).values())
    return {
        **_control_counts(groups),
        "by_target_label": {
            concept: {
                value: _control_counts([group for group in groups if group[0][concept] == value])
                for value in ("0", "1")
            }
            for concept in ("Mass", "Consolidation")
        },
    }


def nested_pair_counts(rows: list[dict], concept: str) -> dict:
    """Count all positive-negative combinations and distinct patients at each tier.

    Input rows are the already excluded test pool. The eight-disease tier holds
    other retained diseases fixed; the strict tier also holds no_finding fixed.
    """
    other_diseases = tuple(field for field in DISEASES if field != concept)
    tiers = {
        "unconstrained": (),
        "same_view": ("view_AP",),
        "same_view_sex": ("view_AP", "sex_M"),
        "same_view_sex_age": MATCH_FIELDS,
        "same_view_sex_age_eight_other_diseases": (*MATCH_FIELDS, *other_diseases),
        "same_view_sex_age_all_other_labels": (*MATCH_FIELDS, *other_diseases, "no_finding"),
    }
    counts = {tier: {
        "patients": 0, "pairs": 0, "patients_with_any_strict_control": 0,
        "patients_with_image_disjoint_strict_control": 0,
    } for tier in tiers}
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient_id"]].append(row)
    controls = defaultdict(list)
    for group in _control_groups(rows).values():
        for a, b in combinations(group, 2):
            controls[a["patient_id"]].append({a["row_id"], b["row_id"]})
    mismatch_pairs = Counter()
    mismatch_patients = defaultdict(set)
    for patient, studies in patients.items():
        positive = [row for row in studies if row[concept] == "1"]
        negative = [row for row in studies if row[concept] == "0"]
        matches = {tier: [] for tier in tiers}
        for a, b in product(positive, negative):
            for tier, fields in tiers.items():
                if all(a[field] == b[field] for field in fields):
                    matches[tier].append({a["row_id"], b["row_id"]})
            for field in (*MATCH_FIELDS, *other_diseases, "no_finding"):
                if a[field] != b[field]:
                    mismatch_pairs[field] += 1
                    mismatch_patients[field].add(patient)
        for tier, pairs in matches.items():
            counts[tier]["pairs"] += len(pairs)
            if pairs:
                counts[tier]["patients"] += 1
                counts[tier]["patients_with_any_strict_control"] += bool(controls[patient])
                counts[tier]["patients_with_image_disjoint_strict_control"] += any(
                    pair.isdisjoint(control) for pair in pairs for control in controls[patient]
                )
    return {
        "tiers": counts,
        "unconstrained_mismatch": {
            field: {"pairs": count, "patients": len(mismatch_patients[field])}
            for field, count in mismatch_pairs.most_common()
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    receipts = {}
    for name, run, filename, _ in COHORTS:
        with (args.runs_root / run / "artifacts" / filename).open(encoding="utf-8") as stream:
            receipts[name] = json.load(stream)
    excluded, provenance = patient_exclusions(rows, receipts)
    pool = remaining_test_rows(rows, excluded)
    test = [row for row in rows if row["split"] == "test"]
    report = {
        "manifest": str(args.manifest),
        "runs_root": str(args.runs_root),
        "fields": fields,
        "test_rows": len(test),
        "test_patients": len({row["patient_id"] for row in test}),
        "exclusions": provenance,
        "excluded_patients": len(excluded),
        "remaining_rows": len(pool),
        "remaining_patients": len({row["patient_id"] for row in pool}),
        "missing_values": {field: sum(row[field] == "" for row in pool) for field in fields},
        "no_finding_with_any_retained_disease": sum(
            row["no_finding"] == "1" and any(row[field] == "1" for field in DISEASES)
            for row in pool
        ),
        "audit": {concept: nested_pair_counts(pool, concept)
                  for concept in ("Mass", "Consolidation")},
        "label_unchanged_controls": unchanged_control_counts(pool),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
