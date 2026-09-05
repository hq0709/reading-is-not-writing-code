"""Audit available NIH within-patient pairs from raw metadata and fixed cohort receipts.

Run from the repository root with python -B -m scripts.audit_nih_paired_pool.
Pair counts enumerate all positive-negative combinations, allowing image reuse.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from scripts.audit_paired_opportunity import COHORTS, patient_exclusions
from src.build_manifest import CONCEPTS, patient_split


DISEASES = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
)
RAW_FIELDS = (
    "Image Index", "Finding Labels", "Follow-up #", "Patient ID", "Patient Age",
    "Patient Sex", "View Position", "OriginalImage[Width", "Height]",
    "OriginalImagePixelSpacing[x", "y]",
)
TIERS = (
    "all_pos_neg", "same_view_sex", "plus_same_age",
    "plus_exact_other13", "plus_same_no_finding",
)
MANIFEST_FIELDS = ("patient_id", "split", "sex_M", "view_AP", "age", "no_finding", *CONCEPTS)


def population_size(rows: list[dict], patient_key: str = "Patient ID") -> dict:
    return {"patients": len({r[patient_key] for r in rows}), "images": len(rows)}


def nested_pair_counts(rows: list[dict], concept: str) -> dict:
    """Count five cumulative tiers using all 13 other diseases, then No Finding."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["Patient ID"]].append((row, set(row["Finding Labels"].split("|"))))
    counts = {tier: {"patients": 0, "pairs": 0, "images": 0,
                     "pairs_negative_no_finding": 0} for tier in TIERS}
    for studies in groups.values():
        positive = [r for r in studies if concept in r[1]]
        negative = [r for r in studies if concept not in r[1]]
        images = {tier: set() for tier in TIERS}
        for (a, a_labels), (b, b_labels) in product(positive, negative):
            conditions = (
                True,
                all(a[k] == b[k] for k in ("View Position", "Patient Sex")),
                int(a["Patient Age"]) == int(b["Patient Age"]),
                a_labels - {concept, "No Finding"} == b_labels - {concept, "No Finding"},
                ("No Finding" in a_labels) == ("No Finding" in b_labels),
            )
            for tier, matched in zip(TIERS, conditions):
                if not matched:
                    break
                counts[tier]["pairs"] += 1
                counts[tier]["pairs_negative_no_finding"] += int("No Finding" in b_labels)
                images[tier].update((a["Image Index"], b["Image Index"]))
        for tier in TIERS:
            counts[tier]["patients"] += bool(images[tier])
            counts[tier]["images"] += len(images[tier])
    return counts


def audit(raw: list[dict], manifest: list[dict], have: set[str],
          receipts: dict, columns: list[str]) -> dict:
    """Validate identities and summarize available and metadata-only populations."""
    raw_by_id = {r["Image Index"][:-4]: r for r in raw}
    if len(raw_by_id) != len(raw):
        raise ValueError("raw image IDs must be unique")
    splits = {r["Patient ID"]: patient_split(r["Patient ID"]) for r in raw}
    for row in manifest:
        original = raw_by_id[row["row_id"]]
        labels = set(original["Finding Labels"].split("|"))
        expected = {
            "patient_id": original["Patient ID"], "split": splits[original["Patient ID"]],
            "sex_M": int(original["Patient Sex"] == "M"),
            "view_AP": int(original["View Position"] == "AP"),
            "age": int(original["Patient Age"]), "no_finding": int("No Finding" in labels),
            **{concept: int(concept in labels) for concept in CONCEPTS},
        }
        for field in MANIFEST_FIELDS:
            actual = row.get(field)
            if isinstance(expected[field], int):
                try:
                    actual = int(actual)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"manifest {field} invalid for {row['row_id']}: {actual!r}"
                    ) from None
            if actual != expected[field]:
                raise ValueError(
                    f"manifest {field} mismatch for {row['row_id']}: "
                    f"stored={actual!r}, expected={expected[field]!r}"
                )
    observed = {label for r in raw for label in r["Finding Labels"].split("|")}
    if observed != set(DISEASES) | {"No Finding"}:
        raise ValueError(f"unexpected raw label schema: {sorted(observed)}")
    excluded, provenance = patient_exclusions(manifest, receipts)
    manifest_patients = {r["patient_id"] for r in manifest}
    manifest_ids = {r["row_id"] for r in manifest}
    available = [r for r in raw if r["Image Index"] in have]
    unavailable = [r for r in raw if r["Image Index"] not in have]

    def residual(rows):
        return [r for r in rows if splits[r["Patient ID"]] == "test"
                and r["Patient ID"] not in excluded]

    test = [r for r in available if splits[r["Patient ID"]] == "test"]
    remaining = residual(available)
    absent = [r for r in remaining if r["Patient ID"] not in manifest_patients]
    present = [r for r in remaining if r["Patient ID"] in manifest_patients]
    manifest_test = [r for r in manifest if r["split"] == "test"]
    manifest_remaining = [r for r in manifest_test if r["patient_id"] not in excluded]
    followup = [int(r["Follow-up #"]) for r in raw if r.get("Follow-up #", "").strip()]
    return {
        "manifest_raw_agreement": {"rows_checked": len(manifest),
                                   "fields": list(MANIFEST_FIELDS),
                                   "values_checked": len(manifest) * len(MANIFEST_FIELDS)},
        "exclusions": provenance,
        "inventory": {
            "png_files": len(have),
            "files_without_metadata": len(have - {r["Image Index"] for r in raw}),
            "metadata": population_size(raw),
            "available": population_size(available),
            "unavailable": population_size(unavailable),
            "original_manifest": population_size(manifest, "patient_id"),
            "original_manifest_test": population_size(manifest_test, "patient_id"),
            "original_manifest_unavailable_images": sum(
                raw_by_id[r["row_id"]]["Image Index"] not in have for r in manifest),
            "exclusion_union": len(excluded),
            "available_test": population_size(test),
            "available_test_residual": population_size(remaining),
            "original_manifest_remaining": population_size(manifest_remaining, "patient_id"),
            "raw_residual_patients_absent_entire_manifest": population_size(absent),
            "raw_residual_patients_present_in_manifest": population_size(present),
            "residual_images_outside_manifest": population_size([
                r for r in remaining if r["Image Index"][:-4] not in manifest_ids]),
            "metadata_only_test_residual": population_size(residual(unavailable)),
        },
        "raw_metadata": {
            "columns": columns,
            "field_availability": {field: field in columns for field in RAW_FIELDS},
            "missing_blank": {field: sum(not (r[field] or "").strip() for r in raw)
                              for field in columns},
            "diseases": list(DISEASES),
            "additional_labels": sorted(observed - set(DISEASES)),
            "no_finding_with_disease": sum(
                "No Finding" in r["Finding Labels"].split("|")
                and len(r["Finding Labels"].split("|")) > 1 for r in raw),
            "view_counts": dict(Counter(r["View Position"] for r in raw)),
            "sex_counts": dict(Counter(r["Patient Sex"] for r in raw)),
            "age_min": min(int(r["Patient Age"]) for r in raw),
            "age_max": max(int(r["Patient Age"]) for r in raw),
            "followup": {"populated": len(followup), "min": min(followup, default=None),
                         "max": max(followup, default=None),
                         "nonzero": sum(n > 0 for n in followup)},
        },
        "pairs": {
            name: {concept: nested_pair_counts(rows, concept)
                   for concept in ("Mass", "Consolidation")}
            for name, rows in (("all_available_test_residual", remaining),
                               ("absent_entire_original_manifest", absent))
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args(argv)
    with (args.dataset_root / "Data_Entry_2017_v2020.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        reader = csv.DictReader(stream)
        raw = list(reader)
        columns = reader.fieldnames
    with (args.dataset_root / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    receipts = {}
    for name, run, filename, _ in COHORTS:
        with (args.runs_root / run / "artifacts" / filename).open(encoding="utf-8") as stream:
            receipts[name] = json.load(stream)
    image_directory = args.dataset_root / "png" / "images"
    have = {p.name for p in image_directory.iterdir() if p.suffix == ".png" and p.is_file()}
    report = audit(raw, manifest, have, receipts, columns)
    report["inputs"] = {"dataset_root": str(args.dataset_root), "runs_root": str(args.runs_root),
                        "image_directory": str(image_directory)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
