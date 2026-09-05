from __future__ import annotations

import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.audit_paired_opportunity import (
    CLINICAL_FIELDS,
    COHORTS,
    main,
    nested_pair_counts,
    patient_exclusions,
    remaining_test_rows,
    unchanged_control_counts,
)


def row(row_id, patient, **values):
    return {
        "row_id": row_id, "image_path": f"{row_id}.png", "patient_id": patient,
        "split": "test", "view_AP": "0", "sex_M": "0", "age": "50",
        **{field: "0" for field in CLINICAL_FIELDS}, **values,
    }


def receipts():
    return {
        "initial": {"eval_row_ids": ["a1", "a2"]},
        "specificity": {"row_ids": ["b1"], "patient_ids": ["b"]},
        "ownership": {"row_ids": ["c1"]},
        "closure": {"pairs": [{"patient_id": "d", "positive_row_id": "d1",
                                "negative_row_id": "d2"}]},
        "active": {"row_ids": ["e1"], "patient_ids": ["e"],
                   "calibration_row_ids": ["c1"], "calibration_patient_ids": ["c"]},
    }


class PairedOpportunityAuditTests(unittest.TestCase):
    def test_patient_exclusion_removes_all_images_and_only_test_pool_remains(self):
        rows = [row(name, name[0]) for name in
                ("a1", "a2", "a3", "b1", "c1", "d1", "d2", "e1", "e2")]
        rows += [row("keep1", "keep"), row("train1", "train", split="train")]
        excluded, provenance = patient_exclusions(rows, receipts())
        self.assertEqual(excluded, {"a", "b", "c", "d", "e"})
        self.assertEqual(provenance[0]["rows"], 2)
        self.assertEqual(provenance[0]["patients"], 1)
        self.assertEqual(provenance[3]["rows"], 2)
        self.assertEqual(provenance[-1]["calibration_coverage"], {
            "rows": 1, "patients": 1, "covered_patients": 1, "uncovered_patients": 0,
        })
        self.assertEqual([r["row_id"] for r in remaining_test_rows(rows, excluded)], ["keep1"])

    def test_calibration_coverage_rejects_invalid_or_uncovered_metadata(self):
        rows = [row(name, name[0]) for name in ("a1", "a2", "b1", "c1", "d1", "d2", "e1")]
        rows += [row("keep1", "keep"), row("train1", "train", split="train")]
        cases = (
            (["keep1"], ["keep"], "outside the exclusion union"),
            (["missing"], [], "absent from manifest"),
            (["train1"], ["train"], "non-test row"),
            (["c1"], ["wrong"], "patient IDs disagree"),
        )
        for row_ids, patient_ids, message in cases:
            with self.subTest(message=message):
                source = receipts()
                source["active"]["calibration_row_ids"] = row_ids
                source["active"]["calibration_patient_ids"] = patient_ids
                with self.assertRaisesRegex(ValueError, message):
                    patient_exclusions(rows, source)

    def test_calibration_coverage_uses_patient_union_without_declared_patient_ids(self):
        rows = [row(name, name[0]) for name in
                ("a1", "a2", "b1", "c1", "c2", "d1", "d2", "e1")]
        source = receipts()
        source["active"]["calibration_row_ids"] = ["c1", "c2"]
        del source["active"]["calibration_patient_ids"]
        excluded, provenance = patient_exclusions(rows, source)
        self.assertEqual(excluded, {"a", "b", "c", "d", "e"})
        self.assertEqual(provenance[-1]["calibration_coverage"], {
            "rows": 2, "patients": 1, "covered_patients": 1, "uncovered_patients": 0,
        })

    def test_pair_counts_are_distinct_from_patient_counts_at_each_tier(self):
        rows = [row("p1", "a", Mass="1"), row("p2", "a", Mass="1"),
                row("n1", "a"), row("n2", "a"), row("n3", "a"),
                row("bp", "b", Mass="1"), row("bn", "b", view_AP="1"),
                row("cp", "c", Mass="1"), row("cn", "c", sex_M="1"),
                row("dp", "d", Mass="1"), row("dn", "d", age="51")]
        tiers = nested_pair_counts(rows, "Mass")["tiers"]
        expected = [(4, 9), (3, 8), (2, 7), (1, 6), (1, 6), (1, 6)]
        self.assertEqual([(v["patients"], v["pairs"]) for v in tiers.values()], expected)

    def test_strict_matching_includes_no_finding_but_disease_tier_does_not(self):
        rows = [row("ap", "a", Mass="1"), row("an", "a", no_finding="1"),
                row("bp", "b", Mass="1", Effusion="1"), row("bn", "b", Effusion="1"),
                row("cp", "c", Mass="1"), row("cn", "c", Nodule="1")]
        tiers = nested_pair_counts(rows, "Mass")["tiers"]
        self.assertEqual(tiers["same_view_sex_age"]["patients"], 3)
        self.assertEqual(tiers["same_view_sex_age_eight_other_diseases"]["patients"], 2)
        self.assertEqual(tiers["same_view_sex_age_all_other_labels"]["patients"], 1)
        consolidation = [row("p", "a", Consolidation="1"), row("n", "a", Mass="1")]
        result = nested_pair_counts(consolidation, "Consolidation")["tiers"]
        self.assertEqual(result["unconstrained"]["pairs"], 1)
        self.assertEqual(result["same_view_sex_age_eight_other_diseases"]["pairs"], 0)

    def test_unchanged_controls_count_unordered_and_image_disjoint_pairs(self):
        rows = [row(f"n{i}", "a") for i in range(3)]
        rows += [row("p1", "a", Mass="1"), row("p2", "a", Mass="1"),
                 row("b1", "b"), row("b2", "b", age="51")]
        controls = unchanged_control_counts(rows)
        self.assertEqual(controls["patients"], 1)
        self.assertEqual(controls["unordered_pairs"], 4)
        self.assertEqual(controls["max_image_disjoint_pairs"], 2)
        self.assertEqual(controls["by_target_label"]["Mass"]["0"]["unordered_pairs"], 3)
        self.assertEqual(controls["by_target_label"]["Mass"]["1"]["unordered_pairs"], 1)
        tier = nested_pair_counts(rows, "Mass")["tiers"]["unconstrained"]
        self.assertEqual(tier["patients_with_image_disjoint_strict_control"], 1)
        shared = nested_pair_counts(rows[:3] + rows[3:4], "Mass")["tiers"]["unconstrained"]
        self.assertEqual(shared["patients_with_image_disjoint_strict_control"], 1)
        shared = nested_pair_counts(rows[:2] + rows[3:4], "Mass")["tiers"]["unconstrained"]
        self.assertEqual(shared["patients_with_any_strict_control"], 1)
        self.assertEqual(shared["patients_with_image_disjoint_strict_control"], 0)

    def test_cli_reads_only_manifest_and_fixed_receipts_and_emits_json(self):
        rows = [row(name, name[0]) for name in ("a1", "a2", "b1", "c1", "d1", "d2", "e1")]
        rows += [row("keep1", "keep", Mass="1"), row("keep2", "keep", no_finding="1")]
        manifest = io.StringIO()
        writer = csv.DictWriter(manifest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        inputs = {Path("manifest.csv"): manifest.getvalue()}
        for name, run, filename, _ in COHORTS:
            inputs[Path("runs") / run / "artifacts" / filename] = json.dumps(receipts()[name])
        opened = []

        def open_input(path, mode="r", **kwargs):
            self.assertEqual(mode, "r")
            opened.append(path)
            return io.StringIO(inputs[path])

        output = io.StringIO()
        with patch.object(Path, "open", open_input), redirect_stdout(output):
            main(["--manifest", "manifest.csv", "--runs-root", "runs"])
        report = json.loads(output.getvalue())
        self.assertEqual(opened, list(inputs))
        self.assertEqual(report["excluded_patients"], 5)
        self.assertEqual(report["remaining_patients"], 1)
        self.assertEqual(report["remaining_rows"], 2)
        self.assertEqual(report["exclusions"][-1]["calibration_coverage"], {
            "rows": 1, "patients": 1, "covered_patients": 1, "uncovered_patients": 0,
        })
        self.assertEqual(report["no_finding_with_any_retained_disease"], 0)
        self.assertEqual(report["audit"]["Mass"]["tiers"]["unconstrained"]["pairs"], 1)


if __name__ == "__main__":
    unittest.main()
