from __future__ import annotations

import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import audit_nih_paired_pool as audit
from scripts.audit_paired_opportunity import COHORTS
from src.build_manifest import CONCEPTS, patient_split


def row(image, patient, labels, **values):
    return {"Image Index": image + ".png", "Patient ID": patient,
            "Finding Labels": labels, "Patient Age": "50", "Patient Sex": "F",
            "View Position": "PA", "Follow-up #": "0", **values}


def fixture():
    test = [str(n) for n in range(100) if patient_split(str(n)) == "test"]
    known, absent, missing, excluded = test[:4]
    train = next(str(n) for n in range(100) if patient_split(str(n)) == "train")
    raw = [row("kp", known, "Mass|Pneumonia"), row("kn", known, "Pneumonia"),
           row("ap", absent, "Mass|Pneumonia"), row("an", absent, "Pneumonia"),
           row("mp", missing, "Mass|Pneumonia"), row("mn", missing, "Pneumonia"),
           row("ep", excluded, "Mass"), row("en", excluded, "No Finding"),
           row("train", train, "|".join(audit.DISEASES))]
    manifest = [{"row_id": r["Image Index"][:-4], "patient_id": r["Patient ID"],
                 "split": patient_split(r["Patient ID"]),
                 "sex_M": str(int(r["Patient Sex"] == "M")),
                 "view_AP": str(int(r["View Position"] == "AP")),
                 "age": str(int(r["Patient Age"])),
                 "no_finding": str(int("No Finding" in r["Finding Labels"].split("|"))),
                 **{c: str(int(c in r["Finding Labels"].split("|"))) for c in CONCEPTS}}
                for r in raw if r["Image Index"] in {"kp.png", "ep.png", "en.png", "train.png"}]
    receipts = {name: {key: ["ep"]} for name, _, _, key in COHORTS}
    receipts["closure"] = {"pairs": [{"patient_id": excluded,
                                       "positive_row_id": "ep", "negative_row_id": "en"}]}
    receipts["active"]["calibration_row_ids"] = ["ep"]
    have = {r["Image Index"] for r in raw} - {"mn.png"}
    return raw, manifest, have, receipts, list(raw[0])


class NihPairedPoolAuditTests(unittest.TestCase):
    def test_absent_patient_differs_from_absent_image_and_unavailable_metadata(self):
        report = audit.audit(*fixture())
        inventory = report["inventory"]
        self.assertEqual(inventory["exclusion_union"], 1)
        self.assertEqual(inventory["available_test_residual"], {"patients": 3, "images": 5})
        self.assertEqual(inventory["original_manifest_remaining"], {"patients": 1, "images": 1})
        self.assertEqual(inventory["raw_residual_patients_absent_entire_manifest"],
                         {"patients": 2, "images": 3})
        self.assertEqual(inventory["residual_images_outside_manifest"],
                         {"patients": 3, "images": 4})
        self.assertEqual(inventory["metadata_only_test_residual"], {"patients": 1, "images": 1})
        for population, expected in [("all_available_test_residual", 2),
                                     ("absent_entire_original_manifest", 1)]:
            self.assertEqual(report["pairs"][population]["Mass"]["plus_same_no_finding"]["pairs"],
                             expected)

    def test_all_other_diseases_and_no_finding_have_separate_tiers(self):
        for concept in ("Mass", "Consolidation"):
            for other in set(audit.DISEASES) - {concept}:
                with self.subTest(concept=concept, other=other):
                    rows = [row("p", "a", concept), row("n", "a", other)]
                    counts = audit.nested_pair_counts(rows, concept)
                    self.assertEqual(counts["plus_same_age"]["pairs"], 1)
                    self.assertEqual(counts["plus_exact_other13"]["pairs"], 0)
            rows = [row("p", "a", concept), row("n", "a", "No Finding"),
                    row("p2", "b", concept + "|Pneumonia"), row("n2", "b", "Pneumonia")]
            counts = audit.nested_pair_counts(rows, concept)
            self.assertEqual(counts["plus_exact_other13"]["pairs"], 2)
            self.assertEqual(counts["plus_exact_other13"]["pairs_negative_no_finding"], 1)
            self.assertEqual(counts["plus_same_no_finding"],
                             {"patients": 1, "pairs": 1, "images": 2,
                              "pairs_negative_no_finding": 0})

    def test_five_tiers_are_nested_and_images_are_distinct(self):
        rows = [row("p1", "a", "Mass"), row("p2", "a", "Mass"),
                row("n1", "a", "No Finding"), row("n2", "a", "No Finding"),
                row("bp", "b", "Mass"), row("bn", "b", "No Finding", **{"View Position": "AP"}),
                row("cp", "c", "Mass"), row("cn", "c", "No Finding", **{"Patient Sex": "M"}),
                row("dp", "d", "Mass"), row("dn", "d", "No Finding", **{"Patient Age": "51"})]
        counts = audit.nested_pair_counts(rows, "Mass")
        self.assertEqual([(v["patients"], v["pairs"], v["images"]) for v in counts.values()],
                         [(4, 7, 10), (2, 5, 6), (1, 4, 4), (1, 4, 4), (0, 0, 0)])

    def test_reuses_authoritative_split_and_rejects_manifest_disagreement(self):
        self.assertIs(audit.patient_split, patient_split)
        for field, value, message in [("split", "train", "manifest split"),
                                      ("patient_id", "other", "manifest patient")]:
            args = fixture()
            args[1][0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                audit.audit(*args)

    def test_manifest_agreement_checks_every_canonical_field(self):
        args = fixture()
        args[1][0]["age"] = "050"
        agreement = audit.audit(*args)["manifest_raw_agreement"]
        self.assertEqual(agreement["rows_checked"], 4)
        self.assertEqual(agreement["values_checked"], 60)
        self.assertEqual(agreement["fields"],
                         ["patient_id", "split", "sex_M", "view_AP", "age", "no_finding",
                          "Effusion", "Atelectasis", "Pneumothorax", "Consolidation",
                          "Cardiomegaly", "Edema", "Infiltration", "Mass", "Nodule"])
        for index in (0, 2, 3):
            for field in ("sex_M", "view_AP", "age", "no_finding", *CONCEPTS):
                with self.subTest(index=index, field=field):
                    args = fixture()
                    changed = args[1][index]
                    changed[field] = str(1 - int(changed[field]))
                    with self.assertRaisesRegex(
                        ValueError, f"manifest {field} mismatch for {changed['row_id']}"
                    ):
                        audit.audit(*args)

    def test_label_schema_raw_fields_and_inventory(self):
        args = fixture()
        args[2].add("extra.png")
        args[0][2]["Finding Labels"] += "|No Finding"
        args[0][0]["Follow-up #"] = ""
        args[0][2]["Patient Age"] = "0"
        args[0][3]["Patient Age"] = "95"
        report = audit.audit(*args)
        self.assertEqual(report["inventory"]["files_without_metadata"], 1)
        metadata = report["raw_metadata"]
        self.assertEqual(metadata["no_finding_with_disease"], 1)
        self.assertEqual(metadata["missing_blank"]["Follow-up #"], 1)
        self.assertTrue(metadata["field_availability"]["Follow-up #"])
        self.assertFalse(metadata["field_availability"]["Height]"])
        self.assertEqual((metadata["age_min"], metadata["age_max"]), (0, 95))
        args[0][2]["Finding Labels"] += "|Unknown"
        with self.assertRaisesRegex(ValueError, "label schema"):
            audit.audit(*args)

    def test_cli_reads_fixed_inputs_and_emits_only_json(self):
        raw, manifest, have, receipts, _ = fixture()
        inputs = {}
        for filename, rows in [("Data_Entry_2017_v2020.csv", raw), ("manifest.csv", manifest)]:
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            inputs[Path("dataset") / filename] = stream.getvalue()
        for name, run, filename, _ in COHORTS:
            inputs[Path("runs") / run / "artifacts" / filename] = json.dumps(receipts[name])
        opened = []

        def open_input(path, mode="r", **kwargs):
            self.assertEqual(mode, "r")
            opened.append(path)
            return io.StringIO(inputs[path])

        image_dir = Path("dataset/png/images")
        entries = [image_dir / name for name in have] + [image_dir / "directory.png"]
        output = io.StringIO()
        with patch.object(Path, "open", open_input), \
             patch.object(Path, "iterdir", return_value=iter(entries)) as listing, \
             patch.object(Path, "is_file", lambda p: p.name in have), redirect_stdout(output):
            audit.main(["--dataset-root", "dataset", "--runs-root", "runs"])
        report = json.loads(output.getvalue())
        self.assertEqual(opened, list(inputs))
        listing.assert_called_once()
        self.assertEqual(report["inventory"]["png_files"], len(have))
        self.assertEqual(report["inventory"]["exclusion_union"], 1)


if __name__ == "__main__":
    unittest.main()
