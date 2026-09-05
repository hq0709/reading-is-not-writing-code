from __future__ import annotations

import base64
import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.build_manifest import patient_split


class BuildManifestTests(unittest.TestCase):
    def test_cli_reads_patient_sex_and_preserves_manifest_fields(self):
        script = Path(__file__).resolve().parents[1] / "src" / "build_manifest.py"
        diseases = ["Effusion", "Atelectasis", "Pneumothorax", "Consolidation",
                    "Cardiomegaly", "Edema", "Infiltration", "Mass", "Nodule"]
        fields = ["row_id", "image_path", "patient_id", "split", "age", "no_finding",
                  "view_AP", "sex_M", *diseases]
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAwMCAO+aD1sAAAAASUVORK5CYII="
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "png" / "images"
            images.mkdir(parents=True)
            raw_rows = []
            expected = {}
            for patient, sex, view, age, split in (
                ("355", "M", "AP", "61", "test"),
                ("1", "F", "PA", "50", "train"),
            ):
                for study, findings in (("000", "Mass|Effusion"), ("001", "No Finding")):
                    row_id = f"{int(patient):08d}_{study}"
                    image = images / f"{row_id}.png"
                    image.write_bytes(png)
                    raw_rows.append({
                        "Image Index": image.name, "Finding Labels": findings,
                        "Patient ID": patient, "Patient Age": age,
                        "Patient Sex": sex, "View Position": view,
                    })
                    positive = study == "000"
                    expected[row_id] = {
                        "row_id": row_id, "image_path": str(image), "patient_id": patient,
                        "split": split, "age": age, "no_finding": str(int(not positive)),
                        "view_AP": str(int(view == "AP")), "sex_M": str(int(sex == "M")),
                        **{disease: str(int(positive and disease in ("Mass", "Effusion")))
                           for disease in diseases},
                    }
            with (root / "Data_Entry_2017_v2020.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
                writer.writeheader()
                writer.writerows(raw_rows)
            output = root / "output" / "manifest.csv"
            result = subprocess.run(
                [sys.executable, "-B", str(script), "--root", str(root),
                 "--out", str(output), "--per-concept", "2", "--seed", "0"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with output.open(newline="") as stream:
                reader = csv.DictReader(stream)
                actual = list(reader)
                self.assertEqual(reader.fieldnames, fields)
            self.assertEqual(len(actual), 4)
            self.assertEqual({row["row_id"]: row for row in actual}, expected)

    def test_patient_split_keeps_registered_assignments(self):
        self.assertEqual(
            {patient: patient_split(patient) for patient in ("1", "2", "355", "100", "1000")},
            {"1": "train", "2": "test", "355": "test", "100": "val", "1000": "train"},
        )


if __name__ == "__main__":
    unittest.main()
