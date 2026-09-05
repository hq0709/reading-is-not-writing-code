from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from src import llava_readout_diagnostic as diagnostic
from src import run_llava_readout_diagnostic as runner


def raw(image, patient, finding="No Finding", age="50", view="PA", sex="F"):
    return {
        "Image Index": image + ".png",
        "Finding Labels": finding,
        "Patient ID": patient,
        "Patient Age": age,
        "Patient Sex": sex,
        "View Position": view,
    }


class AllocationTests(unittest.TestCase):
    def test_cpu_import_and_help_do_not_load_torch(self):
        command = (
            "import src.run_llava_readout_diagnostic; import sys; assert 'torch' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command], capture_output=True, text=True, timeout=30, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        completed = subprocess.run(
            [sys.executable, "-m", "src.run_llava_readout_diagnostic", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_identifier_first_allocation_excludes_every_manifest_patient_and_uses_smallest_image(
        self,
    ):
        rows = [
            raw("m", "manifest"),
            raw("b", "p1", "Effusion"),
            raw("a", "p1"),
            raw("c", "p2"),
            raw("d", "p3", "Effusion"),
            raw("e", "p4"),
            raw("f", "p5"),
        ]
        manifest = [{"row_id": "m", "patient_id": "manifest"}]
        with (
            patch.object(diagnostic, "N_PAIRS", 2),
            patch.object(diagnostic, "patient_split", return_value="val"),
        ):
            allocation = diagnostic.allocate(rows, manifest, "/images")
        selected = allocation["index"] + allocation["donor"]
        self.assertNotIn("manifest", {row["patient_id"] for row in selected})
        chosen = {row["patient_id"]: row for row in selected}
        if "p1" in chosen:
            self.assertEqual(chosen["p1"]["image_index"], "a.png")
            self.assertEqual(chosen["p1"]["Effusion"], 0)
        self.assertEqual(len({row["patient_id"] for row in selected}), 4)

    def test_registered_prompts_and_primary_family_are_exact(self):
        self.assertEqual(
            [condition["name"] for condition in diagnostic.CONDITIONS],
            ["0Y", "0A", "0B", "1Y", "1A", "1B"],
        )
        self.assertEqual(
            diagnostic.CONDITIONS[0]["prompt"],
            "Is there a pleural effusion in this chest radiograph? Answer yes or no.",
        )
        self.assertIn("Answer B if the finding is present", diagnostic.CONDITIONS[-1]["prompt"])
        advantages = np.asarray([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]])
        expected = np.asarray([1.0, 8 / 3, 2.5, -2.0, 1.0, 0.0])
        np.testing.assert_allclose(diagnostic.primary_contrasts(advantages), expected)

    def test_timing_projection_includes_loading_preflight_and_analysis_allowance(self):
        pilot = 100.0
        boundary_elapsed = 3600 - 8406 * pilot / 512 - 600
        self.assertTrue(runner.budget(pilot, boundary_elapsed)["passed"])
        self.assertFalse(runner.budget(pilot, boundary_elapsed + 1e-6)["passed"])

    def test_semantic_preflight_rejects_identity_and_token_drift(self):
        tokens = {}
        for spec in diagnostic.CONDITIONS:
            first, second = runner.PINNED_TOKEN_GROUPS[spec["encoding"]]
            present, absent = (second, first) if spec["encoding"] == "B" else (first, second)
            tokens[spec["name"]] = {
                "raw_first": first,
                "raw_second": second,
                "present": present,
                "absent": absent,
                "candidate_ids": sorted(set(first + second)),
            }
        cases = []
        for identity in runner.expected_mapping_cases():
            sign = 2 * identity["expected_present"] - 1
            record = {**identity, "semantic_margin": float(sign)}
            if identity["encoding"] == "Y":
                record["lse_margin"] = float(sign)
            cases.append(record)
        flight = {
            "source_commit": "a" * 40,
            "passed": True,
            "validation_row_ids": runner.accepted.VALIDATION_ROW_IDS,
            "module": diagnostic.MODULE,
            "model_source": "/model",
            "forwarded_no_cls_exact": True,
            "pooling": "mean over all 577 block-output tokens",
            "clean_repeat_exact": True,
            "mapping_cases": cases,
            "token_ids": tokens,
            "throughput": runner.budget(1.0, 10.0),
        }
        runner.validate_preflight(flight, "a" * 40, "/model")
        changed = {**flight, "mapping_cases": [dict(record) for record in cases]}
        changed["mapping_cases"][0]["statement"] = "The finding might be present."
        with self.assertRaisesRegex(ValueError, "mapping identity"):
            runner.validate_preflight(changed, "a" * 40, "/model")
        changed = {**flight, "token_ids": {key: dict(value) for key, value in tokens.items()}}
        changed["token_ids"]["0Y"]["raw_first"] = [1]
        with self.assertRaisesRegex(ValueError, "pinned singleton"):
            runner.validate_preflight(changed, "a" * 40, "/model")


class GridTests(unittest.TestCase):
    def test_complete_grid_uses_index_labels_for_both_roles(self):
        with patch.object(diagnostic, "N_PAIRS", 3):
            allocation = {
                role: [
                    {
                        "patient_id": f"{role}{i}",
                        "row_id": f"{role}-row{i}",
                        "image_path": f"/{role}{i}.png",
                        "Effusion": int((i + (role == "donor")) % 2),
                    }
                    for i in range(3)
                ]
                for role in ("index", "donor")
            }
            records = []
            for condition in diagnostic.CONDITIONS:
                for role in ("index", "donor"):
                    for i, row in enumerate(allocation[role]):
                        margin = float(i - 1)
                        records.append(
                            {
                                "source_commit": "a" * 40,
                                "condition": condition["name"],
                                "wording": condition["wording"],
                                "encoding": condition["encoding"],
                                "prompt": condition["prompt"],
                                "image_role": role,
                                "pair_index": i,
                                "patient_id": row["patient_id"],
                                "row_id": row["row_id"],
                                "image_path": row["image_path"],
                                "index_label": allocation["index"][i]["Effusion"],
                                "source_label": row["Effusion"],
                                "raw_margin": margin,
                                "semantic_margin": margin,
                                "probability": float(diagnostic.sigmoid(margin)),
                                "answer_token_mass": 0.2,
                                "lse_margin": margin if condition["encoding"] == "Y" else "",
                            }
                        )
            values = diagnostic.validate_grid(records, allocation, "a" * 40)
            self.assertEqual(values["semantic_margin"].shape, (6, 2, 3))
            changed = [dict(row) for row in records]
            changed[3]["index_label"] = 1 - int(changed[3]["index_label"])
            with self.assertRaisesRegex(ValueError, "cohort"):
                diagnostic.validate_grid(changed, allocation, "a" * 40)

            float32_probability = [dict(row) for row in records]
            float32_probability[0]["probability"] = float(
                np.float32(diagnostic.sigmoid(float32_probability[0]["semantic_margin"]))
            )
            self.assertNotEqual(
                float32_probability[0]["probability"],
                float(diagnostic.sigmoid(float32_probability[0]["semantic_margin"])),
            )
            with self.assertRaisesRegex(ValueError, "differs from semantic sigmoid"):
                diagnostic.validate_grid(float32_probability, allocation, "a" * 40)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = np.arange(700) % 2
        cls.indices = diagnostic.bootstrap_indices()

    def test_donor_auroc_is_scored_against_index_label_and_all_six_contrasts_remain(self):
        semantic = np.empty((6, 2, 700), dtype=float)
        semantic[:, 0] = self.labels
        semantic[:, 1] = -self.labels
        arrays, report = diagnostic.answer_statistics(
            semantic, self.labels, 1 - self.labels, self.indices
        )
        np.testing.assert_allclose(arrays["image_advantage"], 1.0)
        self.assertEqual(len(report["contrasts"]), 6)
        self.assertEqual(report["common_valid_draws"], 10_000)
        np.testing.assert_allclose(arrays["donor_own_label_auroc"], 1.0)

    def test_collapsed_bootstrap_is_retained_and_flagged(self):
        semantic = np.tile(self.labels, (6, 2, 1)).astype(float)
        arrays, report = diagnostic.answer_statistics(
            semantic, self.labels, self.labels, self.indices
        )
        np.testing.assert_array_equal(arrays["primary"], 0.0)
        self.assertEqual(report["simultaneous_radius"], 0.0)
        self.assertTrue(report["bootstrap_degenerate"])
        self.assertEqual(
            [row["name"] for row in report["contrasts"]], list(diagnostic.PRIMARY_NAMES)
        )

    def test_nonzero_common_radius_reference(self):
        rng = np.random.default_rng(17)
        semantic = rng.normal(size=(6, 2, 700))
        for condition in range(6):
            semantic[condition, 0] += self.labels * (0.2 + condition / 10)
            semantic[condition, 1] += self.labels * (0.05 - condition / 20)
        arrays, report = diagnostic.answer_statistics(
            semantic, self.labels, 1 - self.labels, self.indices
        )
        radius = report["simultaneous_radius"]
        self.assertGreater(radius, 0)
        expected = np.percentile(
            np.max(
                np.abs(arrays["bootstrap_primary"][arrays["common_valid"]] - arrays["primary"]),
                axis=1,
            ),
            95,
            method="linear",
        )
        self.assertEqual(radius, expected)
        for contrast in report["contrasts"]:
            self.assertAlmostEqual(
                contrast["simultaneous_ci95"][1] - contrast["simultaneous_ci95"][0],
                2 * radius,
            )

    def test_frozen_control_assignments_require_complete_type_coverage(self):
        rows = [
            {"view_AP": 0, "sex_M": 0, "age": 50},
            {"view_AP": 1, "sex_M": 1, "age": 60},
        ]
        types = diagnostic.build_types(rows)
        assignments = [{"assignment": {kind: i % 2 for kind in types}} for i in range(20)]
        with patch.object(diagnostic, "N_PAIRS", 2):
            labels = diagnostic.frozen_control_labels(assignments, rows)
            self.assertEqual(labels.shape, (20, 2))
            assignments[0]["assignment"].pop(types[0])
            with self.assertRaisesRegex(ValueError, "does not cover"):
                diagnostic.frozen_control_labels(assignments, rows)


if __name__ == "__main__":
    unittest.main()
