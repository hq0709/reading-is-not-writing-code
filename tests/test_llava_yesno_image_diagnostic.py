from __future__ import annotations

import inspect
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from src import llava_yesno_image_diagnostic as diagnostic
from src import run_llava_yesno_image_diagnostic as runner


def raw(image, patient, finding="No Finding", age="50", view="PA", sex="F"):
    return {
        "Image Index": image + ".png",
        "Finding Labels": finding,
        "Patient ID": patient,
        "Patient Age": age,
        "Patient Sex": sex,
        "View Position": view,
    }


class ProtocolTests(unittest.TestCase):
    def test_cpu_import_and_help_do_not_load_torch(self):
        command = (
            "import src.run_llava_yesno_image_diagnostic; "
            "import sys; assert 'torch' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        completed = subprocess.run(
            [sys.executable, "-m", "src.run_llava_yesno_image_diagnostic", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_exact_prompts_grid_and_primary_vector(self):
        self.assertEqual([item["name"] for item in diagnostic.CONDITIONS], ["0Y", "1Y"])
        self.assertEqual(
            [item["prompt"] for item in diagnostic.CONDITIONS],
            [
                "Is there a pleural effusion in this chest radiograph? Answer yes or no.",
                "Does this chest radiograph show a pleural effusion? Answer yes or no.",
            ],
        )
        self.assertEqual(runner.protocol()["allocation"], [700, 700, 20260916])
        self.assertEqual(runner.protocol()["bootstrap"], [10_000, 20260917])
        self.assertEqual(runner.protocol()["primary_names"], list(diagnostic.PRIMARY_NAMES))
        np.testing.assert_allclose(
            diagnostic.primary_contrasts(np.asarray([0.8, 0.7]), np.asarray([0.2, 0.35])),
            [0.3, 0.2, 0.15],
        )

    def test_identifier_first_allocation_reuses_parent_algorithm(self):
        rows = [
            raw("m", "manifest"),
            raw("b", "p1", "Effusion"),
            raw("a", "p1"),
            raw("c", "p2"),
            raw("d", "p3", "Effusion"),
            raw("e", "p4"),
            raw("f", "p5"),
        ]
        with (
            patch.object(diagnostic.allocation_source, "N_PAIRS", 2),
            patch.object(diagnostic.allocation_source, "patient_split", return_value="val"),
        ):
            allocation = diagnostic.allocate(
                rows, [{"row_id": "m", "patient_id": "manifest"}], "/images"
            )
        selected = allocation["index"] + allocation["donor"]
        self.assertEqual(len({row["patient_id"] for row in selected}), 4)
        self.assertNotIn("manifest", {row["patient_id"] for row in selected})
        chosen = {row["patient_id"]: row for row in selected}
        if "p1" in chosen:
            self.assertEqual(chosen["p1"]["image_index"], "a.png")
            self.assertEqual(chosen["p1"]["Effusion"], 0)

    def test_timing_includes_preparation_pilot_and_allowance(self):
        pilot, preparation = 100.0, 20.0
        boundary_gpu = 2700 - preparation - 2802 * pilot / 512 - 600
        self.assertTrue(runner.budget(pilot, boundary_gpu, preparation)["passed"])
        self.assertFalse(runner.budget(pilot, boundary_gpu + 1e-6, preparation)["passed"])

    def test_image_grid_precedes_the_two_text_only_outputs(self):
        source = inspect.getsource(runner.run)
        self.assertLess(
            source.index('temporary.replace(out / "per-image.csv")'), source.index("text_only = []")
        )

    def test_yes_no_preflight_requires_primary_and_lse_signs(self):
        token = {
            "raw_first": runner.PINNED_YES,
            "raw_second": runner.PINNED_NO,
            "present": runner.PINNED_YES,
            "absent": runner.PINNED_NO,
            "candidate_ids": sorted(runner.PINNED_YES + runner.PINNED_NO),
        }
        cases = []
        for identity in runner.expected_mapping_cases():
            sign = 2 * identity["expected_present"] - 1
            cases.append({**identity, "semantic_margin": float(sign), "lse_margin": float(sign)})
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
            "token_ids": {"0Y": token, "1Y": dict(token)},
            "throughput": runner.budget(1.0, 10.0, 2.0),
        }
        self.assertTrue(runner.validate_preflight(flight, "a" * 40, "/model", 2.0))
        changed = {**flight, "mapping_cases": [dict(item) for item in cases]}
        changed["mapping_cases"][1]["lse_margin"] = 0.0
        with self.assertRaisesRegex(ValueError, "orientation"):
            runner.validate_preflight(changed, "a" * 40, "/model", 2.0)


class GridTests(unittest.TestCase):
    def test_complete_2800_shape_uses_index_labels_for_donors(self):
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
                    for index, row in enumerate(allocation[role]):
                        margin = float(index - 1)
                        records.append(
                            {
                                "source_commit": "a" * 40,
                                "condition": condition["name"],
                                "wording": condition["wording"],
                                "prompt": condition["prompt"],
                                "image_role": role,
                                "pair_index": index,
                                "patient_id": row["patient_id"],
                                "row_id": row["row_id"],
                                "image_path": row["image_path"],
                                "index_label": allocation["index"][index]["Effusion"],
                                "source_label": row["Effusion"],
                                "raw_margin": margin,
                                "semantic_margin": margin,
                                "probability": float(diagnostic.sigmoid(margin)),
                                "answer_token_mass": 0.2,
                                "lse_margin": margin + 0.1,
                            }
                        )
            values = diagnostic.validate_grid(records, allocation, "a" * 40)
            self.assertEqual(values["semantic_margin"].shape, (2, 2, 3))
            donor = [dict(item) for item in records]
            donor[3]["index_label"] = 1 - int(donor[3]["index_label"])
            with self.assertRaisesRegex(ValueError, "label"):
                diagnostic.validate_grid(donor, allocation, "a" * 40)


class InferenceTests(unittest.TestCase):
    @staticmethod
    def draws(theta, radius=0.01):
        offsets = np.resize(np.asarray([-radius, radius]), 10_000)
        return np.asarray(theta)[None, :] + offsets[:, None]

    def test_strict_opportunity_and_both_wording_directions(self):
        positive = diagnostic.primary_decision(
            np.asarray([0.20, 0.15, 0.08]),
            self.draws([0.20, 0.15, 0.08]),
            np.ones(10_000, dtype=bool),
        )
        self.assertTrue(positive["opportunity_detected"])
        self.assertEqual(positive["wording_dependence"], "positive")
        negative = diagnostic.primary_decision(
            np.asarray([0.20, 0.15, -0.08]),
            self.draws([0.20, 0.15, -0.08]),
            np.ones(10_000, dtype=bool),
        )
        self.assertEqual(negative["wording_dependence"], "negative")
        failed_conjunction = diagnostic.primary_decision(
            np.asarray([0.20, 0.005, 0.08]),
            self.draws([0.20, 0.005, 0.08]),
            np.ones(10_000, dtype=bool),
        )
        self.assertFalse(failed_conjunction["opportunity_detected"])

    def test_unresolved_insufficient_and_degenerate_inference(self):
        unresolved = diagnostic.primary_decision(
            np.asarray([0.20, 0.15, 0.0]),
            self.draws([0.20, 0.15, 0.0]),
            np.ones(10_000, dtype=bool),
        )
        self.assertEqual(unresolved["wording_dependence"], "unresolved")
        insufficient = diagnostic.primary_decision(
            np.asarray([0.20, 0.15, 0.08]),
            self.draws([0.20, 0.15, 0.08]),
            np.arange(10_000) < 9499,
        )
        self.assertEqual(insufficient["population_inference"], "UNAVAILABLE")
        self.assertIsNone(insufficient["opportunity_detected"])
        degenerate = diagnostic.primary_decision(
            np.asarray([0.20, 0.15, 0.08]),
            np.tile([0.20, 0.15, 0.08], (10_000, 1)),
            np.ones(10_000, dtype=bool),
        )
        self.assertTrue(degenerate["bootstrap_degenerate"])
        self.assertIsNone(degenerate["wording_dependence"])

    def test_reader_decision_is_independent_of_answer_family(self):
        self.assertEqual(
            diagnostic.reader_decision(25, 675, 10_000, [0.01, 0.2]),
            {
                "population_inference": "AVAILABLE",
                "unavailable_reason": None,
                "replicated": True,
            },
        )
        unavailable = diagnostic.reader_decision(9, 691, 10_000, [0.01, 0.2])
        self.assertEqual(unavailable["population_inference"], "UNAVAILABLE")
        self.assertFalse(unavailable["replicated"])

    def test_answer_statistics_uses_index_labels_for_both_donor_aurocs(self):
        index_labels = (np.arange(700) % 2).astype(np.int8)
        donor_labels = 1 - index_labels
        semantic = np.empty((2, 2, 700))
        for condition in range(2):
            for role in range(2):
                semantic[condition, role] = condition * 2 + role
        calls = []
        points = {0: 0.80, 1: 0.55, 2: 0.75, 3: 0.45}
        offsets = np.resize(np.asarray([-0.01, 0.01]), 10_000)

        def fake_auc(scores, labels, _counts):
            marker = int(scores[0])
            calls.append(np.asarray(labels).copy())
            point = points[marker] if np.array_equal(labels, index_labels) else 0.60
            return point, point + offsets

        with (
            patch.object(diagnostic, "validate_bootstrap", return_value=np.empty(0)),
            patch.object(diagnostic, "multiplicities", return_value=np.empty(0)),
            patch.object(diagnostic, "auc_point_draws", side_effect=fake_auc),
        ):
            arrays, report = diagnostic.answer_statistics(
                semantic, index_labels, donor_labels, np.empty(0)
            )
        self.assertTrue(np.array_equal(calls[1], index_labels))
        self.assertTrue(np.array_equal(calls[4], index_labels))
        self.assertTrue(np.array_equal(calls[2], donor_labels))
        self.assertTrue(np.array_equal(calls[5], donor_labels))
        np.testing.assert_allclose(arrays["image_advantage"], [0.25, 0.30])
        self.assertEqual(report["population_inference"], "AVAILABLE")

    def test_frozen_reader_replays_every_control_fit_separately(self):
        raw_activation = np.zeros((2, 1024), dtype=np.float16)
        raw_activation[1, :512] = 1
        projection = np.zeros((1024, 512), dtype=np.float32)
        projection[:512] = np.eye(512, dtype=np.float32)
        control_coefficients = np.zeros((20, 512), dtype=np.float32)
        control_intercepts = np.arange(20, dtype=np.float32)
        for index in range(20):
            control_coefficients[index, index] = index + 1
        prepared = {
            "projection": projection,
            "scale": np.ones(512, dtype=np.float64),
            "train_mean": np.zeros(512, dtype=np.float64),
            "coefficients": np.vstack(
                (np.ones((1, 512), dtype=np.float32), np.zeros((5, 512), dtype=np.float32))
            ),
            "control_coefficients": control_coefficients,
            "control_intercepts": control_intercepts,
        }
        projected, features, clinical, controls = diagnostic.frozen_reader_scores(
            raw_activation, prepared
        )
        np.testing.assert_array_equal(projected, features)
        np.testing.assert_allclose(clinical, [0, 512])
        for index in range(20):
            np.testing.assert_allclose(controls[index], [index, index + index + 1])

    def test_all_twenty_frozen_control_maps_must_cover_each_type(self):
        rows = [
            {"view_AP": 0, "sex_M": 0, "age": 50},
            {"view_AP": 1, "sex_M": 1, "age": 60},
        ]
        assignments = [{"assignment": {"0_0_5": 0, "1_1_6": 1}} for _ in range(20)]
        with patch.object(diagnostic.allocation_source, "N_PAIRS", 2):
            labels = diagnostic.frozen_control_labels(assignments, rows)
            self.assertEqual(labels.shape, (20, 2))
            assignments[7]["assignment"].pop("1_1_6")
            with self.assertRaisesRegex(ValueError, "does not cover"):
                diagnostic.frozen_control_labels(assignments, rows)


if __name__ == "__main__":
    unittest.main()
