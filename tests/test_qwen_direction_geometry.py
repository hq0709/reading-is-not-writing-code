from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.qwen_direction_geometry import (
    _finite_difference_summary,
    _load_directions,
    _load_per_image,
    main,
    summarize_direction_geometry,
    summarize_run,
)

CONCEPTS = (
    "Effusion",
    "Atelectasis",
    "Pneumothorax",
    "Cardiomegaly",
    "Mass",
    "Nodule",
)
ARCH = "qwen7b"
CONCEPT = "Effusion"
LOCUS = "vis.last"
EXPECTED_LOCUS = "model.visual.blocks.31"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SOURCE_RUN_ID = "20260903T000321Z-caaae3ef346d-qwen-full"
SOURCE_COMMIT = "caaae3ef346d3ac01c76c53ba99b3b7237066639"
PREVIOUS_COMMIT = "456c81bad460c986d632b1365c0017364b8f6c84"
ALPHA = 0.25
EVAL_ROWS = 400
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_905


class QwenDirectionGeometryTests(unittest.TestCase):
    def test_summary_reports_hand_derived_geometry_and_effect_competitor(self) -> None:
        raw_vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [1.0, 1.0],
                [1.0, -1.0],
                [0.0, -2.0],
            ]
        )
        coefficients = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
            ]
        )

        summary = summarize_direction_geometry(
            CONCEPTS, raw_vectors, coefficients, self._direction_effects()
        )

        self.assertEqual("post_hoc_descriptive", summary["analysis"])
        self.assertEqual(list(CONCEPTS), summary["concepts"])
        np.testing.assert_allclose(
            [
                [1.0, 0.0, -1.0, 0.7071067811865475, 0.7071067811865475, 0.0],
                [0.0, 1.0, 0.0, 0.7071067811865475, -0.7071067811865475, -1.0],
                [-1.0, 0.0, 1.0, -0.7071067811865475, -0.7071067811865475, 0.0],
                [0.7071067811865475, 0.7071067811865475, -0.7071067811865475, 1.0, 0.0, -0.7071067811865475],
                [0.7071067811865475, -0.7071067811865475, -0.7071067811865475, 0.0, 1.0, 0.7071067811865475],
                [0.0, -1.0, 0.0, -0.7071067811865475, 0.7071067811865475, 1.0],
            ],
            summary["raw_vector_cosine_matrix"],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            [
                [1.0, 0.0, -1.0, 0.0, 0.7071067811865475, 0.7071067811865475],
                [0.0, 1.0, 0.0, 0.0, 0.7071067811865475, -0.7071067811865475],
                [-1.0, 0.0, 1.0, 0.0, -0.7071067811865475, -0.7071067811865475],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.7071067811865475, 0.7071067811865475, -0.7071067811865475, 0.0, 1.0, 0.0],
                [0.7071067811865475, -0.7071067811865475, -0.7071067811865475, 0.0, 0.0, 1.0],
            ],
            summary["coefficient_cosine_matrix"],
            atol=1e-15,
        )
        self.assertEqual({"concept": "Effusion", "effect": 0.1}, summary["target"])
        self.assertEqual(
            {
                "concept": "Nodule",
                "effect": 0.15,
                "selected_by": "maximum accepted probability effect over five fixed clinical alternatives",
                "selection_scope": 5,
            },
            summary["strongest_non_target_clinical_competitor"],
        )
        self.assertAlmostEqual(0.05, summary["competitor_minus_target_gap"])
        self.assertEqual(0.0, summary["effusion_vs_competitor_cosines"]["raw_vector"])
        self.assertAlmostEqual(
            0.7071067811865475,
            summary["effusion_vs_competitor_cosines"]["projected_coefficient"],
        )

    def test_summary_rejects_malformed_identity_vectors_shapes_and_effect_keys(self) -> None:
        vectors = np.ones((6, 2), dtype=np.float64)
        coefficients = np.ones((6, 3), dtype=np.float64)
        effects = self._direction_effects()

        with self.assertRaisesRegex(ValueError, "concept identity"):
            summarize_direction_geometry(
                tuple(reversed(CONCEPTS)), vectors, coefficients, effects
            )
        vectors[2] = 0.0
        with self.assertRaisesRegex(ValueError, "finite nonzero"):
            summarize_direction_geometry(CONCEPTS, vectors, coefficients, effects)
        vectors[2] = 1.0
        with self.assertRaisesRegex(ValueError, "coefficient matrix shape"):
            summarize_direction_geometry(CONCEPTS, vectors, coefficients[:5], effects)
        with self.assertRaisesRegex(ValueError, "effect keys"):
            summarize_direction_geometry(
                CONCEPTS,
                vectors,
                coefficients,
                {key: value for key, value in effects.items() if key != "sham"},
            )
        effects["unrelated_Mass"] = 0.2
        with self.assertRaisesRegex(ValueError, "not pinned Nodule"):
            summarize_direction_geometry(CONCEPTS, vectors, coefficients, effects)

    def test_paired_bootstrap_uses_patient_rows_and_reports_selection(self) -> None:
        baseline_margin = np.linspace(-1.0, 1.0, EVAL_ROWS)
        shared_row_effect = np.linspace(-0.05, 0.05, EVAL_ROWS)
        margin_effects = {
            "concept": 0.2 + shared_row_effect,
            **{f"random{index}": np.full(EVAL_ROWS, 0.1) for index in range(20)},
            "sham": np.full(EVAL_ROWS, -0.05),
            "unrelated_Atelectasis": np.full(EVAL_ROWS, 0.05),
            "unrelated_Pneumothorax": np.full(EVAL_ROWS, 0.04),
            "unrelated_Cardiomegaly": np.full(EVAL_ROWS, 0.03),
            "unrelated_Mass": np.full(EVAL_ROWS, 0.02),
            "unrelated_Nodule": 0.4 + shared_row_effect,
        }
        margins = {("concept", 0.0): baseline_margin}
        probabilities = {("concept", 0.0): 1.0 / (1.0 + np.exp(-baseline_margin))}
        accepted_effects = {}
        for name, effect in margin_effects.items():
            shifted = baseline_margin + effect
            margins[(name, ALPHA)] = shifted
            probabilities[(name, ALPHA)] = 1.0 / (1.0 + np.exp(-shifted))
            accepted_effects[name] = float(
                (probabilities[(name, ALPHA)] - probabilities[("concept", 0.0)]).mean()
            )
        raw_cosines = np.eye(len(CONCEPTS)).tolist()
        raw_cosines[0][-1] = raw_cosines[-1][0] = -0.1

        summary = _finite_difference_summary(
            raw_cosines, probabilities, margins, accepted_effects
        )

        nodule_ci = summary["bootstrap"]["nodule_mean_margin_effect"]["descriptive_ci95"]
        paired_ci = summary["bootstrap"]["nodule_minus_effusion_margin_effect"][
            "descriptive_ci95"
        ]
        self.assertLess(nodule_ci[0], nodule_ci[1])
        np.testing.assert_allclose([0.2, 0.2], paired_ci, atol=1e-15)
        self.assertEqual(
            "post_hoc strongest of five fixed clinical alternatives",
            summary["exploratory_sign_inversion"]["selection"],
        )
        raw_cosines[0][-1] = raw_cosines[-1][0] = 0.1
        no_inversion = _finite_difference_summary(
            raw_cosines, probabilities, margins, accepted_effects
        )
        self.assertFalse(no_inversion["exploratory_sign_inversion"]["q_nodule_lt_zero"])
        self.assertFalse(no_inversion["exploratory_sign_inversion"]["observed"])

    def test_cli_loads_real_files_validates_identity_and_writes_deterministically(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            directions = root / "directions.npz"
            specificity = root / "direction-specificity-summary.json"
            per_image = root / "per-image.csv"
            out = root / "geometry.json"
            self._write_directions(directions)
            self._write_per_image(per_image)
            self._write_specificity_summary(specificity, self._sha256(per_image))
            directions_hash = self._sha256(directions)
            specificity_hash = self._sha256(specificity)
            per_image_hash = self._sha256(per_image)

            with (
                patch(
                    "src.qwen_direction_geometry.DIRECTIONS_SHA256", directions_hash
                ),
                patch(
                    "src.qwen_direction_geometry.SPECIFICITY_SUMMARY_SHA256",
                    specificity_hash,
                ),
                patch(
                    "src.qwen_direction_geometry.PER_IMAGE_SHA256", per_image_hash
                ),
                patch.object(
                    sys,
                    "argv",
                    [
                        "qwen_direction_geometry.py",
                        "--directions",
                        str(directions),
                        "--specificity-summary",
                        str(specificity),
                        "--per-image",
                        str(per_image),
                        "--out",
                        str(out),
                    ],
                ),
            ):
                main()
                first = out.read_bytes()
                main()

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(first, out.read_bytes())
            self.assertEqual("post_hoc_descriptive", payload["analysis"])
            self.assertEqual(
                {
                    "directions_sha256": directions_hash,
                    "per_image_sha256": per_image_hash,
                    "specificity_summary_sha256": specificity_hash,
                },
                payload["source_hashes"],
            )
            self.assertNotIn("p_value", json.dumps(payload))
            self.assertNotIn("confirmatory", json.dumps(payload).lower())
            self.assertEqual("exploratory-derived", payload["derivation"])
            self.assertFalse(payload["changes_causal_ownership_route"])
            self.assertEqual(BOOTSTRAP_RESAMPLES, payload["bootstrap"]["resamples"])
            self.assertEqual(BOOTSTRAP_SEED, payload["bootstrap"]["seed"])
            np.testing.assert_allclose(
                [0.2, 0.05, 0.04, 0.03, 0.02, 0.4],
                [
                    payload["clinical_mean_margin_effects"][concept]
                    for concept in CONCEPTS
                ],
                atol=1e-15,
            )
            self.assertAlmostEqual(
                0.04983401298522949,
                payload["clinical_mean_probability_effects"]["Effusion"],
            )
            self.assertAlmostEqual(
                0.09868764877319336,
                payload["clinical_mean_probability_effects"]["Nodule"],
            )
            self.assertEqual(-1.0, payload["raw_direction_cosines"]["Nodule"])
            np.testing.assert_allclose(
                [0.4, 0.4],
                payload["bootstrap"]["nodule_mean_margin_effect"]["descriptive_ci95"],
                atol=1e-15,
            )
            np.testing.assert_allclose(
                [0.2, 0.2],
                payload["bootstrap"]["nodule_minus_effusion_margin_effect"][
                    "descriptive_ci95"
                ],
                atol=1e-15,
            )
            self.assertEqual(
                {
                    "selection": "post_hoc strongest of five fixed clinical alternatives",
                    "q_nodule_lt_zero": True,
                    "both_bootstrap_lower_bounds_gt_zero": True,
                    "nodule_exceeds_all_random_margin_means_and_absolute_sham": True,
                    "observed": True,
                },
                payload["exploratory_sign_inversion"],
            )

    def test_loader_rejects_hash_and_run_identity_mismatches(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            directions = root / "directions.npz"
            specificity = root / "direction-specificity-summary.json"
            per_image = root / "per-image.csv"
            out = root / "geometry.json"
            self._write_directions(directions)
            self._write_per_image(per_image)
            self._write_specificity_summary(specificity, self._sha256(per_image))
            directions_hash = self._sha256(directions)
            specificity_hash = self._sha256(specificity)
            per_image_hash = self._sha256(per_image)

            with (
                patch(
                    "src.qwen_direction_geometry.DIRECTIONS_SHA256", "0" * 64
                ),
                patch(
                    "src.qwen_direction_geometry.SPECIFICITY_SUMMARY_SHA256",
                    specificity_hash,
                ),
                patch(
                    "src.qwen_direction_geometry.PER_IMAGE_SHA256", per_image_hash
                ),
                self.assertRaisesRegex(ValueError, "directions hash"),
            ):
                summarize_run(directions, specificity, per_image, out)

            malformed = json.loads(specificity.read_text(encoding="utf-8"))
            malformed["source_run_id"] = "different-run"
            specificity.write_text(
                json.dumps(malformed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with (
                patch(
                    "src.qwen_direction_geometry.DIRECTIONS_SHA256", directions_hash
                ),
                patch(
                    "src.qwen_direction_geometry.SPECIFICITY_SUMMARY_SHA256",
                    self._sha256(specificity),
                ),
                patch(
                    "src.qwen_direction_geometry.PER_IMAGE_SHA256", per_image_hash
                ),
                self.assertRaisesRegex(ValueError, "specificity summary identity"),
            ):
                summarize_run(directions, specificity, per_image, out)

    def test_loader_rejects_per_image_grid_patient_and_sigmoid_mismatches(self) -> None:
        for malformed, message in (
            ("grid", "grid"),
            ("patient", "patient identity"),
            ("sigmoid", "sigmoid"),
        ):
            with self.subTest(malformed=malformed), TemporaryDirectory() as temp:
                root = Path(temp)
                directions = root / "directions.npz"
                specificity = root / "direction-specificity-summary.json"
                per_image = root / "per-image.csv"
                self._write_directions(directions)
                self._write_per_image(per_image, malformed=malformed)
                self._write_specificity_summary(specificity, self._sha256(per_image))
                with (
                    patch(
                        "src.qwen_direction_geometry.DIRECTIONS_SHA256",
                        self._sha256(directions),
                    ),
                    patch(
                        "src.qwen_direction_geometry.SPECIFICITY_SUMMARY_SHA256",
                        self._sha256(specificity),
                    ),
                    patch(
                        "src.qwen_direction_geometry.PER_IMAGE_SHA256",
                        self._sha256(per_image),
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    summarize_run(directions, specificity, per_image, root / "out.json")

    def test_loader_rejects_raw_vectors_that_do_not_reconstruct(self) -> None:
        with TemporaryDirectory() as temp:
            directions = Path(temp) / "directions.npz"
            self._write_directions(directions, reconstruction_mismatch=True)
            with self.assertRaisesRegex(ValueError, "reconstruct"):
                _load_directions(directions)

    def test_loader_accepts_observed_float32_sigmoid_rounding(self) -> None:
        with TemporaryDirectory() as temp:
            per_image = Path(temp) / "per-image.csv"
            self._write_per_image(
                per_image, baseline_margin=0.625, use_observed_rounding=True
            )

            row_ids, probabilities, _ = _load_per_image(per_image)

        self.assertEqual(EVAL_ROWS, len(row_ids))
        self.assertEqual(0.6513549089431763, probabilities[("concept", 0.0)][0])

    @staticmethod
    def _direction_effects() -> dict[str, float]:
        return {
            "concept": 0.1,
            **{f"random{index}": 0.0 for index in range(20)},
            "sham": 0.0,
            "unrelated_Atelectasis": 0.03,
            "unrelated_Pneumothorax": 0.02,
            "unrelated_Cardiomegaly": 0.04,
            "unrelated_Mass": 0.05,
            "unrelated_Nodule": 0.15,
        }

    @staticmethod
    def _write_directions(path: Path, reconstruction_mismatch: bool = False) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.7071067811865475, 0.7071067811865475, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        coefficients = np.pad(vectors, ((0, 0), (0, 509)))
        if reconstruction_mismatch:
            vectors = vectors.copy()
            vectors[0] = [0.0, 1.0, 0.0]
        projection = np.zeros((3, 512), dtype=np.float64)
        projection[:, :3] = np.eye(3)
        np.savez(
            path,
            names=np.asarray(CONCEPTS),
            vectors=vectors,
            projection=projection,
            scale=np.ones(512, dtype=np.float64),
            coefficients=coefficients,
            locus=np.asarray(LOCUS),
            raw_dim=np.asarray(3),
            projection_dim=np.asarray(512),
            projection_seed=np.asarray(0),
            C=np.asarray(1.0),
            probe_seed=np.asarray(0),
        )

    @classmethod
    def _write_specificity_summary(cls, path: Path, per_image_hash: str) -> None:
        payload = {
            "arch": ARCH,
            "concept": CONCEPT,
            "locus": LOCUS,
            "module": EXPECTED_LOCUS,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_run_id": SOURCE_RUN_ID,
            "source_commit": SOURCE_COMMIT,
            "intervention_commit": PREVIOUS_COMMIT,
            "direction_specific": False,
            "direction_effects": cls._accepted_probability_effects(),
            "n_eval": EVAL_ROWS,
            "n_eval_patients": EVAL_ROWS,
            "alpha": ALPHA,
            "alpha_mode": "reltoken",
            "evidence": {"per_image_sha256": per_image_hash},
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _accepted_probability_effects() -> dict[str, float]:
        return {
            "concept": 0.04983401298522949,
            **{f"random{index}": 0.02497917413711548 for index in range(20)},
            "sham": -0.012497395277023315,
            "unrelated_Atelectasis": 0.012497425079345703,
            "unrelated_Pneumothorax": 0.009998679161071777,
            "unrelated_Cardiomegaly": 0.0074994564056396484,
            "unrelated_Mass": 0.004999816417694092,
            "unrelated_Nodule": 0.09868764877319336,
        }

    @staticmethod
    def _write_per_image(
        path: Path,
        malformed: str | None = None,
        baseline_margin: float = 0.0,
        use_observed_rounding: bool = False,
    ) -> None:
        margin_effects = {
            "concept": 0.2,
            **{f"random{index}": 0.1 for index in range(20)},
            "sham": -0.05,
            "unrelated_Atelectasis": 0.05,
            "unrelated_Pneumothorax": 0.04,
            "unrelated_Cardiomegaly": 0.03,
            "unrelated_Mass": 0.02,
            "unrelated_Nodule": 0.4,
        }
        conditions = [("concept", 0.0, 0.0), *(
            (direction, ALPHA, effect) for direction, effect in margin_effects.items()
        )]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "row_id",
                    "patient_id",
                    "locus",
                    "direction",
                    "alpha",
                    "p_yes",
                    "margin",
                ),
            )
            writer.writeheader()
            for row_index in range(EVAL_ROWS):
                for condition_index, (direction, alpha, effect) in enumerate(conditions):
                    if malformed == "grid" and row_index == EVAL_ROWS - 1 and condition_index == 27:
                        continue
                    patient_id = f"patient-{row_index}"
                    if malformed == "patient" and row_index == 1:
                        patient_id = "patient-0"
                    margin = baseline_margin + effect
                    probability = float(
                        np.float32(1.0 / (1.0 + math.exp(-margin)))
                    )
                    if use_observed_rounding and row_index == 0 and condition_index == 0:
                        probability = 0.6513549089431763
                    if malformed == "sigmoid" and row_index == 0 and condition_index == 0:
                        probability = 0.6
                    writer.writerow(
                        {
                            "row_id": f"row-{row_index}",
                            "patient_id": patient_id,
                            "locus": LOCUS,
                            "direction": direction,
                            "alpha": alpha,
                            "p_yes": probability,
                            "margin": margin,
                        }
                    )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
