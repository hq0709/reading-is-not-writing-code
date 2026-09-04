from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src import qwen_ownership_diagnostics as diagnostics


def probabilities(margin):
    return np.exp(-np.logaddexp(0, -margin))


class OwnershipDiagnosticsTests(unittest.TestCase):
    def arrays(self):
        labels = np.tile([0, 0, 1, 1], (6, 1))
        margins = np.broadcast_to([1.0, -1.0, -0.5, 0.5], (7, 6, 4)).copy()
        return margins, labels

    def test_uniform_shift_changes_calibration_without_changing_discrimination(self):
        margins, labels = self.arrays()
        margins[1:] += 2
        summary = diagnostics.summarize_diagnostics(probabilities(margins), margins, labels)
        for question in diagnostics.CONCEPTS:
            conditions = summary["label_conditioned_performance"][question]
            baseline = conditions["baseline"]
            for direction in diagnostics.CONCEPTS:
                steered = conditions[direction]
                self.assertEqual(baseline["auroc"], steered["auroc"])
                self.assertGreater(steered["brier_score"], baseline["brier_score"])
                self.assertEqual(0, steered["correct_label_margin_effect"])
                self.assertEqual(0, steered["label_gap_change"])
                self.assertEqual([0, 0], steered["label_gap_change_ci95"])
                for subgroup in steered["label_subgroups"].values():
                    self.assertEqual({"count": 2, "mean_margin_shift": 2}, subgroup)
        np.testing.assert_allclose(summary["logit_margin"]["mean_clinical_effect_matrix"], 2)
        self.assertAlmostEqual(1, summary["logit_margin"]["geometry"][
            "raw_top_singular_value_energy_fraction"])
        self.assertEqual(0, summary["logit_margin"]["geometry"][
            "two_way_centered_interaction_energy_fraction"])

    def test_label_oriented_shift_improves_discrimination(self):
        margins, labels = self.arrays()
        margins[1:] += 2 * labels - 1
        summary = diagnostics.summarize_diagnostics(probabilities(margins), margins, labels)
        for question, conditions in summary["label_conditioned_performance"].items():
            steered = conditions[question]
            self.assertGreater(steered["auroc"], conditions["baseline"]["auroc"])
            self.assertLess(steered["brier_score"], conditions["baseline"]["brier_score"])
            self.assertEqual(1, steered["correct_label_margin_effect"])
            self.assertEqual(2, steered["label_gap_change"])
            self.assertEqual([2, 2], steered["label_gap_change_ci95"])

    def test_constant_labels_and_empty_subgroups_are_unavailable(self):
        margins, labels = self.arrays()
        labels[0], labels[1] = 0, 1
        summary = diagnostics.summarize_diagnostics(probabilities(margins), margins, labels)
        for q, absent in ((0, "positive"), (1, "negative")):
            for item in summary["label_conditioned_performance"][diagnostics.CONCEPTS[q]].values():
                self.assertIsNone(item["auroc"])
                self.assertEqual("constant labels", item["auroc_unavailable_reason"])
                self.assertIsNone(item["label_gap_change_ci95"])
                self.assertEqual({"count": 0, "mean_margin_shift": None},
                                 item["label_subgroups"][absent])
        json.dumps(summary, allow_nan=False)

    def test_two_way_centering_and_zero_energy(self):
        additive = np.arange(6)[:, None] + 2 * np.arange(6)[None, :]
        geometry = diagnostics.matrix_geometry(additive)
        self.assertEqual(0, geometry["two_way_centered_interaction_energy_fraction"])
        interaction = np.outer([-1, 1, -1, 1, -1, 1], [-1, 1, -1, 1, -1, 1])
        self.assertAlmostEqual(1, diagnostics.matrix_geometry(interaction)[
            "two_way_centered_interaction_energy_fraction"])
        zero = diagnostics.matrix_geometry(np.zeros((6, 6)))
        self.assertIsNone(zero["raw_top_singular_value_energy_fraction"])
        self.assertIsNone(zero["two_way_centered_interaction_energy_fraction"])

    def test_joint_bootstrap_matches_index_resampling_and_recomputes_maximum(self):
        margins, labels = self.arrays()
        margins[1:] += np.random.default_rng(7).normal(size=(6, 6, 4))
        p = probabilities(margins)
        effects = np.stack([p[1:] - p[0], margins[1:] - margins[0]])
        draws, gaps = diagnostics._bootstrap(effects, labels)
        indices = np.random.default_rng(20260906).integers(0, 4, size=(2000, 4))
        expected = np.stack([effects[..., index].mean(axis=-1) for index in indices])
        np.testing.assert_allclose(draws, expected, atol=1e-15)
        for b, index in enumerate(indices[:20]):
            y = labels[0, index]
            if np.unique(y).size == 2:
                shift = effects[1, 0, 0, index]
                self.assertAlmostEqual(shift[y == 1].mean() - shift[y == 0].mean(), gaps[b, 0, 0])
            else:
                self.assertTrue(np.isnan(gaps[b, 0, 0]))
        summary = diagnostics.summarize_diagnostics(p, margins, labels)
        for s, scale in enumerate(("probability", "logit_margin")):
            np.testing.assert_allclose(summary[scale]["mean_clinical_effect_ci95"],
                                       np.percentile(expected[:, s], [2.5, 97.5], axis=0)
                                       .transpose(1, 2, 0))
            for q, question in enumerate(diagnostics.CONCEPTS):
                others = [d for d in range(6) if d != q]
                ownership = expected[:, s, q, q] - expected[:, s, others, q].max(axis=1)
                item = summary[scale]["questions"][question]
                np.testing.assert_allclose(item["ownership_ci95"], np.percentile(ownership, [2.5, 97.5]))
                matrix = effects[s].mean(axis=-1)
                self.assertEqual(1 + np.sum(matrix[:, q] > matrix[q, q]), item["named_direction_rank"])
                for d, direction in enumerate(diagnostics.CONCEPTS):
                    expected_gap = expected[:, s, q, q] - expected[:, s, d, q]
                    np.testing.assert_allclose(item["gap_differences"][direction]["ci95"],
                                               np.percentile(expected_gap, [2.5, 97.5]))

    def test_numeric_consistency_and_shapes(self):
        margins, labels = self.arrays()
        for bad in (np.nan, np.inf, -0.1, 1.1, 0.1234):
            p = probabilities(margins)
            p[1, 0, 0] = bad
            with self.subTest(probability=bad), self.assertRaises(ValueError):
                diagnostics.summarize_diagnostics(p, margins, labels)
        for bad in (np.nan, np.inf):
            m = margins.copy()
            m[0, 0, 0] = bad
            with self.assertRaises(ValueError):
                diagnostics.summarize_diagnostics(probabilities(margins), m, labels)
        with self.assertRaises(ValueError):
            diagnostics.summarize_diagnostics(probabilities(margins)[:, :, :3], margins, labels)
        labels[0, 0] = 2
        with self.assertRaises(ValueError):
            diagnostics.summarize_diagnostics(probabilities(margins), margins, labels)
        m = np.asarray([-1000, -10, 0, 10, 1000], dtype=float)
        diagnostics._check_numeric(probabilities(m).astype(np.float32), m)

    def fixture(self, root):
        row_ids, patients = [f"r{i}" for i in range(4)], [f"p{i}" for i in range(4)]
        commit = "a3bd883540eb" + "0" * 28
        metadata = {"RUN_ID": diagnostics.OWNERSHIP_RUN_ID, "SOURCE_COMMIT": commit,
                    "COMMAND_STATUS": "0", "DISPATCHER_STATUS": "0",
                    "CLEANUP_STATUS": "0", "ABORT_SIGNAL": "none"}
        (root / "metadata.env").write_text("\n".join(f"{k}={v}" for k, v in metadata.items()))
        for name in ("command_exit_status", "exit_status"):
            (root / name).write_text("0\n")
        artifacts = root / "artifacts"
        artifacts.mkdir()
        (artifacts / "registered-rows.json").write_text(json.dumps(
            {"row_ids": row_ids, "patient_ids": patients}))
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row_id", "patient_id", "split", *diagnostics.CONCEPTS])
            for i in reversed(range(4)):
                writer.writerow([row_ids[i], patients[i], "test", *[(i + q) % 2 for q in range(6)]])
        question_records = []
        for q, question in enumerate(diagnostics.CONCEPTS):
            directory = artifacts / "questions" / question.lower()
            directory.mkdir(parents=True)
            meta = {"source_commit": commit, "concept": question, "n_eval": 4, "n_random": 119,
                    "alpha_mode": "reltoken", "alphas": [0, .25], "control_alphas": [.25],
                    "eval_row_ids": row_ids}
            (directory / "meta.json").write_text(json.dumps(meta))
            (directory / "DONE").write_text(json.dumps({"loci": ["vis.last"], "rows": 127}))
            conditions = [("concept", 0)] + [
                ("concept" if c == question else f"unrelated_{c}", .25) for c in diagnostics.CONCEPTS
            ] + [(f"random{i}", .25) for i in range(119)] + [("sham", .25)]
            records = []
            for d, (direction, alpha) in enumerate(conditions):
                for i in range(4):
                    margin = d * .01 + q * .1 + i * .001
                    records.append({"row_id": row_ids[i], "patient_id": patients[i],
                                    "locus": "vis.last", "direction": direction, "alpha": alpha,
                                    "p_yes": float(probabilities(margin)), "margin": margin})
            np.random.default_rng(q).shuffle(records)
            self.write_records(directory / "per-image.csv", records)
            question_records.append(records)
        return manifest, row_ids, patients, question_records

    def write_records(self, path, records):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def test_loader_alignment_terminal_receipts_and_exact_cohort(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _, _, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "exactly 400"):
                diagnostics.load_ownership_run(root, manifest)
            with patch.object(diagnostics, "EVAL_ROWS", 4):
                p, m, y, provenance = diagnostics.load_ownership_run(root, manifest)
                expected = (np.arange(7)[:, None, None] * .01 + np.arange(6)[None, :, None] * .1
                            + np.arange(4)[None, None, :] * .001)
                np.testing.assert_allclose(m, expected)
                np.testing.assert_allclose(p, probabilities(expected))
                np.testing.assert_array_equal(y, (np.arange(6)[:, None] + np.arange(4)) % 2)
                self.assertEqual(diagnostics.OWNERSHIP_RUN_ID, provenance["source_run_id"])
                (root / "exit_status").write_text("1")
                with self.assertRaisesRegex(ValueError, "exit receipt"):
                    diagnostics.load_ownership_run(root, manifest)

    def test_cli_emits_finite_summary_without_patient_identifiers(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _, _, _ = self.fixture(root)
            out = root / "output" / "diagnostics.json"
            argv = ["diagnostics", "--ownership-run", str(root), "--manifest", str(manifest),
                    "--out", str(out)]
            with patch.object(diagnostics, "EVAL_ROWS", 4), patch("sys.argv", argv):
                diagnostics.main()
            text = out.read_text(encoding="utf-8")
            summary = json.loads(text)
            self.assertLess(len(text), 100_000)
            self.assertNotIn('"row_ids"', text)
            self.assertNotIn('"patient_ids"', text)
            self.assertNotIn("sha256", text)
            self.assertEqual(2000, summary["bootstrap"]["resamples"])
            self.assertEqual(20260906, summary["bootstrap"]["seed"])
            self.assertEqual("a3bd883540eb" + "0" * 28, summary["provenance"]["source_commit"])

    def test_loader_rejects_manifest_and_question_identity_mismatch(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _, _, _ = self.fixture(root)
            with patch.object(diagnostics, "EVAL_ROWS", 4):
                original = manifest.read_text()
                manifest.write_text(original.replace("r0,p0", "r0,p1"))
                with self.assertRaisesRegex(ValueError, "manifest registered patient alignment"):
                    diagnostics.load_ownership_run(root, manifest)
                manifest.write_text(original)
                meta_path = root / "artifacts/questions/effusion/meta.json"
                meta = json.loads(meta_path.read_text())
                meta["eval_row_ids"].reverse()
                meta_path.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, "question metadata"):
                    diagnostics.load_ownership_run(root, manifest)

    def test_malformed_grid_including_controls(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, rows, patients, records_by_question = self.fixture(root)
            records = records_by_question[0]
            path = root / "grid.csv"
            variants = [records[:-1], records + [records[0]]]
            for field, value in (("patient_id", "other"), ("row_id", "other"), ("locus", "other"),
                                 ("direction", "random119"), ("alpha", .5),
                                 ("margin", float("nan")), ("p_yes", .001)):
                changed = [dict(record) for record in records]
                control = next(r for r in changed if r["direction"] == "random0")
                control[field] = value
                variants.append(changed)
            for variant in variants:
                self.write_records(path, variant)
                with self.assertRaises(ValueError):
                    diagnostics._question_values(path, diagnostics.CONCEPTS[0], rows, patients)


if __name__ == "__main__":
    unittest.main()
