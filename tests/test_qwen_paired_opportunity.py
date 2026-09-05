from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from scripts.audit_nih_paired_pool import DISEASES
from scripts.audit_paired_opportunity import COHORTS, patient_exclusions
from src.build_manifest import patient_split
from src import qwen_paired_opportunity as gate


def row(image, patient, labels, **values):
    return {"Image Index": image + ".png", "Patient ID": patient,
            "Finding Labels": labels, "Patient Age": "50", "Patient Sex": "F",
            "View Position": "PA", "Follow-up #": "0", "Height]": "1024", **values}


def fixture(count=110, concept="Mass"):
    patients = [str(i) for i in range(10000) if patient_split(str(i)) == "test"]
    exposed, known, unavailable = patients[:3]
    raw = [row(f"{p}_{label}", p, labels)
           for p in patients[:count + 3]
           for label, labels in (("p", concept), ("n", "No Finding"))]
    manifest = [{"row_id": exposed + "_old", "patient_id": exposed, "split": "test"},
                {"row_id": known + "_old", "patient_id": known, "split": "test"}]
    receipts = {name: {key: [exposed + "_old"]} for name, _, _, key in COHORTS}
    receipts["closure"] = {"pairs": [{"patient_id": exposed,
        "positive_row_id": exposed + "_old", "negative_row_id": exposed + "_old"}]}
    receipts["active"]["calibration_row_ids"] = [exposed + "_old"]
    have = {r["Image Index"] for r in raw} - {unavailable + "_n.png"}
    return raw, manifest, have, receipts


class SelectionTests(unittest.TestCase):
    def test_shuffle_default_count_full_metadata_and_no_scores(self):
        args = fixture()
        selected, receipt = gate.select_pairs(*args, "Mass")
        shuffled = deepcopy(args)
        np.random.default_rng(17).shuffle(shuffled[0])
        shuffled[1].reverse()
        self.assertEqual((selected, receipt), gate.select_pairs(*shuffled, "Mass"))
        self.assertEqual(len(selected), 100)
        self.assertEqual(len({p["patient_id"] for p in selected}), 100)
        self.assertEqual(receipt["eligible_patients"], 110)
        self.assertEqual(receipt["selection_seed"], 20260911)
        self.assertEqual(receipt["pairs"], selected)
        self.assertEqual(receipt["patient_ids"], [p["patient_id"] for p in selected])
        by_id = {r["Image Index"]: r for r in args[0]}
        for pair in selected:
            for side in ("positive", "negative"):
                self.assertEqual(pair[side + "_metadata"], by_id[pair[side + "_image_index"]])
            self.assertEqual(pair["positive_labels"], ["Mass"])
            self.assertEqual(pair["negative_labels"], ["No Finding"])
            self.assertFalse(pair["positive_no_finding"])
            self.assertTrue(pair["negative_no_finding"])
        self.assertNotIn("margin", json.dumps(receipt))
        self.assertNotIn("sha256", json.dumps(receipt))
        args[0][0]["Height]"] = "changed"
        self.assertNotIn("changed", json.dumps(receipt))

    def test_patient_exclusion_availability_and_authoritative_split(self):
        args = fixture(1)
        train = next(str(i) for i in range(100) if patient_split(str(i)) == "train")
        extra = [row("train_p", train, "Mass", split="test"),
                 row("train_n", train, "No Finding", split="test")]
        args[0].extend(extra)
        args[2].update(r["Image Index"] for r in extra)
        selected, receipt = gate.select_pairs(*args, "Mass", n_pairs=1)
        self.assertEqual(receipt["eligible_patients"], 1)
        self.assertNotIn(selected[0]["patient_id"], {r["patient_id"] for r in args[1]})
        self.assertIs(gate.patient_split, patient_split)
        self.assertIs(gate.patient_exclusions, patient_exclusions)
        self.assertEqual(gate.DISEASES, DISEASES)
        args[3]["active"]["calibration_row_ids"] = [args[1][1]["row_id"]]
        with self.assertRaisesRegex(ValueError, "calibration"):
            gate.select_pairs(*args, "Mass", n_pairs=1)

    def test_lexicographic_patient_and_pair_rng_order(self):
        args = fixture(7)
        for r in list(args[0]):
            other = dict(r, **{"Image Index": r["Image Index"].replace(".png", "_2.png")})
            args[0].append(other)
            if r["Image Index"] in args[2]:
                args[2].add(other["Image Index"])
        selected, receipt = gate.select_pairs(*args, "Mass", n_pairs=5)
        excluded = {r["patient_id"] for r in args[1]}
        eligible = sorted({r["Patient ID"] for r in args[0]
                           if r["Finding Labels"] == "No Finding"
                           and r["Image Index"] in args[2] and r["Patient ID"] not in excluded})
        rng = np.random.default_rng(20260911)
        patients = rng.permutation(eligible)[:5].tolist()
        self.assertEqual(receipt["patient_ids"], patients)
        for p, pair in zip(patients, selected):
            positives = [r["Image Index"] for r in args[0]
                         if r["Patient ID"] == p and r["Finding Labels"] == "Mass"]
            negatives = [r["Image Index"] for r in args[0]
                         if r["Patient ID"] == p and r["Finding Labels"] == "No Finding"]
            candidates = sorted((a, b) for a in positives for b in negatives)
            self.assertEqual((pair["positive_image_index"], pair["negative_image_index"]),
                             candidates[rng.integers(len(candidates))])

    def test_all_thirteen_labels_and_no_finding_independent(self):
        for concept in ("Mass", "Consolidation"):
            for other in set(DISEASES) - {concept}:
                with self.subTest(concept=concept, other=other):
                    args = fixture(1, concept)
                    args[0][-1]["Finding Labels"] = other
                    with self.assertRaisesRegex(ValueError, "eligible"):
                        gate.select_pairs(*args, concept, n_pairs=1)
                    args[0][-2]["Finding Labels"] = other + "|" + concept
                    pairs, _ = gate.select_pairs(*args, concept, n_pairs=1)
                    self.assertFalse(pairs[0]["negative_no_finding"])
                    args[0][-2]["Finding Labels"] += "|No Finding"
                    pairs, _ = gate.select_pairs(*args, concept, n_pairs=1)
                    self.assertTrue(pairs[0]["positive_no_finding"])

    def test_supported_nuisance_must_match(self):
        for field, value in (("Patient Age", "51"), ("Patient Sex", "M"),
                             ("View Position", "AP")):
            args = fixture(1)
            args[0][-1][field] = value
            with self.assertRaisesRegex(ValueError, "eligible"):
                gate.select_pairs(*args, "Mass", n_pairs=1)
        args = fixture(1)
        args[0][-1]["Patient Age"] = "050"
        self.assertEqual(len(gate.select_pairs(*args, "Mass", n_pairs=1)[0]), 1)

    def test_rejects_missing_invalid_metadata_target_and_small_pool(self):
        for field, value in (("Patient Age", "50.5"), ("Patient Age", float("nan")),
                             ("Patient Age", -1), ("Patient Age", 50.5),
                             ("Patient Sex", ""), ("Patient Sex", "Unknown"),
                             ("View Position", "LAT"), ("Finding Labels", ""),
                             ("Finding Labels", "Mass|Unknown"), ("Patient ID", "")):
            args = fixture(1)
            args[0][-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                gate.select_pairs(*args, "Mass", n_pairs=1)
        for field in ("Patient Age", "Patient Sex", "View Position", "Finding Labels", "Patient ID"):
            args = fixture(1)
            del args[0][-1][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                gate.select_pairs(*args, "Mass", n_pairs=1)
        for concept in (None, "", "Nodule", "mass"):
            with self.assertRaises(ValueError):
                gate.select_pairs(*fixture(1), concept, n_pairs=1)
        with self.assertRaisesRegex(ValueError, "100"):
            gate.select_pairs(*fixture(99), "Mass")
        args = fixture(1)
        args[0].append(dict(args[0][-1]))
        with self.assertRaisesRegex(ValueError, "unique"):
            gate.select_pairs(*args, "Mass", n_pairs=1)


class OpportunityTests(unittest.TestCase):
    def test_fixed_prompts_match_accepted_functions_without_importing_runner(self):
        root = Path(__file__).resolve().parents[1] / "src"
        namespace = {}
        ae_tree = ast.parse((root / "qwen_answer_encoding.py").read_text(encoding="utf-8"))
        mapping = next(n for n in ae_tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "mapping_instruction")
        exec(compile(ast.Module(body=[mapping], type_ignores=[]), "mapping", "exec"), namespace)
        mass_tree = ast.parse((root / "qwen_mass_prompt_specificity.py").read_text(encoding="utf-8"))
        body = [n for n in mass_tree.body if
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                 and t.id in ("QUESTIONS", "ENCODINGS") for t in n.targets))
                or (isinstance(n, ast.FunctionDef) and n.name in ("instruction", "prompt_for"))]
        from types import SimpleNamespace
        namespace["ae"] = SimpleNamespace(mapping_instruction=namespace["mapping_instruction"])
        exec(compile(ast.Module(body=body, type_ignores=[]), "prompts", "exec"), namespace)
        self.assertEqual(gate.PRIMARY_PROMPTS["Mass"], ("show/A_present", "show/B_present"))
        for cell in gate.PRIMARY_PROMPTS["Mass"]:
            self.assertEqual(gate.PROMPTS["Mass"][cell], namespace["prompt_for"](cell))
        consol_tree = ast.parse((root / "qwen_consolidation_input_closure.py").read_text(encoding="utf-8"))
        prompt = next(ast.literal_eval(n.value) for n in consol_tree.body
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                      and t.id == "PROMPT" for t in n.targets))
        self.assertEqual(gate.PROMPTS["Consolidation"], {"anchor": prompt})

    def test_import_does_not_load_torch(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; import src.qwen_paired_opportunity; assert 'torch' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_positive_negative_ties_and_one_prompt_fails(self):
        for values, available in (([2, 3], True), ([-2, -3], False),
                                  ([0, 0], False), ([2, -1], False), ([2, 0], False)):
            positive = np.repeat(np.array(values)[:, None], 100, axis=1)
            summary = gate.summarize_opportunity(positive, np.zeros((2, 100)), "Mass")
            self.assertIs(summary["opportunity_available"], available)
            for cell, value in zip(gate.PRIMARY_PROMPTS["Mass"], values):
                result = summary["prompts"][cell]
                self.assertAlmostEqual(result["mean_margin_gap"], value)
                self.assertAlmostEqual(result["margin_gap_lower_95"], value)
                np.testing.assert_allclose(result["margin_gap_ci95"], [value, value])
                self.assertEqual(result["mean_pair_ranking"], 1 if value > 0 else .5 if value == 0 else 0)
                self.assertEqual(len(result["per_patient"]["margin_gap"]), 100)
            self.assertNotIn("absence", json.dumps(summary))

    def test_semantic_orientation_and_stable_probability(self):
        positive = np.array([[2.] * 100, [3.] * 100])
        negative = -positive
        summary = gate.summarize_opportunity(positive, negative, "Mass")
        self.assertEqual(summary["prompts"]["show/B_present"]["mean_margin_gap"], 6)
        self.assertAlmostEqual(summary["prompts"]["show/A_present"]["mean_probability_gap"],
                               1 / (1 + np.exp(-2)) - 1 / (1 + np.exp(2)))
        with np.errstate(over="raise"):
            summary = gate.summarize_opportunity(np.full((1, 100), 1000.),
                                                 np.full((1, 100), -1000.), "Consolidation")
        self.assertEqual(summary["prompts"]["anchor"]["mean_probability_gap"], 1.)

    def test_joint_bootstrap_exact_percentiles_and_retention(self):
        gap = np.arange(100, dtype=float) - 65
        positive = np.stack((gap, 2 * gap))
        summary, arrays = gate.summarize_opportunity(positive, np.zeros((2, 100)), "Mass",
                                                   return_arrays=True)
        indices = np.random.default_rng(20260912).integers(0, 100, size=(5000, 100))
        expected = positive[:, indices].mean(axis=2)
        np.testing.assert_array_equal(arrays["bootstrap_indices"], indices)
        np.testing.assert_allclose(arrays["bootstrap_margin_gap"], expected)
        np.testing.assert_allclose(arrays["bootstrap_margin_gap"][1], 2 * expected[0])
        for i, cell in enumerate(gate.PRIMARY_PROMPTS["Mass"]):
            result = summary["prompts"][cell]
            np.testing.assert_allclose(result["margin_gap_ci95"], np.percentile(expected[i], [2.5, 97.5]))
            self.assertEqual(result["margin_gap_lower_95"], np.percentile(expected[i], 5))
            np.testing.assert_array_equal(result["per_patient"]["margin_gap"], positive[i])
        self.assertEqual(summary["bootstrap_seed"], 20260912)
        self.assertEqual(summary["bootstrap_resamples"], 5000)

    def test_no_finding_strata_are_descriptive_and_optional(self):
        positive = np.arange(100, dtype=float)[None, :]
        absent = gate.summarize_opportunity(positive, np.zeros_like(positive), "Consolidation")
        self.assertNotIn("negative_no_finding_strata", absent["prompts"]["anchor"])
        flags = np.arange(100) < 40
        summary = gate.summarize_opportunity(positive, np.zeros_like(positive), "Consolidation",
                                             negative_no_finding=flags)
        strata = summary["prompts"]["anchor"]["negative_no_finding_strata"]
        self.assertEqual(strata, {"true": {"patients": 40, "mean_margin_gap": 19.5},
                                  "false": {"patients": 60, "mean_margin_gap": 69.5}})
        all_true = gate.summarize_opportunity(positive, np.zeros_like(positive), "Consolidation",
                                              negative_no_finding=np.ones(100, dtype=bool))
        self.assertEqual(all_true["prompts"]["anchor"]["negative_no_finding_strata"]["false"],
                         {"patients": 0, "mean_margin_gap": None})

    def test_shape_finite_and_metadata_validation(self):
        valid = np.zeros((2, 100))
        for value in (np.zeros((2, 99)), np.zeros((100, 2)), np.zeros(100),
                      np.full((2, 100), np.nan), np.full((2, 100), np.inf)):
            for positive, negative in ((value, valid), (valid, value)):
                with self.assertRaises(ValueError):
                    gate.summarize_opportunity(positive, negative, "Mass")
        for flags in ([False] * 99, [[False] * 100], [None] * 100,
                      [float("nan")] * 100, [2] * 100, ["False"] * 100):
            with self.assertRaises(ValueError):
                gate.summarize_opportunity(valid, valid, "Mass", negative_no_finding=flags)
        with self.assertRaises(ValueError):
            gate.summarize_opportunity(valid, valid, "Nodule")


if __name__ == "__main__":
    unittest.main()
