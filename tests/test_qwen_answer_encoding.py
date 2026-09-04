from __future__ import annotations

import csv
import subprocess
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src import qwen_answer_encoding as ae


class Tokenizer:
    def __init__(self, tokens=None):
        self.tokens = tokens or {"A": [0], " A": [2], "B": [1], " B": [3]}

    def encode(self, text, add_special_tokens=False):
        return self.tokens.get(text, [99, 100])


def rows_fixture(n=20):
    return [dict(row_id=f"row{i}", patient_id=f"patient{i}",
                 **{q: str(i % 2) for q in ae.CONCEPTS}) for i in range(n)]


def grid_fixture(rows, encodings=ae.ENCODINGS, questions=ae.CONCEPTS):
    for encoding in encodings:
        for qi, question in enumerate(questions):
            for direction, alpha in ae.CONFIGURATIONS:
                di = ae.DIRECTIONS.index(direction) if direction != "baseline" else 0
                for i, row in enumerate(rows):
                    baseline = 2 * int(row[question]) - 1 + i / 100
                    effect = alpha * ((di + 1) * 0.1 + qi * 0.01)
                    semantic = baseline + effect
                    raw = -semantic if encoding == "B_present" else semantic
                    yield {"row_id": row["row_id"], "patient_id": row["patient_id"], "question": question,
                               "encoding": encoding, "direction": direction, "alpha": alpha, "locus": ae.LOCUS,
                               "label": row[question], "raw_margin": raw, "semantic_margin": semantic,
                               "semantic_probability": float(expit(semantic))}


def passed_preflight():
    return {"passed": True, "alpha_zero_exact": True, "nonzero_max_logit_change": 0.25,
                "throughput": {"passed": True},
                "mapping_cases": [{**case, "semantic_margin": 2 * case["expected_present"] - 1}
                               for case in ae.mapping_cases()]}


def write_fixture(root, rows, calibration_rows=None):
    if calibration_rows is None:
        calibration_rows = [{**r, "row_id": "cal-" + r["row_id"], "patient_id": "cal-" + r["patient_id"]}
                            for r in rows]
    meta = {"analysis": "registered_paired_sensitivity", "n_eval": len(rows),
                "ownership_run_id": ae.OWNERSHIP_RUN_ID, "ownership_commit": ae.OWNERSHIP_COMMIT,
                "original_ownership_decisions": {"n_owned_concepts": 0},
                "locus": ae.LOCUS, "module": ae.EXPECTED_LOCUS, "alpha_mode": "reltoken",
                "encodings": list(ae.ENCODINGS), "directions": list(ae.DIRECTIONS), "doses": list(ae.DOSES),
                "prompts": {e: {q: ae.prompt_for(q, e) for q in ae.CONCEPTS} for e in ae.ENCODINGS},
                "rows": rows, "calibration_rows": calibration_rows}
    ae.write_json(root / "meta.json", meta)
    ae.write_json(root / "preflight.json", passed_preflight())
    with (root / "per-image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ae.FIELDS)
        writer.writeheader()
        writer.writerows(grid_fixture(rows))
    with (root / "calibration.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ae.FIELDS)
        writer.writeheader()
        writer.writerows(r for r in grid_fixture(calibration_rows, ae.ENCODINGS[1:]) if r["alpha"] == 0)


class AnswerEncodingTests(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        completed = subprocess.run([sys.executable, "-c",
            "import src.qwen_answer_encoding; import sys; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_exact_prompts_and_grid_size(self):
        self.assertEqual(len(ae.CONFIGURATIONS), 55)
        self.assertEqual(200 * 6 * 3 * len(ae.CONFIGURATIONS), 198000)
        for q in ae.CONCEPTS:
            self.assertEqual(ae.prompt_for(q, "standard"), ae.PROMPTS[q])
        self.assertEqual(ae.prompt_for("Effusion", "A_present"),
            "Does this chest radiograph show a pleural effusion? Answer A if the finding is present "
            "and B if it is absent. Reply with A or B only.")
        self.assertEqual(ae.prompt_for("Nodule", "B_present"),
            "Does this chest radiograph show a lung nodule? Answer B if the finding is present "
            "and A if it is absent. Reply with A or B only.")

    def test_singleton_token_ids_and_semantic_orientation(self):
        self.assertEqual(ae.singleton_ids(Tokenizer()), [[0, 2], [1, 3]])
        self.assertEqual(ae.singleton_ids(Tokenizer({"A": [0], "B": [1]})), [[0], [1]])
        for tokens in ({"A": [0]}, {"A": [0], "B": [0]}):
            with self.assertRaisesRegex(ValueError, "singleton"):
                ae.singleton_ids(Tokenizer(tokens))
        for encoding, expected in (("standard", 3), ("A_present", 3), ("B_present", -3)):
            self.assertEqual(float(ae.orient(3, encoding)), expected)
        with self.assertRaises(ValueError):
            ae.orient(1, "other")

    def test_ab_scoring_uses_max_plain_and_space_logits(self):
        import torch
        logits = torch.tensor([[[2., 7., 5., 3.]], [[8., 4., 1., 6.]]])
        a = ae.score_logits(logits, Tokenizer(), "A_present", [[0, 2], [1, 3]])
        b = ae.score_logits(logits, Tokenizer(), "B_present", [[0, 2], [1, 3]])
        np.testing.assert_array_equal(a[:, 0], [-2, 2])
        np.testing.assert_array_equal(b[:, 0], [-2, 2])
        np.testing.assert_array_equal(b[:, 1], [2, -2])
        np.testing.assert_allclose(a[:, 2], expit([-2, 2]))
        np.testing.assert_allclose(a[:, 2] + b[:, 2], 1)

    def test_pure_semantic_and_pure_token_crossover(self):
        effect = np.arange(24, dtype=float).reshape(2, 3, 4)
        semantic, token = ae.crossover(effect, -effect)
        np.testing.assert_array_equal(semantic, effect)
        np.testing.assert_array_equal(token, 0)
        semantic, token = ae.crossover(effect, effect)
        np.testing.assert_array_equal(semantic, 0)
        np.testing.assert_array_equal(token, effect)

    def test_common_random_and_question_sham_identities(self):
        rng = np.random.default_rng(43)
        vectors = rng.standard_normal((6, 24)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        clinical = dict(zip(ae.CONCEPTS, vectors))
        directions = ae.common_directions(clinical)
        reference = np.random.default_rng(0)
        for i in range(20):
            expected = reference.standard_normal(24).astype(np.float32)
            expected /= np.linalg.norm(expected)
            for question in ae.CONCEPTS:
                np.testing.assert_array_equal(directions[question][f"random{i}"], expected)
        permutation = reference.permutation(24)
        for question in ae.CONCEPTS:
            normalized = clinical[question] / np.linalg.norm(clinical[question])
            sham = normalized[permutation]
            np.testing.assert_array_equal(directions[question]["sham"], sham / np.linalg.norm(sham))
            for concept in ae.CONCEPTS:
                np.testing.assert_array_equal(directions[question][concept], clinical[concept] / np.linalg.norm(clinical[concept]))
            for vector in directions[question].values():
                self.assertAlmostEqual(float(np.linalg.norm(vector)), 1., places=6)

    def test_preflight_requires_all_eight_correct_mapping_signs(self):
        cases = passed_preflight()["mapping_cases"]
        self.assertEqual(len(cases), 8)
        self.assertTrue(ae.mapping_passed(cases))
        self.assertFalse(ae.mapping_passed(cases[:-1]))
        for i in range(8):
            for margin in (0, -cases[i]["semantic_margin"], float("nan")):
                changed = [dict(case) for case in cases]
                changed[i]["semantic_margin"] = margin
                self.assertFalse(ae.mapping_passed(changed))
        self.assertFalse(ae.mapping_passed(cases[::-1]))

    def test_duplicate_incomplete_and_misaligned_grid_rejected(self):
        rows = rows_fixture(2)
        records = list(grid_fixture(rows, ("A_present",), ("Effusion",)))
        with patch.object(ae, "EVAL_ROWS", 2):
            valid = ae.validate_grid(records[::-1], rows, ("A_present",), ("Effusion",))
            self.assertEqual(valid.shape, (1, 1, 55, 2, 3))
            for invalid in (records + records[:1], records[:-1]):
                with self.assertRaises(ValueError):
                    ae.validate_grid(invalid, rows, ("A_present",), ("Effusion",))
            for key, value in (("patient_id", "wrong"), ("label", 7), ("encoding", "standard"),
                               ("direction", "random20"), ("alpha", 0.1), ("raw_margin", float("inf")),
                               ("semantic_margin", 14), ("semantic_probability", 0.99)):
                invalid = [dict(r) for r in records]
                invalid[0][key] = value
                with self.subTest(field=key), self.assertRaises(ValueError):
                    ae.validate_grid(invalid, rows, ("A_present",), ("Effusion",))

    def test_probability_tolerance_accepts_float32_rounding_near_one(self):
        rows = rows_fixture(2)
        records = [r for r in grid_fixture(rows, ("A_present",), ("Effusion",)) if r["alpha"] == 0]
        probability = np.float32(expit(8.0))
        for _ in range(3):
            probability = np.nextafter(probability, np.float32(1.0))
        self.assertGreater(abs(float(probability) - expit(8.0)), 1e-7)
        records[0].update(raw_margin=8.0, semantic_margin=8.0, semantic_probability=float(probability))
        with patch.object(ae, "EVAL_ROWS", 2):
            ae.validate_grid(records, rows, ("A_present",), ("Effusion",), ae.CONFIGURATIONS[:1])
            records[0]["semantic_probability"] = 0.99
            with self.assertRaisesRegex(ValueError, "invalid finite score"):
                ae.validate_grid(records, rows, ("A_present",), ("Effusion",), ae.CONFIGURATIONS[:1])

    def test_weighted_auroc_matches_explicit_resampling_with_ties(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0., 0.5, 0.5, 1.])
        weights = np.array([[1, 1, 1, 1], [2, 0, 1, 1], [4, 0, 0, 0]])
        actual = ae.weighted_auroc(labels, scores, weights)
        for i in (0, 1):
            indices = np.repeat(np.arange(4), weights[i])
            self.assertAlmostEqual(actual[i], roc_auc_score(labels[indices], scores[indices]))
        self.assertTrue(np.isnan(actual[2]))
        with patch.object(ae, "BOOTSTRAP_RESAMPLES", 8):
            first, second = ae.bootstrap_weights(20), ae.bootstrap_weights(20)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first.sum(axis=1), 20)

    def test_energy_uses_mean_matrix_and_eligible_columns(self):
        semantic = np.ones((len(ae.DIRECTIONS), 6, 4))
        semantic[:, 1] = np.array([-3, -1, 1, 3])
        token = np.zeros_like(semantic)
        weights = np.ones((5, 4))
        summary = ae.energy_summary(semantic, token, weights, [0])
        self.assertEqual(summary["primary"]["estimate"], -1)
        self.assertAlmostEqual(summary["all_questions_descriptive"]["estimate"], -5 + 5 / 3)
        self.assertEqual(summary["naive_all_questions_descriptive"]["estimate"], -5)
        self.assertEqual(summary["eligible_controls"]["random0"]["estimate"], -1)
        empty = ae.energy_summary(semantic, token, weights, [])
        self.assertEqual(empty["primary"]["status"], "inconclusive")
        self.assertIsNone(empty["primary"]["estimate"])
        self.assertIsNone(empty["primary"]["descriptive_ci95"])

    def test_unbiased_energy_matches_cross_patient_products_and_resampled_multiset(self):
        values = np.array([1., 2., 5., 8.])[None, None, :]
        weights = np.array([[1., 1., 1., 1.], [2., 0., 1., 1.]])
        actual = ae.unbiased_squared_mean(values, weights)[:, 0, 0]
        for b in range(2):
            sample = np.repeat(values.ravel(), weights[b].astype(int))
            expected = sum(sample[i] * sample[j] for i in range(4) for j in range(4) if i != j) / 12
            self.assertAlmostEqual(actual[b], expected)
        a, b = values + 2, values * -3
        s, t = ae.crossover(a, b)
        actual = ae.unbiased_squared_mean(t, weights) - ae.unbiased_squared_mean(s, weights)
        expected = (a.mean() * b.mean() - (a * b).sum() / 16) * 4 / 3
        self.assertAlmostEqual(actual[0, 0, 0], expected)

    def test_energy_normalization_controls_and_null(self):
        token = np.ones((27, 6, 4))
        semantic = np.zeros_like(token)
        weights = np.ones((5, 4))
        result = ae.energy_summary(semantic, token, weights, [0, 2])
        self.assertEqual(result["primary"]["estimate"], 2)
        self.assertTrue(all(v["estimate"] == 2 for v in result["eligible_controls"].values()))
        self.assertFalse(result["primary"]["encoding_even_beyond_controls"])
        token[6:] *= 0.1
        result = ae.energy_summary(semantic, token, weights, [0, 2])
        self.assertTrue(result["primary"]["encoding_even_beyond_controls"])
        result = ae.energy_summary(semantic, semantic, weights, [0, 2])
        self.assertEqual(result["primary"]["estimate"], 0)
        self.assertFalse(result["primary"]["encoding_even_beyond_controls"])

    def test_full_synthetic_summary_eligibility_orientation_and_determinism(self):
        with TemporaryDirectory() as directory, patch.object(ae, "EVAL_ROWS", 20):
            root = Path(directory)
            rows = rows_fixture()
            write_fixture(root, rows)
            result = ae.summarize(root / "per-image.csv", root / "summary.json")
            first = (root / "summary.json").read_bytes()
            ae.summarize(root / "per-image.csv", root / "summary.json")
            self.assertEqual(first, (root / "summary.json").read_bytes())
            self.assertEqual(result["n_outcomes"], 20 * 3 * 6 * 55)
            self.assertTrue(all(v["status"] == "eligible" for v in result["eligibility"].values()))
            energy = result["crossover_plus_025"]
            expected = np.array([[0.25 * ((d + 1) * 0.1 + q * 0.01) for q in range(6)] for d in range(6)])
            np.testing.assert_allclose(energy["semantic_matrix"], expected)
            np.testing.assert_allclose(energy["token_matrix"], 0, atol=1e-12)
            self.assertAlmostEqual(energy["primary"]["estimate"], -np.square(expected).sum() / 6)
            self.assertEqual(energy["primary"]["eligible_question_count"], 6)
            for encoding in ae.ENCODINGS:
                entry = result["encodings"][encoding]
                self.assertEqual(entry["baseline"]["Effusion"]["auroc"], 1)
                plus = entry["doses"]["0.25"]
                np.testing.assert_allclose(plus["semantic_margin_matrix"], expected)
                sign = -1 if encoding == "B_present" else 1
                np.testing.assert_allclose(plus["raw_margin_matrix"], sign * expected)
                self.assertEqual(len(plus["controls"]), 21)

    def test_summary_reports_ineligible_questions_and_omits_constant_draws(self):
        with TemporaryDirectory() as directory, patch.object(ae, "EVAL_ROWS", 20):
            root = Path(directory)
            rows = rows_fixture()
            for row in rows:
                row["Nodule"] = "0"
            write_fixture(root, rows)
            result = ae.summarize(root / "per-image.csv", root / "summary.json")
            self.assertEqual(result["eligibility"]["Nodule"]["status"], "insufficient_labels")
            self.assertEqual(result["eligibility"]["Nodule"]["valid_bootstrap_count"]["A_present"], 0)
            self.assertEqual(result["crossover_plus_025"]["primary"]["eligible_question_count"], 5)
            preflight = passed_preflight()
            preflight["mapping_cases"][0]["semantic_margin"] = -1
            ae.write_json(root / "preflight.json", preflight)
            result = ae.summarize(root / "per-image.csv", root / "summary.json")
            self.assertTrue(all(v["status"] == "mapping_failed" for v in result["eligibility"].values()))
            self.assertEqual(result["crossover_plus_025"]["primary"]["status"], "inconclusive")
            self.assertEqual(len(result["encodings"]), 3)

    def test_calibration_alone_selects_eligibility_and_patient_overlap_is_rejected(self):
        with TemporaryDirectory() as directory, patch.object(ae, "EVAL_ROWS", 20):
            root = Path(directory)
            effect = rows_fixture()
            calibration = [{**r, "row_id": "cal-" + r["row_id"], "patient_id": "cal-" + r["patient_id"]}
                           for r in rows_fixture()]
            for row in effect:
                row["Effusion"] = "0"
            write_fixture(root, effect, calibration)
            result = ae.summarize(root / "per-image.csv", root / "summary.json")
            self.assertEqual(result["eligibility"]["Effusion"]["status"], "eligible")
            self.assertIsNone(result["encodings"]["A_present"]["baseline"]["Effusion"]["auroc"])
            self.assertEqual(result["eligibility"]["Effusion"]["bootstrap_seed"], 20260908)
            self.assertEqual(result["n_scientific_outcomes"], 20 * (18 * 55 + 12))
            meta = ae.read_json(root / "meta.json")
            meta["calibration_rows"][0]["patient_id"] = effect[0]["patient_id"]
            ae.write_json(root / "meta.json", meta)
            with self.assertRaisesRegex(ValueError, "patient-disjoint"):
                ae.summarize(root / "per-image.csv", root / "summary.json")

    def test_calibration_grid_requires_clean_unique_complete_outcomes(self):
        rows = rows_fixture(2)
        records = [r for r in grid_fixture(rows, ae.ENCODINGS[1:]) if r["alpha"] == 0]
        with patch.object(ae, "EVAL_ROWS", 2):
            result = ae.validate_grid(records, rows, ae.ENCODINGS[1:], conditions=ae.CONFIGURATIONS[:1])
            self.assertEqual(result.shape, (2, 6, 1, 2, 3))
            for invalid in (records[:-1], records + records[:1], [{**records[0], "alpha": .25}, *records[1:]]):
                with self.assertRaises(ValueError):
                    ae.validate_grid(invalid, rows, ae.ENCODINGS[1:], conditions=ae.CONFIGURATIONS[:1])

    def test_summary_cannot_emit_primary_with_any_missing_block(self):
        with TemporaryDirectory() as directory, patch.object(ae, "EVAL_ROWS", 2):
            root = Path(directory)
            rows = rows_fixture(2)
            for name in ("calibration.csv", "per-image.csv"):
                write_fixture(root, rows)
                path = root / name
                with path.open(newline="", encoding="utf-8") as stream:
                    records = list(csv.DictReader(stream))
                first_encoding = records[0]["encoding"]
                records = [r for r in records if (r["encoding"], r["question"]) != (first_encoding, "Effusion")]
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=ae.FIELDS)
                    writer.writeheader()
                    writer.writerows(records)
                with self.subTest(file=name), self.assertRaisesRegex(ValueError, "incomplete"):
                    ae.summarize(root / "per-image.csv", root / "summary.json")
                self.assertFalse((root / "summary.json").exists())

    def test_source_uses_first_and_last_200_accepted_disjoint_patients(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / ae.OWNERSHIP_RUN_ID
            root.mkdir()
            (root / "metadata.env").write_text(
                f"RUN_ID={ae.OWNERSHIP_RUN_ID}\nSOURCE_COMMIT={ae.OWNERSHIP_COMMIT}\n"
                "COMMAND_STATUS=0\nDISPATCHER_STATUS=0\nCLEANUP_STATUS=0\nABORT_SIGNAL=none\n")
            for name in ("command_exit_status", "exit_status"):
                (root / name).write_text("0\n")
            ae.write_json(root / "artifacts/source-reference.json", {"source_run_id": ae.SOURCE_RUN_ID,
                "source_commit": ae.SOURCE_COMMIT, "model_id": ae.MODEL_ID, "model_revision": ae.MODEL_REVISION})
            rows = rows_fixture(400)
            ae.write_json(root / "artifacts/registered-rows.json", {"row_ids": [r["row_id"] for r in rows],
                "patient_ids": [r["patient_id"] for r in rows], "row_ids_sha256": ae.REGISTERED_ROWS_SHA256})
            ae.write_json(root / "artifacts/causal-ownership-summary.json", {"intervention_commit": ae.OWNERSHIP_COMMIT, "n_eval": 400})
            validation = [{**r, "row_id": "val-" + r["row_id"], "patient_id": "val-" + r["patient_id"]}
                          for r in rows_fixture(16)]
            with (root / "manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[*rows[0], "split"])
                writer.writeheader()
                writer.writerows({**r, "split": "test"} for r in rows)
                writer.writerows({**r, "split": "validation"} for r in validation)
            _, effects, calibration, pilot, _ = ae.load_source(root, root / "manifest.csv")
            self.assertEqual([r["row_id"] for r in effects], [r["row_id"] for r in rows[:200]])
            self.assertEqual([r["row_id"] for r in calibration], [r["row_id"] for r in rows[200:]])
            self.assertEqual(len(pilot), 16)
            (root / "metadata.env").write_text("SOURCE_COMMIT=wrong\n")
            with self.assertRaisesRegex(ValueError, "terminal metadata"):
                ae.load_source(root, root / "manifest.csv")

    def test_runner_preflight_failure_and_completed_checkpoint_reuse(self):
        import torch
        from PIL import Image

        from src import intervene

        class Inputs(dict):
            def to(self, device):
                return self

        class Processor:
            tokenizer = Tokenizer({"A": [0], "B": [1], "yes": [0], "no": [1]})

            def __init__(self):
                self.build_sizes = []

            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"][-1]["text"]

            def __call__(self, text, **kwargs):
                self.build_sizes.append(len(text))
                return Inputs(prompt=text[0], size=len(text))

        class Model:
            def __init__(self, fail_after=None, wrong_mapping=False):
                self.block = torch.nn.Identity()
                self.calls, self.fail_after, self.wrong_mapping = 0, fail_after, wrong_mapping

            def eval(self):
                return self

            def named_modules(self):
                return [(ae.EXPECTED_LOCUS, self.block)]

            def __call__(self, prompt, size):
                self.calls += 1
                if self.fail_after and self.calls > self.fail_after:
                    raise RuntimeError("synthetic interrupted batch")
                h = self.block(torch.ones(size, 3, 4))
                if prompt.startswith("The finding"):
                    present = "absent" not in prompt.split(" Is the finding")[0]
                    sign = (1 if present else -1) * (1 if "Answer A" in prompt else -1)
                    if self.wrong_mapping:
                        sign *= -1
                else:
                    sign = 1
                value = sign * h.mean(dim=(1, 2))
                return SimpleNamespace(logits=torch.stack((value, -value), dim=-1)[:, None, :])

        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            rows = rows_fixture(2)
            Image.new("RGB", (336, 336)).save(root / "image.png")
            for row in rows:
                row["image_path"] = str(root / "image.png")
            validation = [{**rows[0], "row_id": f"validation{i}", "patient_id": f"validation{i}"} for i in range(16)]
            calibration_rows = [{**r, "row_id": "cal-" + r["row_id"], "patient_id": "cal-" + r["patient_id"]} for r in rows]
            np.savez(root / "directions.npz", raw_dim=4)
            source = {"model_source": "accepted-model", "directions": str(root / "directions.npz"),
                      "directions_sha256": "accepted-digest"}
            clinical = {q: np.ones(4, dtype=np.float32) / 2 for q in ae.CONCEPTS}
            stack.enter_context(patch.object(ae, "EVAL_ROWS", 2))
            stack.enter_context(patch.object(ae, "load_source", return_value=(source, rows, calibration_rows, validation, {})))
            stack.enter_context(patch.object(intervene, "load_registered_directions", return_value=(clinical, "accepted-digest")))
            stack.enter_context(patch("src.gpu_env.bind_gpu"))
            processor = Processor()
            factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor)
            model = Model(wrong_mapping=True)
            model_factory = unittest.mock.Mock(return_value=model)
            stack.enter_context(patch.dict(sys.modules, {"transformers": SimpleNamespace(
                AutoProcessor=factory, AutoModelForImageTextToText=SimpleNamespace(from_pretrained=model_factory))}))
            self.assertFalse(ae.run(root, root, 0, root / "failed"))
            self.assertEqual(model.calls, 11)
            self.assertFalse((root / "failed/per-image.csv").exists())
            self.assertFalse(ae.read_json(root / "failed/preflight.json")["passed"])
            self.assertEqual(len(model.block._forward_hooks), 0)
            model = Model(fail_after=146)
            model_factory.return_value = model
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                ae.run(root, root, 0, root / "resumed")
            checkpoint = root / "resumed/effusion-standard.csv"
            completed_bytes = checkpoint.read_bytes()
            self.assertEqual(len(model.block._forward_hooks), 0)
            model = Model()
            model_factory.reset_mock()
            model_factory.return_value = model
            self.assertTrue(ae.run(root, root, 0, root / "resumed"))
            model_factory.assert_called_once()
            self.assertEqual(model.calls, 75 + 17 * 55)
            self.assertEqual(checkpoint.read_bytes(), completed_bytes)
            with (root / "resumed/per-image.csv").open(newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
                result = ae.validate_grid(records, rows)
            self.assertEqual(result.shape, (3, 6, 55, 2, 3))
            self.assertEqual([(r["encoding"], r["question"], r["direction"], float(r["alpha"])) for r in records[::2]],
                [(e, q, d, a) for e in ae.ENCODINGS for q in ae.CONCEPTS for d, a in ae.CONFIGURATIONS])
            with (root / "resumed/calibration.csv").open(newline="", encoding="utf-8") as stream:
                calibration_records = list(csv.DictReader(stream))
            self.assertEqual([(r["encoding"], r["question"]) for r in calibration_records[::2]],
                             [(e, q) for e in ae.ENCODINGS[1:] for q in ae.CONCEPTS])
            model = Model()
            model_factory.return_value = model
            processor.build_sizes.clear()
            with patch.object(ae, "score_logits", wraps=ae.score_logits) as scoring:
                self.assertTrue(ae.run(root, root, 0, root / "preflight-only", preflight_only=True))
                self.assertEqual(sum(call.args[0].shape[0] == 16 for call in scoring.call_args_list), 64)
            self.assertEqual(processor.build_sizes.count(16), 2)
            self.assertEqual(model.calls, 75)
            self.assertFalse((root / "preflight-only/per-image.csv").exists())
            model = Model()
            model_factory.return_value = model
            processor.build_sizes.clear()
            with patch.object(ae.time, "perf_counter", side_effect=[0., 100.]), patch("builtins.print"):
                self.assertFalse(ae.run(root, root, 0, root / "slow"))
            self.assertEqual(processor.build_sizes.count(16), 2)
            self.assertEqual(model.calls, 75)
            self.assertFalse((root / "slow/calibration.csv").exists())
            self.assertFalse((root / "slow/per-image.csv").exists())
            preflight = ae.read_json(root / "slow/preflight.json")
            self.assertEqual(preflight["throughput"]["image_condition_equivalents"], 1024)
            self.assertEqual(preflight["throughput"]["predicted_scientific_seconds"], 100 * 200400 / 1024)


if __name__ == "__main__":
    unittest.main()
