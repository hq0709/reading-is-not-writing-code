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

from src import qwen_answer_encoding as ae
from src import qwen_mass_prompt_specificity as mass
from src.qwen_causal_ownership_gate import _simultaneous_lower_bounds


def rows_fixture(n=200, prefix=""):
    return [dict(row_id=f"{prefix}r{i:04}", patient_id=f"{prefix}p{i:04}",
                 split="test", Mass=str(i % 2)) for i in range(n)]


def records_fixture(rows, cells=mass.CELLS, conditions=mass.CONFIGURATIONS):
    for cell in cells:
        for direction, alpha in conditions:
            for row in rows:
                semantic = 2 * int(row["Mass"]) - 1 + alpha
                yield dict(row_id=row["row_id"], patient_id=row["patient_id"],
                           cell=cell, direction=direction, alpha=alpha, locus=ae.LOCUS,
                           label=row["Mass"], raw_margin=semantic * mass.orientation(cell),
                           semantic_margin=semantic, semantic_probability=float(expit(semantic)))


class MassPromptTests(unittest.TestCase):
    def test_cohort_metadata_only_deterministic_and_patient_disjoint(self):
        rows = rows_fixture(450)
        rows += [{**r, "row_id": r["row_id"] + "b"} for r in rows[:300]]
        excluded = [r["row_id"] for r in rows[:40]]
        selected, receipt = mass.select_rows(rows, excluded)
        shuffled = list(reversed([{**r, "Mass": str(1 - int(r["Mass"]))} for r in rows]))
        repeated, _ = mass.select_rows(shuffled, excluded)
        self.assertEqual([r["row_id"] for r in selected], [r["row_id"] for r in repeated])
        self.assertEqual(len(selected), 200)
        self.assertEqual(len({r["patient_id"] for r in selected}), 200)
        self.assertFalse({r["patient_id"] for r in selected} & {r["patient_id"] for r in rows[:40]})
        self.assertEqual(receipt["eligible_patients"], 410)
        rng = np.random.default_rng(20260909)
        patients = rng.permutation(sorted({r["patient_id"] for r in rows[40:450]}))[:200]
        expected = []
        for patient in patients:
            candidates = sorted(r["row_id"] for r in rows if r["patient_id"] == patient)
            expected.append(candidates[rng.integers(len(candidates))])
        self.assertEqual([r["row_id"] for r in selected], expected)

    def test_cohort_shortage_and_missing_exclusion_block(self):
        with self.assertRaisesRegex(ValueError, "200"):
            mass.select_rows(rows_fixture(200), ["r0000"])
        with self.assertRaisesRegex(ValueError, "exclusion"):
            mass.select_rows(rows_fixture(210), ["missing"])
        with self.assertRaisesRegex(ValueError, "unique"):
            mass.select_rows(rows_fixture(210) + rows_fixture(1), [])

    def test_exact_prompts_counts_and_mapping_signs(self):
        self.assertEqual(mass.CELLS, ("show/yes_no", "show/A_present", "show/B_present",
                                     "is/yes_no", "is/A_present", "is/B_present", "anchor"))
        for wording, question in (("show", "Does this chest radiograph show a lung mass?"),
                                  ("is", "Is there a lung mass in this chest radiograph?")):
            for encoding, instruction in (
                ("yes_no", "Answer yes if the finding is present and no if it is absent. Reply with yes or no only."),
                ("A_present", "Answer A if the finding is present and B if it is absent. Reply with A or B only."),
                ("B_present", "Answer B if the finding is present and A if it is absent. Reply with A or B only."),
            ):
                self.assertEqual(mass.prompt_for(f"{wording}/{encoding}"), question + " " + instruction)
        self.assertEqual(mass.prompt_for("anchor"), ae.PROMPTS["Mass"])
        for encoding in ("A_present", "B_present"):
            self.assertEqual(mass.prompt_for("show/" + encoding), ae.prompt_for("Mass", encoding))
        self.assertEqual(len(mass.CONFIGURATIONS), 127)
        self.assertEqual(mass.SCIENTIFIC_OUTCOMES, 179200)
        self.assertEqual(mass.N_RANDOM, 119)
        cases = [{**c, "semantic_margin": 2 * c["expected_present"] - 1} for c in mass.mapping_cases()]
        self.assertEqual(len(cases), 12)
        self.assertTrue(mass.mapping_passed(cases))
        cases[-1]["semantic_margin"] *= -1
        self.assertFalse(mass.mapping_passed(cases))
        self.assertFalse(mass.mapping_passed(cases[:-1]))

    def test_directions_replay_119_draws_then_mass_sham(self):
        clinical = {q: np.arange(1, 17, dtype=np.float32) + i for i, q in enumerate(ae.CONCEPTS)}
        directions = mass.common_directions(clinical)
        self.assertEqual(tuple(directions), mass.DIRECTIONS)
        rng = np.random.default_rng(0)
        for i in range(119):
            vector = rng.standard_normal(16).astype(np.float32)
            np.testing.assert_array_equal(directions[f"random{i}"], vector / np.linalg.norm(vector))
        vector = clinical["Mass"] / np.linalg.norm(clinical["Mass"])
        vector = vector[rng.permutation(16)]
        np.testing.assert_array_equal(directions["sham"], vector / np.linalg.norm(vector))

    def test_complete_grid_and_calibration_failures(self):
        rows = rows_fixture()
        grid = mass.validate_grid(records_fixture(rows), rows)
        self.assertEqual(grid.shape, (7, 127, 200, 3))
        records = list(records_fixture(rows, conditions=mass.CONFIGURATIONS[:1]))
        self.assertEqual(mass.validate_grid(records, rows, conditions=mass.CONFIGURATIONS[:1]).shape,
                         (7, 1, 200, 3))
        for invalid in (records[:-1], records + records[:1],
                        [{**records[0], "patient_id": "wrong"}, *records[1:]],
                        [{**records[0], "semantic_margin": 8}, *records[1:]],
                        [{**records[0], "alpha": .25}, *records[1:]],
                        [{**records[0], "label": "1"}, *records[1:]]):
            with self.subTest(first=invalid[0]), self.assertRaises(ValueError):
                mass.validate_grid(invalid, rows, conditions=mass.CONFIGURATIONS[:1])
        with self.assertRaisesRegex(ValueError, "200"):
            mass.validate_grid([], rows[:-1])

    def test_eligibility_uses_calibration_and_registered_thresholds(self):
        labels = np.arange(200) % 2
        semantic = labels * 2 - 1
        scores = np.stack((semantic, semantic, expit(semantic)), axis=-1)
        weights = ae.bootstrap_weights(200, mass.CALIBRATION_SEED)
        result = mass.calibration_metrics(labels, scores, weights)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["auroc"], 1)
        self.assertEqual(result["positive_patients"], 100)
        for y, s, w, status in (
            (np.zeros(200), scores, weights, "insufficient_labels"),
            (labels, scores, weights[:1899], "insufficient_valid_bootstraps"),
            (labels, -scores, weights, "baseline_ineligible"),
        ):
            self.assertEqual(mass.calibration_metrics(y, s, w)["status"], status)

    def test_shared_family_two_flags_and_all_controls(self):
        rng = np.random.default_rng(4)
        effects = rng.normal(0, .02, (7, 126, 200))
        effects[:, mass.DIRECTIONS.index("Mass")] += 1
        weights = mass.bootstrap_weights(200)
        eligible = {c: True for c in mass.CELLS}
        result = mass.effect_summary(effects, weights, eligible)
        self.assertTrue(result["cross_wording_specificity"])
        self.assertTrue(result["discovery_wording_replication"])
        self.assertEqual(len(result["clinical_family"]), 20)
        self.assertEqual(result["bootstrap"]["resamples"], 5000)
        pairs = [(mass.CELLS.index(c), d) for c in mass.CODED_CELLS
                 for d, name in enumerate(ae.CONCEPTS) if name != "Mass"]
        differences = np.stack([effects[c, 4] - effects[c, d] for c, d in pairs])
        lower, critical = _simultaneous_lower_bounds(differences.mean(axis=1), weights @ differences.T / 200)
        np.testing.assert_allclose([p["simultaneous_lower95"] for p in result["clinical_family"]], lower)
        self.assertAlmostEqual(result["max_t_critical"], critical)
        eligible["is/B_present"] = False
        narrow = mass.effect_summary(effects, weights, eligible)
        self.assertFalse(narrow["cross_wording_specificity"])
        self.assertTrue(narrow["discovery_wording_replication"])
        self.assertEqual(result["clinical_family"], narrow["clinical_family"])
        effects[1, mass.DIRECTIONS.index("random118")] = 2
        failed = mass.effect_summary(effects, weights, eligible)
        self.assertFalse(failed["discovery_wording_replication"])
        effects[1, mass.DIRECTIONS.index("random118")] = 0
        effects[1, -1] = -2
        self.assertFalse(mass.effect_summary(effects, weights, eligible)["discovery_wording_replication"])

    def test_paired_margins_recompute_competitor_per_bootstrap(self):
        effects = np.zeros((7, 126, 4))
        effects[:, mass.DIRECTIONS.index("Mass")] = 3
        effects[:, 0] = [2, 2, 0, 0]
        effects[:, 1] = [0, 0, 2, 2]
        effects[1, mass.DIRECTIONS.index("Mass")] += .5
        weights = np.array([[4, 0, 0, 0], [0, 0, 4, 0], [1, 1, 1, 1]], dtype=float)
        result = mass.effect_summary(effects, weights, {c: True for c in mass.CELLS})
        self.assertEqual(result["cells"]["show/yes_no"]["clinical_margin"]["estimate"], 2)
        self.assertEqual(result["cells"]["show/yes_no"]["clinical_margin"]["descriptive_ci95"][0], 1)
        key = "show/A_present - show/yes_no"
        self.assertEqual(result["paired_margin_comparisons"][key]["estimate"], .5)
        self.assertEqual(result["paired_margin_comparisons"][key]["descriptive_ci95"], [.5, .5])
        self.assertEqual(len(result["paired_margin_comparisons"]), 8)

    def test_source_excludes_repeated_initial_patients_and_closure_pairs(self):
        with TemporaryDirectory() as directory:
            runs = Path(directory)
            rows = [{**r, **{q: r["Mass"] for q in ae.CONCEPTS}} for r in rows_fixture(1400)]
            for i in range(167, 200):
                rows[i]["patient_id"] = rows[i - 167]["patient_id"]
            for i in range(1050, 1100):
                rows[i]["patient_id"] = rows[i - 50]["patient_id"]
            ownership = runs / ae.OWNERSHIP_RUN_ID
            ownership.mkdir()
            (ownership / "metadata.env").write_text(
                f"RUN_ID={ae.OWNERSHIP_RUN_ID}\nSOURCE_COMMIT={ae.OWNERSHIP_COMMIT}\n"
                "COMMAND_STATUS=0\nDISPATCHER_STATUS=0\nCLEANUP_STATUS=0\nABORT_SIGNAL=none\n")
            for name in ("command_exit_status", "exit_status"):
                (ownership / name).write_text("0\n")
            ae.write_json(ownership / "artifacts/source-reference.json", {
                "source_run_id": ae.SOURCE_RUN_ID, "source_commit": ae.SOURCE_COMMIT,
                "model_id": ae.MODEL_ID, "model_revision": ae.MODEL_REVISION})
            ae.write_json(ownership / "artifacts/causal-ownership-summary.json", {
                "intervention_commit": ae.OWNERSHIP_COMMIT, "n_eval": 400})
            ae.write_json(ownership / "artifacts/registered-rows.json", {
                "row_ids": [r["row_id"] for r in rows[600:1000]],
                "patient_ids": [r["patient_id"] for r in rows[600:1000]],
                "row_ids_sha256": ae.REGISTERED_ROWS_SHA256})
            ae.write_json(runs / ae.SOURCE_RUN_ID / "artifacts/intervention-summary.json", {
                "eval_row_ids": [r["row_id"] for r in rows[:200]]})
            ae.write_json(runs / mass.PREVIOUS_RUN_ID / "artifacts/registered-rows.json", {
                "row_ids": [r["row_id"] for r in rows[200:600]]})
            ae.write_json(runs / mass.CLOSURE_RUN_ID / "artifacts/registered-pairs.json", {
                "pairs": [{"patient_id": rows[i]["patient_id"], "positive_row_id": rows[i]["row_id"],
                           "negative_row_id": rows[i + 50]["row_id"]} for i in range(1000, 1050)]})
            ae.write_json(runs / mass.ENCODING_RUN_ID / "artifacts/meta.json", {
                "rows": rows[600:800], "calibration_rows": rows[800:1000]})
            validation = [{**r, "split": "validation", **{q: r["Mass"] for q in ae.CONCEPTS}}
                          for r in rows_fixture(16, "val")]
            manifest = runs / "manifest.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows + validation)
            _, selected, calibration, _, receipt = mass.load_source(ownership, manifest)
            self.assertEqual(receipt["excluded_patients"], 1017)
            self.assertEqual(receipt["eligible_patients"], 300)
            self.assertEqual(calibration, rows[800:1000])
            self.assertFalse({r["patient_id"] for r in selected} & {r["patient_id"] for r in rows[:1100]})
            with patch("src.gpu_env.bind_gpu", side_effect=AssertionError("cohort-only bound GPU")):
                self.assertTrue(mass.run(ownership, manifest, None, runs / "out", cohort_only=True))
            self.assertEqual(ae.read_json(runs / "out/registered-rows.json"), receipt)
            receipt_path = runs / mass.ENCODING_RUN_ID / "artifacts/meta.json"
            corrupted = ae.read_json(receipt_path)
            corrupted["calibration_rows"].reverse()
            ae.write_json(receipt_path, corrupted)
            with self.assertRaisesRegex(ValueError, "calibration"):
                mass.load_source(ownership, manifest)

    def test_import_and_cohort_cli_are_cpu_only(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; from src import qwen_mass_prompt_specificity; assert 'torch' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", "src/qwen_mass_prompt_specificity.py", "cohort", "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--gpu", result.stdout)

    def test_preflight_mapping_and_throughput_block_before_test_images(self):
        import torch
        from PIL import Image
        from src import intervene

        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"A": [0], "B": [1], "yes": [0], "no": [1]}.get(text, [90, 91])

        class Processor:
            tokenizer = Tokenizer()

            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"][-1]["text"]

            def __call__(self, text, **kwargs):
                return Inputs(prompt=text[0], size=len(text))

        class Model:
            def __init__(self, wrong=False):
                self.block = torch.nn.Identity()
                self.calls, self.wrong = 0, wrong

            def eval(self):
                return self

            def named_modules(self):
                return [(ae.EXPECTED_LOCUS, self.block)]

            def __call__(self, prompt, size):
                self.calls += 1
                hidden = self.block(torch.ones(size, 3, 4))
                sign = 1
                if prompt.startswith("The finding"):
                    sign = (1 if "absent" not in prompt.split(" Is the finding")[0] else -1)
                    sign *= -1 if "Answer B" in prompt else 1
                    sign *= -1 if self.wrong else 1
                value = sign * hidden.mean(dim=(1, 2))
                return SimpleNamespace(logits=torch.stack((value, -value), dim=-1)[:, None, :])

        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            Image.new("RGB", (16, 16)).save(root / "image.png")
            rows, calibration = rows_fixture(), rows_fixture(prefix="cal")
            validation = [{**r, "image_path": str(root / "image.png")} for r in rows_fixture(16, "val")]
            np.savez(root / "directions.npz", raw_dim=4)
            source = {"model_source": "accepted-model", "directions": str(root / "directions.npz"),
                      "directions_sha256": "accepted-receipt"}
            receipt = {"row_ids": [r["row_id"] for r in rows], "patient_ids": [r["patient_id"] for r in rows],
                       "calibration_row_ids": [r["row_id"] for r in calibration],
                       "calibration_patient_ids": [r["patient_id"] for r in calibration]}
            stack.enter_context(patch.object(mass, "load_source", return_value=(source, rows, calibration, validation, receipt)))
            stack.enter_context(patch("src.gpu_env.bind_gpu"))
            clinical = {q: np.ones(4, dtype=np.float32) / 2 for q in ae.CONCEPTS}
            stack.enter_context(patch.object(intervene, "load_registered_directions", return_value=(clinical, "accepted-receipt")))
            model = Model(wrong=True)
            model_factory = unittest.mock.Mock(return_value=model)
            stack.enter_context(patch.dict(sys.modules, {"transformers": SimpleNamespace(
                AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: Processor()),
                AutoModelForImageTextToText=SimpleNamespace(from_pretrained=model_factory))}))
            self.assertFalse(mass.run(root, root, 0, root / "mapping-failed"))
            self.assertEqual(model.calls, 15)
            self.assertFalse((root / "mapping-failed/calibration.csv").exists())
            model = Model()
            model_factory.return_value = model
            with patch.object(mass.time, "perf_counter", side_effect=[0., 50.]):
                self.assertFalse(mass.run(root, root, 0, root / "slow"))
            self.assertEqual(model.calls, 79)
            self.assertFalse((root / "slow/calibration.csv").exists())
            model = Model()
            model_factory.return_value = model
            with patch.object(mass.time, "perf_counter", side_effect=[0., 10.]):
                self.assertTrue(mass.run(root, root, 0, root / "preflight", preflight_only=True))
            self.assertEqual(model.calls, 79)
            preflight = ae.read_json(root / "preflight/preflight.json")
            self.assertTrue(mass.measurement_passed(preflight))
            self.assertEqual(preflight["throughput"]["predicted_scientific_seconds"], 1750)
            self.assertFalse((root / "preflight/per-image.csv").exists())
            self.assertEqual(len(model.block._forward_hooks), 0)
            for cell in mass.CELLS:
                for prefix, cohort, conditions in (("", rows, mass.CONFIGURATIONS),
                        ("calibration-", calibration, mass.CONFIGURATIONS[:1])):
                    checkpoint = root / "preflight" / f"{prefix}{cell.replace('/', '-')}.csv"
                    with checkpoint.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=mass.FIELDS)
                        writer.writeheader()
                        writer.writerows(records_fixture(cohort, (cell,), conditions))
            model = Model()
            model_factory.return_value = model
            with patch.object(mass.time, "perf_counter", side_effect=[0., 10.]), patch("builtins.print"):
                self.assertTrue(mass.run(root, root, 0, root / "preflight"))
            self.assertEqual(model.calls, 79)
            with patch.object(mass, "bootstrap_weights", return_value=mass.bootstrap_weights(200)[:30]), patch("builtins.print"):
                result = mass.summarize(root / "preflight/per-image.csv", root / "summary.json")
            self.assertEqual(result["n_scientific_outcomes"], 179200)
            self.assertEqual(result["n_outcomes"], 177800)
            self.assertEqual(result["n_calibration_outcomes"], 1400)
            self.assertFalse(result["cross_wording_specificity"])
            self.assertTrue(all(v["eligible"] for v in result["calibration"].values()))
            self.assertEqual(len(result["calibration_spearman"]), 21)
            self.assertEqual(len(result["cells"]["anchor"]["metric_changes"]), 126)
            self.assertEqual(result["cells"]["anchor"]["metric_changes"]["Mass"]["auroc_change"]["estimate"], 0)
            with (root / "preflight/calibration.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=mass.FIELDS)
                writer.writeheader()
                writer.writerows(records_fixture(calibration, mass.CELLS[:-1], mass.CONFIGURATIONS[:1]))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                mass.summarize(root / "preflight/per-image.csv", root / "incomplete-summary.json")
            self.assertFalse((root / "incomplete-summary.json").exists())
            preflight["mapping_cases"][-1]["semantic_margin"] *= -1
            self.assertFalse(mass.measurement_passed(preflight))


if __name__ == "__main__":
    unittest.main()
