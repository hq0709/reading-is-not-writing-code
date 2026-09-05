from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts.audit_nih_paired_pool import RAW_FIELDS
from scripts.audit_paired_opportunity import COHORTS
from src import qwen_paired_opportunity as core
from src import run_qwen_paired_opportunity as runner
from src.build_manifest import patient_split
from test_qwen_paired_opportunity import fixture, row


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def terminal(root, cross=True, discovery=True):
    run = root / runner.MASS_RUN_ID
    run.mkdir(parents=True, exist_ok=True)
    metadata = {"RUN_ID": runner.MASS_RUN_ID, "SOURCE_COMMIT": runner.MASS_COMMIT,
                "COMMAND_STATUS": "0", "DISPATCHER_STATUS": "0", "CLEANUP_STATUS": "0",
                "ABORT_SIGNAL": "none", "GPU_COUNT": "1"}
    (run / "metadata.env").write_text("\n".join(f"{k}={v}" for k, v in metadata.items()), encoding="utf-8")
    for name in ("command_exit_status", "exit_status"):
        (run / name).write_text("0\n", encoding="utf-8")
    protocol = {"source_commit": runner.MASS_COMMIT, "n_eval": 200, "scientific_outcomes": 179200,
                "gate": "qwen7b-mass-prompt-encoding-specificity"}
    runner.write_json(run / "artifacts/meta.json", protocol)
    runner.write_json(run / "artifacts/mass-prompt-specificity-summary.json",
                      {**protocol, "n_scientific_outcomes": 179200, "n_outcomes": 177800,
                       "n_calibration_outcomes": 1400, "cross_wording_specificity": cross,
                       "discovery_wording_replication": discovery})
    return run


def source_fixture(root, concept="Mass"):
    dataset, runs, out = root / "dataset", root / "runs", root / "out"
    raw, manifest, have, receipts = fixture(105, concept)
    train = next(str(i) for i in range(100) if patient_split(str(i)) == "train")
    raw.append(row("schema", train, "|".join(core.DISEASES)))
    raw = [{field: "0" for field in RAW_FIELDS} | r for r in raw]
    write_csv(dataset / "Data_Entry_2017_v2020.csv", raw, RAW_FIELDS)
    write_csv(dataset / "manifest.csv", manifest, list(manifest[0]))
    images = dataset / "png/images"
    images.mkdir(parents=True)
    for name in have:
        (images / name).touch()
    for name, run, filename, _ in COHORTS:
        runner.write_json(runs / run / "artifacts" / filename, receipts[name])
    terminal(runs, cross=concept == "Mass", discovery=concept == "Mass")
    evidence = root / "evidence.json"
    runner.write_json(evidence, {"source_commit": "b" * 40, "exit_codes": [0, 0],
        "original_pool": {"inputs": {"dataset_root": str(dataset), "runs_root": str(runs),
                                     "image_directory": str(images)}}})
    return {"dataset_root": dataset, "runs_root": runs, "out": out,
            "source_commit": "a" * 40, "concept": concept, "evidence": evidence}


def score(semantic, cell):
    sign = -1 if runner.encoding_for(cell) == "B_present" else 1
    return [sign * semantic, semantic, float(core._sigmoid(np.asarray(semantic)))]


def preflight(concept):
    cells = core.PRIMARY_PROMPTS[concept]
    return {"passed": True, "token_ids": {cell: [[1], [2]] for cell in cells},
            "validation_row_ids": [f"val{i}" for i in range(16)],
            "mapping_cases": [{**case, "scores": score(2 * case["expected_present"] - 1, case["prompt_cell"])}
                              for case in runner.mapping_cases(concept)],
            "repeated_scores": {cell: [score(1., cell), score(1., cell)] for cell in cells},
            "throughput": runner.budget(10., 30., 200 * len(cells))}


def records(receipt):
    return [{"row_id": pair[side + "_row_id"], "patient_id": pair["patient_id"],
             "pair_index": i, "prompt": cell, "label": int(side == "positive"),
             **dict(zip(runner.FIELDS[-3:], score((i - 30.) if side == "positive" else -1., cell)))}
            for cell in core.PRIMARY_PROMPTS[receipt["concept"]]
            for i, pair in enumerate(receipt["pairs"]) for side in ("positive", "negative")]


class RunnerTests(unittest.TestCase):
    def test_terminal_route_all_branches_and_wrong_override(self):
        with tempfile.TemporaryDirectory() as directory:
            for cross, discovery, concept in ((True, True, "Mass"), (False, True, "Mass"),
                                               (False, False, "Consolidation")):
                run = terminal(Path(directory), cross, discovery)
                self.assertEqual(runner.terminal_route(run)["concept"], concept)
                wrong = "Mass" if concept == "Consolidation" else "Consolidation"
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    runner.terminal_route(run, wrong)

    def test_no_summary_read_before_terminal_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            run = terminal(Path(directory))
            for field in ("COMMAND_STATUS", "DISPATCHER_STATUS", "CLEANUP_STATUS", "ABORT_SIGNAL", "SOURCE_COMMIT"):
                terminal(Path(directory))
                path = run / "metadata.env"
                text = path.read_text(encoding="utf-8")
                lines = [line if not line.startswith(field + "=") else field + "=invalid" for line in text.splitlines()]
                path.write_text("\n".join(lines), encoding="utf-8")
                with patch.object(runner, "read_json") as read, self.assertRaisesRegex(ValueError, "terminal"):
                    runner.terminal_route(run)
                read.assert_not_called()
            terminal(Path(directory))
            (run / "command_exit_status").write_text("1", encoding="utf-8")
            with patch.object(runner, "read_json") as read, self.assertRaises(ValueError):
                runner.terminal_route(run)
            read.assert_not_called()

    def test_route_rejects_partial_counts_and_invalid_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            for key, value in (("n_scientific_outcomes", 100), ("n_outcomes", 0), ("n_eval", 199),
                               ("source_commit", "wrong"), ("cross_wording_specificity", 1),
                               ("discovery_wording_replication", False)):
                run = terminal(Path(directory))
                path = run / "artifacts/mass-prompt-specificity-summary.json"
                payload = runner.read_json(path)
                payload[key] = value
                runner.write_json(path, payload)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.terminal_route(run)

    def test_cpu_cohort_cli_receipt_replay_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            args = source_fixture(Path(directory))
            argv = ["cohort"] + [value for key, arg in args.items() for value in ("--" + key.replace("_", "-"), str(arg))]
            runner.main(argv)
            receipt = runner.read_json(args["out"] / "registered-pairs.json")
            self.assertEqual(receipt["source_commit"], "a" * 40)
            self.assertEqual(receipt["route"]["run_id"], runner.MASS_RUN_ID)
            self.assertEqual(len(runner.validate_pairs(receipt)), 100)
            self.assertEqual(runner.cohort(**args), receipt)
            csv_path = args["dataset_root"] / "Data_Entry_2017_v2020.csv"
            csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("Hernia", "Unknown"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                runner.cohort(**args)

    def test_evidence_path_mismatch_and_cohort_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            args = source_fixture(Path(directory))
            runner.cohort(**args)
            with self.assertRaisesRegex(ValueError, "prospective cohort"):
                runner.cohort(**(args | {"source_commit": "c" * 40}))
            evidence = runner.read_json(args["evidence"])
            evidence["original_pool"]["inputs"]["dataset_root"] = "other"
            runner.write_json(args["evidence"], evidence)
            with self.assertRaisesRegex(ValueError, "identities"):
                runner.cohort(**args)

    def test_grid_requires_complete_order_identities_metadata_and_finite_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = runner.cohort(**source_fixture(Path(directory)))
            valid = records(receipt)
            self.assertEqual(runner.validate_grid(valid, receipt).shape, (2, 100, 2, 3))
            for key, value in (("row_id", "wrong"), ("patient_id", "wrong"), ("pair_index", 1),
                               ("prompt", "anchor"), ("label", 0), ("raw_margin", np.nan),
                               ("semantic_margin", np.inf), ("semantic_probability", 1.1)):
                invalid = deepcopy(valid)
                invalid[0][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.validate_grid(invalid, receipt)
            for invalid in (valid[:-1], [valid[0], *valid[:-1]], list(reversed(valid))):
                with self.assertRaises(ValueError):
                    runner.validate_grid(invalid, receipt)
            for side, field, value in (("positive", "Patient Sex", "M"),
                                       ("negative", "Finding Labels", "Mass"),
                                       ("positive", "Patient ID", "other")):
                invalid = deepcopy(receipt)
                invalid["pairs"][0][side + "_metadata"][field] = value
                with self.assertRaises(ValueError):
                    runner.validate_grid(valid, invalid)
            invalid = deepcopy(valid)
            invalid[200]["raw_margin"] *= -1
            with self.assertRaisesRegex(ValueError, "orientation"):
                runner.validate_grid(invalid, receipt)

    def test_budget_includes_elapsed_and_allowance_boundary(self):
        self.assertTrue(runner.budget(32, 440, 400)["passed"])
        self.assertFalse(runner.budget(32, 440.001, 400)["passed"])
        for pilot, elapsed, count in ((0, 1, 400), (2, 1, 400), (np.nan, 3, 400), (1, 2, 999)):
            with self.assertRaises(ValueError):
                runner.budget(pilot, elapsed, count)

    def test_preflight_mapping_repeat_and_budget_failures(self):
        for concept in ("Mass", "Consolidation"):
            valid = preflight(concept)
            finding = "A lung mass" if concept == "Mass" else "Consolidation"
            for case in valid["mapping_cases"]:
                self.assertTrue(case["prompt"].startswith(finding + " is "))
                self.assertTrue(case["prompt"].endswith(core.PROMPTS[concept][case["prompt_cell"]]))
            self.assertTrue(runner.validate_preflight(valid, concept))
            invalid = deepcopy(valid)
            invalid["mapping_cases"][0]["scores"] = score(-1., core.PRIMARY_PROMPTS[concept][0])
            with self.assertRaisesRegex(ValueError, "mapping sign"):
                runner.validate_preflight(invalid, concept)
            invalid = deepcopy(valid)
            cell = core.PRIMARY_PROMPTS[concept][0]
            invalid["repeated_scores"][cell][1] = score(2., cell)
            with self.assertRaisesRegex(ValueError, "repeat exactly"):
                runner.validate_preflight(invalid, concept)
            invalid = deepcopy(valid)
            invalid["throughput"] = runner.budget(100., 500., 200 * len(core.PRIMARY_PROMPTS[concept]))
            with self.assertRaisesRegex(ValueError, "budget"):
                runner.validate_preflight(invalid, concept)

    def test_cpu_summarize_cli_npz_replay_and_retained_patients(self):
        with tempfile.TemporaryDirectory() as directory:
            args = source_fixture(Path(directory), "Consolidation")
            receipt = runner.cohort(**args)
            out = args["out"]
            runner.write_json(out / "meta.json", {**runner.protocol_metadata("Consolidation"),
                "source_commit": receipt["source_commit"], "route": receipt["route"],
                "patient_ids": receipt["patient_ids"], "validation_row_ids": [f"val{i}" for i in range(16)],
                "source": {"model_id": runner.MODEL_ID, "model_revision": runner.MODEL_REVISION,
                           "model_source": "accepted-local-model"}})
            runner.write_json(out / "preflight.json", preflight("Consolidation"))
            write_csv(out / "per-image.csv", records(receipt), runner.FIELDS)
            runner.main(["summarize", "--per-image", str(out / "per-image.csv"), "--out", str(out / "summary.json")])
            summary = runner.read_json(out / "summary.json")
            self.assertEqual(summary["n_outcomes"], 200)
            self.assertEqual(len(summary["patient_ids"]), 100)
            with np.load(out / "summary.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["patient_ids"], receipt["patient_ids"])
                np.testing.assert_array_equal(arrays["bootstrap_indices"],
                    np.random.default_rng(20260912).integers(0, 100, (5000, 100)))
                self.assertEqual(arrays["raw_margin"].shape, (1, 100, 2))
            first = (out / "summary.json").read_bytes()
            runner.summarize(out / "per-image.csv", out / "summary.json")
            self.assertEqual(first, (out / "summary.json").read_bytes())

    def test_fake_model_full_run_and_replay(self):
        self._fake_model_run("Mass")

    def test_fake_model_consolidation_run_and_replay(self):
        self._fake_model_run("Consolidation")

    def _fake_model_run(self, concept):
        from src import qwen_answer_encoding as ae

        class Inputs(dict):
            def to(self, device):
                return self

        class Logits:
            def __init__(self, inputs):
                self.inputs = inputs
            def __getitem__(self, item):
                return self
            def float(self):
                return self

        class Image:
            def __init__(self, path):
                self.path = path
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def convert(self, mode):
                return str(self.path)

        class Processor:
            tokenizer = SimpleNamespace(padding_side="right")
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"][-1]["text"]
            def __call__(self, **kwargs):
                return Inputs(kwargs)

        def scoring(logits, tokenizer, encoding, ids):
            prompt = logits.inputs["text"][0]
            cell = "anchor" if encoding == "standard" else "show/" + encoding
            images = logits.inputs.get("images")
            if images:
                margins = [2. if "_p.png" in path else -1. for path in images]
            else:
                margins = [-1. if "absent" in prompt.split(".", 1)[0] else 1.]
            return np.asarray([score(value, cell) for value in margins])

        from contextlib import nullcontext
        fake_torch = SimpleNamespace(bfloat16="bfloat16", inference_mode=nullcontext,
            cuda=SimpleNamespace(device_count=lambda: 1, get_device_name=lambda index: "NVIDIA A100"))
        model = Mock(side_effect=lambda **inputs: SimpleNamespace(logits=Logits(inputs)))
        model.eval.return_value = model
        models = Mock()
        models.from_pretrained.return_value = model
        processors = Mock()
        processors.from_pretrained.return_value = Processor()
        fake_transformers = SimpleNamespace(AutoModelForImageTextToText=models, AutoProcessor=processors)
        with tempfile.TemporaryDirectory() as directory:
            args = source_fixture(Path(directory), concept)
            model_path = Path(directory) / "model"
            model_path.mkdir()
            source = {"model_source": str(model_path), "model_id": runner.MODEL_ID,
                      "model_revision": runner.MODEL_REVISION}
            validation = [{"row_id": f"val{i}", "patient_id": f"v{i}", "image_path": f"val{i}.png"} for i in range(16)]
            with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}), \
                 patch("PIL.Image.open", Image), patch("src.gpu_env.bind_gpu"), \
                 patch.object(ae, "load_source", return_value=(source, [], [], validation, {})), \
                 patch.object(ae, "singleton_ids", return_value=[[1], [2]]), \
                 patch.object(ae, "score_logits", side_effect=scoring):
                self.assertTrue(runner.run(**args, ownership_run=Path(directory), gpu=0))
            summary = runner.read_json(args["out"] / "paired-opportunity-summary.json")
            self.assertEqual(summary["n_outcomes"], 200 * len(core.PRIMARY_PROMPTS[concept]))
            self.assertTrue(summary["opportunity_available"])
            self.assertEqual(models.from_pretrained.call_args.kwargs,
                             {"dtype": "bfloat16", "device_map": "cuda:0", "local_files_only": True})
            self.assertEqual(processors.from_pretrained.call_args.kwargs,
                {"local_files_only": True, "min_pixels": 336 * 336, "max_pixels": 336 * 336})
            before = model.call_count
            runner.summarize(args["out"] / "per-image.csv", args["out"] / "replayed.json")
            self.assertEqual(model.call_count, before)
            self.assertEqual(summary, runner.read_json(args["out"] / "replayed.json"))
            with self.assertRaisesRegex(ValueError, "already exists"):
                runner.run(**args, ownership_run=Path(directory), gpu=0)

    def test_module_and_cli_help_are_cpu_only(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; import src.run_qwen_paired_opportunity; assert 'torch' not in sys.modules"],
            cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", "-m", "src.run_qwen_paired_opportunity", "--help"],
                                cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cohort,run,summarize", result.stdout)

    def test_launcher_immutable_path_gpu_offline_and_timeout(self):
        script = (Path(__file__).resolve().parents[1] / "scripts/server/run_qwen_paired_opportunity_gate.sh").read_text(encoding="utf-8")
        for fragment in ('GPU_COUNT=1', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                         '900', 'python -B -m src.run_qwen_paired_opportunity',
                         '"$(realpath -- "$source_dir/..")"', 'cd -- "$source_dir"'):
            self.assertIn(fragment, script)


if __name__ == "__main__":
    unittest.main()
