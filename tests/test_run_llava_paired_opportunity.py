from __future__ import annotations

from copy import deepcopy
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from time import perf_counter as time_counter
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src import run_llava_paired_opportunity as runner
from src import run_qwen_paired_opportunity as qwen
from src import qwen_paired_opportunity as core
from test_run_qwen_paired_opportunity import preflight, source_fixture, write_csv


def records(receipt, margins):
    return [{"row_id": pair[side + "_row_id"], "patient_id": pair["patient_id"],
             "pair_index": i, "prompt": "anchor", "label": int(side == "positive"),
             "raw_margin": value, "semantic_margin": value,
             "semantic_probability": float(core._sigmoid(np.asarray(value)))}
            for i, pair in enumerate(receipt["pairs"])
            for side, value in (("positive", float(margins[i])), ("negative", 0.))]


def accepted_fixture(root):
    args = source_fixture(root, "Consolidation")
    args["source_commit"] = runner.QWEN_COMMIT
    original_path = args["runs_root"] / runner.COHORT_RUN_ID / "artifacts/registered-pairs.json"
    original = qwen.read_json(original_path)
    original.update(source_commit=runner.COHORT_COMMIT,
                    sources={"manifest": str(args["dataset_root"] / "manifest.csv"),
                             "image_directory": str(args["dataset_root"] / "png/images")})
    qwen.write_json(original_path, original)
    accepted_run = args["runs_root"] / runner.QWEN_RUN_ID
    args["out"] = accepted_run / "artifacts"
    receipt = qwen.cohort(**args)
    metadata = {"RUN_ID": runner.QWEN_RUN_ID, "SOURCE_COMMIT": runner.QWEN_COMMIT,
                "COMMAND_STATUS": "0", "DISPATCHER_STATUS": "0", "CLEANUP_STATUS": "0",
                "ABORT_SIGNAL": "none", "GPU_COUNT": "1"}
    (accepted_run / "metadata.env").write_text("\n".join(f"{k}={v}" for k, v in metadata.items()), encoding="utf-8")
    for name in ("command_exit_status", "exit_status"):
        (accepted_run / name).write_text("0\n", encoding="utf-8")
    meta = {**qwen.protocol_metadata("Consolidation"), "source_commit": runner.QWEN_COMMIT,
            "route": receipt["route"], "patient_ids": receipt["patient_ids"],
            "validation_row_ids": runner.VALIDATION_ROW_IDS,
            "source": {"model_id": qwen.MODEL_ID, "model_revision": qwen.MODEL_REVISION,
                       "model_source": "accepted-qwen-snapshot"}}
    qwen.write_json(args["out"] / "meta.json", meta)
    flight = preflight("Consolidation")
    flight["validation_row_ids"] = runner.VALIDATION_ROW_IDS
    qwen.write_json(args["out"] / "preflight.json", flight)
    write_csv(args["out"] / "per-image.csv", records(receipt, np.arange(100) - 60.), qwen.FIELDS)
    qwen.summarize(args["out"] / "per-image.csv", args["out"] / "paired-opportunity-summary.json")
    validation = [{"row_id": key, "patient_id": "validation" + str(i), "split": "val",
                   "image_path": str(root / (key + ".png"))} for i, key in enumerate(runner.VALIDATION_ROW_IDS)]
    write_csv(args["dataset_root"] / "manifest.csv", validation[::-1], list(validation[0]))
    model_root = root / "models/huggingface"
    snapshot = model_root / runner.SNAPSHOT_RELATIVE_PATH
    snapshot.mkdir(parents=True)
    asset = {"asset": runner.ARCH, "repo_id": runner.MODEL_ID, "revision": runner.MODEL_REVISION,
             "snapshot_relative_path": runner.SNAPSHOT_RELATIVE_PATH, "source": "accepted-source",
             "source_commit": "c" * 40, "verified_files": {"config.json": {"bytes": 1, "sha256": "d" * 64}}}
    qwen.write_json(model_root / "asset-receipt.json", asset)
    inputs = {"model_id": runner.MODEL_ID, "model_revision": runner.MODEL_REVISION,
              "snapshot": str(snapshot), "model_receipt": str(model_root / "asset-receipt.json"),
              "model_receipt_sha256": "b" * 64, "source_commit": runner.ASSET_COMMIT,
              "vision_feature_layer": -2}
    qwen.write_json(args["runs_root"] / runner.ASSET_RUN_ID / "artifacts/input-verification.json", inputs)
    return {"dataset_root": args["dataset_root"], "runs_root": args["runs_root"],
            "model_root": model_root, "out": root / "llava", "source_commit": "a" * 40, "gpu": 0}


def fake_score(prompt, rows=()):
    if rows:
        margins = np.asarray([2. if "_p.png" in r["image_path"] else -1. for r in rows])
    else:
        margins = np.asarray([-1. if "absent" in prompt.split(".")[0] else 1.])
    return np.stack((margins, margins, core._sigmoid(margins)), axis=-1)


def scorer_mock():
    score = Mock(side_effect=fake_score)
    return score, [[1, 3], [2, 4]], {"gpu_name": "NVIDIA A100", "gpu_count": 1,
                                    "chat_template": "native template"}


class StatisticsTests(unittest.TestCase):
    def test_stored_draws_ties_and_patient_paired_concordance(self):
        indices = np.tile(np.arange(100, dtype=np.int64), (5000, 1))
        indices[:, :30] = 99
        llava = np.resize([-2., 0., 3.], 100)[None, :]
        qwen = np.resize([1., 0., -1., -2.], 100)[None, :]
        with patch.object(np.random, "default_rng", side_effect=AssertionError("fresh sampling")):
            result, arrays = runner.summarize_arrays(
                llava, np.zeros_like(llava), qwen, np.zeros_like(qwen), indices,
                np.arange(100) < 40)
        ranking = (llava > 0) + .5 * (llava == 0)
        qwen_ranking = (qwen > 0) + .5 * (qwen == 0)
        delta = ranking - qwen_ranking
        boot = llava[:, indices].mean(axis=2)
        np.testing.assert_array_equal(arrays["bootstrap_indices"], indices)
        np.testing.assert_array_equal(arrays["concordance_difference"], delta)
        np.testing.assert_array_equal(arrays["bootstrap_concordance_difference"], delta[:, indices].mean(axis=2))
        anchor = result["prompts"]["anchor"]
        self.assertEqual(anchor["margin_gap_lower_95"], np.percentile(boot[0], 5))
        self.assertEqual(len(anchor["per_patient"]["margin_gap"]), 100)
        self.assertEqual(anchor["negative_no_finding_strata"]["true"]["patients"], 40)
        self.assertEqual(result["cross_model"]["mean_concordance_difference"], delta.mean())
        np.testing.assert_array_equal(result["cross_model"]["concordance_difference_ci95"],
                                      np.percentile(delta[:, indices].mean(axis=2)[0], [2.5, 97.5]))

    def test_strict_zero_threshold_and_cross_model_scale_invariance(self):
        indices = np.tile(np.arange(100, dtype=np.int64), (5000, 1))
        for gap, passed in ((-1., False), (0., False), (1., True)):
            values = np.full((1, 100), gap)
            result, _ = runner.summarize_arrays(values, values * 0, values * 100,
                                                values * 0, indices, np.ones(100, dtype=bool))
            self.assertIs(result["opportunity_available"], passed)
            self.assertEqual(result["cross_model"]["mean_concordance_difference"], 0.)

    def test_invalid_indices_scores_and_strata(self):
        indices = np.tile(np.arange(100, dtype=np.int64), (5000, 1))
        values, flags = np.zeros((1, 100)), np.ones(100, dtype=bool)
        for bad in (indices[:4999], indices.astype(float), indices.astype(np.int32),
                    np.full_like(indices, -1), np.full_like(indices, 100)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                runner.summarize_arrays(values, values, values, values, bad, flags)
        for bad in (np.zeros((100,)), np.full((1, 100), np.inf), np.full((1, 100), np.nan)):
            with self.assertRaises(ValueError):
                runner.summarize_arrays(bad, values, values, values, indices, flags)
        for bad in (flags[:99], ["False"] * 100, np.full(100, 2)):
            with self.assertRaises(ValueError):
                runner.summarize_arrays(values, values, values, values, indices, bad)


class RunnerTests(unittest.TestCase):
    def test_reference_import_preserves_receipts_and_never_reselects(self):
        with tempfile.TemporaryDirectory() as directory:
            args = accepted_fixture(Path(directory))
            (args["dataset_root"] / "Data_Entry_2017_v2020.csv").unlink()
            with patch.object(core, "select_pairs", side_effect=AssertionError("selection")), \
                 patch.object(np.random, "default_rng", side_effect=AssertionError("fresh sampling")):
                reference, arrays = runner.load_reference(args["runs_root"])
            original = qwen.read_json(args["runs_root"] / runner.COHORT_RUN_ID / "artifacts/registered-pairs.json")
            self.assertEqual(reference["cohort_receipt"], original)
            np.testing.assert_array_equal(arrays["row_ids"], runner.row_ids(original))
            self.assertEqual([r["row_id"] for r in runner.validation_rows(args["dataset_root"], reference)],
                             runner.VALIDATION_ROW_IDS)

    def test_reference_rejects_provenance_cohort_order_model_and_grid_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args = accepted_fixture(Path(directory))
            reference, arrays = runner.load_reference(args["runs_root"])
            mutations = [
                lambda r: r.update(qwen_run_id="other"),
                lambda r: r["cohort_receipt"].update(source_commit="a" * 40),
                lambda r: r["qwen_meta"].update(source_commit="a" * 40),
                lambda r: r["qwen_meta"]["source"].update(model_id=runner.MODEL_ID),
                lambda r: r["qwen_meta"]["source"].update(model_revision="wrong"),
                lambda r: r["qwen_meta"]["validation_row_ids"].reverse(),
                lambda r: r["cohort_receipt"]["patient_ids"].reverse(),
                lambda r: r["qwen_cohort_receipt"]["pairs"][0]["positive_metadata"].update({"Height]": "2048"}),
                lambda r: r["qwen_records"].reverse(),
                lambda r: r["qwen_records"].pop(),
                lambda r: r["qwen_records"][0].update(raw_margin="nan"),
                lambda r: r["qwen_summary"].update(opportunity_available=True),
            ]
            for i, mutate in enumerate(mutations):
                changed = deepcopy(reference)
                mutate(changed)
                with self.subTest(mutation=i), self.assertRaises(ValueError):
                    runner.validate_reference(changed, arrays)
            for key in ("bootstrap_indices", "patient_ids", "row_ids", "positive_margin", "negative_margin",
                        "pair_ranking", "raw_margin", "semantic_probability"):
                changed = {k: v.copy() for k, v in arrays.items()}
                if key == "bootstrap_indices":
                    changed[key][0, 0] = 100
                elif key in ("patient_ids", "row_ids"):
                    changed[key] = changed[key][::-1]
                else:
                    changed[key].flat[0] += .25
                with self.subTest(array=key), self.assertRaises(ValueError):
                    runner.validate_reference(reference, changed)

    def test_historical_bootstrap_statistics_are_preserved_as_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            args = accepted_fixture(Path(directory))
            path = args["runs_root"] / runner.QWEN_RUN_ID / "artifacts/paired-opportunity-summary.npz"
            with np.load(path, allow_pickle=False) as bundle:
                arrays = {key: bundle[key] for key in bundle.files}
            arrays["bootstrap_margin_gap"] = np.nextafter(arrays["bootstrap_margin_gap"], np.inf)
            arrays["bootstrap_indices"] = arrays["bootstrap_indices"][::-1].copy()
            np.savez_compressed(path, **arrays)
            with patch.object(np.random, "default_rng", side_effect=AssertionError("fresh sampling")), \
                 patch.object(runner, "load_scorer", return_value=scorer_mock()):
                runner.run(**args)
            with np.load(args["out"] / "paired-opportunity-summary.npz", allow_pickle=False) as result:
                np.testing.assert_array_equal(result["qwen_bootstrap_margin_gap"], arrays["bootstrap_margin_gap"])
                np.testing.assert_array_equal(result["bootstrap_indices"], arrays["bootstrap_indices"])
                np.testing.assert_array_equal(result["bootstrap_concordance_difference"],
                    result["concordance_difference"][:, arrays["bootstrap_indices"]].mean(axis=2))

    def test_wrong_terminal_fails_before_reading_outcomes_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            args = accepted_fixture(Path(directory))
            path = args["runs_root"] / runner.QWEN_RUN_ID / "command_exit_status"
            path.write_text("1", encoding="utf-8")
            with patch.object(runner, "read_json") as read, patch.object(runner, "load_scorer") as gpu, \
                 self.assertRaisesRegex(ValueError, "terminal"):
                runner.run(**args)
            read.assert_not_called()
            gpu.assert_not_called()

    def test_asset_source_and_validation_mismatches_fail_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            args = accepted_fixture(Path(directory))
            reference, _ = runner.load_reference(args["runs_root"])
            source = runner.load_model_source(args["model_root"], args["runs_root"])
            self.assertEqual(source["input_verification"]["model_receipt_sha256"], "b" * 64)
            path = args["model_root"] / "asset-receipt.json"
            original = qwen.read_json(path)
            for key in ("asset", "repo_id", "revision", "snapshot_relative_path"):
                qwen.write_json(path, original | {key: "wrong"})
                with patch.object(runner, "load_scorer") as gpu, self.assertRaises(ValueError):
                    runner.run(**args)
                gpu.assert_not_called()
            qwen.write_json(path, original)
            rows = runner.validation_rows(args["dataset_root"], reference)
            for key, value in (("split", "test"), ("patient_id", reference["cohort_receipt"]["patient_ids"][0])):
                changed = deepcopy(rows)
                changed[0][key] = value
                write_csv(args["dataset_root"] / "manifest.csv", changed, list(changed[0]))
                with self.assertRaises(ValueError):
                    runner.validation_rows(args["dataset_root"], reference)

    def test_complete_mocked_boundary_real_serialization_and_cpu_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            args = accepted_fixture(Path(directory))
            score, ids, runtime = scorer_mock()
            with patch.object(runner, "load_scorer", return_value=(score, ids, runtime)), \
                 patch.object(core, "select_pairs", side_effect=AssertionError("selection")), \
                 patch.object(np.random, "default_rng", side_effect=AssertionError("fresh sampling")):
                self.assertTrue(runner.run(**args))
            summary_path = args["out"] / "paired-opportunity-summary.json"
            summary = qwen.read_json(summary_path)
            self.assertTrue(summary["opportunity_available"])
            self.assertFalse(summary["qwen_reference"]["qwen_summary"]["opportunity_available"])
            self.assertEqual(summary["n_outcomes"], 200)
            self.assertEqual(summary["cross_model"]["mean_concordance_difference"], .605)
            self.assertEqual(summary["cohort_receipt"]["source_commit"], runner.COHORT_COMMIT)
            self.assertEqual(summary["source_commit"], "a" * 40)
            scientific = score.call_args_list[14:]
            self.assertEqual(len(scientific), 13)
            self.assertEqual([len(call.args[1]) for call in scientific], [16] * 12 + [8])
            self.assertTrue(all(call.args[0] == runner.PROMPT for call in scientific))
            with np.load(summary_path.with_suffix(".npz"), allow_pickle=False) as bundle:
                self.assertEqual(bundle["raw_margin"].shape, (1, 100, 2))
                self.assertEqual(bundle["concordance_difference"].shape, (1, 100))
                self.assertEqual(bundle["bootstrap_concordance_difference"].shape, (1, 5000))
                np.testing.assert_array_equal(bundle["bootstrap_indices"], bundle["qwen_bootstrap_indices"])
                with np.load(args["out"] / "qwen-reference.npz", allow_pickle=False) as accepted:
                    for key in accepted.files:
                        np.testing.assert_array_equal(bundle["qwen_" + key], accepted[key])
            with patch.object(runner, "load_scorer", side_effect=AssertionError("GPU")):
                self.assertEqual(runner.summarize(args["out"] / "per-image.csv", args["out"] / "replay.json"), summary)
            self.assertEqual(summary_path.read_bytes(), (args["out"] / "replay.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "already exists"):
                runner.run(**args)

    def test_preflight_failure_prevents_scientific_images_and_records_error(self):
        for kind in ("mapping", "repeat", "budget", "tokens"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                args = accepted_fixture(Path(directory))
                score, ids, runtime = scorer_mock()
                if kind == "mapping":
                    score.side_effect = lambda prompt, rows=(): fake_score("The finding is present.", rows)
                elif kind == "repeat":
                    def changing(prompt, rows=()):
                        values = fake_score(prompt, rows)
                        if rows and len(rows) == 1 and score.call_count == 6:
                            values[:, :2] = 2
                            values[:, 2] = core._sigmoid(values[:, 1])
                        return values
                    score.side_effect = changing
                elif kind == "tokens":
                    ids = [[1], [1]]
                with patch.object(runner, "load_scorer", return_value=(score, ids, runtime)), \
                     patch.object(runner.time, "perf_counter", side_effect=[0., 100., 450., 500.] if kind == "budget" else None,
                                  wraps=time_counter), self.assertRaises(ValueError):
                    runner.run(**args)
                self.assertFalse((args["out"] / "per-image.csv").exists())
                self.assertFalse((args["out"] / "per-image.csv.tmp").exists())
                flight = qwen.read_json(args["out"] / "preflight.json")
                self.assertFalse(flight["passed"])
                self.assertIn("ValueError", flight["error"])
                self.assertFalse(any("_p.png" in r["image_path"] for call in score.call_args_list
                                     for r in (call.args[1] if len(call.args) > 1 else [])))

    def test_saved_model_identity_protocol_and_preflight_reject_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args = accepted_fixture(Path(directory))
            with patch.object(runner, "load_scorer", return_value=scorer_mock()):
                runner.run(**args)
            path = args["out"] / "meta.json"
            original = qwen.read_json(path)
            mutations = [
                lambda m: m.update(model_id=qwen.MODEL_ID),
                lambda m: m.update(model_revision="wrong"),
                lambda m: m.update(source_commit="short"),
                lambda m: m["source"].update(model_source="another-snapshot"),
                lambda m: m["source"]["asset_receipt"].update(repo_id="other"),
                lambda m: m["source"]["input_verification"].update(model_revision="other"),
                lambda m: m["source"]["input_verification"].update(source_commit="a" * 40),
                lambda m: m["runtime"].update(gpu_count=2),
                lambda m: m["token_ids"].update(anchor=[[9], [10]]),
                lambda m: m["validation_rows"].reverse(),
                lambda m: m["prompts"].update(anchor="different prompt"),
                lambda m: m["patient_ids"].reverse(),
            ]
            for i, mutate in enumerate(mutations):
                changed = deepcopy(original)
                mutate(changed)
                qwen.write_json(path, changed)
                with self.subTest(mutation=i), self.assertRaises(ValueError):
                    runner.summarize(args["out"] / "per-image.csv", args["out"] / "rejected.json")
            self.assertFalse((args["out"] / "rejected.json").exists())

    def test_preflight_only_and_hard_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            args = accepted_fixture(Path(directory))
            score, ids, runtime = scorer_mock()
            with patch.object(runner, "load_scorer", return_value=(score, ids, runtime)):
                self.assertTrue(runner.run(**args, preflight_only=True))
            self.assertEqual(score.call_count, 14)
            self.assertFalse((args["out"] / "per-image.csv").exists())
            self.assertTrue(qwen.read_json(args["out"] / "preflight.json")["passed"])
        with patch.object(runner.time, "perf_counter", return_value=900):
            with self.assertRaises(TimeoutError):
                runner.check_deadline(0)

    def test_local_factory_native_processor_and_real_cpu_variant_max_scoring(self):
        import torch
        from src import intervene
        from transformers.image_utils import SizeDict

        class Inputs(dict):
            def to(self, device):
                self.device = device
                return self

        class Tokenizer:
            padding_side = "right"
            def encode(self, text, **kwargs):
                table = {"yes": 1, " yes": 3, "Yes": 5, " Yes": 7, "YES": 9, " YES": 11,
                         "no": 2, " no": 4, "No": 6, " No": 8, "NO": 10, " NO": 12}
                return [table[text]]

        image_processor = SimpleNamespace(
            do_resize=True, size=SizeDict(shortest_edge=336), do_center_crop=True,
            crop_size=SizeDict(height=336, width=336), do_convert_rgb=True,
            do_normalize=True, do_rescale=True, resample=3, rescale_factor=1 / 255,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711])
        processor = Mock(image_processor=image_processor, patch_size=14, num_additional_image_tokens=1,
                         vision_feature_select_strategy="default", image_token="<image>",
                         chat_template="native template", tokenizer=Tokenizer())
        processor.apply_chat_template.return_value = "native rendered prompt"
        processor.side_effect = lambda **kwargs: Inputs(kwargs)
        logits = torch.zeros((2, 1, 16), dtype=torch.bfloat16)
        logits[:, :, 11] = 5
        logits[:, :, 2] = 2
        model = Mock(return_value=SimpleNamespace(logits=logits))
        model.eval.return_value = model
        model.config = SimpleNamespace(vision_feature_layer=-2, vision_feature_select_strategy="default",
            vision_config=SimpleNamespace(patch_size=14, image_size=336))
        runner.validate_processor(processor, model.config)
        image_processor.size = SizeDict(shortest_edge=224)
        with self.assertRaises(ValueError):
            runner.validate_processor(processor, model.config)
        image_processor.size = SizeDict(shortest_edge=336)
        factories = SimpleNamespace(AutoProcessor=Mock(), AutoModelForImageTextToText=Mock())
        factories.AutoProcessor.from_pretrained.return_value = processor
        factories.AutoModelForImageTextToText.from_pretrained.return_value = model
        image = Mock()
        image.__enter__ = Mock(return_value=image)
        image.__exit__ = Mock(return_value=False)
        image.convert.return_value = "RGB image"
        with patch.dict(sys.modules, {"transformers": factories}), patch("src.gpu_env.bind_gpu") as bind, \
             patch.object(torch.cuda, "device_count", return_value=1), \
             patch.object(torch.cuda, "get_device_name", return_value="NVIDIA A100"), \
             patch("PIL.Image.open", return_value=image), \
             patch.object(intervene, "yes_margin_batch", wraps=intervene.yes_margin_batch) as margin:
            score, ids, runtime = runner.load_scorer({"model_source": "local-snapshot"}, 3, time_counter())
            result = score(runner.PROMPT, [{"image_path": "one.png"}, {"image_path": "two.png"}])
        bind.assert_called_once_with(3)
        margin.assert_called_once()
        self.assertEqual(ids, [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]])
        np.testing.assert_array_equal(result[:, :2], np.full((2, 2), 3.))
        np.testing.assert_allclose(result[:, 2], core._sigmoid(np.asarray([3., 3.])))
        self.assertEqual(processor.tokenizer.padding_side, "left")
        self.assertEqual(processor.call_args.kwargs["images"], ["RGB image", "RGB image"])
        self.assertTrue(processor.call_args.kwargs["padding"])
        self.assertEqual(processor.apply_chat_template.call_args.args[0][0]["content"],
                         [{"type": "image"}, {"type": "text", "text": runner.PROMPT}])
        self.assertEqual(runtime["chat_template"], "native template")
        factories.AutoProcessor.from_pretrained.assert_called_once_with("local-snapshot", local_files_only=True)
        factories.AutoModelForImageTextToText.from_pretrained.assert_called_once_with(
            "local-snapshot", dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True)

    def test_budget_boundary_and_preflight_validation(self):
        self.assertTrue(qwen.budget(64, 440, 200)["passed"])
        self.assertFalse(qwen.budget(64, 440.001, 200)["passed"])
        valid = preflight("Consolidation")
        valid["validation_row_ids"] = runner.VALIDATION_ROW_IDS
        self.assertTrue(runner.validate_preflight(valid))
        for ids in ([], [[], [2]], [[1], [1]]):
            changed = deepcopy(valid)
            changed["token_ids"]["anchor"] = ids
            with self.assertRaises(ValueError):
                runner.validate_preflight(changed)
        changed = deepcopy(valid)
        changed["mapping_cases"][0]["scores"] = [0., 0., .5]
        with self.assertRaisesRegex(ValueError, "sign"):
            runner.validate_preflight(changed)

    def test_cpu_import_help_and_launcher_contract(self):
        root = Path(__file__).resolve().parents[1]
        for argv in (["-c", "import sys; import src.run_llava_paired_opportunity; assert 'torch' not in sys.modules"],
                     ["-m", "src.run_llava_paired_opportunity", "--help"]):
            process = subprocess.run([sys.executable, "-B", *argv], cwd=root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
        script = (root / "scripts/server/run_llava_paired_opportunity_gate.sh").read_text(encoding="utf-8")
        for fragment in ('GPU_COUNT=1', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                         'HF_HOME="$data_root/models/huggingface"', 'SOURCE_PATH=$source_dir',
                         'SOURCE_COMMIT=$expected_sha', '"$source_dir" = "$run_dir/source"',
                         'source scripts/server/activate_env.sh', '--kill-after=10 900',
                         'python -B -m src.run_llava_paired_opportunity'):
            self.assertIn(fragment, script)


if __name__ == "__main__":
    unittest.main()
