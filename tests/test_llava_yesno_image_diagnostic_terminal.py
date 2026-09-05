import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from scripts.server import validate_llava_yesno_image_diagnostic as terminal
from src import llava_yesno_image_diagnostic as diagnostic


class TerminalReplayTests(unittest.TestCase):
    def test_string_receipt_paths_are_converted_for_shared_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            self.assertEqual(terminal.read_path(str(path)), {"status": "ok"})

    def test_independent_primary_vector_matches_registration(self):
        np.testing.assert_allclose(
            terminal.primary(np.asarray([0.8, 0.7]), np.asarray([0.2, 0.35])),
            [0.3, 0.2, 0.15],
        )

    def test_terminal_lse_tolerance_covers_observed_float32_reduction_only(self):
        terminal.same(
            np.asarray([0.0]),
            np.asarray([3.814697265625e-6]),
            "lse reduction",
            terminal.LSE_REPLAY_ATOL,
        )
        with self.assertRaisesRegex(AssertionError, "lse reduction"):
            terminal.same(
                np.asarray([0.0]),
                np.asarray([terminal.LSE_REPLAY_ATOL * 1.01]),
                "lse reduction",
                terminal.LSE_REPLAY_ATOL,
            )
        terminal.same(
            np.asarray([0.0]),
            np.asarray([2.0265579223632812e-6]),
            "token mass reduction",
            terminal.TOKEN_MASS_REPLAY_ATOL,
        )
        with self.assertRaisesRegex(AssertionError, "token mass reduction"):
            terminal.same(
                np.asarray([0.0]),
                np.asarray([terminal.TOKEN_MASS_REPLAY_ATOL * 1.01]),
                "token mass reduction",
                terminal.TOKEN_MASS_REPLAY_ATOL,
            )

    def test_independent_prompt_and_mapping_identities_are_exact(self):
        self.assertEqual([item["name"] for item in terminal.expected_conditions()], ["0Y", "1Y"])
        self.assertEqual(len(terminal.expected_mapping_cases()), 4)
        self.assertEqual(
            terminal.expected_mapping_cases()[0]["prompt"],
            "The finding is present. Is the finding present? Answer yes or no.",
        )

    def test_independent_token_check_rejects_drift(self):
        expected = {
            "raw_first": terminal.YES,
            "raw_second": terminal.NO,
            "present": terminal.YES,
            "absent": terminal.NO,
            "candidate_ids": sorted(terminal.YES + terminal.NO),
        }
        terminal.validate_token_metadata({"0Y": expected, "1Y": dict(expected)})
        changed = {"0Y": dict(expected), "1Y": dict(expected)}
        changed["1Y"]["raw_first"] = [1]
        with self.assertRaisesRegex(AssertionError, "token identity"):
            terminal.validate_token_metadata(changed)

    def test_complete_protocol_and_model_fields_fail_closed_on_drift(self):
        protocol = terminal.expected_protocol()
        terminal.validate_protocol(protocol)
        changed = deepcopy(protocol)
        changed["preprocessing"]["shortest_edge"] = 448
        with self.assertRaisesRegex(AssertionError, "protocol identity"):
            terminal.validate_protocol(changed)
        model = {
            "model_id": terminal.MODEL_ID,
            "model_revision": terminal.MODEL_REVISION,
            "model_source": terminal.MODEL_SOURCE,
            "model_receipt": terminal.MODEL_RECEIPT,
            "input_verification_path": terminal.INPUT_VERIFICATION,
            "asset_receipt": {
                "asset": terminal.ASSET,
                "repo_id": terminal.MODEL_ID,
                "revision": terminal.MODEL_REVISION,
                "snapshot_relative_path": terminal.SNAPSHOT_RELATIVE_PATH,
                "source_commit": terminal.ASSET_SOURCE_COMMIT,
            },
            "input_verification": {
                "model_id": terminal.MODEL_ID,
                "model_revision": terminal.MODEL_REVISION,
                "source_commit": terminal.ASSET_COMMIT,
                "snapshot": terminal.MODEL_SOURCE,
                "model_receipt": terminal.MODEL_RECEIPT,
                "model_receipt_sha256": terminal.MODEL_RECEIPT_SHA256,
                "vision_feature_layer": -2,
                "verification_mode": "accepted-asset-receipt",
            },
        }
        terminal.validate_model_source(model)
        model["model_revision"] = "0" * 40
        with self.assertRaisesRegex(AssertionError, "model identity"):
            terminal.validate_model_source(model)
        model["model_revision"] = terminal.MODEL_REVISION
        for section, key in (
            (model, "model_receipt"),
            (model, "input_verification_path"),
            (model["asset_receipt"], "snapshot_relative_path"),
            (model["input_verification"], "source_commit"),
            (model["input_verification"], "model_receipt_sha256"),
        ):
            original = section[key]
            section[key] = "drift"
            with self.assertRaisesRegex(AssertionError, "receipt binding"):
                terminal.validate_model_source(model)
            section[key] = original

    def test_terminal_reader_arithmetic_matches_registered_float32_replay(self):
        rng = np.random.default_rng(41)
        raw = rng.normal(size=(7, 1024)).astype(np.float16)
        prepared = {
            "projection": rng.normal(size=(1024, 512)).astype(np.float32),
            "train_mean": rng.normal(size=512).astype(np.float64),
            "scale": rng.uniform(0.25, 2.0, size=512).astype(np.float64),
            "coefficients": rng.normal(size=(6, 512)).astype(np.float64),
            "control_coefficients": rng.normal(size=(20, 512)).astype(np.float64),
            "control_intercepts": rng.normal(size=20).astype(np.float64),
        }
        expected = diagnostic.frozen_reader_scores(raw, prepared)
        observed = terminal.reader_scores(raw, prepared)
        self.assertEqual(observed[1].dtype, np.float32)
        for actual, wanted in zip(observed, expected, strict=True):
            np.testing.assert_array_equal(actual, wanted)

    def test_preflight_binds_commit_model_runtime_and_timing(self):
        token = {
            "raw_first": terminal.YES,
            "raw_second": terminal.NO,
            "present": terminal.YES,
            "absent": terminal.NO,
            "candidate_ids": sorted(terminal.YES + terminal.NO),
        }
        cases = []
        for identity in terminal.expected_mapping_cases():
            sign = 2 * identity["expected_present"] - 1
            cases.append({**identity, "semantic_margin": float(sign), "lse_margin": float(sign)})
        preparation, gpu_elapsed, pilot = 2.0, 10.0, 1.0
        timing = {
            "pilot_equivalents": 512,
            "pilot_seconds": pilot,
            "gpu_phase_elapsed_seconds": gpu_elapsed,
            "preparation_seconds": preparation,
            "elapsed_preparation_loading_preflight_and_pilot": preparation + gpu_elapsed,
            "scientific_outcomes": 2802,
            "analysis_allowance_seconds": 600,
            "projected_seconds": preparation + gpu_elapsed + 2802 * pilot / 512 + 600,
            "limit_seconds": 2700,
            "passed": True,
        }
        preflight = {
            "source_commit": "a" * 40,
            "model_source": terminal.MODEL_SOURCE,
            "validation_row_ids": list(terminal.VALIDATION_ROW_IDS),
            "module": "model.vision_tower.encoder.layers.22",
            "pooling": "mean over all 577 block-output tokens",
            "forwarded_no_cls_exact": True,
            "clean_repeat_exact": True,
            "consumed_prompts": terminal.expected_consumed_prompts(),
            "passed": True,
            "runtime": {
                "gpu_name": "NVIDIA A100 80GB PCIe",
                "gpu_count": 1,
                "chat_template": "accepted template",
                "module": "model.vision_tower.encoder.layers.22",
                "positions": "all",
                "model_source": terminal.MODEL_SOURCE,
            },
            "token_ids": {"0Y": token, "1Y": dict(token)},
            "mapping_cases": cases,
            "throughput": timing,
        }
        terminal.validate_preflight(preflight, preparation, "a" * 40, terminal.MODEL_SOURCE)
        changed = deepcopy(preflight)
        changed["runtime"]["positions"] = "cls"
        with self.assertRaisesRegex(AssertionError, "runtime"):
            terminal.validate_preflight(changed, preparation, "a" * 40, terminal.MODEL_SOURCE)
        changed = deepcopy(preflight)
        changed["consumed_prompts"][0]["prompt"] = "drift"
        with self.assertRaisesRegex(AssertionError, "consumed-locus"):
            terminal.validate_preflight(changed, preparation, "a" * 40, terminal.MODEL_SOURCE)

    def test_text_only_native_logits_and_partition_are_replayed(self):
        logits = np.asarray([0.0, -1.0, 2.0, 1.0, -2.0, 0.5], dtype=np.float32)
        lookup = {
            token_id: index for index, token_id in enumerate(sorted(terminal.YES + terminal.NO))
        }
        yes = [lookup[value] for value in terminal.YES]
        no = [lookup[value] for value in terminal.NO]
        raw = float(logits[yes].max() - logits[no].max())
        lse = float(np.logaddexp.reduce(logits[yes]) - np.logaddexp.reduce(logits[no]))
        partition = 4.0
        mass = float(np.exp(np.logaddexp.reduce(logits) - partition))
        records = [
            {
                "condition": spec["name"],
                "wording": spec["wording"],
                "prompt": spec["prompt"],
                "raw_margin": raw,
                "semantic_margin": raw,
                "probability": float(1 / (1 + np.exp(-raw))),
                "answer_token_mass": mass,
                "lse_margin": lse,
                "candidate_logits": logits.tolist(),
                "log_partition": partition,
                "constant_auroc_against_index_labels": 0.5,
                "subtracting_constant_preserves_image_auroc": True,
            }
            for spec in terminal.expected_conditions()
        ]
        terminal.validate_text_only(records)
        changed = deepcopy(records)
        changed[0]["candidate_logits"][2] += 0.5
        with self.assertRaisesRegex(AssertionError, "native raw"):
            terminal.validate_text_only(changed)

    def test_parent_review_receipt_identity_is_pinned(self):
        receipt = {
            "valid": True,
            "readOnly": True,
            "modelExpected": "claude-fable-5-1",
            "modelsObserved": ["claude-fable-5-1"],
            "effort": "medium",
            "arisFullSha": terminal.ARIS_FULL_SHA,
            "wrapperSha256": terminal.REVIEWER_WRAPPER_SHA256,
            "checkoutBefore": {"head": terminal.PARENT_COMMIT, "status": ""},
            "checkoutAfter": {"head": terminal.PARENT_COMMIT, "status": ""},
        }
        terminal.validate_parent_review(receipt)
        receipt["readOnly"] = False
        with self.assertRaisesRegex(AssertionError, "read-only"):
            terminal.validate_parent_review(receipt)


if __name__ == "__main__":
    unittest.main()
