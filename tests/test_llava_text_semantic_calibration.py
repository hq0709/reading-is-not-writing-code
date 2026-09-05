from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src import llava_text_semantic_calibration as core
from src import run_llava_text_semantic_calibration as runner


def outcome(identity, raw):
    metadata = core.token_metadata(identity["encoding"])
    token_ids = metadata["candidate_ids"]
    logits = np.zeros(len(token_ids), dtype=np.float32)
    position = {token_id: i for i, token_id in enumerate(token_ids)}
    for token_id in metadata["raw_first"]:
        logits[position[token_id]] = np.float32(raw)
    return core.derive_record(identity, "rendered:" + identity["user_prompt"], logits, 10.0, token_ids)


def complete(correct=True):
    records = []
    for identity in core.cases():
        sign = 1.0 if identity["expected_present"] else -1.0
        raw = -sign if identity["encoding"] == "B" else sign
        if not correct and identity["condition"] == "candidate-A" and identity["case_id"].endswith("positive"):
            raw = -raw
        records.append(outcome(identity, raw))
    return records


def independently_expected_cases():
    objects = ("red square", "blue circle")
    pairs = (
        ("The scene contains a {object}.", "The scene contains no {object}."),
        ("A {object} is present.", "A {object} is absent."),
        (
            "It is false that the {object} is absent.",
            "It is false that the {object} is present.",
        ),
        (
            "The list is complete. Listed objects: {object}, green triangle.",
            "The list is complete. Listed objects: green triangle.",
        ),
    )
    conditions = (
        ("Y", "Y", False, "Answer yes or no."),
        (
            "incumbent-A",
            "A",
            False,
            "Answer A if the finding is present and B if it is absent. Reply with A or B only.",
        ),
        (
            "incumbent-B",
            "B",
            False,
            "Answer B if the finding is present and A if it is absent. Reply with A or B only.",
        ),
        (
            "candidate-A",
            "A",
            True,
            (
                "Choose the correct option.\nA. The finding is present.\n"
                "B. The finding is absent.\nReply with A or B only."
            ),
        ),
        (
            "candidate-B",
            "B",
            True,
            (
                "Choose the correct option.\nA. The finding is absent.\n"
                "B. The finding is present.\nReply with A or B only."
            ),
        ),
    )
    expected = []
    for object_index, object_name in enumerate(objects, start=1):
        prefix = f"The target finding is a {object_name}."
        for pair_index, pair in enumerate(pairs, start=1):
            for present, sentence in ((True, pair[0]), (False, pair[1])):
                description = f"{prefix} {sentence.format(object=object_name)}"
                polarity = "positive" if present else "negative"
                for condition, encoding, candidate, suffix in conditions:
                    expected.append(
                        {
                            "case_id": f"object-{object_index}-pair-{pair_index}-{polarity}",
                            "object": object_name,
                            "pair": pair_index,
                            "expected_present": present,
                            "expected_sign": 1 if present else -1,
                            "description": description,
                            "condition": condition,
                            "encoding": encoding,
                            "candidate": candidate,
                            "user_prompt": f"{description} Is the finding present? {suffix}",
                        }
                    )
    return expected


class ProtocolTests(unittest.TestCase):
    def test_cpu_import_and_help_do_not_load_torch(self):
        command = "import src.run_llava_text_semantic_calibration; import sys; assert 'torch' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-c", command], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [sys.executable, "-m", "src.run_llava_text_semantic_calibration", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_eighty_identities_order_and_literal_newlines(self):
        cases = core.cases()
        self.assertEqual(cases, independently_expected_cases())
        self.assertEqual(sum(row["expected_present"] for row in cases), 40)
        self.assertIn(
            "\nA. The finding is absent.\nB. The finding is present.\n",
            cases[4]["user_prompt"],
        )

    def test_collection_passes_only_identity_to_text_scorer(self):
        seen = []

        def score(identity):
            seen.append(identity)
            sign = 1.0 if identity["expected_present"] else -1.0
            raw = -sign if identity["encoding"] == "B" else sign
            return outcome(identity, raw)

        records = runner.collect(score)
        self.assertEqual(seen, core.cases())
        self.assertEqual(len(records), 80)

    def test_token_groups_and_semantic_reversal_are_exact(self):
        tokens = {row["name"]: core.token_metadata(row["encoding"]) for row in core.CONDITIONS}
        self.assertTrue(core.validate_token_metadata(tokens))
        identity = next(row for row in core.cases() if row["condition"] == "candidate-B")
        record = outcome(identity, -2.0)
        self.assertEqual(record["raw_margin"], -2.0)
        self.assertEqual(record["semantic_margin"], 2.0)
        tokens["candidate-B"]["raw_first"] = [1]
        with self.assertRaisesRegex(ValueError, "pinned singleton"):
            core.validate_token_metadata(tokens)

    def test_routes_are_strict_exhaustive_and_ignore_incumbent_success(self):
        records = complete()
        self.assertEqual(core.route(records), "candidate_eligible")
        failed_candidate = complete(correct=False)
        self.assertEqual(core.route(failed_candidate), "yes_no_route")
        tie_index = next(i for i, row in enumerate(records) if row["condition"] == "candidate-A")
        records[tie_index] = outcome(core.cases()[tie_index], 0.0)
        self.assertEqual(core.route(records), "yes_no_route")
        y_index = next(i for i, row in enumerate(records) if row["condition"] == "Y")
        records[y_index] = outcome(core.cases()[y_index], 0.0)
        self.assertEqual(core.route(records), "comprehension_unresolved")

    def test_y_logsumexp_requirement_is_independent_of_max_margin(self):
        records = complete()
        index = next(
            i
            for i, identity in enumerate(core.cases())
            if identity["condition"] == "Y" and identity["expected_present"]
        )
        identity = core.cases()[index]
        metadata = core.token_metadata("Y")
        positions = {token_id: i for i, token_id in enumerate(metadata["candidate_ids"])}
        logits = np.full(len(positions), -100.0, dtype=np.float32)
        logits[positions[metadata["raw_first"][0]]] = 1.0
        for token_id in metadata["raw_second"]:
            logits[positions[token_id]] = 0.0
        records[index] = core.derive_record(
            identity,
            "rendered:" + identity["user_prompt"],
            logits,
            10.0,
            metadata["candidate_ids"],
        )
        self.assertTrue(records[index]["primary_sign_correct"])
        self.assertFalse(records[index]["lse_sign_correct"])
        self.assertEqual(core.route(records), "comprehension_unresolved")

    def test_summary_reports_all_errors_pairs_and_replays_logits(self):
        records = complete(correct=False)
        summary = core.summarize(records)
        self.assertEqual(summary["route"], "yes_no_route")
        self.assertEqual(len(summary["conditions"]), 5)
        self.assertEqual(len(summary["paired_positive_minus_negative"]), 40)
        changed = [dict(row) for row in records]
        changed[0]["semantic_margin"] += 1
        with self.assertRaisesRegex(ValueError, "derived score"):
            core.validate_outcomes(changed)

    def test_json_round_trip_and_full_offline_replay(self):
        commit = "a" * 40
        source = {"model_id": "model", "model_revision": "revision"}
        tokens = {row["name"]: core.token_metadata(row["encoding"]) for row in core.CONDITIONS}
        records = complete()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "artifacts"
            out.mkdir()
            runner.write_json(
                out / runner.META,
                {
                    "source_commit": commit,
                    "protocol": runner.protocol(),
                    "source": source,
                    "token_ids": tokens,
                    "runtime": {"gpu_count": 1},
                },
            )
            runner.write_json(out / runner.OUTCOMES, records)
            runner.write_json(
                out / runner.SUMMARY,
                {
                    **core.summarize(records),
                    "source_commit": commit,
                    "elapsed_seconds": 1.0,
                    "protocol": runner.protocol(),
                    "source": source,
                    "token_ids": tokens,
                },
            )
            with patch.object(runner, "load_source", return_value=source):
                replayed = runner.replay(root, out, commit)
        self.assertEqual(replayed["route"], "candidate_eligible")

    def test_source_loader_uses_only_the_model_asset_receipt(self):
        source_text = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("input_verification", source_text)
        self.assertNotIn("datasets/", source_text)
        self.assertNotIn("manifest.csv", source_text)


if __name__ == "__main__":
    unittest.main()
