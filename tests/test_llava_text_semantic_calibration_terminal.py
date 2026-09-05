import unittest

import numpy as np

from scripts.server import validate_llava_text_semantic_calibration as validator
from src import llava_text_semantic_calibration as core


class TerminalTests(unittest.TestCase):
    def test_independent_case_count_order_and_newlines(self):
        cases = validator.expected_cases()
        self.assertEqual(cases, core.cases())
        self.assertIn("\nA. The finding is present.\n", cases[3]["user_prompt"])
        self.assertIn("\nA. The finding is absent.\n", cases[4]["user_prompt"])

    def test_terminal_route_ignores_incumbents(self):
        rows = []
        for identity in validator.expected_cases():
            rows.append(
                {
                    **identity,
                    "primary_sign_correct": identity["condition"] != "candidate-A",
                    "lse_sign_correct": True if identity["condition"] == "Y" else None,
                }
            )
        self.assertEqual(validator.terminal_route(rows), "yes_no_route")
        rows[0]["primary_sign_correct"] = False
        self.assertEqual(validator.terminal_route(rows), "comprehension_unresolved")

    def test_impossible_candidate_partition_is_rejected(self):
        identity = validator.expected_cases()[0]
        token_ids = sorted(set(validator.TOKENS["Y"][0] + validator.TOKENS["Y"][1]))
        record = {
            **identity,
            "rendered_prompt": "rendered",
            "candidate_token_ids": token_ids,
            "candidate_logits": np.zeros(len(token_ids), dtype=np.float32).tolist(),
            "log_partition": 0.0,
        }
        with self.assertRaisesRegex(AssertionError, "answer-token mass"):
            validator.replay_record(record, identity)


if __name__ == "__main__":
    unittest.main()
