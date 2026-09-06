from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reviewer_model_routing_adapter", ROOT / "scripts/server/claude_review_adapter.py"
)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
FABLE = "claude-fable-5-1"
OPUS = "claude-opus-5"


class ReviewerModelRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patches = ExitStack()
        self.addCleanup(patches.close)
        home = Path(temporary.name).resolve()
        self.state = home / "receipts"
        for name, value in {
            "HOME": home,
            "REPO": home / "repo",
            "CLAUDE": home / "claude",
            "STATE": self.state,
        }.items():
            patches.enter_context(mock.patch.object(adapter, name, value))
        patches.enter_context(mock.patch.object(Path, "home", return_value=home))
        self.paths = patches.enter_context(mock.patch.object(adapter, "inside_home", return_value=True))
        self.git = patches.enter_context(mock.patch.object(
            adapter.subprocess, "check_output", return_value=adapter.ARIS_SHA + "\n"
        ))
        self.before = {"head": "a" * 40, "status": ""}
        self.fingerprint = patches.enter_context(mock.patch.object(
            adapter, "git_fingerprint", side_effect=[self.before, dict(self.before)]
        ))
        self.run = patches.enter_context(mock.patch.object(adapter.subprocess, "run"))

    def review(self, models: list[str], *, exit_code: int = 0, valid: bool = True) -> dict:
        self.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=exit_code,
            stdout=json.dumps({
                "modelUsage": {str(i): {"canonicalModel": model} for i, model in enumerate(models)},
                "result": "READY",
            }), stderr="",
        )
        if valid:
            self.assertEqual(adapter.run_review("Review evidence."), "READY")
        else:
            with self.assertRaisesRegex(RuntimeError, "review identity/read-only gate failed"):
                adapter.run_review("Review evidence.")
        receipts = list(self.state.glob("review-*.json"))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
        self.assertIs(receipt["valid"], valid)
        self.assertEqual(receipt["modelsObserved"], sorted(models))
        self.assertEqual(receipt["modelsAccepted"], sorted([FABLE, OPUS]))
        self.assertEqual(receipt["modelExpected"], FABLE)
        self.assertEqual(receipt["exitCode"], exit_code)
        self.git.assert_called_once_with(
            ["git", "-C", os.fspath(adapter.HOME / "aris_repo"), "rev-parse", "HEAD"], text=True
        )
        self.assertEqual(self.fingerprint.call_count, 2)
        self.assertEqual(self.paths.call_args_list, [
            mock.call(adapter.REPO), mock.call(adapter.CLAUDE), mock.call(adapter.HOME / "aris_repo")
        ])
        return receipt

    def test_accepts_fable(self) -> None:
        self.review([FABLE])

    def test_accepts_opus_preserving_observation_and_fixed_transport(self) -> None:
        receipt = self.review([OPUS])
        argv = [
            os.fspath(adapter.CLAUDE), "-p", "--name", "pinned-review",
            "--no-session-persistence", "--model", FABLE, "--effort", "medium",
            "--permission-mode", "plan", "--tools", "", "--disallowedTools", "mcp__*",
            "--safe-mode", "--output-format", "json", "Review evidence.",
        ]
        pins = {
            "CLAUDE_REVIEW_MODEL": FABLE,
            "CLAUDE_CODE_EFFORT_LEVEL": "medium",
            "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        }
        self.run.assert_called_once_with(
            argv, cwd=adapter.REPO, text=True, capture_output=True, timeout=1800,
            check=False, env={**os.environ, **pins},
        )
        self.assertEqual(receipt["argvWithoutPrompt"], argv[:-1])
        self.assertEqual(receipt["environmentPins"], pins)
        self.assertEqual(receipt["effort"], "medium")
        self.assertIs(receipt["readOnly"], True)

    def test_accepts_both_allowed_models(self) -> None:
        self.review([FABLE, OPUS])

    def test_rejects_unknown_model(self) -> None:
        self.review(["unknown-model"], valid=False)

    def test_rejects_unknown_alongside_allowed_model(self) -> None:
        self.review([OPUS, "unknown-model"], valid=False)

    def test_rejects_empty_models(self) -> None:
        self.review([], valid=False)

    def test_rejects_changed_checkout_with_allowed_model(self) -> None:
        for after in ({"head": "b" * 40, "status": ""},
                      {"head": self.before["head"], "status": " M tracked.txt"}):
            with self.subTest(after=after):
                self.fingerprint.side_effect = [self.before, after]
                self.fingerprint.reset_mock()
                self.git.reset_mock()
                self.paths.reset_mock()
                receipt = self.review([OPUS], valid=False)
                self.assertIs(receipt["readOnly"], False)
                self.assertEqual(receipt["checkoutBefore"], self.before)
                self.assertEqual(receipt["checkoutAfter"], after)
                for path in self.state.glob("review-*.json"):
                    path.chmod(0o600)
                    path.unlink()

    def test_rejects_nonzero_exit_with_allowed_model(self) -> None:
        receipt = self.review([OPUS], exit_code=1, valid=False)
        self.assertIs(receipt["readOnly"], True)


class SupervisorReceiptRoutingTests(unittest.TestCase):
    def test_live_receipt_predicate(self) -> None:
        source = (ROOT / "scripts/server/supervisor.sh").read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY", source, flags=re.DOTALL)
        predicates = [block for block in blocks if 'receipt.get("modelsObserved")' in block
                      or 'receipt.get("modelsObserved",' in block]
        self.assertEqual(len(predicates), 1)
        predicate = compile(predicates[0], "supervisor-receipt-predicate", "exec")
        cases = [
            ([FABLE], {}, True), ([OPUS], {}, True), ([FABLE, OPUS], {}, True),
            ([], {}, False), (["unknown-model"], {}, False),
            ([OPUS, "unknown-model"], {}, False),
            ([OPUS], {"valid": False}, False), ([OPUS], {"readOnly": False}, False),
            ([OPUS], {"arisFullSha": "wrong"}, False),
            ([OPUS], {"wrapperSha256": "wrong"}, False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            for models, overrides, accepted in cases:
                with self.subTest(models=models, overrides=overrides):
                    path.write_text(json.dumps({
                        "valid": True, "readOnly": True, "modelsObserved": models,
                        "arisFullSha": adapter.ARIS_SHA, "wrapperSha256": "wrapper-pin",
                        **overrides,
                    }), encoding="utf-8")
                    with mock.patch("sys.argv", [
                        "-", str(path), FABLE, adapter.ARIS_SHA, "wrapper-pin"
                    ]):
                        if accepted:
                            exec(predicate, {})
                        else:
                            with self.assertRaises(AssertionError):
                                exec(predicate, {})


if __name__ == "__main__":
    unittest.main()
