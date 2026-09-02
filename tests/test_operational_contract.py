from __future__ import annotations

import os
import re
import stat
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OperationalContractTests(unittest.TestCase):
    def test_required_bootstrap_files_exist(self) -> None:
        required = [
            "AGENTS.md",
            "docs/AUTORESEARCH_STARTUP.md",
            "docs/RESEARCH_WRITING_AND_RECORDING.md",
            "docs/REMOTE_RESEARCH_OPERATIONS.md",
            "docs/RESEARCH_PLAN.md",
            "docs/RESEARCH_STATE.md",
            "docs/EXPERIMENT_REGISTRY.md",
            "config/aris-run-prompt.md",
            "config/aris-skills.txt",
            "config/aris-forbidden-skills.txt",
            "config/reviewer-routing.tsv",
            "scripts/start_aris.sh",
            "scripts/server_run.sh",
            "scripts/remote_run.sh",
            "scripts/remote_run.ps1",
            "scripts/fetch_results.sh",
            "scripts/fetch_results.ps1",
            "scripts/sync_overleaf.sh",
            "scripts/server/dispatch_run.sh",
            "scripts/server/run_command.sh",
            "scripts/server/supervisor.sh",
            "scripts/server/install_profile.sh",
        ]
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual([], missing)

    def test_server_scripts_fail_closed_and_never_use_default_tmux_socket(self) -> None:
        scripts = [
            ROOT / "scripts/start_aris.sh",
            ROOT / "scripts/server_run.sh",
            ROOT / "scripts/server/dispatch_run.sh",
            ROOT / "scripts/server/run_command.sh",
        ]
        for path in scripts:
            text = path.read_text(encoding="utf-8")
            self.assertIn("/home/qingchan", text, path)
            self.assertNotRegex(text, r"\bsudo\b", path)
            if "tmux" in text:
                self.assertRegex(text, r"tmux\s+-S\s+", path)

    def test_dispatch_contract_has_immutable_receipt_and_process_group(self) -> None:
        text = (ROOT / "scripts/server/dispatch_run.sh").read_text(encoding="utf-8")
        for token in (
            "metadata.env",
            "command.sh",
            "command_exit_status",
            "exit_status",
            "stdout.log",
            "stderr.log",
            "SHA256SUMS",
            "source",
            "setsid",
        ):
            self.assertIn(token, text)
        self.assertRegex(text, r"git\s+worktree\s+add|git\s+archive")
        self.assertIn("START_EPOCH", text)
        self.assertIn("GPU_COUNT", text)
        self.assertIn("ABORT_SIGNAL", text)
        self.assertIn("trap finalize EXIT", text)

    def test_start_gate_checks_logins_profile_reviewer_and_uses_codex_exec(self) -> None:
        text = (ROOT / "scripts/start_aris.sh").read_text(encoding="utf-8")
        for token in ("codex login status", "--strict-config", "--probe", "exec --profile"):
            self.assertIn(token, text)
        self.assertIn("supervisor.sh", text)

    def test_remote_dispatch_refreshes_git_and_uses_encoded_arguments(self) -> None:
        bash = (ROOT / "scripts/remote_run.sh").read_text(encoding="utf-8")
        server = (ROOT / "scripts/server_run.sh").read_text(encoding="utf-8")
        self.assertIn("git fetch origin", bash)
        self.assertIn("payload", bash)
        self.assertIn("git pull --ff-only", server)
        self.assertIn("new-window", server)

    def test_project_config_has_explicit_identity_and_safety_limits(self) -> None:
        text = (ROOT / "config/autoresearch.env").read_text(encoding="utf-8")
        expected = {
            "PROJECT_SLUG": "concept-flow",
            "SERVER_HOME": "/home/qingchan",
            "SERVER_REPO": "/home/qingchan/work/concept-flow",
            "DATA_ROOT": "/home/qingchan/data/concept-flow",
            "CONDA_ENV": "conceptflow",
            "PYTHON_VERSION": "3.13",
            "AGENT_TMUX": "concept-flow-aris",
            "EXPERIMENT_TMUX": "concept-flow-exp",
        }
        for key, value in expected.items():
            self.assertRegex(text, rf"(?m)^{key}={re.escape(value)}$")
        for key in (
            "MIN_FREE_DISK_GB",
            "MAX_CONSECUTIVE_FAILURES",
            "MAX_WALL_CLOCK_HOURS",
            "MAX_GPU_HOURS",
            "MAX_SINGLE_RUN_HOURS",
        ):
            self.assertRegex(text, rf"(?m)^{key}=.+$")

    def test_research_state_does_not_present_plans_as_observations(self) -> None:
        text = (ROOT / "docs/RESEARCH_STATE.md").read_text(encoding="utf-8")
        self.assertIn("PLANNED", text)
        self.assertIn("OBSERVED", text)
        self.assertIn("BLOCKED", text)
        self.assertIn("first hard gate", text.lower())
        self.assertLess(len(text.splitlines()), 80)

    def test_prompt_references_single_writing_contract(self) -> None:
        prompt = (ROOT / "config/aris-run-prompt.md").read_text(encoding="utf-8")
        self.assertIn("docs/RESEARCH_WRITING_AND_RECORDING.md", prompt)
        self.assertIn("docs/EXPERIMENT_REGISTRY.md", prompt)

    def test_reviewer_adapter_schema_cannot_override_identity_or_tools(self) -> None:
        adapter = ROOT / "scripts/server/claude_review_adapter.py"
        self.assertTrue(adapter.is_file())
        result = subprocess.run(
            [os.fspath(Path(os.sys.executable)), os.fspath(adapter), "--print-tool-schema"],
            check=True,
            capture_output=True,
            text=True,
        )
        schema = result.stdout
        self.assertIn('"prompt"', schema)
        for forbidden in ('"model"', '"tools"', '"effort"', '"permission_mode"'):
            self.assertNotIn(forbidden, schema)


if __name__ == "__main__":
    unittest.main()
