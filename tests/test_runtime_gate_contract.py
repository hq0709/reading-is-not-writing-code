from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def wsl_path(path: Path) -> str:
    if os.name != "nt":
        return os.fspath(path.resolve())
    return subprocess.check_output(
        ["bash", "-lc", f"wslpath -a {shlex_quote(os.fspath(path))}"], text=True
    ).strip()


def shlex_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


class RuntimeGateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.repo = self.home / "work/concept-flow"
        (self.repo / "scripts/server").mkdir(parents=True)
        (self.repo / "config").mkdir()
        (self.repo / "scripts/server/supervisor.sh").write_text(
            (ROOT / "scripts/server/supervisor.sh")
            .read_text(encoding="utf-8")
            .replace(
                "readonly AUTHORIZED_HOME=/home/qingchan",
                "readonly AUTHORIZED_HOME=${HOME:?}",
            ),
            encoding="utf-8",
            newline="\n",
        )
        self.home_wsl = wsl_path(self.home)
        self.repo_wsl = wsl_path(self.repo)
        self.data_wsl = f"{self.home_wsl}/data/concept-flow"
        self.aris_wsl = f"{self.home_wsl}/aris_repo"
        self.socket_wsl = f"{self.home_wsl}/.cache/tmux/concept-flow.sock"
        config = f"""SERVER_HOME={self.home_wsl}
DATA_ROOT={self.data_wsl}
ARIS_REPO={self.aris_wsl}
CONDA_ROOT={self.home_wsl}/miniforge3
CONDA_ENV=conceptflow
TMUX_SOCKET={self.socket_wsl}
AGENT_TMUX=concept-flow-aris
EXPERIMENT_TMUX=concept-flow-exp
CODEX_PROFILE=concept-flow
CODEX_MODEL=gpt-5.6-sol
REVIEW_TRANSPORT=claude-review-concept-flow
CLAUDE_REVIEW_MODEL=claude-fable-5-1
STOP_SENTINEL={self.data_wsl}/STOP
SUPERVISOR_INTERVAL_SECONDS=1
MIN_FREE_DISK_GB=0
MAX_CONSECUTIVE_FAILURES=3
MAX_WALL_CLOCK_HOURS=1
MAX_GPU_HOURS=1
MAX_API_SPEND_USD=0
MAX_SINGLE_RUN_HOURS=24
ARIS_FULL_SHA=94d8093ed21d20a790830318190095b9f5036ce8
"""
        (self.repo / "config/autoresearch.env").write_text(config, encoding="utf-8", newline="\n")
        setup = f"""
set -e
mkdir -p {shlex_quote(self.data_wsl)}/state {shlex_quote(self.data_wsl)}/runs {shlex_quote(self.aris_wsl)}
date +%s > {shlex_quote(self.data_wsl)}/state/bootstrap_started_epoch
git -C {shlex_quote(self.repo_wsl)} init -q
git -C {shlex_quote(self.repo_wsl)} config user.email test@example.com
git -C {shlex_quote(self.repo_wsl)} config user.name test
git -C {shlex_quote(self.repo_wsl)} add .
git -C {shlex_quote(self.repo_wsl)} commit -qm initial
"""
        subprocess.run(["bash", "-lc", setup], check=True)

    def run_supervisor(self) -> subprocess.CompletedProcess[str]:
        command = (
            f"cd {shlex_quote(self.repo_wsl)} && "
            f"HOME={shlex_quote(self.home_wsl)} bash scripts/server/supervisor.sh --once"
        )
        return subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )

    def install_recovery_harness(
        self, *, launcher_succeeds: bool
    ) -> tuple[Path, Path, Path]:
        fake_bin = self.home / "recovery-bin"
        fake_bin.mkdir()
        launcher_log = self.home / "launcher.log"
        agent_live = self.home / "agent-live"
        clock = self.home / "recovery-clock"
        clock.write_text(str(int(time.time()) + 60) + "\n", encoding="utf-8", newline="\n")

        date = fake_bin / "date"
        date.write_text(
            f"""#!/usr/bin/env bash
if test "$*" = +%s; then cat {shlex_quote(wsl_path(clock))}; else exec /usr/bin/date "$@"; fi
""",
            encoding="utf-8",
            newline="\n",
        )

        git = fake_bin / "git"
        git.write_text(
            f"""#!/usr/bin/env bash
case "$*" in
  'rev-parse --show-toplevel') echo {self.repo_wsl} ;;
  'status --porcelain --untracked-files=all'|'fetch origin --quiet') ;;
  'rev-parse HEAD'|'rev-parse @{{upstream}}') printf '%040d\n' 1 ;;
  -C*' rev-parse HEAD') echo 94d8093ed21d20a790830318190095b9f5036ce8 ;;
esac
""",
            encoding="utf-8",
            newline="\n",
        )
        tmux = fake_bin / "tmux"
        tmux.write_text(
            f"""#!/usr/bin/env bash
case "$*" in
  *'has-session -t concept-flow-aris'*) test -e {shlex_quote(wsl_path(agent_live))} ;;
  *'has-session -t concept-flow-exp'*) exit 0 ;;
  *'list-windows -t concept-flow-exp'*) printf 'control 1\n' ;;
  *) exit 64 ;;
esac
""",
            encoding="utf-8",
            newline="\n",
        )
        codex = self.home / ".local/bin/codex"
        codex.parent.mkdir(parents=True)
        codex.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")

        adapter = self.repo / "scripts/server/claude_review_adapter.py"
        adapter.write_text("# reviewer\n", encoding="utf-8", newline="\n")
        for relative in (
            "config/aris-skills.txt",
            "config/aris-forbidden-skills.txt",
            "config/reviewer-routing.tsv",
        ):
            (self.repo / relative).write_text(relative + "\n", encoding="utf-8", newline="\n")

        def digest(relative: str) -> str:
            return hashlib.sha256((self.repo / relative).read_bytes()).hexdigest()

        audit = self.home / "data/concept-flow/state/aris-audit.env"
        audit.write_text(
            "\n".join(
                [
                    "ARIS_FULL_SHA=94d8093ed21d20a790830318190095b9f5036ce8",
                    f"SKILL_LIST_SHA256={digest('config/aris-skills.txt')}",
                    f"FORBIDDEN_LIST_SHA256={digest('config/aris-forbidden-skills.txt')}",
                    f"REVIEWER_ROUTING_SHA256={digest('config/reviewer-routing.tsv')}",
                    f"REVIEWER_WRAPPER_SHA256={digest('scripts/server/claude_review_adapter.py')}",
                ]
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        receipt_dir = self.home / ".codex/state/claude-review-concept-flow"
        receipt_dir.mkdir(parents=True)
        receipt_dir.joinpath("review-recovery.json").write_text(
            """{"valid":true,"readOnly":true,"modelsObserved":["claude-fable-5-1"],
"arisFullSha":"94d8093ed21d20a790830318190095b9f5036ce8",
"wrapperSha256":"%s"}\n""" % digest("scripts/server/claude_review_adapter.py"),
            encoding="utf-8",
            newline="\n",
        )
        interpreter = self.home / "miniforge3/envs/conceptflow/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text(
            '#!/usr/bin/env bash\nexec /usr/bin/python3 "$@"\n', encoding="utf-8", newline="\n"
        )
        launcher = self.repo / "scripts/start_aris.sh"
        launcher.parent.mkdir(exist_ok=True)
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' \"$0\" >> {shlex_quote(wsl_path(launcher_log))}\n"
            + (f"touch {shlex_quote(wsl_path(agent_live))}\nexit 0\n" if launcher_succeeds else "exit 64\n"),
            encoding="utf-8",
            newline="\n",
        )
        setup = "chmod 700 " + " ".join(
            shlex_quote(wsl_path(path))
            for path in (date, git, tmux, codex, interpreter, launcher)
        )
        subprocess.run(["bash", "-lc", setup], check=True)
        return fake_bin, launcher_log, agent_live

    def run_recovery_supervisor(
        self, fake_bin: Path, *, once: bool = True
    ) -> subprocess.CompletedProcess[str] | subprocess.Popen[str]:
        once_arg = " --once" if once else ""
        command = (
            f"cd {shlex_quote(self.repo_wsl)} && HOME={shlex_quote(self.home_wsl)} "
            f"PATH={shlex_quote(wsl_path(fake_bin))}:/usr/bin:/bin "
            f"bash scripts/server/supervisor.sh{once_arg}"
        )
        if not once:
            return subprocess.Popen(
                ["bash", "-lc", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        return subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )

    def test_expired_wall_budget_blocks_dispatch_without_killing_active_work(self) -> None:
        old_epoch = str(1)
        (self.home / "data/concept-flow/state/bootstrap_started_epoch").write_text(
            old_epoch + "\n", encoding="utf-8"
        )
        active = self.home / "data/concept-flow/runs/active"
        active.mkdir(parents=True)
        marker = active / "still-running"
        marker.write_text("alive\n", encoding="utf-8")

        result = self.run_supervisor()

        self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertTrue((self.home / "data/concept-flow/STOP").is_file())
        self.assertTrue(marker.is_file(), "supervisor must not terminate or delete active work")
        health = (self.home / "data/concept-flow/state/health.env").read_text(encoding="utf-8")
        self.assertIn("WALL_CLOCK_OK=0", health)

    def test_each_dispatch_budget_is_reported_as_closed(self) -> None:
        state = self.home / "data/concept-flow/state"
        runs = self.home / "data/concept-flow/runs"
        state.joinpath("bootstrap_started_epoch").write_text(
            str(int(time.time())) + "\n", encoding="utf-8"
        )
        cases = {
            "disk": ("MIN_FREE_DISK_GB=0", "MIN_FREE_DISK_GB=999999999", "DISK_OK=0"),
            "gpu": (None, None, "GPU_BUDGET_OK=0"),
            "failures": (None, None, "FAILURE_BUDGET_OK=0"),
        }
        for name, (old, new, expected) in cases.items():
            with self.subTest(name=name):
                if old:
                    config_path = self.repo / "config/autoresearch.env"
                    original = config_path.read_text(encoding="utf-8")
                    config_path.write_text(original.replace(old, new), encoding="utf-8", newline="\n")
                if name == "gpu":
                    run = runs / "gpu"
                    run.mkdir(parents=True, exist_ok=True)
                    run.joinpath("metadata.env").write_text(
                        "ELAPSED_SECONDS=3600\nGPU_COUNT=1\n", encoding="utf-8"
                    )
                if name == "failures":
                    for index in range(3):
                        run = runs / f"failure-{index}"
                        run.mkdir(parents=True, exist_ok=True)
                        run.joinpath("command_exit_status").write_text("1\n", encoding="utf-8")

                result = self.run_supervisor()
                self.assertTrue(state.joinpath("health.env").is_file(), result.stdout + result.stderr)
                health = state.joinpath("health.env").read_text(encoding="utf-8")
                self.assertIn(expected, health)

                if old:
                    config_path.write_text(original, encoding="utf-8", newline="\n")
                shutil.rmtree(runs)
                runs.mkdir()

    def test_start_refuses_launch_when_runtime_gate_fails(self) -> None:
        start = self.repo / "scripts/start_aris.sh"
        start.write_text(
            (ROOT / "scripts/start_aris.sh")
            .read_text(encoding="utf-8")
            .replace("readonly AUTHORIZED_HOME=/home/qingchan", "readonly AUTHORIZED_HOME=${HOME:?}"),
            encoding="utf-8",
            newline="\n",
        )
        start.write_text(
            start.read_text(encoding="utf-8").replace(
                "/home/qingchan/.local/bin/codex", f"{self.home_wsl}/.local/bin/codex"
            ),
            encoding="utf-8",
            newline="\n",
        )
        fake_bin = self.home / "fake-bin"
        fake_bin.mkdir()
        git = fake_bin / "git"
        git.write_text(
            f"""#!/usr/bin/env bash
if test "${{1:-}}" = -C; then echo 94d8093ed21d20a790830318190095b9f5036ce8; exit 0; fi
case "$*" in
  'rev-parse --show-toplevel') echo {self.repo_wsl} ;;
  'status --porcelain --untracked-files=all') ;;
  'fetch origin') ;;
  'rev-parse HEAD'|'rev-parse @{{upstream}}') printf '%040d\n' 1 ;;
esac
""",
            encoding="utf-8",
            newline="\n",
        )
        tmux_log = self.home / "tmux.log"
        tmux = fake_bin / "tmux"
        tmux.write_text(
            f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> {shlex_quote(wsl_path(tmux_log))}\n",
            encoding="utf-8",
            newline="\n",
        )
        stat = fake_bin / "stat"
        stat.write_text("#!/usr/bin/env bash\necho 600\n", encoding="utf-8", newline="\n")
        python = fake_bin / "python"
        python.write_text("#!/usr/bin/env bash\nexec /usr/bin/python3 \"$@\"\n", encoding="utf-8", newline="\n")
        codex = self.home / ".local/bin/codex"
        codex.parent.mkdir(parents=True)
        codex.write_text(
            """#!/usr/bin/env bash
case "$*" in
  'plugin list') echo 'No marketplace plugins found.' ;;
  *'mcp list --json') echo '[{"name":"claude-review-concept-flow","enabled":true}]' ;;
esac
""",
            encoding="utf-8",
            newline="\n",
        )
        profile = self.home / ".codex/concept-flow.config.toml"
        profile.parent.mkdir()
        profile.write_text('model = "gpt-5.6-sol"\nmodel_reasoning_effort = "high"\n', encoding="utf-8")
        activate = self.repo / "scripts/server/activate_env.sh"
        activate.write_text("#!/usr/bin/env bash\n", encoding="utf-8", newline="\n")
        adapter = self.repo / "scripts/server/claude_review_adapter.py"
        adapter.write_text("#!/usr/bin/env python3\nprint('READY')\n", encoding="utf-8", newline="\n")
        for relative in ("config/aris-skills.txt", "config/aris-forbidden-skills.txt", "config/reviewer-routing.tsv"):
            (self.repo / relative).write_text(relative + "\n", encoding="utf-8")
        def digest(relative: str) -> str:
            return hashlib.sha256((self.repo / relative).read_bytes()).hexdigest()

        audit = self.home / "data/concept-flow/state/aris-audit.env"
        audit.write_text(
            "\n".join(
                [
                    "ARIS_FULL_SHA=94d8093ed21d20a790830318190095b9f5036ce8",
                    f"SKILL_LIST_SHA256={digest('config/aris-skills.txt')}",
                    f"FORBIDDEN_LIST_SHA256={digest('config/aris-forbidden-skills.txt')}",
                    f"REVIEWER_ROUTING_SHA256={digest('config/reviewer-routing.tsv')}",
                    f"REVIEWER_WRAPPER_SHA256={digest('scripts/server/claude_review_adapter.py')}",
                ]
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        gate = self.repo / "scripts/server/supervisor.sh"
        gate.write_text("#!/usr/bin/env bash\nexit 64\n", encoding="utf-8", newline="\n")
        python_path = self.home / "miniforge3/envs/conceptflow/bin/python"
        python_path.parent.mkdir(parents=True)
        setup = f"""
chmod 700 {shlex_quote(wsl_path(git))} {shlex_quote(wsl_path(tmux))} {shlex_quote(wsl_path(stat))} {shlex_quote(wsl_path(python))} {shlex_quote(wsl_path(codex))}
chmod 700 {shlex_quote(wsl_path(start))} {shlex_quote(wsl_path(gate))}
chmod 600 {shlex_quote(wsl_path(profile))}
ln -s /usr/bin/python3 {shlex_quote(wsl_path(python_path))}
"""
        subprocess.run(["bash", "-lc", setup], check=True)

        command = (
            f"cd {shlex_quote(self.repo_wsl)} && HOME={shlex_quote(self.home_wsl)} "
            f"PATH={shlex_quote(wsl_path(fake_bin))}:/usr/bin:/bin bash scripts/start_aris.sh"
        )
        result = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=10)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("runtime health gate failed", result.stderr)
        self.assertFalse(tmux_log.exists(), "no tmux session may launch after a failed runtime gate")

        stop = self.home / "data/concept-flow/STOP"
        stop.write_text("AUTORESEARCH_SUPERVISOR_BLOCK_V1\n", encoding="utf-8", newline="\n")
        gate.write_text(
            f"#!/usr/bin/env bash\nrm -f {shlex_quote(wsl_path(stop))}\nexit 0\n",
            encoding="utf-8",
            newline="\n",
        )
        recovered = subprocess.run(
            ["bash", "-lc", command], capture_output=True, text=True, encoding="utf-8", timeout=10
        )
        self.assertEqual(0, recovered.returncode, recovered.stdout + recovered.stderr)
        self.assertFalse(stop.exists())

    def test_manual_stop_and_supervisor_singleton_are_preserved(self) -> None:
        stop = self.home / "data/concept-flow/STOP"
        stop.write_text("operator requested stop\n", encoding="utf-8", newline="\n")
        self.run_supervisor()
        self.assertEqual("operator requested stop\n", stop.read_text(encoding="utf-8"))

        command = (
            f"cd {shlex_quote(self.repo_wsl)} && HOME={shlex_quote(self.home_wsl)} "
            "bash scripts/server/supervisor.sh"
        )
        daemon = subprocess.Popen(
            ["bash", "-lc", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            time.sleep(1)
            second = self.run_supervisor()
            self.assertEqual(64, second.returncode)
            self.assertIn("supervisor already running", second.stderr)
        finally:
            daemon.terminate()
            daemon.communicate(timeout=5)

    def test_recovery_never_bypasses_user_stop_or_failed_gate(self) -> None:
        fake_bin, launcher_log, _ = self.install_recovery_harness(launcher_succeeds=True)
        bootstrap = self.home / "data/concept-flow/state/bootstrap_started_epoch"
        bootstrap.write_text("1\n", encoding="utf-8", newline="\n")

        failed_gate = self.run_recovery_supervisor(fake_bin)
        self.assertNotEqual(0, failed_gate.returncode)
        self.assertFalse(launcher_log.exists(), "failed non-agent gates must suppress recovery")

        bootstrap.write_text(str(int(time.time())) + "\n", encoding="utf-8", newline="\n")
        stop = self.home / "data/concept-flow/STOP"
        stop.write_text("operator requested stop\n", encoding="utf-8", newline="\n")
        user_stop = self.run_recovery_supervisor(fake_bin)
        self.assertNotEqual(0, user_stop.returncode)
        self.assertFalse(launcher_log.exists(), "a user stop must suppress recovery")
        self.assertEqual("operator requested stop\n", stop.read_text(encoding="utf-8"))

    def test_recovery_backoff_prevents_restart_storm(self) -> None:
        fake_bin, launcher_log, _ = self.install_recovery_harness(launcher_succeeds=False)
        first = self.run_recovery_supervisor(fake_bin)
        second = self.run_recovery_supervisor(fake_bin)

        attempts = launcher_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(1, len(attempts), first.stdout + first.stderr)
        self.assertNotEqual(0, second.returncode)
        health = (self.home / "data/concept-flow/state/health.env").read_text(encoding="utf-8")
        self.assertIn("RECOVERY_BACKOFF_ACTIVE=1", health)

    def test_recovery_backoff_survives_transient_agent_exit(self) -> None:
        fake_bin, launcher_log, agent_live = self.install_recovery_harness(launcher_succeeds=True)

        first = self.run_recovery_supervisor(fake_bin)
        agent_live.unlink()
        second = self.run_recovery_supervisor(fake_bin)

        self.assertEqual(0, first.returncode, first.stdout + first.stderr)
        self.assertNotEqual(0, second.returncode)
        self.assertEqual(1, len(launcher_log.read_text(encoding="utf-8").splitlines()))
        health = (self.home / "data/concept-flow/state/health.env").read_text(encoding="utf-8")
        self.assertIn("RECOVERY_BACKOFF_ACTIVE=1", health)

    def test_recovery_backoff_resets_after_stable_agent_window(self) -> None:
        fake_bin, _, _ = self.install_recovery_harness(launcher_succeeds=True)
        clock = self.home / "recovery-clock"

        first = self.run_recovery_supervisor(fake_bin)
        now = int(clock.read_text(encoding="utf-8"))
        clock.write_text(str(now + 120) + "\n", encoding="utf-8", newline="\n")
        stable = self.run_recovery_supervisor(fake_bin)

        self.assertEqual(0, first.returncode, first.stdout + first.stderr)
        self.assertEqual(0, stable.returncode, stable.stdout + stable.stderr)
        recovery = (self.home / "data/concept-flow/state/recovery.env").read_text(
            encoding="utf-8"
        )
        self.assertIn("RECOVERY_BACKOFF_SECONDS=1", recovery)
        self.assertIn("RECOVERY_NEXT_EPOCH=0", recovery)

    def test_recovery_uses_normal_launcher_and_managed_stop(self) -> None:
        fake_bin, launcher_log, agent_live = self.install_recovery_harness(launcher_succeeds=True)
        stop = self.home / "data/concept-flow/STOP"
        stop.write_text("AUTORESEARCH_SUPERVISOR_BLOCK_V1\n", encoding="utf-8", newline="\n")

        result = self.run_recovery_supervisor(fake_bin)

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertTrue(agent_live.exists())
        self.assertEqual(
            [f"{self.repo_wsl}/scripts/start_aris.sh"],
            launcher_log.read_text(encoding="utf-8").splitlines(),
        )
        self.assertFalse(stop.exists(), "healthy recovery must clear only the managed sentinel")

    def test_reviewer_health_uses_fresh_structured_receipt_and_pinned_python(self) -> None:
        adapter = self.repo / "scripts/server/claude_review_adapter.py"
        adapter.write_text("# reviewer\n", encoding="utf-8", newline="\n")
        for relative in ("config/aris-skills.txt", "config/aris-forbidden-skills.txt", "config/reviewer-routing.tsv"):
            (self.repo / relative).write_text(relative + "\n", encoding="utf-8", newline="\n")
        interpreter = self.home / "miniforge3/envs/conceptflow/bin/python"
        interpreter.parent.mkdir(parents=True)
        marker = self.home / "pinned-python-used"
        interpreter.write_text(
            f"#!/usr/bin/env bash\ntouch {shlex_quote(wsl_path(marker))}\nexec /usr/bin/python3 \"$@\"\n",
            encoding="utf-8",
            newline="\n",
        )
        receipt_dir = self.home / ".codex/state/claude-review-concept-flow"
        receipt_dir.mkdir(parents=True)
        receipt = receipt_dir / "review-test.json"
        receipt.write_text(
            """{"valid":true,"readOnly":true,"modelsObserved":["claude-fable-5-1"],
"arisFullSha":"94d8093ed21d20a790830318190095b9f5036ce8",
"wrapperSha256":"%s"}\n""" % hashlib.sha256(adapter.read_bytes()).hexdigest(),
            encoding="utf-8",
            newline="\n",
        )
        subprocess.run(["bash", "-lc", f"chmod 700 {shlex_quote(wsl_path(interpreter))}"], check=True)

        self.run_supervisor()

        health = (self.home / "data/concept-flow/state/health.env").read_text(encoding="utf-8")
        self.assertIn("REVIEWER_RECEIPT_OK=1", health)
        self.assertTrue(marker.exists(), "receipt validation must use the pinned environment Python")

    def test_first_gate_rejects_run_dir_not_bound_to_its_archived_source(self) -> None:
        actual_run = self.home / "data/concept-flow/runs/actual"
        source = actual_run / "source"
        runner = source / "scripts/server/run_first_gate.sh"
        runner.parent.mkdir(parents=True)
        runner.write_text(
            (ROOT / "scripts/server/run_first_gate.sh")
            .read_text(encoding="utf-8")
            .replace("readonly AUTHORIZED_HOME=/home/qingchan", "readonly AUTHORIZED_HOME=${HOME:?}"),
            encoding="utf-8",
            newline="\n",
        )
        substituted_run = self.home / "data/concept-flow/runs/substituted"
        substituted_run.mkdir(parents=True)
        command = (
            f"HOME={shlex_quote(self.home_wsl)} "
            f"RUN_DIR={shlex_quote(wsl_path(substituted_run))} "
            f"bash {shlex_quote(wsl_path(runner))} hook"
        )

        result = subprocess.run(
            ["bash", "-lc", command], capture_output=True, text=True, encoding="utf-8", timeout=10
        )

        self.assertEqual(64, result.returncode, result.stdout + result.stderr)
        self.assertIn("RUN_DIR does not match archived source", result.stderr)


if __name__ == "__main__":
    unittest.main()
