from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fetch_results.ps1"


class FetchResultsContractTests(unittest.TestCase):
    def test_missing_exit_status_fails_closed_before_atomic_move(self) -> None:
        pwsh = shutil.which("pwsh")
        self.assertIsNotNone(pwsh, "pwsh is required for the Windows fetch contract test")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            repo = temp_root / "repo"
            fixture = temp_root / "fixture"
            bin_dir = temp_root / "bin"
            (repo / "scripts").mkdir(parents=True)
            fixture.mkdir()
            bin_dir.mkdir()
            shutil.copy2(SCRIPT, repo / "scripts/fetch_results.ps1")

            metadata = b"RUN_ID=missing-exit-status\n"
            (fixture / "metadata.env").write_bytes(metadata)
            digest = hashlib.sha256(metadata).hexdigest()
            (fixture / "SHA256SUMS").write_text(
                f"{digest}  ./metadata.env\n", encoding="utf-8"
            )
            (bin_dir / "scp.cmd").write_text(
                "@echo off\r\n"
                'xcopy "%FETCH_RESULTS_FIXTURE%\\*" "%~5" /E /I /Y >nul\r\n'
                "exit /b %ERRORLEVEL%\r\n",
                encoding="ascii",
            )
            subprocess.run(
                ["git", "init", "--quiet", os.fspath(repo)], check=True
            )
            env = os.environ.copy()
            env["FETCH_RESULTS_FIXTURE"] = os.fspath(fixture)
            env["PATH"] = os.pathsep.join((os.fspath(bin_dir), env["PATH"]))

            result = subprocess.run(
                [
                    pwsh,
                    "-NoProfile",
                    "-NonInteractive",
                    "-File",
                    os.fspath(repo / "scripts/fetch_results.ps1"),
                    "missing-exit-status",
                ],
                cwd=repo,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertIn("exit_status missing", result.stderr)
            results = repo / "results/remote"
            self.assertFalse((results / "missing-exit-status").exists())
            self.assertEqual([], list(results.glob(".partial-*")))


if __name__ == "__main__":
    unittest.main()
