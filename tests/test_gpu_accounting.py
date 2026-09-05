from __future__ import annotations

import itertools
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt":
    git = shutil.which("git")
    BASH = str(Path(git).resolve().parents[1] / "bin/bash.exe") if git else None
else:
    BASH = shutil.which("bash")


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash is required")
class GpuAccountingTests(unittest.TestCase):
    def assert_gpu_seconds(self, metadata: list[str], expected: int) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            for index, content in enumerate(metadata):
                run = runs / f"run-{index}"
                run.mkdir()
                (run / "metadata.env").write_text(content, encoding="utf-8", newline="\n")
            for name in ("supervisor.sh", "dispatch_run.sh"):
                with self.subTest(script=name):
                    source = (ROOT / "scripts/server" / name).read_text(encoding="utf-8")
                    assignment = re.search(r"gpu_seconds=\$\(.*?\)\n", source, re.DOTALL)
                    self.assertIsNotNone(assignment)
                    result = subprocess.run(
                        [BASH, "--noprofile", "--norc", "-c",
                         'set -euo pipefail\nruns_dir="$1"\n' + assignment.group()
                         + 'printf "%s\\n" "${gpu_seconds:-0}"',
                         "gpu-accounting", runs.as_posix()],
                        check=True, capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(expected, int(result.stdout))

    def test_mixed_cpu_and_gpu_runs_are_independent_of_file_order(self) -> None:
        metadata = [
            "RUN_ID=cpu\nGPU_COUNT=0\nELAPSED_SECONDS=20\nCLEANUP_STATUS=0\n",
            "RUN_ID=one-gpu\nGPU_COUNT=1\nELAPSED_SECONDS=7\nCLEANUP_STATUS=0\n",
            "RUN_ID=two-gpu\nGPU_COUNT=2\nELAPSED_SECONDS=11\nCLEANUP_STATUS=0\n",
        ]
        for order in itertools.permutations(metadata):
            with self.subTest(order=order):
                self.assert_gpu_seconds(list(order), 29)

    def test_single_final_run_is_counted(self) -> None:
        self.assert_gpu_seconds(["GPU_COUNT=2\nELAPSED_SECONDS=11\n"], 22)

    def test_active_run_does_not_inherit_completed_run_elapsed(self) -> None:
        completed = "GPU_COUNT=1\nELAPSED_SECONDS=17\n"
        active = "GPU_COUNT=2\n"
        self.assert_gpu_seconds([completed, active], 17)
        self.assert_gpu_seconds([active, completed], 17)

    def test_elapsed_before_gpu_count_is_also_supported(self) -> None:
        self.assert_gpu_seconds(
            ["ELAPSED_SECONDS=20\nGPU_COUNT=0\n", "ELAPSED_SECONDS=11\nGPU_COUNT=2\n"],
            22,
        )

    def test_empty_run_store_uses_zero(self) -> None:
        self.assert_gpu_seconds([], 0)


if __name__ == "__main__":
    unittest.main()
