import unittest
from pathlib import Path

LAUNCHER = (
    Path(__file__).resolve().parents[1] / "scripts/server/run_llava_readout_diagnostic_gate.sh"
)


class LauncherContractTests(unittest.TestCase):
    def test_immutable_binding_environment_and_registered_limits(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        for value in (
            'timeout --signal=TERM --kill-after=60 90m bash "$0" "$@"',
            'test "$source_dir" = "$run_dir/source"',
            "SOURCE_COMMIT=$expected_sha",
            "source scripts/server/activate_env.sh",
            "HF_HUB_OFFLINE=1",
            "OMP_NUM_THREADS=4",
            'test "$used" -lt 500',
            "grep -Fx 'GPU_COUNT=1'",
        ):
            self.assertIn(value, source)
        self.assertIn("python -B -m src.run_llava_readout_diagnostic prepare", source)
        self.assertIn("python -B -m src.run_llava_readout_diagnostic run", source)
        self.assertIn("python -B -m src.run_llava_readout_diagnostic summarize", source)

    def test_shell_mode_and_line_endings(self):
        raw = LAUNCHER.read_bytes()
        self.assertNotIn(b"\r\n", raw)
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertIn(b"[full|preflight|summarize] EXPECTED_SHA", raw)


if __name__ == "__main__":
    unittest.main()
