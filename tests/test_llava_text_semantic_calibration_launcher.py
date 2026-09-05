import unittest
from pathlib import Path

LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts/server/run_llava_text_semantic_calibration_gate.sh"
)


class LauncherTests(unittest.TestCase):
    def test_immutable_binding_and_registered_limits(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        for text in (
            'timeout --signal=TERM --kill-after=60 14m bash "$0" "$@"',
            'test "$source_dir" = "$run_dir/source"',
            'test "$used" -lt 500',
            "source scripts/server/activate_env.sh",
            "HF_HUB_OFFLINE=1",
            "OMP_NUM_THREADS=4",
            "python -B -m src.run_llava_text_semantic_calibration run",
            "python -B -m src.run_llava_text_semantic_calibration replay",
        ):
            self.assertIn(text, source)

    def test_shell_mode_and_line_endings(self):
        raw = LAUNCHER.read_bytes()
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertNotIn(b"\r\n", raw)


if __name__ == "__main__":
    unittest.main()
