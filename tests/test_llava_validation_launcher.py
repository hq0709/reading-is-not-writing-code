"""Immutable launch contract for the LLaVA validation opportunity screen."""

from pathlib import Path
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/server/run_llava_validation_opportunity_gate.sh"


class LauncherContractTests(unittest.TestCase):
    def test_immutable_source_binding_and_fixed_environment(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            'test "$source_dir" = "$run_dir/source"',
            'SOURCE_COMMIT=$expected_sha', 'SOURCE_PATH=$source_dir',
            'RUN_ID=${run_dir##*/}', 'source scripts/server/activate_env.sh',
            'GPU_COUNT=1', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
        ):
            self.assertIn(required, source)

    def test_cpu_gpu_and_summary_phases_have_bounded_timeouts(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        for phase, duration in (("prepare", "30m"), ("run", "90m"), ("summarize", "5m")):
            self.assertRegex(source, rf"timeout[^\n]+{duration}\s*\\\n\s*python -B -m src\.run_llava_validation_opportunity {phase}")
        module = "python -B -m src.run_llava_validation_opportunity "
        self.assertLess(source.index(module + "prepare "), source.index(module + "run "))
        self.assertIn('if test "$mode" = full;', source)
        self.assertIn('preflight) args+=(--preflight-only)', source)

    def test_shell_bytes_are_lf_and_modes_explicit(self):
        raw = LAUNCHER.read_bytes()
        self.assertNotIn(b"\r\n", raw)
        source = raw.decode("utf-8")
        self.assertIn("set -euo pipefail", source)
        self.assertIn("[full|preflight|summarize] EXPECTED_SHA", source)
        self.assertIn('"$data_root"/runs/*)', source)


if __name__ == "__main__":
    unittest.main()
