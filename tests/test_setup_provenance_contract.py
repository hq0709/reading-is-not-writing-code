from __future__ import annotations

import csv
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SetupProvenanceContractTests(unittest.TestCase):
    def test_bootstrap_and_first_scientific_gate_statuses_are_separate(self) -> None:
        state = (ROOT / "docs/RESEARCH_STATE.md").read_text(encoding="utf-8")
        bootstrap = state.split("## Bootstrap gate", 1)[1].split(
            "## First hard gate", 1
        )[0]
        first_gate = state.split("## First hard gate", 1)[1].split(
            "## Transition rules", 1
        )[0]

        self.assertIn("Gate disposition: `READY`", bootstrap)
        self.assertIn("Scientific state: `OBSERVED`", first_gate)
        self.assertIn("Gate decision: `PASS`", first_gate)
        self.assertIn("Gate disposition: `READY`", first_gate)
        for condition in (
            "model",
            "dataset",
            "hook",
            "bootstrap",
            "control",
            "intervention",
            "receipt",
            "review",
        ):
            self.assertIn(condition, first_gate.lower())

    def test_provenance_manifest_preserves_observed_pins(self) -> None:
        manifest_path = ROOT / "config/toolchain-provenance.tsv"
        self.assertTrue(manifest_path.is_file())
        with manifest_path.open(encoding="utf-8", newline="") as stream:
            rows = {row["component"]: row for row in csv.DictReader(stream, delimiter="\t")}

        expected = {
            "miniforge-conda": (
                "26.3.2",
                "42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94",
                "e4417d0a99b86d7cf8ab384e132f2a41b4db9c8d955dfcc6359b91c2f178aa3c",
            ),
            "uv": (
                "0.12.8",
                "9d360fe3d4a2c26157ecab7b44892acfe6b43e490c940ed0b29ab12e2085b1d3",
                "fe60af3d6b1771ecea15fdfd3821959912009e1612b3767a4988f135bc3cdd9d",
            ),
            "uvx": (
                "0.12.8",
                "9d360fe3d4a2c26157ecab7b44892acfe6b43e490c940ed0b29ab12e2085b1d3",
                "88cc93ffee37a4cf3470c091b64bd187cc666a9f21b79dd1bfd5beaa026a02bb",
            ),
            "codex": (
                "0.152.1",
                "ba92dd27e5c06f0d3bbc58bfa4b9cfb6599cd2742fbb1f92a2765e6c07dedb5a",
                "b82018241214a4a7c6b97b198585192d1dbc3aab1ddcdc640f04d8dee8c606f9",
            ),
            "claude": (
                "2.1.258",
                "3a68d3406cf674e17bed1733a4dcf37805e2e47d87417700007d7e1aa766a944",
                "704f1334ac65d3e89e1c6c1d7663293ad786a6166afdb71b5075337df630f976",
            ),
        }
        self.assertEqual(set(expected), set(rows))
        for component, (version, installer_hash, binary_hash) in expected.items():
            row = rows[component]
            self.assertEqual(version, row["expected_version"])
            self.assertEqual(installer_hash, row["installer_sha256"])
            self.assertEqual(binary_hash, row["expected_binary_sha256"])
            self.assertRegex(row["official_source_url"], r"^https://")
            self.assertTrue(row["binary_home_relative_path"])
            self.assertTrue(row["sanitized_install_command"])
            self.assertNotRegex(
                row["sanitized_install_command"],
                re.compile(r"(?i)(token|password|secret|authorization)"),
            )

    def test_receipt_writer_is_home_bound_atomic_and_private(self) -> None:
        script = (ROOT / "scripts/server/write_setup_receipt.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "AUTHORIZED_HOME=/home/qingchan",
            "config/toolchain-provenance.tsv",
            "setup-receipt.tsv",
            "mktemp",
            "umask 077",
            "chmod 600",
            "mv -f",
            "trap",
            "date -u",
            "hostname",
            "git rev-parse HEAD",
            "sha256sum",
            "ARIS_FULL_SHA",
            "git -C \"$ARIS_REPO\" rev-parse HEAD",
        ):
            self.assertIn(required, script)
        self.assertIn('test "$HOME" = "$AUTHORIZED_HOME"', script)
        self.assertIn('"$AUTHORIZED_HOME"/*', script)
        self.assertNotRegex(script, r"\bsudo\b")


if __name__ == "__main__":
    unittest.main()
