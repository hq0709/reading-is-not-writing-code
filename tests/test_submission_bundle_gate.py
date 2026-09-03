from __future__ import annotations

import hashlib
import importlib.util
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/server/build_submission_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_submission_bundle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SubmissionBundleGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        for relative in BUNDLE.SOURCE_FILES:
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = b"asset\n"
            if relative == "main.tex":
                content = b"\\author{Anonymous Authors}\n"
            path.write_bytes(content)

    def build(self, name: str) -> tuple[Path, Path]:
        return BUNDLE.build(
            source=self.source,
            output=self.root / name,
            source_date_epoch=1788439147,
            tectonic_version="Tectonic 0.17.0",
            tectonic_sha256="a" * 64,
            pdf_sha256="b" * 64,
        )

    def test_archive_is_deterministic_minimal_and_manifested(self) -> None:
        _, first = self.build("first")
        _, second = self.build("second")
        self.assertEqual(digest(first), digest(second))

        with tarfile.open(first, "r:gz") as archive:
            names = archive.getnames()
            expected = sorted(
                f"submission-source/{name}"
                for name in (*BUNDLE.SOURCE_FILES, *BUNDLE.GENERATED_FILES)
            )
            self.assertEqual(expected, names)
            self.assertTrue(all(member.isfile() for member in archive.getmembers()))
            manifest = archive.extractfile("submission-source/MANIFEST.sha256")
            assert manifest is not None
            lines = manifest.read().decode().splitlines()
            self.assertEqual(len(BUNDLE.SOURCE_FILES), len(lines))

    def test_identity_marker_is_rejected(self) -> None:
        (self.source / "sections/0_abstract.tex").write_text(
            "artifact at /home/qingchan/private\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(SystemExit, "forbidden identity marker"):
            self.build("rejected")

    def test_missing_or_symlinked_source_is_rejected(self) -> None:
        target = self.source / "sections/0_abstract.tex"
        target.unlink()
        target.symlink_to(self.source / "main.tex")
        with self.assertRaisesRegex(SystemExit, "required regular source file"):
            self.build("rejected")

    def test_runner_is_bound_to_accepted_compile_receipt(self) -> None:
        runner = (ROOT / "scripts/server/run_submission_bundle_gate.sh").read_text(encoding="utf-8")
        for value in (
            "20260903T123856Z-00f0adc-paper-compile",
            "00f0adc32a4ac36dba3d0f21feca485f1c8ef972",
            "e88c7f9a42baa4f04c7ccdbfdf9653501e4f96a81bd1c1ee238163671e5106e2",
            "SOURCE_DATE_EPOCH=1788439147",
            'env SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" FORCE_SOURCE_DATE=1',
            "byteIdenticalToAcceptedPdf",
        ):
            self.assertIn(value, runner)


if __name__ == "__main__":
    unittest.main()
