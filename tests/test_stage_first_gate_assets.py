from __future__ import annotations

import importlib.util
import datetime
import csv
import hashlib
import gzip
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/server/stage_first_gate_assets.py"
if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
SPEC = importlib.util.spec_from_file_location("stage_first_gate_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


class ManagedPathTests(unittest.TestCase):
    def test_rejects_symlink_in_managed_path_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            real = home / "real"
            real.mkdir()
            link = home / "link"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with mock.patch.object(stage, "AUTHORIZED_HOME", home):
                with self.assertRaisesRegex(SystemExit, "symbolic link"):
                    stage.assert_managed_path(link / "partial")

    def test_rejects_resolved_path_outside_authorized_home(self) -> None:
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as outside_dir:
            home = Path(home_dir)
            link = home / "escape"
            try:
                link.symlink_to(Path(outside_dir), target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with mock.patch.object(stage, "AUTHORIZED_HOME", home):
                with self.assertRaisesRegex(SystemExit, "authorized home"):
                    stage.assert_managed_path(link / "download")

    def test_model_tree_rejects_mocked_symlink_target_outside_partial(self) -> None:
        root = Path("/home/qingchan/data/concept-flow/models/.partial-model")
        link = root / "hub/snapshots/revision/weights.safetensors"
        outside = Path("/home/qingchan/other-project/weights.safetensors")

        def is_symlink(path: Path) -> bool:
            return path == link

        def resolve(path: Path, strict: bool = False) -> Path:
            if path == link:
                return outside
            return path

        def exists(path: Path) -> bool:
            return path == root

        with (
            mock.patch.object(stage, "AUTHORIZED_HOME", Path("/home/qingchan")),
            mock.patch.object(stage, "assert_managed_path"),
            mock.patch.object(stage.os, "walk", return_value=[(link.parent, [], [link.name])]),
            mock.patch.object(Path, "is_symlink", is_symlink),
            mock.patch.object(Path, "resolve", resolve),
            mock.patch.object(Path, "exists", exists),
        ):
            with self.assertRaisesRegex(SystemExit, "model symlink escapes staging partial"):
                stage.assert_model_tree_symlinks_stay_within(root)

    def test_model_tree_accepts_mocked_symlink_target_inside_partial(self) -> None:
        root = Path("/home/qingchan/data/concept-flow/models/.partial-model")
        link = root / "hub/snapshots/revision/weights.safetensors"
        blob = root / "hub/blobs/abc123"

        def is_symlink(path: Path) -> bool:
            return path == link

        def resolve(path: Path, strict: bool = False) -> Path:
            if path == link:
                return blob
            return path

        def exists(path: Path) -> bool:
            return path == root

        with (
            mock.patch.object(stage, "assert_managed_path"),
            mock.patch.object(stage.os, "walk", return_value=[(link.parent, [], [link.name])]),
            mock.patch.object(Path, "is_symlink", is_symlink),
            mock.patch.object(Path, "resolve", resolve),
            mock.patch.object(Path, "exists", exists),
        ):
            stage.assert_model_tree_symlinks_stay_within(root)


class RuntimeSafetyTests(unittest.TestCase):
    def test_runtime_check_rejects_head_drift_from_captured_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            data_root.mkdir()
            outputs = iter(("new-head\n", "captured-head\n"))
            with (
                mock.patch.object(stage, "AUTHORIZED_HOME", root),
                mock.patch.object(stage, "REPO", root / "repo"),
                mock.patch.object(stage, "DATA_ROOT", data_root),
                mock.patch.object(stage, "STOP", data_root / "STOP"),
                mock.patch.object(stage.subprocess, "check_output", side_effect=lambda *a, **k: next(outputs)),
                mock.patch.object(stage.shutil, "disk_usage", return_value=mock.Mock(free=10**15)),
            ):
                with self.assertRaisesRegex(SystemExit, "source commit changed"):
                    stage.assert_runtime_safe("captured-head")

    def test_runtime_check_rejects_stop_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "STOP").touch()
            with (
                mock.patch.object(stage, "AUTHORIZED_HOME", root),
                mock.patch.object(stage, "REPO", root / "repo"),
                mock.patch.object(stage, "DATA_ROOT", data_root),
                mock.patch.object(stage, "STOP", data_root / "STOP"),
            ):
                with self.assertRaisesRegex(SystemExit, "stop sentinel"):
                    stage.assert_runtime_safe("captured-head")

    def test_stop_sentinel_prevents_final_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            partial = data_root / "datasets/.partial-asset"
            final = data_root / "datasets/asset"
            partial.mkdir(parents=True)
            (data_root / "STOP").touch()
            with (
                mock.patch.object(stage, "AUTHORIZED_HOME", root),
                mock.patch.object(stage, "REPO", root / "repo"),
                mock.patch.object(stage, "DATA_ROOT", data_root),
                mock.patch.object(stage, "STOP", data_root / "STOP"),
                mock.patch.object(stage.os, "replace") as replace,
            ):
                with self.assertRaisesRegex(SystemExit, "stop sentinel"):
                    stage.atomic_publish(partial, final, "captured-head")
            replace.assert_not_called()


class CommittedManifestTests(unittest.TestCase):
    def test_reads_registered_manifest_from_captured_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "data").mkdir()
            (repo / "data/manifest.csv").write_text("live checkout", encoding="utf-8")
            with (
                mock.patch.object(stage, "REPO", repo),
                mock.patch.object(stage.subprocess, "check_output", return_value=b"committed manifest" ) as call,
            ):
                observed = stage.read_committed_file("abc123", "data/manifest.csv")
            self.assertEqual(b"committed manifest", observed)
            self.assertEqual(
                ["git", "-C", os.fspath(repo), "show", "abc123:data/manifest.csv"],
                call.call_args.args[0],
            )


class SingletonLockTests(unittest.TestCase):
    def test_duplicate_staging_invocation_fails_without_deleting_partial_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            partial = data_root / "datasets/.partial-nih-chestxray14"
            partial.mkdir(parents=True)
            marker = partial / "resume.bin"
            marker.write_bytes(b"partial")
            with (
                mock.patch.object(stage, "AUTHORIZED_HOME", root),
                mock.patch.object(stage, "DATA_ROOT", data_root),
                stage.staging_lock(),
            ):
                with self.assertRaisesRegex(SystemExit, "already in progress"):
                    with stage.staging_lock():
                        pass
                self.assertEqual(b"partial", marker.read_bytes())


class NihHashTests(unittest.TestCase):
    def test_normal_exact_size_download_skips_curl_and_verifies_expected_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            download_dir = Path(temp_dir)
            payload = b"committed-checksum-payload"
            path = download_dir / "images_001.tar.gz"
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            spec = (path.name, len(payload), "https://official/file")
            with (
                mock.patch.object(stage, "assert_managed_path"),
                mock.patch.object(stage, "assert_runtime_safe"),
                mock.patch.object(stage.subprocess, "run") as run,
            ):
                record = stage.download_one(download_dir, spec, expected, "captured-head")
            run.assert_not_called()
            self.assertEqual(expected, record["sha256"])

    def test_normal_exact_size_download_skips_curl_but_rejects_wrong_expected_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            download_dir = Path(temp_dir)
            payload = b"committed-checksum-payload"
            path = download_dir / "images_001.tar.gz"
            path.write_bytes(payload)
            spec = (path.name, len(payload), "https://official/file")
            with (
                mock.patch.object(stage, "assert_managed_path"),
                mock.patch.object(stage, "assert_runtime_safe"),
                mock.patch.object(stage.subprocess, "run") as run,
            ):
                with self.assertRaisesRegex(SystemExit, "NIH SHA-256 mismatch"):
                    stage.download_one(download_dir, spec, "0" * 64, "captured-head")
            run.assert_not_called()

    def test_bootstrap_exact_size_resume_skips_curl_and_still_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            download_dir = Path(temp_dir)
            payload = b"already complete"
            path = download_dir / "images_001.tar.gz"
            path.write_bytes(payload)
            spec = (path.name, len(payload), "https://official/file")
            with (
                mock.patch.object(stage, "assert_managed_path"),
                mock.patch.object(stage, "assert_runtime_safe"),
                mock.patch.object(stage.subprocess, "run") as run,
            ):
                record = stage.download_nih_candidate(download_dir, spec, "captured-head")
            run.assert_not_called()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])

    def test_single_pass_archive_validation_accepts_good_and_rejects_gzip_corruption(self) -> None:
        raw_tar = io.BytesIO()
        with tarfile.open(fileobj=raw_tar, mode="w") as archive:
            content = b"png-fixture" * 100
            member = tarfile.TarInfo("images/example.png")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        valid = gzip.compress(raw_tar.getvalue())
        corrupt_crc = valid[:-8] + bytes([valid[-8] ^ 0x01]) + valid[-7:]
        missing_trailer = valid[:-8]
        truncated_deflate = valid[:-20] + valid[-8:]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good = root / "good.tar.gz"
            good.write_bytes(valid)
            with (
                mock.patch.object(stage, "assert_managed_path"),
                mock.patch.object(stage, "assert_runtime_safe"),
            ):
                self.assertEqual(1, stage.validate_nih_archive_stream(good, "captured-head"))

            for name, payload in (
                ("corrupt-crc.tar.gz", corrupt_crc),
                ("missing-trailer.tar.gz", missing_trailer),
                ("truncated-deflate.tar.gz", truncated_deflate),
            ):
                path = root / name
                path.write_bytes(payload)
                with (
                    self.subTest(name=name),
                    mock.patch.object(stage, "assert_managed_path"),
                    mock.patch.object(stage, "assert_runtime_safe"),
                ):
                    with self.assertRaisesRegex(SystemExit, "invalid gzip stream"):
                        stage.validate_nih_archive_stream(path, "captured-head")

    def test_archive_validation_passes_one_gzip_stream_to_streaming_tar_reader(self) -> None:
        raw_tar = io.BytesIO()
        with tarfile.open(fileobj=raw_tar, mode="w") as archive:
            content = b"fixture"
            member = tarfile.TarInfo("images/example.png")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "archive.tar.gz"
            path.write_bytes(gzip.compress(raw_tar.getvalue()))
            real_tar_open = tarfile.open
            with (
                mock.patch.object(stage, "assert_managed_path"),
                mock.patch.object(stage, "assert_runtime_safe"),
                mock.patch.object(stage.tarfile, "open", wraps=real_tar_open) as tar_open,
            ):
                stage.validate_nih_archive_stream(path, "captured-head")
            self.assertEqual("r|", tar_open.call_args.kwargs["mode"])
            self.assertIn("fileobj", tar_open.call_args.kwargs)

    def test_interrupted_candidate_bundle_has_no_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state = root / "state"
            state.mkdir()
            records = [{"name": "images_001.tar.gz", "sha256": "a" * 64}]
            receipt = {"source_commit": "captured-head"}
            with (
                mock.patch.object(stage, "AUTHORIZED_HOME", root),
                mock.patch.object(stage, "DATA_ROOT", root),
                mock.patch.object(stage, "write_json", side_effect=RuntimeError("interrupted")),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    stage.write_nih_candidate_bundle(records, receipt, "captured-head")
            bundle = root / stage.NIH_CANDIDATE_BUNDLE
            self.assertFalse(bundle.exists())
            self.assertEqual([], list(state.rglob("receipt.json")))

    def test_runtime_failure_immediately_before_bundle_rename_leaves_no_final_receipt(self) -> None:
        for reason in ("stop sentinel exists", "source commit changed"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                (root / "state").mkdir()
                records = [{"name": "images_001.tar.gz", "sha256": "a" * 64}]
                with (
                    mock.patch.object(stage, "AUTHORIZED_HOME", root),
                    mock.patch.object(stage, "DATA_ROOT", root),
                    mock.patch.object(stage, "assert_runtime_safe", side_effect=SystemExit(reason)),
                ):
                    with self.assertRaisesRegex(SystemExit, reason):
                        stage.write_nih_candidate_bundle(
                            records, {"source_commit": "captured-head"}, "captured-head"
                        )
                bundle = root / stage.NIH_CANDIDATE_BUNDLE
                self.assertFalse(bundle.exists())
                self.assertFalse((bundle / "receipt.json").exists())

    def test_hash_manifest_requires_every_registered_download(self) -> None:
        manifest = b"name\tsha256\nimages_001.tar.gz\t" + b"a" * 64 + b"\n"
        specs = (
            ("images_001.tar.gz", 1, "https://example/1"),
            ("Data_Entry_2017_v2020.csv", 1, "https://example/labels"),
        )
        with mock.patch.object(stage, "read_committed_file", return_value=manifest):
            with self.assertRaisesRegex(SystemExit, "missing trusted SHA-256"):
                stage.load_nih_hashes("abc123", specs)

    def test_download_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "archive.tar.gz"
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(SystemExit, "NIH SHA-256 mismatch"):
                stage.verify_nih_hash(path, "0" * 64)

    def test_bootstrap_empty_manifest_writes_data_root_candidates_without_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            repo = home / "work/concept-flow"
            data_root = home / "data/concept-flow"
            (repo / "config").mkdir(parents=True)
            (data_root / "datasets").mkdir(parents=True)
            (data_root / "state").mkdir(parents=True)
            committed_manifest = repo / stage.NIH_HASH_MANIFEST
            committed_manifest.write_text("name\tsha256\n", encoding="ascii")

            tar_payload = io.BytesIO()
            with tarfile.open(fileobj=tar_payload, mode="w:gz") as archive:
                content = b"png-fixture"
                member = tarfile.TarInfo("images/00000001_000.png")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            csv_payload = (
                "Image Index,Finding Labels,Patient ID\n"
                "00000001_000.png,Effusion,1\n"
                "00000002_000.png,No Finding,2\n"
            ).encode()
            archives = tuple(
                (
                    f"images_{index:03d}.tar.gz",
                    len(tar_payload.getvalue()),
                    f"https://official/tar-{index:03d}",
                )
                for index in range(1, 13)
            )
            labels_spec = (
                "Data_Entry_2017_v2020.csv",
                len(csv_payload),
                "https://official/csv",
            )
            specs = (*archives, labels_spec)
            payloads = {name: tar_payload.getvalue() for name, _, _ in archives}
            payloads[labels_spec[0]] = csv_payload

            def fake_download(download_dir: Path, spec: tuple[str, int, str], head: str) -> dict:
                name, expected_size, url = spec
                path = download_dir / name
                path.write_bytes(payloads[name])
                return {
                    "name": name,
                    "bytes": expected_size,
                    "sha256": hashlib.sha256(payloads[name]).hexdigest(),
                    "source": url,
                }

            with (
                mock.patch.object(stage, "AUTHORIZED_HOME", home),
                mock.patch.object(stage, "REPO", repo),
                mock.patch.object(stage, "DATA_ROOT", data_root),
                mock.patch.object(stage, "STOP", data_root / "STOP"),
                mock.patch.object(stage, "NIH_ARCHIVES", archives),
                mock.patch.object(stage, "NIH_LABELS", labels_spec),
                mock.patch.object(stage, "NIH_EXPECTED_LABEL_ROWS", 2),
                mock.patch.object(stage, "validate_nih_source"),
                mock.patch.object(stage, "assert_runtime_safe"),
                mock.patch.object(stage, "download_nih_candidate", side_effect=fake_download),
            ):
                stage.stage_nih_bootstrap("captured-head")

            candidate_bundle = data_root / stage.NIH_CANDIDATE_BUNDLE
            candidate_tsv = candidate_bundle / "checksums.tsv"
            candidate_json = candidate_bundle / "receipt.json"
            self.assertTrue(candidate_tsv.is_file())
            self.assertTrue(candidate_json.is_file())
            with candidate_tsv.open(newline="", encoding="ascii") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([spec[0] for spec in specs], [row["name"] for row in rows])
            self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))
            receipt = json.loads(candidate_json.read_text(encoding="utf-8"))
            self.assertEqual("captured-head", receipt["source_commit"])
            self.assertEqual(12, receipt["validated_tar_archives"])
            self.assertEqual(2, receipt["validated_label_rows"])
            self.assertFalse((data_root / "datasets/nih-chestxray14").exists())
            self.assertEqual("name\tsha256\n", committed_manifest.read_text(encoding="ascii"))


if __name__ == "__main__":
    unittest.main()
