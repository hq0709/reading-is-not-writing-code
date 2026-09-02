from __future__ import annotations

import importlib.util
import datetime
import os
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


if __name__ == "__main__":
    unittest.main()
