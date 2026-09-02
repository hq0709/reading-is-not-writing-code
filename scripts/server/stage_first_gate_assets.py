#!/usr/bin/env python3
"""Atomically stage the public assets required by the first scientific gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


AUTHORIZED_HOME = Path("/home/qingchan")
REPO = AUTHORIZED_HOME / "work/concept-flow"
DATA_ROOT = AUTHORIZED_HOME / "data/concept-flow"
STOP = DATA_ROOT / "STOP"
MIN_FREE_BYTES_AFTER_STAGE = 500 * 1024**3
STAGE_RESERVE_BYTES = 120 * 1024**3
LOCK_NAME = ".stage-first-gate-assets.lock"
NIH_HASH_MANIFEST = "config/nih-chestxray14-sha256.tsv"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
MODEL_HASHES = {
    "model-00001-of-00003.safetensors": "c11dbf016ee7d35ee130c19b67e20eb04873996006f424b7bcbd453b6517ee66",
    "model-00002-of-00003.safetensors": "46df6c6e5fad297fe7fbca4963dcf180cb1d0528cabc884d1f603311fed01328",
    "model-00003-of-00003.safetensors": "4f06177c37ca13944e73b57dd6d051767b950dffc12400c700dfa2509c24c697",
    "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
}

BOX_SHARED_NAME = "zx7xbtg7oj9ko9ghmgmsm5zd7tnhhscp"
NIH_ROOT_PAGE = "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345"
NIH_PAGE = "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737"


def box_file_url(file_id: int) -> str:
    return (
        "https://nihcc.app.box.com/index.php?rm=box_download_shared_file"
        f"&shared_name={BOX_SHARED_NAME}&file_id=f_{file_id}"
    )


NIH_ARCHIVES = (
    ("images_001.tar.gz", 2_008_470_987, box_file_url(219_764_235_225)),
    ("images_002.tar.gz", 3_952_623_504, box_file_url(219_767_703_471)),
    ("images_003.tar.gz", 3_929_234_850, box_file_url(219_770_039_352)),
    ("images_004.tar.gz", 3_838_903_983, box_file_url(221_185_642_661)),
    ("images_005.tar.gz", 3_935_496_531, box_file_url(219_776_556_743)),
    ("images_006.tar.gz", 3_986_301_172, box_file_url(219_777_758_783)),
    ("images_007.tar.gz", 4_016_328_426, box_file_url(220_610_700_915)),
    ("images_008.tar.gz", 4_018_347_353, box_file_url(219_776_273_384)),
    ("images_009.tar.gz", 4_111_327_929, box_file_url(219_782_291_318)),
    ("images_010.tar.gz", 4_181_556_296, box_file_url(219_781_375_034)),
    ("images_011.tar.gz", 4_187_084_020, box_file_url(219_777_519_815)),
    ("images_012.tar.gz", 2_914_187_733, box_file_url(219_778_785_923)),
)
NIH_LABELS = (
    "Data_Entry_2017_v2020.csv",
    9_003_496,
    box_file_url(219_760_887_468),
)


def fail(message: str) -> None:
    raise SystemExit(f"asset staging refused: {message}")


def assert_managed_path(path: Path) -> Path:
    """Reject paths outside the authorized home or crossing any symlink."""
    home = AUTHORIZED_HOME.absolute()
    candidate = path.absolute()
    if not candidate.is_relative_to(home):
        fail(f"path escapes authorized home: {path}")
    resolved_home = home.resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(resolved_home):
        fail(f"resolved path escapes authorized home: {path} -> {resolved}")
    current = home
    if current.is_symlink():
        fail(f"symbolic link in managed path: {current}")
    for part in candidate.relative_to(home).parts:
        current = current / part
        if current.is_symlink():
            fail(f"symbolic link in managed path: {current}")
    return resolved


def assert_tree_has_no_symlinks(root: Path) -> None:
    assert_managed_path(root)
    if not root.exists():
        return
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            path = Path(parent) / name
            if path.is_symlink():
                fail(f"symbolic link in managed tree: {path}")


def assert_model_tree_symlinks_stay_within(root: Path) -> None:
    """Allow Hugging Face cache links only when their targets stay in root."""
    assert_managed_path(root)
    resolved_root = root.resolve(strict=False)
    if not root.exists():
        return
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            path = Path(parent) / name
            if not path.is_symlink():
                continue
            try:
                target = path.resolve(strict=True)
            except (FileNotFoundError, RuntimeError, OSError) as error:
                fail(f"invalid model symlink in staging partial: {path}: {error}")
            if not target.is_relative_to(resolved_root):
                fail(f"model symlink escapes staging partial: {path} -> {target}")


def assert_runtime_safe(head: str) -> None:
    """Recheck mutable safety gates around every long phase and publish."""
    for path in (REPO, DATA_ROOT, STOP):
        assert_managed_path(path)
    if STOP.exists():
        fail("stop sentinel exists")
    current = subprocess.check_output(
        ["git", "-C", os.fspath(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    upstream = subprocess.check_output(
        ["git", "-C", os.fspath(REPO), "rev-parse", "@{upstream}"], text=True
    ).strip()
    if current != head or upstream != head:
        fail(f"source commit changed from captured commit {head}: HEAD={current} upstream={upstream}")
    free = shutil.disk_usage(DATA_ROOT).free
    if free - STAGE_RESERVE_BYTES < MIN_FREE_BYTES_AFTER_STAGE:
        fail("insufficient disk reserve for atomic staging")


def assert_safe() -> str:
    if Path.home().resolve() != AUTHORIZED_HOME.resolve(strict=False):
        fail(f"unexpected home {Path.home()}")
    for path in (AUTHORIZED_HOME, REPO, DATA_ROOT, STOP):
        assert_managed_path(path)
    if subprocess.check_output(
        ["git", "-C", os.fspath(REPO), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ):
        fail("source tree is dirty")
    head = subprocess.check_output(["git", "-C", os.fspath(REPO), "rev-parse", "HEAD"], text=True).strip()
    upstream = subprocess.check_output(
        ["git", "-C", os.fspath(REPO), "rev-parse", "@{upstream}"], text=True
    ).strip()
    if head != upstream:
        fail("source commit is not pushed")
    assert_runtime_safe(head)
    return head


@contextmanager
def staging_lock():
    """Hold an atomic fail-closed singleton lock without touching partials."""
    assert_managed_path(DATA_ROOT)
    lock = DATA_ROOT / LOCK_NAME
    assert_managed_path(lock)
    try:
        os.mkdir(lock, 0o700)
    except FileExistsError:
        fail(f"asset staging already in progress (or stale lock requires audit): {lock}")
    owner = lock / "owner.json"
    try:
        assert_managed_path(owner)
        owner.write_text(
            json.dumps({"pid": os.getpid(), "started_utc": datetime.now(UTC).isoformat()}) + "\n",
            encoding="utf-8",
        )
        owner.chmod(0o600)
        yield
    finally:
        assert_managed_path(lock)
        assert_managed_path(owner)
        if owner.exists():
            owner.unlink()
        lock.rmdir()


def read_committed_file(head: str, repo_path: str) -> bytes:
    if repo_path.startswith("/") or ".." in PurePosixPath(repo_path).parts:
        fail(f"invalid committed path: {repo_path}")
    return subprocess.check_output(
        ["git", "-C", os.fspath(REPO), "show", f"{head}:{repo_path}"]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    assert_managed_path(path.parent)
    assert_managed_path(path)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    assert_managed_path(temp)
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.chmod(0o600)
    os.replace(temp, path)


def atomic_publish(
    partial: Path,
    final: Path,
    head: str,
    *,
    allow_internal_symlinks_within_partial: bool = False,
) -> None:
    assert_managed_path(partial)
    assert_managed_path(final)
    assert_runtime_safe(head)
    if allow_internal_symlinks_within_partial:
        assert_model_tree_symlinks_stay_within(partial)
    else:
        assert_tree_has_no_symlinks(partial)
    os.replace(partial, final)


def stage_model(head: str) -> None:
    final = DATA_ROOT / "models/huggingface"
    partial = DATA_ROOT / f"models/.partial-huggingface-{MODEL_REVISION}"
    hub = partial / "hub"
    for path in (final.parent, final, partial, hub):
        assert_managed_path(path)
    if final.exists():
        fail(f"model target already exists: {final}")
    partial.mkdir(parents=True, exist_ok=True)
    assert_model_tree_symlinks_stay_within(partial)
    assert_managed_path(hub)
    hub.mkdir(exist_ok=True)
    assert_managed_path(hub)
    assert_model_tree_symlinks_stay_within(partial)
    os.environ["HF_HOME"] = os.fspath(partial)
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoProcessor

    assert_runtime_safe(head)
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    assert_runtime_safe(head)
    if info.sha != MODEL_REVISION or info.private or info.gated:
        fail("model identity, access, or revision mismatch")
    assert_runtime_safe(head)
    print(f"staging {MODEL_ID}@{MODEL_REVISION}", flush=True)
    snapshot = Path(
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=hub)
    ).resolve()
    if snapshot.name != MODEL_REVISION or not snapshot.is_relative_to(partial.resolve()):
        fail(f"unexpected model snapshot path: {snapshot}")
    assert_model_tree_symlinks_stay_within(partial)
    assert_runtime_safe(head)
    observed = {}
    for name, expected in MODEL_HASHES.items():
        path = snapshot / name
        # Hugging Face snapshots intentionally use file symlinks into their
        # content-addressed blob store. The managed snapshot directory and the
        # resolved blob must both remain inside this staging partial.
        assert_managed_path(path.parent)
        resolved_file = path.resolve(strict=True)
        if not resolved_file.is_relative_to(partial.resolve(strict=True)):
            fail(f"model blob escapes staging partial: {path} -> {resolved_file}")
        assert_runtime_safe(head)
        actual = sha256(path)
        assert_runtime_safe(head)
        if actual != expected:
            fail(f"model hash mismatch for {name}: {actual}")
        observed[name] = {"bytes": path.stat().st_size, "sha256": actual}
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    if getattr(config, "vision_feature_layer", None) != -2:
        fail("model config no longer selects vision feature layer -2")
    assert_runtime_safe(head)
    write_json(
        partial / "asset-receipt.json",
        {
            "asset": "llava15_7b",
            "completed_utc": datetime.now(UTC).isoformat(),
            "source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "source_commit": head,
            "snapshot_relative_path": os.fspath(snapshot.relative_to(partial.resolve())),
            "verified_files": observed,
        },
    )
    atomic_publish(partial, final, head, allow_internal_symlinks_within_partial=True)
    print(f"model staged at {final}", flush=True)


def verify_nih_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        fail(f"NIH SHA-256 mismatch for {path.name}: {actual} != {expected}")
    return actual


def load_nih_hashes(head: str, specs: tuple[tuple[str, int, str], ...]) -> dict[str, str]:
    """Load checksums committed at the captured source SHA; absence blocks NIH staging."""
    payload = read_committed_file(head, NIH_HASH_MANIFEST).decode("ascii")
    rows = csv.DictReader(io.StringIO(payload), delimiter="\t")
    if rows.fieldnames != ["name", "sha256"]:
        fail(f"trusted NIH SHA-256 manifest has invalid schema: {NIH_HASH_MANIFEST}")
    hashes: dict[str, str] = {}
    for row in rows:
        name = row["name"]
        digest = row["sha256"].lower()
        if name in hashes or not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail(f"invalid trusted NIH SHA-256 entry for {name!r}")
        hashes[name] = digest
    expected_names = {name for name, _, _ in specs}
    missing = sorted(expected_names - hashes.keys())
    extra = sorted(hashes.keys() - expected_names)
    if missing:
        fail(f"missing trusted SHA-256 for NIH downloads: {', '.join(missing)}")
    if extra:
        fail(f"unexpected entries in trusted NIH SHA-256 manifest: {', '.join(extra)}")
    return hashes


def download_one(
    download_dir: Path,
    spec: tuple[str, int, str],
    expected_sha256: str,
    head: str,
) -> dict:
    name, expected_size, url = spec
    assert_managed_path(download_dir)
    path = download_dir / name
    assert_managed_path(path)
    if path.exists() and path.stat().st_size > expected_size:
        fail(f"oversized partial download: {path}")
    assert_runtime_safe(head)
    print(f"downloading {name} ({expected_size / 1024**3:.2f} GiB)", flush=True)
    subprocess.run(
        [
            "curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
            "--continue-at", "-", "--silent", "--show-error", "--output", os.fspath(path), url,
        ],
        check=True,
    )
    assert_runtime_safe(head)
    assert_managed_path(path)
    if path.stat().st_size != expected_size:
        fail(f"size mismatch for {name}: {path.stat().st_size} != {expected_size}")
    assert_runtime_safe(head)
    actual = verify_nih_hash(path, expected_sha256)
    assert_runtime_safe(head)
    result = {"name": name, "bytes": expected_size, "sha256": actual, "source": url}
    print(f"verified download {name} {result['sha256']}", flush=True)
    return result


def safe_extract(archive: Path, destination: Path) -> None:
    assert_managed_path(archive)
    assert_managed_path(destination)
    assert_tree_has_no_symlinks(destination)
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                fail(f"unsafe archive member in {archive.name}: {member.name}")
        stream.extractall(destination, filter="data")
    assert_tree_has_no_symlinks(destination)


def box_items(page: str) -> dict[str, dict]:
    html = urllib.request.urlopen(page, timeout=30).read().decode("utf-8")
    match = re.search(r"Box\.postStreamData = (\{.*?\});\s*</script>", html)
    if not match:
        fail(f"Box metadata missing from {page}")
    payload = json.loads(match.group(1))
    folder = payload.get("/app-api/enduserapp/shared-folder", {})
    return {item["name"]: item for item in folder.get("items", [])}


def validate_nih_source() -> None:
    archive_items = box_items(NIH_PAGE)
    root_items = box_items(NIH_ROOT_PAGE)
    expected = [
        (name, size, int(url.rsplit("f_", 1)[1]), archive_items)
        for name, size, url in NIH_ARCHIVES
    ]
    expected.append((NIH_LABELS[0], NIH_LABELS[1], 219_760_887_468, root_items))
    for name, size, file_id, items in expected:
        item = items.get(name)
        if not item or item.get("type") != "file":
            fail(f"official NIH Box item missing: {name}")
        if item.get("id") != file_id or item.get("itemSize") != size:
            fail(f"official NIH Box metadata changed for {name}")


def stage_nih(head: str) -> None:
    final = DATA_ROOT / "datasets/nih-chestxray14"
    partial = DATA_ROOT / "datasets/.partial-nih-chestxray14"
    downloads = partial / "downloads"
    for path in (final.parent, final, partial, downloads):
        assert_managed_path(path)
    if final.exists():
        fail(f"dataset target already exists: {final}")
    downloads.mkdir(parents=True, exist_ok=True)
    assert_tree_has_no_symlinks(partial)
    specs = (*NIH_ARCHIVES, NIH_LABELS)
    hashes = load_nih_hashes(head, specs)
    assert_runtime_safe(head)
    validate_nih_source()
    assert_runtime_safe(head)
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(download_one, downloads, spec, hashes[spec[0]], head)
            for spec in specs
        ]
        for future in as_completed(futures):
            records.append(future.result())
    assert_runtime_safe(head)
    assert_tree_has_no_symlinks(partial)
    png_root = partial / "png"
    assert_managed_path(png_root)
    png_root.mkdir(exist_ok=True)
    for name, _, _ in NIH_ARCHIVES:
        assert_runtime_safe(head)
        print(f"validating and extracting {name}", flush=True)
        safe_extract(downloads / name, png_root)
        assert_runtime_safe(head)
    images = png_root / "images"
    assert_managed_path(images)
    assert_tree_has_no_symlinks(images)
    assert_runtime_safe(head)
    image_count = sum(1 for path in images.iterdir() if path.suffix.lower() == ".png")
    assert_runtime_safe(head)
    if image_count != 112_120:
        fail(f"NIH image count mismatch: {image_count}")
    labels = downloads / NIH_LABELS[0]
    assert_managed_path(labels)
    shutil.copy2(labels, partial / NIH_LABELS[0])
    with labels.open(newline="", encoding="utf-8-sig") as stream:
        label_rows = sum(1 for _ in csv.DictReader(stream))
    if label_rows != 112_120:
        fail(f"NIH label row count mismatch: {label_rows}")

    manifest_temp = partial / ".manifest.csv.tmp"
    manifest_final = partial / "manifest.csv"
    for path in (manifest_temp, manifest_final):
        assert_managed_path(path)
    source_manifest = io.StringIO(read_committed_file(head, "data/manifest.csv").decode("utf-8"))
    with source_manifest as source, manifest_temp.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "row_id" not in reader.fieldnames or "image_path" in reader.fieldnames:
            fail("registered manifest schema mismatch")
        fields = [reader.fieldnames[0], "image_path", *reader.fieldnames[1:]]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        registered_rows = 0
        for row in reader:
            image = final / "png/images" / f"{row['row_id']}.png"
            staged_image = images / image.name
            if not staged_image.is_file():
                fail(f"registered image missing from NIH release: {image.name}")
            row["image_path"] = os.fspath(image)
            writer.writerow(row)
            registered_rows += 1
    os.replace(manifest_temp, manifest_final)
    assert_runtime_safe(head)
    assert_tree_has_no_symlinks(partial)
    write_json(
        partial / "asset-receipt.json",
        {
            "asset": "nih-chestxray14",
            "completed_utc": datetime.now(UTC).isoformat(),
            "official_page": NIH_PAGE,
            "source_commit": head,
            "archives": sorted(records, key=lambda row: row["name"]),
            "image_count": image_count,
            "label_rows": label_rows,
            "registered_manifest_rows": registered_rows,
            "registered_manifest_sha256": sha256(manifest_final),
        },
    )
    atomic_publish(partial, final, head)
    print(f"NIH dataset staged at {final}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=("model", "nih", "all"))
    args = parser.parse_args()
    with staging_lock():
        head = assert_safe()
        if args.asset in ("model", "all"):
            stage_model(head)
        if args.asset in ("nih", "all"):
            stage_nih(head)


if __name__ == "__main__":
    main()
