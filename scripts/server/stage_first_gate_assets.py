#!/usr/bin/env python3
"""Atomically stage the public assets required by the first scientific gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


AUTHORIZED_HOME = Path("/home/qingchan")
REPO = AUTHORIZED_HOME / "work/concept-flow"
DATA_ROOT = AUTHORIZED_HOME / "data/concept-flow"
STOP = DATA_ROOT / "STOP"
MIN_FREE_BYTES_AFTER_STAGE = 500 * 1024**3
STAGE_RESERVE_BYTES = 120 * 1024**3

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


def assert_safe() -> str:
    if Path.home().resolve() != AUTHORIZED_HOME:
        fail(f"unexpected home {Path.home()}")
    for path in (REPO, DATA_ROOT):
        if not path.resolve().is_relative_to(AUTHORIZED_HOME):
            fail(f"path escapes authorized home: {path}")
    if STOP.exists():
        fail("stop sentinel exists")
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
    free = shutil.disk_usage(DATA_ROOT).free
    if free - STAGE_RESERVE_BYTES < MIN_FREE_BYTES_AFTER_STAGE:
        fail("insufficient disk reserve for atomic staging")
    return head


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.chmod(0o600)
    os.replace(temp, path)


def stage_model(head: str) -> None:
    final = DATA_ROOT / "models/huggingface"
    partial = DATA_ROOT / f"models/.partial-huggingface-{MODEL_REVISION}"
    if final.exists():
        fail(f"model target already exists: {final}")
    partial.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = os.fspath(partial)
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoProcessor

    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION or info.private or info.gated:
        fail("model identity, access, or revision mismatch")
    print(f"staging {MODEL_ID}@{MODEL_REVISION}", flush=True)
    snapshot = Path(
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=partial / "hub")
    ).resolve()
    if snapshot.name != MODEL_REVISION or not snapshot.is_relative_to(partial.resolve()):
        fail(f"unexpected model snapshot path: {snapshot}")
    observed = {}
    for name, expected in MODEL_HASHES.items():
        path = snapshot / name
        actual = sha256(path)
        if actual != expected:
            fail(f"model hash mismatch for {name}: {actual}")
        observed[name] = {"bytes": path.stat().st_size, "sha256": actual}
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    if getattr(config, "vision_feature_layer", None) != -2:
        fail("model config no longer selects vision feature layer -2")
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
    os.replace(partial, final)
    print(f"model staged at {final}", flush=True)


def download_one(download_dir: Path, spec: tuple[str, int, str]) -> dict:
    name, expected_size, url = spec
    path = download_dir / name
    if path.exists() and path.stat().st_size > expected_size:
        fail(f"oversized partial download: {path}")
    print(f"downloading {name} ({expected_size / 1024**3:.2f} GiB)", flush=True)
    subprocess.run(
        [
            "curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
            "--continue-at", "-", "--silent", "--show-error", "--output", os.fspath(path), url,
        ],
        check=True,
    )
    if path.stat().st_size != expected_size:
        fail(f"size mismatch for {name}: {path.stat().st_size} != {expected_size}")
    result = {"name": name, "bytes": expected_size, "sha256": sha256(path), "source": url}
    print(f"verified download {name} {result['sha256']}", flush=True)
    return result


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                fail(f"unsafe archive member in {archive.name}: {member.name}")
        stream.extractall(destination, filter="data")


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
    if final.exists():
        fail(f"dataset target already exists: {final}")
    downloads = partial / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    validate_nih_source()
    specs = (*NIH_ARCHIVES, NIH_LABELS)
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_one, downloads, spec) for spec in specs]
        for future in as_completed(futures):
            records.append(future.result())
    if STOP.exists():
        fail("stop sentinel appeared during download")
    png_root = partial / "png"
    png_root.mkdir(exist_ok=True)
    for name, _, _ in NIH_ARCHIVES:
        print(f"validating and extracting {name}", flush=True)
        safe_extract(downloads / name, png_root)
    images = png_root / "images"
    image_count = sum(1 for path in images.iterdir() if path.suffix.lower() == ".png")
    if image_count != 112_120:
        fail(f"NIH image count mismatch: {image_count}")
    labels = downloads / NIH_LABELS[0]
    shutil.copy2(labels, partial / NIH_LABELS[0])
    with labels.open(newline="", encoding="utf-8-sig") as stream:
        label_rows = sum(1 for _ in csv.DictReader(stream))
    if label_rows != 112_120:
        fail(f"NIH label row count mismatch: {label_rows}")

    source_manifest = REPO / "data/manifest.csv"
    manifest_temp = partial / ".manifest.csv.tmp"
    manifest_final = partial / "manifest.csv"
    with source_manifest.open(newline="", encoding="utf-8") as source, manifest_temp.open(
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
    os.replace(partial, final)
    print(f"NIH dataset staged at {final}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=("model", "nih", "all"))
    args = parser.parse_args()
    head = assert_safe()
    if args.asset in ("model", "all"):
        stage_model(head)
    if args.asset in ("nih", "all"):
        stage_nih(head)


if __name__ == "__main__":
    main()
