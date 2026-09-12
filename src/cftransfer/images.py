"""Image loading against the frozen manifests. Paths are executor-local; row ids are the join key."""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

DATA_ROOT = Path("/rodata/azradonc_dev/m253405/cf-transfer/data")
IMAGE_ROOTS = {
    "nih": Path("/rodata/azradonc_dev/m253405/cache/nih"),           # + images/<row_id>.png
    "coco": DATA_ROOT / "coco",                                        # + train2017/… val2017/…
    "chexpert": DATA_ROOT / "chexpert" / "images",
}


def load_cohort(dataset_id: str, roles: tuple[str, ...] | None = None) -> list[dict]:
    rows = list(csv.DictReader((DATA_ROOT / dataset_id / "manifests" / "cohort.csv").open(newline="", encoding="utf-8")))
    if roles:
        rows = [r for r in rows if r["role"] in roles]
    rows.sort(key=lambda r: (["train", "preflight", "calibration", "test"].index(r["role"]), int(r["order"])))
    return rows


def load_labels(dataset_id: str) -> dict[tuple[str, str], dict]:
    out = {}
    for r in csv.DictReader((DATA_ROOT / dataset_id / "manifests" / "labels.csv").open(newline="", encoding="utf-8")):
        out[(r["row_id"], r["concept"])] = r
    return out


def image_path(dataset_id: str, row: dict) -> Path:
    return IMAGE_ROOTS[dataset_id] / row["relative_image_path"]


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")
