"""Build the frozen cohort and label manifests for each dataset (README section 4, return-format.md).

Outputs per dataset, under <data_root>/<dataset_id>/manifests/:
  cohort.csv : dataset_id,row_id,unit_id,role,order,relative_image_path,original_split
  labels.csv : dataset_id,row_id,concept,label,label_known,label_raw,view,sex,age,width,height,type_id

Sampling uses SHA-256 of tagged identifiers exactly as the package specifies; labels never enter the
ordering. NIH reuses the package's cohorts.csv verbatim (verified to reproduce from data/manifest.csv).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from .protocol import CONCEPTS, COCO_CATEGORY_IDS, PKG, REPO

COHORT_COLS = ["dataset_id", "row_id", "unit_id", "role", "order", "relative_image_path", "original_split"]
LABEL_COLS = ["dataset_id", "row_id", "concept", "label", "label_known", "label_raw", "view", "sex", "age",
              "width", "height", "type_id"]
NIH_EXTRA_LABELS = ["Consolidation", "Edema", "Infiltration", "no_finding"]
CHEXPERT_RAW_TO_CONCEPT = {"Pleural Effusion": "Effusion", "Atelectasis": "Atelectasis", "Pneumothorax": "Pneumothorax",
                           "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation", "Edema": "Edema"}
CHEXPERT_EXTRA = ["No Finding", "Enlarged Cardiomediastinum", "Lung Opacity", "Lung Lesion", "Pneumonia",
                  "Pleural Other", "Fracture", "Support Devices"]


def sha(tag: str, value: str) -> str:
    return hashlib.sha256(f"{tag}{value}".encode("utf-8")).hexdigest()


def age_decade(age) -> str:
    try:
        return str(min(max(int(age) // 10, 0), 9))
    except (TypeError, ValueError):
        return "na"


def xray_type_id(view_ap, sex_m, age) -> str:
    """view_AP x sex_M x age-decade, the recurring input type of src/probe.py:build_types."""
    return f"{view_ap}_{sex_m}_{age_decade(age)}"


def coco_type_id(width: int, height: int) -> str:
    aspect = width / height
    a = "a0" if aspect < 0.8 else ("a1" if aspect <= 1.25 else "a2")
    area = width * height
    s = "s0" if area < 65536 else ("s1" if area < 262144 else ("s2" if area < 1048576 else "s3"))
    return f"{a}_{s}"


def write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# ----------------------------------------------------------------------------------------- NIH
def build_nih(data_root: Path, nih_meta_csv: Path) -> dict:
    manifest = list(csv.DictReader((REPO / "data" / "manifest.csv").open(newline="", encoding="utf-8")))
    cohorts = list(csv.DictReader((PKG / "cohorts.csv").open(newline="", encoding="utf-8")))
    by_id = {r["row_id"]: r for r in manifest}
    # original image sizes from the official label table
    meta = {}
    with nih_meta_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            meta[row[0].replace(".png", "")] = {"width": row[7], "height": row[8]}
    cohort_rows, label_rows = [], []
    train = [r for r in manifest if r["split"] == "train"]
    roles = [("train", r["row_id"], i) for i, r in enumerate(train)]
    counters = defaultdict(int)
    for c in cohorts:
        roles.append((c["role"], c["row_id"], counters[c["role"]]))
        counters[c["role"]] += 1
    concepts = CONCEPTS["nih"] + NIH_EXTRA_LABELS
    for role, row_id, order in roles:
        m = by_id[row_id]
        cohort_rows.append({"dataset_id": "nih", "row_id": row_id, "unit_id": m["patient_id"], "role": role,
                            "order": order, "relative_image_path": f"images/{row_id}.png", "original_split": m["split"]})
        tid = xray_type_id(m["view_AP"], m["sex_M"], m["age"])
        for concept in concepts:
            raw = m[concept]
            label_rows.append({"dataset_id": "nih", "row_id": row_id, "concept": concept, "label": raw,
                               "label_known": "true", "label_raw": raw,
                               "view": "AP" if m["view_AP"] == "1" else "PA", "sex": "M" if m["sex_M"] == "1" else "F",
                               "age": m["age"], "width": meta.get(row_id, {}).get("width", ""),
                               "height": meta.get(row_id, {}).get("height", ""), "type_id": tid})
    out = data_root / "nih" / "manifests"
    write_csv(out / "cohort.csv", COHORT_COLS, cohort_rows)
    write_csv(out / "labels.csv", LABEL_COLS, label_rows)
    return summarize("nih", cohort_rows, label_rows)


# ---------------------------------------------------------------------------------------- COCO
def build_coco(data_root: Path) -> dict:
    ann_dir = data_root / "coco" / "annotations"
    wanted = set(COCO_CATEGORY_IDS.values())
    cat_to_concept = {v: k for k, v in COCO_CATEGORY_IDS.items()}

    def load(split: str):
        d = json.loads((ann_dir / f"instances_{split}2017.json").read_text(encoding="utf-8"))
        images = {im["id"]: im for im in d["images"]}
        present = defaultdict(set)
        for a in d["annotations"]:               # crowd instances count as present (README 4)
            if a["category_id"] in wanted:
                present[a["image_id"]].add(cat_to_concept[a["category_id"]])
        ids = sorted(images, key=lambda i: sha("cf-transfer-v1-coco:", f"{i:012d}"))
        return images, present, ids

    tr_images, tr_present, tr_ids = load("train")
    va_images, va_present, va_ids = load("val")
    roles = [("train", i, k) for k, i in enumerate(tr_ids[:20000])]
    roles += [("preflight", i, k) for k, i in enumerate(tr_ids[20000:20016])]
    roles += [("calibration", i, k) for k, i in enumerate(tr_ids[20016:20416])]
    roles += [("test", i, k) for k, i in enumerate(va_ids[:600])]
    assert len({i for _, i, _ in roles}) == len(roles), "image ids must be unique across roles"
    cohort_rows, label_rows = [], []
    for role, image_id, order in roles:
        split = "val" if role == "test" else "train"
        im = (va_images if split == "val" else tr_images)[image_id]
        present = (va_present if split == "val" else tr_present).get(image_id, set())
        row_id = f"{image_id:012d}"
        cohort_rows.append({"dataset_id": "coco", "row_id": row_id, "unit_id": row_id, "role": role, "order": order,
                            "relative_image_path": f"{split}2017/{im['file_name']}", "original_split": f"{split}2017"})
        tid = coco_type_id(im["width"], im["height"])
        for concept in CONCEPTS["coco"]:
            lab = "1" if concept in present else "0"
            label_rows.append({"dataset_id": "coco", "row_id": row_id, "concept": concept, "label": lab,
                               "label_known": "true", "label_raw": lab, "view": "", "sex": "", "age": "",
                               "width": im["width"], "height": im["height"], "type_id": tid})
    out = data_root / "coco" / "manifests"
    write_csv(out / "cohort.csv", COHORT_COLS, cohort_rows)
    write_csv(out / "labels.csv", LABEL_COLS, label_rows)
    return summarize("coco", cohort_rows, label_rows)


# ------------------------------------------------------------------------------------ CheXpert
def build_chexpert(data_root: Path, train_csv: Path, release: str) -> dict:
    """Official training release: one frontal image per patient, tagged-hash order, first 20,000/16/400/600.

    Path column looks like 'CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg'. The official
    validation set is left untouched. Unknown labels (-1 or blank) are kept unknown.
    """
    rows = list(csv.DictReader(train_csv.open(newline="", encoding="utf-8")))
    frontal = [r for r in rows if r.get("Frontal/Lateral", "").strip() == "Frontal"]
    per_patient = defaultdict(list)
    for r in frontal:
        parts = r["Path"].split("/")
        pid = next(p for p in parts if p.startswith("patient"))
        per_patient[pid].append(r)
    chosen = {pid: min(v, key=lambda r: sha("cf-transfer-v1-chexpert-row:", r["Path"])) for pid, v in per_patient.items()}
    order = sorted(chosen, key=lambda pid: sha("cf-transfer-v1-chexpert-patient:", pid))
    roles = [("train", p, k) for k, p in enumerate(order[:20000])]
    roles += [("preflight", p, k) for k, p in enumerate(order[20000:20016])]
    roles += [("calibration", p, k) for k, p in enumerate(order[20016:20416])]
    roles += [("test", p, k) for k, p in enumerate(order[20416:21016])]
    if len(order) < 21016:
        raise SystemExit(f"CheXpert release has only {len(order)} patients with frontal images; need 21,016")
    cohort_rows, label_rows = [], []
    for role, pid, k in roles:
        r = chosen[pid]
        rel = r["Path"].split("/", 1)[1] if r["Path"].startswith("CheXpert") else r["Path"]
        row_id = rel.replace("/", "__").rsplit(".", 1)[0]
        cohort_rows.append({"dataset_id": "chexpert", "row_id": row_id, "unit_id": pid, "role": role, "order": k,
                            "relative_image_path": rel, "original_split": "train"})
        view = r.get("AP/PA", "").strip()
        sex = r.get("Sex", "").strip()
        tid = xray_type_id("1" if view == "AP" else ("0" if view == "PA" else "na"),
                           "1" if sex == "Male" else ("0" if sex == "Female" else "na"), r.get("Age", ""))
        for raw_name, concept in list(CHEXPERT_RAW_TO_CONCEPT.items()) + [(x, x) for x in CHEXPERT_EXTRA]:
            raw = r.get(raw_name, "").strip()
            known = raw in ("1.0", "0.0", "1", "0")
            label = ("1" if raw in ("1.0", "1") else "0") if known else ""
            label_rows.append({"dataset_id": "chexpert", "row_id": row_id, "concept": concept, "label": label,
                               "label_known": "true" if known else "false", "label_raw": raw, "view": view,
                               "sex": sex[:1] if sex else "", "age": r.get("Age", ""), "width": "", "height": "",
                               "type_id": tid})
    out = data_root / "chexpert" / "manifests"
    write_csv(out / "cohort.csv", COHORT_COLS, cohort_rows)
    write_csv(out / "labels.csv", LABEL_COLS, label_rows)
    (out / "release.json").write_text(json.dumps({"dataset_release": release, "source_csv": str(train_csv)}, indent=2))
    return summarize("chexpert", cohort_rows, label_rows)


def summarize(dataset_id: str, cohort_rows: list[dict], label_rows: list[dict]) -> dict:
    roles = defaultdict(int)
    for r in cohort_rows:
        roles[r["role"]] += 1
    lab = {}
    concepts = CONCEPTS[dataset_id]
    by_role = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))   # role -> concept -> [pos, neg, unknown]
    role_of = {r["row_id"]: r["role"] for r in cohort_rows}
    for r in label_rows:
        if r["concept"] not in concepts:
            continue
        cell = by_role[role_of[r["row_id"]]][r["concept"]]
        if r["label_known"] != "true":
            cell[2] += 1
        elif r["label"] == "1":
            cell[0] += 1
        else:
            cell[1] += 1
    for role in ("train", "preflight", "calibration", "test"):
        lab[role] = {c: by_role[role][c] for c in concepts}
    types = defaultdict(set)
    for r in label_rows:
        types[role_of[r["row_id"]]].add(r["type_id"])
    return {"dataset_id": dataset_id, "roles": dict(roles), "labels_pos_neg_unknown": lab,
            "type_count_train": len(types["train"])}


# ------------------------------------------------------------------------------- CheXpert Plus (Redivis)
def build_chexpert_plus(data_root: Path, release: str) -> dict:
    """CheXpert Plus (Stanford AIMI on Redivis, aimi.chexpert_plus:5yyj v1.0): same patients, studies and CheXpert
    labels as CheXpert-v1.0, full-resolution PNGs. Inputs staged under <data_root>/chexpert/:
      df_chexpert_plus_240401.parquet  (path_to_image, frontal_lateral, ap_pa, deid_patient_id, age, sex, split)
      impression_fixed.json            (JSON lines: path_to_image + 14 CheXpert labels, 1.0/0.0/-1.0/null)
      png_train_index.parquet          (Redivis file index: file_name, file_id, size, md5_hash)
    Rule (README 4): train split, frontal images; one image per patient by SHA-256 of the original relative path;
    patients ordered by SHA-256 of the patient id; first 20,000 train, then 16/400/600. Labels never enter the order.
    """
    import json
    import pandas as pd
    d = data_root / "chexpert"
    meta = pd.read_parquet(d / "df_chexpert_plus_240401.parquet")
    labels = {}
    with (d / "impression_fixed.json").open() as f:
        for ln in f:
            r = json.loads(ln); labels[r["path_to_image"]] = r
    idx = pd.read_parquet(d / "png_train_index.parquet")
    png_files = set(idx["file_name"])
    fr = meta[(meta["split"] == "train") & (meta["frontal_lateral"] == "Frontal")].copy()
    fr["png_rel"] = fr["path_to_image"].str.replace(r"^train/", "", regex=True).str.replace(r"\.jpg$", ".png", regex=True)
    missing_png = (~fr["png_rel"].isin(png_files)).sum()
    fr = fr[fr["png_rel"].isin(png_files) & fr["path_to_image"].isin(labels)]
    per_patient = defaultdict(list)
    for r in fr.itertuples(index=False):
        per_patient[r.deid_patient_id].append(r)
    chosen = {pid: min(v, key=lambda r: sha("cf-transfer-v1-chexpert-row:", r.path_to_image)) for pid, v in per_patient.items()}
    order = sorted(chosen, key=lambda pid: sha("cf-transfer-v1-chexpert-patient:", pid))
    if len(order) < 21016:
        raise SystemExit(f"only {len(order)} patients with a labelled frontal PNG; need 21,016")
    roles = [("train", p, k) for k, p in enumerate(order[:20000])]
    roles += [("preflight", p, k) for k, p in enumerate(order[20000:20016])]
    roles += [("calibration", p, k) for k, p in enumerate(order[20016:20416])]
    roles += [("test", p, k) for k, p in enumerate(order[20416:21016])]
    cohort_rows, label_rows = [], []
    for role, pid, k in roles:
        r = chosen[pid]
        row_id = r.png_rel.replace("/", "__").rsplit(".", 1)[0]
        cohort_rows.append({"dataset_id": "chexpert", "row_id": row_id, "unit_id": pid, "role": role, "order": k,
                            "relative_image_path": f"train/{r.png_rel}", "original_split": "train"})
        view = r.ap_pa if isinstance(r.ap_pa, str) else ""
        sex = r.sex if isinstance(r.sex, str) else ""
        tid = xray_type_id("1" if view == "AP" else ("0" if view == "PA" else "na"),
                           "1" if sex == "Male" else ("0" if sex == "Female" else "na"), r.age)
        lab = labels[r.path_to_image]
        for raw_name, concept in list(CHEXPERT_RAW_TO_CONCEPT.items()) + [(x, x) for x in CHEXPERT_EXTRA]:
            raw = lab.get(raw_name)
            known = raw in (0.0, 1.0)
            label_rows.append({"dataset_id": "chexpert", "row_id": row_id, "concept": concept,
                               "label": ("1" if raw == 1.0 else "0") if known else "",
                               "label_known": "true" if known else "false", "label_raw": "" if raw is None else str(raw),
                               "view": view, "sex": sex[:1] if sex else "", "age": "" if pd.isna(r.age) else int(r.age),
                               "width": "", "height": "", "type_id": tid})
    out = d / "manifests"
    write_csv(out / "cohort.csv", COHORT_COLS, cohort_rows)
    write_csv(out / "labels.csv", LABEL_COLS, label_rows)
    (out / "release.json").write_text(json.dumps({
        "dataset_release": release, "source": "Redivis aimi.chexpert_plus:5yyj:v1_0 (stanford.redivis.com)",
        "label_file": "impression_fixed.json (CheXpert labeler output per image; 1/0 known, -1/null unknown)",
        "image_files": "PNG_train file index (full-resolution PNG, same paths as CheXpert-v1.0 train/*.jpg)",
        "frontal_train_rows": int(len(fr)), "patients_with_frontal": len(order), "png_missing_for_frontal_rows": int(missing_png),
        "selection": "one frontal per patient by SHA-256('cf-transfer-v1-chexpert-row:'+path_to_image); patients by "
                     "SHA-256('cf-transfer-v1-chexpert-patient:'+deid_patient_id); 20000/16/400/600"}, indent=2))
    return summarize("chexpert", cohort_rows, label_rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["nih", "coco", "chexpert", "chexpert_plus"])
    ap.add_argument("--data-root", default="/rodata/azradonc_dev/m253405/cf-transfer/data")
    ap.add_argument("--nih-meta", default="/rodata/azradonc_dev/m253405/cache/nih/Data_Entry_2017_v2020.csv")
    ap.add_argument("--chexpert-train-csv")
    ap.add_argument("--chexpert-release", default="")
    a = ap.parse_args()
    root = Path(a.data_root)
    if a.dataset == "nih":
        s = build_nih(root, Path(a.nih_meta))
    elif a.dataset == "coco":
        s = build_coco(root)
    elif a.dataset == "chexpert_plus":
        s = build_chexpert_plus(root, a.chexpert_release or "CheXpert Plus v1.0 (Redivis aimi.chexpert_plus:5yyj), full-resolution PNG")
    else:
        s = build_chexpert(root, Path(a.chexpert_train_csv), a.chexpert_release)
    print(json.dumps(s, indent=1))
    (root / "chexpert" / "manifests" / "summary.json").write_text(json.dumps(s, indent=1)) if a.dataset.startswith("chexpert") else \
        (root / a.dataset / "manifests" / "summary.json").write_text(json.dumps(s, indent=1))
