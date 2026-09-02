"""Build the dermoscopy manifest from the crawled ISIC index.

The second modality exists to answer one question about the first: are the chest-radiograph findings a
property of medical vision-language models, or of chest radiographs? To answer it the design has to be the
same design, not merely another dataset. It is:

    chest radiograph                     dermoscopy
    ----------------------------------   ------------------------------------
    9 findings from report labels        7 diagnoses from the archive taxonomy
    view position (AP / PA)              anatomic site
    sex, age decade                      sex, age decade
    patient-level split by hash          patient-level split by the same hash
    control task over view x sex x age   control task over site x sex x age

Restricted to `image_type == dermoscopic`: it is 97% of the records that carry full metadata, and the
remaining 3% are total-body-photography tiles, a different instrument whose inclusion would confound the
acquisition nuisance with the imaging device rather than isolate it.

    python src/build_isic_manifest.py --out data/isic_manifest.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os

# Subtypes of one disease are merged. The archive splits melanoma five ways by stage and morphology, which
# are distinctions a radiology-style presence/absence probe cannot express and which would leave each arm
# too small to fit. Everything else is kept at the granularity the archive reports.
MERGE = {
    "Melanoma, NOS": "Melanoma", "Melanoma in situ": "Melanoma", "Melanoma Invasive": "Melanoma",
    "Melanoma metastasis": "Melanoma", "Melanoma, NOS, Superficial spreading": "Melanoma",
    "Squamous cell carcinoma, NOS": "SquamousCellCarcinoma",
    "Squamous cell carcinoma in situ": "SquamousCellCarcinoma",
    "Squamous cell carcinoma, Invasive": "SquamousCellCarcinoma",
    "Solar or actinic keratosis": "ActinicKeratosis",
    "Seborrheic keratosis": "SeborrheicKeratosis",
    "Pigmented benign keratosis": "PigmentedBenignKeratosis",
    "Lichen planus like keratosis": "PigmentedBenignKeratosis",
    "Basal cell carcinoma": "BasalCellCarcinoma",
    "Nevus": "Nevus",
    "Dermatofibroma": "Dermatofibroma",
    "Solar lentigo": "Lentigo", "Lentigo NOS": "Lentigo",
    "Fibroepithelial polyp": "FibroepithelialPolyp",
}


def patient_split(pid, test_frac=0.2, val_frac=0.1, salt="concept-flow-v1"):
    """Identical to the chest-radiograph split, salt included, so the two are the same procedure."""
    h = int(hashlib.sha256(f"{salt}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/isic/images.jsonl")
    ap.add_argument("--images", default="data/isic/images")
    ap.add_argument("--out", default="data/isic_manifest.csv")
    ap.add_argument("--min-per-concept", type=int, default=300)
    ap.add_argument("--min-per-site", type=int, default=50)
    ap.add_argument("--image-type", default="dermoscopic")
    args = ap.parse_args()

    rows, dropped = [], collections.Counter()
    for line in open(args.index):
        r = json.loads(line)
        flat = {}
        for grp, d in r["metadata"].items():
            flat.update(d)
        if flat.get("image_type") != args.image_type:
            dropped["image_type"] += 1; continue
        if not flat.get("patient_id"):
            dropped["no_patient_id"] += 1; continue
        if not (flat.get("sex") and flat.get("anatom_site_1") and flat.get("diagnosis_3")):
            dropped["missing_clinical"] += 1; continue
        if flat.get("age_approx") is None:
            dropped["no_age"] += 1; continue
        label = MERGE.get(flat["diagnosis_3"])
        if label is None:
            dropped["diagnosis_not_in_taxonomy"] += 1; continue
        rows.append({
            "row_id": r["isic_id"],
            "image_path": os.path.join(args.images, r["isic_id"] + ".jpg"),
            "url": r["files"]["full"]["url"],
            "patient_id": flat["patient_id"],
            "lesion_id": flat.get("lesion_id", ""),
            "age": int(flat["age_approx"]),
            "sex_M": int(str(flat["sex"]).lower().startswith("m")),
            "site": flat["anatom_site_1"],
            "malignant": int(str(flat.get("diagnosis_1", "")).lower().startswith("malig")),
            "label": label,
        })

    counts = collections.Counter(r["label"] for r in rows)
    concepts = sorted(c for c, n in counts.items() if n >= args.min_per_concept)
    rows = [r for r in rows if r["label"] in concepts]

    # A site with a handful of images cannot support a control task: its type appears on one side of the
    # patient-level split only, so the random label attached to it is trained on and never tested, or the
    # reverse. Dropping it costs a few images and keeps every type genuinely recurrent.
    site_n = collections.Counter(r["site"] for r in rows)
    small = {s for s, n in site_n.items() if n < args.min_per_site}
    if small:
        print(f"dropping {sum(site_n[s] for s in small)} rows from rare sites: "
              + ", ".join(f"{s} ({site_n[s]})" for s in sorted(small)))
        rows = [r for r in rows if r["site"] not in small]

    sites = sorted({r["site"] for r in rows})
    for r in rows:
        r["split"] = patient_split(r["patient_id"])
        for c in concepts:
            r[c] = int(r["label"] == c)
        for s in sites:
            r["site_" + s.replace(" ", "")] = int(r["site"] == s)

    # The same assertion the chest-radiograph manifest carries. A patient on both sides of the split turns
    # a held-out score into a partially in-sample one, and it is silent unless it is checked.
    by_pat = collections.defaultdict(set)
    for r in rows:
        by_pat[r["patient_id"]].add(r["split"])
    spanning = [p for p, s in by_pat.items() if len(s) > 1]
    assert not spanning, f"{len(spanning)} patients span splits"

    cols = (["row_id", "image_path", "url", "patient_id", "lesion_id", "split", "age", "sex_M",
             "site", "malignant", "label"] + concepts + ["site_" + s.replace(" ", "") for s in sites])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    sp = collections.Counter(r["split"] for r in rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"  patients {len(by_pat)}  |  train {sp['train']}  val {sp['val']}  test {sp['test']}")
    print(f"  patients spanning splits: 0 (asserted)")
    print(f"\ndropped: " + ", ".join(f"{k}={v}" for k, v in dropped.most_common()))
    print(f"\nconcepts ({len(concepts)}):")
    for c in concepts:
        n = sum(r[c] for r in rows)
        ntr = sum(r[c] for r in rows if r["split"] == "train")
        nte = sum(r[c] for r in rows if r["split"] == "test")
        print(f"  {c:26s} {n:6d}  ({ntr} train / {nte} test)")
    print(f"\nnuisance, anatomic site ({len(sites)}):")
    for s in sites:
        print(f"  {s:26s} {sum(1 for r in rows if r['site'] == s):6d}")
    types = collections.Counter(f"{r['site']}_{r['sex_M']}_{min(max(r['age'] // 10, 0), 9)}" for r in rows)
    print(f"\ncontrol-task types: {len(types)}, median {sorted(types.values())[len(types)//2]} rows each, "
          f"smallest {min(types.values())}")


if __name__ == "__main__":
    main()
