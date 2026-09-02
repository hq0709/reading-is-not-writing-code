"""Build the experiment manifest from NIH ChestX-ray14, with patient-level splits.

Splits are on Patient ID, never on image, because the dataset has multiple studies per patient and an
image-level split leaks. The survey found unstated or image-level splits to be one of the commonest
defects in this literature, so the split rule is enforced here once and every experiment inherits it.

Emits, per row: image path, patient id, split, one binary column per clinical concept, and the nuisance
columns (view position, sex, age band) that the design uses as controls.

    python src/build_manifest.py --out data/manifest.csv --per-concept 4000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
from collections import Counter, defaultdict

ROOT = "${CXR14_ROOT:?set CXR14_ROOT to the ChestX-ray14 image directory}"

# The clinical concepts we probe for. Chosen for prevalence in this dataset and for being the findings a
# radiologist would name, so a probe result is clinically interpretable.
CONCEPTS = ["Effusion", "Atelectasis", "Pneumothorax", "Consolidation", "Cardiomegaly", "Edema",
            "Infiltration", "Mass", "Nodule"]

# Nuisance concepts. The survey's most consistent finding across 135 studies was that acquisition and
# demographic variables are the most recoverable properties of a medical representation. If a finding
# probe is really reading one of these, these columns expose it.
NUISANCE = ["view_AP", "sex_M"]


def patient_split(pid: str, test_frac=0.2, val_frac=0.1, salt="concept-flow-v1"):
    """Deterministic, patient-level, stable across reruns and machines."""
    h = int(hashlib.sha256(f"{salt}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--per-concept", type=int, default=4000,
                    help="target positives per concept; negatives are matched 1:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = os.path.join(args.root, "Data_Entry_2017_v2020.csv")
    imgdir = os.path.join(args.root, "png", "images")
    if not os.path.isdir(imgdir):
        alt = os.path.join(args.root, "png")
        imgdir = alt if os.path.isdir(alt) else imgdir
    have = {f for f in os.listdir(imgdir) if f.endswith(".png")}
    print(f"{len(have)} images unzipped so far at {imgdir}")

    rows = list(csv.DictReader(open(labels)))
    print(f"{len(rows)} label rows, {len({r['Patient ID'] for r in rows})} patients")

    pool = []
    for r in rows:
        fn = r["Image Index"]
        if fn not in have:
            continue
        found = set(r["Finding Labels"].split("|"))
        try:
            age = int(r["Patient Age"])
        except ValueError:
            age = -1
        rec = {
            "row_id": fn[:-4],
            "image_path": os.path.join(imgdir, fn),
            "patient_id": r["Patient ID"],
            "split": patient_split(r["Patient ID"]),
            "view_AP": int(r["View Position"] == "AP"),
            "sex_M": int(r["Patient Gender"] == "M"),
            "age": age,
            "no_finding": int("No Finding" in found),
        }
        for c in CONCEPTS:
            rec[c] = int(c in found)
        pool.append(rec)
    print(f"{len(pool)} rows have an image on disk")

    # Sample a balanced set per concept so every probe has enough positives without extracting all 112k.
    rng = random.Random(args.seed)
    chosen: dict[str, dict] = {}
    for c in CONCEPTS:
        pos = [r for r in pool if r[c] == 1]
        neg = [r for r in pool if r[c] == 0 and r["no_finding"] == 1]
        rng.shuffle(pos)
        rng.shuffle(neg)
        k = min(args.per_concept, len(pos), len(neg))
        for r in pos[:k] + neg[:k]:
            chosen[r["row_id"]] = r
        print(f"  {c:14s} pos {len(pos):6d} -> take {k:5d}  (clean negatives available {len(neg)})")

    out_rows = list(chosen.values())
    rng.shuffle(out_rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = (["row_id", "image_path", "patient_id", "split", "age", "no_finding"]
            + NUISANCE + CONCEPTS)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    sp = Counter(r["split"] for r in out_rows)
    print(f"\nwrote {len(out_rows)} rows to {args.out}")
    print(f"split: {dict(sp)}")
    overlap = defaultdict(set)
    for r in out_rows:
        overlap[r["patient_id"]].add(r["split"])
    bad = [p for p, s in overlap.items() if len(s) > 1]
    print(f"patients spanning more than one split: {len(bad)}  (must be 0)")
    print("\npositives per concept and split:")
    for c in CONCEPTS + NUISANCE:
        line = "  ".join(f"{s} {sum(1 for r in out_rows if r['split'] == s and r[c] == 1):5d}"
                         for s in ("train", "val", "test"))
        print(f"  {c:14s} {line}")


if __name__ == "__main__":
    main()
