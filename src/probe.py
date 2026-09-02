"""Fit concept probes at every locus, with the validity battery, and emit the survival curve.

The survey of 135 medical interpretability studies found that probe results are almost never controlled:
a permutation null appeared in 3 of 69 applicable records, a randomly initialised encoder in 5 of 121,
and a randomised control task in none of the 135. A probe number without those is not interpretable, so
this module refuses to report accuracy alone. Every locus gets:

  real          AUROC for the true labels
  perm          AUROC after permuting labels within the training split. Chance floor for this fit
  control       AUROC on a control task: each input TYPE gets a fixed random label (Hewitt and Liang).
                For medical images the type is the view position, which recurs and crosses a
                patient-level split. Selectivity = real - control is the number the field has never had
  nuisance      AUROC for view position and sex at the same locus, from the same features. If a finding
                probe tracks these, it is reading acquisition rather than pathology

Comparability across loci is enforced, not assumed: the decoder class and its capacity are fixed, features
are standardised, and an optional random projection puts every locus at a common dimension so a wider
representation cannot win on width alone.

    python src/probe.py --acts runs/act_qwen7b --manifest data/manifest.csv \
        --concepts Effusion,Pneumothorax --out runs/probe_qwen7b
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEFAULT_NUISANCE = ["view_AP", "sex_M"]


def load_acts(act_dir):
    """Merge the per-GPU shards into one array per locus, keyed by row_id."""
    files = sorted(glob.glob(os.path.join(act_dir, "shard*.npz")))
    if not files:
        raise SystemExit(f"no shards in {act_dir}")
    ids, per_locus = [], {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        rid = [str(x) for x in z["row_id"]]
        ids.extend(rid)
        for k in z.files:
            if k == "row_id":
                continue
            per_locus.setdefault(k, []).append(z[k])
    acts = {k: np.concatenate(v, axis=0) for k, v in per_locus.items()}
    n = len(ids)
    for k, v in acts.items():
        if v.shape[0] != n:
            raise SystemExit(f"locus {k}: {v.shape[0]} rows but {n} ids")
    print(f"loaded {n} rows over {len(acts)} loci from {len(files)} shards")
    return ids, acts


def load_manifest(path):
    rows = {r["row_id"]: r for r in csv.DictReader(open(path))}
    print(f"manifest {len(rows)} rows")
    return rows


def fit_auroc(Xtr, ytr, Xte, yte, C=1.0, seed=0, proj=None, rng=None):
    """One probe fit. Capacity is fixed by C and by the projection, identically at every locus."""
    if proj is not None:
        Xtr, Xte = Xtr @ proj, Xte @ proj
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def build_types(meta, cols=("view_AP", "sex_M", "age")):
    """The input TYPE a control task random-labels: the cross product of the given columns.

    A control task needs many types. With only two (say the view position) the random assignment is one
    of four functions, two constant and two equal to the type itself up to a flip, so the "control" just
    asks whether the type is decodable. Measured on Qwen2.5-VL-7B, view position decodes at AUROC 0.999
    at every locus, which drove the control to 0.999 and selectivity negative. That is a degenerate
    control, not a strong probe.

    Crossing an acquisition variable with sex and age decade gives tens of types that all recur on both
    sides of a patient-level split. The columns are named rather than hardcoded so that the identical
    procedure runs on both modalities: view position x sex x age for chest radiographs, anatomic site x
    sex x age for dermoscopy. `age` is bucketed by decade; every other column is used as it stands.

    It is not a strict Hewitt and Liang control task, because images have no equivalent of word identity,
    and the paper must say so rather than imply otherwise.
    """
    out = []
    for m in meta:
        parts = []
        for c in cols:
            v = m.get(c, "")
            if c == "age":
                try:
                    v = str(min(max(int(v) // 10, 0), 9))
                except (TypeError, ValueError):
                    v = "na"
            parts.append(f"{v}")
        out.append("_".join(parts))
    return np.array(out)


def control_labels(types, rng, min_types=8):
    """A fixed random label per type. Returns the labels, the assignment, and whether it is usable."""
    uniq = sorted(set(types))
    if len(uniq) < min_types:
        return None, {}, False
    assign = {t: int(rng.integers(0, 2)) for t in uniq}
    if len(set(assign.values())) < 2:
        assign[uniq[0]] = 1 - assign[uniq[0]]
    return np.array([assign[t] for t in types]), assign, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--concepts", default="Effusion,Pneumothorax,Cardiomegaly,Atelectasis,Consolidation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--proj-dim", type=int, default=0,
                    help="if >0, random-project every locus to this width so dimensionality is matched")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for the test CI")
    ap.add_argument("--type-cols", default="view_AP,sex_M,age",
                    help="manifest columns crossed to define an input type for the control task. "
                         "chest radiographs: view_AP,sex_M,age  |  dermoscopy: site,sex_M,age")
    ap.add_argument("--nuisance", default="",
                    help="binary manifest columns probed alongside the concepts. Defaults to "
                         "view_AP,sex_M when present, otherwise every site_* column plus sex_M")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ids, acts = load_acts(args.acts)
    man = load_manifest(args.manifest)

    keep = [i for i, r in enumerate(ids) if r in man]
    if len(keep) != len(ids):
        print(f"dropping {len(ids) - len(keep)} rows with no manifest entry")
    ids = [ids[i] for i in keep]
    acts = {k: v[keep] for k, v in acts.items()}
    meta = [man[r] for r in ids]

    split = np.array([m["split"] for m in meta])
    tr, te = split == "train", split == "test"
    type_cols = [c.strip() for c in args.type_cols.split(",") if c.strip()]
    missing = [c for c in type_cols if c not in meta[0]]
    if missing:
        raise SystemExit(f"--type-cols names columns the manifest does not have: {missing}. "
                         f"available: {sorted(meta[0])[:25]}")
    if args.nuisance:
        nuisance = [c.strip() for c in args.nuisance.split(",") if c.strip()]
    elif all(c in meta[0] for c in DEFAULT_NUISANCE):
        nuisance = list(DEFAULT_NUISANCE)
    else:
        nuisance = sorted(c for c in meta[0] if c.startswith("site_")) + ["sex_M"]
    nuisance = [c for c in nuisance if c in meta[0]]
    print(f"nuisance probes: {nuisance}")
    types = build_types(meta, type_cols)
    n_types = len(set(types))
    per_type = np.bincount(np.unique(types, return_inverse=True)[1])
    print(f"control-task types: {n_types} ({'x'.join(type_cols)}), "
          f"median {int(np.median(per_type))} rows each, smallest {per_type.min()}")
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]

    loci = sorted(acts, key=lambda k: (0 if k.startswith("vis") else 1 if k == "connector" else 2,
                                       int(k.split(".L")[1].split(".")[0]) if ".L" in k else 0,
                                       k))
    print(f"\n{tr.sum()} train / {te.sum()} test rows | {len(loci)} loci | concepts {concepts}")

    projections = {}
    if args.proj_dim:
        for k, v in acts.items():
            d = v.shape[1]
            projections[k] = (rng.standard_normal((d, args.proj_dim)) / np.sqrt(args.proj_dim)
                              if d > args.proj_dim else np.eye(d)[:, :min(d, args.proj_dim)])

    results = []
    t0 = time.time()
    for ci, concept in enumerate(concepts):
        y = np.array([int(m[concept]) for m in meta])
        yc, assign, ctrl_ok = control_labels(types, np.random.default_rng(args.seed + ci))
        yperm = y.copy()
        perm_idx = np.where(tr)[0]
        shuffled = perm_idx.copy()
        np.random.default_rng(args.seed + 100 + ci).shuffle(shuffled)
        yperm[perm_idx] = y[shuffled]
        print(f"\n{concept}: {y[tr].sum()} train pos, {y[te].sum()} test pos | "
              f"control task over {n_types} types" + ("" if ctrl_ok else " [DEGENERATE, reported as nan]"))

        for lo in loci:
            X = acts[lo].astype(np.float32)
            proj = projections.get(lo) if args.proj_dim else None
            row = {"concept": concept, "locus": lo, "dim": int(X.shape[1])}
            row["real"] = fit_auroc(X[tr], y[tr], X[te], y[te], args.C, args.seed, proj)
            row["perm"] = fit_auroc(X[tr], yperm[tr], X[te], y[te], args.C, args.seed, proj)
            row["control"] = (fit_auroc(X[tr], yc[tr], X[te], yc[te], args.C, args.seed, proj)
                              if ctrl_ok else float("nan"))
            row["selectivity"] = row["real"] - row["control"] if ctrl_ok else float("nan")
            row["n_types"] = n_types
            for nz in nuisance:
                yn = np.array([int(m[nz]) for m in meta])
                row[f"nuis_{nz}"] = fit_auroc(X[tr], yn[tr], X[te], yn[te], args.C, args.seed, proj)
            results.append(row)
            if lo in ("vis.last", "connector") or lo.endswith(".L0.vis") or lo == loci[-1]:
                nz0 = f"nuis_{nuisance[0]}" if nuisance else None
                extra = f"  {nuisance[0][:9]} {row[nz0]:.3f}" if nz0 and nz0 in row else ""
                print(f"    {lo:16s} real {row['real']:.3f}  perm {row['perm']:.3f}  "
                      f"ctrl {row['control']:.3f}  sel {row['selectivity']:+.3f}{extra}")

    cols = list(results[0])
    with open(os.path.join(args.out, "probe_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    json.dump({"acts": args.acts, "manifest": args.manifest, "C": args.C,
               "type_cols": type_cols, "nuisance": nuisance, "n_types": n_types,
               "proj_dim": args.proj_dim, "seed": args.seed, "concepts": concepts,
               "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "elapsed_s": round(time.time() - t0, 1)},
              open(os.path.join(args.out, "probe_meta.json"), "w"), indent=1)
    print(f"\nwrote {len(results)} rows to {args.out}/probe_results.csv in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
