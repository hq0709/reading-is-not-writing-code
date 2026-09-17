"""FGOBJ: fine-grained object concepts on COCO -- the control that separates "clinical concept" from "hard question".

The campaign's headline contrast is 29 of 246 chest cells owned against 101 of 120 COCO cells. A reviewer can answer
that COCO differs from chest radiographs in DIFFICULTY, not in concept type: the six COCO concepts in the grid
(person, dog, car, chair, bottle, bicycle) are easy -- median clean-answer AUROC 0.98 over the 120 easy COCO cells
against 0.67 over the 246 chest cells -- and
matching chest cells to them on answerability and probe selectivity leaves the matched comparison starved, because
the natural-image side has no hard concepts to match to. FGOBJ adds six MORE COCO categories that are small, often
occluded or fine-grained, fitted, written and graded exactly like the six easy ones, so the natural-image side spans
the chest side's difficulty range and the matched comparison has partners on both sides.

The module has three parts:

  labels       build_fgobj_labels(data_root) APPENDS the six categories to <data>/coco/manifests/labels.csv with the
               rule of manifests.build_coco -- presence from the instance annotations of the row's own split, crowd
               instances count as present -- and writes fgobj.json next to it (selection rule, category ids, per-role
               counts, annotation-file digests). Append-only and idempotent: no existing row is rewritten, cohort.csv
               is not touched, and a second call finds the rows already present (and refuses to continue if any
               differ). Every other consumer is unaffected: fit, calibration, coverage and the manifest all iterate
               protocol.CONCEPTS[dataset], which the append does not change.

  directions   build_fgobj(model, dataset) fits the six probes with the campaign's own settings on the SAME training
               rows as the block's seed-0 fit: that fit's projection R and its TRAIN-ONLY scaler (mu, s are the stored
               ones), LogisticRegression(C=1, lbfgs, 2000 iterations, random_state 0), lifted with
               fit.direction_from_projected. The family is SELF-CONTAINED: its own 119 random directions and its own
               per-concept coordinate-permutation shams, drawn from PCG64(FGOBJ_RANDOM_SEED) in the order fit.fit_locus
               uses (the n_random x D normals first, row-normalised, then one permutation per concept), so an FGOBJ cell
               is graded against a random p95 and a sham of its own rather than against the easy concepts' references.
               The prep also stores, on the calibration rows, the real probe logits and the 20 type->random-label control
               logits (the campaign's own control labelling of the COCO aspect x area strata), so the analysis can apply
               the campaign's readability rule unchanged. CPU only; seconds.

  grid         module FGOBJ scores the 127-condition CORE grid (own clean baseline + 6 concept directions + 119 random
               + the question's sham) at the primary template and PRIMARY_ALPHA on the 600 test rows, and module
               FGOBJ_CALIBRATION scores the six clean questions on the 400 calibration rows, exactly as CALIBRATION does
               for the easy six, so readability and answerability are measured on the same cohort and by the same rule
               for both families.

Writes fits/<locus>/fgobj_seed0.npz.

    python -m cftransfer.fgobj --build-labels                        # once, per data root
    python -m cftransfer.fgobj --model-key M --dataset coco          # per block (CPU, seconds)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .fit import direction_from_projected, fit_lr, load_features, load_fit
from .images import DATA_ROOT, load_labels
from .manifests import LABEL_COLS, coco_type_id
from .protocol import (CONCEPTS, CONTROL_SEEDS, FGOBJ_CANDIDATE_POOL, FGOBJ_CATEGORY_IDS, FGOBJ_CONCEPTS,
                       FGOBJ_MAX_MEAN_AREA, FGOBJ_MIN_ROLE_POS, FGOBJ_MIN_TRAIN_POS, FGOBJ_RANDOM_SEED, N_RANDOM)
from .runpaths import fits_dir

FGOBJ_FILE = "fgobj_seed0.npz"
PROVENANCE = "fgobj.json"
FGOBJ_DATASET = "coco"


# ------------------------------------------------------------------------------------ selection (prespecified rule)
def select_fgobj_concepts(stats: dict[str, dict], pool: tuple[str, ...] = FGOBJ_CANDIDATE_POOL, n: int = 6) -> list[str]:
    """The six fine-grained categories, by a rule fixed BEFORE any FGOBJ outcome existed and using no outcome of any
    kind -- only cohort support and instance size:

      A  support   >= FGOBJ_MIN_ROLE_POS positives among the 400 calibration rows AND among the 600 test rows (the
                   10/10 rule the readability and answerability grades need), and >= FGOBJ_MIN_TRAIN_POS positives
                   among the 20,000 training rows (the training support of `bicycle`, the weakest of the six easy
                   concepts, so no FGOBJ probe is fitted on less than the campaign's own floor)
      B  difficulty  mean relative instance area < FGOBJ_MAX_MEAN_AREA (the MEDIAN of the six easy concepts' mean
                   relative areas), i.e. smaller than half the existing natural-image grid
      C  rank      calibration positives descending; ties by mean relative area ascending, then by COCO category id

    `pool` is the candidate pool; `stats` maps a category name to cal_pos / test_pos / train_pos / mean_rel_area /
    category_id (coco_category_stats)."""
    elig = []
    for name in pool:
        s = stats[name]
        if s["cal_pos"] < FGOBJ_MIN_ROLE_POS or s["test_pos"] < FGOBJ_MIN_ROLE_POS:
            continue
        if s["train_pos"] < FGOBJ_MIN_TRAIN_POS or s["mean_rel_area"] >= FGOBJ_MAX_MEAN_AREA:
            continue
        elig.append(name)
    elig.sort(key=lambda k: (-stats[k]["cal_pos"], stats[k]["mean_rel_area"], stats[k]["category_id"]))
    if len(elig) < n:
        raise SystemExit(f"only {len(elig)} of {len(pool)} candidates pass the FGOBJ support and size rule: {elig}")
    return elig[:n]


def coco_category_stats(data_root: Path = DATA_ROOT, names: tuple[str, ...] | None = None) -> dict[str, dict]:
    """Per COCO category, the frozen cohort's support and the mean relative instance area (instance area divided by
    the image area, averaged over every instance of the category in the cohort). Reads the instance annotations of
    both splits once; `names` restricts the output."""
    ann = data_root / FGOBJ_DATASET / "annotations"
    cohort = _cohort_rows(data_root)
    role_of = {r["row_id"]: r["role"] for r in cohort}
    split_of = {r["row_id"]: r["original_split"] for r in cohort}
    pos = defaultdict(lambda: defaultdict(set))          # name -> role -> row ids
    area_sum, area_n, cat_id = defaultdict(float), defaultdict(int), {}
    for split in ("train", "val"):
        d = json.loads((ann / f"instances_{split}2017.json").read_text(encoding="utf-8"))
        cat = {c["id"]: c["name"] for c in d["categories"]}
        cat_id.update({v: k for k, v in cat.items()})
        images = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            rid = f"{a['image_id']:012d}"
            if role_of.get(rid) is None or split_of[rid] != f"{split}2017":
                continue
            nm = cat[a["category_id"]]
            pos[nm][role_of[rid]].add(rid)
            im = images[a["image_id"]]
            area_sum[nm] += float(a["area"]) / float(im["width"] * im["height"])
            area_n[nm] += 1
    out = {}
    for nm, cid in sorted(cat_id.items(), key=lambda kv: kv[1]):
        if names is not None and nm not in names:
            continue
        out[nm] = {"category_id": cid, "train_pos": len(pos[nm]["train"]), "calibration_pos": len(pos[nm]["calibration"]),
                   "cal_pos": len(pos[nm]["calibration"]), "test_pos": len(pos[nm]["test"]),
                   "preflight_pos": len(pos[nm]["preflight"]), "n_instances": area_n[nm],
                   "mean_rel_area": (area_sum[nm] / area_n[nm]) if area_n[nm] else float("inf")}
    return out


# ------------------------------------------------------------------------------------------------ labels (append)
def _cohort_rows(data_root: Path) -> list[dict]:
    p = data_root / FGOBJ_DATASET / "manifests" / "cohort.csv"
    return list(csv.DictReader(p.open(newline="", encoding="utf-8")))


def coco_presence(data_root: Path, category_ids: dict[str, int], cohort: list[dict]) -> dict[str, set[str]]:
    """row_id -> set of category names present, by the rule of manifests.build_coco: presence from the instance
    annotations of the row's OWN original split, crowd instances count as present."""
    ann = data_root / FGOBJ_DATASET / "annotations"
    wanted = set(category_ids.values())
    cat_to_concept = {v: k for k, v in category_ids.items()}
    split_of = {r["row_id"]: r["original_split"] for r in cohort}
    present: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "val"):
        d = json.loads((ann / f"instances_{split}2017.json").read_text(encoding="utf-8"))
        for a in d["annotations"]:                        # crowd instances count as present (README 4)
            if a["category_id"] not in wanted:
                continue
            rid = f"{a['image_id']:012d}"
            if split_of.get(rid) == f"{split}2017":
                present[rid].add(cat_to_concept[a["category_id"]])
    return present


def fgobj_label_rows(data_root: Path) -> list[dict]:
    """The labels.csv rows the six fine-grained concepts add, in cohort order then FGOBJ_CONCEPTS order. Every other
    column (view/sex/age/width/height/type_id) is copied from the row's EXISTING label rows, so an appended row and a
    campaign row of the same image carry byte-identical metadata."""
    cohort = _cohort_rows(data_root)
    present = coco_presence(data_root, FGOBJ_CATEGORY_IDS, cohort)
    existing = {}
    for r in csv.DictReader((data_root / FGOBJ_DATASET / "manifests" / "labels.csv").open(newline="", encoding="utf-8")):
        if r["concept"] == CONCEPTS[FGOBJ_DATASET][0]:
            existing[r["row_id"]] = r
    rows = []
    for c in cohort:
        rid = c["row_id"]
        meta = existing.get(rid)
        if meta is None:
            raise SystemExit(f"{rid}: cohort row has no labels.csv row for {CONCEPTS[FGOBJ_DATASET][0]}; build the COCO "
                             f"manifests first (python -m cftransfer.manifests coco)")
        if meta["type_id"] != coco_type_id(int(meta["width"]), int(meta["height"])):
            raise SystemExit(f"{rid}: stored type_id {meta['type_id']} is not the protocol's bucket of "
                             f"{meta['width']}x{meta['height']}; the COCO manifests are not the frozen ones")
        for concept in FGOBJ_CONCEPTS:
            lab = "1" if concept in present.get(rid, ()) else "0"
            rows.append({"dataset_id": FGOBJ_DATASET, "row_id": rid, "concept": concept, "label": lab,
                         "label_known": "true", "label_raw": lab, "view": "", "sex": "", "age": "",
                         "width": meta["width"], "height": meta["height"], "type_id": meta["type_id"]})
    return rows


def build_fgobj_labels(data_root: Path = DATA_ROOT) -> dict:
    """Append the six fine-grained concepts to the COCO label manifest. Append-only and idempotent; cohort.csv, the
    cohort rows and their roles are never touched. Writes manifests/fgobj.json (provenance)."""
    out = data_root / FGOBJ_DATASET / "manifests"
    lab_path = out / "labels.csv"
    rows = fgobj_label_rows(data_root)
    want = [{c: str(r[c]) for c in LABEL_COLS} for r in rows]
    have = [{c: str(r[c]) for c in LABEL_COLS}
            for r in csv.DictReader(lab_path.open(newline="", encoding="utf-8")) if r["concept"] in set(FGOBJ_CONCEPTS)]
    appended = not have
    if have:
        if have != want:
            raise SystemExit(f"labels.csv already carries {len(have)} rows for {FGOBJ_CONCEPTS} that differ from this "
                             f"build; nothing written")
    else:
        with lab_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LABEL_COLS).writerows(rows)
    cohort = _cohort_rows(data_root)
    role_of = {r["row_id"]: r["role"] for r in cohort}
    counts = {c: {} for c in FGOBJ_CONCEPTS}
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        per[r["concept"]][role_of[r["row_id"]]][0 if r["label"] == "1" else 1] += 1
    for c in FGOBJ_CONCEPTS:
        counts[c] = {role: {"pos": per[c][role][0], "neg": per[c][role][1]} for role in ("train", "preflight", "calibration", "test")}
    ann = data_root / FGOBJ_DATASET / "annotations"
    prov = {
        "concepts": list(FGOBJ_CONCEPTS), "category_ids": dict(FGOBJ_CATEGORY_IDS),
        "candidate_pool": list(FGOBJ_CANDIDATE_POOL),
        "label_rule": "presence from the instance annotations of the row's own original split; crowd instances count as "
                      "present (manifests.build_coco, README 4); every label known",
        "selection_rule": {"min_role_positives": FGOBJ_MIN_ROLE_POS, "min_train_positives": FGOBJ_MIN_TRAIN_POS,
                           "max_mean_relative_area": FGOBJ_MAX_MEAN_AREA,
                           "rank": "calibration positives descending, ties by mean relative area ascending, then category id",
                           "note": "fixed before any FGOBJ outcome existed; uses cohort support and instance size only"},
        "append_only": "labels.csv appended; cohort.csv, the cohort rows and their roles unchanged; the six protocol "
                       "COCO concepts' rows are untouched",
        "rows_appended": len(rows) if appended else 0, "rows_present": len(want), "appended": appended,
        "labels_pos_neg_by_role": counts,
        "annotations_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(ann.glob("instances_*2017.json"))},
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / PROVENANCE).write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return prov


# ------------------------------------------------------------------------------------------------ directions
def fgobj_labels(dataset_id: str, ids: np.ndarray, concepts: list[str] | None = None) -> np.ndarray:
    """(K, n) manifest labels of the fine-grained concepts, -1 unknown (COCO labels are always known)."""
    concepts = list(concepts or FGOBJ_CONCEPTS)
    labels = load_labels(dataset_id)
    Y = np.full((len(concepts), len(ids)), -1, np.int8)
    for ci, c in enumerate(concepts):
        for j, rid in enumerate(ids):
            lab = labels.get((rid, c))
            if lab is None:
                raise FileNotFoundError(
                    f"labels.csv has no row for ({rid}, {c}); append the fine-grained concepts first: "
                    f"python -m cftransfer.fgobj --build-labels")
            if lab["label_known"] == "true":
                Y[ci, j] = int(lab["label"])
    return Y


def control_probes(fit: dict, Zs: np.ndarray, train_idx: np.ndarray, mask: np.ndarray, type_ix: np.ndarray) -> tuple:
    """The 20 type->random-label control probes for a concept whose known-training mask is `mask`.

    The campaign fits one set per concept on that concept's known-label training subset. When the mask equals a mask
    the seed-0 fit already recorded (on COCO every label is known, so all six coincide), the STORED control
    coefficients are reused verbatim -- identical to the campaign's, and no refit. Otherwise the controls are refitted
    with the same rule. Returns (coefficients (20, 512), intercepts (20,), source)."""
    stored = fit.get("control_train_mask")
    if stored is not None and bool(fit["controls_eligible"]):
        for j in range(stored.shape[0]):
            if np.array_equal(stored[j], mask):
                return fit["control_coefficients"][j], fit["control_intercepts"][j], f"seed0 control probes of concept {j}"
    ctrl_labels = fit["control_type_labels"]
    coef = np.zeros((len(CONTROL_SEEDS), Zs.shape[1]), np.float32)
    inter = np.zeros(len(CONTROL_SEEDS), np.float32)
    tr_types = type_ix[train_idx][mask]
    for k in range(len(CONTROL_SEEDS)):
        yk = ctrl_labels[k][tr_types]
        if yk.min() == yk.max():
            coef[k] = np.nan
            continue
        ck = fit_lr(Zs[train_idx][mask], yk)
        coef[k], inter[k] = ck.coef_[0], ck.intercept_[0]
    return coef, inter, "refitted on this concept's known-label training rows"


def build_fgobj(model_key: str, dataset_id: str = FGOBJ_DATASET, locus_id: str = "vis.last",
                out_dir: Path | None = None, concepts: list[str] | None = None) -> tuple[Path, dict]:
    t0 = time.time()
    if dataset_id != FGOBJ_DATASET:
        raise SystemExit(f"FGOBJ is NOT_REQUESTED for {dataset_id} (fine-grained OBJECT concepts live on COCO)")
    concepts = list(concepts or FGOBJ_CONCEPTS)
    K = len(concepts)
    fit = load_fit(model_key, dataset_id, locus_id, 0)
    easy = list(fit["concept_names"].astype(str))
    if easy != list(CONCEPTS[dataset_id]):
        raise RuntimeError("seed0.npz concept order differs from the protocol")
    ids, X, _vtc, roles = load_features(model_key, dataset_id, locus_id)
    train_idx = np.array([i for i, r in enumerate(roles) if r == "train"])
    cal_idx = np.array([i for i, r in enumerate(roles) if r == "calibration"])
    test_idx = np.array([i for i, r in enumerate(roles) if r == "test"])
    if not np.array_equal(ids[train_idx], fit["train_row_ids"].astype(str)):
        raise RuntimeError("training rows of features/<locus>.npz differ from the fit's train_row_ids")
    P, mu, s = fit["projection"], fit["scaler_mean"], fit["scaler_scale"]
    D = X.shape[1]
    Zs = ((X.astype(np.float32) @ P - mu) / np.maximum(s, 1e-8)).astype(np.float32)
    Y = fgobj_labels(dataset_id, ids, concepts)
    labels = load_labels(dataset_id)
    types = np.array([labels[(rid, easy[0])]["type_id"] for rid in ids])
    type_names = list(fit["type_names"].astype(str))
    type_ix = np.array([type_names.index(t) if t in type_names else -1 for t in types])
    ctrl_labels = fit["control_type_labels"]
    controls_eligible = bool(fit["controls_eligible"])

    coef = np.zeros((K, Zs.shape[1]), np.float32); inter = np.zeros(K, np.float32)
    n_pos = np.zeros((K, 3), np.int64); n_neg = np.zeros((K, 3), np.int64)          # train / calibration / test
    auc_cal = np.full(K, np.nan); auc_test = np.full(K, np.nan)
    cal_real = np.full((K, len(cal_idx)), np.nan, np.float32)
    cal_ctrl = np.full((K, len(CONTROL_SEEDS), len(cal_idx)), np.nan, np.float32)
    ctrl_source = []
    for ci, c in enumerate(concepts):
        m = Y[ci][train_idx] >= 0
        yy = Y[ci][train_idx][m]
        for ri, idx in enumerate((train_idx, cal_idx, test_idx)):
            kn = Y[ci][idx] >= 0
            n_pos[ci, ri], n_neg[ci, ri] = int((Y[ci][idx][kn] == 1).sum()), int((Y[ci][idx][kn] == 0).sum())
        if yy.min() == yy.max():
            raise RuntimeError(f"{c}: single-class training labels")
        clf = fit_lr(Zs[train_idx][m], yy)
        coef[ci], inter[ci] = clf.coef_[0], clf.intercept_[0]
        for idx, store in ((cal_idx, auc_cal), (test_idx, auc_test)):
            kn = Y[ci][idx] >= 0
            if kn.sum() and Y[ci][idx][kn].min() != Y[ci][idx][kn].max():
                store[ci] = float(roc_auc_score(Y[ci][idx][kn], Zs[idx][kn] @ coef[ci] + inter[ci]))
        cal_real[ci] = Zs[cal_idx] @ coef[ci] + inter[ci]
        if controls_eligible:
            cc, cb, src = control_probes(fit, Zs, train_idx, m, type_ix)
            ctrl_source.append(src)
            for k in range(len(CONTROL_SEEDS)):
                if np.isfinite(cc[k]).all():
                    cal_ctrl[ci, k] = Zs[cal_idx] @ cc[k] + cb[k]
    vec = np.stack([direction_from_projected(P, s, coef[ci]) for ci in range(K)])
    # the family's OWN random directions and shams: fit.fit_locus's construction re-drawn from PCG64(FGOBJ_RANDOM_SEED)
    rng = np.random.default_rng(FGOBJ_RANDOM_SEED)
    R = rng.standard_normal((N_RANDOM, D))
    R = (R / np.linalg.norm(R, axis=1, keepdims=True)).astype(np.float32)
    perms = np.stack([rng.permutation(D) for _ in range(K)]).astype(np.int32)
    shams = np.stack([vec[ci][perms[ci]] for ci in range(K)]).astype(np.float32)
    cos_model = vec.astype(np.float64) @ fit["clinical_vectors"].astype(np.float64).T
    c64, c6 = coef.astype(np.float64), fit["coefficients"].astype(np.float64)
    cos_proj = (c64 @ c6.T) / (np.linalg.norm(c64, axis=1)[:, None] * np.linalg.norm(c6, axis=1)[None, :])
    cal_ctrl_labels = np.where(type_ix[cal_idx][None, :] >= 0, ctrl_labels[:, np.maximum(type_ix[cal_idx], 0)], -1).astype(np.int8)
    arrays = {
        "fgobj_names": np.array(concepts), "concept_names": fit["concept_names"], "coefficients": coef, "intercepts": inter,
        "fgobj_vectors": vec, "random_vectors": R, "sham_vectors": shams, "sham_permutations": perms,
        "random_seed": np.int64(FGOBJ_RANDOM_SEED), "n_random": np.int64(N_RANDOM), "fit_seed": np.int64(0),
        "projection_seed": fit["projection_seed"], "n_pos": n_pos, "n_neg": n_neg,
        "auroc_calibration": auc_cal, "auroc_test": auc_test, "cos_model": cos_model, "cos_projected": cos_proj,
        "controls_eligible": np.bool_(controls_eligible), "controls_source": np.array(sorted(set(ctrl_source)) or ["none"]),
        "cal_row_ids": ids[cal_idx], "cal_labels": Y[:, cal_idx], "cal_real_logits": cal_real,
        "cal_control_logits": cal_ctrl, "cal_control_labels": cal_ctrl_labels,
        "test_row_ids": ids[test_idx], "test_labels": Y[:, test_idx],
        "train_row_ids": fit["train_row_ids"], "locus_id": np.array(locus_id)}
    out_dir = Path(out_dir) if out_dir else fits_dir(model_key, dataset_id, locus_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FGOBJ_FILE
    np.savez(path, **arrays)
    print(f"[{model_key}/{dataset_id}/{locus_id}] fgobj written to {path} in {time.time() - t0:.1f}s", flush=True)
    return path, arrays


def load_fgobj(model_key: str, dataset_id: str, locus_id: str) -> dict:
    path = fits_dir(model_key, dataset_id, locus_id) / FGOBJ_FILE
    if not path.exists():
        raise FileNotFoundError(f"FGOBJ needs {path}; compute it first (CPU, seconds): "
                                f"python -m cftransfer.fgobj --model-key {model_key} --dataset {dataset_id}")
    return dict(np.load(path, allow_pickle=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-key")
    ap.add_argument("--dataset", default=FGOBJ_DATASET)
    ap.add_argument("--locus", default="vis.last")
    ap.add_argument("--out-dir", default=None, help="write fgobj_seed0.npz here instead of fits/<locus>/ (verification runs)")
    ap.add_argument("--build-labels", action="store_true", help="append the six concepts to the COCO label manifest and exit")
    ap.add_argument("--show-selection", action="store_true", help="recompute the candidate statistics and the selection rule")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    a = ap.parse_args()
    if a.show_selection:
        st = coco_category_stats(Path(a.data_root))
        pick = select_fgobj_concepts(st)
        for nm in FGOBJ_CANDIDATE_POOL:
            s = st[nm]
            print(f"  {nm:16s} id={s['category_id']:3d} train={s['train_pos']:6d} cal={s['cal_pos']:4d} test={s['test_pos']:4d} "
                  f"mean_rel_area={s['mean_rel_area']:.5f}  {'SELECTED' if nm in pick else ''}")
        print(json.dumps({"selected": pick, "frozen": list(FGOBJ_CONCEPTS), "agrees": pick == list(FGOBJ_CONCEPTS)}))
    elif a.build_labels:
        prov = build_fgobj_labels(Path(a.data_root))
        print(json.dumps({k: v for k, v in prov.items() if k != "labels_pos_neg_by_role"}, indent=1))
        for c, roles in prov["labels_pos_neg_by_role"].items():
            print(f"  {c:16s} " + "  ".join(f"{r}={v['pos']}/{v['pos'] + v['neg']}" for r, v in roles.items()))
    else:
        if not a.model_key:
            ap.error("--model-key is required unless --build-labels or --show-selection is given")
        path, arr = build_fgobj(a.model_key, a.dataset, a.locus, a.out_dir)
        six = arr["concept_names"].astype(str)
        for ci, c in enumerate(arr["fgobj_names"].astype(str)):
            cos = " ".join(f"{d}={arr['cos_model'][ci, di]:+.3f}" for di, d in enumerate(six))
            print(f"  {c:16s} train={int(arr['n_pos'][ci, 0]):6d}/{int(arr['n_pos'][ci, 0] + arr['n_neg'][ci, 0]):6d} "
                  f"cal={int(arr['n_pos'][ci, 1]):4d} test={int(arr['n_pos'][ci, 2]):4d} "
                  f"auroc_cal={arr['auroc_calibration'][ci]:.3f} auroc_test={arr['auroc_test'][ci]:.3f}  cos_easy: {cos}")
        print(json.dumps({"path": str(path), "controls_eligible": bool(arr["controls_eligible"]),
                          "controls_source": list(arr["controls_source"].astype(str))}))
