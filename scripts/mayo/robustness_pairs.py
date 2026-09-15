#!/usr/bin/env python
"""Shared-tower paired deltas, ceiling marking, tower sharing, effect floors, readability ranking and the
denominator audit over the cf-transfer campaign grid.

Read-only over <RUN_ROOT>/<model_key>/<dataset>/ (summary.json, outcomes/CORE.parquet, outcomes/REFIT.parquet,
features/vis.last.npz, fits/vis.last/seed0.npz, manifests/cohort.csv). Writes <RUN_ROOT>/robustness/pairs.json and
pairs.md only.

A block enters the write-matrix statistics when it is included under the paper's one rule
(cftransfer.manifest.block_included: CORE and CALIBRATION in run.json completed_modules) AND its summary.json carries
core.W for all six concepts with a verdict per concept AND a merged outcomes/CORE.parquet exists. Blocks that fail the
rule (e.g. llama32-11/nih, whose CORE is incomplete and whose partial core.W was rebuilt from unmerged shards) are
excluded and flagged in the denominator audit, matching the manifest's write-matrix cell count.

Tasks (numbered as in the request):
  1. shared-tower paired deltas   Delta O_q = O_q(larger) - O_q(smaller) for the Gemma-3 / MedGemma size pairs on every
                                  dataset, paired patient bootstrap (the campaign's core_bootstrap_indices, seed
                                  BOOT_CORE_SEED, 2,000 draws; test rows are one per unit on all three datasets),
                                  max competitor recomputed inside every draw; Delta W_qq and Delta max competitor too;
                                  |Delta O_q| against the blocks' refit SD (summary t3 all|median_refit_O_sd)
  2. ceiling marking              share of test rows with clean P(yes) > 0.99 or < 0.01 per (block, concept); cells
                                  above 0.9; margin-scale ownership O^m_q with its interval for gemma3-4 nih/chexpert
  3. tower sharing                md5 of the pooled vis.last test features per size within a family; max abs
                                  difference and the cosine between the fitted logistic write vectors per concept
  4. effect floor and stability   owned cells with W_qq >= 0.05 and O_q >= 0.05; owned cells with O_q > 0 under both
                                  refit seeds; all cells with O_q >= 0.05 and W_qq >= 0.05
  5. readability ranking          mean probe selectivity S per concept over chest blocks, NIH and CheXpert separately
  6. denominator audit            probe-graded / write-matrix / owned / answer-capable / readable cells per dataset
                                  with the contributing blocks

Usage (from the concept-flow repo):
  PYTHONPATH=src python scripts/mayo/robustness_pairs.py [--draws 2000] [--jobs 4] [--out <dir>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from cftransfer.analysis import _O_from_delta, _load_module, _matrix, core_bootstrap_indices          # noqa: E402
from cftransfer.manifest import block_included                                                        # noqa: E402
from cftransfer.protocol import BOOT_CORE_SEED, CONCEPTS, DATASETS, MODEL_ORDER, PRIMARY_ALPHA, primary_template  # noqa: E402
from cftransfer.runpaths import RUN_ROOT, run_dir                                                      # noqa: E402

OUT_DIR = RUN_ROOT / "robustness"
SKIP_DIRS = {"robustness", "figures"}
CHEST = ("nih", "chexpert")
EXCLUDED_MODELS: tuple = ()                # filled at discovery: models with no block under the inclusion rule
PAIRS = [("gemma3-4", "gemma3-12"), ("gemma3-12", "gemma3-27"), ("gemma3-4", "gemma3-27"), ("medgemma-4", "medgemma-27")]
FAMILIES = {
    "qwen25vl": ["q25-3", "q25-7", "q25-32", "q25-72"],
    "qwen3vl": ["q3-4", "q3-8", "q3-32"],
    "internvl35": ["iv35-8", "iv35-14", "iv35-38"],
    "lingshu": ["lingshu-7", "lingshu-32"],
    "llava15": ["llava15-7", "llava15-13"],
    "gemma3": ["gemma3-4", "gemma3-12", "gemma3-27"],
    "medgemma": ["medgemma-4", "medgemma-27"],
}
DRAWS = 2000
FLOOR = 0.05                      # effect floor on W_qq and O_q (p scale)
CEIL_HI, CEIL_LO, CEIL_SHARE = 0.99, 0.01, 0.9
TOWER_ABS_TOL, TOWER_COS_TOL = 1e-2, 0.998   # write vectors (unit norm): max |diff| and per-concept cosine
TOWER_REL_TOL = 2.0 ** -7                    # features: bf16 ulp, relative to the element magnitude
TOWER_REL_FLOOR = 1e-3                       # magnitude floor as a fraction of max|x| (near-zero elements)
CEILING_EXAMPLES = (("gemma3-4", "nih"), ("gemma3-4", "chexpert"))


# ------------------------------------------------------------------------------------------------ helpers
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def f4(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def test_order(model_key: str, dataset_id: str) -> dict[str, int]:
    """Frozen test order from the block's own manifests/cohort.csv (data-root copy as fallback)."""
    p = run_dir(model_key, dataset_id) / "manifests" / "cohort.csv"
    if not p.exists():
        from cftransfer.images import load_cohort
        rows = load_cohort(dataset_id, ("test",))
    else:
        rows = [r for r in csv.DictReader(p.open(newline="", encoding="utf-8")) if r["role"] == "test"]
        rows.sort(key=lambda r: int(r["order"]))
    return {r["row_id"]: i for i, r in enumerate(rows)}


def clinical_stack(delta: dict, concepts) -> np.ndarray:
    """(6, 6, n) per-sample deltas delta[(q, concept:d)]; raises when a cell is missing."""
    return np.stack([np.stack([delta[(q, f"concept:{d}")] for d in concepts]) for q in concepts]).astype(float)


def count_matrix(idx: np.ndarray) -> np.ndarray:
    """(B, n) multiplicity of every row in every bootstrap draw."""
    B, n = idx.shape
    C = np.zeros((B, n))
    np.add.at(C, (np.repeat(np.arange(B), n), idx.ravel()), 1.0)
    return C


def boot_means(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """nan-aware bootstrap means: X (k, n) -> (B, k), means recomputed inside every draw."""
    F = np.isfinite(X)
    Xz = np.where(F, X, 0.0)
    return (C @ Xz.T) / (C @ F.T.astype(float))


def ownership_from_matrix(M: np.ndarray) -> dict:
    """M (..., k, k) write matrix -> diag, max competitor (off-diagonal max), O = diag - max competitor."""
    k = M.shape[-1]
    diag = np.diagonal(M, axis1=-2, axis2=-1)
    off = np.where(np.eye(k, dtype=bool), -np.inf, M)
    mx = off.max(axis=-1)
    return {"W_qq": diag, "max_competitor": mx, "O": diag - mx}


def pct_ci(draws: np.ndarray) -> list[float]:
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


# --------------------------------------------------------------------------------------------- discovery
def discover() -> list[dict]:
    blocks = []
    for sj in sorted(RUN_ROOT.glob("*/*/summary.json")):
        mk, ds = sj.parts[-3], sj.parts[-2]
        if mk in SKIP_DIRS or ds not in DATASETS:
            continue
        s = json.loads(sj.read_text())
        rj = sj.parent / "run.json"
        included = rj.exists() and block_included(json.loads(rj.read_text()))
        cs = CONCEPTS[ds]
        core = s.get("core") or {}
        W = core.get("W") or {}
        pq_ = core.get("per_question") or {}
        w_complete = all(f"concept:{d}" in W.get(q, {}) for q in cs for d in cs)
        verdicts = all(pq_.get(q, {}).get("verdict") for q in cs)
        parquet = (sj.parent / "outcomes" / "CORE.parquet").exists()
        cal = s.get("calibration") or {}
        rec = {"model": mk, "dataset": ds, "primary_template": s.get("primary_template") or primary_template(sj.parent),
               "calibration_present": all(q in cal for q in cs),
               "core_W_in_summary": w_complete, "core_verdicts_in_summary": verdicts, "core_parquet": parquet,
               "refit_parquet": (sj.parent / "outcomes" / "REFIT.parquet").exists(),
               "excluded_model": not included,       # excluded under the inclusion rule (CORE and CALIBRATION completed)
               "refit_sd": (s.get("t3") or {}).get("all|median_refit_O_sd", {}).get("estimate"),
               "summary": s}
        rec["write_matrix"] = bool(w_complete and verdicts and parquet and not rec["excluded_model"])
        rec["write_matrix_reason"] = ("" if rec["write_matrix"] else
                                      "not included (run.json completed_modules lacks CORE or CALIBRATION)" if rec["excluded_model"] else
                                      "core.W incomplete in summary" if not w_complete else
                                      "no verdicts in summary" if not verdicts else "no merged outcomes/CORE.parquet")
        blocks.append(rec)
    # blocks without a summary
    for d in sorted(RUN_ROOT.glob("*/*/")):
        mk, ds = d.parts[-2], d.parts[-1]
        if mk in SKIP_DIRS or ds not in DATASETS or (d / "summary.json").exists():
            continue
        blocks.append({"model": mk, "dataset": ds, "primary_template": None, "calibration_present": False,
                       "core_W_in_summary": False, "core_verdicts_in_summary": False, "core_parquet": (d / "outcomes" / "CORE.parquet").exists(),
                       "refit_parquet": False, "excluded_model": True, "refit_sd": None, "summary": {},
                       "write_matrix": False, "write_matrix_reason": "no summary.json"})
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    blocks.sort(key=lambda b: (order.get(b["model"], 999), b["model"], DATASETS.index(b["dataset"])))
    global EXCLUDED_MODELS
    EXCLUDED_MODELS = tuple(sorted({b["model"] for b in blocks} - {b["model"] for b in blocks if not b["excluded_model"]}))
    return blocks


# ----------------------------------------------------------------------------------------- block loading
def load_block(args) -> dict:
    """Per-sample clinical deltas (p and margin), clean P(yes), refit-seed ownership. Runs in a worker."""
    mk, ds, primary = args
    t0 = time.time()
    cs = CONCEPTS[ds]
    order = test_order(mk, ds)
    n = len(order)
    core = _load_module(mk, ds, "CORE")
    core = core[(core.template_id == primary) & (core.fit_seed == 0)]
    base = core[core.direction_id == "baseline"]
    dp = _matrix(core, cs, order, PRIMARY_ALPHA, template_id=primary, baseline=base)
    dm = _matrix(core, cs, order, PRIMARY_ALPHA, template_id=primary, baseline=base, value="semantic_margin")
    P0 = np.full((len(cs), n), np.nan)
    for qi, q in enumerate(cs):
        g = base[base.concept == q]
        keep = [i for i, r in enumerate(g.row_id.values) if r in order]
        P0[qi, [order[r] for r in g.row_id.values[keep]]] = g.p_present.values[keep]
    out = {"model": mk, "dataset": ds, "n_rows": n, "row_ids": [None] * n, "Dp": clinical_stack(dp, cs), "Dm": clinical_stack(dm, cs), "P0": P0}
    for r, i in order.items():
        out["row_ids"][i] = r
    out["min_finite_clinical"] = int(np.isfinite(out["Dp"]).sum(axis=2).min())
    # refit seeds 1, 2 against the CORE baseline (robustness_scale approach)
    refit = _load_module(mk, ds, "REFIT")
    out["refit_O"] = {}
    if refit is not None and not refit.empty:
        refit = refit[refit.template_id == primary]
        for k in (1, 2):
            dk = _matrix(refit, cs, order, PRIMARY_ALPHA, template_id=primary, fit_seed=k, baseline=base)
            o = _O_from_delta(dk, cs)
            if o is not None:
                out["refit_O"][k] = o
    out["seconds"] = round(time.time() - t0, 1)
    return out


def test_features(mk: str, ds: str, order: dict[str, int]):
    """(x_test (600, D) float16 in frozen test order, md5 of its bytes)."""
    f = np.load(run_dir(mk, ds) / "features" / "vis.last.npz", allow_pickle=False)
    ids = f["row_id"].astype(str)
    pos = {r: i for i, r in enumerate(ids)}
    missing = [r for r in order if r not in pos]
    if missing:
        raise RuntimeError(f"{mk}/{ds}: {len(missing)} test rows missing from features")
    sel = np.array([pos[r] for r in sorted(order, key=order.get)])
    x = np.ascontiguousarray(f["x"][sel])
    return x, hashlib.md5(x.tobytes()).hexdigest()


def write_vectors(mk: str, ds: str) -> np.ndarray:
    z = np.load(run_dir(mk, ds) / "fits" / "vis.last" / "seed0.npz", allow_pickle=False)
    names = [str(c) for c in z["concept_names"]]
    assert names == CONCEPTS[ds], (mk, ds, names)
    return z["clinical_vectors"].astype(np.float64)


def feature_diff(xa: np.ndarray, xb: np.ndarray) -> dict:
    """Element-wise comparison of two same-width feature blocks (float16 as stored, compared in float32)."""
    A, B = xa.astype(np.float32), xb.astype(np.float32)
    diff = np.abs(A - B)
    mag = np.maximum(np.abs(A), np.abs(B))
    floor = TOWER_REL_FLOOR * float(mag.max())
    rel = diff / np.maximum(mag, floor)
    nz = diff > 0
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "differing_elements": int(nz.sum()), "differing_fraction": float(nz.mean()),
            "rows_identical_fraction": float((diff.max(axis=1) == 0).mean()),
            "max_rel_diff": float(rel.max()), "rel_floor_abs": floor,
            "features_bf16_equal": bool(rel.max() <= TOWER_REL_TOL)}


def compare_vectors(Va: np.ndarray, Vb: np.ndarray) -> dict:
    cos = [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)) for a, b in zip(Va, Vb)]
    return {"max_abs_diff": float(np.abs(Va - Vb).max()), "identical": bool(np.array_equal(Va, Vb)),
            "cos_per_concept": cos, "min_cos": float(min(cos))}


# ----------------------------------------------------------------------------------------------- task 1
def paired_deltas(small: dict, large: dict, ds: str, C: np.ndarray, sd_small, sd_large) -> dict:
    cs = CONCEPTS[ds]
    k = len(cs)
    stats = {}
    for name, blk in (("small", small), ("large", large)):
        X = blk["Dp"].reshape(k * k, -1)
        point = ownership_from_matrix(np.nanmean(X, axis=1).reshape(k, k))
        draws = ownership_from_matrix(boot_means(X, C).reshape(-1, k, k))
        stats[name] = {"point": point, "draws": draws}
    sds = [s for s in (sd_small, sd_large) if isinstance(s, (int, float))]
    sd_ref = float(np.mean(sds)) if sds else None
    per = {}
    n_excl = {"O": 0, "W_qq": 0, "max_competitor": 0}
    for qi, q in enumerate(cs):
        cell = {}
        for key in ("O", "W_qq", "max_competitor"):
            est = float(stats["large"]["point"][key][qi] - stats["small"]["point"][key][qi])
            dr = stats["large"]["draws"][key][:, qi] - stats["small"]["draws"][key][:, qi]
            lo, hi = pct_ci(dr)
            excl = bool(lo > 0 or hi < 0)
            n_excl[key] += excl
            cell[f"delta_{key}"] = {"estimate": est, "ci95": [lo, hi], "excludes_zero": excl,
                                    "small": float(stats["small"]["point"][key][qi]), "large": float(stats["large"]["point"][key][qi])}
        cell["abs_delta_O_over_refit_sd"] = (abs(cell["delta_O"]["estimate"]) / sd_ref) if sd_ref else None
        per[q] = cell
    abs_dO = [abs(per[q]["delta_O"]["estimate"]) for q in cs]
    return {"per_concept": per, "n_concepts_delta_O_ci_excludes_zero": n_excl["O"],
            "n_concepts_delta_W_qq_ci_excludes_zero": n_excl["W_qq"],
            "n_concepts_delta_max_competitor_ci_excludes_zero": n_excl["max_competitor"],
            "median_abs_delta_O": float(np.median(abs_dO)), "max_abs_delta_O": float(np.max(abs_dO)),
            "refit_sd_small": sd_small, "refit_sd_large": sd_large, "refit_sd_reference": sd_ref,
            "median_abs_delta_O_over_refit_sd": (float(np.median(abs_dO)) / sd_ref) if sd_ref else None,
            "max_abs_delta_O_over_refit_sd": (float(np.max(abs_dO)) / sd_ref) if sd_ref else None,
            "n_concepts_abs_delta_O_above_refit_sd": (int(sum(a > sd_ref for a in abs_dO)) if sd_ref else None)}


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    t0 = time.time()
    blocks = discover()
    by_key = {(b["model"], b["dataset"]): b for b in blocks}
    wm = [b for b in blocks if b["write_matrix"]]
    print(f"{len(blocks)} blocks surveyed, {len(wm)} with a write matrix", flush=True)

    # ---- load per-sample deltas for every write-matrix block
    jobs = [(b["model"], b["dataset"], b["primary_template"]) for b in wm]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            loaded = list(ex.map(load_block, jobs))
    else:
        loaded = [load_block(j) for j in jobs]
    data = {(d["model"], d["dataset"]): d for d in loaded}
    for d in loaded:
        print(f"  loaded {d['model']:12s} {d['dataset']:9s} finite>= {d['min_finite_clinical']}/{d['n_rows']} refit seeds {sorted(d['refit_O'])} {d['seconds']}s", flush=True)

    # ---- shared bootstrap indices (campaign helper, BOOT_CORE_SEED)
    idx = {ds: core_bootstrap_indices(ds, 600, a.draws) for ds in DATASETS}
    Cm = {ds: count_matrix(idx[ds]) for ds in DATASETS}
    boot_meta = {"helper": "cftransfer.analysis.core_bootstrap_indices(dataset, 600, draws)", "seed": BOOT_CORE_SEED, "draws": a.draws,
                 "unit": "test row = one unit (patient on NIH/CheXpert, image on COCO); 600 unique units per dataset",
                 "interval": "percentile 2.5/97.5 over paired draws, max competitor recomputed inside every draw"}

    # ================================================================================ task 3: tower sharing
    feats, vecs = {}, {}

    def get_feat(mk, ds):
        if (mk, ds) not in feats:
            feats[(mk, ds)] = test_features(mk, ds, test_order(mk, ds))
        return feats[(mk, ds)]

    def get_vec(mk, ds):
        if (mk, ds) not in vecs:
            vecs[(mk, ds)] = write_vectors(mk, ds)
        return vecs[(mk, ds)]

    tower = {}
    for fam, sizes in FAMILIES.items():
        tower[fam] = {"sizes": sizes, "per_dataset": {}}
        for ds in DATASETS:
            present = [m for m in sizes if (run_dir(m, ds) / "features" / "vis.last.npz").exists()]
            if len(present) < 2:
                tower[fam]["per_dataset"][ds] = {"present": present, "note": "fewer than two sizes with features"}
                continue
            info, pairs = {}, []
            for m in present:
                x, h = get_feat(m, ds)
                info[m] = {"D": int(x.shape[1]), "md5": h, "feature_abs_max": float(np.abs(x.astype(np.float32)).max()),
                           "feature_abs_mean": float(np.abs(x.astype(np.float32)).mean()),
                           "fit_seed0": (run_dir(m, ds) / "fits" / "vis.last" / "seed0.npz").exists()}
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    ma, mb = present[i], present[j]
                    xa, ha = get_feat(ma, ds); xb, hb = get_feat(mb, ds)
                    rec = {"a": ma, "b": mb, "same_width": xa.shape[1] == xb.shape[1], "md5_identical": ha == hb}
                    if rec["same_width"]:
                        rec.update(feature_diff(xa, xb))
                        if info[ma]["fit_seed0"] and info[mb]["fit_seed0"]:
                            rec["write_vectors"] = compare_vectors(get_vec(ma, ds), get_vec(mb, ds))
                        rec["shared_tower_bf16"] = bool(rec["features_bf16_equal"] and
                                                        rec.get("write_vectors", {}).get("min_cos", -1) > TOWER_COS_TOL)
                    else:
                        rec["max_abs_diff"] = None; rec["features_bf16_equal"] = False; rec["shared_tower_bf16"] = False
                        rec["note"] = f"different width ({xa.shape[1]} vs {xb.shape[1]}): different tower"
                    rec["relationship"] = "bit-identical" if rec["md5_identical"] else ("bf16-equal" if rec["shared_tower_bf16"] else "different")
                    pairs.append(rec)
            # connected components under the shared relation
            groups, seen = [], set()
            for m in present:
                if m in seen:
                    continue
                comp, stack = [], [m]
                while stack:
                    u = stack.pop()
                    if u in seen:
                        continue
                    seen.add(u); comp.append(u)
                    for p in pairs:
                        if p["shared_tower_bf16"]:
                            v = p["b"] if p["a"] == u else p["a"] if p["b"] == u else None
                            if v and v not in seen:
                                stack.append(v)
                groups.append(sorted(comp, key=present.index))
            tower[fam]["per_dataset"][ds] = {"present": present, "blocks": info, "pairs": pairs, "shared_groups": groups}
        # family verdict: groups consistent across datasets?
        gs = [tuple(tuple(g) for g in v["shared_groups"]) for v in tower[fam]["per_dataset"].values() if "shared_groups" in v]
        tower[fam]["shared_groups_consistent_across_datasets"] = bool(gs and all(g == gs[0] for g in gs))
        tower[fam]["shared_groups"] = [list(g) for g in gs[0]] if gs else []
    print("tower sharing done", flush=True)

    # ================================================================================ task 1: paired deltas
    pairs_out = []
    for sm, lg in PAIRS:
        for ds in DATASETS:
            rec = {"pair": f"{sm}->{lg}", "small": sm, "large": lg, "dataset": ds}
            if (sm, ds) not in data or (lg, ds) not in data:
                rec["status"] = "SKIPPED"; rec["reason"] = "one block lacks a write matrix"
                pairs_out.append(rec); continue
            A, B = data[(sm, ds)], data[(lg, ds)]
            rec["same_test_rows"] = bool(A["row_ids"] == B["row_ids"])
            rec["primary_template"] = [by_key[(sm, ds)]["primary_template"], by_key[(lg, ds)]["primary_template"]]
            rec["write_vectors"] = compare_vectors(get_vec(sm, ds), get_vec(lg, ds))
            fa, fb = get_feat(sm, ds), get_feat(lg, ds)
            rec["features"] = {"md5_identical": fa[1] == fb[1], **(feature_diff(fa[0], fb[0]) if fa[0].shape == fb[0].shape else {"max_abs_diff": None, "features_bf16_equal": False})}
            rec["shared_write_vectors_bf16"] = bool(rec["write_vectors"]["max_abs_diff"] < TOWER_ABS_TOL and rec["write_vectors"]["min_cos"] > TOWER_COS_TOL)
            rec["relationship"] = ("bit-identical" if rec["features"]["md5_identical"] and rec["write_vectors"]["identical"] else
                                   "bf16-equal" if rec["features"]["features_bf16_equal"] and rec["shared_write_vectors_bf16"] else "different")
            rec.update(paired_deltas(A, B, ds, Cm[ds], by_key[(sm, ds)]["refit_sd"], by_key[(lg, ds)]["refit_sd"]))
            rec["status"] = "OK"
            pairs_out.append(rec)
            print(f"  pair {sm}->{lg} {ds}: dO CI excl 0 in {rec['n_concepts_delta_O_ci_excludes_zero']}/6, median|dO| {rec['median_abs_delta_O']:.4f}", flush=True)

    # ================================================================================ task 2: ceiling marking
    ceiling_cells, ceiling_blocks = [], []
    for d in loaded:
        cs = CONCEPTS[d["dataset"]]
        P0 = d["P0"]
        sat = (P0 > CEIL_HI) | (P0 < CEIL_LO)
        fin = np.isfinite(P0)
        shares = sat.sum(axis=1) / np.maximum(fin.sum(axis=1), 1)
        hi = (P0 > CEIL_HI).sum(axis=1) / np.maximum(fin.sum(axis=1), 1)
        lo = (P0 < CEIL_LO).sum(axis=1) / np.maximum(fin.sum(axis=1), 1)
        pq_ = by_key[(d["model"], d["dataset"])]["summary"]["core"]["per_question"]
        for qi, q in enumerate(cs):
            ceiling_cells.append({"model": d["model"], "dataset": d["dataset"], "concept": q, "share": float(shares[qi]),
                                  "share_high": float(hi[qi]), "share_low": float(lo[qi]), "ceiling": bool(shares[qi] > CEIL_SHARE),
                                  "owned": bool(pq_[q].get("verdict") == "fixed_family_advantage" and pq_[q].get("steering_reference")),
                                  "O_q": pq_[q].get("O_q"), "W_qq": pq_[q].get("W_qq")})
        ceiling_blocks.append({"model": d["model"], "dataset": d["dataset"], "share_pooled": float(sat[fin].mean()),
                               "n_ceiling_concepts": int((shares > CEIL_SHARE).sum())})
    ceiling_examples = {}
    for mk, ds in CEILING_EXAMPLES:
        if (mk, ds) not in data:
            ceiling_examples[f"{mk}/{ds}"] = {"status": "no write matrix"}; continue
        d = data[(mk, ds)]; cs = CONCEPTS[ds]; k = len(cs)
        Xm = d["Dm"].reshape(k * k, -1); Xp = d["Dp"].reshape(k * k, -1)
        pm = ownership_from_matrix(np.nanmean(Xm, axis=1).reshape(k, k)); dm_ = ownership_from_matrix(boot_means(Xm, Cm[ds]).reshape(-1, k, k))
        pp = ownership_from_matrix(np.nanmean(Xp, axis=1).reshape(k, k)); dp_ = ownership_from_matrix(boot_means(Xp, Cm[ds]).reshape(-1, k, k))
        Wm = np.nanmean(Xm, axis=1).reshape(k, k)
        cells = {}
        for qi, q in enumerate(cs):
            others = {cs[di]: float(Wm[qi, di]) for di in range(k) if di != qi}
            cells[q] = {"O_m": float(pm["O"][qi]), "O_m_ci95": pct_ci(dm_["O"][:, qi]), "O_m_ci_excludes_zero": bool(pct_ci(dm_["O"][:, qi])[0] > 0 or pct_ci(dm_["O"][:, qi])[1] < 0),
                        "W_m_qq": float(pm["W_qq"][qi]), "W_m_qq_ci95": pct_ci(dm_["W_qq"][:, qi]),
                        "max_competitor_m": float(pm["max_competitor"][qi]), "argmax_competitor_m": max(others, key=others.get),
                        "O_p": float(pp["O"][qi]), "O_p_ci95": pct_ci(dp_["O"][:, qi]),
                        "ceiling_share": next(c["share"] for c in ceiling_cells if c["model"] == mk and c["dataset"] == ds and c["concept"] == q)}
        ceiling_examples[f"{mk}/{ds}"] = {"status": "OK", "per_concept": cells}
    print("ceiling done", flush=True)

    # ================================================================================ task 4: floors and stability
    floors = {}
    for ds in DATASETS:
        rows = []
        for b in wm:
            if b["dataset"] != ds:
                continue
            pq_ = b["summary"]["core"]["per_question"]
            ro = data[(b["model"], ds)]["refit_O"]
            for q in CONCEPTS[ds]:
                c = pq_[q]
                owned = bool(c.get("verdict") == "fixed_family_advantage" and c.get("steering_reference"))
                floored = bool(c["W_qq"] >= FLOOR and c["O_q"] >= FLOOR)
                both = (all(ro[k][q] > 0 for k in (1, 2)) if len(ro) == 2 else None)
                rows.append({"model": b["model"], "concept": q, "owned": owned, "floored": floored, "W_qq": c["W_qq"], "O_q": c["O_q"],
                             "refit_seeds": len(ro), "O_pos_both_refits": both,
                             "refit_O": {f"seed{k}": ro[k][q] for k in ro}})
        own = [r for r in rows if r["owned"]]
        with_refit = [r for r in rows if r["refit_seeds"] == 2]
        own_refit = [r for r in own if r["refit_seeds"] == 2]
        floors[ds] = {"n_cells": len(rows), "n_blocks": len({r["model"] for r in rows}),
                      "owned": len(own), "owned_and_floored": sum(r["floored"] for r in own),
                      "owned_and_floored_cells": [f'{r["model"]}/{r["concept"]}' for r in own if r["floored"]],
                      "owned_not_floored_cells": [{"cell": f'{r["model"]}/{r["concept"]}', "W_qq": r["W_qq"], "O_q": r["O_q"]} for r in own if not r["floored"]],
                      "cells_with_refit": len(with_refit), "blocks_with_refit": len({r["model"] for r in with_refit}),
                      "owned_with_refit": len(own_refit),
                      "owned_O_pos_both_refits": sum(bool(r["O_pos_both_refits"]) for r in own_refit),
                      "owned_unstable_cells": [{"cell": f'{r["model"]}/{r["concept"]}', "O_seed0": r["O_q"], **r["refit_O"]} for r in own_refit if not r["O_pos_both_refits"]],
                      "all_cells_floored": sum(r["floored"] for r in rows),
                      "all_cells_floored_not_owned": [{"cell": f'{r["model"]}/{r["concept"]}', "W_qq": r["W_qq"], "O_q": r["O_q"]} for r in rows if r["floored"] and not r["owned"]],
                      "all_cells_O_pos_both_refits": sum(bool(r["O_pos_both_refits"]) for r in with_refit),
                      "cells": rows}
    for g, members in (("chest", CHEST), ("all", tuple(DATASETS))):
        floors[g] = {k: sum(floors[ds][k] for ds in members) for k in
                     ("n_cells", "owned", "owned_and_floored", "cells_with_refit", "owned_with_refit", "owned_O_pos_both_refits", "all_cells_floored", "all_cells_O_pos_both_refits")}

    # ================================================================================ task 5: readability ranking
    readability = {}
    for ds in CHEST:
        cs = CONCEPTS[ds]
        for scope, sel in (("probe_graded", [b for b in blocks if b["dataset"] == ds and b["calibration_present"]]),
                           ("write_matrix", [b for b in wm if b["dataset"] == ds])):
            per = {}
            for q in cs:
                S = [b["summary"]["calibration"][q]["selectivity"] for b in sel]
                R = [bool(b["summary"]["calibration"][q].get("readable")) for b in sel]
                per[q] = {"mean_S": float(np.mean(S)), "median_S": float(np.median(S)), "min_S": float(np.min(S)), "max_S": float(np.max(S)),
                          "n_blocks": len(S), "readable_blocks": int(sum(R))}
            ranked = sorted(cs, key=lambda q: -per[q]["mean_S"])
            for r, q in enumerate(ranked):
                per[q]["rank"] = r + 1
            readability[f"{ds}|{scope}"] = {"dataset": ds, "scope": scope, "blocks": [b["model"] for b in sel], "per_concept": per, "ranked": ranked}
    # grid-wide over chest (both datasets) for the concepts shared by NIH and CheXpert
    shared = [c for c in CONCEPTS["nih"] if c in CONCEPTS["chexpert"]]
    grid = {}
    for q in shared:
        S = [b["summary"]["calibration"][q]["selectivity"] for b in blocks if b["dataset"] in CHEST and b["calibration_present"]]
        grid[q] = {"mean_S": float(np.mean(S)), "n_blocks": len(S)}
    readability["chest|probe_graded|shared_concepts"] = {"per_concept": grid, "ranked": sorted(shared, key=lambda q: -grid[q]["mean_S"])}

    # ================================================================================ task 6: denominator audit
    audit = {}
    for ds in DATASETS:
        cs = CONCEPTS[ds]
        graded = [b for b in blocks if b["dataset"] == ds and b["calibration_present"]]
        wmb = [b for b in wm if b["dataset"] == ds]
        no_summary = [b for b in blocks if b["dataset"] == ds and not b["summary"]]
        excluded = [b for b in blocks if b["dataset"] == ds and b["summary"] and not b["write_matrix"]]

        def cal_flag(b, q, key):
            return bool(b["summary"]["calibration"][q].get(key, False))

        owned_cells = [(b["model"], q) for b in wmb for q in cs
                       if b["summary"]["core"]["per_question"][q].get("verdict") == "fixed_family_advantage" and b["summary"]["core"]["per_question"][q].get("steering_reference")]
        ra_cells = [(b["model"], q) for b in wmb for q in cs if cal_flag(b, q, "readable") and cal_flag(b, q, "answer_capable")]
        audit[ds] = {
            "blocks_surveyed": [f'{b["model"]}' for b in blocks if b["dataset"] == ds],
            "probe_graded": {"cells": 6 * len(graded), "blocks": [b["model"] for b in graded]},
            "write_matrix": {"cells": 6 * len(wmb), "blocks": [b["model"] for b in wmb]},
            "excluded_from_write_matrix": [{"block": b["model"], "reason": b["write_matrix_reason"],
                                            "core_W_in_summary": b["core_W_in_summary"], "primary_template": b["primary_template"]} for b in excluded + no_summary],
            "owned": {"cells": len(owned_cells), "cells_list": [f"{m}/{q}" for m, q in owned_cells], "blocks": sorted({m for m, _ in owned_cells}, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999)},
            "readable": {"cells_graded": sum(cal_flag(b, q, "readable") for b in graded for q in cs),
                         "cells_write_matrix": sum(cal_flag(b, q, "readable") for b in wmb for q in cs)},
            "answer_capable": {"cells_graded": sum(cal_flag(b, q, "answer_capable") for b in graded for q in cs),
                               "cells_write_matrix": sum(cal_flag(b, q, "answer_capable") for b in wmb for q in cs),
                               "blocks_without_answer_grade": [b["model"] for b in graded if not all("answer_capable" in b["summary"]["calibration"][q] for q in cs)]},
            "readable_and_answer_capable": {"cells_write_matrix": len(ra_cells),
                                            "owned_among_them": sum((m, q) in owned_cells for m, q in ra_cells),
                                            "owned_among_rest": sum((m, q) not in ra_cells for m, q in owned_cells)},
            "verdicts_write_matrix": {v: sum(b["summary"]["core"]["per_question"][q].get("verdict") == v for b in wmb for q in cs)
                                      for v in ("fixed_family_advantage", "stronger_competitor", "unresolved")},
            "steering_reference_write_matrix": sum(bool(b["summary"]["core"]["per_question"][q].get("steering_reference")) for b in wmb for q in cs),
        }
    chest = {}
    for key in ("probe_graded", "write_matrix", "owned"):
        chest[key] = {"cells": sum(audit[ds][key]["cells"] for ds in CHEST)}
    chest["readable_cells_graded"] = sum(audit[ds]["readable"]["cells_graded"] for ds in CHEST)
    chest["readable_cells_write_matrix"] = sum(audit[ds]["readable"]["cells_write_matrix"] for ds in CHEST)
    chest["answer_capable_cells_write_matrix"] = sum(audit[ds]["answer_capable"]["cells_write_matrix"] for ds in CHEST)
    chest["answer_capable_cells_graded"] = sum(audit[ds]["answer_capable"]["cells_graded"] for ds in CHEST)
    chest["readable_and_answer_capable_write_matrix"] = sum(audit[ds]["readable_and_answer_capable"]["cells_write_matrix"] for ds in CHEST)
    chest["owned_among_readable_and_answer_capable"] = sum(audit[ds]["readable_and_answer_capable"]["owned_among_them"] for ds in CHEST)
    chest["stronger_competitor_write_matrix"] = sum(audit[ds]["verdicts_write_matrix"]["stronger_competitor"] for ds in CHEST)
    chest["steering_reference_write_matrix"] = sum(audit[ds]["steering_reference_write_matrix"] for ds in CHEST)
    audit["chest"] = chest
    audit["table1_superscripts"] = {ds: f'{audit[ds]["owned"]["cells"]}/{audit[ds]["probe_graded"]["cells"]} (owned over probe-graded cells); '
                                        f'write-matrix denominator would be {audit[ds]["write_matrix"]["cells"]}' for ds in DATASETS}

    res = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_root": str(RUN_ROOT), "alpha": PRIMARY_ALPHA,
                    "bootstrap": boot_meta, "floor": FLOOR, "ceiling": {"hi": CEIL_HI, "lo": CEIL_LO, "share": CEIL_SHARE},
                    "tower_tolerance": {"feature_rel_tol": TOWER_REL_TOL, "feature_rel_floor": TOWER_REL_FLOOR, "write_vector_max_abs_diff": TOWER_ABS_TOL, "write_vector_min_cos": TOWER_COS_TOL},
                    "blocks_surveyed": len(blocks), "blocks_write_matrix": len(wm), "excluded_models": list(EXCLUDED_MODELS),
                    "script": str(Path(__file__).resolve())},
           "blocks": [{k: v for k, v in b.items() if k != "summary"} for b in blocks],
           "task1_paired_deltas": pairs_out,
           "task2_ceiling": {"cells": ceiling_cells, "blocks": ceiling_blocks,
                             "ceiling_cells": [c for c in ceiling_cells if c["ceiling"]],
                             "n_ceiling_cells": sum(c["ceiling"] for c in ceiling_cells),
                             "n_ceiling_cells_owned": sum(c["ceiling"] and c["owned"] for c in ceiling_cells),
                             "per_dataset": {ds: {"n_cells": sum(c["dataset"] == ds for c in ceiling_cells), "n_ceiling": sum(c["ceiling"] for c in ceiling_cells if c["dataset"] == ds),
                                                  "n_ceiling_owned": sum(c["ceiling"] and c["owned"] for c in ceiling_cells if c["dataset"] == ds)} for ds in DATASETS},
                             "examples": ceiling_examples},
           "task3_tower_sharing": tower,
           "task4_floors": floors,
           "task5_readability": readability,
           "task6_denominators": audit}
    res = jsonable(res)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_md(res, out / "pairs.md")
    print(f"wrote {out / 'pairs.json'} and pairs.md in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------- markdown
def write_md(R: dict, path: Path):
    m = R["meta"]
    L = ["# Shared-tower pairs, ceiling, tower sharing, effect floors, readability, denominators",
         f"Generated {m['generated_utc']} by `scripts/mayo/robustness_pairs.py` from `{m['run_root']}`; alpha {m['alpha']}. "
         f"Blocks surveyed {m['blocks_surveyed']}, with a write matrix {m['blocks_write_matrix']} (excluded models: {', '.join(m['excluded_models'])}). "
         f"Bootstrap: {m['bootstrap']['helper']}, seed {m['bootstrap']['seed']}, {m['bootstrap']['draws']} draws; {m['bootstrap']['unit']}; {m['bootstrap']['interval']}.", ""]
    # ---- task 1
    L += ["## 1. Shared-tower paired deltas (larger minus smaller)", ""]
    rows = []
    for p in R["task1_paired_deltas"]:
        if p["status"] != "OK":
            rows.append([p["pair"], p["dataset"], p["reason"], "", "", "", "", "", "", "", ""]); continue
        wv = p["write_vectors"]
        rows.append([p["pair"], p["dataset"], p["relationship"], f'{p["same_test_rows"]}',
                     f'{p["features"]["md5_identical"]} / {p["features"].get("differing_elements", "n/a")} / {f4(p["features"]["max_abs_diff"])} / {p["features"].get("max_rel_diff", float("nan")):.1e}',
                     f'{wv["identical"]} / {wv["max_abs_diff"]:.1e} / {f4(wv["min_cos"], 5)}',
                     f'{p["n_concepts_delta_O_ci_excludes_zero"]}/6', f'{p["n_concepts_delta_W_qq_ci_excludes_zero"]}/6', f'{p["n_concepts_delta_max_competitor_ci_excludes_zero"]}/6',
                     f4(p["median_abs_delta_O"]), f'{f4(p["refit_sd_small"])} / {f4(p["refit_sd_large"])} -> med ratio {f4(p["median_abs_delta_O_over_refit_sd"], 2)}, max {f4(p["max_abs_delta_O_over_refit_sd"], 2)}'])
    L.append(md_table(["pair", "dataset", "relationship", "same rows", "features md5 same / differing elements / max|dx| / max rel", "write vectors identical / max|diff| / min cos",
                       "dO CI excl 0", "dW_qq CI excl 0", "dMaxComp CI excl 0", "median |dO|", "refit SD small / large -> |dO|/SD"], rows))
    L.append("")
    for p in R["task1_paired_deltas"]:
        if p["status"] != "OK":
            continue
        L.append(f"### {p['pair']} on {p['dataset']}")
        rows = []
        for q, c in p["per_concept"].items():
            dO, dW, dM = c["delta_O"], c["delta_W_qq"], c["delta_max_competitor"]
            rows.append([q, f4(dO["small"]), f4(dO["large"]), f'{dO["estimate"]:+.4f}', f'[{dO["ci95"][0]:+.4f}, {dO["ci95"][1]:+.4f}]', "yes" if dO["excludes_zero"] else "no",
                         f'{dW["estimate"]:+.4f} [{dW["ci95"][0]:+.4f}, {dW["ci95"][1]:+.4f}]', "yes" if dW["excludes_zero"] else "no",
                         f'{dM["estimate"]:+.4f} [{dM["ci95"][0]:+.4f}, {dM["ci95"][1]:+.4f}]', "yes" if dM["excludes_zero"] else "no",
                         f4(c["abs_delta_O_over_refit_sd"], 2)])
        L.append(md_table(["concept", "O small", "O large", "dO", "dO CI95", "excl 0", "dW_qq [CI]", "excl 0", "dMaxComp [CI]", "excl 0", "|dO|/refit SD"], rows))
        L.append("")
    # ---- task 2
    T = R["task2_ceiling"]
    L += ["## 2. Ceiling marking (clean P(yes) > 0.99 or < 0.01)",
          f"Cells with share > {m['ceiling']['share']}: {T['n_ceiling_cells']} of {len(T['cells'])} (owned among them: {T['n_ceiling_cells_owned']}); per dataset " +
          ", ".join(f"{ds} {v['n_ceiling']}/{v['n_cells']} (owned {v['n_ceiling_owned']})" for ds, v in T["per_dataset"].items()) + ".", ""]
    L.append(md_table(["model", "dataset", "concept", "share", "high", "low", "owned", "O_q", "W_qq"],
                      [[c["model"], c["dataset"], c["concept"], f4(c["share"], 3), f4(c["share_high"], 3), f4(c["share_low"], 3), c["owned"], f4(c["O_q"]), f4(c["W_qq"])] for c in T["ceiling_cells"]]))
    L.append("")
    L.append("Per-block pooled share (six concepts x 600 rows) and number of ceiling concepts:")
    L.append(md_table(["model", "dataset", "pooled share", "ceiling concepts"], [[b["model"], b["dataset"], f4(b["share_pooled"], 3), b["n_ceiling_concepts"]] for b in T["blocks"]]))
    L.append("")
    for key, ex in T["examples"].items():
        L.append(f"Margin-scale ownership for {key} (O^m = W^m_qq - max_d W^m_qd on the semantic margin, same draws):")
        if ex["status"] != "OK":
            L.append(ex["status"]); continue
        L.append(md_table(["concept", "ceiling share", "O^m", "O^m CI95", "excl 0", "W^m_qq [CI]", "max comp^m (argmax)", "O^p", "O^p CI95"],
                          [[q, f4(c["ceiling_share"], 3), f'{c["O_m"]:+.4f}', f'[{c["O_m_ci95"][0]:+.4f}, {c["O_m_ci95"][1]:+.4f}]', "yes" if c["O_m_ci_excludes_zero"] else "no",
                            f'{c["W_m_qq"]:+.4f} [{c["W_m_qq_ci95"][0]:+.4f}, {c["W_m_qq_ci95"][1]:+.4f}]', f'{c["max_competitor_m"]:+.4f} ({c["argmax_competitor_m"]})',
                            f'{c["O_p"]:+.4f}', f'[{c["O_p_ci95"][0]:+.4f}, {c["O_p_ci95"][1]:+.4f}]'] for q, c in ex["per_concept"].items()]))
        L.append("")
    # ---- task 3
    L += ["## 3. Tower sharing (pooled vis.last test features, 600 rows, float16 as stored)",
          f"Shared to bf16 tolerance = same width, every feature element within a bf16 ulp (|dx| <= {m['tower_tolerance']['feature_rel_tol']:.2e} x magnitude, magnitude floored at {m['tower_tolerance']['feature_rel_floor']} x max|x|) "
          f"and cosine between the fitted logistic write vectors > {m['tower_tolerance']['write_vector_min_cos']} for every concept. bit-identical = equal md5 of the stored float16 test features.", ""]
    rows = []
    for fam, F in R["task3_tower_sharing"].items():
        rows.append([fam, ", ".join(F["sizes"]), " | ".join("+".join(g) for g in F["shared_groups"]) or "n/a", F["shared_groups_consistent_across_datasets"]])
    L.append(md_table(["family", "sizes", "shared-tower groups", "consistent across datasets"], rows))
    L.append("")
    rows = []
    for fam, F in R["task3_tower_sharing"].items():
        for ds, v in F["per_dataset"].items():
            if "pairs" not in v:
                continue
            for p in v["pairs"]:
                wv = p.get("write_vectors")
                rows.append([fam, ds, p["a"], p["b"], f'{v["blocks"][p["a"]]["D"]}/{v["blocks"][p["b"]]["D"]}', p["md5_identical"],
                             f'{p["differing_elements"]} ({p["differing_fraction"]:.1e})' if "differing_elements" in p else "width differs",
                             f4(p.get("max_abs_diff"), 4) if p.get("max_abs_diff") is not None else "",
                             f'{p["max_rel_diff"]:.1e}' if "max_rel_diff" in p else "",
                             f4(v["blocks"][p["a"]]["feature_abs_max"], 2),
                             f4(wv["min_cos"], 5) if wv else "n/a", " ".join(f"{c:.3f}" for c in wv["cos_per_concept"]) if wv else "", p["relationship"]])
    L.append(md_table(["family", "dataset", "a", "b", "D a/b", "md5 same", "differing elements (fraction)", "max|dx|", "max rel", "max|x| (a)", "min cos", "cos per concept", "relationship"], rows))
    L.append("")
    # ---- task 4
    F = R["task4_floors"]
    L += ["## 4. Effect floor and refit stability (p scale)",
          f"floored = W_qq >= {m['floor']} and O_q >= {m['floor']}; owned = verdict fixed_family_advantage and steering_reference; refit = O_q > 0 under both REFIT seeds (CORE baseline).", ""]
    rows = []
    for g in ("nih", "chexpert", "coco", "chest", "all"):
        v = F[g]
        rows.append([g, v["n_cells"], v["owned"], v["owned_and_floored"], f'{v["owned_O_pos_both_refits"]}/{v["owned_with_refit"]}' if v["cells_with_refit"] else "no REFIT",
                     v["all_cells_floored"], f'{v["all_cells_O_pos_both_refits"]}/{v["cells_with_refit"]}' if v["cells_with_refit"] else "no REFIT"])
    L.append(md_table(["group", "cells", "owned", "owned & floored", "owned & O>0 both refits", "all cells floored", "all cells O>0 both refits"], rows))
    for ds in DATASETS:
        v = F[ds]
        if v["owned_not_floored_cells"]:
            L += ["", f"{ds}: owned cells below the floor: " + "; ".join(f'{c["cell"]} (W_qq {c["W_qq"]:.4f}, O_q {c["O_q"]:.4f})' for c in v["owned_not_floored_cells"])]
        if v["all_cells_floored_not_owned"]:
            L += ["", f"{ds}: floored cells that are not owned: " + "; ".join(f'{c["cell"]} (W_qq {c["W_qq"]:.4f}, O_q {c["O_q"]:.4f})' for c in v["all_cells_floored_not_owned"])]
        if v["owned_unstable_cells"]:
            L += ["", f"{ds}: owned cells with O <= 0 under a refit: " + "; ".join(f'{c["cell"]} (s0 {c["O_seed0"]:.4f}, s1 {c["seed1"]:.4f}, s2 {c["seed2"]:.4f})' for c in v["owned_unstable_cells"])]
    L.append("")
    # ---- task 5
    L += ["## 5. Readability ranking (probe selectivity S = AUROC_real - mean control AUROC, calibration rows)", ""]
    for key, v in R["task5_readability"].items():
        if "dataset" not in v:
            continue
        L.append(f"{v['dataset']} ({v['scope']}, {len(v['blocks'])} blocks):")
        L.append(md_table(["rank", "concept", "mean S", "median S", "min", "max", "readable blocks"],
                          [[v["per_concept"][q]["rank"], q, f4(v["per_concept"][q]["mean_S"]), f4(v["per_concept"][q]["median_S"]), f4(v["per_concept"][q]["min_S"]),
                            f4(v["per_concept"][q]["max_S"]), f'{v["per_concept"][q]["readable_blocks"]}/{v["per_concept"][q]["n_blocks"]}'] for q in v["ranked"]]))
        L.append("")
    g = R["task5_readability"]["chest|probe_graded|shared_concepts"]
    L.append("Concepts shared by NIH and CheXpert, mean S over all chest probe-graded blocks: " + ", ".join(f'{q} {g["per_concept"][q]["mean_S"]:.4f} (n={g["per_concept"][q]["n_blocks"]})' for q in g["ranked"]))
    L.append("")
    # ---- task 6
    A = R["task6_denominators"]
    L += ["## 6. Denominator audit", ""]
    rows = []
    for ds in DATASETS:
        v = A[ds]
        rows.append([ds, len(v["blocks_surveyed"]), f'{v["probe_graded"]["cells"]} ({len(v["probe_graded"]["blocks"])} blocks)', f'{v["write_matrix"]["cells"]} ({len(v["write_matrix"]["blocks"])} blocks)',
                     v["owned"]["cells"], f'{v["readable"]["cells_graded"]} / {v["readable"]["cells_write_matrix"]}', f'{v["answer_capable"]["cells_graded"]} / {v["answer_capable"]["cells_write_matrix"]}',
                     f'{v["readable_and_answer_capable"]["cells_write_matrix"]} (owned {v["readable_and_answer_capable"]["owned_among_them"]})',
                     v["verdicts_write_matrix"]["stronger_competitor"], v["steering_reference_write_matrix"]])
    L.append(md_table(["dataset", "blocks", "probe-graded cells", "write-matrix cells", "owned", "readable (graded / write-matrix)", "answer-capable (graded / write-matrix)",
                       "readable & answerable (write-matrix)", "stronger competitor", "steering reference"], rows))
    c = A["chest"]
    L += ["", f"Chest totals: probe-graded {c['probe_graded']['cells']}, write matrix {c['write_matrix']['cells']}, owned {c['owned']['cells']}, readable {c['readable_cells_graded']} of graded "
          f"({c['readable_cells_write_matrix']} within write-matrix blocks), answer-capable {c['answer_capable_cells_write_matrix']} within write-matrix blocks ({c['answer_capable_cells_graded']} of graded), "
          f"readable & answerable {c['readable_and_answer_capable_write_matrix']} (owned among them {c['owned_among_readable_and_answer_capable']}), stronger competitor {c['stronger_competitor_write_matrix']}, "
          f"steering reference {c['steering_reference_write_matrix']}.", ""]
    for ds in DATASETS:
        v = A[ds]
        L.append(f"{ds}: probe-graded blocks: {', '.join(v['probe_graded']['blocks'])}.")
        L.append(f"{ds}: write-matrix blocks: {', '.join(v['write_matrix']['blocks'])}.")
        if v["excluded_from_write_matrix"]:
            L.append(f"{ds}: excluded from the write matrix: " + "; ".join(f'{e["block"]} ({e["reason"]}; core.W in summary: {e["core_W_in_summary"]}, template {e["primary_template"]})' for e in v["excluded_from_write_matrix"]) + ".")
        if v["answer_capable"]["blocks_without_answer_grade"]:
            L.append(f"{ds}: graded blocks without an answer grade: {', '.join(v['answer_capable']['blocks_without_answer_grade'])}.")
        L.append(f"{ds}: owned cells: {', '.join(v['owned']['cells_list']) or 'none'}.")
        L.append(f"{ds}: Table 1 superscript: {A['table1_superscripts'][ds]}.")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
