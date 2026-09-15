#!/usr/bin/env python
"""Robustness of the ownership headline to the probability scale, answer saturation, direction refits, the
connector locus and bf16 batch effects (reviewer items 1-6). Read-only over runs/<model>/<dataset>/; writes
runs/robustness/scale.json and scale.md.

Ownership on the p scale: W_qd = mean_i [p_i(q,d) - p_i(q,baseline)] at alpha +0.25, O_q = W_qq - max_{d!=q} W_qd;
a cell is "owned" when summary.core.per_question[q].verdict == fixed_family_advantage and steering_reference is
true. This script recomputes W/O from the outcome parquets (p and semantic-margin scales), checks its p-scale O_q
against summary.json, and adds: margin-scale O and random reference (item 1), stratified O by clean P(yes) and
by label plus saturation shares (item 2), refit-seed O and direction cosines (item 3), connector-locus W/O with
its own random family (item 4), preflight batch-vs-single checks (item 5) and the ownership x grade association
over chest cells (item 6).

Usage (from the repo root):
  PYTHONPATH=src python scripts/mayo/robustness_scale.py [--models q25-7,q3-8] [--datasets nih,coco]
                                                          [--draws 5000] [--jobs 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from cftransfer.analysis import _load_module, _matrix, _O_from_delta, bootstrap_indices          # noqa: E402
from cftransfer.manifest import block_included                                                   # noqa: E402
from cftransfer.protocol import (BOOT_CORE_DRAWS, BOOT_CORE_SEED, CONCEPTS, DATASETS, MODEL_ORDER, N_RANDOM,  # noqa: E402
                                 PRIMARY_ALPHA, primary_template)
from cftransfer.runpaths import RUN_ROOT, run_dir                                                # noqa: E402

OUT_DIR = RUN_ROOT / "robustness"
CHEST = ("nih", "chexpert")
MIN_STRATUM = 10            # a stratum contributes an O only with at least this many rows
SAT_HI, SAT_LO = 0.99, 0.01  # clean P(yes) beyond these is "saturated"
STRATA = ("p_lt_0.5", "p_ge_0.5", "label_present", "label_absent", "unsaturated")
REFIT_SENTENCE = ("A refit seed (1 or 2) resamples the independent training units (patients for NIH/CheXpert, images "
                  "for COCO) with replacement, keeps every training row of a sampled unit with its multiplicity, and "
                  "refits the scaler and the six concept probes on the same fixed random projection P; the control "
                  "probes, the random family and the sham stay at seed 0 (src/cftransfer/fit.py docstring; "
                  "docs/external-replication/README.md, REFIT paragraph).")


# ------------------------------------------------------------------------------------------------ helpers
def nanpct(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a[np.isfinite(a)], p)) if np.isfinite(a).any() else float("nan")


def nanmed(a):
    a = np.asarray(a, float)
    return float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")


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


def block_labels(model_key: str, dataset_id: str, concepts, order) -> dict[str, np.ndarray]:
    """y[q] over test rows: 1 present, 0 absent, -1 unknown (manifests/labels.csv)."""
    y = {q: np.full(len(order), -1, np.int8) for q in concepts}
    p = run_dir(model_key, dataset_id) / "manifests" / "labels.csv"
    for r in csv.DictReader(p.open(newline="", encoding="utf-8")):
        if r["row_id"] in order and r["concept"] in y and r["label_known"] == "true":
            y[r["concept"]][order[r["row_id"]]] = int(r["label"])
    return y


def clean_vector(base, q, order, value):
    v = np.full(len(order), np.nan)
    g = base[base.concept == q]
    keep = [i for i, r in enumerate(g.row_id.values) if r in order]
    v[[order[r] for r in g.row_id.values[keep]]] = g[value].values[keep]
    return v


def wmat(delta, mask=None) -> dict:
    """W[q][direction] = mean over rows (optionally within a boolean mask) of the per-sample delta."""
    W = {}
    for (q, d), v in delta.items():
        vv = v if mask is None else v[mask]
        W.setdefault(q, {})[d] = float(np.nanmean(vv)) if np.isfinite(vv).any() else float("nan")
    return W


def cell_stats(W, q, concepts) -> dict:
    """Own effect, strongest clinical competitor, O, random-family reference and sham for one question."""
    w = W.get(q, {})
    own = w.get(f"concept:{q}", float("nan"))
    others = [w[f"concept:{d}"] for d in concepts if d != q and f"concept:{d}" in w]
    rand = np.array([w[k] for k in w if k.startswith("random:")], float)
    sham = w.get(f"sham:{q}", float("nan"))
    full = len(others) == len(concepts) - 1
    out = {"W_qq": own, "max_competitor": max(others) if full else float("nan"),
           "argmax_competitor": max(((w[f"concept:{d}"], d) for d in concepts if d != q and f"concept:{d}" in w),
                                    default=(float("nan"), None))[1],
           "O": own - max(others) if full else float("nan"),
           "random_p95": nanpct(rand, 95), "random_max": float(np.nanmax(rand)) if len(rand) else float("nan"),
           "random_abs_median": nanmed(np.abs(rand)), "random_n": int(len(rand)), "abs_sham": abs(sham),
           "rank_in_random": int(1 + (rand >= own).sum()) if len(rand) and np.isfinite(own) else None}
    out["beats_random_p95"] = bool(np.isfinite(own) and own > 0 and len(rand) and own > out["random_p95"])
    out["steering_reference"] = bool(len(rand) == N_RANDOM and np.isfinite(own) and own > 0 and own > out["random_p95"]
                                     and own > out["abs_sham"])
    return out


def boot_O(delta, concepts, idx) -> dict:
    """Percentile 95% interval of O_q over the shared unit-bootstrap draws (means recomputed inside every draw)."""
    k = len(concepts)
    mat = np.stack([delta[(q, f"concept:{d}")] for q in concepts for d in concepts])      # (k*k, n)
    Os = np.empty((idx.shape[0], k))
    for b in range(idx.shape[0]):
        M = np.nanmean(mat[:, idx[b]], axis=1).reshape(k, k)
        diag = np.diag(M).copy()
        np.fill_diagonal(M, -np.inf)
        Os[b] = diag - M.max(axis=1)
    return {q: [float(np.percentile(Os[:, i], 2.5)), float(np.percentile(Os[:, i], 97.5))] for i, q in enumerate(concepts)}


def O_masked(delta, q, concepts, mask) -> dict:
    """O_q recomputed on the rows in `mask` (clinical directions only)."""
    n = int(mask.sum())
    if n < MIN_STRATUM:
        return {"n": n, "O": float("nan"), "W_qq": float("nan"), "max_competitor": float("nan")}
    w = {d: float(np.nanmean(delta[(q, f"concept:{d}")][mask])) for d in concepts}
    return {"n": n, "O": w[q] - max(w[d] for d in concepts if d != q), "W_qq": w[q],
            "max_competitor": max(w[d] for d in concepts if d != q)}


def complete_matrix(delta, concepts, n) -> tuple[bool, str]:
    missing = [(q, d) for q in concepts for d in concepts if (q, f"concept:{d}") not in delta]
    if missing:
        return False, f"{len(missing)} clinical cells missing"
    low = min(int(np.isfinite(delta[(q, f"concept:{d}")]).sum()) for q in concepts for d in concepts)
    if low < 0.99 * n:
        return False, f"only {low}/{n} finite rows in the thinnest clinical cell"
    return True, ""


def direction_cosines(model_key, dataset_id, concepts) -> dict:
    """cos(v_c^seed0, v_c^seedk) of the fitted clinical write vectors at the primary locus, per concept."""
    fd = run_dir(model_key, dataset_id) / "fits" / "vis.last"
    out = {}
    try:
        z0 = np.load(fd / "seed0.npz", allow_pickle=False)
        names = [str(x) for x in z0["concept_names"]]
        v0 = z0["clinical_vectors"].astype(np.float64)
        for k in (1, 2):
            zk = np.load(fd / f"seed{k}.npz", allow_pickle=False)
            vk = zk["clinical_vectors"].astype(np.float64)
            for i, c in enumerate(names):
                if c in concepts:
                    out.setdefault(c, {})[f"cos_seed{k}"] = float(v0[i] @ vk[i] / (np.linalg.norm(v0[i]) * np.linalg.norm(vk[i]) + 1e-12))
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return out


def preflight_checks(rd: Path) -> dict:
    out = {}
    for locus, fn in (("vis.last", "preflight.vis.last.json"), ("connector", "preflight.connector.json")):
        p = rd / fn
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        ch = j.get("checks", {})
        D = ch.get("D_batch_vs_single", {}) or {}
        D2 = ch.get("D2_replicated_batch_vs_single", {}) or {}
        G = ch.get("G_fp32_vs_model_logits", {}) or {}
        out[locus] = {"max_abs_candidate_logit_diff": D.get("max_abs_candidate_logit_diff"),
                      "max_abs_margin_diff": D.get("max_abs_margin_diff"),
                      "margin_sign_agreement": D.get("margin_sign_agreement"),
                      "declared_tolerance_logit": D.get("declared_tolerance_logit"), "D_pass": D.get("pass"),
                      "D2_max_abs_candidate_logit_diff": D2.get("max_abs_candidate_logit_diff"),
                      "D2_within_batch_max_spread": D2.get("within_batch_max_spread"),
                      "G_fp32_vs_model_max_abs_diff": G.get("max_abs_diff"),
                      "preflight_pass": j.get("pass"), "tolerance_deviations": j.get("tolerance_deviations"),
                      "batch_policy": j.get("batch_policy")}
    return out


# ------------------------------------------------------------------------------------------- one block
def analyse_block(model_key: str, dataset_id: str, draws: int) -> dict:
    t0 = time.time()
    rd = run_dir(model_key, dataset_id)
    concepts = CONCEPTS[dataset_id]
    rec = {"model": model_key, "dataset": dataset_id, "preflight": preflight_checks(rd)}
    if not (rd / "summary.json").exists():
        rec["status"] = "NO_SUMMARY"
        return rec
    if not ((rd / "run.json").exists() and block_included(json.loads((rd / "run.json").read_text()))):
        rec["status"] = "NOT_INCLUDED"       # the paper's one rule (cftransfer.manifest): CORE and CALIBRATION completed
        rec["reason"] = "run.json completed_modules lacks CORE or CALIBRATION"
        return rec
    sm = json.loads((rd / "summary.json").read_text())
    primary = sm.get("primary_template") or primary_template(rd)
    rec["primary_template"] = primary
    pq_ = sm.get("core", {}).get("per_question", {})
    if not all(pq_.get(q, {}).get("verdict") for q in concepts):     # packaged with CORE statistics = headline cell
        rec["status"] = "CORE_NOT_IN_SUMMARY"
        rec["reason"] = "summary.json carries no CORE verdicts (block not yet packaged with core statistics)"
        return rec
    core = _load_module(model_key, dataset_id, "CORE")
    if core is None or core.empty:
        rec["status"] = "NO_CORE"
        return rec
    core = core[(core.template_id == primary) & (core.fit_seed == 0)]
    order = test_order(model_key, dataset_id)
    n = len(order)
    base = core[core.direction_id == "baseline"]
    dp = _matrix(core, concepts, order, PRIMARY_ALPHA, template_id=primary, baseline=base)
    ok, why = complete_matrix(dp, concepts, n)
    if not ok or base.empty:
        rec["status"] = "INCOMPLETE_CORE"; rec["reason"] = why or "no baseline rows"
        return rec
    dm = _matrix(core, concepts, order, PRIMARY_ALPHA, template_id=primary, baseline=base, value="semantic_margin")
    P0 = {q: clean_vector(base, q, order, "p_present") for q in concepts}
    y = block_labels(model_key, dataset_id, concepts, order)
    Wp, Wm = wmat(dp), wmat(dm)
    idx = bootstrap_indices(dataset_id, 600, BOOT_CORE_SEED, draws)[:, :n] if n == 600 else \
        np.random.Generator(np.random.PCG64(BOOT_CORE_SEED)).integers(0, n, size=(draws, n))
    ci_p, ci_m = boot_O(dp, concepts, idx), boot_O(dm, concepts, idx)
    cal = sm.get("calibration", {})
    # ---- refit (seeds 1, 2; CORE baseline)
    refit = _load_module(model_key, dataset_id, "REFIT")
    O_ref, W_ref = {}, {}
    if refit is not None and not refit.empty:
        refit = refit[refit.template_id == primary]
        for k in (1, 2):
            dk = _matrix(refit, concepts, order, PRIMARY_ALPHA, template_id=primary, fit_seed=k, baseline=base)
            o = _O_from_delta(dk, concepts)
            if o is not None:
                O_ref[k] = o
                W_ref[k] = wmat(dk)
    cos = direction_cosines(model_key, dataset_id, concepts) if O_ref else {}
    # ---- connector locus (own baseline, own random family)
    locus = _load_module(model_key, dataset_id, "LOCUS")
    loc = None
    if locus is not None and not locus.empty:
        locus = locus[(locus.template_id == primary) & (locus.fit_seed == 0)]
        base_l = locus[locus.direction_id == "baseline"]
        dl_p = _matrix(locus, concepts, order, PRIMARY_ALPHA, template_id=primary, baseline=base_l)
        okl, whyl = complete_matrix(dl_p, concepts, n)
        if okl and not base_l.empty:
            dl_m = _matrix(locus, concepts, order, PRIMARY_ALPHA, template_id=primary, baseline=base_l, value="semantic_margin")
            P0l = {q: clean_vector(base_l, q, order, "p_present") for q in concepts}
            M0 = np.concatenate([clean_vector(base, q, order, "semantic_margin") for q in concepts])
            M0l = np.concatenate([clean_vector(base_l, q, order, "semantic_margin") for q in concepts])
            dP = np.abs(np.concatenate([P0l[q] - P0[q] for q in concepts])); dM = np.abs(M0l - M0)
            fin = np.isfinite(dP) & np.isfinite(dM)
            loc = {"Wp": wmat(dl_p), "Wm": wmat(dl_m),
                   "baseline_max_abs_p_diff_vs_core": float(np.nanmax(dP)),
                   "baseline_reproducibility": {"n_rows": int(fin.sum()), "frac_identical": float(np.mean(dM[fin] == 0)),
                                                "frac_abs_dp_gt_0.1": float(np.mean(dP[fin] > 0.1)),
                                                "frac_sign_flip": float(np.mean(np.sign(M0[fin]) != np.sign(M0l[fin]))),
                                                "median_abs_dmargin": float(np.median(dM[fin])), "p99_abs_dmargin": float(np.percentile(dM[fin], 99)),
                                                "max_abs_dmargin": float(dM[fin].max()), "max_abs_dp": float(dP[fin].max())}}
        else:
            rec["locus_status"] = "INCOMPLETE_LOCUS: " + (whyl or "no baseline rows")
    # ---- per cell
    cells = []
    sat_block = []
    for q in concepts:
        s = pq_.get(q, {})
        sp, smg = cell_stats(Wp, q, concepts), cell_stats(Wm, q, concepts)
        p0 = P0[q]
        strata_masks = {"p_lt_0.5": p0 < 0.5, "p_ge_0.5": p0 >= 0.5, "label_present": y[q] == 1, "label_absent": y[q] == 0,
                        "unsaturated": (p0 >= SAT_LO) & (p0 <= SAT_HI)}
        sat_hi, sat_lo = float(np.nanmean(p0 > SAT_HI)), float(np.nanmean(p0 < SAT_LO))
        sat_block.append(p0)
        c = {"model": model_key, "dataset": dataset_id, "concept": q, "primary_template": primary,
             "summary": {"verdict": s.get("verdict"), "steering_reference": s.get("steering_reference"),
                         "O_q": s.get("O_q"), "O_q_ci95_percentile": s.get("O_q_ci95_percentile"), "W_qq": s.get("W_qq")},
             "owned_p": bool(s.get("verdict") == "fixed_family_advantage" and s.get("steering_reference")),
             "grade": {"readable": bool(cal.get(q, {}).get("readable", False)),
                       "answer_capable": bool(cal.get(q, {}).get("answer_capable", False)),
                       "answer_auroc": cal.get(q, {}).get("answer_auroc")},
             "p_scale": {**sp, "O_ci95": ci_p[q]}, "margin_scale": {**smg, "O_ci95": ci_m[q]},
             "clean": {"mean_p": float(np.nanmean(p0)), "frac_p_ge_0.5": float(np.nanmean(p0 >= 0.5)),
                       "sat_hi": sat_hi, "sat_lo": sat_lo, "sat": sat_hi + sat_lo,
                       "n_label_present": int((y[q] == 1).sum()), "n_label_absent": int((y[q] == 0).sum())},
             "strata": {name: O_masked(dp, q, concepts, m) for name, m in strata_masks.items()}}
        c["check_O_p_minus_summary"] = (sp["O"] - s["O_q"]) if isinstance(s.get("O_q"), (int, float)) else None
        if O_ref:
            c["refit"] = {f"O_seed{k}": O_ref[k][q] for k in O_ref}
            c["refit"].update({f"W_qq_seed{k}": W_ref[k][q].get(f"concept:{q}") for k in W_ref})
            c["refit"].update(cos.get(q, {}))
            c["refit"]["O_pos_both"] = bool(all(O_ref[k][q] > 0 for k in (1, 2)) if len(O_ref) == 2 else False)
            c["refit"]["n_seeds"] = len(O_ref)
        if loc is not None:
            c["connector"] = {"p_scale": cell_stats(loc["Wp"], q, concepts), "margin_scale": cell_stats(loc["Wm"], q, concepts)}
        cells.append(c)
    allp = np.concatenate(sat_block)
    rec.update({"status": "OK", "n_rows": n, "draws": int(idx.shape[0]),
                "saturation": {"sat_hi": float(np.nanmean(allp > SAT_HI)), "sat_lo": float(np.nanmean(allp < SAT_LO)),
                               "sat": float(np.nanmean((allp > SAT_HI) | (allp < SAT_LO))),
                               "per_concept": {q: {"sat_hi": c["clean"]["sat_hi"], "sat_lo": c["clean"]["sat_lo"]} for q, c in zip(concepts, cells)}},
                "has_refit": bool(O_ref), "has_locus": loc is not None,
                "locus_baseline_max_abs_p_diff_vs_core": loc["baseline_max_abs_p_diff_vs_core"] if loc else None,
                "baseline_reproducibility": loc["baseline_reproducibility"] if loc else None,
                "W_p": Wp, "W_m": Wm, "cells": cells, "seconds": round(time.time() - t0, 1)})
    return rec


def _job(args):
    m, d, draws = args
    try:
        return analyse_block(m, d, draws)
    except Exception as e:                      # keep the campaign sweep going; report the failure per block
        return {"model": m, "dataset": d, "status": "ERROR", "reason": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------------------- aggregation
def two_by_two(cells, a_key, b_key):
    a = np.array([c[a_key] for c in cells], bool); b = np.array([c[b_key] for c in cells], bool)
    return {"a_yes_b_yes": int((a & b).sum()), "a_yes_b_no": int((a & ~b).sum()),
            "a_no_b_yes": int((~a & b).sum()), "a_no_b_no": int((~a & ~b).sum()), "n": int(len(a))}


def aggregate(blocks: list[dict]) -> dict:
    ok = [b for b in blocks if b.get("status") == "OK"]
    cells = [c for b in ok for c in b["cells"]]
    for c in cells:
        c["Om_pos"] = bool(c["margin_scale"]["O"] > 0)
        c["Om_ci_pos"] = bool(c["margin_scale"]["O_ci95"][0] > 0)
        c["Om_beats_random"] = bool(c["margin_scale"]["beats_random_p95"])
        c["Om_reference"] = bool(c["margin_scale"]["steering_reference"])
        c["Op_pos"] = bool(c["p_scale"]["O"] > 0)
        c["owned_m"] = bool(c["Om_ci_pos"] and c["Om_reference"])       # O^m percentile CI > 0, W^m_qq > random p95 and > |sham|
    res = {"blocks_total": len(blocks), "blocks_ok": len(ok),
           "blocks_excluded": [{"model": b["model"], "dataset": b["dataset"], "status": b["status"], "reason": b.get("reason")}
                               for b in blocks if b.get("status") != "OK"],
           "check_max_abs_O_p_minus_summary": float(np.nanmax([abs(c["check_O_p_minus_summary"]) for c in cells if c["check_O_p_minus_summary"] is not None])),
           "n_cells": len(cells)}
    groups = {ds: [c for c in cells if c["dataset"] == ds] for ds in DATASETS}
    groups["chest"] = [c for c in cells if c["dataset"] in CHEST]
    groups["all"] = cells
    # ---- item 1: margin scale
    it1 = {}
    for g, cs in groups.items():
        if not cs:
            continue
        it1[g] = {"n_cells": len(cs), "n_blocks": len({(c["model"], c["dataset"]) for c in cs}),
                  "owned_p": sum(c["owned_p"] for c in cs), "Op_pos": sum(c["Op_pos"] for c in cs),
                  "Om_pos": sum(c["Om_pos"] for c in cs), "Om_ci_lower_pos": sum(c["Om_ci_pos"] for c in cs),
                  "Wm_qq_beats_random_p95": sum(c["Om_beats_random"] for c in cs),
                  "Om_reference_(Wqq>p95,>|sham|,>0)": sum(c["Om_reference"] for c in cs),
                  "owned_m_(CI>0&ref)": sum(c["owned_m"] for c in cs),
                  "owned_p_and_Om_pos": sum(c["owned_p"] and c["Om_pos"] for c in cs),
                  "agreement_owned_p_vs_owned_m": two_by_two(cs, "owned_p", "owned_m"),
                  "owned_m_not_owned_p": [{"model": c["model"], "dataset": c["dataset"], "concept": c["concept"], "O_p": round(c["p_scale"]["O"], 4),
                                           "verdict_p": c["summary"]["verdict"], "O_m": round(c["margin_scale"]["O"], 4)}
                                          for c in cs if c["owned_m"] and not c["owned_p"]],
                  "owned_p_and_Om_ci_pos_and_beats_random_m": sum(c["owned_p"] and c["Om_ci_pos"] and c["Om_beats_random"] for c in cs),
                  "agreement_owned_p_vs_Om_pos": two_by_two(cs, "owned_p", "Om_pos"),
                  "agreement_owned_p_vs_Om_ci_pos": two_by_two(cs, "owned_p", "Om_ci_pos"),
                  "agreement_Op_pos_vs_Om_pos": two_by_two(cs, "Op_pos", "Om_pos"),
                  "median_Om": nanmed([c["margin_scale"]["O"] for c in cs]),
                  "median_Wm_qq": nanmed([c["margin_scale"]["W_qq"] for c in cs]),
                  "median_random_p95_m": nanmed([c["margin_scale"]["random_p95"] for c in cs]),
                  "disagreements": [{"model": c["model"], "dataset": c["dataset"], "concept": c["concept"], "owned_p": c["owned_p"],
                                     "O_p": round(c["p_scale"]["O"], 4), "O_m": round(c["margin_scale"]["O"], 4),
                                     "O_m_ci95": [round(x, 4) for x in c["margin_scale"]["O_ci95"]]}
                                    for c in cs if c["owned_p"] != c["Om_pos"]]}
    res["item1_margin_scale"] = it1
    # ---- item 2: strata and saturation
    it2 = {"min_rows_per_stratum": MIN_STRATUM, "strata": {}, "saturation": {}}
    for g, cs in groups.items():
        if not cs:
            continue
        row = {}
        for st in STRATA:
            ev = [c for c in cs if np.isfinite(c["strata"][st]["O"])]
            row[st] = {"evaluable": len(ev), "O_pos": sum(c["strata"][st]["O"] > 0 for c in ev),
                       "owned_p_evaluable": sum(c["owned_p"] for c in ev),
                       "owned_p_O_pos": sum(c["owned_p"] and c["strata"][st]["O"] > 0 for c in ev),
                       "not_owned_evaluable": sum(not c["owned_p"] for c in ev),
                       "not_owned_O_pos": sum((not c["owned_p"]) and c["strata"][st]["O"] > 0 for c in ev),
                       "median_n_rows": nanmed([c["strata"][st]["n"] for c in cs])}
        it2["strata"][g] = row
    sat_blocks = [{"model": b["model"], "dataset": b["dataset"], **{k: round(v, 4) for k, v in b["saturation"].items() if k != "per_concept"},
                   "per_concept_sat": {q: round(v["sat_hi"] + v["sat_lo"], 3) for q, v in b["saturation"]["per_concept"].items()}}
                  for b in ok]
    it2["saturation"] = {"blocks": sat_blocks,
                         "blocks_more_than_half_saturated": [f'{b["model"]}/{b["dataset"]}' for b in sat_blocks if b["sat"] > 0.5],
                         "blocks_more_than_half_saturated_high": [f'{b["model"]}/{b["dataset"]}' for b in sat_blocks if b["sat_hi"] > 0.5],
                         "blocks_more_than_half_saturated_low": [f'{b["model"]}/{b["dataset"]}' for b in sat_blocks if b["sat_lo"] > 0.5],
                         "cells_more_than_half_saturated": sum(c["clean"]["sat"] > 0.5 for c in cells),
                         "owned_cells_more_than_half_saturated": sum(c["owned_p"] and c["clean"]["sat"] > 0.5 for c in cells),
                         "per_dataset_median_block_sat": {ds: nanmed([b["sat"] for b in sat_blocks if b["dataset"] == ds]) for ds in DATASETS}}
    res["item2_ceiling"] = it2
    # ---- item 3: refit
    it3 = {"what_a_refit_seed_changes": REFIT_SENTENCE, "per_dataset": {}}
    for g, cs in groups.items():
        rc = [c for c in cs if c.get("refit", {}).get("n_seeds") == 2]
        if not rc:
            continue
        d1 = np.array([abs(c["refit"]["O_seed1"] - c["p_scale"]["O"]) for c in rc])
        d2 = np.array([abs(c["refit"]["O_seed2"] - c["p_scale"]["O"]) for c in rc])
        own = [c for c in rc if c["owned_p"]]; noto = [c for c in rc if not c["owned_p"]]
        it3["per_dataset"][g] = {
            "n_cells": len(rc), "n_blocks": len({(c["model"], c["dataset"]) for c in rc}),
            "median_abs_dO_seed1": float(np.median(d1)), "median_abs_dO_seed2": float(np.median(d2)),
            "median_abs_dO_pooled": float(np.median(np.concatenate([d1, d2]))),
            "p90_abs_dO_pooled": float(np.percentile(np.concatenate([d1, d2]), 90)),
            "owned_p": len(own), "owned_p_O_pos_both_refits": sum(c["refit"]["O_pos_both"] for c in own),
            "owned_p_fraction_stable": (sum(c["refit"]["O_pos_both"] for c in own) / len(own)) if own else None,
            "not_owned": len(noto), "not_owned_O_pos_both_refits": sum(c["refit"]["O_pos_both"] for c in noto),
            "not_owned_fraction_O_pos_both": (sum(c["refit"]["O_pos_both"] for c in noto) / len(noto)) if noto else None,
            "not_owned_O_pos_seed0": sum(c["Op_pos"] for c in noto),
            "median_cos_seed1": nanmed([c["refit"].get("cos_seed1", np.nan) for c in rc]),
            "median_cos_seed2": nanmed([c["refit"].get("cos_seed2", np.nan) for c in rc]),
            "min_cos": float(np.nanmin([c["refit"].get(f"cos_seed{k}", np.nan) for c in rc for k in (1, 2)])),
            "owned_unstable": [{"model": c["model"], "dataset": c["dataset"], "concept": c["concept"], "O_seed0": round(c["p_scale"]["O"], 4),
                                "O_seed1": round(c["refit"]["O_seed1"], 4), "O_seed2": round(c["refit"]["O_seed2"], 4)}
                               for c in own if not c["refit"]["O_pos_both"]]}
    res["item3_refit"] = it3
    # ---- item 4: connector locus
    it4 = {"per_dataset": {}}
    for g, cs in groups.items():
        lc = [c for c in cs if "connector" in c]
        if not lc:
            continue
        row = {"n_cells": len(lc), "n_blocks": len({(c["model"], c["dataset"]) for c in lc})}
        for scale in ("p_scale", "margin_scale"):
            P = [c[scale] for c in lc]; L = [c["connector"][scale] for c in lc]
            row[scale] = {
                "median_abs_W_qq_primary": nanmed([abs(x["W_qq"]) for x in P]), "median_abs_W_qq_connector": nanmed([abs(x["W_qq"]) for x in L]),
                "median_max_competitor_primary": nanmed([x["max_competitor"] for x in P]), "median_max_competitor_connector": nanmed([x["max_competitor"] for x in L]),
                "median_random_p95_primary": nanmed([x["random_p95"] for x in P]), "median_random_p95_connector": nanmed([x["random_p95"] for x in L]),
                "median_random_abs_median_primary": nanmed([x["random_abs_median"] for x in P]), "median_random_abs_median_connector": nanmed([x["random_abs_median"] for x in L]),
                "median_abs_sham_primary": nanmed([x["abs_sham"] for x in P]), "median_abs_sham_connector": nanmed([x["abs_sham"] for x in L]),
                "median_O_primary": nanmed([x["O"] for x in P]), "median_O_connector": nanmed([x["O"] for x in L]),
                "connector_W_qq_gt_random_p95": sum(x["beats_random_p95"] for x in L), "primary_W_qq_gt_random_p95": sum(x["beats_random_p95"] for x in P),
                "connector_steering_reference": sum(x["steering_reference"] for x in L), "connector_O_pos": sum(x["O"] > 0 for x in L),
                "connector_abs_W_qq_gt_primary": sum(abs(l["W_qq"]) > abs(p["W_qq"]) for p, l in zip(P, L)),
                "connector_random_n_median": nanmed([x["random_n"] for x in L])}
        row["owned_p_cells_with_connector_O_pos"] = sum(c["connector"]["p_scale"]["O"] > 0 for c in lc if c["owned_p"])
        row["owned_p_cells"] = sum(c["owned_p"] for c in lc)
        it4["per_dataset"][g] = row
    it4["locus_baseline_max_abs_p_diff_vs_core_max_over_blocks"] = float(np.nanmax([b["locus_baseline_max_abs_p_diff_vs_core"] for b in ok if b.get("locus_baseline_max_abs_p_diff_vs_core") is not None] or [np.nan]))
    res["item4_connector"] = it4
    # ---- item 5: batch effects (every block with a preflight file, CORE-complete or not)
    it5 = {"per_block": [], "distribution": {}, "sign_disagreement": [], "over_tolerance": [],
           "note": "D_batch_vs_single is a property of the clean forward pass, not of the locus; the connector preflight file "
                   "carries the same check, so one row per block (vis.last) is reported and connector files that differ are counted."}
    keys = ("max_abs_candidate_logit_diff", "max_abs_margin_diff", "margin_sign_agreement")
    it5["connector_file_differs_from_vis_last"] = [f'{b["model"]}/{b["dataset"]}' for b in blocks
                                                   if "connector" in b.get("preflight", {}) and "vis.last" in b.get("preflight", {})
                                                   and any(b["preflight"]["connector"].get(k) != b["preflight"]["vis.last"].get(k) for k in keys)]
    it5["blocks_without_connector_preflight"] = [f'{b["model"]}/{b["dataset"]}' for b in blocks
                                                 if "vis.last" in b.get("preflight", {}) and "connector" not in b.get("preflight", {})]
    for b in blocks:
        if "vis.last" in b.get("preflight", {}):
            it5["per_block"].append({"model": b["model"], "dataset": b["dataset"], "locus": "vis.last", "core_complete": b.get("status") == "OK", **b["preflight"]["vis.last"]})
    for locus in ("vis.last",):
        rows = [r for r in it5["per_block"] if r["locus"] == locus]
        dist = {}
        for key in ("max_abs_candidate_logit_diff", "max_abs_margin_diff", "D2_max_abs_candidate_logit_diff", "G_fp32_vs_model_max_abs_diff"):
            v = np.array([r[key] for r in rows if isinstance(r.get(key), (int, float))], float)
            dist[key] = {"n": int(len(v)), "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())} if len(v) else None
        dist["margin_sign_agreement_true"] = sum(r.get("margin_sign_agreement") is True for r in rows)
        dist["margin_sign_agreement_false"] = sum(r.get("margin_sign_agreement") is False for r in rows)
        dist["n_blocks"] = len(rows)
        it5["distribution"][locus] = dist
    it5["sign_disagreement"] = [f'{r["model"]}/{r["dataset"]}@{r["locus"]}' for r in it5["per_block"] if r.get("margin_sign_agreement") is False]
    it5["over_tolerance"] = [f'{r["model"]}/{r["dataset"]}@{r["locus"]}' for r in it5["per_block"] if r.get("D_pass") is False]
    flagged_sign = {f'{r["model"]}/{r["dataset"]}' for r in it5["per_block"] if r.get("margin_sign_agreement") is False}
    flagged_tol = {f'{r["model"]}/{r["dataset"]}' for r in it5["per_block"] if r.get("D_pass") is False}
    it5["owned_cells_in_sign_disagreement_blocks"] = sum(c["owned_p"] for c in cells if f'{c["model"]}/{c["dataset"]}' in flagged_sign)
    it5["owned_cells_in_D_over_tolerance_blocks"] = sum(c["owned_p"] for c in cells if f'{c["model"]}/{c["dataset"]}' in flagged_tol)
    it5["batch_policy"] = next((r["batch_policy"] for r in it5["per_block"] if r.get("batch_policy")), None)
    it5["clean_baseline_core_vs_locus"] = [{"model": b["model"], "dataset": b["dataset"], **b["baseline_reproducibility"]} for b in ok if b.get("baseline_reproducibility")]
    br = it5["clean_baseline_core_vs_locus"]
    it5["clean_baseline_distribution"] = {k: {"min": float(np.min([r[k] for r in br])), "median": float(np.median([r[k] for r in br])), "max": float(np.max([r[k] for r in br]))}
                                          for k in ("frac_identical", "frac_abs_dp_gt_0.1", "frac_sign_flip", "median_abs_dmargin", "p99_abs_dmargin", "max_abs_dmargin")} if br else {}
    it5["clean_baseline_blocks_not_identical"] = [f'{r["model"]}/{r["dataset"]}' for r in br if r["frac_identical"] < 1.0]
    it5["core_complete_blocks_with_D_over_0.25_logit"] = [f'{r["model"]}/{r["dataset"]}@{r["locus"]}' for r in it5["per_block"]
                                                          if r["core_complete"] and isinstance(r.get("max_abs_candidate_logit_diff"), (int, float)) and r["max_abs_candidate_logit_diff"] > 0.25]
    res["item5_batch"] = it5
    # ---- item 6: ownership x grade (chest cells)
    it6 = {}
    for g in ("chest", "nih", "chexpert", "coco"):
        cs = groups[g]
        if not cs:
            continue
        def tab(pred, name):
            a = [c for c in cs if pred(c)]; r = [c for c in cs if not pred(c)]
            t = [[sum(c["owned_p"] for c in a), sum(not c["owned_p"] for c in a)],
                 [sum(c["owned_p"] for c in r), sum(not c["owned_p"] for c in r)]]
            orr, p = fisher_exact(t, alternative="two-sided")
            return {"grade": name, "graded_owned": t[0][0], "graded_not_owned": t[0][1], "rest_owned": t[1][0], "rest_not_owned": t[1][1],
                    "owned_rate_graded": t[0][0] / max(1, sum(t[0])), "owned_rate_rest": t[1][0] / max(1, sum(t[1])),
                    "odds_ratio": float(orr) if np.isfinite(orr) else None, "fisher_p_two_sided": float(p)}
        it6[g] = {"readable_and_answerable": tab(lambda c: c["grade"]["readable"] and c["grade"]["answer_capable"], "readable & answer_capable"),
                  "readable": tab(lambda c: c["grade"]["readable"], "readable"),
                  "answer_capable": tab(lambda c: c["grade"]["answer_capable"], "answer_capable")}
    res["item6_grade_association"] = it6
    return res


# ------------------------------------------------------------------------------------------ markdown
def md_table(headers, rows) -> str:
    def f(x):
        if isinstance(x, float):
            if not np.isfinite(x):
                return "nan"
            return str(int(x)) if x.is_integer() and abs(x) >= 1 else (f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}")
        return str(x)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(f(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def write_md(res: dict, path: Path):
    L = ["# Ownership robustness: scale, ceiling, refits, connector, batch effects",
         f"Blocks with a complete CORE write matrix: {res['blocks_ok']} of {res['blocks_total']} surveyed; cells: {res['n_cells']}. "
         f"Excluded: {', '.join(f'{b['model']}/{b['dataset']} ({b['status']})' for b in res['blocks_excluded']) or 'none'}. "
         f"Max |O_q(recomputed, p scale) - summary O_q| = {res['check_max_abs_O_p_minus_summary']:.2e}.", ""]
    # item 1
    L += ["## 1. Margin-scale ownership (W^m at alpha +0.25, O^m = W^m_qq - max_d W^m_qd)"]
    rows = []
    for g, r in res["item1_margin_scale"].items():
        rows.append([g, r["n_cells"], r["owned_p"], r["Op_pos"], r["Om_pos"], r["Om_ci_lower_pos"], r["Wm_qq_beats_random_p95"],
                     r["Om_reference_(Wqq>p95,>|sham|,>0)"], r["owned_m_(CI>0&ref)"], r["owned_p_and_Om_pos"], r["owned_p_and_Om_ci_pos_and_beats_random_m"],
                     r["median_Om"], r["median_Wm_qq"], r["median_random_p95_m"]])
    L.append(md_table(["group", "cells", "owned (p)", "O^p>0", "O^m>0", "O^m CI>0", "W^m_qq>rand p95", "ref (m)", "owned (m) = CI>0 & ref", "owned(p)&O^m>0",
                       "owned(p)&CI>0&>p95", "med O^m", "med W^m_qq", "med rand p95 (m)"], rows))
    L += ["", "owned (p): summary verdict fixed_family_advantage (max-T simultaneous intervals) and steering_reference (W_qq > random p95 and > |sham|). "
          "owned (m): the same shape on the margin scale with a percentile interval for O^m (5000 shared unit-bootstrap draws) in place of the max-T family.", ""]
    rows = []
    for g, r in res["item1_margin_scale"].items():
        a = r["agreement_owned_p_vs_owned_m"]
        rows.append([g, a["a_yes_b_yes"], a["a_yes_b_no"], a["a_no_b_yes"], a["a_no_b_no"]])
    L += ["Agreement, owned (p) vs owned (m):", md_table(["group", "both", "p only", "m only", "neither"], rows)]
    extra = [d for g in ("nih", "chexpert", "coco") for d in res["item1_margin_scale"].get(g, {}).get("owned_m_not_owned_p", [])]
    if extra:
        L += ["", "Cells owned on the margin scale but not on the p scale:", md_table(["model", "dataset", "concept", "O^p", "verdict (p)", "O^m"],
                                                                                    [[d["model"], d["dataset"], d["concept"], d["O_p"], d["verdict_p"], d["O_m"]] for d in extra])]
    L += ["", "Agreement, p-scale ownership (verdict fixed_family_advantage & steering_reference) vs O^m > 0:"]
    rows = []
    for g, r in res["item1_margin_scale"].items():
        a = r["agreement_owned_p_vs_Om_pos"]; b = r["agreement_owned_p_vs_Om_ci_pos"]
        rows.append([g, a["a_yes_b_yes"], a["a_yes_b_no"], a["a_no_b_yes"], a["a_no_b_no"], b["a_yes_b_no"], b["a_no_b_yes"]])
    L.append(md_table(["group", "owned & O^m>0", "owned & O^m<=0", "not owned & O^m>0", "not owned & O^m<=0", "owned & CI^m<=0", "not owned & CI^m>0"], rows))
    dis = [d for g in ("nih", "chexpert", "coco") for d in res["item1_margin_scale"].get(g, {}).get("disagreements", [])]
    if dis:
        L += ["", "Disagreeing cells (owned_p != O^m>0):", md_table(["model", "dataset", "concept", "owned_p", "O^p", "O^m", "O^m CI95"],
                                                                   [[d["model"], d["dataset"], d["concept"], d["owned_p"], d["O_p"], d["O_m"], d["O_m_ci95"]] for d in dis])]
    # item 2
    L += ["", f"## 2. Ceiling stratification (p scale; a stratum counts with >= {res['item2_ceiling']['min_rows_per_stratum']} rows)"]
    rows = []
    for g, strata in res["item2_ceiling"]["strata"].items():
        for st, r in strata.items():
            rows.append([g, st, r["evaluable"], r["O_pos"], f'{r["owned_p_O_pos"]}/{r["owned_p_evaluable"]}', f'{r["not_owned_O_pos"]}/{r["not_owned_evaluable"]}', r["median_n_rows"]])
    L.append(md_table(["group", "stratum", "evaluable cells", "O>0", "owned: O>0/evaluable", "not owned: O>0/evaluable", "median rows"], rows))
    s = res["item2_ceiling"]["saturation"]
    L += ["", f"Saturation (clean P(yes) > {SAT_HI} or < {SAT_LO}); per-dataset median block share: " +
          ", ".join(f"{k} {v:.3f}" for k, v in s["per_dataset_median_block_sat"].items()) +
          f". Blocks > 50% saturated: {', '.join(s['blocks_more_than_half_saturated']) or 'none'} "
          f"(high side: {', '.join(s['blocks_more_than_half_saturated_high']) or 'none'}; low side: {', '.join(s['blocks_more_than_half_saturated_low']) or 'none'}). "
          f"Cells > 50% saturated: {s['cells_more_than_half_saturated']} (owned among them: {s['owned_cells_more_than_half_saturated']}).", ""]
    L.append(md_table(["model", "dataset", "sat", "sat high", "sat low", "per concept sat"],
                      [[b["model"], b["dataset"], b["sat"], b["sat_hi"], b["sat_lo"], " ".join(f"{k[:5]}={v:.2f}" for k, v in b["per_concept_sat"].items())] for b in s["blocks"]]))
    # item 3
    L += ["", "## 3. Refit stability (REFIT seeds 1, 2 with the CORE baseline; p scale)", res["item3_refit"]["what_a_refit_seed_changes"],
          "REFIT exists for the NIH and COCO blocks only (CheXpert blocks were packaged before REFIT was extended to that dataset), so 'chest' here is NIH. "
          "|dO| = |O_q(seed k) - O_q(seed 0)|; cos = cosine between the seed-0 and seed-k clinical write vectors at the primary locus.", ""]
    rows = []
    for g, r in res["item3_refit"]["per_dataset"].items():
        rows.append([g, r["n_cells"], r["median_abs_dO_seed1"], r["median_abs_dO_seed2"], r["median_abs_dO_pooled"], r["p90_abs_dO_pooled"],
                     f'{r["owned_p_O_pos_both_refits"]}/{r["owned_p"]}', r["owned_p_fraction_stable"] if r["owned_p_fraction_stable"] is not None else "nan",
                     f'{r["not_owned_O_pos_both_refits"]}/{r["not_owned"]}', r["not_owned_fraction_O_pos_both"] if r["not_owned_fraction_O_pos_both"] is not None else "nan",
                     r["not_owned_O_pos_seed0"], r["median_cos_seed1"], r["median_cos_seed2"], r["min_cos"]])
    L.append(md_table(["group", "cells", "med|dO| s1", "med|dO| s2", "med|dO| pooled", "p90|dO|", "owned O>0 both", "frac", "not owned O>0 both", "frac",
                       "not owned O>0 seed0", "med cos s1", "med cos s2", "min cos"], rows))
    un = [u for g in ("nih", "chexpert", "coco") for u in res["item3_refit"]["per_dataset"].get(g, {}).get("owned_unstable", [])]
    if un:
        L += ["", "Owned cells with O <= 0 under at least one refit:", md_table(["model", "dataset", "concept", "O s0", "O s1", "O s2"],
                                                                                 [[u["model"], u["dataset"], u["concept"], u["O_seed0"], u["O_seed1"], u["O_seed2"]] for u in un])]
    # item 4
    L += ["", "## 4. Connector locus (LOCUS.parquet, own baseline and own random family) vs primary locus"]
    for scale in ("p_scale", "margin_scale"):
        rows = []
        for g, r in res["item4_connector"]["per_dataset"].items():
            x = r[scale]
            rows.append([g, r["n_cells"], x["median_abs_W_qq_primary"], x["median_abs_W_qq_connector"], x["median_max_competitor_primary"], x["median_max_competitor_connector"],
                         x["median_random_p95_primary"], x["median_random_p95_connector"], x["median_abs_sham_primary"], x["median_abs_sham_connector"],
                         x["median_O_primary"], x["median_O_connector"], x["primary_W_qq_gt_random_p95"], x["connector_W_qq_gt_random_p95"],
                         x["connector_steering_reference"], x["connector_O_pos"], x["connector_abs_W_qq_gt_primary"]])
        L += ["", f"{scale}:", md_table(["group", "cells", "med|W_qq| prim", "med|W_qq| conn", "med maxcomp prim", "med maxcomp conn", "med rand p95 prim", "med rand p95 conn",
                                        "med|sham| prim", "med|sham| conn", "med O prim", "med O conn", "W_qq>p95 prim", "W_qq>p95 conn", "ref conn", "O>0 conn", "|W_qq| conn>prim"], rows)]
    L += ["", f"Owned (p-scale, primary) cells with O > 0 at the connector: " +
          ", ".join(f'{g} {r["owned_p_cells_with_connector_O_pos"]}/{r["owned_p_cells"]}' for g, r in res["item4_connector"]["per_dataset"].items()) +
          f". The LOCUS baseline is the block's own clean pass; its agreement with the CORE baseline is tabulated in item 5."]
    # item 5
    L += ["", "## 5. Batch effects (preflight D_batch_vs_single; D2 replicated batch; G fp32 vs model logits)"]
    rows = []
    for locus, d in res["item5_batch"]["distribution"].items():
        for key in ("max_abs_candidate_logit_diff", "max_abs_margin_diff", "D2_max_abs_candidate_logit_diff", "G_fp32_vs_model_max_abs_diff"):
            v = d.get(key)
            if v:
                rows.append([locus, key, v["n"], v["min"], v["median"], v["max"]])
        rows.append([locus, "margin_sign_agreement true/false", d["n_blocks"], d["margin_sign_agreement_true"], d["margin_sign_agreement_false"], ""])
    L.append(md_table(["locus", "check", "n", "min", "median", "max"], rows))
    L += ["", f"Sign disagreement: {', '.join(res['item5_batch']['sign_disagreement']) or 'none'}. D over declared tolerance: {', '.join(res['item5_batch']['over_tolerance']) or 'none'}. "
          f"CORE-complete blocks with D > 0.25 logit: {', '.join(res['item5_batch']['core_complete_blocks_with_D_over_0.25_logit']) or 'none'}. "
          f"{res['item5_batch']['note']} Connector files differing: {', '.join(res['item5_batch']['connector_file_differs_from_vis_last']) or 'none'}; "
          f"blocks without a connector preflight: {', '.join(res['item5_batch']['blocks_without_connector_preflight']) or 'none'}.", ""]
    L += [f"Owned cells inside sign-disagreement blocks: {res['item5_batch']['owned_cells_in_sign_disagreement_blocks']}; inside D-over-tolerance blocks: "
          f"{res['item5_batch']['owned_cells_in_D_over_tolerance_blocks']}. Batch policy recorded in the preflight: {res['item5_batch']['batch_policy']}", ""]
    L.append(md_table(["model", "dataset", "CORE", "logit diff", "margin diff", "sign agree", "tolerance", "D pass", "D2 logit diff", "G fp32 diff", "preflight pass"],
                      [[r["model"], r["dataset"], "y" if r["core_complete"] else "n", r.get("max_abs_candidate_logit_diff"), r.get("max_abs_margin_diff"),
                        r.get("margin_sign_agreement"), r.get("declared_tolerance_logit"), r.get("D_pass"), r.get("D2_max_abs_candidate_logit_diff"),
                        r.get("G_fp32_vs_model_max_abs_diff"), r.get("preflight_pass")]
                       for r in res["item5_batch"]["per_block"]]))
    br = res["item5_batch"].get("clean_baseline_core_vs_locus", [])
    if br:
        d = res["item5_batch"]["clean_baseline_distribution"]
        L += ["", "In-situ clean-forward reproducibility: the CORE and LOCUS modules each score the clean baseline of every test row (hook passive); "
              "per block, over the 6 x 600 rows, share of rows with identical margins, share with |dP(yes)| > 0.1, share of margin sign flips, and |d margin| quantiles.",
              f"Blocks whose baselines are not fully identical: {', '.join(res['item5_batch']['clean_baseline_blocks_not_identical']) or 'none'}. "
              + "; ".join(f"{k} min/median/max {v['min']:.4g}/{v['median']:.4g}/{v['max']:.4g}" for k, v in d.items()), "",
              md_table(["model", "dataset", "rows", "identical", "|dP|>0.1", "sign flip", "med |dm|", "p99 |dm|", "max |dm|", "max |dP|"],
                       [[r["model"], r["dataset"], r["n_rows"], r["frac_identical"], r["frac_abs_dp_gt_0.1"], r["frac_sign_flip"], r["median_abs_dmargin"],
                         r["p99_abs_dmargin"], r["max_abs_dmargin"], r["max_abs_dp"]] for r in br])]
    # item 6
    L += ["", "## 6. Ownership x grade (Fisher exact, two-sided)"]
    rows = []
    for g, d in res["item6_grade_association"].items():
        for k, t in d.items():
            rows.append([g, t["grade"], f'{t["graded_owned"]}/{t["graded_owned"] + t["graded_not_owned"]}', t["owned_rate_graded"],
                         f'{t["rest_owned"]}/{t["rest_owned"] + t["rest_not_owned"]}', t["owned_rate_rest"], t["odds_ratio"] if t["odds_ratio"] is not None else "nan", t["fisher_p_two_sided"]])
    L.append(md_table(["group", "grade", "owned/graded", "rate", "owned/rest", "rate", "odds ratio", "p"], rows))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list; default every model directory under the run root")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--draws", type=int, default=BOOT_CORE_DRAWS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    models = a.models.split(",") if a.models else sorted([p.name for p in RUN_ROOT.iterdir() if p.is_dir() and any((p / d).is_dir() for d in DATASETS)],
                                                         key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 999, m))
    jobs = [(m, d, a.draws) for m in models for d in a.datasets.split(",") if (run_dir(m, d)).is_dir()]
    t0 = time.time()
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            blocks = list(ex.map(_job, jobs))
    else:
        blocks = [_job(j) for j in jobs]
    for b in blocks:
        print(f"{b['model']:12s} {b['dataset']:9s} {b['status']:16s} {b.get('reason', '')} {b.get('seconds', '')}", flush=True)
    res = aggregate(blocks)
    res["blocks"] = [{k: v for k, v in b.items() if k not in ("W_p", "W_m", "cells")} for b in blocks]
    res["cells"] = [c for b in blocks if b.get("status") == "OK" for c in b["cells"]]
    res["W"] = {f'{b["model"]}/{b["dataset"]}': {"p": b["W_p"], "m": b["W_m"]} for b in blocks if b.get("status") == "OK"}
    res["meta"] = {"alpha": PRIMARY_ALPHA, "draws": a.draws, "bootstrap_seed": BOOT_CORE_SEED, "run_root": str(RUN_ROOT),
                   "seconds": round(time.time() - t0, 1), "script": str(Path(__file__).resolve())}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "scale.json").write_text(json.dumps(jsonable(res), indent=1), encoding="utf-8")
    write_md(res, out / "scale.md")
    print(f"wrote {out / 'scale.json'} and {out / 'scale.md'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
