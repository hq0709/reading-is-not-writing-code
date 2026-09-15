#!/usr/bin/env python
"""Answer-direction validation, column selectivity and known-label ownership over the cf-transfer campaign grid.

Read-only over <RUN_ROOT>/<model_key>/<dataset>/ (summary.json, outcomes/CORE.parquet, features/<locus>.npz,
fits/<locus>/seed<k>.npz, fits/<locus>/ansdir_seed0.npz, probe_scores.<locus>.parquet, manifests/labels.csv,
manifests/cohort.csv). Writes <RUN_ROOT>/robustness/validation.json and validation.md only.

Tasks (numbered as in the review request):
  1. answer-direction validation  every block whose summary.json carries `ansdir`. Per question q, with x the projected,
                                  train-scaled 512-d test features (seed-0 P, mu, s), a_q the ridge answer coefficients
                                  (fits/<locus>/ansdir_seed0.npz answer_coefficients) and w_q the logistic probe coefficients:
                                  (a) AUROC of a_q.x and of w_q.x against the test label of q on known-label rows, with the
                                      paired unit-bootstrap interval of the difference (shared CORE draws restricted to the
                                      known rows of each draw);
                                  (b) Pearson and Spearman correlation of the two scores over the 600 test rows;
                                  (c) raw cosine cos(a_q, w_q) (projected space and model space, as stored by ansdir.py) beside
                                      the covariance-aware cosine cos_S = a'S w / sqrt(a'S a . w'S w) with S the covariance of
                                      the training-role rows (projected space; model-space analogue with the pooled features);
                                  (d) AUROC of both scores against the model's own clean answer (CORE baseline P(yes) > 0.5),
                                      and the correlation of a_q.x with the clean margin (out-of-sample fit of the oracle).
  2. column selectivity           every block with a complete 6x6 core.W and a merged CORE.parquet. For direction d,
                                  S_d = W_dd / sum_q |W_qd| and S+_d = W_dd / sum_q max(W_qd, 0); per dataset the median,
                                  the count with S_d >= 0.5, the joint table with row ownership (verdict fixed_family_advantage
                                  with steering_reference) and the "selective but not owned" cells (S_d >= 0.5, W_dd >= 0.05,
                                  not owned).
  3. known-label ownership        the same blocks; W and O_q recomputed from CORE.parquet on the rows whose label for q is
                                  known, with a 2,000-draw percentile interval (the campaign's shared draws when every row is
                                  known, else a PCG64(BOOT_CORE_SEED) draw of the known subset's own size), the percentile
                                  verdict on all rows versus known rows, and the agreement counts per dataset. NIH and COCO have
                                  every label known, so the known-row numbers reproduce the all-row numbers exactly.
  W2 support table                per dataset and concept the known positives / negatives among the 600 test rows and among the
                                  400 calibration rows (summary calibration), and whether an answer AUROC is defined under the
                                  10/10 support rule.

Usage (from the concept-flow repo):
  PYTHONPATH=src python scripts/mayo/robustness_validation.py [--run-root ...] [--out-dir ...] [--locus vis.last]
                                                              [--fit-seed 0] [--draws 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from cftransfer.analysis import _matrix, core_bootstrap_indices          # noqa: E402
from cftransfer.manifest import block_included                            # noqa: E402
from cftransfer.protocol import BOOT_CORE_SEED, CONCEPTS, DATASETS, N_RANDOM, PRIMARY_ALPHA, primary_template  # noqa: E402
from cftransfer.runpaths import RUN_ROOT                                  # noqa: E402

SKIP_DIRS = {"robustness", "figures"}
EXCLUDED_MODELS: tuple = ()            # filled at discovery: models with no block under the inclusion rule
CHEST = ("nih", "chexpert")
SUPPORT_MIN = 10                       # the 10/10 support rule of the calibration grading
S_THRESHOLD = 0.5
W_FLOOR = 0.05
OWNED_VERDICT = "fixed_family_advantage"


# ------------------------------------------------------------------------------------------------ helpers
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def f3(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def fs(x, nd=3):
    """signed"""
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def ci(lo, hi, nd=3):
    return f"[{fs(lo, nd)}, {fs(hi, nd)}]"


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else np.nan


def auroc_fast(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties; NaN when a class is absent."""
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pct_verdict(lo: float, hi: float) -> str:
    if np.isfinite(lo) and lo > 0:
        return OWNED_VERDICT
    if np.isfinite(hi) and hi < 0:
        return "stronger_competitor"
    return "unresolved"


def boot_counts(idx: np.ndarray, n: int) -> np.ndarray:
    """(B, n) multiplicity matrix of the with-replacement draws."""
    B = idx.shape[0]
    C = np.zeros((B, n), np.float64)
    for b in range(B):
        C[b] = np.bincount(idx[b], minlength=n)
    return C


def boot_means(D: np.ndarray, C: np.ndarray) -> np.ndarray:
    """nan-aware bootstrap means: D (k, n) with NaN for unscored rows, C (B, n) counts -> (B, k)."""
    fin = np.isfinite(D)
    num = C @ np.where(fin, D, 0.0).T
    den = C @ fin.T.astype(np.float64)
    return num / np.where(den > 0, den, np.nan)


# ------------------------------------------------------------------------------------------- data loading
def discover_blocks(run_root: Path):
    """Blocks under the paper's one inclusion rule (cftransfer.manifest.block_included: CORE and CALIBRATION in
    run.json completed_modules) with a complete 6x6 core.W, a verdict per question and a merged CORE.parquet."""
    global EXCLUDED_MODELS
    used, skipped, seen, kept = [], [], set(), set()
    for sj in sorted(run_root.glob("*/*/summary.json")):
        mk, ds = sj.parts[-3], sj.parts[-2]
        if mk in SKIP_DIRS or ds not in DATASETS:
            continue
        s = json.loads(sj.read_text())
        core = s.get("core") or {}
        W = core.get("W") or {}
        pqn = core.get("per_question") or {}
        cs = CONCEPTS[ds]
        complete = all(f"concept:{d}" in W.get(q, {}) for q in cs for d in cs) and all("verdict" in pqn.get(q, {}) for q in cs)
        parquet = sj.parent / "outcomes" / "CORE.parquet"
        rj = sj.parent / "run.json"
        seen.add(mk)
        if not (rj.exists() and block_included(json.loads(rj.read_text()))):
            skipped.append({"block": f"{mk}/{ds}", "reason": f"not included (run.json completed_modules lacks CORE or CALIBRATION); core.W complete: {complete}"})
            continue
        kept.add(mk)
        if not complete:
            skipped.append({"block": f"{mk}/{ds}", "reason": "core.W incomplete or without verdicts"}); continue
        if not parquet.exists():
            skipped.append({"block": f"{mk}/{ds}", "reason": "no merged outcomes/CORE.parquet"}); continue
        used.append((mk, ds, sj.parent, s))
    EXCLUDED_MODELS = tuple(sorted(seen - kept))
    return used, skipped


def load_test_labels(block_dir: Path, ds: str):
    """(row_ids in frozen test order, Y (n, 6) float with NaN for unknown)."""
    coh = pd.read_csv(block_dir / "manifests" / "cohort.csv", dtype=str)
    lab = pd.read_csv(block_dir / "manifests" / "labels.csv", dtype=str)
    test = coh[coh.role == "test"].copy()
    test["order"] = test["order"].astype(int)
    test = test.sort_values("order")
    rows = test.row_id.tolist()
    cs = CONCEPTS[ds]
    lab = lab[lab.row_id.isin(rows) & lab.concept.isin(cs)]
    piv_l = lab.pivot(index="row_id", columns="concept", values="label").reindex(rows)[cs]
    piv_k = lab.pivot(index="row_id", columns="concept", values="label_known").reindex(rows)[cs]
    Y = np.array(piv_l.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), dtype=float, copy=True)
    Y[(piv_k.to_numpy() != "true")] = np.nan
    return rows, Y


def load_core(block_dir: Path):
    cols = ["row_id", "concept", "template_id", "fit_seed", "direction_id", "alpha", "p_present", "semantic_margin", "sample_status"]
    df = pq.read_table(block_dir / "outcomes" / "CORE.parquet", columns=cols).to_pandas()
    return df[df.sample_status == "OK"]


def block_W(summary: dict, ds: str) -> np.ndarray:
    cs = CONCEPTS[ds]
    W = summary["core"]["W"]
    return np.array([[W[q][f"concept:{d}"] for d in cs] for q in cs], float)


def owned_flags(summary: dict, ds: str):
    pqn = summary["core"]["per_question"]
    return {q: bool(pqn[q].get("verdict") == OWNED_VERDICT and pqn[q].get("steering_reference")) for q in CONCEPTS[ds]}


# ------------------------------------------------------------------------------------------ W2 support
def support_table(used, labels_by_ds: dict) -> dict:
    out = {}
    for ds in DATASETS:
        if ds not in labels_by_ds:
            continue
        cs = CONCEPTS[ds]; Y = labels_by_ds[ds]
        cal = {}
        for mk, d2, bdir, s in used:
            if d2 != ds or "calibration" not in s:
                continue
            for q in cs:
                c = s["calibration"].get(q, {})
                cal.setdefault(q, set()).add((c.get("n_pos"), c.get("n_neg"), c.get("n_unknown")))
        rows = []
        for qi, q in enumerate(cs):
            y = Y[:, qi]
            n_pos, n_neg = int(np.nansum(y == 1)), int(np.nansum(y == 0))
            calv = sorted(cal.get(q, set()))
            cal_pos, cal_neg, cal_unk = calv[0] if calv else (None, None, None)
            rows.append({"concept": q, "test_n_known": n_pos + n_neg, "test_n_pos": n_pos, "test_n_neg": n_neg,
                         "test_n_unknown": int(np.isnan(y).sum()),
                         "test_answer_auroc_defined": bool(n_pos >= SUPPORT_MIN and n_neg >= SUPPORT_MIN),
                         "calibration_n_pos": cal_pos, "calibration_n_neg": cal_neg, "calibration_n_unknown": cal_unk,
                         "calibration_answer_auroc_defined": (bool(cal_pos >= SUPPORT_MIN and cal_neg >= SUPPORT_MIN) if cal_pos is not None else None),
                         "calibration_counts_identical_across_blocks": (len(calv) == 1) if calv else None})
        out[ds] = rows
    return out


# ------------------------------------------------------------------------------------ task 1: answer directions
def answer_direction_block(mk, ds, bdir, summary, locus, fit_seed, draws, Y, rows) -> dict:
    cs = CONCEPTS[ds]; k = len(cs)
    fit = np.load(bdir / "fits" / locus / f"seed{fit_seed}.npz", allow_pickle=False)
    ans = np.load(bdir / "fits" / locus / "ansdir_seed0.npz", allow_pickle=False)
    assert [str(c) for c in fit["concept_names"]] == cs and [str(c) for c in ans["concept_names"]] == cs
    P = fit["projection"].astype(np.float32); mu = fit["scaler_mean"].astype(np.float32); s = fit["scaler_scale"].astype(np.float32)
    Wc = fit["coefficients"].astype(np.float64); bc = fit["intercepts"].astype(np.float64)
    A = ans["answer_coefficients"].astype(np.float64); bA = ans["answer_intercepts"].astype(np.float64)
    V = fit["clinical_vectors"].astype(np.float64); Va = ans["answer_vectors"].astype(np.float64)
    feat = np.load(bdir / "features" / f"{locus}.npz", allow_pickle=False)
    ids = feat["row_id"].astype(str); roles = feat["fit_role"].astype(str); X = feat["x"]
    pos = {r: i for i, r in enumerate(ids)}
    missing = [r for r in rows if r not in pos]
    if missing:
        raise RuntimeError(f"{mk}/{ds}: {len(missing)} test rows without pooled features, e.g. {missing[:3]}")
    ti = np.array([pos[r] for r in rows])
    tr = np.where(roles == "train")[0]
    Zs_test = ((X[ti].astype(np.float32) @ P - mu) / np.maximum(s, 1e-8)).astype(np.float64)
    Zs_train = ((X[tr].astype(np.float32) @ P - mu) / np.maximum(s, 1e-8)).astype(np.float64)
    Sig = np.cov(Zs_train, rowvar=False)                                   # (512, 512) projected, train-scaled space
    Xtr = X[tr].astype(np.float64)
    SigX = np.cov(Xtr, rowvar=False)                                       # (D, D) model space (pooled features)
    del Xtr
    n = len(rows)
    # clean answers of the CORE baseline (primary template, seed 0)
    primary = summary.get("primary_template") or primary_template(bdir)
    core = load_core(bdir)
    base = core[(core.direction_id == "baseline") & (core.template_id == primary) & (core.fit_seed == 0)]
    order = {r: i for i, r in enumerate(rows)}
    p_clean, m_clean = {}, {}
    for q in cs:
        g = base[base.concept == q]
        v = np.full(n, np.nan); w = np.full(n, np.nan)
        keep = [i for i, r in enumerate(g.row_id) if r in order]
        v[[order[r] for r in g.row_id.values[keep]]] = g.p_present.values[keep]
        w[[order[r] for r in g.row_id.values[keep]]] = g.semantic_margin.values[keep]
        p_clean[q], m_clean[q] = v, w
    # cross-check against the stored probe scores (fit-time logits of the same rows)
    ps_path = bdir / f"probe_scores.{locus}.parquet"
    ps = None
    if ps_path.exists():
        ps = pq.read_table(ps_path, columns=["row_id", "role", "fit_seed", "concept", "probe_kind", "logit"]).to_pandas()
        ps = ps[(ps.role == "test") & (ps.fit_seed == fit_seed) & (ps.probe_kind == "real")]
    idx = core_bootstrap_indices(ds, n, draws)
    B = idx.shape[0]
    ansum = summary["ansdir"]["per_question"]; coresum = summary["core"]["per_question"]
    per_q, cross = {}, {"max_abs_probe_logit_diff": 0.0, "max_abs_auroc_diff_vs_fit_summary": 0.0}
    fit_sum = json.loads((bdir / "fits" / locus / "summary.json").read_text()).get("seeds", {}).get(str(fit_seed), {})
    for qi, q in enumerate(cs):
        sa = Zs_test @ A[qi] + bA[qi]
        sw = Zs_test @ Wc[qi] + bc[qi]
        if ps is not None:
            g = ps[ps.concept == q].set_index("row_id").logit
            common = [r for r in rows if r in g.index]
            if common:
                d = float(np.abs(g.loc[common].values.astype(np.float64) - sw[[order[r] for r in common]]).max())
                cross["max_abs_probe_logit_diff"] = max(cross["max_abs_probe_logit_diff"], d)
        y = Y[:, qi]; known = np.isfinite(y)
        yk = y[known].astype(int)
        n_pos, n_neg = int((yk == 1).sum()), int((yk == 0).sum())
        au_a = auroc_fast(yk, sa[known]); au_w = auroc_fast(yk, sw[known])
        if fit_sum.get(q, {}).get("auroc_test") is not None and np.isfinite(au_w):
            cross["max_abs_auroc_diff_vs_fit_summary"] = max(cross["max_abs_auroc_diff_vs_fit_summary"], abs(au_w - fit_sum[q]["auroc_test"]))
        # paired bootstrap of the AUROC difference on the known rows of every shared draw
        diffs = []
        for b in range(B):
            ii = idx[b]; kk = known[ii]
            yy = y[ii][kk].astype(int)
            da, dw = auroc_fast(yy, sa[ii][kk]), auroc_fast(yy, sw[ii][kk])
            if np.isfinite(da) and np.isfinite(dw):
                diffs.append(da - dw)
        diffs = np.array(diffs)
        d_ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if len(diffs) >= 0.95 * B else [np.nan, np.nan]
        pr = pearsonr(sa, sw); spr = spearmanr(sa, sw)
        a, w = A[qi], Wc[qi]
        cos_raw_proj = float(a @ w / (np.linalg.norm(a) * np.linalg.norm(w)))
        cos_sig_proj = float(a @ Sig @ w / np.sqrt((a @ Sig @ a) * (w @ Sig @ w)))
        va, vw = Va[qi], V[qi]
        cos_raw_model = float(va @ vw / (np.linalg.norm(va) * np.linalg.norm(vw)))
        cos_sig_model = float(va @ SigX @ vw / np.sqrt((va @ SigX @ va) * (vw @ SigX @ vw)))
        # clean answer of the model on the test rows
        pc, mc = p_clean[q], m_clean[q]
        ok = np.isfinite(pc)
        yes = (pc[ok] > 0.5).astype(int)
        n_yes, n_no = int(yes.sum()), int((1 - yes).sum())
        au_a_ans = auroc_fast(yes, sa[ok]); au_w_ans = auroc_fast(yes, sw[ok])
        sp_margin = spearmanr(sa[ok], mc[ok]); pe_margin = pearsonr(sa[ok], mc[ok])
        sp_margin_w = spearmanr(sw[ok], mc[ok])
        kk = known & ok
        answer_auroc_test = auroc_fast(y[kk].astype(int), mc[kk])
        per_q[q] = {
            "n_known": int(known.sum()), "n_pos": n_pos, "n_neg": n_neg, "support_ok": bool(n_pos >= SUPPORT_MIN and n_neg >= SUPPORT_MIN),
            "auroc_answer_dir": au_a, "auroc_probe": au_w, "auroc_diff": (au_a - au_w) if np.isfinite(au_a) and np.isfinite(au_w) else np.nan,
            "auroc_diff_ci95_percentile": d_ci, "auroc_diff_valid_draws": int(len(diffs)),
            "answer_dir_reads_label_better": bool(np.isfinite(au_a) and np.isfinite(au_w) and au_a > au_w),
            "answer_dir_better_ci_excludes_0": bool(np.isfinite(d_ci[0]) and d_ci[0] > 0),
            "pearson_scores": float(pr.statistic), "spearman_scores": float(spr.statistic),
            "cos_raw_projected": cos_raw_proj, "cos_raw_projected_stored": float(ans["cos_projected"][qi, qi]),
            "cos_raw_model": cos_raw_model, "cos_raw_model_stored": float(ans["cos_to_logistic"][qi]),
            "cos_sigma_projected": cos_sig_proj, "cos_sigma_model": cos_sig_model,
            "n_clean_yes": n_yes, "n_clean_no": n_no, "auroc_answer_dir_vs_clean_answer": au_a_ans, "auroc_probe_vs_clean_answer": au_w_ans,
            "spearman_answer_dir_vs_clean_margin": float(sp_margin.statistic), "pearson_answer_dir_vs_clean_margin": float(pe_margin.statistic),
            "spearman_probe_vs_clean_margin": float(sp_margin_w.statistic),
            "clean_answer_auroc_vs_label_test": answer_auroc_test,
            "cv_r2_train": float(ans["cv_r2"][qi]), "ridge_alpha": float(ans["ridge_alpha"][qi]),
            "ansdir_W_qq": ansum[q]["W_qq"], "ansdir_O_q": ansum[q]["O_q"], "ansdir_verdict": ansum[q]["verdict"],
            "core_W_qq": coresum[q]["W_qq"], "core_O_q": coresum[q]["O_q"], "core_verdict": coresum[q]["verdict"],
            "core_owned": bool(coresum[q]["verdict"] == OWNED_VERDICT and coresum[q].get("steering_reference")),
        }
    vals = list(per_q.values())
    better = [q for q, c in per_q.items() if c["answer_dir_reads_label_better"]]
    better_sig = [q for q, c in per_q.items() if c["answer_dir_better_ci_excludes_0"]]
    return {"model_key": mk, "dataset": ds, "D": int(X.shape[1]), "n_test": n, "n_train_rows_sigma": int(len(tr)),
            "n_train_rows_ansdir": int(ans["n_train_rows"]), "primary_template": primary, "draws": int(B), "cross_checks": cross,
            "per_question": per_q,
            "summary": {"median_auroc_answer_dir": med([c["auroc_answer_dir"] for c in vals]),
                        "median_auroc_probe": med([c["auroc_probe"] for c in vals]),
                        "median_auroc_diff": med([c["auroc_diff"] for c in vals]),
                        "median_pearson_scores": med([c["pearson_scores"] for c in vals]),
                        "median_spearman_scores": med([c["spearman_scores"] for c in vals]),
                        "median_cos_raw_projected": med([c["cos_raw_projected"] for c in vals]),
                        "median_cos_raw_model": med([c["cos_raw_model"] for c in vals]),
                        "median_cos_sigma_projected": med([c["cos_sigma_projected"] for c in vals]),
                        "median_cos_sigma_model": med([c["cos_sigma_model"] for c in vals]),
                        "median_auroc_answer_dir_vs_clean_answer": med([c["auroc_answer_dir_vs_clean_answer"] for c in vals]),
                        "median_auroc_probe_vs_clean_answer": med([c["auroc_probe_vs_clean_answer"] for c in vals]),
                        "median_spearman_answer_dir_vs_clean_margin": med([c["spearman_answer_dir_vs_clean_margin"] for c in vals]),
                        "n_answer_dir_reads_label_better": len(better), "cells_answer_dir_reads_label_better": better,
                        "n_answer_dir_better_ci_excludes_0": len(better_sig), "cells_answer_dir_better_ci_excludes_0": better_sig,
                        "n_label_auroc_defined": int(sum(np.isfinite(c["auroc_answer_dir"]) for c in vals)),
                        "n_support_ok": int(sum(c["support_ok"] for c in vals))}}


# ------------------------------------------------------------------------------------ task 2: column selectivity
def column_selectivity(used) -> dict:
    cells = []
    for mk, ds, bdir, s in used:
        cs = CONCEPTS[ds]; W = block_W(s, ds); own = owned_flags(s, ds); pqn = s["core"]["per_question"]
        for di, d in enumerate(cs):
            col = W[:, di]
            den_abs = float(np.abs(col).sum()); den_pos = float(np.maximum(col, 0).sum())
            S = float(W[di, di] / den_abs) if den_abs > 0 else np.nan
            Sp = float(W[di, di] / den_pos) if den_pos > 0 else np.nan
            row = W[di]
            R = float(W[di, di] / np.abs(row).sum()) if np.abs(row).sum() > 0 else np.nan
            others = [j for j in range(len(cs)) if j != di]
            col_max_other = others[int(np.argmax(col[others]))]
            cells.append({"block": f"{mk}/{ds}", "model_key": mk, "dataset": ds, "d": d, "W_dd": float(W[di, di]),
                          "S_d": S, "S_plus_d": Sp, "R_d_row": R, "O_d": float(pqn[d]["O_q"]), "verdict": pqn[d]["verdict"],
                          "steering_reference": bool(pqn[d].get("steering_reference")), "owned": own[d],
                          "selective": bool(np.isfinite(S) and S >= S_THRESHOLD),
                          "selective_plus": bool(np.isfinite(Sp) and Sp >= S_THRESHOLD),
                          "col_abs_sum": den_abs, "col_max_other_question": cs[col_max_other],
                          "col_max_other_W": float(col[col_max_other]), "col_share_max_other": float(abs(col[col_max_other]) / den_abs) if den_abs > 0 else np.nan,
                          "random_p95": pqn[d].get("random_p95"), "abs_sham": pqn[d].get("abs_sham"), "argmax_other_row": pqn[d].get("argmax_other"),
                          "max_other_row": pqn[d].get("max_other_clinical"),
                          "selective_not_owned": bool(np.isfinite(S) and S >= S_THRESHOLD and W[di, di] >= W_FLOOR and not own[d]),
                          "owned_not_selective": bool(own[d] and not (np.isfinite(S) and S >= S_THRESHOLD))})
    df = pd.DataFrame(cells)

    def summarise(g: pd.DataFrame) -> dict:
        sel = g.selective; own = g.owned
        return {"n_blocks": int(g.block.nunique()), "n_cells": int(len(g)),
                "median_S_d": med(g.S_d.tolist()), "median_S_plus_d": med(g.S_plus_d.tolist()),
                "mean_S_d": float(np.nanmean(g.S_d)), "q25_S_d": float(np.nanpercentile(g.S_d, 25)), "q75_S_d": float(np.nanpercentile(g.S_d, 75)),
                "n_S_ge_0.5": int(sel.sum()), "n_S_plus_ge_0.5": int(g.selective_plus.sum()),
                "n_S_negative": int((g.S_d < 0).sum()), "n_owned": int(own.sum()),
                "joint": {"owned_and_selective": int((own & sel).sum()), "owned_not_selective": int((own & ~sel).sum()),
                          "not_owned_selective": int((~own & sel).sum()), "not_owned_not_selective": int((~own & ~sel).sum())},
                "joint_S_plus": {"owned_and_selective": int((own & g.selective_plus).sum()), "owned_not_selective": int((own & ~g.selective_plus).sum()),
                                 "not_owned_selective": int((~own & g.selective_plus).sum()), "not_owned_not_selective": int((~own & ~g.selective_plus).sum())},
                "median_S_d_owned": med(g.S_d[own].tolist()), "median_S_d_not_owned": med(g.S_d[~own].tolist()),
                "n_selective_not_owned_floor": int(g.selective_not_owned.sum()),
                "selective_not_owned_cells": [{k: r[k] for k in ("block", "d", "S_d", "S_plus_d", "W_dd", "O_d", "verdict", "steering_reference",
                                                                  "col_max_other_question", "col_max_other_W", "random_p95", "abs_sham",
                                                                  "argmax_other_row", "max_other_row")}
                                              for r in g[g.selective_not_owned].sort_values("S_d", ascending=False).to_dict("records")],
                "selective_not_owned_any_W_cells": [{k: r[k] for k in ("block", "d", "S_d", "W_dd", "O_d", "verdict", "steering_reference", "random_p95", "abs_sham")}
                                                    for r in g[g.selective & ~g.owned].sort_values("S_d", ascending=False).to_dict("records")],
                "owned_not_selective_cells": [{k: r[k] for k in ("block", "d", "S_d", "S_plus_d", "W_dd", "O_d", "col_max_other_question", "col_max_other_W")}
                                              for r in g[g.owned_not_selective].sort_values("S_d").to_dict("records")]}

    out = {"definitions": {"S_d": "W_dd / sum_q |W_qd|", "S_plus_d": "W_dd / sum_q max(W_qd, 0)", "R_d_row": "W_dd / sum_q |W_dq| (row analogue)",
                           "owned": f"summary verdict {OWNED_VERDICT} with steering_reference", "selective": f"S_d >= {S_THRESHOLD}",
                           "selective_not_owned": f"S_d >= {S_THRESHOLD} and W_dd >= {W_FLOOR} and not owned"},
           "per_dataset": {ds: summarise(df[df.dataset == ds]) for ds in DATASETS if (df.dataset == ds).any()},
           "chest": summarise(df[df.dataset.isin(CHEST)]), "all": summarise(df), "cells": cells}
    return out


# --------------------------------------------------------------------------------- task 3: known-label ownership
def known_label_block(mk, ds, bdir, summary, draws, Y, rows) -> dict:
    cs = CONCEPTS[ds]; k = len(cs); n = len(rows)
    primary = summary.get("primary_template") or primary_template(bdir)
    core = load_core(bdir)
    order = {r: i for i, r in enumerate(rows)}
    delta = _matrix(core, cs, order, PRIMARY_ALPHA, template_id=primary, baseline=core[core.direction_id == "baseline"])
    missing = [(q, d) for q in cs for d in cs if (q, f"concept:{d}") not in delta]
    if missing:
        raise RuntimeError(f"{mk}/{ds}: missing clinical cells {missing[:3]}")
    idx_all = core_bootstrap_indices(ds, n, draws)
    C_all = boot_counts(idx_all, n)
    pqn = summary["core"]["per_question"]
    per_q, cross = {}, {"max_abs_O_all_diff_vs_summary": 0.0, "max_abs_W_all_diff_vs_summary": 0.0}
    for qi, q in enumerate(cs):
        Dq = np.stack([delta[(q, f"concept:{d}")] for d in cs]).astype(np.float64)        # (6, n)
        rand = np.stack([delta[(q, f"random:{i:03d}")] for i in range(N_RANDOM) if (q, f"random:{i:03d}") in delta]).astype(np.float64) if any((q, f"random:{i:03d}") in delta for i in range(N_RANDOM)) else None
        sham = delta.get((q, f"sham:{q}"))
        others = [j for j in range(k) if j != qi]

        def stats(mask: np.ndarray, C: np.ndarray):
            Dm = Dq[:, mask]
            W = np.nanmean(Dm, axis=1)
            O = float(W[qi] - W[others].max())
            Wb = boot_means(Dm, C)                                                          # (B, 6)
            Ob = Wb[:, qi] - Wb[:, others].max(axis=1)
            Ob = Ob[np.isfinite(Ob)]
            lo, hi = (float(np.percentile(Ob, 2.5)), float(np.percentile(Ob, 97.5))) if len(Ob) >= 0.95 * C.shape[0] else (np.nan, np.nan)
            rp95 = float(np.percentile(np.nanmean(rand[:, mask], axis=1), 95)) if rand is not None else np.nan
            ash = abs(float(np.nanmean(sham[mask]))) if sham is not None else np.nan
            ref = bool(rand is not None and rand.shape[0] == N_RANDOM and W[qi] > 0 and W[qi] > rp95 and W[qi] > ash)
            return {"W_row": {d: float(W[j]) for j, d in enumerate(cs)}, "W_qq": float(W[qi]), "O_q": O,
                    "max_other": float(W[others].max()), "argmax_other": cs[others[int(np.argmax(W[others]))]],
                    "O_q_ci95_percentile": [lo, hi], "verdict_percentile": pct_verdict(lo, hi), "valid_draws": int(len(Ob)),
                    "random_p95": rp95, "abs_sham": ash, "steering_reference": ref,
                    "n_scored_rows_min": int(np.isfinite(Dm).sum(axis=1).min())}

        all_mask = np.ones(n, bool)
        a = stats(all_mask, C_all)
        cross["max_abs_O_all_diff_vs_summary"] = max(cross["max_abs_O_all_diff_vs_summary"], abs(a["O_q"] - pqn[q]["O_q"]))
        cross["max_abs_W_all_diff_vs_summary"] = max(cross["max_abs_W_all_diff_vs_summary"], max(abs(a["W_row"][d] - summary["core"]["W"][q][f"concept:{d}"]) for d in cs))
        known = np.isfinite(Y[:, qi]); nk = int(known.sum())
        if nk == n:
            kn = a
            same_draws = True
        else:
            idx_k = np.random.Generator(np.random.PCG64(BOOT_CORE_SEED)).integers(0, nk, size=(draws, nk))
            kn = stats(known, boot_counts(idx_k, nk))
            same_draws = False
        n_pos = int(np.nansum(Y[:, qi] == 1)); n_neg = int(np.nansum(Y[:, qi] == 0))
        per_q[q] = {"n_rows_all": n, "n_rows_known": nk, "n_pos_known": n_pos, "n_neg_known": n_neg, "known_rows_are_all_rows": same_draws,
                    "all": a, "known": kn,
                    "summary_O_q": pqn[q]["O_q"], "summary_O_q_ci95_percentile": pqn[q].get("O_q_ci95_percentile"),
                    "summary_verdict_maxT": pqn[q]["verdict"], "summary_steering_reference": bool(pqn[q].get("steering_reference")),
                    "summary_owned": bool(pqn[q]["verdict"] == OWNED_VERDICT and pqn[q].get("steering_reference")),
                    "sign_agrees": bool(np.sign(a["O_q"]) == np.sign(kn["O_q"])),
                    "percentile_verdict_agrees": a["verdict_percentile"] == kn["verdict_percentile"],
                    "known_percentile_verdict_vs_summary_maxT_agrees": kn["verdict_percentile"] == pqn[q]["verdict"],
                    "argmax_other_agrees": a["argmax_other"] == kn["argmax_other"],
                    "owned_known": bool(kn["verdict_percentile"] == OWNED_VERDICT and kn["steering_reference"]),
                    "owned_all_percentile": bool(a["verdict_percentile"] == OWNED_VERDICT and a["steering_reference"]),
                    "delta_O_known_minus_all": kn["O_q"] - a["O_q"]}
    return {"model_key": mk, "dataset": ds, "primary_template": primary, "draws": int(draws), "cross_checks": cross, "per_question": per_q}


def known_label_dataset_summary(blocks: list[dict]) -> dict:
    cells = [(b["model_key"], b["dataset"], q, c) for b in blocks for q, c in b["per_question"].items()]
    n = len(cells)
    return {"n_blocks": len(blocks), "n_cells": n,
            "median_n_rows_known": med([c["n_rows_known"] for *_, c in cells]),
            "min_n_rows_known": int(min(c["n_rows_known"] for *_, c in cells)) if n else None,
            "max_n_rows_known": int(max(c["n_rows_known"] for *_, c in cells)) if n else None,
            "n_sign_agrees": int(sum(c["sign_agrees"] for *_, c in cells)),
            "n_percentile_verdict_agrees": int(sum(c["percentile_verdict_agrees"] for *_, c in cells)),
            "n_argmax_other_agrees": int(sum(c["argmax_other_agrees"] for *_, c in cells)),
            "n_known_percentile_verdict_vs_summary_maxT_agrees": int(sum(c["known_percentile_verdict_vs_summary_maxT_agrees"] for *_, c in cells)),
            "n_all_percentile_verdict_vs_summary_maxT_agrees": int(sum(c["all"]["verdict_percentile"] == c["summary_verdict_maxT"] for *_, c in cells)),
            "n_owned_summary": int(sum(c["summary_owned"] for *_, c in cells)),
            "n_owned_all_percentile": int(sum(c["owned_all_percentile"] for *_, c in cells)),
            "n_owned_known": int(sum(c["owned_known"] for *_, c in cells)),
            "verdicts_all_percentile": dict(pd.Series([c["all"]["verdict_percentile"] for *_, c in cells]).value_counts()) if n else {},
            "verdicts_known_percentile": dict(pd.Series([c["known"]["verdict_percentile"] for *_, c in cells]).value_counts()) if n else {},
            "median_abs_delta_O": med([abs(c["delta_O_known_minus_all"]) for *_, c in cells]),
            "max_abs_delta_O": float(max(abs(c["delta_O_known_minus_all"]) for *_, c in cells)) if n else None,
            "disagreeing_cells": [{"block": f"{mk}/{ds}", "q": q, "n_known": c["n_rows_known"], "O_all": c["all"]["O_q"], "O_known": c["known"]["O_q"],
                                   "ci_all": c["all"]["O_q_ci95_percentile"], "ci_known": c["known"]["O_q_ci95_percentile"],
                                   "verdict_all": c["all"]["verdict_percentile"], "verdict_known": c["known"]["verdict_percentile"],
                                   "sign_agrees": c["sign_agrees"], "argmax_all": c["all"]["argmax_other"], "argmax_known": c["known"]["argmax_other"]}
                                  for mk, ds, q, c in cells if not (c["sign_agrees"] and c["percentile_verdict_agrees"])],
            "max_cross_check_O_diff": float(max(b["cross_checks"]["max_abs_O_all_diff_vs_summary"] for b in blocks)) if blocks else None,
            "max_cross_check_W_diff": float(max(b["cross_checks"]["max_abs_W_all_diff_vs_summary"] for b in blocks)) if blocks else None}


# ------------------------------------------------------------------------------------------- markdown
def write_markdown(R: dict, path: Path):
    L = []
    m = R["meta"]
    L.append("# Answer-direction validation, column selectivity, known-label ownership\n")
    L.append(f"Generated {m['generated_utc']} by `scripts/mayo/robustness_validation.py` from `{m['run_root']}` (locus `{m['locus']}`, fit seed "
             f"{m['fit_seed']}, alpha {PRIMARY_ALPHA}, {m['draws']} bootstrap draws, seed {BOOT_CORE_SEED}). Blocks with a complete core.W, verdicts and a "
             f"merged CORE.parquet: {m['n_blocks']} (" + ", ".join(f"{ds} {n}" for ds, n in m["n_blocks_by_dataset"].items()) + f"; {m['n_cells']} cells). "
             f"Blocks with `ansdir`: {m['n_ansdir_blocks']} (" + ", ".join(m["ansdir_blocks"]) + "). Skipped: "
             + ("; ".join(f"{s['block']} ({s['reason']})" for s in m["skipped"]) or "none") + ".\n")
    L.append("Notation. W_qd = mean over test rows of p(present | direction d, alpha) - p(present | baseline) for question q; O_q = W_qq - max_{d != q} W_qd; "
             "a cell is *owned* when the summary verdict is fixed_family_advantage (max-T over the 6x5 contrasts) and the steering reference holds. "
             "x = projected, train-scaled 512-d test features (seed-0 P, mu, s); w_q = logistic probe coefficients; a_q = ridge answer-direction coefficients "
             "(fits/vis.last/ansdir_seed0.npz). The model-space score a_q^model . x_raw equals a_q . x up to a constant, so its AUROCs are the same.\n")

    # headline
    agg = R["answer_direction"]["aggregate"]; cs2 = R["column_selectivity"]; kl = R["known_label"]["per_dataset"]
    L.append("## Summary\n")
    L.append(f"1. Answer directions ({agg['n_cells']} cells, {m['n_ansdir_blocks']} blocks). Median label-AUROC on known-label test rows: a_q {f3(agg['median_auroc_answer_dir'])} vs probe "
             f"{f3(agg['median_auroc_probe'])} (median paired difference {fs(agg['median_auroc_diff'])}); a_q reads the label better than the probe in "
             f"{agg['n_answer_dir_reads_label_better']}/{agg['n_label_auroc_defined']} cells by point estimate, in {agg['n_answer_dir_better_ci_excludes_0']} with a bootstrap interval "
             f"excluding 0, while the probe is better with an interval excluding 0 in {agg['n_probe_better_ci_excludes_0']}. Per dataset (a_q / probe): nih "
             f"{f3(agg.get('nih_median_auroc_answer_dir'))} / {f3(agg.get('nih_median_auroc_probe'))}, chexpert {f3(agg.get('chexpert_median_auroc_answer_dir'))} / "
             f"{f3(agg.get('chexpert_median_auroc_probe'))}, coco {f3(agg.get('coco_median_auroc_answer_dir'))} / {f3(agg.get('coco_median_auroc_probe'))}. "
             f"Score correlation over the 600 test rows: median Pearson {f3(agg['median_pearson_scores'])}, Spearman {f3(agg['median_spearman_scores'])}. "
             f"Raw cosine cos(a_q, w_q): median {f3(agg['median_cos_raw_projected'])} (projected) / {f3(agg['median_cos_raw_model'])} (model space); covariance-aware cosine "
             f"cos_S: median {f3(agg['median_cos_sigma_projected'])} (nih {f3(agg.get('nih_median_cos_sigma_projected'))}, chexpert {f3(agg.get('chexpert_median_cos_sigma_projected'))}, "
             f"coco {f3(agg.get('coco_median_cos_sigma_projected'))}). Against the model's own clean answer the a_q score has median AUROC "
             f"{f3(agg['median_auroc_answer_dir_vs_clean_answer'])} (probe {f3(agg['median_auroc_probe_vs_clean_answer'])}); median Spearman of a_q . x with the clean margin "
             f"{f3(agg['median_spearman_answer_dir_vs_clean_margin'])}.\n")
    ch = cs2["chest"]; co = cs2["per_dataset"].get("coco", {})
    L.append(f"2. Column selectivity ({cs2['all']['n_cells']} cells, {cs2['all']['n_blocks']} blocks). Median S_d: nih {f3(cs2['per_dataset']['nih']['median_S_d'])}, chexpert "
             f"{f3(cs2['per_dataset']['chexpert']['median_S_d'])}, coco {f3(co.get('median_S_d'))}; S_d >= 0.5 in {cs2['per_dataset']['nih']['n_S_ge_0.5']}/{cs2['per_dataset']['nih']['n_cells']} (nih), "
             f"{cs2['per_dataset']['chexpert']['n_S_ge_0.5']}/{cs2['per_dataset']['chexpert']['n_cells']} (chexpert), {co.get('n_S_ge_0.5')}/{co.get('n_cells')} (coco). "
             f"Chest joint table (owned x selective): owned & selective {ch['joint']['owned_and_selective']}, owned & not selective {ch['joint']['owned_not_selective']}, "
             f"not owned & selective {ch['joint']['not_owned_selective']}, neither {ch['joint']['not_owned_not_selective']}; COCO: {co.get('joint', {}).get('owned_and_selective')}, "
             f"{co.get('joint', {}).get('owned_not_selective')}, {co.get('joint', {}).get('not_owned_selective')}, {co.get('joint', {}).get('not_owned_not_selective')}. "
             f"Selective but not owned with W_dd >= {W_FLOOR}: nih {cs2['per_dataset']['nih']['n_selective_not_owned_floor']}, chexpert {cs2['per_dataset']['chexpert']['n_selective_not_owned_floor']}, "
             f"coco {co.get('n_selective_not_owned_floor')} (listed in section 2 with the steering-reference numbers).\n")
    cx = kl.get("chexpert", {})
    if cx:
        L.append(f"3. Known-label ownership (CheXpert, {cx['n_cells']} cells, {cx['n_blocks']} blocks; median {f3(cx['median_n_rows_known'], 0)} known rows per cell, range "
                 f"{cx['min_n_rows_known']}-{cx['max_n_rows_known']}). Sign of O_q agrees on {cx['n_sign_agrees']}/{cx['n_cells']} cells, the percentile verdict on "
                 f"{cx['n_percentile_verdict_agrees']}/{cx['n_cells']}, the strongest competitor on {cx['n_argmax_other_agrees']}/{cx['n_cells']}; owned cells {cx['n_owned_summary']} (summary) -> "
                 f"{cx['n_owned_known']} (known rows); median |dO| {f3(cx['median_abs_delta_O'], 4)}, max {f3(cx['max_abs_delta_O'], 3)}. NIH and COCO: every label known, numbers identical "
                 f"(max |dO| {f3(kl['nih']['max_abs_delta_O'], 6)} / {f3(kl['coco']['max_abs_delta_O'], 6)}).\n")

    # W2
    L.append("## W2. Label support on the test rows (600) and the calibration rows (400)\n")
    L.append("Labels are identical across blocks within a dataset. Answer AUROC defined = at least 10 known positives and 10 known negatives.\n")
    L.append("| dataset | concept | test known | test pos | test neg | test unknown | AUROC defined (test) | calibration pos | calibration neg | calibration unknown | AUROC defined (calibration) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, rows in R["support"].items():
        for r in rows:
            L.append(f"| {ds} | {r['concept']} | {r['test_n_known']} | {r['test_n_pos']} | {r['test_n_neg']} | {r['test_n_unknown']} | {r['test_answer_auroc_defined']} | "
                     f"{r['calibration_n_pos']} | {r['calibration_n_neg']} | {r['calibration_n_unknown']} | {r['calibration_answer_auroc_defined']} |")
    L.append("")

    # task 1
    L.append("## 1. Answer-direction validation (blocks with ANSDIR)\n")
    L.append("Per block: medians over the six questions. label-AUROC on known-label test rows (paired bootstrap of the difference on the known rows of each shared draw); "
             "score correlation over the 600 test rows; cos raw = cos(a_q, w_q) in the standardised projected space and in model space (as stored by ansdir.py); "
             "cos_S = a'S w / sqrt(a'S a . w'S w) with S the covariance of the training-role rows in the projected, train-scaled space. cos_S is the Pearson correlation "
             "of the two linear scores over the training rows, so the test-row Pearson column is its out-of-sample counterpart. The model-space version with the pooled-feature "
             "covariance is identical by construction (v = normalize(P (u / s)) for both families, so v_a' S_X v_w is proportional to a' S w); it is kept in validation.json "
             "(cos_sigma_model) and not repeated here. Clean-answer AUROC = the score against the model's own baseline P(yes) > 0.5 on the test rows.\n")
    L.append("| block | D | median AUROC a_q | median AUROC probe | median diff | a_q better (n/defined) | CI excl. 0 | median Pearson | median Spearman | median cos raw proj | median cos raw model | median cos_S | median AUROC a_q vs clean answer | median AUROC probe vs clean answer | median Spearman(a_q, clean margin) | probe-logit cross-check max abs diff |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, b in R["answer_direction"]["blocks"].items():
        s = b["summary"]
        L.append(f"| {key} | {b['D']} | {f3(s['median_auroc_answer_dir'])} | {f3(s['median_auroc_probe'])} | {fs(s['median_auroc_diff'])} | "
                 f"{s['n_answer_dir_reads_label_better']}/{s['n_label_auroc_defined']} ({', '.join(s['cells_answer_dir_reads_label_better']) or '-'}) | "
                 f"{s['n_answer_dir_better_ci_excludes_0']} ({', '.join(s['cells_answer_dir_better_ci_excludes_0']) or '-'}) | "
                 f"{f3(s['median_pearson_scores'])} | {f3(s['median_spearman_scores'])} | {f3(s['median_cos_raw_projected'])} | {f3(s['median_cos_raw_model'])} | "
                 f"{f3(s['median_cos_sigma_projected'])} | {f3(s['median_auroc_answer_dir_vs_clean_answer'])} | "
                 f"{f3(s['median_auroc_probe_vs_clean_answer'])} | {f3(s['median_spearman_answer_dir_vs_clean_margin'])} | {b['cross_checks']['max_abs_probe_logit_diff']:.1e} |")
    L.append("")
    L.append("Pooled over all ANSDIR cells: " + "; ".join(f"{k} {f3(v) if isinstance(v, float) else v}" for k, v in agg.items() if not isinstance(v, (list, dict))) + ".\n")
    L.append("Cells where a_q reads the label better than the probe (point estimate):\n")
    L.append("| block | q | known (pos/neg) | AUROC a_q | AUROC probe | diff [CI95] | support >= 10/10 |")
    L.append("|---|---|---|---|---|---|---|")
    for key, b in R["answer_direction"]["blocks"].items():
        for q, c in b["per_question"].items():
            if c["answer_dir_reads_label_better"]:
                lo, hi = c["auroc_diff_ci95_percentile"]
                L.append(f"| {key} | {q} | {c['n_known']} ({c['n_pos']}/{c['n_neg']}) | {f3(c['auroc_answer_dir'])} | {f3(c['auroc_probe'])} | {fs(c['auroc_diff'])} {ci(lo, hi)} | {c['support_ok']} |")
    L.append("")
    L.append("Per cell:\n")
    L.append("| block | q | known (pos/neg) | AUROC a_q | AUROC probe | diff [CI95] | Pearson | Spearman | cos raw proj | cos raw model | cos_S | clean yes/no | AUROC a_q vs clean | AUROC probe vs clean | Spearman(a_q, margin) | cv R2 (train) | O^a_q (ansdir) | O_q (core) | core verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, b in R["answer_direction"]["blocks"].items():
        for q, c in b["per_question"].items():
            lo, hi = c["auroc_diff_ci95_percentile"]
            flag = " **" if c["answer_dir_reads_label_better"] else ""
            L.append(f"| {key} | {q}{flag} | {c['n_known']} ({c['n_pos']}/{c['n_neg']}) | {f3(c['auroc_answer_dir'])} | {f3(c['auroc_probe'])} | "
                     f"{fs(c['auroc_diff'])} {ci(lo, hi)} | {f3(c['pearson_scores'])} | {f3(c['spearman_scores'])} | {f3(c['cos_raw_projected'])} | "
                     f"{f3(c['cos_raw_model'])} | {f3(c['cos_sigma_projected'])} | {c['n_clean_yes']}/{c['n_clean_no']} | "
                     f"{f3(c['auroc_answer_dir_vs_clean_answer'])} | {f3(c['auroc_probe_vs_clean_answer'])} | {f3(c['spearman_answer_dir_vs_clean_margin'])} | "
                     f"{f3(c['cv_r2_train'], 2)} | {fs(c['ansdir_O_q'])} | {fs(c['core_O_q'])} | {c['core_verdict']} |")
    L.append("")
    L.append("** = a_q reads the label better than the probe (point estimate). A diff interval of n/a means fewer than 95% of the draws had both classes among the known rows "
             "(CheXpert Atelectasis: 84 known positives, 2 known negatives; the point AUROC is computed but the cell is below the 10/10 support rule). "
             "A clean-answer AUROC of n/a means the model gave the same clean answer on every test row (clean yes/no column).\n")

    # task 2
    L.append("## 2. Column selectivity of the write matrix\n")
    L.append(f"S_d = W_dd / sum_q |W_qd| (share of direction d's absolute effect landing on its own question); S+_d = W_dd / sum_q max(W_qd, 0). "
             f"Selective = S_d >= {S_THRESHOLD}. Owned = summary verdict fixed_family_advantage with steering reference (row-based). "
             f"'Selective but not owned' additionally requires W_dd >= {W_FLOOR}.\n")
    L.append("| dataset | blocks | cells | median S_d | IQR | median S+_d | S_d >= 0.5 | S+_d >= 0.5 | S_d < 0 | owned | owned & selective | owned & not selective | not owned & selective | not owned & not selective | selective, W_dd >= 0.05, not owned | median S_d owned / not owned |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, g in list(R["column_selectivity"]["per_dataset"].items()) + [("chest", R["column_selectivity"]["chest"]), ("all", R["column_selectivity"]["all"])]:
        j = g["joint"]
        L.append(f"| {name} | {g['n_blocks']} | {g['n_cells']} | {f3(g['median_S_d'])} | [{f3(g['q25_S_d'])}, {f3(g['q75_S_d'])}] | {f3(g['median_S_plus_d'])} | "
                 f"{g['n_S_ge_0.5']} | {g['n_S_plus_ge_0.5']} | {g['n_S_negative']} | {g['n_owned']} | {j['owned_and_selective']} | {j['owned_not_selective']} | "
                 f"{j['not_owned_selective']} | {j['not_owned_not_selective']} | {g['n_selective_not_owned_floor']} | {f3(g['median_S_d_owned'])} / {f3(g['median_S_d_not_owned'])} |")
    L.append("")
    L.append("Selective but not owned cells (S_d >= 0.5, W_dd >= 0.05, not owned), by dataset, with the row-side numbers that decide ownership "
             "(strongest competitor on the question, CORE random p95 and |sham| of the steering reference):\n")
    L.append("| dataset | block | d | S_d | S+_d | W_dd | O_d (row) | strongest competitor on question d (W) | random p95 | abs sham | verdict | steering ref | largest other question in column (W) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, g in R["column_selectivity"]["per_dataset"].items():
        for c in g["selective_not_owned_cells"]:
            L.append(f"| {ds} | {c['block']} | {c['d']} | {f3(c['S_d'])} | {f3(c['S_plus_d'])} | {fs(c['W_dd'])} | {fs(c['O_d'])} | {c['argmax_other_row']} ({fs(c['max_other_row'])}) | "
                     f"{f3(c['random_p95'])} | {f3(c['abs_sham'])} | {c['verdict']} | {c['steering_reference']} | {c['col_max_other_question']} ({fs(c['col_max_other_W'])}) |")
        if not g["selective_not_owned_cells"]:
            L.append(f"| {ds} | - | - | - | - | - | - | - | - | - | - | - | - |")
    L.append("")
    L.append("Selective but not owned without the W_dd floor (S_d >= 0.5, not owned):\n")
    L.append("| dataset | block | d | S_d | W_dd | O_d (row) | random p95 | abs sham | verdict | steering ref |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds, g in R["column_selectivity"]["per_dataset"].items():
        for c in g["selective_not_owned_any_W_cells"]:
            L.append(f"| {ds} | {c['block']} | {c['d']} | {f3(c['S_d'])} | {fs(c['W_dd'])} | {fs(c['O_d'])} | {f3(c['random_p95'])} | {f3(c['abs_sham'])} | {c['verdict']} | {c['steering_reference']} |")
        if not g["selective_not_owned_any_W_cells"]:
            L.append(f"| {ds} | - | - | - | - | - | - | - | - | - |")
    L.append("")
    L.append("Owned but not column-selective cells (S_d < 0.5):\n")
    L.append("| dataset | block | d | S_d | S+_d | W_dd | O_d (row) | largest other question in column (W) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, g in R["column_selectivity"]["per_dataset"].items():
        for c in g["owned_not_selective_cells"]:
            L.append(f"| {ds} | {c['block']} | {c['d']} | {f3(c['S_d'])} | {f3(c['S_plus_d'])} | {fs(c['W_dd'])} | {fs(c['O_d'])} | {c['col_max_other_question']} ({fs(c['col_max_other_W'])}) |")
        if not g["owned_not_selective_cells"]:
            L.append(f"| {ds} | - | - | - | - | - | - | - |")
    L.append("")

    # task 3
    L.append("## 3. Ownership on known-label rows\n")
    L.append("Per cell W and O_q recomputed from CORE.parquet (primary template, seed 0, alpha 0.25) on all 600 rows and on the rows whose label for q is known. "
             f"Percentile interval: {m['draws']} draws; the campaign's shared unit-bootstrap draws (core_bootstrap_indices, seed {BOOT_CORE_SEED}) when every row is known, "
             f"else a PCG64({BOOT_CORE_SEED}) draw of the known subset's own size, max competitor recomputed inside every draw. Percentile verdict: interval > 0 -> fixed_family_advantage, "
             "< 0 -> stronger_competitor, else unresolved. The summary verdict is the max-T one (5,000 draws), listed for reference.\n")
    L.append("| dataset | blocks | cells | median n known | min-max n known | sign agrees | percentile verdict agrees | argmax competitor agrees | known percentile verdict = summary max-T | all-rows percentile verdict = summary max-T | owned (summary) | owned (all rows, percentile) | owned (known rows) | median abs dO | max abs dO | verdicts all (pct) | verdicts known (pct) | recompute check max abs dO vs summary |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, g in R["known_label"]["per_dataset"].items():
        L.append(f"| {ds} | {g['n_blocks']} | {g['n_cells']} | {f3(g['median_n_rows_known'], 0)} | {g['min_n_rows_known']}-{g['max_n_rows_known']} | {g['n_sign_agrees']}/{g['n_cells']} | "
                 f"{g['n_percentile_verdict_agrees']}/{g['n_cells']} | {g['n_argmax_other_agrees']}/{g['n_cells']} | {g['n_known_percentile_verdict_vs_summary_maxT_agrees']}/{g['n_cells']} | "
                 f"{g['n_all_percentile_verdict_vs_summary_maxT_agrees']}/{g['n_cells']} | {g['n_owned_summary']} | {g['n_owned_all_percentile']} | {g['n_owned_known']} | "
                 f"{f3(g['median_abs_delta_O'], 4)} | {f3(g['max_abs_delta_O'], 4)} | {g['verdicts_all_percentile']} | {g['verdicts_known_percentile']} | {g['max_cross_check_O_diff']:.1e} |")
    L.append("")
    L.append("NIH and COCO: every test label is known, so the known-row subset is the full row set and the same draws apply; the numbers are identical by construction "
             "(max |dO| = 0 above).\n")
    for ds in ("chexpert",):
        g = R["known_label"]["per_dataset"].get(ds)
        if not g:
            continue
        L.append(f"Disagreeing cells on {ds} (sign or percentile verdict):\n")
        L.append("| block | q | n known | O_q all [CI] | verdict all | O_q known [CI] | verdict known | sign agrees | competitor all -> known |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c in g["disagreeing_cells"]:
            L.append(f"| {c['block']} | {c['q']} | {c['n_known']} | {fs(c['O_all'])} {ci(*c['ci_all'])} | {c['verdict_all']} | {fs(c['O_known'])} {ci(*c['ci_known'])} | "
                     f"{c['verdict_known']} | {c['sign_agrees']} | {c['argmax_all']} -> {c['argmax_known']} |")
        if not g["disagreeing_cells"]:
            L.append("| - | - | - | - | - | - | - | - | - |")
        L.append("")
        L.append(f"Ownership changes on {ds} (summary owned = max-T verdict with steering reference on all rows; known = percentile verdict with the steering reference recomputed on the known rows):\n")
        L.append("| block | q | n known | owned summary | owned all rows (pct) | owned known rows | O_q all [CI] | O_q known [CI] | steering ref all / known |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        n_flip = 0
        for b in R["known_label"]["blocks"]:
            if b["dataset"] != ds:
                continue
            for q, c in b["per_question"].items():
                if c["summary_owned"] != c["owned_known"] or c["owned_all_percentile"] != c["owned_known"]:
                    n_flip += 1
                    a2, k2 = c["all"], c["known"]
                    L.append(f"| {b['model_key']}/{ds} | {q} | {c['n_rows_known']} | {c['summary_owned']} | {c['owned_all_percentile']} | {c['owned_known']} | "
                             f"{fs(a2['O_q'], 4)} {ci(*a2['O_q_ci95_percentile'], 4)} | {fs(k2['O_q'], 4)} {ci(*k2['O_q_ci95_percentile'], 4)} | {a2['steering_reference']} / {k2['steering_reference']} |")
        if n_flip == 0:
            L.append("| - | - | - | - | - | - | - | - | - |")
        L.append("")
        L.append(f"All {ds} cells:\n")
        L.append("| block | q | n known (pos/neg) | O_q all [CI] | pct verdict all | O_q known [CI] | pct verdict known | summary max-T verdict | owned summary / known | agree sign / verdict |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for b in R["known_label"]["blocks"]:
            if b["dataset"] != ds:
                continue
            for q, c in b["per_question"].items():
                a, kn = c["all"], c["known"]
                L.append(f"| {b['model_key']}/{ds} | {q} | {c['n_rows_known']} ({c['n_pos_known']}/{c['n_neg_known']}) | {fs(a['O_q'])} {ci(*a['O_q_ci95_percentile'])} | {a['verdict_percentile']} | "
                         f"{fs(kn['O_q'])} {ci(*kn['O_q_ci95_percentile'])} | {kn['verdict_percentile']} | {c['summary_verdict_maxT']} | {c['summary_owned']} / {c['owned_known']} | "
                         f"{c['sign_agrees']} / {c['percentile_verdict_agrees']} |")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(RUN_ROOT))
    ap.add_argument("--out-dir", default=None, help="default <run-root>/robustness")
    ap.add_argument("--locus", default="vis.last")
    ap.add_argument("--fit-seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    run_root = Path(a.run_root)
    out_dir = Path(a.out_dir) if a.out_dir else run_root / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    # sanity: the fast AUROC agrees with sklearn
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 300); ss = rng.standard_normal(300) + yy; ss[:40] = np.round(ss[:40])
    assert abs(auroc_fast(yy, ss) - roc_auc_score(yy, ss)) < 1e-12

    used, skipped = discover_blocks(run_root)
    if not used:
        raise SystemExit("no complete blocks found")
    labels, rows_by_ds = {}, {}
    for mk, ds, bdir, s in used:
        rows, Y = load_test_labels(bdir, ds)
        if ds in labels:
            assert rows == rows_by_ds[ds] and np.array_equal(np.nan_to_num(Y, nan=-9), np.nan_to_num(labels[ds], nan=-9)), f"{mk}/{ds}: test labels differ"
        else:
            labels[ds], rows_by_ds[ds] = Y, rows

    support = support_table(used, labels)
    print(f"[validation] {len(used)} blocks, support table done, {time.time() - t0:.0f}s", flush=True)

    # task 1
    ans_blocks = {}
    for mk, ds, bdir, s in used:
        if "ansdir" not in s or not (bdir / "fits" / a.locus / "ansdir_seed0.npz").exists():
            continue
        ans_blocks[f"{mk}/{ds}"] = answer_direction_block(mk, ds, bdir, s, a.locus, a.fit_seed, a.draws, labels[ds], rows_by_ds[ds])
        print(f"[validation] ansdir {mk}/{ds} done, {time.time() - t0:.0f}s", flush=True)
    cells = [c for b in ans_blocks.values() for c in b["per_question"].values()]
    aggregate = {"n_cells": len(cells),
                 "n_label_auroc_defined": int(sum(np.isfinite(c["auroc_answer_dir"]) for c in cells)),
                 "n_answer_dir_reads_label_better": int(sum(c["answer_dir_reads_label_better"] for c in cells)),
                 "n_answer_dir_better_ci_excludes_0": int(sum(c["answer_dir_better_ci_excludes_0"] for c in cells)),
                 "n_probe_better_ci_excludes_0": int(sum(np.isfinite(c["auroc_diff_ci95_percentile"][1]) and c["auroc_diff_ci95_percentile"][1] < 0 for c in cells)),
                 "median_auroc_answer_dir": med([c["auroc_answer_dir"] for c in cells]), "median_auroc_probe": med([c["auroc_probe"] for c in cells]),
                 "median_auroc_diff": med([c["auroc_diff"] for c in cells]),
                 "median_pearson_scores": med([c["pearson_scores"] for c in cells]), "median_spearman_scores": med([c["spearman_scores"] for c in cells]),
                 "median_cos_raw_projected": med([c["cos_raw_projected"] for c in cells]), "median_cos_raw_model": med([c["cos_raw_model"] for c in cells]),
                 "median_cos_sigma_projected": med([c["cos_sigma_projected"] for c in cells]), "median_cos_sigma_model": med([c["cos_sigma_model"] for c in cells]),
                 "median_auroc_answer_dir_vs_clean_answer": med([c["auroc_answer_dir_vs_clean_answer"] for c in cells]),
                 "median_auroc_probe_vs_clean_answer": med([c["auroc_probe_vs_clean_answer"] for c in cells]),
                 "median_spearman_answer_dir_vs_clean_margin": med([c["spearman_answer_dir_vs_clean_margin"] for c in cells]),
                 "n_ansdir_owned_by_verdict": int(sum(c["ansdir_verdict"] == OWNED_VERDICT for c in cells)),
                 "n_core_owned": int(sum(c["core_owned"] for c in cells))}
    for ds in DATASETS:
        cd = [c for k, b in ans_blocks.items() for c in b["per_question"].values() if b["dataset"] == ds]
        if cd:
            aggregate[f"{ds}_median_auroc_answer_dir"] = med([c["auroc_answer_dir"] for c in cd])
            aggregate[f"{ds}_median_auroc_probe"] = med([c["auroc_probe"] for c in cd])
            aggregate[f"{ds}_median_cos_sigma_projected"] = med([c["cos_sigma_projected"] for c in cd])
            aggregate[f"{ds}_median_pearson_scores"] = med([c["pearson_scores"] for c in cd])
            aggregate[f"{ds}_n_answer_dir_reads_label_better"] = int(sum(c["answer_dir_reads_label_better"] for c in cd))
            aggregate[f"{ds}_n_label_auroc_defined"] = int(sum(np.isfinite(c["auroc_answer_dir"]) for c in cd))

    # task 2
    colsel = column_selectivity(used)
    print(f"[validation] column selectivity done, {time.time() - t0:.0f}s", flush=True)

    # task 3
    kl_blocks = []
    for mk, ds, bdir, s in used:
        kl_blocks.append(known_label_block(mk, ds, bdir, s, a.draws, labels[ds], rows_by_ds[ds]))
        print(f"[validation] known-label {mk}/{ds} done, {time.time() - t0:.0f}s", flush=True)
    kl_summary = {ds: known_label_dataset_summary([b for b in kl_blocks if b["dataset"] == ds]) for ds in DATASETS if any(b["dataset"] == ds for b in kl_blocks)}

    by_ds = {ds: sum(1 for _mk, d2, *_ in used if d2 == ds) for ds in DATASETS}
    R = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_root": str(run_root), "locus": a.locus, "fit_seed": a.fit_seed,
                  "draws": a.draws, "alpha": PRIMARY_ALPHA, "bootstrap_seed": BOOT_CORE_SEED, "n_blocks": len(used), "n_blocks_by_dataset": by_ds,
                  "n_cells": 6 * len(used), "blocks": [f"{mk}/{ds}" for mk, ds, *_ in used], "skipped": skipped,
                  "n_ansdir_blocks": len(ans_blocks), "ansdir_blocks": list(ans_blocks), "excluded_models": list(EXCLUDED_MODELS),
                  "seconds": round(time.time() - t0, 1)},
         "support": support,
         "answer_direction": {"blocks": ans_blocks, "aggregate": aggregate},
         "column_selectivity": colsel,
         "known_label": {"per_dataset": kl_summary, "blocks": kl_blocks}}
    (out_dir / "validation.json").write_text(json.dumps(clean(R), indent=1), encoding="utf-8")
    write_markdown(clean(R), out_dir / "validation.md")
    print(f"[validation] wrote {out_dir / 'validation.json'} and validation.md in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
