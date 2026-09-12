"""CPU statistics over the returned artefacts (README section 6). The project owner recomputes these
from the same files; this module lets the executor check its own deliverable and fill main-tables rows.

Implemented:
  calibration(model, dataset, locus)  -> per concept: real AUROC, control mean/range, selectivity S with
                                         2,000 unit-bootstrap draws (seed 2026090602), readable / answer-capable flags
  core(model, dataset, module)        -> W_qd (6x6 + random + sham), O_q, random p95 / |sham| reference, rank of the
                                         matched direction among the random family, per-sample vectors
  max_t(contrasts, draws)             -> two-sided simultaneous interval via max|Z| with fixed bootstrap SD (ddof=1)
  bootstrap_indices(dataset, n, seed) -> shared unit-bootstrap indices (5000 x n) in frozen test order
  label_shifts(...)                   -> positive-minus-negative margin shift per (question, direction)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from .images import DATA_ROOT, load_cohort, load_labels
from .protocol import (BOOT_CALIBRATION_DRAWS, BOOT_CALIBRATION_SEED, BOOT_CORE_DRAWS, BOOT_CORE_SEED, CONCEPTS, DATASETS,
                       N_RANDOM)
from .runpaths import outcomes_dir, run_dir

DATASET_ORDER = ["nih", "chexpert", "coco"]


# --------------------------------------------------------------------------------------------- helpers
def bootstrap_indices(dataset_id: str, n: int, seed: int, draws: int) -> np.ndarray:
    """Shared with-replacement indices. One generator per seed, datasets drawn in nih, chexpert, coco order so
    every model of one dataset uses identical draws and different datasets combine by draw number."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out = None
    for ds in DATASET_ORDER:
        idx = rng.integers(0, n, size=(draws, n), dtype=np.int64)
        if ds == dataset_id:
            out = idx
    return out


def auroc_safe(y, s) -> float:
    y = np.asarray(y); s = np.asarray(s)
    if len(y) == 0 or y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, s))


def max_t(estimates: np.ndarray, draws: np.ndarray, level: float = 0.95) -> dict:
    """Two-sided simultaneous intervals: Z = (C* - C)/sd, sd = bootstrap SD (ddof=1) fixed per contrast; critical
    value = linear 95th percentile of max_j |Z_bj|; components with sd <= 1e-12 have Z set to zero; if every
    component is degenerate critical = 0. Intervals still use the actual sd."""
    est = np.asarray(estimates, float)
    dr = np.asarray(draws, float)                      # (B, m)
    sd = dr.std(axis=0, ddof=1)
    ok = sd > 1e-12
    Z = np.zeros_like(dr)
    Z[:, ok] = (dr[:, ok] - est[ok]) / sd[ok]
    crit = float(np.percentile(np.abs(Z).max(axis=1), level * 100, method="linear")) if ok.any() else 0.0
    return {"estimate": est, "sd": sd, "critical": crit, "lower": est - crit * sd, "upper": est + crit * sd,
            "degenerate": (~ok).astype(int)}


# ----------------------------------------------------------------------------------------- calibration
def calibration(model_key: str, dataset_id: str, locus_id: str = "vis.last") -> dict:
    ps = pq.read_table(run_dir(model_key, dataset_id) / f"probe_scores.{locus_id}.parquet").to_pandas()
    ps = ps[(ps.role == "calibration") & (ps.fit_seed == 0)]
    cal_rows = load_cohort(dataset_id, ("calibration",))
    order = {r["row_id"]: i for i, r in enumerate(cal_rows)}
    n = len(cal_rows)
    idx = bootstrap_indices(dataset_id, n, BOOT_CALIBRATION_SEED, BOOT_CALIBRATION_DRAWS)
    # answers (clean CALIBRATION IY margins)
    cal_path = outcomes_dir(model_key, dataset_id) / "CALIBRATION.parquet"
    ans = pq.read_table(cal_path).to_pandas() if cal_path.exists() else None
    labels = load_labels(dataset_id)
    out = {}
    for c in CONCEPTS[dataset_id]:
        y = np.full(n, -1, np.int8)
        for r in cal_rows:
            lab = labels[(r["row_id"], c)]
            if lab["label_known"] == "true":
                y[order[r["row_id"]]] = int(lab["label"])
        known = y >= 0
        real = ps[(ps.concept == c) & (ps.probe_kind == "real")]
        s_real = np.full(n, np.nan); s_real[[order[r] for r in real.row_id]] = real.logit.values
        ctrl = ps[(ps.concept == c) & (ps.probe_kind == "control")]
        cs = {}
        for k, g in ctrl.groupby("control_seed"):
            v = np.full(n, np.nan); t = np.full(n, -1, np.int8)
            v[[order[r] for r in g.row_id]] = g.logit.values
            t[[order[r] for r in g.row_id]] = g.target_label.fillna(-1).astype(int).values
            cs[int(k)] = (v, t)
        cell = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "n_unknown": int((~known).sum())}
        cell["auroc_real"] = auroc_safe(y[known], s_real[known])
        ctrl_auc = [auroc_safe(t[t >= 0], v[t >= 0]) for v, t in cs.values()]
        cell["control_mean"] = float(np.nanmean(ctrl_auc)) if ctrl_auc else np.nan
        cell["control_min"], cell["control_max"] = (float(np.nanmin(ctrl_auc)), float(np.nanmax(ctrl_auc))) if ctrl_auc else (np.nan, np.nan)
        cell["selectivity"] = cell["auroc_real"] - cell["control_mean"]
        # bootstrap over units (one row per unit in calibration)
        S, A, valid = [], [], 0
        for b in range(idx.shape[0]):
            ii = idx[b]
            kk = known[ii]
            a = auroc_safe(y[ii][kk], s_real[ii][kk])
            cm = [auroc_safe(t[ii][t[ii] >= 0], v[ii][t[ii] >= 0]) for v, t in cs.values()]
            if np.isnan(a) or (cs and any(np.isnan(cm))):
                continue
            valid += 1; A.append(a); S.append(a - np.mean(cm) if cs else np.nan)
        cell["bootstrap_valid_draws"] = valid
        if valid >= 1900:
            cell["selectivity_lower95_one_sided"] = float(np.percentile(S, 5))
            cell["selectivity_ci95"] = [float(np.percentile(S, 2.5)), float(np.percentile(S, 97.5))]
            cell["auroc_real_ci95"] = [float(np.percentile(A, 2.5)), float(np.percentile(A, 97.5))]
        cell["readable"] = bool(cell["n_pos"] >= 10 and cell["n_neg"] >= 10 and valid >= 1900 and cs
                                and cell.get("selectivity_lower95_one_sided", -1) > 0)
        cell["readable_status"] = "readable" if cell["readable"] else (
            "insufficient_support" if cell["n_pos"] < 10 or cell["n_neg"] < 10 else
            ("controls_ineligible" if not cs else ("insufficient_draws" if valid < 1900 else "not_readable")))
        # answer capability from clean IY margins
        if ans is not None:
            a_c = ans[(ans.concept == c) & (ans.template_id == "IY") & (ans.sample_status == "OK")]
            m = np.full(n, np.nan); m[[order[r] for r in a_c.row_id]] = a_c.semantic_margin.values
            p = 1 / (1 + np.exp(-m))
            cell["answer_auroc"] = auroc_safe(y[known & ~np.isnan(m)], m[known & ~np.isnan(m)])
            cell["answer_brier"] = float(np.nanmean((p[known] - y[known]) ** 2))
            AA = []
            for b in range(idx.shape[0]):
                ii = idx[b]; kk = known[ii] & ~np.isnan(m[ii])
                a = auroc_safe(y[ii][kk], m[ii][kk])
                if not np.isnan(a):
                    AA.append(a)
            cell["answer_valid_draws"] = len(AA)
            if len(AA) >= 1900:
                cell["answer_auroc_lower95_one_sided"] = float(np.percentile(AA, 5))
            cell["answer_capable"] = bool(cell["n_pos"] >= 10 and cell["n_neg"] >= 10 and len(AA) >= 1900
                                          and cell.get("answer_auroc_lower95_one_sided", 0) > 0.5)
        out[c] = cell
    return out


# ------------------------------------------------------------------------------------------------- CORE
def core(model_key: str, dataset_id: str, module: str = "CORE", template_id: str = "IY", fit_seed: int = 0,
         alpha: float = 0.25, n_boot: int | None = None) -> dict:
    """W_qd = mean_i [p_i(q,d) - p_i(q,baseline)], O_q = W_qq - max_{d != q} W_qd, references, ranks, and the
    per-contrast bootstrap C_qd = W_qq - W_qd with max-T simultaneous intervals over the 6x5 family."""
    df = pq.read_table(outcomes_dir(model_key, dataset_id) / f"{module}.parquet",
                       columns=["row_id", "concept", "template_id", "fit_seed", "direction_id", "direction_kind", "alpha",
                                "p_present", "semantic_margin", "sample_status"]).to_pandas()
    df = df[(df.template_id == template_id) & (df.fit_seed == fit_seed) & (df.sample_status == "OK")]
    concepts = CONCEPTS[dataset_id]
    rows = load_cohort(dataset_id, ("test",))
    if module == "DOSE":
        rows = rows[:200]
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    base = df[df.direction_id == "baseline"]
    if base.empty:                                   # DOSE/REFIT reuse the CORE baseline
        cb = pq.read_table(outcomes_dir(model_key, dataset_id) / "CORE.parquet",
                           columns=["row_id", "concept", "template_id", "fit_seed", "direction_id", "p_present", "sample_status"]).to_pandas()
        base = cb[(cb.direction_id == "baseline") & (cb.template_id == template_id) & (cb.fit_seed == 0) & (cb.sample_status == "OK")]
    P0 = {}
    for q in concepts:
        v = np.full(n, np.nan); g = base[base.concept == q]
        v[[order[r] for r in g.row_id]] = g.p_present.values; P0[q] = v
    st = df[(df.direction_id != "baseline") & (np.isclose(df.alpha, alpha))]
    delta = {}                                       # (q, d) -> per-sample delta vector
    for (q, d), g in st.groupby(["concept", "direction_id"]):
        v = np.full(n, np.nan); v[[order[r] for r in g.row_id]] = g.p_present.values
        delta[(q, d)] = v - P0[q]
    W = {q: {d: float(np.nanmean(v)) for (qq, d), v in delta.items() if qq == q} for q in concepts}
    res = {"n_rows": n, "alpha": alpha, "W": W, "per_question": {}}
    for q in concepts:
        w = W[q]
        clin = {d: w[f"concept:{d}"] for d in concepts if f"concept:{d}" in w}
        rand = np.array([w[f"random:{i:03d}"] for i in range(N_RANDOM) if f"random:{i:03d}" in w])
        sham = w.get(f"sham:{q}", np.nan)
        own = clin[q] - max(v for d, v in clin.items() if d != q) if len(clin) == len(concepts) else np.nan
        cell = {"W_qq": clin.get(q), "O_q": own, "max_other_clinical": max((v for d, v in clin.items() if d != q), default=np.nan),
                "argmax_other": max(((v, d) for d, v in clin.items() if d != q), default=(np.nan, None))[1],
                "random_p95": float(np.percentile(rand, 95)) if len(rand) else np.nan, "random_max": float(rand.max()) if len(rand) else np.nan,
                "random_n": int(len(rand)), "abs_sham": abs(sham) if not np.isnan(sham) else np.nan,
                "rank_in_random_family": int(1 + (rand >= clin.get(q, np.nan)).sum()) if len(rand) else None}
        cell["steering_reference"] = bool(len(rand) == N_RANDOM and clin.get(q, -1) > 0 and clin[q] > cell["random_p95"]
                                          and clin[q] > cell["abs_sham"])
        res["per_question"][q] = cell
    # bootstrap of the 6x5 family C_qd = W_qq - W_qd, shared unit indices in frozen test order
    if all(f"concept:{d}" in W[q] for q in concepts for d in concepts):
        draws_n = n_boot or BOOT_CORE_DRAWS
        idx = bootstrap_indices(dataset_id, 600, BOOT_CORE_SEED, draws_n)[:, :n] if n == 600 else \
            np.random.Generator(np.random.PCG64(BOOT_CORE_SEED)).integers(0, n, size=(draws_n, n))
        names, est, mat = [], [], []
        for q in concepts:
            dq = delta[(q, f"concept:{q}")]
            for d in concepts:
                if d == q:
                    continue
                dd = delta[(q, f"concept:{d}")]
                c = dq - dd
                names.append(f"{q}-{d}"); est.append(float(np.nanmean(c))); mat.append(c)
        mat = np.stack(mat)                                 # (30, n)
        boot = np.stack([np.nanmean(mat[:, idx[b]], axis=1) for b in range(idx.shape[0])])   # (B, 30)
        mt = max_t(np.array(est), boot)
        res["contrasts"] = {nm: {"estimate": float(e), "sd": float(s), "lower": float(lo), "upper": float(hi)}
                            for nm, e, s, lo, hi in zip(names, mt["estimate"], mt["sd"], mt["lower"], mt["upper"])}
        res["max_t_critical"] = mt["critical"]
        # direct percentile interval for O_q (max recomputed inside every draw)
        Oq_draws = {}
        for qi, q in enumerate(concepts):
            block = boot[:, qi * 5:(qi + 1) * 5]
            Oq_draws[q] = block.min(axis=1)                 # min over contrasts = W_qq - max_d W_qd
            res["per_question"][q]["O_q_ci95_percentile"] = [float(np.percentile(Oq_draws[q], 2.5)), float(np.percentile(Oq_draws[q], 97.5))]
            lows = [res["contrasts"][f"{q}-{d}"]["lower"] for d in concepts if d != q]
            highs = [res["contrasts"][f"{q}-{d}"]["upper"] for d in concepts if d != q]
            res["per_question"][q]["verdict"] = ("stronger_competitor" if any(h < 0 for h in highs) else
                                                 ("fixed_family_advantage" if all(l > 0 for l in lows) else "unresolved"))
    return res


def label_shifts(model_key: str, dataset_id: str, module: str = "CORE", alpha: float = 0.25) -> dict:
    """Positive-minus-negative difference in mean semantic-margin shift, per (question, clinical direction)."""
    df = pq.read_table(outcomes_dir(model_key, dataset_id) / f"{module}.parquet",
                       columns=["row_id", "concept", "template_id", "fit_seed", "direction_id", "alpha", "semantic_margin", "sample_status"]).to_pandas()
    df = df[(df.template_id == "IY") & (df.fit_seed == 0) & (df.sample_status == "OK")]
    labels = load_labels(dataset_id)
    concepts = CONCEPTS[dataset_id]
    base = df[df.direction_id == "baseline"].set_index(["concept", "row_id"]).semantic_margin
    out = {}
    for q in concepts:
        out[q] = {}
        for d in concepts:
            g = df[(df.concept == q) & (df.direction_id == f"concept:{d}") & np.isclose(df.alpha, alpha)]
            shift = g.semantic_margin.values - base.loc[[(q, r) for r in g.row_id]].values
            y = np.array([int(labels[(r, q)]["label"]) if labels[(r, q)]["label_known"] == "true" else -1 for r in g.row_id])
            pos, neg = shift[y == 1], shift[y == 0]
            out[q][d] = {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "shift_pos": float(pos.mean()) if len(pos) else np.nan,
                         "shift_neg": float(neg.mean()) if len(neg) else np.nan,
                         "gap": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else np.nan}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--what", default="calibration,core", help="comma list of calibration,core,shifts")
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args()
    rd = run_dir(a.model_key, a.dataset)
    report = {}
    for w in a.what.split(","):
        if w == "calibration":
            report["calibration"] = calibration(a.model_key, a.dataset)
        elif w == "core":
            report["core"] = core(a.model_key, a.dataset, n_boot=a.n_boot)
        elif w == "shifts":
            report["label_shifts"] = label_shifts(a.model_key, a.dataset)
    (rd / "summary.json").write_text(json.dumps(report, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    if "calibration" in report:
        for c, cell in report["calibration"].items():
            print(f"CAL {c:14s} auroc={cell['auroc_real']:.4f} ctrl={cell['control_mean']:.4f} S={cell['selectivity']:.4f} "
                  f"readable={cell['readable']} answer_auroc={cell.get('answer_auroc', float('nan')):.4f} capable={cell.get('answer_capable')}")
    if "core" in report:
        for q, cell in report["core"]["per_question"].items():
            print(f"CORE {q:14s} W_qq={cell['W_qq']:.4f} O_q={cell['O_q']:.4f} (vs {cell['argmax_other']}) p95rand={cell['random_p95']:.4f} "
                  f"|sham|={cell['abs_sham']:.4f} rank={cell['rank_in_random_family']} ref={cell['steering_reference']} verdict={cell.get('verdict')}")
