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
  altdir(model, dataset)              -> ALTDIR: per alternative direction family (dom, pattern, orth, resid) the 6x6 write
                                         matrix, within-family O_q with percentile interval and max-T verdict, own write
                                         minus the strongest logistic competitor of CORE, and the per-concept cosines
  extcomp(model, dataset)             -> EXTCOMP: W_q,extra per extra direction, ownership over the extended competitor
                                         family (five protocol + extras) with interval and verdict, extras beating W_qq
  tokenw(model, dataset)              -> TOKENW: W / O per token-weighting variant with interval and verdict
  precision(model, dataset)           -> PRECISION: W / O per numerics setting on the first 200 rows against CORE
  ansdir(model, dataset)              -> ANSDIR: W^a / O^a of the answer directions with interval, verdict, references, cosines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from .altdir import load_altdir
from .ansdir import load_ansdir
from .extcomp import load_extcomp
from .images import DATA_ROOT, load_cohort, load_labels
from .protocol import (ALTDIR_FAMILIES, BOOT_CALIBRATION_DRAWS, BOOT_CALIBRATION_SEED, BOOT_CORE_DRAWS, BOOT_CORE_SEED,
                       BOOT_DIAGNOSTIC_SEED, CONCEPTS, DATASETS, EXTCOMP_LABELS, LOCI, MODULES, N_RANDOM, NUMERICS_DEFAULT,
                       PRECISION_SETTINGS, PRIMARY_ALPHA, PROMPT_CONCEPTS, TOKENW_VARIANTS, expected_rows, primary_template)
from .runpaths import outcomes_dir, run_dir

DATASET_ORDER = ["nih", "chexpert", "coco"]
MIN_CONTROLS_PER_DRAW = 10      # declared: a draw counts when at least half the 20 control AUROCs are estimable
PROMPT_CONTRASTS = (("wording_IY_minus_WY_O", "IY", "WY"), ("mapping_IA_minus_IB_O", "IA", "IB"))


def _primary(model_key: str, dataset_id: str, template_id: str | None = None) -> str:
    """Template the block's single-template modules were scored with: the caller's choice, else the block's primary
    template (protocol.primary_template: IY, or IB when IY failed preflight check E)."""
    return template_id or primary_template(run_dir(model_key, dataset_id))


def _eligibility(model_key: str, dataset_id: str) -> dict:
    p = run_dir(model_key, dataset_id) / "template_eligibility.json"
    return json.loads(p.read_text()) if p.exists() else {}


def prompt_contrast_ineligible(primary: str, elig: dict, ta: str, tb: str) -> str | None:
    """Why the PROMPT contrast O(ta) - O(tb) is not defined for a block (None when it is). Both contrasts are defined
    against the protocol's IY reference, so a block whose primary template is not IY, or whose ta/tb template is
    INELIGIBLE by preflight check E, gets an explicit INELIGIBLE cell instead of a number."""
    if primary != "IY":
        return f"primary template is {primary}, not IY"
    bad = [t for t in (ta, tb) if not elig.get(t, {}).get("eligible", True)]
    if bad:
        return "INELIGIBLE template(s): " + ", ".join(bad)
    return None


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


def core_bootstrap_indices(dataset_id: str, n: int, draws: int) -> np.ndarray:
    """Unit-bootstrap indices of the CORE family (README 6, computation details): the shared 600-row draws of
    BOOT_CORE_SEED when the block has the full 600 test rows, else a PCG64(BOOT_CORE_SEED) draw of the block's own size."""
    if n == 600:
        return bootstrap_indices(dataset_id, 600, BOOT_CORE_SEED, draws)
    return np.random.Generator(np.random.PCG64(BOOT_CORE_SEED)).integers(0, n, size=(draws, n))


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
def calibration(model_key: str, dataset_id: str, locus_id: str = "vis.last", template_id: str | None = None) -> dict:
    template_id = _primary(model_key, dataset_id, template_id)
    ps = pq.read_table(run_dir(model_key, dataset_id) / f"probe_scores.{locus_id}.parquet").to_pandas()
    ps = ps[(ps.role == "calibration") & (ps.fit_seed == 0)]
    cal_rows = load_cohort(dataset_id, ("calibration",))
    order = {r["row_id"]: i for i, r in enumerate(cal_rows)}
    n = len(cal_rows)
    idx = bootstrap_indices(dataset_id, n, BOOT_CALIBRATION_SEED, BOOT_CALIBRATION_DRAWS)
    # answers (clean CALIBRATION margins of the block's primary template)
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
        # single-class draws are excluded only for the affected AUROC (README 6, computation details): a draw is
        # valid when the real AUROC is finite and at least MIN_CONTROLS_PER_DRAW control AUROCs are finite
        S, A, valid, used = [], [], 0, []
        for b in range(idx.shape[0]):
            ii = idx[b]
            kk = known[ii]
            a = auroc_safe(y[ii][kk], s_real[ii][kk])
            cm = np.array([auroc_safe(t[ii][t[ii] >= 0], v[ii][t[ii] >= 0]) for v, t in cs.values()], dtype=float)
            fin = cm[np.isfinite(cm)] if cs else np.array([])
            if np.isnan(a) or (cs and len(fin) < MIN_CONTROLS_PER_DRAW):
                continue
            valid += 1; A.append(a); S.append(a - fin.mean() if cs else np.nan); used.append(len(fin))
        cell["bootstrap_valid_draws"] = valid
        cell["controls_used_per_draw_min"] = int(min(used)) if used else 0
        cell["controls_used_per_draw_mean"] = float(np.mean(used)) if used else 0.0
        if valid >= 1900:
            cell["selectivity_lower95_one_sided"] = float(np.percentile(S, 5))
            cell["selectivity_ci95"] = [float(np.percentile(S, 2.5)), float(np.percentile(S, 97.5))]
            cell["auroc_real_ci95"] = [float(np.percentile(A, 2.5)), float(np.percentile(A, 97.5))]
        cell["readable"] = bool(cell["n_pos"] >= 10 and cell["n_neg"] >= 10 and valid >= 1900 and cs
                                and cell.get("selectivity_lower95_one_sided", -1) > 0)
        cell["readable_status"] = "readable" if cell["readable"] else (
            "insufficient_support" if cell["n_pos"] < 10 or cell["n_neg"] < 10 else
            ("controls_ineligible" if not cs else ("insufficient_draws" if valid < 1900 else "not_readable")))
        # answer capability from the clean margins of the primary template
        if ans is not None:
            a_c = ans[(ans.concept == c) & (ans.template_id == template_id) & (ans.sample_status == "OK")]
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
def core(model_key: str, dataset_id: str, module: str = "CORE", template_id: str | None = None, fit_seed: int = 0,
         alpha: float = 0.25, n_boot: int | None = None) -> dict:
    """W_qd = mean_i [p_i(q,d) - p_i(q,baseline)], O_q = W_qq - max_{d != q} W_qd, references, ranks, and the
    per-contrast bootstrap C_qd = W_qq - W_qd with max-T simultaneous intervals over the 6x5 family.
    template_id defaults to the block's primary template."""
    template_id = _primary(model_key, dataset_id, template_id)
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
        idx = core_bootstrap_indices(dataset_id, n, n_boot or BOOT_CORE_DRAWS)
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


def label_shifts(model_key: str, dataset_id: str, module: str = "CORE", alpha: float = 0.25, template_id: str | None = None) -> dict:
    """Positive-minus-negative difference in mean semantic-margin shift, per (question, clinical direction), on the
    block's primary template unless template_id is given."""
    template_id = _primary(model_key, dataset_id, template_id)
    df = pq.read_table(outcomes_dir(model_key, dataset_id) / f"{module}.parquet",
                       columns=["row_id", "concept", "template_id", "fit_seed", "direction_id", "alpha", "semantic_margin", "sample_status"]).to_pandas()
    df = df[(df.template_id == template_id) & (df.fit_seed == 0) & (df.sample_status == "OK")]
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


# ------------------------------------------------------------------------------------------------- T3
def _load_module(model_key, dataset_id, module, cols=("row_id", "concept", "template_id", "fit_seed", "direction_id", "alpha",
                                                    "p_present", "semantic_margin", "sample_status")):
    p = outcomes_dir(model_key, dataset_id) / f"{module}.parquet"
    if not p.exists():
        return None
    df = pq.read_table(p, columns=list(cols)).to_pandas()
    return df[df.sample_status == "OK"]


def _matrix(df, concepts, order, alpha, *, template_id, fit_seed=0, baseline=None, value="p_present"):
    """per-sample delta[q][d] arrays (n,) for clinical directions at one alpha/template/seed; baseline df separate.
    template_id is explicit: the block's primary template for CORE/DOSE/REFIT/LOCUS, a PROMPT template otherwise."""
    n = len(order)
    base = baseline if baseline is not None else df[(df.direction_id == "baseline") & (df.template_id == template_id)]
    P0 = {}
    for q in concepts:
        v = np.full(n, np.nan); g = base[(base.concept == q) & (base.fit_seed == 0)]
        v[[order[r] for r in g.row_id if r in order]] = g[value].values[[i for i, r in enumerate(g.row_id) if r in order]]
        P0[q] = v
    st = df[(df.template_id == template_id) & (df.fit_seed == fit_seed) & np.isclose(df.alpha, alpha) & (df.direction_id != "baseline")]
    delta = {}
    for (q, d), g in st.groupby(["concept", "direction_id"]):
        keep = [i for i, r in enumerate(g.row_id) if r in order]
        v = np.full(n, np.nan); v[[order[r] for r in g.row_id.values[keep]]] = g[value].values[keep]
        delta[(q, d)] = v - P0[q]
    return delta


def _O_from_delta(delta, concepts, idx=None, sign=1.0, questions=None, prefix="concept"):
    """O_q = sign*W_qq - max_{d != q} sign*W_qd from per-sample deltas (optionally over a bootstrap index).
    `questions` restricts the rows of the surface (PROMPT scores only two questions); directions span `concepts`;
    `prefix` is the direction family (concept, or an ALTDIR family)."""
    out = {}
    for q in (questions or concepts):
        vals = {}
        for d in concepts:
            v = delta.get((q, f"{prefix}:{d}"))
            if v is None:
                return None
            vals[d] = sign * float(np.nanmean(v if idx is None else v[idx]))
        out[q] = vals[q] - max(vals[d] for d in concepts if d != q)
    return out


def t3(model_key: str, dataset_id: str, draws: int = 2000, template_id: str | None = None) -> dict:
    """Table 3 aggregates with paired unit-bootstrap intervals recomputed inside every draw (seed 2026090603).
    CORE/DOSE/REFIT/LOCUS deltas are taken on the block's primary template; the PROMPT contrasts need IY."""
    primary = _primary(model_key, dataset_id, template_id)
    concepts = CONCEPTS[dataset_id]
    rows = load_cohort(dataset_id, ("test",))
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    order200 = {r["row_id"]: i for i, r in enumerate(rows[:200])}
    core = _load_module(model_key, dataset_id, "CORE")
    out = {}
    if core is None:
        return out
    base_core = core[core.direction_id == "baseline"]
    rng = np.random.Generator(np.random.PCG64(BOOT_DIAGNOSTIC_SEED))
    idx600 = rng.integers(0, 600, size=(draws, 600))
    idx200 = rng.integers(0, 200, size=(draws, 200))

    def interval(fn, idx):
        est = fn(None)
        if est is None:
            return None
        boot = [fn(idx[b]) for b in range(idx.shape[0])]
        boot = [b for b in boot if b is not None and np.isfinite(b)]
        if len(boot) < 0.95 * idx.shape[0]:
            return {"estimate": est, "ci_low": np.nan, "ci_high": np.nan, "status": "INSUFFICIENT_DRAWS"}
        return {"estimate": est, "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)), "status": "COMPLETE"}

    # ---- DOSE: signed O range over six doses, median over concepts (first 200 rows; +0.25 from CORE)
    dose = _load_module(model_key, dataset_id, "DOSE")
    if dose is not None and len(dose):
        deltas = {0.25: _matrix(core, concepts, order200, 0.25, template_id=primary, baseline=base_core)}
        for a in DOSE_ALPHAS_T3:
            deltas[a] = _matrix(dose, concepts, order200, a, template_id=primary, baseline=base_core)
        def dose_stat(idx):
            per = []
            for q in concepts:
                Os = []
                for a, dl in deltas.items():
                    o = _O_from_delta(dl, [q] + [c for c in concepts if c != q], idx, sign=np.sign(a))
                    if o is None:
                        return None
                    Os.append(o[q])
                per.append(max(Os) - min(Os))
            return float(np.median(per))
        r = interval(dose_stat, idx200)
        if r:
            out["all|median_signed_dose_O_range"] = r
    # ---- REFIT: SD of O over seeds 0/1/2, median over concepts
    refit = _load_module(model_key, dataset_id, "REFIT")
    if refit is not None and len(refit):
        d0 = _matrix(core, concepts, order, 0.25, template_id=primary, baseline=base_core)
        d1 = _matrix(refit, concepts, order, 0.25, template_id=primary, fit_seed=1, baseline=base_core)
        d2 = _matrix(refit, concepts, order, 0.25, template_id=primary, fit_seed=2, baseline=base_core)
        def refit_stat(idx):
            per = []
            for q in concepts:
                Os = [_O_from_delta(dl, concepts, idx) for dl in (d0, d1, d2)]
                if any(o is None for o in Os):
                    return None
                per.append(float(np.std([o[q] for o in Os], ddof=1)))
            return float(np.median(per))
        r = interval(refit_stat, idx600)
        if r:
            out["all|median_refit_O_sd"] = r
    # ---- LOCUS: median O at the connector locus (its own baseline)
    locus = _load_module(model_key, dataset_id, "LOCUS")
    if locus is not None and len(locus):
        dl = _matrix(locus, concepts, order, 0.25, template_id=primary, baseline=locus[locus.direction_id == "baseline"])
        def locus_stat(idx):
            o = _O_from_delta(dl, concepts, idx)
            return None if o is None else float(np.median([o[q] for q in concepts]))
        r = interval(locus_stat, idx600)
        if r:
            out["all|connector_median_O"] = r
    # ---- label gap: matched direction, positive-minus-negative margin shift, median over concepts
    labels = load_labels(dataset_id)
    dm = _matrix(core, concepts, order, 0.25, template_id=primary, baseline=base_core, value="semantic_margin")
    dm_p = _matrix(core, concepts, order, 0.25, template_id=primary, baseline=base_core)      # p_present deltas (PROMPT wording)
    y = {q: np.array([int(labels[(r["row_id"], q)]["label"]) if labels[(r["row_id"], q)]["label_known"] == "true" else -1 for r in rows]) for q in concepts}
    def gap_stat(idx):
        per = []
        for q in concepts:
            v = dm.get((q, f"concept:{q}"))
            if v is None:
                return None
            vv, yy = (v, y[q]) if idx is None else (v[idx], y[q][idx])
            pos, neg = vv[yy == 1], vv[yy == 0]
            if len(pos) == 0 or len(neg) == 0:
                return None
            per.append(float(np.nanmean(pos) - np.nanmean(neg)))
        return float(np.median(per))
    r = interval(gap_stat, idx600)
    if r:
        out["all|median_label_gap"] = r
    # ---- PROMPT: wording (IY - WY) and mapping (IA - IB) O differences for the two designated concepts. Both are
    # defined against the IY reference: a block whose primary template is not IY, or whose IY/WY (IA/IB) template is
    # INELIGIBLE, gets an explicit INELIGIBLE cell (no number, no exception) so the table cell is filled either way.
    prompt = _load_module(model_key, dataset_id, "PROMPT")
    elig = _eligibility(model_key, dataset_id)
    complete = prompt is not None and len(prompt) > 0 and \
        len(prompt) >= 0.999 * expected_rows("PROMPT", dataset_id) - _ineligible_prompt_rows(model_key, dataset_id)
    base_prompt = prompt[prompt.direction_id == "baseline"] if complete else None
    for c in PROMPT_CONCEPTS[dataset_id]:
        for name, ta, tb in PROMPT_CONTRASTS:
            why = prompt_contrast_ineligible(primary, elig, ta, tb)
            if why:
                out[f"{c}|{name}"] = {"estimate": None, "ci_low": None, "ci_high": None, "status": "INELIGIBLE", "reason": why}
                continue
            if not complete:
                continue
            da = dm_p if ta == "IY" else _matrix(prompt, concepts, order, 0.25, template_id=ta, baseline=base_prompt)
            db = _matrix(prompt, concepts, order, 0.25, template_id=tb, baseline=base_prompt)
            def pstat(idx, da=da, db=db, c=c):
                oa = _O_from_delta(da, concepts, idx, questions=[c]); ob = _O_from_delta(db, concepts, idx, questions=[c])
                if oa is None or ob is None or c not in oa or c not in ob:
                    return None
                return oa[c] - ob[c]
            r = interval(pstat, idx600)
            if r:
                out[f"{c}|{name}"] = r
    return out


DOSE_ALPHAS_T3 = [-0.5, -0.25, -0.1, 0.1, 0.5]


# ------------------------------------------------------------------------------------ direction-family helpers
def _family_stack(delta: dict, concepts: list[str], prefix: str, n: int) -> np.ndarray:
    """(6, 6, n) per-sample deltas delta[(q, prefix:d)] in concept order; raises when a cell is missing."""
    missing = [f"{q}<-{prefix}:{d}" for q in concepts for d in concepts if (q, f"{prefix}:{d}") not in delta]
    if missing:
        raise RuntimeError(f"{prefix}: no scored rows for {len(missing)} (question, direction) cells, e.g. {missing[:3]}")
    return np.stack([np.stack([delta[(q, f"{prefix}:{d}")] for d in concepts]) for q in concepts]).astype(float).reshape(len(concepts), len(concepts), n)


def _direction_rows(delta: dict, concepts: list[str], direction_id: str, n: int) -> np.ndarray:
    """(6, n) per-sample deltas of one direction applied to every question; raises when a question is missing."""
    missing = [q for q in concepts if (q, direction_id) not in delta]
    if missing:
        raise RuntimeError(f"{direction_id}: no scored rows for question(s) {missing}")
    return np.stack([delta[(q, direction_id)] for q in concepts]).astype(float).reshape(len(concepts), n)


def _logistic_reference(d_log: dict, concepts: list[str], idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """CORE's logistic family: the 6x6 matrix, the per-draw strongest competitor (B, 6), and per question the own
    write, strongest competitor, O_q and the random-p95 / |sham| references."""
    n = idx.shape[1]
    L = _family_stack(d_log, concepts, "concept", n)
    Wlog = np.nanmean(L, axis=2)
    off = ~np.eye(len(concepts), dtype=bool)
    comp_boot = np.stack([np.where(off, np.nanmean(L[:, :, idx[b]], axis=2), -np.inf).max(axis=1) for b in range(idx.shape[0])])
    ref = {}
    for qi, q in enumerate(concepts):
        rand = np.array([np.nanmean(d_log[(q, f"random:{i:03d}")]) for i in range(N_RANDOM) if (q, f"random:{i:03d}") in d_log])
        sham = float(np.nanmean(d_log[(q, f"sham:{q}")])) if (q, f"sham:{q}") in d_log else np.nan
        others = {d: float(Wlog[qi, di]) for di, d in enumerate(concepts) if d != q}
        ref[q] = {"W_logistic_qq": float(Wlog[qi, qi]), "max_other_logistic": max(others.values()),
                  "argmax_other_logistic": max(others, key=others.get), "O_logistic_q": float(Wlog[qi, qi] - max(others.values())),
                  "random_p95": float(np.percentile(rand, 95)) if len(rand) else np.nan, "random_n": int(len(rand)),
                  "abs_sham": abs(sham) if not np.isnan(sham) else np.nan}
    return Wlog, comp_boot, ref


def _ownership_block(own: np.ndarray, fam: dict, concepts: list[str], idx: np.ndarray) -> dict:
    """Ownership of an own write against a competitor family, with the CORE rules.
    own: (6, n) per-sample deltas of each question's own direction; fam: name -> (6, n) deltas of that direction on
    every question; the competitors of question q are every name except q itself. Returns the write matrix W[q][name]
    (own under q's own name), per question W_qq / O_q / strongest competitor / percentile interval of O_q (max
    recomputed per draw) / verdict from the max-T simultaneous intervals over all (q, competitor) contrasts."""
    names = list(fam)
    F = np.stack([fam[nm] for nm in names], axis=1)                                       # (6, K, n)
    B = idx.shape[0]
    Wf, Wo = np.nanmean(F, axis=2), np.nanmean(own, axis=1)                                 # (6, K), (6,)
    Fb = np.stack([np.nanmean(F[:, :, idx[b]], axis=2) for b in range(B)])                  # (B, 6, K)
    Ob = np.stack([np.nanmean(own[:, idx[b]], axis=1) for b in range(B)])                   # (B, 6)
    cnames, est, boot, blocks = [], [], [], {}
    for qi, q in enumerate(concepts):
        blocks[q] = []
        for ki, nm in enumerate(names):
            if nm == q:
                continue
            blocks[q].append(len(cnames))
            cnames.append(f"{q}-{nm}"); est.append(float(Wo[qi] - Wf[qi, ki])); boot.append(Ob[:, qi] - Fb[:, qi, ki])
    boot = np.stack(boot, axis=1)
    mt = max_t(np.array(est), boot)
    res = {"W": {q: {**{nm: float(Wf[qi, ki]) for ki, nm in enumerate(names) if nm != q}, q: float(Wo[qi])} for qi, q in enumerate(concepts)},
           "n_scored_rows_min": int(min(np.min((~np.isnan(F)).sum(axis=2)), np.min((~np.isnan(own)).sum(axis=1)))),
           "contrasts": {nm: {"estimate": float(e), "sd": float(sd), "lower": float(lo), "upper": float(hi)}
                         for nm, e, sd, lo, hi in zip(cnames, mt["estimate"], mt["sd"], mt["lower"], mt["upper"])},
           "max_t_critical": mt["critical"], "per_question": {}, "own_boot": Ob}
    for qi, q in enumerate(concepts):
        others = {nm: float(Wf[qi, ki]) for ki, nm in enumerate(names) if nm != q}
        cols = blocks[q]
        lows = [mt["lower"][j] for j in cols]; highs = [mt["upper"][j] for j in cols]
        O_draws = boot[:, cols].min(axis=1)
        res["per_question"][q] = {
            "W_qq": float(Wo[qi]), "O_q": float(Wo[qi] - max(others.values())),
            "max_other": max(others.values()), "argmax_other": max(others, key=others.get), "n_competitors": len(others),
            "O_q_ci95_percentile": [float(np.percentile(O_draws, 2.5)), float(np.percentile(O_draws, 97.5))],
            "verdict": ("stronger_competitor" if any(h < 0 for h in highs) else
                        ("fixed_family_advantage" if all(l > 0 for l in lows) else "unresolved"))}
    return res


def _cross_reference(cell: dict, own_b: np.ndarray, comp_boot_q: np.ndarray, ref: dict) -> None:
    """Add the comparison against CORE's logistic family to a per-question cell: own write minus the strongest
    logistic competitor (paired percentile interval) and the CORE steering reference applied to the own write."""
    X = own_b - comp_boot_q
    w = cell["W_qq"]
    cell.update({"own_minus_max_logistic_competitor": float(w - ref["max_other_logistic"]),
                 "own_minus_max_logistic_competitor_ci95_percentile": [float(np.percentile(X, 2.5)), float(np.percentile(X, 97.5))],
                 "W_logistic_qq": ref["W_logistic_qq"], "O_logistic_q": ref["O_logistic_q"],
                 "random_p95": ref["random_p95"], "abs_sham": ref["abs_sham"],
                 "steering_reference": bool(ref["random_n"] == N_RANDOM and w > 0 and w > ref["random_p95"] and w > ref["abs_sham"])})


def _module_deltas(model_key: str, dataset_id: str, module: str, alpha: float, template_id: str | None, draws: int):
    """Shared preamble of the CORE-baseline modules: primary template, test order, CORE + module deltas against the
    CORE clean baseline, bootstrap indices and the logistic reference."""
    primary = _primary(model_key, dataset_id, template_id)
    concepts = CONCEPTS[dataset_id]
    rows = load_cohort(dataset_id, ("test",))
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    core_df = _load_module(model_key, dataset_id, "CORE")
    mod_df = _load_module(model_key, dataset_id, module)
    if core_df is None or mod_df is None or len(mod_df) == 0:
        raise FileNotFoundError(f"{module.lower()} needs merged outcomes/CORE.parquet and outcomes/{module}.parquet (python -m cftransfer.package)")
    base_core = core_df[core_df.direction_id == "baseline"]
    d_log = _matrix(core_df, concepts, order, alpha, template_id=primary, baseline=base_core)
    d_mod = _matrix(mod_df, concepts, order, alpha, template_id=primary, baseline=base_core)
    idx = core_bootstrap_indices(dataset_id, n, draws)
    Wlog, comp_boot, ref = _logistic_reference(d_log, concepts, idx)
    return primary, concepts, n, d_log, d_mod, idx, Wlog, comp_boot, ref


# ----------------------------------------------------------------------------------------------- ALTDIR
def altdir(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
           draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """ALTDIR statistics on the block's primary template against the CORE clean baseline, same rows / dose / locus /
    competitor protocol as CORE. For every alternative family F (dom, pattern, orth, resid):
      W^F_qd = mean_i [p_i(q, F:d) - p_i(q, baseline)]           6x6 write matrix
      O^F_q  = W^F_qq - max_{d != q} W^F_qd                        within-family ownership, percentile interval from the
                                                                  CORE unit-bootstrap indices (`draws`, default 2,000; max
                                                                  recomputed inside every draw)
      C^F_qd = W^F_qq - W^F_qd, 6x5 max-T simultaneous intervals   verdict as CORE: stronger_competitor / fixed_family_advantage / unresolved
      X^F_q  = W^F_qq - max_{d != q} W^log_qd                      own alternative write minus the strongest LOGISTIC competitor
                                                                  of CORE (percentile interval, paired draws)
    plus the CORE steering reference applied to W^F_qq (> 0, > random p95, > |sham| of CORE at the same dose) and the
    per-concept cosines / norm fractions from fits/<locus>/altdir_seed0.npz."""
    primary, concepts, n, d_log, d_alt, idx, Wlog, comp_boot, log_ref = _module_deltas(model_key, dataset_id, "ALTDIR", alpha, template_id, draws)
    # cosines from the prep file
    arr = load_altdir(model_key, dataset_id, LOCI["primary"])
    fo = list(arr["family_order"].astype(str))
    names = list(arr["concept_names"].astype(str))
    # resid bookkeeping arrays were added after the first NIH/COCO files were written; those files (bit-identical
    # directions) carry the all-rows, all-covariates, no-fallback case, which is what the defaults below state
    n_c = len(names)
    rows_used = arr["resid_rows_used"] if "resid_rows_used" in arr else np.full(n_c, int(arr["n_train_rows"]))
    fallback = arr["resid_fallback"] if "resid_fallback" in arr else np.zeros(n_c, bool)
    cov_used = arr["resid_covariates_used"] if "resid_covariates_used" in arr else ~np.eye(n_c, dtype=bool)
    cos = {}
    for ci, c in enumerate(names):
        cos[c] = {"model": {a: {b: float(arr["cos_model"][ci, ai, bi]) for bi, b in enumerate(fo)} for ai, a in enumerate(fo)},
                  "projected": {a: {b: float(arr["cos_projected"][ci, ai, bi]) for bi, b in enumerate(fo)} for ai, a in enumerate(fo)},
                  "orth_norm_removed_fraction": float(arr["orth_norm_removed_fraction"][ci]),
                  "resid_cos_to_logistic_projected": float(arr["resid_cos_to_logistic_projected"][ci]),
                  "resid_rows_used": int(rows_used[ci]), "resid_fallback": bool(fallback[ci]),
                  "resid_covariates_used": [d for di, d in enumerate(names) if cov_used[ci, di]]}
    res = {"n_rows": n, "alpha": alpha, "template_id": primary, "baseline_module": "CORE", "draws": int(idx.shape[0]),
           "families": list(ALTDIR_FAMILIES), "logistic_reference": log_ref, "cosines": cos}
    for fam in ALTDIR_FAMILIES:
        A = _family_stack(d_alt, concepts, fam, n)
        block = _ownership_block(np.stack([A[qi, qi] for qi in range(len(concepts))]), {d: A[:, di, :] for di, d in enumerate(concepts)}, concepts, idx)
        own_b = block.pop("own_boot")
        for qi, q in enumerate(concepts):
            cell = block["per_question"][q]
            _cross_reference(cell, own_b[:, qi], comp_boot[:, qi], log_ref[q])
            cell["cos_to_logistic_model"] = cos[q]["model"]["logistic"][fam]
            cell["cos_to_logistic_projected"] = cos[q]["projected"]["logistic"][fam]
            if fam == "orth":
                cell["norm_removed_fraction"] = cos[q]["orth_norm_removed_fraction"]
            if fam == "resid":
                cell["fallback"] = cos[q]["resid_fallback"]; cell["rows_used"] = cos[q]["resid_rows_used"]
                cell["covariates_used"] = cos[q]["resid_covariates_used"]
        res[fam] = block
    return res


# ---------------------------------------------------------------------------------------------- EXTCOMP
def extcomp(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
            draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """EXTCOMP statistics: how CORE ownership changes when the competitor family grows. With W_q,extra the write of each
    extra direction on question q (CORE baseline), the extended ownership is
      O^ext_q = W_qq - max( W_qd over the five protocol competitors, W_q,extra over the extra directions )
    with the percentile interval (max recomputed per draw), the count of extra directions whose write beats W_qq, and
    the verdict over the extended 6 x (5 + K) max-T family. The five-competitor O_q of CORE is kept alongside."""
    primary, concepts, n, d_log, d_ext, idx, Wlog, comp_boot, log_ref = _module_deltas(model_key, dataset_id, "EXTCOMP", alpha, template_id, draws)
    extras = list(EXTCOMP_LABELS[dataset_id])
    L = _family_stack(d_log, concepts, "concept", n)
    fam = {d: L[:, di, :] for di, d in enumerate(concepts)}
    fam.update({f"extra:{k}": _direction_rows(d_ext, concepts, f"extra:{k}", n) for k in extras})
    block = _ownership_block(np.stack([L[qi, qi] for qi in range(len(concepts))]), fam, concepts, idx)
    block.pop("own_boot")
    arr = load_extcomp(model_key, dataset_id, LOCI["primary"])
    en = list(arr["extra_names"].astype(str))
    info = {k: {"n_pos": int(arr["n_pos"][en.index(k)]), "n_neg": int(arr["n_neg"][en.index(k)]),
                "auroc_calibration": float(arr["auroc_calibration"][en.index(k)]), "auroc_test": float(arr["auroc_test"][en.index(k)]),
                "cos_model_to_protocol": {c: float(arr["cos_model"][en.index(k), ci]) for ci, c in enumerate(concepts)}} for k in extras}
    for qi, q in enumerate(concepts):
        cell = block["per_question"][q]
        w_extra = {k: block["W"][q][f"extra:{k}"] for k in extras}
        cell.update({"W_extra": w_extra, "n_extra_beating_own": int(sum(v > cell["W_qq"] for v in w_extra.values())),
                     "max_extra": max(w_extra.values()), "argmax_extra": max(w_extra, key=w_extra.get),
                     "O_core_q": log_ref[q]["O_logistic_q"], "max_other_protocol": log_ref[q]["max_other_logistic"],
                     "argmax_other_protocol": log_ref[q]["argmax_other_logistic"],
                     "random_p95": log_ref[q]["random_p95"], "abs_sham": log_ref[q]["abs_sham"]})
        cell["O_ext_q"] = cell["O_q"]
    return {"n_rows": n, "alpha": alpha, "template_id": primary, "baseline_module": "CORE", "draws": int(idx.shape[0]),
            "extra_directions": extras, "extra_info": info, "n_competitors": 5 + len(extras), **block}


# ----------------------------------------------------------------------------------------------- TOKENW
def tokenw(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
           draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """TOKENW statistics per weighting variant (tokenw = softmax token weights, topq = top-quarter tokens): the 6x6
    write matrix of the weighted logistic writes against the CORE baseline, O_q with percentile interval and max-T
    verdict within the variant, and the own weighted write minus the strongest uniform logistic competitor of CORE."""
    primary, concepts, n, d_log, d_tw, idx, Wlog, comp_boot, log_ref = _module_deltas(model_key, dataset_id, "TOKENW", alpha, template_id, draws)
    res = {"n_rows": n, "alpha": alpha, "template_id": primary, "baseline_module": "CORE", "draws": int(idx.shape[0]),
           "variants": list(TOKENW_VARIANTS), "logistic_reference": log_ref}
    for v in TOKENW_VARIANTS:
        A = _family_stack(d_tw, concepts, v, n)
        block = _ownership_block(np.stack([A[qi, qi] for qi in range(len(concepts))]), {d: A[:, di, :] for di, d in enumerate(concepts)}, concepts, idx)
        own_b = block.pop("own_boot")
        for qi, q in enumerate(concepts):
            _cross_reference(block["per_question"][q], own_b[:, qi], comp_boot[:, qi], log_ref[q])
        res[v] = block
    return res


# ----------------------------------------------------------------------------------------------- ANSDIR
def ansdir(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
           draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """ANSDIR statistics: the answer directions a_d written on every question against the CORE baseline.
      W^a_qd (6x6), O^a_q = W^a_qq - max_{d != q} W^a_qd with the percentile interval and the CORE-style verdict over
      the 6x5 max-T family; steering reference W^a_qq > 0, > the CORE random p95 (same rows and dose) and > |W(q, a-sham)|;
      own answer write minus the strongest LOGISTIC competitor of CORE (paired interval); cos(a_q, w_q), the 6x6 cosine
      matrix, CV R^2 and ridge alpha from fits/<locus>/ansdir_seed0.npz."""
    primary, concepts, n, d_log, d_ans, idx, Wlog, comp_boot, log_ref = _module_deltas(model_key, dataset_id, "ANSDIR", alpha, template_id, draws)
    A = _family_stack(d_ans, concepts, "ans", n)
    block = _ownership_block(np.stack([A[qi, qi] for qi in range(len(concepts))]), {d: A[:, di, :] for di, d in enumerate(concepts)}, concepts, idx)
    own_b = block.pop("own_boot")
    arr = load_ansdir(model_key, dataset_id, LOCI["primary"])
    names = list(arr["concept_names"].astype(str))
    for qi, q in enumerate(concepts):
        cell = block["per_question"][q]
        _cross_reference(cell, own_b[:, qi], comp_boot[:, qi], log_ref[q])
        sham = float(np.nanmean(d_ans[(q, f"anssham:{q}")])) if (q, f"anssham:{q}") in d_ans else np.nan
        cell["abs_sham_logistic"] = cell["abs_sham"]
        cell["abs_sham"] = abs(sham) if not np.isnan(sham) else np.nan          # the a-sham is this module's sham reference
        cell["steering_reference"] = bool(log_ref[q]["random_n"] == N_RANDOM and cell["W_qq"] > 0 and cell["W_qq"] > cell["random_p95"]
                                          and cell["W_qq"] > cell["abs_sham"])
        k = names.index(q)
        cell.update({"cos_to_logistic_model": float(arr["cos_to_logistic"][k]), "cos_to_logistic_projected": float(arr["cos_projected"][k, k]),
                     "cv_r2": float(arr["cv_r2"][k]), "ridge_alpha": float(arr["ridge_alpha"][k]), "n_rows_used": int(arr["n_rows_used"][k])})
    cos = {q: {d: float(arr["cos_model"][names.index(q), names.index(d)]) for d in concepts} for q in concepts}
    return {"n_rows": n, "alpha": alpha, "template_id": primary, "baseline_module": "CORE", "draws": int(idx.shape[0]),
            "n_train_rows": int(arr["n_train_rows"]), "logistic_reference": log_ref, "cos_model": cos, **block}


# -------------------------------------------------------------------------------------------- PRECISION
def _core_cells(delta: dict, concepts: list[str]) -> dict:
    """Point W (6x6 + random p95 + |sham|) and the two reference comparisons per question from per-sample deltas."""
    out = {}
    for q in concepts:
        clin = {d: float(np.nanmean(delta[(q, f"concept:{d}")])) for d in concepts if (q, f"concept:{d}") in delta}
        rand = np.array([np.nanmean(delta[(q, f"random:{i:03d}")]) for i in range(N_RANDOM) if (q, f"random:{i:03d}") in delta])
        sham = float(np.nanmean(delta[(q, f"sham:{q}")])) if (q, f"sham:{q}") in delta else np.nan
        if len(clin) != len(concepts):
            raise RuntimeError(f"{q}: clinical writes missing ({sorted(clin)})")
        others = {d: v for d, v in clin.items() if d != q}
        out[q] = {"W": clin, "W_qq": clin[q], "O_q": clin[q] - max(others.values()), "max_other": max(others.values()),
                  "argmax_other": max(others, key=others.get), "random_p95": float(np.percentile(rand, 95)) if len(rand) else np.nan,
                  "random_n": int(len(rand)), "abs_sham": abs(sham) if not np.isnan(sham) else np.nan,
                  "own_gt_competitor": bool(clin[q] > max(others.values())),
                  "own_gt_random_p95": bool(len(rand) and clin[q] > np.percentile(rand, 95))}
    return out


def precision(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA) -> dict:
    """PRECISION: the CORE grid on the first PRECISION_ROWS test rows under each numerics setting (fp32, batch1), each
    with its own clean baseline, against CORE (bf16, batched) on the same rows: W and O per setting, the maximum |dW|
    over the 6x6 clinical cells, dO per question, and whether the two point verdicts (own > strongest competitor,
    own > random p95) agree with CORE's on those rows. No bootstrap: the question is numerical, not sampling."""
    primary = _primary(model_key, dataset_id, template_id)
    concepts = CONCEPTS[dataset_id]
    n_rows = MODULES["PRECISION"].row_limit
    rows = load_cohort(dataset_id, ("test",))[:n_rows]
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    core_df = _load_module(model_key, dataset_id, "CORE")
    p = outcomes_dir(model_key, dataset_id) / "PRECISION.parquet"
    if core_df is None or not p.exists():
        raise FileNotFoundError("precision needs merged outcomes/CORE.parquet and outcomes/PRECISION.parquet (python -m cftransfer.package)")
    prec = pq.read_table(p, columns=["row_id", "concept", "template_id", "fit_seed", "direction_id", "alpha", "p_present",
                                     "semantic_margin", "sample_status", "numerics"]).to_pandas()
    prec = prec[prec.sample_status == "OK"]
    d_core = _matrix(core_df, concepts, order, alpha, template_id=primary, baseline=core_df[core_df.direction_id == "baseline"])
    core_cells = _core_cells(d_core, concepts)
    res = {"n_rows": len(rows), "alpha": alpha, "template_id": primary, "settings": list(PRECISION_SETTINGS),
           "core": {"numerics": NUMERICS_DEFAULT, "per_question": core_cells}}
    for st in PRECISION_SETTINGS:
        df = prec[prec.numerics == st]
        if df.empty:
            res[st] = {"status": "NOT_STARTED"}
            continue
        d_st = _matrix(df, concepts, order, alpha, template_id=primary, baseline=df[df.direction_id == "baseline"])
        cells = _core_cells(d_st, concepts)
        dW = np.array([[cells[q]["W"][d] - core_cells[q]["W"][d] for d in concepts] for q in concepts])
        for q in concepts:
            c = cells[q]
            c.update({"dW_qq": c["W_qq"] - core_cells[q]["W_qq"], "dO_q": c["O_q"] - core_cells[q]["O_q"],
                      "max_abs_dW_row": float(np.abs(dW[concepts.index(q)]).max()),
                      "agree_competitor": c["own_gt_competitor"] == core_cells[q]["own_gt_competitor"],
                      "agree_random_p95": c["own_gt_random_p95"] == core_cells[q]["own_gt_random_p95"]})
        n_scored = int(min((~np.isnan(d_st[(q, f"concept:{q}")])).sum() for q in concepts))
        res[st] = {"status": "COMPLETE" if n_scored == len(rows) else "RUNNING", "n_scored_rows_min": n_scored,
                   "max_abs_dW": float(np.abs(dW).max()), "max_abs_dO": float(max(abs(cells[q]["dO_q"]) for q in concepts)),
                   "n_agree_competitor": int(sum(cells[q]["agree_competitor"] for q in concepts)),
                   "n_agree_random_p95": int(sum(cells[q]["agree_random_p95"] for q in concepts)), "per_question": cells}
    return res


def _ineligible_prompt_rows(model_key, dataset_id) -> int:
    """Rows the PROMPT block legitimately lacks because templates were INELIGIBLE at preflight."""
    elig = _eligibility(model_key, dataset_id)
    bad = [t for t in ("WY", "IA", "IB", "WA", "WB") if not elig.get(t, {}).get("eligible", True)]
    return len(bad) * len(PROMPT_CONCEPTS[dataset_id]) * 127 * 600


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--what", default="calibration,core", help="comma list of calibration,core,shifts,t3,altdir,extcomp,tokenw,precision,ansdir")
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args()
    rd = run_dir(a.model_key, a.dataset)
    report = json.loads((rd / "summary.json").read_text()) if (rd / "summary.json").exists() else {}
    for w in a.what.split(","):
        if w == "calibration":
            report["calibration"] = calibration(a.model_key, a.dataset)
        elif w == "core":
            report["core"] = core(a.model_key, a.dataset, n_boot=a.n_boot)
        elif w == "shifts":
            report["label_shifts"] = label_shifts(a.model_key, a.dataset)
        elif w == "t3":
            report["t3"] = t3(a.model_key, a.dataset)
        elif w == "altdir":
            report["altdir"] = altdir(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "extcomp":
            report["extcomp"] = extcomp(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "tokenw":
            report["tokenw"] = tokenw(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "precision":
            report["precision"] = precision(a.model_key, a.dataset)
        elif w == "ansdir":
            report["ansdir"] = ansdir(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
    report["primary_template"] = primary_template(rd)       # template the single-template statistics above refer to
    (rd / "summary.json").write_text(json.dumps(report, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    if "calibration" in report:
        for c, cell in report["calibration"].items():
            print(f"CAL {c:14s} auroc={cell['auroc_real']:.4f} ctrl={cell['control_mean']:.4f} S={cell['selectivity']:.4f} "
                  f"readable={cell['readable']} answer_auroc={cell.get('answer_auroc', float('nan')):.4f} capable={cell.get('answer_capable')}")
    if "t3" in report:
        for k, v in report["t3"].items():
            print(f"T3  {k:40s} {v['estimate']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] {v['status']}")
    if "altdir" in report and "altdir" in a.what:
        for fam in report["altdir"]["families"]:
            for q, cell in report["altdir"][fam]["per_question"].items():
                lo, hi = cell["O_q_ci95_percentile"]; xlo, xhi = cell["own_minus_max_logistic_competitor_ci95_percentile"]
                print(f"ALTDIR {fam:8s} {q:14s} W_qq={cell['W_qq']:+.4f} O_q={cell['O_q']:+.4f} [{lo:+.4f},{hi:+.4f}] (vs {cell['argmax_other']}) "
                      f"own-maxlog={cell['own_minus_max_logistic_competitor']:+.4f} [{xlo:+.4f},{xhi:+.4f}] "
                      f"cos={cell['cos_to_logistic_model']:+.3f} ref={cell['steering_reference']} verdict={cell['verdict']}")
    if "extcomp" in report and "extcomp" in a.what:
        for q, cell in report["extcomp"]["per_question"].items():
            lo, hi = cell["O_q_ci95_percentile"]
            print(f"EXTCOMP {q:14s} W_qq={cell['W_qq']:+.4f} O_core={cell['O_core_q']:+.4f} O_ext={cell['O_ext_q']:+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"(vs {cell['argmax_other']}) extras_beating_own={cell['n_extra_beating_own']} verdict={cell['verdict']}")
    if "tokenw" in report and "tokenw" in a.what:
        for v in report["tokenw"]["variants"]:
            for q, cell in report["tokenw"][v]["per_question"].items():
                lo, hi = cell["O_q_ci95_percentile"]; xlo, xhi = cell["own_minus_max_logistic_competitor_ci95_percentile"]
                print(f"TOKENW {v:7s} {q:14s} W_qq={cell['W_qq']:+.4f} O_q={cell['O_q']:+.4f} [{lo:+.4f},{hi:+.4f}] "
                      f"own-maxlog={cell['own_minus_max_logistic_competitor']:+.4f} [{xlo:+.4f},{xhi:+.4f}] ref={cell['steering_reference']} verdict={cell['verdict']}")
    if "ansdir" in report and "ansdir" in a.what:
        for q, cell in report["ansdir"]["per_question"].items():
            lo, hi = cell["O_q_ci95_percentile"]; xlo, xhi = cell["own_minus_max_logistic_competitor_ci95_percentile"]
            print(f"ANSDIR {q:14s} W_qq={cell['W_qq']:+.4f} O_q={cell['O_q']:+.4f} [{lo:+.4f},{hi:+.4f}] (vs {cell['argmax_other']}) "
                  f"own-maxlog={cell['own_minus_max_logistic_competitor']:+.4f} [{xlo:+.4f},{xhi:+.4f}] cos={cell['cos_to_logistic_model']:+.3f} "
                  f"cv_r2={cell['cv_r2']:+.3f} ref={cell['steering_reference']} verdict={cell['verdict']}")
    if "precision" in report and "precision" in a.what:
        for st in report["precision"]["settings"]:
            r = report["precision"][st]
            if r.get("status") == "NOT_STARTED":
                print(f"PRECISION {st:7s} NOT_STARTED"); continue
            print(f"PRECISION {st:7s} {r['status']} max|dW|={r['max_abs_dW']:.4f} max|dO|={r['max_abs_dO']:.4f} "
                  f"agree competitor {r['n_agree_competitor']}/6, random p95 {r['n_agree_random_p95']}/6")
    if "core" in report and "core" in a.what:
        for q, cell in report["core"]["per_question"].items():
            print(f"CORE {q:14s} W_qq={cell['W_qq']:.4f} O_q={cell['O_q']:.4f} (vs {cell['argmax_other']}) p95rand={cell['random_p95']:.4f} "
                  f"|sham|={cell['abs_sham']:.4f} rank={cell['rank_in_random_family']} ref={cell['steering_reference']} verdict={cell.get('verdict')}")
