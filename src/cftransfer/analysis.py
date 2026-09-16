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
  altdird(model, dataset)             -> ALTDIRD: as altdir for the displacement-lifted dom / pattern families
  attr(model, dataset)                -> ATTR: 9x9 write matrix, ownership over {3 attributes + 6 clinical}, attribute readability
  ansdirt(model, dataset)             -> ANSDIRT: per template O^a with interval / verdict, PROMPT cross reference, transfer counts
  valid(model, dataset)               -> VALID: the CORE grade on the radiologist-labelled CheXpert valid rows, answer capability and
                                         probe readability against those labels, paired comparison with the block's CORE grade on test
  attr(model, dataset)                -> also folds in ATTRRAND when its outcomes exist: the attribute cells then carry the SAME
                                         random p95 reference as the clinical cells (sham-only and matched verdicts both recorded)
  validfit(model, dataset)            -> VALIDFIT: W / O of the expert-label (radiologist) refits against CORE's competitor family,
                                         random p95 and sham, plus the CPU cosines and cross-fitted AUROCs of the prep
  projseed(model, dataset)            -> PROJSEED: per projection seed the 6x6 write matrix, O_q with that projection's own
                                         119-random p95 and sham, interval, max-T verdict, and agreement with the seed-0 grade
  towerswap(model, dataset)           -> TOWERSWAP: the write matrix and ownership grade of the four {tower} x {reader}
                                         combinations of a Gemma 3 / MedGemma pair, the paired crossed-minus-native contrast,
                                         and the reader effect at a fixed tower next to the tower effect at a fixed reader
  replay(model, dataset)              -> REPLAY: the CORE grade on a stored consumed-block tensor replayed into this reader,
                                         paired against the block's own CORE grade on the same rows, the drift the replacement
                                         removed, and the cross-reader comparison on byte-identical input
  semend(model, dataset)              -> SEMEND: per endpoint (negated question, counterbalanced forced choice, finding-word
                                         log-probability) and per family (label / answer) a full 6x6 write matrix, the signed
                                         effect with its own random / sham steering reference, the max-T ownership verdict, the
                                         spillover onto the other five concepts' endpoints, the affirmative/negated correlation,
                                         and the incremental validity of ownership over write magnitude, probe selectivity and
                                         clean-answer AUROC (reported the same way whether or not it adds anything)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from .altdir import gram_spectrum_summary, load_altdir
from .ansdir import load_ansdir
from .attr import attribute_labels, attribute_value, load_attr
from .extcomp import load_extcomp
from .fit import load_fit
from .projseed import load_projseed
from .images import DATA_ROOT, load_cohort, load_labels
from .protocol import (ALTDIR_FAMILIES, ALTDIRD_FAMILIES, ANSDIRT_TEMPLATES, ATTR_CONCEPTS, BOOT_CALIBRATION_DRAWS,
                       BOOT_CALIBRATION_SEED, BOOT_CORE_DRAWS, BOOT_CORE_SEED, BOOT_DIAGNOSTIC_SEED, CONCEPTS, CONTROL_SEEDS, DATASETS,
                       EXTCOMP_LABELS, LOCI, MODULE_SETTINGS, MODULES, N_RANDOM, NUMERICS_DEFAULT, PRECISION_SETTINGS, PRIMARY_ALPHA,
                       PROJSEED_SEEDS, PROMPT_CONCEPTS, REPLAY_SOURCE, SEMEND_N_RANDOM, SEMEND_SIGN, SEMEND_TEMPLATES,
                       TOKENW_VARIANTS, TOWERSWAP_PAIRS, conditions_for, expected_rows, primary_template, semend_finding_word)
from .semend import load_semend
from .runpaths import outcomes_dir, run_dir, valid_features_dir
from .validfit import load_validfit

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
         alpha: float = 0.25, n_boot: int | None = None, row_limit: int | None = None, numerics: str | None = None) -> dict:
    """W_qd = mean_i [p_i(q,d) - p_i(q,baseline)], O_q = W_qq - max_{d != q} W_qd, references, ranks, and the
    per-contrast bootstrap C_qd = W_qq - W_qd with max-T simultaneous intervals over the 6x5 family.
    template_id defaults to the block's primary template. Rows: the module's cohort role (test; valid for VALID), its own
    row limit (DOSE / PRECISION: 200), or `row_limit` when given (CORE on the first 200 rows). `numerics` selects one
    setting of a per-setting module."""
    template_id = _primary(model_key, dataset_id, template_id)
    path = outcomes_dir(model_key, dataset_id) / f"{module}.parquet"
    cols = ["row_id", "concept", "template_id", "fit_seed", "direction_id", "direction_kind", "alpha", "p_present", "semantic_margin",
            "sample_status"]
    has_numerics = "numerics" in pq.read_schema(path).names
    df = pq.read_table(path, columns=cols + (["numerics"] if has_numerics else [])).to_pandas()
    df = df[(df.template_id == template_id) & (df.fit_seed == fit_seed) & (df.sample_status == "OK")]
    if numerics is not None:
        settings = MODULE_SETTINGS.get(module, (NUMERICS_DEFAULT,))
        if numerics not in settings:
            raise ValueError(f"{module}: numerics must be one of {settings}, got {numerics!r}")
        df = df[df.numerics == numerics] if has_numerics else df
    if df.empty:
        raise RuntimeError(f"{module}: no OK rows for template {template_id}, fit seed {fit_seed}" + (f", numerics {numerics}" if numerics else ""))
    concepts = CONCEPTS[dataset_id]
    rows = load_cohort(dataset_id, (MODULES[module].role,))
    limit = row_limit or MODULES[module].row_limit
    if limit:
        rows = rows[:limit]
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    df = df[df.row_id.isin(order)]                   # a row-limited module (DOSE, PRECISION, CORE on 200 rows) keeps only its rows
    base = df[df.direction_id == "baseline"]
    if base.empty:                                   # DOSE/REFIT reuse the CORE baseline
        cb = pq.read_table(outcomes_dir(model_key, dataset_id) / "CORE.parquet",
                           columns=["row_id", "concept", "template_id", "fit_seed", "direction_id", "p_present", "sample_status"]).to_pandas()
        base = cb[(cb.direction_id == "baseline") & (cb.template_id == template_id) & (cb.fit_seed == 0) & (cb.sample_status == "OK")
                  & cb.row_id.isin(order)]
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


def _ownership_block(own: np.ndarray, fam: dict, concepts: list[str], idx: np.ndarray, own_names: dict | None = None) -> dict:
    """Ownership of an own write against a competitor family, with the CORE rules.
    own: (Q, n) per-sample deltas of each question's own direction; fam: name -> (Q, n) deltas of that direction on
    every question; the competitors of question q are every name except its own (`own_names[q]`, default q).
    Returns the write matrix W[q][name] (own under its own name), per question W_qq / O_q / strongest competitor /
    percentile interval of O_q (max recomputed per draw) / verdict from the max-T simultaneous intervals over all
    (q, competitor) contrasts."""
    own_names = own_names or {q: q for q in concepts}
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
            if nm == own_names[q]:
                continue
            blocks[q].append(len(cnames))
            cnames.append(f"{q}-{nm}"); est.append(float(Wo[qi] - Wf[qi, ki])); boot.append(Ob[:, qi] - Fb[:, qi, ki])
    boot = np.stack(boot, axis=1)
    mt = max_t(np.array(est), boot)
    res = {"W": {q: {**{nm: float(Wf[qi, ki]) for ki, nm in enumerate(names) if nm != own_names[q]}, own_names[q]: float(Wo[qi])}
                 for qi, q in enumerate(concepts)},
           "n_scored_rows_min": int(min(np.min((~np.isnan(F)).sum(axis=2)), np.min((~np.isnan(own)).sum(axis=1)))),
           "contrasts": {nm: {"estimate": float(e), "sd": float(sd), "lower": float(lo), "upper": float(hi)}
                         for nm, e, sd, lo, hi in zip(cnames, mt["estimate"], mt["sd"], mt["lower"], mt["upper"])},
           "max_t_critical": mt["critical"], "per_question": {}, "own_boot": Ob}
    for qi, q in enumerate(concepts):
        others = {nm: float(Wf[qi, ki]) for ki, nm in enumerate(names) if nm != own_names[q]}
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


# ---------------------------------------------------------------------------------------------- ALTDIRD
def altdird(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
            draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """ALTDIRD statistics: as altdir for the displacement-lifted families dom_disp / pattern_disp (altdir.displacement_lift),
    with cosines to the logistic normal and to the coefficient-lifted dom / pattern, and the R^T R / D spectrum."""
    primary, concepts, n, d_log, d_alt, idx, Wlog, comp_boot, log_ref = _module_deltas(model_key, dataset_id, "ALTDIRD", alpha, template_id, draws)
    arr = load_altdir(model_key, dataset_id, LOCI["primary"])
    if "cos_model_ext" not in arr:
        raise FileNotFoundError(f"altdir_seed0.npz lacks the displacement families; re-run python -m cftransfer.altdir --model-key {model_key} --dataset {dataset_id}")
    fe = list(arr["family_order_ext"].astype(str))
    names = list(arr["concept_names"].astype(str))
    ce, cpx = arr["cos_model_ext"], arr["cos_projected_ext"]
    res = {"n_rows": n, "alpha": alpha, "template_id": primary, "baseline_module": "CORE", "draws": int(idx.shape[0]),
           "families": list(ALTDIRD_FAMILIES), "logistic_reference": log_ref,
           "gram_spectrum": gram_spectrum_summary(arr["disp_gram_eigenvalues"]),
           "cosines": {c: {a: {b: float(ce[ci, ai, bi]) for bi, b in enumerate(fe)} for ai, a in enumerate(fe)} for ci, c in enumerate(names)}}
    for fam in ALTDIRD_FAMILIES:
        src = fam.replace("_disp", "")
        A = _family_stack(d_alt, concepts, fam, n)
        block = _ownership_block(np.stack([A[qi, qi] for qi in range(len(concepts))]), {d: A[:, di, :] for di, d in enumerate(concepts)}, concepts, idx)
        own_b = block.pop("own_boot")
        for qi, q in enumerate(concepts):
            cell = block["per_question"][q]
            _cross_reference(cell, own_b[:, qi], comp_boot[:, qi], log_ref[q])
            k = names.index(q)
            cell.update({"cos_to_logistic_model": float(ce[k, 0, fe.index(fam)]), "cos_to_logistic_projected": float(cpx[k, 0, fe.index(fam)]),
                         f"cos_to_{src}_model": float(ce[k, fe.index(src), fe.index(fam)])})
        res[fam] = block
    return res


# ------------------------------------------------------------------------------------------------- ATTR
def _readability_cell(y: np.ndarray, s_real: np.ndarray, controls: list[tuple[np.ndarray, np.ndarray]], idx: np.ndarray) -> dict:
    """The campaign's readability rule (analysis.calibration) on given calibration-row scores: real AUROC, control
    mean / range, selectivity S with unit-bootstrap draws (valid when the real AUROC is finite and >= MIN_CONTROLS_PER_DRAW
    controls are), one-sided 95% lower bound, readable flag."""
    known = y >= 0
    cell = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "n_unknown": int((~known).sum())}
    cell["auroc_real"] = auroc_safe(y[known], s_real[known])
    ctrl_auc = [auroc_safe(t[t >= 0], v[t >= 0]) for v, t in controls]
    cell["control_mean"] = float(np.nanmean(ctrl_auc)) if ctrl_auc else np.nan
    cell["control_min"], cell["control_max"] = (float(np.nanmin(ctrl_auc)), float(np.nanmax(ctrl_auc))) if ctrl_auc else (np.nan, np.nan)
    cell["selectivity"] = cell["auroc_real"] - cell["control_mean"]
    S, A, used = [], [], []
    for b in range(idx.shape[0]):
        ii = idx[b]; kk = known[ii]
        a = auroc_safe(y[ii][kk], s_real[ii][kk])
        cm = np.array([auroc_safe(t[ii][t[ii] >= 0], v[ii][t[ii] >= 0]) for v, t in controls], dtype=float)
        fin = cm[np.isfinite(cm)] if controls else np.array([])
        if np.isnan(a) or (controls and len(fin) < MIN_CONTROLS_PER_DRAW):
            continue
        A.append(a); S.append(a - fin.mean() if controls else np.nan); used.append(len(fin))
    cell["bootstrap_valid_draws"] = len(A)
    cell["controls_used_per_draw_min"] = int(min(used)) if used else 0
    if len(A) >= 1900:
        cell["selectivity_lower95_one_sided"] = float(np.percentile(S, 5))
        cell["selectivity_ci95"] = [float(np.percentile(S, 2.5)), float(np.percentile(S, 97.5))]
        cell["auroc_real_ci95"] = [float(np.percentile(A, 2.5)), float(np.percentile(A, 97.5))]
    cell["readable"] = bool(cell["n_pos"] >= 10 and cell["n_neg"] >= 10 and len(A) >= 1900 and controls
                            and cell.get("selectivity_lower95_one_sided", -1) > 0)
    cell["readable_status"] = "readable" if cell["readable"] else (
        "insufficient_support" if cell["n_pos"] < 10 or cell["n_neg"] < 10 else
        ("controls_ineligible" if not controls else ("insufficient_draws" if len(A) < 1900 else "not_readable")))
    return cell


def _answer_cell(y: np.ndarray, margin: np.ndarray, idx: np.ndarray) -> dict:
    """Answer capability rule (analysis.calibration) for clean margins against a binary label."""
    known = (y >= 0) & ~np.isnan(margin)
    cell = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "answer_auroc": auroc_safe(y[known], margin[known]),
            "answer_brier": float(np.nanmean((1 / (1 + np.exp(-margin[known])) - y[known]) ** 2)) if known.any() else np.nan}
    AA = []
    for b in range(idx.shape[0]):
        ii = idx[b]; kk = known[ii]
        a = auroc_safe(y[ii][kk], margin[ii][kk])
        if not np.isnan(a):
            AA.append(a)
    cell["answer_valid_draws"] = len(AA)
    if len(AA) >= 1900:
        cell["answer_auroc_lower95_one_sided"] = float(np.percentile(AA, 5))
    cell["answer_capable"] = bool(cell["n_pos"] >= 10 and cell["n_neg"] >= 10 and len(AA) >= 1900 and cell.get("answer_auroc_lower95_one_sided", 0) > 0.5)
    return cell


def _attrrand_deltas(model_key: str, dataset_id: str, attr_df, order: dict, alpha: float, primary: str):
    """Per-sample deltas of the 119 protocol random directions on the three attribute questions (ATTRRAND) against
    ATTR's own attribute baseline, or None when the module has not been scored for this block."""
    ar = _load_module(model_key, dataset_id, "ATTRRAND")
    if ar is None or len(ar) == 0:
        return None
    ar = ar[ar.template_id == primary]
    if ar.empty:
        return None
    return _matrix(ar, list(ATTR_CONCEPTS), order, alpha, template_id=primary,
                   baseline=attr_df[attr_df.direction_id == "baseline"])


def attr(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
         draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """ATTR statistics. Questions Q = 3 attributes + 6 clinical, directions = 3 attribute directions + 6 clinical normals:
      W[q][d] (9x9) against the question's clean baseline (attribute questions: ATTR's own; clinical: CORE's)
      ownership of every question over the 9-direction family (8 competitors): O_q with percentile interval and max-T
      verdict; steering reference W_qq > 0, > |sham_q| (the question's sham scored in ATTR) and > the random p95 on the
      same rows -- CORE's 119-random family for a clinical question, and, once ATTRRAND has been scored, the SAME
      119-direction family written on the attribute questions (module ATTRRAND) for an attribute question, so both kinds
      of cell are graded against one reference. Every cell records `steering_reference_sham_only` (the sham-only rule,
      which is all an attribute cell had before ATTRRAND), `steering_reference_matched` (the matched random-p95 + sham
      rule, None when no random family is available for that cell) and `steering_reference_rule`, which names the
      reference the cell's `steering_reference` actually used. Clinical cells are unchanged by ATTRRAND.
      readability of the attribute probes on the calibration rows (selectivity against the 20 type->random-label controls,
      campaign rule) and answer capability of the clean attribute questions on the test rows, reported twice: the
      `answer_*` fields as the module has recorded them since its first run (labels from the prep file's stored test
      labels), and the `label_answer_*` / `label_n_*` fields recomputed here from the outcomes on disk against the
      attribute's manifest ground-truth label with exactly the clinical answerability definition of core()/calibration
      (>= 10 positives and negatives, `draws` unit-bootstrap draws, one-sided 95% lower AUROC bound > 0.5)."""
    primary = _primary(model_key, dataset_id, template_id)
    clin = CONCEPTS[dataset_id]
    Q = list(ATTR_CONCEPTS) + list(clin)
    rows = load_cohort(dataset_id, ("test",))
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    core_df = _load_module(model_key, dataset_id, "CORE")
    attr_df = _load_module(model_key, dataset_id, "ATTR")
    if core_df is None or attr_df is None or len(attr_df) == 0:
        raise FileNotFoundError("attr needs merged outcomes/CORE.parquet and outcomes/ATTR.parquet (python -m cftransfer.package)")
    base_core = core_df[core_df.direction_id == "baseline"]
    base = pd.concat([base_core[base_core.concept.isin(clin)], attr_df[attr_df.direction_id == "baseline"]], ignore_index=True)
    d = _matrix(attr_df, Q, order, alpha, template_id=primary, baseline=base)
    d_log = _matrix(core_df, clin, order, alpha, template_id=primary, baseline=base_core)
    d_ar = _attrrand_deltas(model_key, dataset_id, attr_df, order, alpha, primary)
    dir_names = [f"attr:{a}" for a in ATTR_CONCEPTS] + [f"concept:{c}" for c in clin]
    own_names = {q: (f"attr:{q}" if q in ATTR_CONCEPTS else f"concept:{q}") for q in Q}
    fam = {nm: _direction_rows(d, Q, nm, n) for nm in dir_names}
    own = np.stack([d[(q, own_names[q])] for q in Q]).astype(float)
    idx = core_bootstrap_indices(dataset_id, n, draws)
    block = _ownership_block(own, fam, Q, idx, own_names)
    block.pop("own_boot")
    n_attr_random = 0
    for q in Q:
        cell = block["per_question"][q]
        attribute = q in ATTR_CONCEPTS
        sk = f"attrsham:{q}" if attribute else f"sham:{q}"
        sham = float(np.nanmean(d[(q, sk)])) if (q, sk) in d else np.nan
        src = d_ar if attribute else d_log
        rand = np.array([]) if src is None else \
            np.array([np.nanmean(src[(q, f"random:{i:03d}")]) for i in range(N_RANDOM) if (q, f"random:{i:03d}") in src])
        if attribute:
            n_attr_random = max(n_attr_random, len(rand))
        w = cell["W_qq"]
        p95 = float(np.percentile(rand, 95)) if len(rand) == N_RANDOM else None
        cell.update({"question_kind": "attribute" if attribute else "clinical", "abs_sham": abs(sham) if not np.isnan(sham) else np.nan,
                     "random_reference": bool(len(rand) == N_RANDOM), "random_p95": p95, "random_n": int(len(rand)),
                     "random_max": float(rand.max()) if len(rand) else None,
                     "random_source": (("ATTRRAND" if attribute else "CORE") if len(rand) == N_RANDOM else None),
                     "rank_in_random_family": int(1 + (rand >= w).sum()) if len(rand) == N_RANDOM else None,
                     "own_direction": own_names[q]})
        sham_only = bool(w > 0 and w > cell["abs_sham"])
        matched = bool(sham_only and w > p95) if cell["random_reference"] else None
        cell.update({"steering_reference_sham_only": sham_only, "steering_reference_matched": matched,
                     "steering_reference": bool(matched) if matched is not None else sham_only,
                     "steering_reference_rule": ("random_p95_and_sham (%s 119-random family)" % cell["random_source"])
                     if cell["random_reference"] else "sham_only (no random family scored for this cell)"})
    # attribute readability (calibration rows) and answer capability (test rows) from the prep file and ATTR baselines
    arr = load_attr(model_key, dataset_id, LOCI["primary"])
    an = list(arr["attr_names"].astype(str))
    n_cal = len(arr["cal_row_ids"])
    idx_cal = bootstrap_indices(dataset_id, n_cal, BOOT_CALIBRATION_SEED, draws)
    idx_test = bootstrap_indices(dataset_id, n, BOOT_CALIBRATION_SEED, draws)
    test_pos = {r: i for i, r in enumerate(arr["test_row_ids"].astype(str))}
    labels = load_labels(dataset_id)
    c0 = CONCEPTS[dataset_id][0]
    attributes = {}
    for a in ATTR_CONCEPTS:
        k = an.index(a)
        controls = [(arr["cal_control_logits"][k, j], arr["cal_control_labels"][j]) for j in range(arr["cal_control_logits"].shape[1])
                    if np.isfinite(arr["cal_control_logits"][k, j]).all()] if bool(arr["controls_eligible"]) else []
        cell = _readability_cell(arr["cal_labels"][k].astype(int), arr["cal_real_logits"][k].astype(float), controls, idx_cal)
        cell["controls_note"] = "type->random-label controls over the view x sex x age-decade strata; an attribute label is one particular labelling of those strata"
        y_test = np.full(n, -1, int); m_test = np.full(n, np.nan)
        for r, i in order.items():
            if r in test_pos:
                y_test[i] = int(arr["test_labels"][k, test_pos[r]])
        g = attr_df[(attr_df.direction_id == "baseline") & (attr_df.concept == a)]
        m_test[[order[r] for r in g.row_id if r in order]] = g.semantic_margin.values[[i for i, r in enumerate(g.row_id) if r in order]]
        cell.update(_answer_cell(y_test, m_test, idx_test))
        # the quantity a reviewer expects, recomputed from the outcomes on disk against the MANIFEST attribute label,
        # with the clinical answerability definition of core()/calibration (10 positives / 10 negatives, lower bound > .5)
        y_manifest = np.array([attribute_value(a, labels[(r["row_id"], c0)]) for r in rows], int)
        cell.update({f"label_{kk}": vv for kk, vv in _answer_cell(y_manifest, m_test, idx_test).items()})
        cell["answer_reference"] = ("answer_*: clean answer margin of the attribute question vs the attribute label stored in "
                                    "fits/<locus>/attr_seed0.npz (test_labels). label_answer_*: the same margins vs the attribute's "
                                    "manifest ground-truth label read here from labels.csv, clinical answerability rule.")
        cell.update({"auroc_calibration_prep": float(arr["auroc_calibration"][k]), "auroc_test_prep": float(arr["auroc_test"][k]),
                     "n_train_pos": int(arr["n_pos"][k]), "n_train_neg": int(arr["n_neg"][k]),
                     "cos_model_to_clinical": {c: float(arr["cos_model"][k, ci]) for ci, c in enumerate(clin)}})
        attributes[a] = cell
    attrrand = {"available": d_ar is not None, "n_random": int(n_attr_random), "expected_random": N_RANDOM,
                "questions": list(ATTR_CONCEPTS), "baseline_module": "ATTR",
                "note": ("the three attribute questions carry the protocol's own 119-direction random family (module ATTRRAND), "
                         "so an attribute cell is graded against the same random p95 bar as a clinical cell")
                if d_ar is not None else
                ("ATTRRAND not scored for this block: attribute cells fall back to the sham-only reference "
                 "(steering_reference_rule records this per cell)")}
    return {"n_rows": n, "alpha": alpha, "template_id": primary, "draws": int(idx.shape[0]), "questions": Q, "directions": dir_names,
            "attributes": attributes, "attrrand": attrrand, **block}


# --------------------------------------------------------------------------------------------- ANSDIRT
def ansdirt(model_key: str, dataset_id: str, alpha: float = PRIMARY_ALPHA, draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """ANSDIRT statistics: the IY-fitted answer directions written under each other eligible template, with the
    module's own clean baseline per (question, template). Per template: W^a (6x6), O^a_q with percentile interval and
    6x5 max-T verdict, the a-sham reference, and for PROMPT's designated concepts the cross comparison against the
    logistic competitors and random p95 scored by PROMPT under that template (own answer write minus the strongest
    logistic competitor, paired interval). Transfer: the IY-owned a_q cells (from ansdir(): steering reference and
    fixed_family_advantage) and how many stay owned under each template (same rule; random p95 where PROMPT provides it)."""
    concepts = CONCEPTS[dataset_id]
    rows = load_cohort(dataset_id, ("test",))
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    df = _load_module(model_key, dataset_id, "ANSDIRT")
    if df is None or len(df) == 0:
        raise FileNotFoundError("ansdirt needs merged outcomes/ANSDIRT.parquet (python -m cftransfer.package)")
    prompt = _load_module(model_key, dataset_id, "PROMPT")
    elig = _eligibility(model_key, dataset_id)
    idx = core_bootstrap_indices(dataset_id, n, draws)
    B = idx.shape[0]
    try:
        iy = ansdir(model_key, dataset_id, alpha=alpha, draws=draws)
        owned_iy = {q for q, c in iy["per_question"].items() if c["steering_reference"] and c["verdict"] == "fixed_family_advantage"}
    except FileNotFoundError:
        iy, owned_iy = None, None
    res = {"n_rows": n, "alpha": alpha, "draws": B, "templates": [], "ineligible_templates": [], "not_started_templates": [],
           "iy_owned": sorted(owned_iy) if owned_iy is not None else None, "per_template": {}, "transfer": {}}
    for t in ANSDIRT_TEMPLATES:
        if not elig.get(t, {}).get("eligible", True):
            res["ineligible_templates"].append(t)
            continue
        dt = df[df.template_id == t]
        if dt.empty:
            res["not_started_templates"].append(t)
            continue
        d = _matrix(dt, concepts, order, alpha, template_id=t, baseline=dt[dt.direction_id == "baseline"])
        A = _family_stack(d, concepts, "ans", n)
        block = _ownership_block(np.stack([A[qi, qi] for qi in range(len(concepts))]), {c: A[:, ci, :] for ci, c in enumerate(concepts)}, concepts, idx)
        own_b = block.pop("own_boot")
        dp = None
        if prompt is not None and (prompt.template_id == t).any():
            pt = prompt[prompt.template_id == t]
            dp = _matrix(pt, concepts, order, alpha, template_id=t, baseline=pt[pt.direction_id == "baseline"])
        owned_t = set()
        for qi, q in enumerate(concepts):
            cell = block["per_question"][q]
            sham = float(np.nanmean(d[(q, f"anssham:{q}")])) if (q, f"anssham:{q}") in d else np.nan
            cell["abs_sham"] = abs(sham) if not np.isnan(sham) else np.nan
            cell["random_reference"], cell["random_p95"] = False, None
            if dp is not None and all((q, f"concept:{c}") in dp and np.isfinite(dp[(q, f"concept:{c}")]).any() for c in concepts):
                Wl = {c: float(np.nanmean(dp[(q, f"concept:{c}")])) for c in concepts}
                others = {c: v for c, v in Wl.items() if c != q}
                comp_b = np.stack([max(np.nanmean(dp[(q, f"concept:{c}")][idx[b]]) for c in concepts if c != q) for b in range(B)])
                X = own_b[:, qi] - comp_b
                rand = np.array([np.nanmean(dp[(q, f"random:{i:03d}")]) for i in range(N_RANDOM) if (q, f"random:{i:03d}") in dp])
                cell.update({"logistic_source": "PROMPT", "W_logistic_qq": Wl[q], "max_other_logistic": max(others.values()),
                             "argmax_other_logistic": max(others, key=others.get), "O_logistic_q": Wl[q] - max(others.values()),
                             "own_minus_max_logistic_competitor": float(cell["W_qq"] - max(others.values())),
                             "own_minus_max_logistic_competitor_ci95_percentile": [float(np.percentile(X, 2.5)), float(np.percentile(X, 97.5))],
                             "random_p95": float(np.percentile(rand, 95)) if len(rand) == N_RANDOM else None,
                             "random_reference": bool(len(rand) == N_RANDOM)})
            w = cell["W_qq"]
            cell["steering_reference"] = bool(w > 0 and w > cell["abs_sham"] and (not cell["random_reference"] or w > cell["random_p95"]))
            if cell["steering_reference"] and cell["verdict"] == "fixed_family_advantage":
                owned_t.add(q)
        block.update({"owned": sorted(owned_t), "n_owned": len(owned_t), "n_scored_rows_min": block["n_scored_rows_min"]})
        res["templates"].append(t)
        res["per_template"][t] = block
        if owned_iy is not None:
            res["transfer"][t] = {"n_owned_IY": len(owned_iy), "n_stay_owned": len(owned_iy & owned_t), "stay": sorted(owned_iy & owned_t),
                                  "lost": sorted(owned_iy - owned_t), "gained": sorted(owned_t - owned_iy)}
    return res


# -------------------------------------------------------------------------------------------- PRECISION
def _grade_cells(c: dict) -> dict:
    """The campaign grade of one core() result, per question: verdict, steering reference, O_q and its interval."""
    return {q: {k: v.get(k) for k in ("W_qq", "O_q", "O_q_ci95_percentile", "verdict", "steering_reference", "random_p95", "abs_sham",
                                      "argmax_other")} for q, v in c["per_question"].items()}


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


def precision(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
              draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """PRECISION: the CORE grid on the first PRECISION_ROWS test rows under each numerics setting (fp32, batch1), each
    with its own clean baseline, against CORE (bf16, batched) on the same rows: W and O per setting, the maximum |dW|
    over the 6x6 clinical cells, dO per question, and whether the two point verdicts (own > strongest competitor,
    own > random p95) agree with CORE's on those rows (point estimates), and, under "grade", the FULL campaign grade of
    every setting and of CORE on the 200 rows (core(): steering reference against the 119-random p95 and the sham on those
    rows, 6x5 max-T contrasts with `draws` unit-bootstrap draws, verdicts) with the count of cells whose verdict or
    steering reference changes, max |dW| over the whole 127-direction grid and max |d contrast|."""
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
    core_grade = core(model_key, dataset_id, "CORE", template_id=primary, alpha=alpha, n_boot=draws, row_limit=n_rows)
    res = {"n_rows": len(rows), "alpha": alpha, "template_id": primary, "settings": list(PRECISION_SETTINGS), "draws": draws,
           "core": {"numerics": NUMERICS_DEFAULT, "per_question": core_cells, "grade": _grade_cells(core_grade),
                    "max_t_critical": core_grade.get("max_t_critical")}}
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
        # full campaign grade under this setting versus CORE on the same rows
        g = core(model_key, dataset_id, "PRECISION", template_id=primary, alpha=alpha, n_boot=draws, numerics=st)
        grade = _grade_cells(g)
        cg = res["core"]["grade"]
        for q in concepts:
            grade[q]["verdict_core"] = cg[q]["verdict"]; grade[q]["verdict_changed"] = grade[q]["verdict"] != cg[q]["verdict"]
            grade[q]["steering_reference_core"] = cg[q]["steering_reference"]
            grade[q]["steering_reference_changed"] = grade[q]["steering_reference"] != cg[q]["steering_reference"]
            grade[q]["dO_q"] = (grade[q]["O_q"] - cg[q]["O_q"]) if grade[q]["O_q"] is not None and cg[q]["O_q"] is not None else None
        common = {q: sorted(set(g["W"][q]) & set(core_grade["W"][q])) for q in concepts}
        dW_grid = max(abs(g["W"][q][k] - core_grade["W"][q][k]) for q in concepts for k in common[q])
        dC = max(abs(g["contrasts"][k]["estimate"] - core_grade["contrasts"][k]["estimate"]) for k in g.get("contrasts", {}) if k in core_grade.get("contrasts", {}))
        res[st]["grade"] = {"per_question": grade, "n_verdict_changes": int(sum(grade[q]["verdict_changed"] for q in concepts)),
                            "n_steering_reference_changes": int(sum(grade[q]["steering_reference_changed"] for q in concepts)),
                            "max_abs_dW_grid": float(dW_grid), "n_grid_cells_compared": int(sum(len(v) for v in common.values())),
                            "max_abs_dcontrast": float(dC), "max_t_critical": g.get("max_t_critical")}
    return res


# ------------------------------------------------------------------------------------------------ VALID
def _valid_probe_scores(model_key: str, dataset_id: str, rows: list[dict], locus_id: str = "vis.last") -> dict | None:
    """Seed-0 probe logits on the valid rows from features/valid/<locus>.npz (None when those features are not extracted):
    per concept the real-probe logit vector (NaN for a row without features) and the (logit, target) pairs of the 20
    type->random-label control probes (target -1 for a row whose type was not seen in training)."""
    path = valid_features_dir(model_key, dataset_id) / f"{locus_id}.npz"
    if not path.exists():
        return None
    f = np.load(path)
    pos = {r: i for i, r in enumerate(f["row_id"].astype(str))}
    fit = load_fit(model_key, dataset_id, locus_id, 0)
    Zs = ((f["x"].astype(np.float32) @ fit["projection"] - fit["scaler_mean"]) / np.maximum(fit["scaler_scale"], 1e-8)).astype(np.float32)
    labels = load_labels(dataset_id)
    concepts = list(fit["concept_names"].astype(str))
    type_names = list(fit["type_names"].astype(str))
    n = len(rows)
    sel = np.array([pos.get(r["row_id"], -1) for r in rows])
    ok = sel >= 0
    type_ix = np.array([type_names.index(t) if (t := labels[(r["row_id"], concepts[0])]["type_id"]) in type_names else -1 for r in rows])
    out = {}
    for ci, c in enumerate(concepts):
        real = np.full(n, np.nan)
        real[ok] = Zs[sel[ok]] @ fit["coefficients"][ci] + fit["intercepts"][ci]
        controls = []
        if bool(fit["controls_eligible"]):
            for k in range(len(CONTROL_SEEDS)):
                v = np.full(n, np.nan)
                v[ok] = Zs[sel[ok]] @ fit["control_coefficients"][ci, k] + fit["control_intercepts"][ci, k]
                t = np.where(type_ix >= 0, fit["control_type_labels"][k][np.maximum(type_ix, 0)], -1).astype(np.int8)
                controls.append((v, t))
        out[c] = (real, controls)
    return out


def valid(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
          draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """VALID: the CORE grade on the radiologist-labelled valid rows (CheXpert, 200 rows) and its paired comparison with the
    block's CORE grade on the labeler-labelled test rows. Per concept on the valid rows:
      W_qq, strongest competitor, O_q with percentile interval and the 6x5 max-T verdict (core() on VALID.parquet: `draws`
      paired draws of PCG64(BOOT_CORE_SEED) at n = 200, as the PRECISION grade), the steering reference (random p95 / |sham|
      on the valid rows);
      answer-capable: the module's own clean answers against the radiologist label, campaign rule (>= 10 positives and
      negatives, `draws` unit-bootstrap draws of BOOT_CALIBRATION_SEED, one-sided 95% lower AUROC bound > 0.5);
      readable: the seed-0 probe on features/valid/vis.last.npz against the 20 type->random-label control probes, campaign
      rule (selectivity lower bound > 0); readable None / "features_not_available" until those features are extracted.
    `comparison`: per concept the test grade (CORE verdict, O_q and steering reference recomputed on the test rows with
    `draws` draws; readable and answer-capable from the block's recorded calibration grade in summary.json) next to the
    valid grade, dO_q, and the counts of concepts whose verdict, ownership, readability and answer capability agree."""
    primary = _primary(model_key, dataset_id, template_id)
    concepts = CONCEPTS[dataset_id]
    rows = load_cohort(dataset_id, ("valid",))
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    df = _load_module(model_key, dataset_id, "VALID")
    if df is None or len(df) == 0 or _load_module(model_key, dataset_id, "CORE") is None:
        raise FileNotFoundError("valid needs merged outcomes/VALID.parquet and outcomes/CORE.parquet (python -m cftransfer.package)")
    g = core(model_key, dataset_id, "VALID", template_id=primary, alpha=alpha, n_boot=draws)
    grade = _grade_cells(g)
    labels = load_labels(dataset_id)
    idx = bootstrap_indices(dataset_id, n, BOOT_CALIBRATION_SEED, draws)
    probes = _valid_probe_scores(model_key, dataset_id, rows)
    base = df[(df.direction_id == "baseline") & (df.template_id == primary)]
    per = {}
    for q in concepts:
        y = np.array([int(labels[(r["row_id"], q)]["label"]) if labels[(r["row_id"], q)]["label_known"] == "true" else -1 for r in rows])
        cell = dict(grade[q])
        cell["max_other"] = g["per_question"][q]["max_other_clinical"]
        gb = base[base.concept == q]
        m = np.full(n, np.nan); m[[order[r] for r in gb.row_id]] = gb.semantic_margin.values
        cell.update(_answer_cell(y, m, idx))
        if probes is None:
            cell.update({"readable": None, "readable_status": "features_not_available"})
        else:
            real, controls = probes[q]
            cell.update(_readability_cell(np.where(np.isnan(real), -1, y), real, controls, idx))
        cell["owned"] = bool(cell["steering_reference"] and cell["verdict"] == "fixed_family_advantage")
        cell["label_source"] = "radiologist"
        per[q] = cell
    # the same block's CORE grade on the test rows: verdict / O_q / reference recomputed, probe grades as recorded
    test_grade = _grade_cells(core(model_key, dataset_id, "CORE", template_id=primary, alpha=alpha, n_boot=draws))
    sp = run_dir(model_key, dataset_id) / "summary.json"
    cal = (json.loads(sp.read_text()).get("calibration") or {}) if sp.exists() else {}
    keys = ("verdict", "owned", "readable", "answer_capable")
    agree, compared = {k: 0 for k in keys}, {k: 0 for k in keys}
    comparison = {"test_rows": len(load_cohort(dataset_id, ("test",))), "valid_rows": n, "per_question": {}}
    for q in concepts:
        t = test_grade[q]
        test = {"W_qq": t["W_qq"], "O_q": t["O_q"], "verdict": t["verdict"], "steering_reference": t["steering_reference"],
                "owned": bool(t["steering_reference"] and t["verdict"] == "fixed_family_advantage"),
                "readable": (cal.get(q) or {}).get("readable"), "answer_capable": (cal.get(q) or {}).get("answer_capable"),
                "label_source": "labeler"}
        val = {k: per[q].get(k) for k in ("W_qq", "O_q", "verdict", "steering_reference", "owned", "readable", "answer_capable", "label_source")}
        for k in keys:
            if test[k] is not None and val[k] is not None:
                compared[k] += 1; agree[k] += int(test[k] == val[k])
        comparison["per_question"][q] = {"test": test, "valid": val,
                                         "dO_q": (val["O_q"] - test["O_q"]) if val["O_q"] is not None and test["O_q"] is not None else None}
    comparison.update({"n_agree": agree, "n_compared": compared})
    return {"n_rows": n, "alpha": alpha, "template_id": primary, "draws": int(draws), "label_source": "radiologist",
            "features_available": probes is not None, "W": g["W"], "contrasts": g.get("contrasts"), "max_t_critical": g.get("max_t_critical"),
            "per_question": per, "comparison": comparison}


# --------------------------------------------------------------------------------------------- VALIDFIT
def validfit(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
             draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """VALIDFIT statistics: the six concept directions REFITTED on the radiologist-labelled valid rows, written on the
    600 test rows at the primary template and dose against the CORE clean baseline, i.e. the same rows / dose / locus /
    competitor protocol as CORE, so only the LABEL SOURCE of the fit differs.
      W^e_qd (6x6), O^e_q = W^e_qq - max_{d != q} W^e_qd with the percentile interval (max recomputed per draw) and the
      CORE-style verdict over the 6x5 max-T family; the steering reference is CORE's own on these rows (W^e_qq > 0,
      > the 119-random p95, > |sham_q|), so ownership is graded against exactly the references the report-label
      directions are graded against; own expert write minus the strongest LOGISTIC competitor of CORE (paired interval).
    Per concept the CPU quantities of the prep are attached: the raw (model-space and projected) and covariance-whitened
    cosines to the report-label direction, the cross-fitted held-out probe AUROC against the expert labels and against
    the report-derived labels of the same rows, and the report-label direction's AUROC on those rows."""
    primary, concepts, n, d_log, d_vf, idx, Wlog, comp_boot, log_ref = _module_deltas(model_key, dataset_id, "VALIDFIT", alpha,
                                                                                     template_id, draws)
    A = _family_stack(d_vf, concepts, "vfit", n)
    block = _ownership_block(np.stack([A[qi, qi] for qi in range(len(concepts))]),
                             {d: A[:, di, :] for di, d in enumerate(concepts)}, concepts, idx)
    own_b = block.pop("own_boot")
    arr = load_validfit(model_key, dataset_id, LOCI["primary"])
    names = list(arr["concept_names"].astype(str))
    for qi, q in enumerate(concepts):
        cell = block["per_question"][q]
        _cross_reference(cell, own_b[:, qi], comp_boot[:, qi], log_ref[q])
        k = names.index(q)
        cell.update({"cos_to_report_model": float(arr["cos_model"][k]), "cos_to_report_projected": float(arr["cos_projected"][k]),
                     "cos_to_report_whitened": float(arr["cos_whitened"][k]),
                     "auroc_expert_heldout": float(arr["auroc_expert_heldout"][k]),
                     "auroc_report_labels_heldout": float(arr["auroc_report_labels_heldout"][k]),
                     "auroc_expert_in_sample": float(arr["auroc_expert_in_sample"][k]),
                     "report_direction_auroc_expert": float(arr["report_direction_auroc_expert"][k]),
                     "report_direction_auroc_report_labels": float(arr["report_direction_auroc_report_labels"][k]),
                     "n_valid_pos": int(arr["n_pos"][k]), "n_valid_neg": int(arr["n_neg"][k]),
                     "n_report_pos": int(arr["n_pos_report"][k]), "n_report_neg": int(arr["n_neg_report"][k]),
                     "folds_used": int(arr["folds_used"][k]), "n_heldout_rows": int(arr["n_heldout_rows"][k])})
        cell["owned"] = bool(cell["steering_reference"] and cell["verdict"] == "fixed_family_advantage")
        cell["owned_report_labels"] = bool(log_ref[q]["W_logistic_qq"] > 0 and log_ref[q]["O_logistic_q"] > 0)
    return {"n_rows": n, "alpha": alpha, "template_id": primary, "baseline_module": "CORE", "draws": int(idx.shape[0]),
            "label_source": "radiologist", "fit_rows": int(arr["n_valid_rows"]), "n_folds": int(arr["n_folds"]),
            "cv_seed": int(arr["cv_seed"]), "report_label_source": str(arr["report_label_source"]),
            "logistic_reference": log_ref, **block}


# --------------------------------------------------------------------------------------------- PROJSEED
def projseed(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
             draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """PROJSEED statistics: one complete ownership grade per further projection seed. For each k in PROJSEED_SEEDS (the
    outcomes carry k in the fit_seed column):
      W^k_qd (6x6) against the CORE clean baseline, O^k_q with the percentile interval and the 6x5 max-T verdict;
      the steering reference is that projection's OWN family: W^k_qq > 0, > the p95 of its 119 re-drawn random
      directions and > |W(q, projsham_q)|; and the own write minus the strongest logistic competitor of the seed-0 CORE
      family (paired interval), plus the model-space cosine between the seed-k and seed-0 direction of the concept.
    `agreement` counts, over the six concepts, how many verdicts / ownership decisions match CORE's on the same rows."""
    primary, concepts, n, d_log, _d_mod, idx, Wlog, comp_boot, log_ref = _module_deltas(model_key, dataset_id, "PROJSEED", alpha,
                                                                                       template_id, draws)
    core_df = _load_module(model_key, dataset_id, "CORE")
    mod_df = _load_module(model_key, dataset_id, "PROJSEED")
    base_core = core_df[core_df.direction_id == "baseline"]
    rows = load_cohort(dataset_id, ("test",))
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    core_grade = core(model_key, dataset_id, "CORE", template_id=primary, alpha=alpha, n_boot=draws)
    res = {"n_rows": n, "alpha": alpha, "template_id": primary, "baseline_module": "CORE", "draws": int(idx.shape[0]),
           "seeds": list(PROJSEED_SEEDS), "not_started_seeds": [], "logistic_reference": log_ref,
           "core": {"projection_seed": 0, "grade": _grade_cells(core_grade), "max_t_critical": core_grade.get("max_t_critical")}}
    for k in PROJSEED_SEEDS:
        dk = mod_df[mod_df.fit_seed == k]
        if dk.empty:
            res["not_started_seeds"].append(k)
            continue
        d = _matrix(dk, concepts, order, alpha, template_id=primary, fit_seed=k, baseline=base_core)
        A = _family_stack(d, concepts, "proj", n)
        block = _ownership_block(np.stack([A[qi, qi] for qi in range(len(concepts))]),
                                 {c: A[:, ci, :] for ci, c in enumerate(concepts)}, concepts, idx)
        own_b = block.pop("own_boot")
        arr = load_projseed(model_key, dataset_id, LOCI["primary"], k)
        names = list(arr["concept_names"].astype(str))
        agree = {"verdict": 0, "owned": 0}
        for qi, q in enumerate(concepts):
            cell = block["per_question"][q]
            _cross_reference(cell, own_b[:, qi], comp_boot[:, qi], log_ref[q])
            rand = np.array([np.nanmean(d[(q, f"projrand:{i:03d}")]) for i in range(N_RANDOM) if (q, f"projrand:{i:03d}") in d])
            sham = float(np.nanmean(d[(q, f"projsham:{q}")])) if (q, f"projsham:{q}") in d else np.nan
            w = cell["W_qq"]
            # this projection's own references replace CORE's (which _cross_reference stored under the *_logistic names)
            cell["random_p95_logistic"] = cell["random_p95"]; cell["abs_sham_logistic"] = cell["abs_sham"]
            cell.update({"random_p95": float(np.percentile(rand, 95)) if len(rand) == N_RANDOM else None,
                         "random_n": int(len(rand)), "random_max": float(rand.max()) if len(rand) else None,
                         "random_reference": bool(len(rand) == N_RANDOM), "abs_sham": abs(sham) if not np.isnan(sham) else np.nan,
                         "rank_in_random_family": int(1 + (rand >= w).sum()) if len(rand) == N_RANDOM else None,
                         "cos_to_seed0_model": float(arr["cos_model_to_seed0"][names.index(q)]),
                         "auroc_test_probe": float(arr["auroc_test"][names.index(q)])})
            cell["steering_reference"] = bool(cell["random_reference"] and w > 0 and w > cell["random_p95"] and w > cell["abs_sham"])
            cell["owned"] = bool(cell["steering_reference"] and cell["verdict"] == "fixed_family_advantage")
            cg = res["core"]["grade"][q]
            core_owned = bool(cg["steering_reference"] and cg["verdict"] == "fixed_family_advantage")
            cell.update({"verdict_core": cg["verdict"], "verdict_changed": cell["verdict"] != cg["verdict"],
                         "owned_core": core_owned, "owned_changed": cell["owned"] != core_owned,
                         "dO_q": (cell["O_q"] - cg["O_q"]) if cg["O_q"] is not None else None})
            agree["verdict"] += int(not cell["verdict_changed"]); agree["owned"] += int(not cell["owned_changed"])
        block.update({"projection_seed": k, "n_agree": agree, "n_compared": len(concepts),
                      "owned": sorted(q for q in concepts if block["per_question"][q]["owned"]),
                      "median_abs_cos_to_seed0": float(np.median(np.abs(arr["cos_model_to_seed0"])))})
        res[f"seed{k}"] = block
    return res


# ------------------------------------------------------------------------------------------- TOWERSWAP
def _arm_deltas(model_key: str, dataset_id: str, module: str, concepts: list[str], order: dict, alpha: float,
                template_id: str, value: str = "p_present"):
    """Per-sample deltas of one scored arm against ITS OWN clean baseline (None when the arm is not packaged)."""
    df = _load_module(model_key, dataset_id, module)
    if df is None or df.empty:
        return None
    base = df[df.direction_id == "baseline"]
    if base.empty:
        raise RuntimeError(f"{model_key}/{dataset_id}/{module}: no clean baseline rows; this module scores its own")
    return _matrix(df, concepts, order, alpha, template_id=template_id, baseline=base, value=value)


def _own_rows(delta: dict, concepts: list[str], n: int, prefix: str = "concept") -> np.ndarray:
    """(6, n) per-sample deltas of each question's OWN direction."""
    return np.stack([delta[(q, f"{prefix}:{q}")] for q in concepts]).astype(float).reshape(len(concepts), n)


def _paired_contrast(a: np.ndarray, b: np.ndarray, concepts: list[str], idx: np.ndarray, names: list[str]) -> dict:
    """Question-wise a - b with percentile intervals and max-T simultaneous intervals over the six questions."""
    est = np.nanmean(a - b, axis=1)
    boot = np.stack([np.nanmean((a - b)[:, idx[j]], axis=1) for j in range(idx.shape[0])])      # (B, 6)
    mt = max_t(est, boot)
    return {q: {"estimate": float(est[qi]), "sd": float(mt["sd"][qi]),
                "ci95_percentile": [float(np.percentile(boot[:, qi], 2.5)), float(np.percentile(boot[:, qi], 97.5))],
                "max_t_lower": float(mt["lower"][qi]), "max_t_upper": float(mt["upper"][qi]),
                "nonzero_simultaneous": bool(mt["lower"][qi] > 0 or mt["upper"][qi] < 0)}
            for qi, q in enumerate(concepts)} | {"max_t_critical": mt["critical"], "contrast": names}


def towerswap(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
              draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """TOWERSWAP: the write matrix of the same reader under two towers, and of the same tower under two readers.

    The block's CROSSED arm (its own reader with the partner checkpoint's vision tower, written with the PARTNER's
    seed-0 directions) is graded exactly as CORE -- 6x6 write matrix, W_qq, O_q with the percentile interval, the
    119-random p95 / |sham| steering reference and the 6x5 max-T verdict -- and compared, paired on the same
    TOWERSWAP_ROWS test rows, with the block's NATIVE arm, which is its own CORE grade restricted to those rows (so no
    GPU time is spent re-running CORE). When the partner block is packaged too, all four combinations of
    {tower A, tower B} x {reader A, reader B} are reported, together with the two contrasts the claim turns on:
      reader effect at a fixed tower   W_qq(T, reader A) - W_qq(T, reader B)
      tower effect at a fixed reader   W_qq(A, reader R) - W_qq(B, reader R)
    each with max-T simultaneous intervals over the six questions. If what a written direction does is decided by the
    reader, the tower effect is small where the reader effect is not.
    """
    partner = TOWERSWAP_PAIRS.get(model_key)
    if partner is None:
        raise KeyError(f"{model_key}: not a TOWERSWAP pair member ({sorted(TOWERSWAP_PAIRS)})")
    primary = _primary(model_key, dataset_id, template_id)
    concepts = CONCEPTS[dataset_id]
    n_rows = MODULES["TOWERSWAP"].row_limit
    rows = load_cohort(dataset_id, ("test",))[:n_rows]
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    idx = core_bootstrap_indices(dataset_id, n, draws)
    res = {"n_rows": n, "alpha": alpha, "template_id": primary, "draws": int(idx.shape[0]), "reader": model_key,
           "partner": partner, "own_tower": model_key, "swapped_tower": partner, "combinations": {}, "arms": {}}
    swaps = sorted((outcomes_dir(model_key, dataset_id) / "TOWERSWAP").glob("swap-*.json"))
    res["swap_receipt"] = json.loads(swaps[-1].read_text()) if swaps else None
    arms: dict[str, np.ndarray] = {}
    for reader, tower, module in ((model_key, model_key, "CORE"), (model_key, partner, "TOWERSWAP"),
                                  (partner, partner, "CORE"), (partner, model_key, "TOWERSWAP")):
        name = f"tower={tower}|reader={reader}"
        tpl = _primary(reader, dataset_id, template_id)
        try:
            d = _arm_deltas(reader, dataset_id, module, concepts, order, alpha, tpl)
        except (FileNotFoundError, RuntimeError, KeyError):
            d = None
        if d is None or any((q, f"concept:{q}") not in d for q in concepts):
            res["combinations"][name] = {"status": "NOT_STARTED", "module": module, "tower": tower, "reader": reader}
            continue
        g = core(reader, dataset_id, module, template_id=tpl, alpha=alpha, n_boot=draws,
                 row_limit=n_rows if module == "CORE" else None)
        cell = _grade_cells(g)
        for q in concepts:
            cell[q]["owned"] = bool(cell[q]["steering_reference"] and cell[q]["verdict"] == "fixed_family_advantage")
        res["combinations"][name] = {"status": "COMPLETE", "module": module, "tower": tower, "reader": reader,
                                     "directions_from": tower, "template_id": tpl, "W": g["W"], "grade": cell,
                                     "max_t_critical": g.get("max_t_critical"),
                                     "owned": sorted(q for q in concepts if cell[q]["owned"])}
        arms[name] = _own_rows(d, concepts, n)
    res["arms"] = sorted(arms)
    a_self, a_cross = f"tower={model_key}|reader={model_key}", f"tower={partner}|reader={model_key}"
    if a_self in arms and a_cross in arms:                     # within the block: the tower effect at this reader
        res["swapped_minus_native"] = _paired_contrast(arms[a_cross], arms[a_self], concepts, idx,
                                                       [a_cross, a_self])
        for q in concepts:
            sw, na = res["combinations"][a_cross]["grade"][q], res["combinations"][a_self]["grade"][q]
            res["swapped_minus_native"][q].update({
                "W_qq_swapped": sw["W_qq"], "W_qq_native": na["W_qq"], "dO_q": (sw["O_q"] - na["O_q"]),
                "verdict_swapped": sw["verdict"], "verdict_native": na["verdict"],
                "verdict_changed": sw["verdict"] != na["verdict"], "owned_swapped": sw["owned"],
                "owned_native": na["owned"], "owned_changed": sw["owned"] != na["owned"]})
    b_self, b_cross = f"tower={partner}|reader={partner}", f"tower={model_key}|reader={partner}"
    have_all = all(k in arms for k in (a_self, a_cross, b_self, b_cross))
    res["crossover_complete"] = have_all
    if have_all:
        res["crossover"] = {
            "reader_effect_at_own_tower": _paired_contrast(arms[a_self], arms[b_cross], concepts, idx, [a_self, b_cross]),
            "reader_effect_at_partner_tower": _paired_contrast(arms[a_cross], arms[b_self], concepts, idx, [a_cross, b_self]),
            "tower_effect_at_own_reader": _paired_contrast(arms[a_cross], arms[a_self], concepts, idx, [a_cross, a_self]),
            "tower_effect_at_partner_reader": _paired_contrast(arms[b_cross], arms[b_self], concepts, idx, [b_cross, b_self])}
        for k, blk in res["crossover"].items():
            vals = [abs(blk[q]["estimate"]) for q in concepts]
            blk["mean_abs_effect"] = float(np.mean(vals))
            blk["n_simultaneously_nonzero"] = int(sum(blk[q]["nonzero_simultaneous"] for q in concepts))
        reader_eff = float(np.mean([res["crossover"]["reader_effect_at_own_tower"]["mean_abs_effect"],
                                    res["crossover"]["reader_effect_at_partner_tower"]["mean_abs_effect"]]))
        tower_eff = float(np.mean([res["crossover"]["tower_effect_at_own_reader"]["mean_abs_effect"],
                                   res["crossover"]["tower_effect_at_partner_reader"]["mean_abs_effect"]]))
        res["crossover"]["mean_abs_reader_effect"] = reader_eff
        res["crossover"]["mean_abs_tower_effect"] = tower_eff
        res["crossover"]["reader_over_tower"] = float(reader_eff / tower_eff) if tower_eff > 0 else None
        res["crossover"]["ownership_by_combination"] = {k: res["combinations"][k]["owned"]
                                                        for k in (a_self, a_cross, b_self, b_cross)}
        res["crossover"]["ownership_follows"] = {
            q: ("reader" if (res["combinations"][a_self]["grade"][q]["owned"] ==
                             res["combinations"][a_cross]["grade"][q]["owned"] and
                             res["combinations"][b_self]["grade"][q]["owned"] ==
                             res["combinations"][b_cross]["grade"][q]["owned"] and
                             res["combinations"][a_self]["grade"][q]["owned"] !=
                             res["combinations"][b_self]["grade"][q]["owned"])
                 else "tower" if (res["combinations"][a_self]["grade"][q]["owned"] ==
                                  res["combinations"][b_cross]["grade"][q]["owned"] and
                                  res["combinations"][a_cross]["grade"][q]["owned"] ==
                                  res["combinations"][b_self]["grade"][q]["owned"] and
                                  res["combinations"][a_self]["grade"][q]["owned"] !=
                                  res["combinations"][a_cross]["grade"][q]["owned"])
                 else "neither (all four agree or the pattern is mixed)") for q in concepts}
    return res


# ---------------------------------------------------------------------------------------------- REPLAY
def replay(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
           draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """REPLAY: the CORE grid scored on a stored consumed-block tensor, so two readers see byte-identical input.

    The block's REPLAY arm is graded exactly as CORE and compared, paired on the same REPLAY_ROWS test rows, with the
    block's own CORE grade on those rows (`replay_minus_core`). `drift` is what the replacement removed: the distance
    between this reader's own tower output and the stored tensor, recorded by the hook while it fired. When another
    block of the same shared-tower group is packaged, `group` reports the cross-reader difference of W_qq on the
    byte-identical input next to the same difference computed from the two blocks' own CORE runs, so the part of the
    reader gap that was bf16 kernel noise is separated from the part that is the reader.
    """
    source = REPLAY_SOURCE.get(model_key)
    if source is None:
        raise KeyError(f"{model_key}: not in a shared-tower group ({sorted(set(REPLAY_SOURCE.values()))})")
    primary = _primary(model_key, dataset_id, template_id)
    concepts = CONCEPTS[dataset_id]
    n_rows = MODULES["REPLAY"].row_limit
    rows = load_cohort(dataset_id, ("test",))[:n_rows]
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    idx = core_bootstrap_indices(dataset_id, n, draws)
    d_rep = _arm_deltas(model_key, dataset_id, "REPLAY", concepts, order, alpha, primary)
    d_core = _arm_deltas(model_key, dataset_id, "CORE", concepts, order, alpha, primary)
    if d_rep is None or d_core is None:
        raise FileNotFoundError("replay needs merged outcomes/CORE.parquet and outcomes/REPLAY.parquet "
                                "(python -m cftransfer.package)")
    g_rep = core(model_key, dataset_id, "REPLAY", template_id=primary, alpha=alpha, n_boot=draws)
    g_core = core(model_key, dataset_id, "CORE", template_id=primary, alpha=alpha, n_boot=draws, row_limit=n_rows)
    rep_cells, core_cells = _grade_cells(g_rep), _grade_cells(g_core)
    for cells in (rep_cells, core_cells):
        for q in concepts:
            cells[q]["owned"] = bool(cells[q]["steering_reference"] and cells[q]["verdict"] == "fixed_family_advantage")
    res = {"n_rows": n, "alpha": alpha, "template_id": primary, "draws": int(idx.shape[0]), "reader": model_key,
           "source_block": source, "directions_from": source,
           "group": sorted(k for k, v in REPLAY_SOURCE.items() if v == source),
           "replay": {"grade": rep_cells, "W": g_rep["W"], "max_t_critical": g_rep.get("max_t_critical"),
                      "owned": sorted(q for q in concepts if rep_cells[q]["owned"])},
           "core": {"grade": core_cells, "W": g_core["W"], "max_t_critical": g_core.get("max_t_critical"),
                    "owned": sorted(q for q in concepts if core_cells[q]["owned"])}}
    a, b = _own_rows(d_rep, concepts, n), _own_rows(d_core, concepts, n)
    res["replay_minus_core"] = _paired_contrast(a, b, concepts, idx, ["REPLAY", "CORE"])
    for q in concepts:
        res["replay_minus_core"][q].update({"dO_q": rep_cells[q]["O_q"] - core_cells[q]["O_q"],
                                            "verdict_changed": rep_cells[q]["verdict"] != core_cells[q]["verdict"],
                                            "owned_changed": rep_cells[q]["owned"] != core_cells[q]["owned"]})
    metas = [json.loads(p.read_text()) for p in (outcomes_dir(model_key, dataset_id) / "REPLAY").glob("meta-*.json")]
    drift = [m["replay"] for m in metas if isinstance(m.get("replay"), dict) and m["replay"].get("rows_replayed")]
    res["drift"] = ({"max_abs": max(d["drift_max_abs"] for d in drift),
                     "mean_abs": float(np.mean([d["drift_mean_abs"] for d in drift])),
                     "block_mean_abs": float(np.mean([d["block_mean_abs"] for d in drift])),
                     "rows_replayed": int(sum(d["rows_replayed"] for d in drift)),
                     "note": "|this reader's own tower output - the stored tensor|, measured before the replacement"}
                    if drift else {"available": False})
    res["group_comparison"] = {}
    for other in res["group"]:
        if other == model_key:
            continue
        try:
            d_other_rep = _arm_deltas(other, dataset_id, "REPLAY", concepts, order, alpha,
                                      _primary(other, dataset_id, template_id))
            d_other_core = _arm_deltas(other, dataset_id, "CORE", concepts, order, alpha,
                                       _primary(other, dataset_id, template_id))
        except (FileNotFoundError, RuntimeError, KeyError):
            d_other_rep = d_other_core = None
        if d_other_rep is None or any((q, f"concept:{q}") not in d_other_rep for q in concepts):
            res["group_comparison"][other] = {"status": "NOT_STARTED"}
            continue
        blk = {"status": "COMPLETE",
               "exact": _paired_contrast(a, _own_rows(d_other_rep, concepts, n), concepts, idx, [model_key, other])}
        if d_other_core is not None and all((q, f"concept:{q}") in d_other_core for q in concepts):
            blk["own_towers"] = _paired_contrast(b, _own_rows(d_other_core, concepts, n), concepts, idx,
                                                 [model_key, other])
            blk["mean_abs_exact"] = float(np.mean([abs(blk["exact"][q]["estimate"]) for q in concepts]))
            blk["mean_abs_own_towers"] = float(np.mean([abs(blk["own_towers"][q]["estimate"]) for q in concepts]))
            blk["mean_abs_change"] = blk["mean_abs_exact"] - blk["mean_abs_own_towers"]
        res["group_comparison"][other] = blk
    return res


# ---------------------------------------------------------------------------------------------- SEMEND
SEMEND_FAMILIES = {"label": ("concept", "sham"), "answer": ("ans", "anssham")}
SEMEND_BASE_PREDICTORS = ("W_qq_core", "probe_selectivity", "answer_auroc")
SEMEND_ADDED_PREDICTORS = ("owned_core", "O_q_core")


def _semend_frame(model_key: str, dataset_id: str):
    """SEMEND outcomes with the report endpoint's log p(finding word) added as a scored column."""
    p = outcomes_dir(model_key, dataset_id) / "SEMEND.parquet"
    if not p.exists():
        raise FileNotFoundError(f"semend needs {p} (python -m cftransfer.package)")
    df = pq.read_table(p, columns=["row_id", "concept", "template_id", "fit_seed", "direction_id", "alpha", "p_present",
                                   "semantic_margin", "positive_logits", "vocab_logsumexp", "sample_status"]).to_pandas()
    df = df[df.sample_status == "OK"].copy()
    pos = df["positive_logits"].map(lambda v: float(np.max(v)) if v is not None and len(v) else np.nan)
    df["logp"] = pos - df["vocab_logsumexp"].astype(float)      # log p(finding word) at the continuation position
    return df


def _r2(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    return beta, (1.0 - float((resid ** 2).sum()) / sst if sst > 0 else float("nan"))


def _incremental_validity(target_rows: np.ndarray, cells: list[dict], concepts: list[str], endpoints: list[str],
                          core_own: np.ndarray, core_off: dict, fixed: dict, idx: np.ndarray) -> dict:
    """Does the ownership label add signal about endpoint behaviour beyond write magnitude, readability, answerability?

    target_rows: (K, n) per-cell, per-sample intended-minus-unintended change (the selectivity of the write on this
    endpoint). Every predictor is a BLOCK-LEVEL per-concept quantity, so the pooled design has six distinct predictor
    rows repeated across the endpoints; endpoint dummies are in BOTH models, so the change in explained variance is
    entirely the concept-level contribution of ownership. W_qq and O_q are recomputed from the CORE writes inside every
    patient-bootstrap draw together with the target; probe selectivity and the clean-answer AUROC are computed on the
    CALIBRATION rows, so the test-row bootstrap leaves them fixed by construction.
    """
    K = len(cells)
    qi = np.array([concepts.index(c["concept"]) for c in cells])
    ep = np.array([endpoints.index(c["endpoint"]) for c in cells])
    dummies = np.stack([(ep == j).astype(float) for j in range(1, len(endpoints))], axis=1) if len(endpoints) > 1 \
        else np.zeros((K, 0))
    sel = np.array([fixed["probe_selectivity"].get(c["concept"], np.nan) for c in cells], float)
    auroc = np.array([fixed["answer_auroc"].get(c["concept"], np.nan) for c in cells], float)
    owned = np.array([1.0 if fixed["owned_core"].get(c["concept"]) else 0.0 for c in cells])
    if not np.isfinite(sel).all() or not np.isfinite(auroc).all():
        return {"available": False, "reason": "summary.json has no calibration selectivity / answer AUROC for every "
                                              "concept; run cftransfer.analysis --what calibration first"}

    def design(w, o):
        base = np.column_stack([np.ones(K), dummies, w[qi], sel, auroc])
        return base, np.column_stack([base, owned, o[qi]])

    def fit(w, o, y):
        base, full = design(w, o)
        bb, r2b = _r2(y, base)
        bf, r2f = _r2(y, full)
        return bb, r2b, bf, r2f
    w0 = np.nanmean(core_own, axis=1)
    o0 = np.array([w0[k] - max(np.nanmean(core_off[q], axis=1)) for k, q in enumerate(concepts)])
    y0 = np.nanmean(target_rows, axis=1)
    bb, r2b, bf, r2f = fit(w0, o0, y0)
    names_base = ["intercept"] + [f"endpoint:{e}" for e in endpoints[1:]] + list(SEMEND_BASE_PREDICTORS)
    names_full = names_base + list(SEMEND_ADDED_PREDICTORS)
    draws = []
    for b in range(idx.shape[0]):
        wb = np.nanmean(core_own[:, idx[b]], axis=1)
        ob = np.array([wb[k] - max(np.nanmean(core_off[q][:, idx[b]], axis=1)) for k, q in enumerate(concepts)])
        yb = np.nanmean(target_rows[:, idx[b]], axis=1)
        _bb, r2bb, bfb, r2fb = fit(wb, ob, yb)
        draws.append(np.concatenate([bfb, [r2fb - r2bb, r2bb, r2fb]]))
    D = np.stack(draws)

    def ci(col):
        return [float(np.percentile(D[:, col], 2.5)), float(np.percentile(D[:, col], 97.5))]
    k_owned, k_oq = names_full.index("owned_core"), names_full.index("O_q_core")
    add = {"delta_r2": float(r2f - r2b), "delta_r2_ci95": ci(len(names_full)),
           "coef_owned_core": float(bf[k_owned]), "coef_owned_core_ci95": ci(k_owned),
           "coef_O_q_core": float(bf[k_oq]), "coef_O_q_core_ci95": ci(k_oq)}
    add["owned_core_interval_excludes_zero"] = bool(add["coef_owned_core_ci95"][0] > 0 or add["coef_owned_core_ci95"][1] < 0)
    add["O_q_core_interval_excludes_zero"] = bool(add["coef_O_q_core_ci95"][0] > 0 or add["coef_O_q_core_ci95"][1] < 0)
    add["delta_r2_interval_excludes_zero"] = bool(add["delta_r2_ci95"][0] > 0)
    # the contrast ownership would have to explain: owned cells versus cells that met the steering reference but were
    # graded not owned, on endpoints no direction was fitted against
    grp = {"owned": [], "reference_met_not_owned": [], "neither": []}
    for k, c in enumerate(cells):
        q = c["concept"]
        g = "owned" if fixed["owned_core"].get(q) else (
            "reference_met_not_owned" if fixed["reference_core"].get(q) else "neither")
        grp[g].append(k)
        c["group_core"] = g
    groups = {}
    for g, ks in grp.items():
        groups[g] = {"n_cells": len(ks), "concepts": sorted({cells[k]["concept"] for k in ks}),
                     "mean_target": float(np.mean([y0[k] for k in ks])) if ks else None,
                     "mean_intended": float(np.mean([cells[k]["intended"] for k in ks])) if ks else None,
                     "mean_unintended": float(np.mean([cells[k]["unintended_mean"] for k in ks])) if ks else None}
    out = {"available": True, "n_cells": K, "n_concepts": len(concepts), "endpoints": endpoints,
           "base_predictors": list(SEMEND_BASE_PREDICTORS), "added_predictors": list(SEMEND_ADDED_PREDICTORS),
           "base_model": {"r2": float(r2b), "coefficients": dict(zip(names_base, [float(x) for x in bb]))},
           "full_model": {"r2": float(r2f), "coefficients": dict(zip(names_full, [float(x) for x in bf]))},
           "ownership_adds": add, "groups": groups,
           "note": "predictors are per-concept block-level quantities with endpoint dummies in both models; the effective "
                   "sample is the six concepts, not the K cells, and the patient bootstrap carries the dependence between "
                   "cells that share rows"}
    a, b_ = grp["owned"], grp["reference_met_not_owned"]
    if a and b_:
        diff = np.nanmean(target_rows[a], axis=0) - np.nanmean(target_rows[b_], axis=0)          # (n,)
        db = np.array([float(np.nanmean(diff[idx[j]])) for j in range(idx.shape[0])])
        out["owned_minus_reference_met_not_owned"] = {
            "estimate": float(np.nanmean(diff)), "ci95_percentile": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
            "n_owned_cells": len(a), "n_reference_met_not_owned_cells": len(b_),
            "interval_excludes_zero": bool(np.percentile(db, 2.5) > 0 or np.percentile(db, 97.5) < 0)}
    else:
        out["owned_minus_reference_met_not_owned"] = {
            "available": False, "n_owned_cells": len(a), "n_reference_met_not_owned_cells": len(b_),
            "reason": "one of the two groups is empty in this block"}
    return out


def semend(model_key: str, dataset_id: str, template_id: str | None = None, alpha: float = PRIMARY_ALPHA,
           draws: int = BOOT_CALIBRATION_DRAWS) -> dict:
    """SEMEND: three endpoints that are not the six templates, each with its own baseline and steering reference.

    Per endpoint t, family f (label = the six logistic normals, answer = the six ridge answer directions) and question q,
    with the endpoint's sign s_t (NY: -1, everything else +1) and the endpoint's own clean rows:
      E^{t,f}_{q,d} = s_t * mean_i [score_i(q, t, f:d) - score_i(q, t, baseline)]        a full 6x6 write matrix
                      score = p_present for NY / DA / DB, log p(finding word) for RF
      ownership     the campaign rule, unchanged: W_qq = E_{q,q}, O_q = W_qq - max_{d != q} E_{q,d} with the percentile
                    interval and the 6x5 max-T verdict, and a steering reference of the endpoint's own (W_qq > 0, above
                    every one of the SEMEND_N_RANDOM random directions, above that family's |sham|)
      spillover     intended = E_{q,q}; unintended = the same direction on the OTHER five concepts' endpoints,
                    mean and max of E_{d,q}; selectivity = intended - mean unintended
    `incremental_validity` then asks the question that decides whether the module was worth its compute: pooled over
    cells, does the CORE ownership label add anything about that selectivity once the CORE write magnitude, the probe
    selectivity and the clean-answer AUROC are in the model? It is reported separately for the two families, with the
    cells that met the steering reference but were graded NOT owned kept as their own group and compared directly with
    the owned cells. A null result is reported the same way as a positive one.
    """
    primary = _primary(model_key, dataset_id, template_id)
    concepts = CONCEPTS[dataset_id]
    rows = load_cohort(dataset_id, ("test",))
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    idx = core_bootstrap_indices(dataset_id, n, draws)
    df = _semend_frame(model_key, dataset_id)
    arr = load_semend(model_key, dataset_id, LOCI["primary"])
    comp = dict(zip(arr["concept_names"].astype(str), arr["competitor_names"].astype(str)))
    core_df = _load_module(model_key, dataset_id, "CORE")
    if core_df is None:
        raise FileNotFoundError("semend needs merged outcomes/CORE.parquet (python -m cftransfer.package)")
    d_log = _matrix(core_df, concepts, order, alpha, template_id=primary,
                    baseline=core_df[core_df.direction_id == "baseline"])
    core_grade = _grade_cells(core(model_key, dataset_id, "CORE", template_id=primary, alpha=alpha, n_boot=draws))
    core_own = np.stack([d_log[(q, f"concept:{q}")] for q in concepts]).astype(float)
    core_off = {q: np.stack([d_log[(q, f"concept:{c}")] for c in concepts if c != q]).astype(float) for q in concepts}
    sp = run_dir(model_key, dataset_id) / "summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    cal = {c: v for c, v in (summary.get("calibration") or {}).items() if isinstance(v, dict)}
    fixed = {"probe_selectivity": {q: cal.get(q, {}).get("selectivity", np.nan) for q in concepts},
             "answer_auroc": {q: cal.get(q, {}).get("answer_auroc", np.nan) for q in concepts},
             "owned_core": {q: bool(core_grade[q]["steering_reference"] and core_grade[q]["verdict"] == "fixed_family_advantage")
                            for q in concepts},
             "reference_core": {q: bool(core_grade[q]["steering_reference"]) for q in concepts}}
    res = {"n_rows": n, "alpha": alpha, "primary_template": primary, "draws": int(idx.shape[0]),
           "endpoints": list(SEMEND_TEMPLATES), "families": list(SEMEND_FAMILIES), "n_random": SEMEND_N_RANDOM,
           "competitor": comp, "competitor_source": json.loads(str(arr["meta_json"]))["competitor_source"],
           "prompts": {k: v["question"] for k, v in json.loads(str(arr["prompts_json"])).items()},
           "core": core_grade, "per_endpoint": {}, "cells": []}
    target_rows = {f: [] for f in SEMEND_FAMILIES}
    cells = {f: [] for f in SEMEND_FAMILIES}
    for t in SEMEND_TEMPLATES:
        sub = df[df.template_id == t]
        if sub.empty:
            res["per_endpoint"][t] = {"status": "NOT_STARTED"}
            continue
        value = "logp" if t == "RF" else "p_present"
        d = _matrix(sub, concepts, order, alpha, template_id=t, baseline=sub[sub.direction_id == "baseline"], value=value)
        s = SEMEND_SIGN[t]
        ep = {"status": None, "sign": s, "value": value,
              "n_conditions": len(conditions_for("SEMEND", dataset_id, concepts[0])), "families": {}}
        rand = {q: np.array([s * np.nanmean(d[(q, f"random:{i:03d}")]) for i in range(SEMEND_N_RANDOM)
                             if (q, f"random:{i:03d}") in d]) for q in concepts}
        n_scored = None
        for fam_name, (prefix, sham_prefix) in SEMEND_FAMILIES.items():
            A = s * _family_stack(d, concepts, prefix, n)                       # (question, direction, n)
            own = np.stack([A[qi, qi] for qi in range(len(concepts))])
            block = _ownership_block(own, {c: A[:, ci, :] for ci, c in enumerate(concepts)}, concepts, idx)
            block.pop("own_boot", None)
            n_scored = int(min((~np.isnan(own[qi])).sum() for qi in range(len(concepts))))
            for qi, q in enumerate(concepts):
                cell = block["per_question"][q]
                sham = float(np.nanmean(s * d[(q, f"{sham_prefix}:{q}")])) if (q, f"{sham_prefix}:{q}") in d else np.nan
                others = [di for di in range(len(concepts)) if di != qi]
                unint_rows = np.nanmean(A[others, qi, :], axis=0)               # this direction on the other endpoints
                tgt_rows = A[qi, qi, :] - unint_rows
                cell.update({"family": fam_name, "sign": s, "value": value, "competitor": comp[q],
                             "E_sham": sham, "abs_sham": abs(sham) if not np.isnan(sham) else np.nan,
                             "random_n": int(len(rand[q])),
                             "random_max": float(rand[q].max()) if len(rand[q]) else None,
                             "random_mean": float(rand[q].mean()) if len(rand[q]) else None,
                             "rank_in_random_family": int(1 + (rand[q] >= cell["W_qq"]).sum()) if len(rand[q]) else None,
                             "intended": float(np.nanmean(A[qi, qi, :])),
                             "unintended_mean": float(np.nanmean(unint_rows)),
                             "unintended_max": float(max(np.nanmean(A[di, qi, :]) for di in others)),
                             "selectivity": float(np.nanmean(tgt_rows)),
                             "W_qq_core": core_grade[q]["W_qq"], "O_q_core": core_grade[q]["O_q"],
                             "owned_core": fixed["owned_core"][q], "reference_core": fixed["reference_core"][q],
                             "probe_selectivity": fixed["probe_selectivity"][q],
                             "answer_auroc": fixed["answer_auroc"][q]})
                cell["steering_reference"] = bool(len(rand[q]) == SEMEND_N_RANDOM and cell["W_qq"] > 0
                                                  and cell["W_qq"] > cell["random_max"] and cell["W_qq"] > abs(sham))
                cell["owned"] = bool(cell["steering_reference"] and cell["verdict"] == "fixed_family_advantage")
                target_rows[fam_name].append(tgt_rows)
                rec = {"concept": q, "endpoint": t, "family": fam_name, "intended": cell["intended"],
                       "unintended_mean": cell["unintended_mean"], "unintended_max": cell["unintended_max"],
                       "selectivity": cell["selectivity"], "O_q_endpoint": cell["O_q"], "verdict": cell["verdict"],
                       "steering_reference": cell["steering_reference"], "owned": cell["owned"],
                       "owned_core": cell["owned_core"], "reference_core": cell["reference_core"]}
                cells[fam_name].append(rec)
                res["cells"].append(rec)
            block.update({"owned": sorted(q for q in concepts if block["per_question"][q]["owned"]),
                          "reference_met": sorted(q for q in concepts if block["per_question"][q]["steering_reference"])})
            ep["families"][fam_name] = block
        ep["status"] = "COMPLETE" if n_scored == n else "RUNNING"
        ep["n_scored_rows_min"] = n_scored
        ep["per_question"] = ep["families"]["label"]["per_question"]        # the label family is the module's primary read
        ep["owned"] = ep["families"]["label"]["owned"]
        ep["reference_met"] = ep["families"]["label"]["reference_met"]
        res["per_endpoint"][t] = ep
    done = [t for t in SEMEND_TEMPLATES if res["per_endpoint"][t].get("status") in ("COMPLETE", "RUNNING")]
    res["incremental_validity"] = {
        f: (_incremental_validity(np.stack(target_rows[f]), cells[f], concepts, done, core_own, core_off, fixed, idx)
            if target_rows[f] else {"available": False, "reason": "no scored endpoint"})
        for f in SEMEND_FAMILIES}
    # negated question: on the RAW p(yes) scale a concept direction must move the negated answer DOWN
    if "NY" in done:
        ny = res["per_endpoint"]["NY"]["families"]["label"]["per_question"]
        raw = {q: -ny[q]["W_qq"] for q in concepts}                              # undo the sign convention
        aff = {q: core_grade[q]["W_qq"] for q in concepts}
        x, y = np.array([aff[q] for q in concepts]), np.array([raw[q] for q in concepts])
        res["negation"] = {"raw_negated_effect": raw, "affirmative_effect_core": aff,
                           "n_opposite_sign": int(sum((x[i] > 0) != (y[i] > 0) for i in range(len(concepts)))),
                           "correlation_affirmative_vs_raw_negated":
                               float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None,
                           "correlation_affirmative_vs_signed_negated":
                               float(np.corrcoef(x, -y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None,
                           "n_reference_met": len(res["per_endpoint"]["NY"]["reference_met"]),
                           "rule": "a direction that carries the concept raises p(yes) to the affirmative question and "
                                   "lowers it to the negated one, so the two raw effects must anti-correlate"}
    # forced choice: the concept option under both orders
    if "DA" in done and "DB" in done:
        da = res["per_endpoint"]["DA"]["families"]["label"]["per_question"]
        db = res["per_endpoint"]["DB"]["families"]["label"]["per_question"]
        res["forced_choice"] = {q: {"DA": da[q]["W_qq"], "DB": db[q]["W_qq"],
                                    "mean": 0.5 * (da[q]["W_qq"] + db[q]["W_qq"]),
                                    "order_gap": da[q]["W_qq"] - db[q]["W_qq"], "competitor": comp[q],
                                    "both_reference_met": bool(da[q]["steering_reference"] and db[q]["steering_reference"]),
                                    "both_owned": bool(da[q]["owned"] and db[q]["owned"])} for q in concepts}
        res["forced_choice"]["n_both_owned"] = int(sum(res["forced_choice"][q]["both_owned"] for q in concepts))
        res["forced_choice"]["mean_abs_order_gap"] = float(np.mean([abs(res["forced_choice"][q]["order_gap"])
                                                                    for q in concepts]))
    if "RF" in done:
        rf = res["per_endpoint"]["RF"]["families"]["label"]["per_question"]
        res["report"] = {q: {"delta_logp": rf[q]["W_qq"], "delta_logp_max_competitor": rf[q]["max_other"],
                             "delta_logp_sham": rf[q]["E_sham"], "finding_word": semend_finding_word(dataset_id, q),
                             "reference_met": rf[q]["steering_reference"], "owned": rf[q]["owned"]} for q in concepts}
    res["summary"] = {"endpoints_scored": done, "n_cells": len(concepts) * len(done),
                      "n_reference_met": {f: sum(c["steering_reference"] for c in cells[f]) for f in SEMEND_FAMILIES},
                      "n_owned": {f: sum(c["owned"] for c in cells[f]) for f in SEMEND_FAMILIES},
                      "owned_core": sorted(q for q in concepts if fixed["owned_core"][q]),
                      "reference_met_not_owned_core": sorted(q for q in concepts if fixed["reference_core"][q]
                                                             and not fixed["owned_core"][q]),
                      "owned_on_every_endpoint": {
                          f: sorted(q for q in concepts
                                    if all(any(c["concept"] == q and c["endpoint"] == t and c["owned"] for c in cells[f])
                                           for t in done)) for f in SEMEND_FAMILIES}}
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
    ap.add_argument("--what", default="calibration,core",
                    help="comma list of calibration,core,shifts,t3,altdir,extcomp,tokenw,precision,ansdir,altdird,attr,ansdirt,"
                         "valid,validfit,projseed,towerswap,replay,semend (ATTRRAND is folded into attr)")
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
            report["precision"] = precision(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "ansdir":
            report["ansdir"] = ansdir(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "altdird":
            report["altdird"] = altdird(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "attr":
            report["attr"] = attr(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "ansdirt":
            report["ansdirt"] = ansdirt(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "valid":
            report["valid"] = valid(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "validfit":
            report["validfit"] = validfit(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "projseed":
            report["projseed"] = projseed(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "towerswap":
            report["towerswap"] = towerswap(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "replay":
            report["replay"] = replay(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
        elif w == "semend":
            report["semend"] = semend(a.model_key, a.dataset, draws=a.n_boot or BOOT_CALIBRATION_DRAWS)
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
            g = r.get("grade", {})
            print(f"PRECISION {st:7s} {r['status']} max|dW|={r['max_abs_dW']:.4f} max|dO|={r['max_abs_dO']:.4f} "
                  f"agree competitor {r['n_agree_competitor']}/6, random p95 {r['n_agree_random_p95']}/6; grade: verdict changes "
                  f"{g.get('n_verdict_changes')}/6, reference changes {g.get('n_steering_reference_changes')}/6, "
                  f"max|dW| grid={g.get('max_abs_dW_grid', float('nan')):.4f} max|dC|={g.get('max_abs_dcontrast', float('nan')):.4f}")
    if "altdird" in report and "altdird" in a.what:
        gs = report["altdird"]["gram_spectrum"]
        print(f"ALTDIRD R^T R / D spectrum: min={gs['min']:.6f} max={gs['max']:.6f} condition={gs['condition']:.2f}")
        for fam in report["altdird"]["families"]:
            for q, cell in report["altdird"][fam]["per_question"].items():
                lo, hi = cell["O_q_ci95_percentile"]
                print(f"ALTDIRD {fam:12s} {q:14s} W_qq={cell['W_qq']:+.4f} O_q={cell['O_q']:+.4f} [{lo:+.4f},{hi:+.4f}] "
                      f"own-maxlog={cell['own_minus_max_logistic_competitor']:+.4f} cos_log={cell['cos_to_logistic_model']:+.3f} ref={cell['steering_reference']} verdict={cell['verdict']}")
    if "attr" in report and "attr" in a.what:
        ar = report["attr"].get("attrrand") or {}
        print(f"ATTR attribute random family (ATTRRAND): available={ar.get('available')} n_random={ar.get('n_random')}")
        for a_name, cell in report["attr"]["attributes"].items():
            print(f"ATTR probe {a_name:8s} auroc={cell['auroc_real']:.4f} S={cell['selectivity']:+.4f} readable={cell['readable']} "
                  f"answer_auroc={cell.get('answer_auroc', float('nan')):.4f} capable={cell.get('answer_capable')} | "
                  f"label_answer_auroc={cell.get('label_answer_auroc', float('nan')):.4f} "
                  f"label_capable={cell.get('label_answer_capable')} (pos/neg {cell.get('label_n_pos')}/{cell.get('label_n_neg')})")
        for q, cell in report["attr"]["per_question"].items():
            lo, hi = cell["O_q_ci95_percentile"]
            print(f"ATTR {cell['question_kind']:9s} {q:14s} W_qq={cell['W_qq']:+.4f} O_q={cell['O_q']:+.4f} [{lo:+.4f},{hi:+.4f}] (vs {cell['argmax_other']}) "
                  f"ref={cell['steering_reference']} (sham_only={cell['steering_reference_sham_only']}, matched={cell['steering_reference_matched']}, "
                  f"{cell['steering_reference_rule']}) verdict={cell['verdict']}")
    if "validfit" in report and "validfit" in a.what:
        r = report["validfit"]
        print(f"VALIDFIT fitted on {r['fit_rows']} radiologist-labelled rows, {r['n_folds']}-fold cross-fitting; "
              f"report labels: {r['report_label_source']}")
        for q, cell in r["per_question"].items():
            lo, hi = cell["O_q_ci95_percentile"]
            print(f"VALIDFIT {q:14s} W_qq={cell['W_qq']:+.4f} O_q={cell['O_q']:+.4f} [{lo:+.4f},{hi:+.4f}] (vs {cell['argmax_other']}) "
                  f"cos_report={cell['cos_to_report_model']:+.4f} (whitened {cell['cos_to_report_whitened']:+.4f}) "
                  f"auroc_expert_heldout={cell['auroc_expert_heldout']:.4f} auroc_report_heldout={cell['auroc_report_labels_heldout']:.4f} "
                  f"ref={cell['steering_reference']} verdict={cell['verdict']} owned={cell['owned']}")
    if "projseed" in report and "projseed" in a.what:
        r = report["projseed"]
        for k in r["seeds"]:
            blk = r.get(f"seed{k}")
            if blk is None:
                print(f"PROJSEED seed{k} NOT_STARTED"); continue
            print(f"PROJSEED seed{k}: owned {len(blk['owned'])}/6 ({', '.join(blk['owned']) or '-'}); agrees with CORE on "
                  f"{blk['n_agree']['verdict']}/6 verdicts, {blk['n_agree']['owned']}/6 ownership decisions; "
                  f"median |cos| to seed 0 = {blk['median_abs_cos_to_seed0']:.4f}")
            for q, cell in blk["per_question"].items():
                lo, hi = cell["O_q_ci95_percentile"]
                print(f"PROJSEED seed{k} {q:14s} W_qq={cell['W_qq']:+.4f} O_q={cell['O_q']:+.4f} [{lo:+.4f},{hi:+.4f}] "
                      f"p95rand={cell['random_p95']:+.4f} |sham|={cell['abs_sham']:.4f} cos_seed0={cell['cos_to_seed0_model']:+.4f} "
                      f"ref={cell['steering_reference']} verdict={cell['verdict']} (core {cell['verdict_core']})")
    if "ansdirt" in report and "ansdirt" in a.what:
        r = report["ansdirt"]
        for t in r["templates"]:
            tr = r["transfer"].get(t, {})
            print(f"ANSDIRT {t}: owned {r['per_template'][t]['n_owned']}/6 ({', '.join(r['per_template'][t]['owned']) or '-'}); "
                  f"IY-owned staying: {tr.get('n_stay_owned', '?')}/{tr.get('n_owned_IY', '?')}")
        if r["ineligible_templates"]:
            print(f"ANSDIRT ineligible templates: {r['ineligible_templates']}")
    if "valid" in report and "valid" in a.what:
        r = report["valid"]
        for q, cell in r["per_question"].items():
            t = r["comparison"]["per_question"][q]["test"]
            print(f"VALID {q:14s} W_qq={cell['W_qq']:+.4f} O_q={cell['O_q']:+.4f} (vs {cell['argmax_other']}) ref={cell['steering_reference']} "
                  f"verdict={cell['verdict']} answer_auroc={cell.get('answer_auroc', float('nan')):.4f} capable={cell.get('answer_capable')} "
                  f"readable={cell['readable']} [{cell['readable_status']}] | test: O_q={t['O_q']:+.4f} verdict={t['verdict']} owned={t['owned']}")
        ag, nc = r["comparison"]["n_agree"], r["comparison"]["n_compared"]
        print("VALID agreement test vs valid: " + ", ".join(f"{k} {ag[k]}/{nc[k]}" for k in ag) + f"; valid features {'present' if r['features_available'] else 'absent'}")
    if "towerswap" in report and "towerswap" in a.what:
        r = report["towerswap"]
        for name, c in r["combinations"].items():
            if c.get("status") != "COMPLETE":
                print(f"TOWERSWAP {name:38s} {c.get('status')}"); continue
            print(f"TOWERSWAP {name:38s} owned {len(c['owned'])}/6 ({', '.join(c['owned']) or '-'}) "
                  f"directions from {c['directions_from']}")
        if "swapped_minus_native" in r:
            sm = r["swapped_minus_native"]
            for q, cell in sm.items():
                if not isinstance(cell, dict) or "estimate" not in cell:
                    continue
                lo, hi = cell["ci95_percentile"]
                print(f"TOWERSWAP dW {q:14s} {cell['estimate']:+.4f} [{lo:+.4f},{hi:+.4f}] swapped={cell['W_qq_swapped']:+.4f} "
                      f"native={cell['W_qq_native']:+.4f} verdict {cell['verdict_native']} -> {cell['verdict_swapped']} "
                      f"owned {cell['owned_native']} -> {cell['owned_swapped']}")
        if r.get("crossover_complete"):
            x = r["crossover"]
            print(f"TOWERSWAP crossover: mean |reader effect| = {x['mean_abs_reader_effect']:.4f}, "
                  f"mean |tower effect| = {x['mean_abs_tower_effect']:.4f}, ratio = {x['reader_over_tower']}")
            print(f"TOWERSWAP ownership follows: {x['ownership_follows']}")
    if "replay" in report and "replay" in a.what:
        r = report["replay"]
        d = r["drift"]
        print(f"REPLAY reader={r['reader']} source={r['source_block']} group={r['group']}; drift max|.|="
              f"{d.get('max_abs')} mean|.|={d.get('mean_abs')} against block mean |h|={d.get('block_mean_abs')}")
        for q in r["replay"]["grade"]:
            cell, cc = r["replay"]["grade"][q], r["core"]["grade"][q]
            dd = r["replay_minus_core"][q]
            lo, hi = dd["ci95_percentile"]
            print(f"REPLAY {q:14s} W_qq={cell['W_qq']:+.4f} (CORE {cc['W_qq']:+.4f}) d={dd['estimate']:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] owned {cc['owned']} -> {cell['owned']} verdict={cell['verdict']}")
        for other, blk in r["group_comparison"].items():
            if blk.get("status") != "COMPLETE":
                print(f"REPLAY vs {other}: {blk.get('status')}"); continue
            print(f"REPLAY vs {other}: mean |dW_qq| exact={blk.get('mean_abs_exact')} own towers={blk.get('mean_abs_own_towers')}")
    if "semend" in report and "semend" in a.what:
        r = report["semend"]
        for t in r["endpoints"]:
            blk = r["per_endpoint"][t]
            if blk.get("status") == "NOT_STARTED":
                print(f"SEMEND {t}: NOT_STARTED"); continue
            print(f"SEMEND {t} ({blk['value']}, sign {blk['sign']:+.0f}, {blk['n_conditions']} conditions):")
            for fam, fb in blk["families"].items():
                print(f"  SEMEND {t} {fam:6s}: reference met {len(fb['reference_met'])}/6, owned {len(fb['owned'])}/6 "
                      f"({', '.join(fb['owned']) or '-'})")
                for q, cell in fb["per_question"].items():
                    lo, hi = cell["O_q_ci95_percentile"]
                    print(f"    {q:14s} E={cell['W_qq']:+.4f} maxother={cell['max_other']:+.4f} sham={cell['E_sham']:+.4f} "
                          f"randmax={cell['random_max']} rank={cell['rank_in_random_family']} O={cell['O_q']:+.4f} "
                          f"[{lo:+.4f},{hi:+.4f}] intended={cell['intended']:+.4f} unintended={cell['unintended_mean']:+.4f} "
                          f"sel={cell['selectivity']:+.4f} ref={cell['steering_reference']} verdict={cell['verdict']}")
        if "negation" in r:
            ng = r["negation"]
            print(f"SEMEND negation: opposite sign {ng['n_opposite_sign']}/6, corr(affirmative, raw negated) = "
                  f"{ng['correlation_affirmative_vs_raw_negated']}")
        if "forced_choice" in r:
            print(f"SEMEND forced choice: both orders owned {r['forced_choice']['n_both_owned']}/6, mean |order gap| = "
                  f"{r['forced_choice']['mean_abs_order_gap']:.4f}")
        for fam, iv in r["incremental_validity"].items():
            if not iv.get("available"):
                print(f"SEMEND incremental validity [{fam}]: NOT AVAILABLE ({iv.get('reason')})"); continue
            add = iv["ownership_adds"]
            print(f"SEMEND incremental validity [{fam}]: base R2 {iv['base_model']['r2']:+.3f} -> full R2 "
                  f"{iv['full_model']['r2']:+.3f}, dR2 {add['delta_r2']:+.4f} {add['delta_r2_ci95']}; "
                  f"coef(owned) {add['coef_owned_core']:+.4f} {add['coef_owned_core_ci95']} "
                  f"excludes 0: {add['owned_core_interval_excludes_zero']}; "
                  f"coef(O_q) {add['coef_O_q_core']:+.4f} {add['coef_O_q_core_ci95']} "
                  f"excludes 0: {add['O_q_core_interval_excludes_zero']}")
            for g, gv in iv["groups"].items():
                print(f"  group {g:26s} cells={gv['n_cells']:2d} mean selectivity={gv['mean_target']} "
                      f"(intended {gv['mean_intended']}, unintended {gv['mean_unintended']})")
            d = iv["owned_minus_reference_met_not_owned"]
            print(f"  owned minus reference-met-not-owned: {d.get('estimate')} {d.get('ci95_percentile')} "
                  f"excludes 0: {d.get('interval_excludes_zero', d.get('reason'))}")
        print(f"SEMEND summary: {r['summary']}")
    if "core" in report and "core" in a.what:
        for q, cell in report["core"]["per_question"].items():
            print(f"CORE {q:14s} W_qq={cell['W_qq']:.4f} O_q={cell['O_q']:.4f} (vs {cell['argmax_other']}) p95rand={cell['random_p95']:.4f} "
                  f"|sham|={cell['abs_sham']:.4f} rank={cell['rank_in_random_family']} ref={cell['steering_reference']} verdict={cell.get('verdict')}")
