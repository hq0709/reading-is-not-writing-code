#!/usr/bin/env python
"""Direction / label geometry versus off-diagonal dominance in the CORE write matrix.

Reviewer alternative under test: on chest radiographs a competing concept's direction moves the target answer more
than the target's own direction (low ownership O_q) merely because the six clinical directions are correlated
(correlated report labels / shared components), not because clinical concepts are hard to write.

For every packaged block <RUN_ROOT>/<model_key>/<dataset>/ whose summary.json carries a complete 6x6 core.W and whose
fits/<locus>/seed<k>.npz exists, this script computes
  1. direction geometry: pairwise cosine between the six concept directions in model space (clinical_vectors, i.e.
     v_c = normalize(P @ (w_c / max(s, 1e-8)))) and in the 512-d projected space (probe coefficients w_c in the
     standardised projected space, and u_c = w_c / s in the un-standardised projected space); per-dataset mean/median
     off-diagonal cosine; whether the strongest competitor argmax_other(q) is the direction with the highest cosine
     to v_q;
  2. label geometry: pairwise phi coefficient and co-occurrence counts between concept labels over the test rows
     (both-known rows only); whether the strongest competitor is the most co-occurring label;
  3. association: Spearman rho between W_qd and cos(v_q, v_d), and between W_qd and phi(q, d), over all off-diagonal
     cells (pooled per dataset, pooled over chest = nih + chexpert, and within block); the same for the competitor
     advantage W_qd - W_qq;
  4. leave-one-competitor-out ownership O^{-d*} = W_qq - max over the four remaining competitors after removing the
     strongest, counts of cells with O^{-d*} > 0 vs O_q > 0, and per cell the number of competitors whose W_qd
     exceeds W_qq;
  5. COCO competitor families restricted to co-occurring objects: O_q restricted to competitors inside a family;
  6. the seed cell (default q25-7 / nih / Effusion): cosine and phi to its strongest competitor and that competitor's
     rank among the five competitors by cosine and by phi.

Reads only summary.json, fits/<locus>/seed<k>.npz, manifests/labels.csv and manifests/cohort.csv. Writes
<RUN_ROOT>/robustness/geometry.json and geometry.md. Nothing under the block directories is modified.

Usage (from the concept-flow repo):
  PYTHONPATH=src python scripts/mayo/robustness_geometry.py [--run-root ...] [--out-dir ...] [--locus vis.last]
                                                            [--fit-seed 0] [--seed-cell q25-7:nih:Effusion]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from cftransfer.manifest import block_included  # noqa: E402
from cftransfer.protocol import CONCEPTS, DATASETS  # noqa: E402
from cftransfer.runpaths import RUN_ROOT  # noqa: E402

CHEST = ("nih", "chexpert")
SKIP_DIRS = {"robustness", "figures"}
# COCO competitor families that visually / semantically co-occur (README figure discussion)
COCO_FAMILIES = {
    "person+bicycle": ("person", "bicycle"),
    "person+dog": ("person", "dog"),
    "chair+bottle": ("chair", "bottle"),
    "car+bicycle": ("car", "bicycle"),
    "person+chair+bottle": ("person", "chair", "bottle"),
}


# ------------------------------------------------------------------------------------------------ helpers
def clean(o):
    """JSON-safe: numpy scalars -> python, NaN -> None, arrays -> lists."""
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


def cosine_matrix(V: np.ndarray) -> np.ndarray:
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return Vn @ Vn.T


def offdiag(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    return M[~np.eye(n, dtype=bool)]


def upper(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    return M[np.triu_indices(n, 1)]


def spearman(x, y) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return {"rho": np.nan, "p": np.nan, "n": int(m.sum())}
    r = spearmanr(x[m], y[m])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(m.sum())}


def phi_and_counts(Y: np.ndarray):
    """Y (n, k) with values 0/1 or NaN (unknown). Returns phi (k,k), n11 (k,k), n_both_known (k,k), 2x2 tables."""
    k = Y.shape[1]
    phi = np.full((k, k), np.nan); n11 = np.zeros((k, k), int); nk = np.zeros((k, k), int)
    tables = {}
    for i in range(k):
        for j in range(k):
            m = np.isfinite(Y[:, i]) & np.isfinite(Y[:, j])
            a, b = Y[m, i], Y[m, j]
            t11 = int(((a == 1) & (b == 1)).sum()); t10 = int(((a == 1) & (b == 0)).sum())
            t01 = int(((a == 0) & (b == 1)).sum()); t00 = int(((a == 0) & (b == 0)).sum())
            n11[i, j] = t11; nk[i, j] = int(m.sum())
            den = (t11 + t10) * (t01 + t00) * (t11 + t01) * (t10 + t00)
            phi[i, j] = (t11 * t00 - t10 * t01) / np.sqrt(den) if den > 0 else np.nan
            tables[(i, j)] = (t11, t10, t01, t00)
    return phi, n11, nk, tables


# ------------------------------------------------------------------------------------------- data loading
def discover_blocks(run_root: Path, locus: str, fit_seed: int):
    """Blocks under the paper's one inclusion rule (cftransfer.manifest.block_included: CORE and CALIBRATION in
    run.json completed_modules) with a complete 6x6 core.W and a seed-<fit_seed> fit at the requested locus."""
    used, skipped = [], []
    for sj in sorted(run_root.glob("*/*/summary.json")):
        mk, ds = sj.parts[-3], sj.parts[-2]
        if mk in SKIP_DIRS or ds not in DATASETS:
            continue
        rj = sj.parent / "run.json"
        if not (rj.exists() and block_included(json.loads(rj.read_text()))):
            skipped.append({"block": f"{mk}/{ds}", "reason": "not included: run.json completed_modules lacks CORE or CALIBRATION"}); continue
        s = json.loads(sj.read_text())
        core = s.get("core") or {}
        W = core.get("W") or {}
        cs = CONCEPTS[ds]
        complete = all(f"concept:{d}" in W.get(q, {}) for q in cs for d in cs)
        npz = sj.parent / "fits" / locus / f"seed{fit_seed}.npz"
        man = sj.parent / "manifests"
        if not complete:
            skipped.append({"block": f"{mk}/{ds}", "reason": "core.W incomplete or missing"}); continue
        if not npz.exists():
            skipped.append({"block": f"{mk}/{ds}", "reason": f"missing {npz.relative_to(sj.parent)}"}); continue
        if not (man / "labels.csv").exists() or not (man / "cohort.csv").exists():
            skipped.append({"block": f"{mk}/{ds}", "reason": "missing manifests"}); continue
        used.append((mk, ds, sj.parent, s))
    return used, skipped


def load_test_labels(block_dir: Path, ds: str):
    """(row_ids in frozen test order, Y (n,6) float with NaN for unknown, md5 of the table)."""
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
    digest = hashlib.md5(pd.DataFrame(Y, index=rows, columns=cs).to_csv().encode()).hexdigest()
    return rows, Y, digest


def load_directions(block_dir: Path, ds: str, locus: str, fit_seed: int) -> dict:
    f = np.load(block_dir / "fits" / locus / f"seed{fit_seed}.npz", allow_pickle=False)
    names = [str(c) for c in f["concept_names"]]
    assert names == CONCEPTS[ds], (block_dir, names)
    V = f["clinical_vectors"].astype(np.float64)                       # (6, D) unit vectors in model space
    P = f["projection"].astype(np.float64)                             # (D, 512)
    w = f["coefficients"].astype(np.float64)                           # (6, 512) probe weights (standardised space)
    s = f["scaler_scale"].astype(np.float64)                           # (512,) train-only scale
    u = w / np.maximum(s, 1e-8)                                        # (6, 512) pre-projection direction
    raw = u @ P.T                                                      # (6, D)
    rec = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    recon = float(np.abs(rec - V).max())
    out = {"D": int(V.shape[1]), "recon_maxabs": recon,
           "cos_model": cosine_matrix(V), "cos_proj_w": cosine_matrix(w), "cos_proj_u": cosine_matrix(u)}
    if "random_vectors" in f.files:
        R = f["random_vectors"].astype(np.float64)
        CR = cosine_matrix(R)
        out["random_pair_abs_cos_mean"] = float(np.abs(upper(CR)).mean())
        out["concept_random_abs_cos_mean"] = float(np.abs(V @ R.T).mean())
    return out


def block_W(summary: dict, ds: str) -> np.ndarray:
    cs = CONCEPTS[ds]
    W = summary["core"]["W"]
    return np.array([[W[q][f"concept:{d}"] for d in cs] for q in cs], float)


# ------------------------------------------------------------------------------------------- per block
def analyse_block(mk, ds, summary, geo, phi, n11) -> dict:
    cs = CONCEPTS[ds]; k = len(cs)
    W = block_W(summary, ds)
    pq = summary["core"]["per_question"]
    per_q, cells = {}, []
    for i, q in enumerate(cs):
        others = [j for j in range(k) if j != i]
        wrow = W[i]
        arg_w = others[int(np.argmax(wrow[others]))]
        am_summary = pq[q].get("argmax_other")
        assert am_summary == cs[arg_w], (mk, ds, q, am_summary, cs[arg_w])
        sorted_comp = sorted(others, key=lambda j: -wrow[j])
        O_q = float(wrow[i] - wrow[arg_w])
        O_loo = float(wrow[i] - wrow[sorted_comp[1]])            # strongest competitor removed
        n_above = int(sum(wrow[j] > wrow[i] for j in others))

        def argmax_by(M):
            vals = M[i, others]
            if not np.isfinite(vals).any():
                return None
            vals = np.where(np.isfinite(vals), vals, -np.inf)
            return others[int(np.argmax(vals))]

        def rank_of(M, j):
            vals = M[i, others]
            if not np.isfinite(M[i, j]):
                return None
            return int(1 + sum(1 for v in vals if np.isfinite(v) and v > M[i, j]))

        am_cos = argmax_by(geo["cos_model"]); am_cw = argmax_by(geo["cos_proj_w"]); am_cu = argmax_by(geo["cos_proj_u"])
        am_phi = argmax_by(phi); am_n11 = argmax_by(n11.astype(float))
        per_q[q] = {
            "W_qq": float(wrow[i]), "O_q": O_q, "O_q_summary": pq[q].get("O_q"), "verdict": pq[q].get("verdict"),
            "steering_reference": pq[q].get("steering_reference"),
            "argmax_other": cs[arg_w], "second_competitor": cs[sorted_comp[1]], "O_loo": O_loo,
            "n_competitors_above_own": n_above,
            "argmax_cos_model": cs[am_cos] if am_cos is not None else None,
            "argmax_cos_proj_w": cs[am_cw] if am_cw is not None else None,
            "argmax_cos_proj_u": cs[am_cu] if am_cu is not None else None,
            "argmax_phi": cs[am_phi] if am_phi is not None else None,
            "argmax_cooccurrence": cs[am_n11] if am_n11 is not None else None,
            "coincide_cos_model": (am_cos == arg_w) if am_cos is not None else None,
            "coincide_cos_proj_w": (am_cw == arg_w) if am_cw is not None else None,
            "coincide_cos_proj_u": (am_cu == arg_w) if am_cu is not None else None,
            "coincide_phi": (am_phi == arg_w) if am_phi is not None else None,
            "coincide_cooccurrence": (am_n11 == arg_w) if am_n11 is not None else None,
            "cos_model_to_argmax_other": float(geo["cos_model"][i, arg_w]),
            "phi_to_argmax_other": float(phi[i, arg_w]),
            "rank_of_argmax_other_by_cos_model": rank_of(geo["cos_model"], arg_w),
            "rank_of_argmax_other_by_phi": rank_of(phi, arg_w),
        }
        for j in others:
            cells.append({"block": f"{mk}/{ds}", "model_key": mk, "dataset": ds, "q": q, "d": cs[j],
                          "W_qd": float(wrow[j]), "W_qq": float(wrow[i]), "adv": float(wrow[j] - wrow[i]),
                          "cos_model": float(geo["cos_model"][i, j]), "cos_proj_w": float(geo["cos_proj_w"][i, j]),
                          "cos_proj_u": float(geo["cos_proj_u"][i, j]), "phi": float(phi[i, j]), "n11": int(n11[i, j]),
                          "is_argmax_other": bool(j == arg_w)})
    return {"W": W, "per_question": per_q, "cells": cells,
            "offdiag_cos_model_mean": float(offdiag(geo["cos_model"]).mean()),
            "offdiag_cos_proj_w_mean": float(offdiag(geo["cos_proj_w"]).mean()),
            "offdiag_cos_proj_u_mean": float(offdiag(geo["cos_proj_u"]).mean()),
            "offdiag_abs_cos_model_mean": float(np.abs(offdiag(geo["cos_model"])).mean())}


# ------------------------------------------------------------------------------------------- aggregates
def frac(flags):
    flags = [f for f in flags if f is not None]
    return {"fraction": float(np.mean(flags)) if flags else np.nan, "n_true": int(sum(flags)), "n": len(flags)}


def association(cells: list[dict]) -> dict:
    df = pd.DataFrame(cells)
    out = {"n_cells": int(len(df)), "n_blocks": int(df.block.nunique())}
    for xname, yname, key in [("W_qd", "cos_model", "rho_W_vs_cos_model"), ("W_qd", "cos_proj_w", "rho_W_vs_cos_proj_w"),
                              ("W_qd", "cos_proj_u", "rho_W_vs_cos_proj_u"), ("W_qd", "phi", "rho_W_vs_phi"),
                              ("W_qd", "n11", "rho_W_vs_cooccurrence"),
                              ("adv", "cos_model", "rho_adv_vs_cos_model"), ("adv", "cos_proj_w", "rho_adv_vs_cos_proj_w"),
                              ("adv", "phi", "rho_adv_vs_phi")]:
        out[key] = spearman(df[xname], df[yname])
        # within-block: one rho per block over its 30 off-diagonal cells, then summarised across blocks
        per = [spearman(g[xname], g[yname])["rho"] for _b, g in df.groupby("block")]
        per = [r for r in per if np.isfinite(r)]
        out[key + "_within_block"] = {"mean": float(np.mean(per)) if per else np.nan,
                                      "median": float(np.median(per)) if per else np.nan,
                                      "fraction_positive": float(np.mean([r > 0 for r in per])) if per else np.nan,
                                      "n_blocks": len(per)}
    # cells where the competitor beats the own direction: how similar are those directions, versus the rest
    dom = df[df.adv > 0]; rest = df[df.adv <= 0]
    out["dominating_cells"] = {"n": int(len(dom)), "cos_model_mean": float(dom.cos_model.mean()) if len(dom) else np.nan,
                               "phi_mean": float(dom.phi.mean()) if len(dom) else np.nan}
    out["non_dominating_cells"] = {"n": int(len(rest)), "cos_model_mean": float(rest.cos_model.mean()) if len(rest) else np.nan,
                                   "phi_mean": float(rest.phi.mean()) if len(rest) else np.nan}
    return out


def leave_one_out(per_q_rows: list[dict]) -> dict:
    n = len(per_q_rows)
    na = np.array([r["n_competitors_above_own"] for r in per_q_rows], int)
    return {"n_cells": n, "n_blocks": len({r["block"] for r in per_q_rows}),
            "O_q_positive": int(sum(r["O_q"] > 0 for r in per_q_rows)),
            "O_loo_positive": int(sum(r["O_loo"] > 0 for r in per_q_rows)),
            "O_q_positive_fraction": float(np.mean([r["O_q"] > 0 for r in per_q_rows])) if n else np.nan,
            "O_loo_positive_fraction": float(np.mean([r["O_loo"] > 0 for r in per_q_rows])) if n else np.nan,
            "n_above_distribution": {str(k): int((na == k).sum()) for k in range(6)},
            "n_above_ge1": int((na >= 1).sum()), "n_above_ge2": int((na >= 2).sum()), "n_above_ge3": int((na >= 3).sum()),
            "n_above_mean": float(na.mean()) if n else np.nan}


def coco_families(blocks: dict, phi_coco: np.ndarray) -> dict:
    cs = CONCEPTS["coco"]; ix = {c: i for i, c in enumerate(cs)}
    out = {}
    for fam, members in COCO_FAMILIES.items():
        rows = []
        for key, b in blocks.items():
            if b["dataset"] != "coco":
                continue
            W = np.array(b["W"])
            for q in members:
                i = ix[q]
                comp = [ix[d] for d in members if d != q]
                allc = [j for j in range(len(cs)) if j != i]
                rows.append({"block": key, "q": q, "O_family": float(W[i, i] - W[i, comp].max()),
                             "O_full": float(W[i, i] - W[i, allc].max()),
                             "family_argmax": cs[comp[int(np.argmax(W[i, comp]))]]})
        pairs = [(ix[a], ix[b]) for a in members for b in members if a < b]
        cos_vals = [float(np.array(blocks[k]["cos_model"])[i, j]) for k in blocks if blocks[k]["dataset"] == "coco" for i, j in pairs]
        out[fam] = {"members": list(members), "n_cells": len(rows),
                    "O_family_positive": int(sum(r["O_family"] > 0 for r in rows)),
                    "O_family_positive_fraction": float(np.mean([r["O_family"] > 0 for r in rows])) if rows else np.nan,
                    "O_full_positive": int(sum(r["O_full"] > 0 for r in rows)),
                    "O_full_positive_fraction": float(np.mean([r["O_full"] > 0 for r in rows])) if rows else np.nan,
                    "O_family_median": float(np.median([r["O_family"] for r in rows])) if rows else np.nan,
                    "O_full_median": float(np.median([r["O_full"] for r in rows])) if rows else np.nan,
                    "phi_within_family": {f"{cs[i]}-{cs[j]}": float(phi_coco[i, j]) for i, j in pairs},
                    "phi_within_family_mean": float(np.mean([phi_coco[i, j] for i, j in pairs])),
                    "cos_model_within_family_mean": float(np.mean(cos_vals)) if cos_vals else np.nan,
                    "cells": rows}
    return out


# ------------------------------------------------------------------------------------------- markdown
def f3(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def matrix_table(names, M, nd=3, diag_blank=True):
    head = "| | " + " | ".join(names) + " |\n|" + "---|" * (len(names) + 1) + "\n"
    body = ""
    for i, a in enumerate(names):
        cells = []
        for j in range(len(names)):
            if diag_blank and i == j:
                cells.append("-")
            else:
                v = M[i][j]
                cells.append("n/a" if v is None else (f3(v, nd) if isinstance(v, float) else str(v)))
        body += f"| {a} | " + " | ".join(cells) + " |\n"
    return head + body


def write_markdown(R: dict, path: Path, locus: str, fit_seed: int):
    L = []
    L.append("# Direction / label geometry versus off-diagonal dominance\n")
    L.append(f"Generated {R['meta']['generated_utc']} by `scripts/mayo/robustness_geometry.py` from `{R['meta']['run_root']}` "
             f"(locus `{locus}`, fit seed {fit_seed}, alpha {R['meta']['alpha']}). Blocks used: "
             + ", ".join(f"{ds} {n}" for ds, n in R["meta"]["n_blocks_by_dataset"].items())
             + f" (total {R['meta']['n_blocks']}). Skipped: "
             + ("; ".join(f"{s['block']} ({s['reason']})" for s in R["meta"]["skipped"]) or "none") + ".\n")
    L.append("Definitions. W_qd = mean over the 600 test rows of p(present | direction d, alpha) - p(present | baseline) for "
             "question q (core.W). O_q = W_qq - max_{d != q} W_qd. Model-space direction v_c = normalize(P @ (w_c / max(s, 1e-8))) "
             "(fits/<locus>/seed0.npz clinical_vectors; reconstruction error <= "
             f"{f3(R['meta']['max_recon_error'], 9)}). cos_model = cos(v_q, v_d); cos_proj_w = cos(w_q, w_d) in the standardised 512-d "
             "projected space; cos_proj_u = cos(w_q/s, w_d/s). phi = phi coefficient of the two test-label columns over both-known rows; "
             "n11 = both-positive count. Chance level for an argmax coincidence is 1/5 = 0.200. Test-label tables are identical across "
             f"blocks within a dataset: {R['meta']['labels_identical_across_blocks']}.\n")

    # 1 direction geometry
    L.append("## 1. Direction geometry\n")
    L.append("| dataset | blocks | mean off-diag cos_model | median | mean |cos_model| | mean off-diag cos_proj_w | median | random-pair |cos| ref | concept-random |cos| ref | argmax_other = argmax cos_model | = argmax cos_proj_w | = argmax cos_proj_u |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, g in R["direction_geometry"].items():
        L.append(f"| {ds} | {g['n_blocks']} | {f3(g['offdiag_cos_model_mean'])} | {f3(g['offdiag_cos_model_median'])} | "
                 f"{f3(g['offdiag_abs_cos_model_mean'])} | {f3(g['offdiag_cos_proj_w_mean'])} | {f3(g['offdiag_cos_proj_w_median'])} | "
                 f"{f3(g['random_pair_abs_cos_mean'])} | {f3(g['concept_random_abs_cos_mean'])} | "
                 f"{f3(g['coincide_cos_model']['fraction'])} ({g['coincide_cos_model']['n_true']}/{g['coincide_cos_model']['n']}) | "
                 f"{f3(g['coincide_cos_proj_w']['fraction'])} ({g['coincide_cos_proj_w']['n_true']}/{g['coincide_cos_proj_w']['n']}) | "
                 f"{f3(g['coincide_cos_proj_u']['fraction'])} ({g['coincide_cos_proj_u']['n_true']}/{g['coincide_cos_proj_u']['n']}) |")
    L.append("")
    for ds, g in R["direction_geometry"].items():
        L.append(f"Mean cos_model over {ds} blocks:\n")
        L.append(matrix_table(CONCEPTS[ds], g["cos_model_mean_matrix"]))
    L.append("Rank of the strongest competitor argmax_other(q) among the five competitors by cos_model (1 = most similar direction; uniform = 20% each):\n")
    L.append("| dataset | rank 1 | 2 | 3 | 4 | 5 | n |")
    L.append("|---|---|---|---|---|---|---|")
    for ds, g in R["direction_geometry"].items():
        d = g["rank_of_argmax_other_by_cos_model_distribution"]
        L.append(f"| {ds} | " + " | ".join(str(d[str(r)]) for r in range(1, 6)) + f" | {sum(d.values())} |")
    L.append("")
    L.append("Per-block mean off-diagonal cos_model:\n")
    L.append("| block | D | mean cos_model | mean cos_proj_w | argmax coincidences (cos_model) |")
    L.append("|---|---|---|---|---|")
    for key, b in R["blocks"].items():
        c = sum(1 for q in b["per_question"].values() if q["coincide_cos_model"])
        L.append(f"| {key} | {b['D']} | {f3(b['offdiag_cos_model_mean'])} | {f3(b['offdiag_cos_proj_w_mean'])} | {c}/6 |")
    L.append("")

    # 2 label geometry
    L.append("## 2. Label geometry (test rows)\n")
    L.append("| dataset | n test | mean off-diag phi | median | max |phi| pair | argmax_other = argmax phi | = argmax n11 |")
    L.append("|---|---|---|---|---|---|---|")
    for ds, g in R["label_geometry"].items():
        L.append(f"| {ds} | {R['labels'][ds]['n_test']} | {f3(g['offdiag_phi_mean'])} | {f3(g['offdiag_phi_median'])} | "
                 f"{g['max_abs_phi_pair']} ({f3(g['max_abs_phi'])}) | "
                 f"{f3(g['coincide_phi']['fraction'])} ({g['coincide_phi']['n_true']}/{g['coincide_phi']['n']}) | "
                 f"{f3(g['coincide_cooccurrence']['fraction'])} ({g['coincide_cooccurrence']['n_true']}/{g['coincide_cooccurrence']['n']}) |")
    L.append("")
    L.append("Rank of argmax_other(q) among the five competitors by phi (1 = most co-occurring label; undefined phi ranks last):\n")
    L.append("| dataset | rank 1 | 2 | 3 | 4 | 5 | n |")
    L.append("|---|---|---|---|---|---|---|")
    for ds, g in R["label_geometry"].items():
        d = g["rank_of_argmax_other_by_phi_distribution"]
        L.append(f"| {ds} | " + " | ".join(str(d[str(r)]) for r in range(1, 6)) + f" | {sum(d.values())} |")
    L.append("")
    L.append("phi is undefined (n/a) when a 2x2 marginal over the both-known rows is zero; on CheXpert the known-label subset is the "
             "rows whose report mentions the finding, so both-known counts are small and, for Atelectasis, almost all known rows are positive.\n")
    for ds, lab in R["labels"].items():
        cs = CONCEPTS[ds]
        L.append(f"{ds}: positives per concept " + ", ".join(f"{c} {n}" for c, n in zip(cs, lab["n_pos"]))
                 + "; known per concept " + ", ".join(f"{c} {n}" for c, n in zip(cs, lab["n_known"])) + ".\n")
        L.append("phi (both-known rows):\n")
        L.append(matrix_table(cs, lab["phi"]))
        L.append("co-occurrence n11 (both positive) with n both-known in parentheses:\n")
        M = [[f"{lab['cooccurrence'][i][j]} ({lab['n_both_known'][i][j]})" for j in range(6)] for i in range(6)]
        L.append(matrix_table(cs, M))

    # 3 association
    L.append("## 3. Association between W_qd and direction / label similarity (off-diagonal cells)\n")
    L.append("Pooled Spearman rho over all (block, q, d != q) cells; within-block = rho computed inside each block over its 30 cells, then summarised across blocks.\n")
    L.append("| cells | n cells | n blocks | rho(W_qd, cos_model) | rho(W_qd, cos_proj_w) | rho(W_qd, phi) | rho(W_qd, n11) | rho(W_qd - W_qq, cos_model) | rho(W_qd - W_qq, phi) | within-block rho(W_qd, cos_model) median [frac > 0] | within-block rho(W_qd, phi) median [frac > 0] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, a in R["association"].items():
        def cell(k):
            return f"{f3(a[k]['rho'])} (p={a[k]['p']:.2g})" if a[k]["p"] is not None and np.isfinite(a[k]["p"]) else "n/a"
        L.append(f"| {name} | {a['n_cells']} | {a['n_blocks']} | {cell('rho_W_vs_cos_model')} | {cell('rho_W_vs_cos_proj_w')} | "
                 f"{cell('rho_W_vs_phi')} | {cell('rho_W_vs_cooccurrence')} | {cell('rho_adv_vs_cos_model')} | {cell('rho_adv_vs_phi')} | "
                 f"{f3(a['rho_W_vs_cos_model_within_block']['median'])} [{f3(a['rho_W_vs_cos_model_within_block']['fraction_positive'], 2)}] | "
                 f"{f3(a['rho_W_vs_phi_within_block']['median'])} [{f3(a['rho_W_vs_phi_within_block']['fraction_positive'], 2)}] |")
    L.append("")
    L.append("Cells where the competitor beats the own direction (W_qd > W_qq) versus the rest:\n")
    L.append("| cells | dominating n | mean cos_model | mean phi | non-dominating n | mean cos_model | mean phi |")
    L.append("|---|---|---|---|---|---|---|")
    for name, a in R["association"].items():
        d, r = a["dominating_cells"], a["non_dominating_cells"]
        L.append(f"| {name} | {d['n']} | {f3(d['cos_model_mean'])} | {f3(d['phi_mean'])} | {r['n']} | {f3(r['cos_model_mean'])} | {f3(r['phi_mean'])} |")
    L.append("")

    # 4 leave one out
    L.append("## 4. Leave-one-competitor-out ownership and competitor counts\n")
    L.append("O^{-d*} = W_qq - max over the four competitors left after removing the strongest one. n_above = number of competitors with W_qd > W_qq.\n")
    L.append("| cells | n (block, q) | O_q > 0 | O^{-d*} > 0 | n_above = 0 | 1 | 2 | 3 | 4 | 5 | n_above >= 2 | >= 3 | mean n_above |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, g in R["leave_one_out"].items():
        d = g["n_above_distribution"]
        L.append(f"| {name} | {g['n_cells']} | {g['O_q_positive']} ({f3(g['O_q_positive_fraction'])}) | {g['O_loo_positive']} ({f3(g['O_loo_positive_fraction'])}) | "
                 f"{d['0']} | {d['1']} | {d['2']} | {d['3']} | {d['4']} | {d['5']} | {g['n_above_ge2']} | {g['n_above_ge3']} | {f3(g['n_above_mean'], 2)} |")
    L.append("")

    # 5 coco families
    L.append("## 5. COCO competitor families (co-occurring objects)\n")
    L.append("O restricted to competitors inside the family, for every COCO block and every q in the family; O_full = the same q cells with all five competitors. Chest reference rows use the full family.\n")
    L.append("| family | n (block, q) | O_family > 0 | O_full > 0 (same cells) | median O_family | median O_full | mean phi within family | mean cos_model within family |")
    L.append("|---|---|---|---|---|---|---|---|")
    for fam, g in R["coco_families"].items():
        L.append(f"| {fam} | {g['n_cells']} | {g['O_family_positive']} ({f3(g['O_family_positive_fraction'])}) | "
                 f"{g['O_full_positive']} ({f3(g['O_full_positive_fraction'])}) | {f3(g['O_family_median'])} | {f3(g['O_full_median'])} | "
                 f"{f3(g['phi_within_family_mean'])} | {f3(g['cos_model_within_family_mean'])} |")
    for name in ("nih", "chexpert", "chest", "coco"):
        g = R["leave_one_out"][name]
        phi_m = f3(R["label_geometry"][name]["offdiag_phi_mean"]) if name in R["label_geometry"] else "n/a"
        cos_m = f3(R["direction_geometry"][name]["offdiag_cos_model_mean"]) if name in R["direction_geometry"] else "n/a"
        L.append(f"| {name} full family (reference) | {g['n_cells']} | - | {g['O_q_positive']} ({f3(g['O_q_positive_fraction'])}) | - | "
                 f"{f3(R['ownership_median'][name])} | {phi_m} | {cos_m} |")
    L.append("")

    # 6 seed cell
    sc = R["seed_cell"]
    L.append(f"## 6. Seed cell {sc['block']} / {sc['q']}\n")
    L.append(f"Strongest competitor (argmax_other): {sc['argmax_other']}. W_qq = {f3(sc['W_qq'], 4)}, W_q,{sc['argmax_other']} = {f3(sc['W_q_argmax'], 4)}, O_q = {f3(sc['O_q'], 4)}, O^(-d*) = {f3(sc['O_loo'], 4)}, verdict {sc['verdict']}.\n")
    L.append(f"cos_model({sc['q']}, {sc['argmax_other']}) = {f3(sc['cos_model'], 4)}; cos_proj_w = {f3(sc['cos_proj_w'], 4)}; cos_proj_u = {f3(sc['cos_proj_u'], 4)}; "
             f"phi = {f3(sc['phi'], 4)}; n11 = {sc['n11']} (2x2 n11/n10/n01/n00 = {sc['table_2x2']}).\n")
    L.append(f"Rank of {sc['argmax_other']} among the five competitors of {sc['q']}: by cos_model {sc['rank_by_cos_model']}/5, by cos_proj_w {sc['rank_by_cos_proj_w']}/5, by phi {sc['rank_by_phi']}/5, by n11 {sc['rank_by_cooccurrence']}/5 (1 = highest).\n")
    L.append("| competitor d | W_qd | cos_model | cos_proj_w | phi | n11 |")
    L.append("|---|---|---|---|---|---|")
    for r in sc["competitors"]:
        L.append(f"| {r['d']} | {f3(r['W_qd'], 4)} | {f3(r['cos_model'], 4)} | {f3(r['cos_proj_w'], 4)} | {f3(r['phi'], 4)} | {r['n11']} |")
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(RUN_ROOT))
    ap.add_argument("--out-dir", default=None, help="default <run-root>/robustness")
    ap.add_argument("--locus", default="vis.last")
    ap.add_argument("--fit-seed", type=int, default=0)
    ap.add_argument("--seed-cell", default="q25-7:nih:Effusion", help="model_key:dataset:question")
    a = ap.parse_args()
    run_root = Path(a.run_root)
    out_dir = Path(a.out_dir) if a.out_dir else run_root / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    used, skipped = discover_blocks(run_root, a.locus, a.fit_seed)
    if not used:
        raise SystemExit("no complete blocks found")

    # labels: one table per dataset, verified identical across blocks
    labels, digests = {}, {}
    for mk, ds, bdir, _s in used:
        rows, Y, dg = load_test_labels(bdir, ds)
        digests.setdefault(ds, set()).add(dg)
        if ds not in labels:
            phi, n11, nk, tables = phi_and_counts(Y)
            labels[ds] = {"concepts": CONCEPTS[ds], "n_test": len(rows), "source_block": f"{mk}/{ds}",
                          "n_pos": np.nansum(Y, axis=0).astype(int).tolist(), "n_known": np.isfinite(Y).sum(axis=0).tolist(),
                          "phi": phi, "cooccurrence": n11, "n_both_known": nk, "_tables": tables}
    labels_identical = all(len(v) == 1 for v in digests.values())

    blocks, alphas, recon = {}, set(), []
    for mk, ds, bdir, s in used:
        geo = load_directions(bdir, ds, a.locus, a.fit_seed)
        recon.append(geo["recon_maxabs"])
        alphas.add(s["core"].get("alpha"))
        res = analyse_block(mk, ds, s, geo, labels[ds]["phi"], labels[ds]["cooccurrence"])
        blocks[f"{mk}/{ds}"] = {"model_key": mk, "dataset": ds, "D": geo["D"], "recon_maxabs": geo["recon_maxabs"],
                                "n_rows": s["core"].get("n_rows"), "primary_template": s.get("primary_template"),
                                "cos_model": geo["cos_model"], "cos_proj_w": geo["cos_proj_w"], "cos_proj_u": geo["cos_proj_u"],
                                "random_pair_abs_cos_mean": geo.get("random_pair_abs_cos_mean"),
                                "concept_random_abs_cos_mean": geo.get("concept_random_abs_cos_mean"), **res}

    groups = {ds: [k for k, b in blocks.items() if b["dataset"] == ds] for ds in DATASETS}
    groups["chest"] = groups["nih"] + groups["chexpert"]

    # 1 direction geometry per dataset
    direction_geometry = {}
    for ds in DATASETS:
        keys = groups[ds]
        if not keys:
            continue
        cm = np.stack([blocks[k]["cos_model"] for k in keys]); cw = np.stack([blocks[k]["cos_proj_w"] for k in keys])
        cu = np.stack([blocks[k]["cos_proj_u"] for k in keys])
        od = np.concatenate([offdiag(m) for m in cm]); odw = np.concatenate([offdiag(m) for m in cw]); odu = np.concatenate([offdiag(m) for m in cu])
        pqs = [blocks[k]["per_question"][q] for k in keys for q in CONCEPTS[ds]]
        direction_geometry[ds] = {
            "n_blocks": len(keys),
            "offdiag_cos_model_mean": float(od.mean()), "offdiag_cos_model_median": float(np.median(od)),
            "offdiag_abs_cos_model_mean": float(np.abs(od).mean()),
            "offdiag_cos_proj_w_mean": float(odw.mean()), "offdiag_cos_proj_w_median": float(np.median(odw)),
            "offdiag_cos_proj_u_mean": float(odu.mean()), "offdiag_cos_proj_u_median": float(np.median(odu)),
            "per_block_offdiag_cos_model_mean": {k: blocks[k]["offdiag_cos_model_mean"] for k in keys},
            "cos_model_mean_matrix": cm.mean(axis=0), "cos_proj_w_mean_matrix": cw.mean(axis=0),
            "random_pair_abs_cos_mean": float(np.nanmean([blocks[k]["random_pair_abs_cos_mean"] for k in keys])),
            "concept_random_abs_cos_mean": float(np.nanmean([blocks[k]["concept_random_abs_cos_mean"] for k in keys])),
            "coincide_cos_model": frac([p["coincide_cos_model"] for p in pqs]),
            "coincide_cos_proj_w": frac([p["coincide_cos_proj_w"] for p in pqs]),
            "coincide_cos_proj_u": frac([p["coincide_cos_proj_u"] for p in pqs]),
            "rank_of_argmax_other_by_cos_model_distribution": {str(r): int(sum(p["rank_of_argmax_other_by_cos_model"] == r for p in pqs)) for r in range(1, 6)},
        }

    # 2 label geometry per dataset
    label_geometry = {}
    for ds in DATASETS:
        keys = groups[ds]
        if not keys:
            continue
        phi = labels[ds]["phi"]; od = upper(phi); od = od[np.isfinite(od)]
        cs = CONCEPTS[ds]
        iu = np.triu_indices(6, 1); absphi = np.abs(phi[iu]); j = int(np.nanargmax(absphi))
        pqs = [blocks[k]["per_question"][q] for k in keys for q in cs]
        label_geometry[ds] = {
            "offdiag_phi_mean": float(od.mean()), "offdiag_phi_median": float(np.median(od)),
            "max_abs_phi": float(absphi[j]), "max_abs_phi_pair": f"{cs[iu[0][j]]}-{cs[iu[1][j]]}",
            "coincide_phi": frac([p["coincide_phi"] for p in pqs]),
            "coincide_cooccurrence": frac([p["coincide_cooccurrence"] for p in pqs]),
            "rank_of_argmax_other_by_phi_distribution": {str(r): int(sum(p["rank_of_argmax_other_by_phi"] == r for p in pqs)) for r in range(1, 6)},
        }

    # 3 association, 4 leave-one-out, ownership medians
    assoc, loo, own_med = {}, {}, {}
    for name in ("chest", "nih", "chexpert", "coco"):
        keys = groups[name]
        if not keys:
            continue
        cells = [c for k in keys for c in blocks[k]["cells"]]
        assoc[name] = association(cells)
        pq_rows = [{"block": k, "q": q, **blocks[k]["per_question"][q]} for k in keys for q in CONCEPTS[blocks[k]["dataset"]]]
        loo[name] = leave_one_out(pq_rows)
        own_med[name] = float(np.median([r["O_q"] for r in pq_rows]))

    # 5 coco families
    fam = coco_families(blocks, labels["coco"]["phi"]) if groups["coco"] else {}

    # 6 seed cell
    mk, ds, q = a.seed_cell.split(":")
    key = f"{mk}/{ds}"
    if key not in blocks:
        raise SystemExit(f"seed cell block {key} not among complete blocks")
    b = blocks[key]; cs = CONCEPTS[ds]; i = cs.index(q); p = b["per_question"][q]
    dstar = p["argmax_other"]; j = cs.index(dstar)
    comps = [c for c in b["cells"] if c["q"] == q]
    seed_cell = {"block": key, "q": q, "argmax_other": dstar, "W_qq": p["W_qq"], "W_q_argmax": float(np.array(b["W"])[i, j]),
                 "O_q": p["O_q"], "O_loo": p["O_loo"], "verdict": p["verdict"],
                 "cos_model": float(b["cos_model"][i, j]), "cos_proj_w": float(b["cos_proj_w"][i, j]), "cos_proj_u": float(b["cos_proj_u"][i, j]),
                 "phi": float(labels[ds]["phi"][i, j]), "n11": int(labels[ds]["cooccurrence"][i, j]),
                 "table_2x2": list(labels[ds]["_tables"][(i, j)]),
                 "rank_by_cos_model": p["rank_of_argmax_other_by_cos_model"],
                 "rank_by_cos_proj_w": int(1 + sum(c["cos_proj_w"] > b["cos_proj_w"][i, j] for c in comps)),
                 "rank_by_phi": p["rank_of_argmax_other_by_phi"],
                 "rank_by_cooccurrence": int(1 + sum(c["n11"] > labels[ds]["cooccurrence"][i, j] for c in comps)),
                 "competitors": sorted(comps, key=lambda c: -c["W_qd"])}

    R = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_root": str(run_root),
                  "locus": a.locus, "fit_seed": a.fit_seed, "alpha": sorted(alphas)[0] if len(alphas) == 1 else sorted(alphas),
                  "n_blocks": len(blocks), "n_blocks_by_dataset": {ds: len(groups[ds]) for ds in DATASETS},
                  "blocks": list(blocks), "skipped": skipped, "labels_identical_across_blocks": labels_identical,
                  "label_digests": {ds: sorted(v) for ds, v in digests.items()}, "max_recon_error": float(max(recon)),
                  "coco_families": {k: list(v) for k, v in COCO_FAMILIES.items()},
                  "direction_formula": "v_c = normalize(P @ (w_c / max(s, 1e-8))); cos_proj_w = cos(w_q, w_d); cos_proj_u = cos(w_q/s, w_d/s)"},
         "labels": {ds: {k: v for k, v in lab.items() if not k.startswith("_")} for ds, lab in labels.items()},
         "direction_geometry": direction_geometry, "label_geometry": label_geometry, "association": assoc,
         "leave_one_out": loo, "ownership_median": own_med, "coco_families": fam, "seed_cell": seed_cell,
         "blocks": {k: {kk: vv for kk, vv in b.items() if kk != "cells"} for k, b in blocks.items()},
         "cells": [c for b in blocks.values() for c in b["cells"]]}
    R = clean(R)
    (out_dir / "geometry.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    write_markdown(R, out_dir / "geometry.md", a.locus, a.fit_seed)
    print(f"wrote {out_dir / 'geometry.json'} and geometry.md ({len(blocks)} blocks)")
    for name in ("chest", "nih", "chexpert", "coco"):
        if name in assoc:
            A = assoc[name]; G = loo[name]
            print(f"{name:9s} cells={A['n_cells']:4d} rho(W,cos)={A['rho_W_vs_cos_model']['rho']:+.3f} rho(W,phi)={A['rho_W_vs_phi']['rho']:+.3f} "
                  f"rho(adv,cos)={A['rho_adv_vs_cos_model']['rho']:+.3f} | O>0 {G['O_q_positive']}/{G['n_cells']} loo>0 {G['O_loo_positive']}/{G['n_cells']} "
                  f"n_above>=2 {G['n_above_ge2']} >=3 {G['n_above_ge3']}")
    for ds, g in direction_geometry.items():
        print(f"{ds:9s} mean offdiag cos_model {g['offdiag_cos_model_mean']:+.3f} argmax coincide {g['coincide_cos_model']['fraction']:.3f} "
              f"phi-coincide {label_geometry[ds]['coincide_phi']['fraction']:.3f}")
    print(f"seed cell {seed_cell['block']}/{seed_cell['q']} vs {seed_cell['argmax_other']}: cos_model {seed_cell['cos_model']:+.4f} "
          f"phi {seed_cell['phi']:+.4f} rank cos {seed_cell['rank_by_cos_model']} rank phi {seed_cell['rank_by_phi']}")


if __name__ == "__main__":
    main()
