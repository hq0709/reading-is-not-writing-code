#!/usr/bin/env python
"""Round-2 robustness modules over the cf-transfer campaign grid: VALID, ALTDIRD, ANSDIRT, ATTR, PRECISION (full grade).

Read-only over <RUN_ROOT>/<model_key>/<dataset>/{run.json,summary.json}. Only blocks under the paper's one inclusion rule
(cftransfer.manifest.block_included: CORE and CALIBRATION in run.json completed_modules) enter; within them a module
enters only when its statistics cover every row (the rules below). Writes <RUN_ROOT>/robustness/round2.json and
round2.md; nothing under the block directories is modified.

Ownership everywhere: owned = steering_reference and verdict == "fixed_family_advantage" (cftransfer.manifest.owned).

  valid      summary.json `valid` of CheXpert blocks with VALID in run.json completed_modules (as the manifest's valid_*
             columns). Per block and pooled: owned on the labeler-labelled test rows (the module's test regrade, cross-
             checked against the paper's core grade) and on the radiologist-labelled valid rows, verdict and ownership
             agreement, the owned disagreements, |dO_q| = |O_q(valid) - O_q(test)| (median, 90th percentile), and
             readable / answer-capable agreement restricted to cells with >= 10 positives and >= 10 negatives on both
             the valid rows and the block's calibration rows (the 10/10 support rule of the calibration grading).
  altdird    summary.json `altdird` of blocks with ALTDIRD completed and both displacement-lifted families (dom_disp,
             pattern_disp) scored on every test row. Per dataset (+ chest, all): blocks, cells, owned per family, the
             logistic normal's owned count on the same blocks (the paper's core grade), shares, median O_q, median
             model-space cosine to the logistic normal (cosines[c]["logistic"][family]); the R^T R / D spectrum
             (min eigenvalue, max eigenvalue, condition number range, full rank) across blocks.
  ansdirt    summary.json `ansdirt`: the answer direction a_q fitted on IY written under WY, IA, IB, WA, WB. A template
             enters when it is eligible and scored on every test row (n_scored_rows_min == n_rows); a block enters when
             every eligible template does and the IY grade exists. Per block and template: owned, IY-owned kept; pooled
             chest / COCO: IY-owned cells, kept per template, and kept (cell, template) pairs over all eligible templates.
             Blocks whose summary was scored on fewer rows (module still running, or a summary older than the completed
             module) are listed under partial_blocks with their row counts and never aggregated.
  attr       summary.json `attr` of blocks with ATTR in run.json completed_modules and the 9 x 9 write matrix scored on every
             test row (n_scored_rows_min == n_rows). Three non-clinical radiographic attributes (view_AP, sex_F, age_60) are
             fitted on the same images, features, projection and probe settings as the clinical probes and written against one
             nine-direction family (3 attributes + 6 findings) at the primary dose and template. Per block: owned attribute and
             owned clinical cells inside that family, per-attribute ownership, probe AUROC, selectivity, answer AUROC, and the
             median |cosine| of the attribute direction to the six clinical normals. Per dataset (+ chest, all): blocks, cells and
             owned counts for both kinds with their shares, per-attribute owned counts, medians of probe AUROC / selectivity /
             answer AUROC / |W_qq|, and the median |cosine| between attribute and clinical directions.
             The attribute-versus-finding ownership comparison is reported THREE ways (protocol.json attr_comparison), all
             prespecified and all reported whatever they show: `all_cells` (every cell, the comparison as it stood),
             `answer_capable_cells` (only cells whose clean answer clears the campaign's answer-capable rule, applied to
             attributes and findings alike) and `answerability_matched` (each attribute cell paired with the clinical cells
             of the SAME block whose clean-answer AUROC is within ATTR_MATCH_WINDOW of it, with the unmatched attribute
             cells named). Each mode carries the two ownership rates, their difference and a patient-bootstrap interval of
             that difference over the campaign's shared draws; the per-block rows carry every cell's answer AUROC and its
             source, so the matching is auditable. The clinical answer AUROC is READ from the block's own summary.json
             `calibration` and the attribute one from `attrq` (or, until ATTRQ is scored for the block, from the ATTR
             module's own test-row measurement) -- every cell records which, and on which rows.
  fgobj      summary.json `fgobj` of COCO blocks with FGOBJ in run.json completed_modules, the six-direction write matrix
             scored on every test row, and FGOBJ_CALIBRATION scored (so the fine-grained cells carry a clean-answer AUROC
             from the same calibration rows and the same rule as every other cell). Six COCO categories that are small,
             often occluded or fine-grained, built from the same instance annotations by the same rule, fitted with the
             same projection / scaler / probe settings on the same training rows, and written as a self-contained
             127-condition family with its own 119 random directions and its own shams. Per block: fine-grained and easy
             COCO owned cells, readability, answerability, median clean-answer AUROC and median probe selectivity for both
             families. Pooled over chest cells, easy COCO cells and fine-grained COCO cells: ownership rate and the two
             medians; then the DIFFICULTY-MATCHED comparison -- every chest cell paired with the natural-image cells of any
             block within the prespecified window, under three matchings (both variables, clean-answer AUROC alone, probe
             selectivity alone) and, for reference, the same matching with only the easy COCO cells as partners, which is
             the starved comparison FGOBJ exists to un-starve.
  precision  summary.json `precision` settings (fp32, batch1) with status COMPLETE and a full grade: verdict changes,
             steering-reference changes, cells whose verdict or reference changes, and ownership changes against CORE on the same 200 rows, max |dW| over all
             6 x 126 written cells (clinical, random, sham), max |d contrast|; pooled over blocks and settings.

Usage (from the concept-flow repo):
  PYTHONPATH=src python scripts/mayo/robustness_round2.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from cftransfer.analysis import unpack_draw_flags                                             # noqa: E402
from cftransfer.manifest import block_included, owned                                          # noqa: E402
from cftransfer.protocol import (ALTDIRD_FAMILIES, ANSDIRT_TEMPLATES, ATTR_CONCEPTS, ATTR_MATCH_WINDOW, BOOT_CORE_SEED,  # noqa: E402
                                 CONCEPTS, DATASETS, FGOBJ_CONCEPTS, MODEL_ORDER, PRECISION_SETTINGS)
from cftransfer.runpaths import RUN_ROOT                                                        # noqa: E402

OUT_DIR = RUN_ROOT / "robustness"
CHEST = ("nih", "chexpert")
SUPPORT_MIN = 10
GROUPS = {"nih": ("nih",), "chexpert": ("chexpert",), "coco": ("coco",), "chest": CHEST, "all": tuple(DATASETS)}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def p90(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.percentile(xs, 90)) if xs else None


def block_order(key: tuple[str, str]):
    mk, ds = key
    return (MODEL_ORDER.index(mk) if mk in MODEL_ORDER else 999, mk, DATASETS.index(ds))


def discover(run_root: Path) -> tuple[dict, list]:
    """(model, dataset) -> {run, summary} for every block under the inclusion rule; skipped blocks with the reason."""
    blocks, skipped = {}, []
    for rj in sorted(run_root.glob("*/*/run.json")):
        mk, ds = rj.parent.parent.name, rj.parent.name
        if ds not in DATASETS:
            continue
        run = json.loads(rj.read_text())
        if not block_included(run):
            skipped.append({"block": f"{mk}/{ds}", "reason": "not included: run.json completed_modules lacks CORE or CALIBRATION"})
            continue
        sp = rj.parent / "summary.json"
        blocks[(mk, ds)] = {"run": run, "summary": json.loads(sp.read_text()) if sp.exists() else {}}
    return dict(sorted(blocks.items(), key=lambda kv: block_order(kv[0]))), skipped


def completed(b: dict, module: str) -> bool:
    return module in set(b["run"].get("completed_modules") or [])


# ------------------------------------------------------------------------------------------------ VALID
def valid_section(blocks: dict) -> dict:
    rows, skipped = [], []
    for (mk, ds), b in blocks.items():
        s = b["summary"]
        if ds != "chexpert":
            continue
        v = s.get("valid")
        if not v:
            skipped.append({"block": f"{mk}/{ds}", "reason": "no summary.json valid"}); continue
        if not completed(b, "VALID"):
            skipped.append({"block": f"{mk}/{ds}", "reason": "VALID not in run.json completed_modules"}); continue
        core, cal, comp = s["core"]["per_question"], s.get("calibration") or {}, v["comparison"]["per_question"]
        per_q, mismatches = {}, []
        for q in CONCEPTS[ds]:
            t, va = comp[q]["test"], comp[q]["valid"]
            paper_owned = owned(core[q])
            if paper_owned != bool(t["owned"]) or core[q]["verdict"] != t["verdict"]:
                mismatches.append({"concept": q, "paper_owned": paper_owned, "regrade_owned": bool(t["owned"]),
                                   "paper_verdict": core[q]["verdict"], "regrade_verdict": t["verdict"]})
            vq, cq = v["per_question"][q], cal.get(q) or {}
            supported = bool(min(vq.get("n_pos") or 0, vq.get("n_neg") or 0) >= SUPPORT_MIN
                             and min(cq.get("n_pos") or 0, cq.get("n_neg") or 0) >= SUPPORT_MIN)
            per_q[q] = {"test": {k: t.get(k) for k in ("W_qq", "O_q", "verdict", "steering_reference", "owned", "readable", "answer_capable")},
                        "valid": {k: va.get(k) for k in ("W_qq", "O_q", "verdict", "steering_reference", "owned", "readable", "answer_capable")},
                        "paper_owned": paper_owned, "dO_q": comp[q].get("dO_q"),
                        "abs_dO_q": abs(comp[q]["dO_q"]) if comp[q].get("dO_q") is not None else None,
                        "verdict_agree": t["verdict"] == va["verdict"], "owned_agree": bool(t["owned"]) == bool(va["owned"]),
                        "valid_n_pos": vq.get("n_pos"), "valid_n_neg": vq.get("n_neg"),
                        "calibration_n_pos": cq.get("n_pos"), "calibration_n_neg": cq.get("n_neg"), "supported_10_10": supported,
                        "valid_readable_status": vq.get("readable_status"),
                        "readable_agree": (t["readable"] == va["readable"]) if supported and t.get("readable") is not None and va.get("readable") is not None else None,
                        "answer_capable_agree": (t["answer_capable"] == va["answer_capable"]) if supported and t.get("answer_capable") is not None and va.get("answer_capable") is not None else None}
        rows.append({"block": f"{mk}/{ds}", "model": mk, "dataset": ds, "n_rows": v["n_rows"], "test_rows": v["comparison"]["test_rows"],
                     "features_available": v.get("features_available"), "label_source": v.get("label_source"),
                     "owned_test": sum(c["test"]["owned"] for c in per_q.values()), "owned_valid": sum(bool(c["valid"]["owned"]) for c in per_q.values()),
                     "owned_paper": sum(c["paper_owned"] for c in per_q.values()),
                     "verdict_agree": sum(c["verdict_agree"] for c in per_q.values()), "owned_agree": sum(c["owned_agree"] for c in per_q.values()),
                     "n_cells": len(per_q), "median_abs_dO_q": med([c["abs_dO_q"] for c in per_q.values()]),
                     "module_n_agree": v["comparison"]["n_agree"], "module_n_compared": v["comparison"]["n_compared"],
                     "test_regrade_vs_paper_mismatches": mismatches, "per_question": per_q})
    cells = [(r, q, c) for r in rows for q, c in r["per_question"].items()]
    sup = [c for _, _, c in cells if c["supported_10_10"]]
    agg = {"blocks": len(rows), "cells": len(cells), "valid_rows": sorted({r["n_rows"] for r in rows}),
           "owned_agree": sum(c["owned_agree"] for _, _, c in cells), "verdict_agree": sum(c["verdict_agree"] for _, _, c in cells),
           "owned_test": sum(r["owned_test"] for r in rows), "owned_valid": sum(r["owned_valid"] for r in rows),
           "owned_paper": sum(r["owned_paper"] for r in rows),
           "test_regrade_vs_paper_mismatches": sum(len(r["test_regrade_vs_paper_mismatches"]) for r in rows),
           "median_abs_dO_q": med([c["abs_dO_q"] for _, _, c in cells]), "p90_abs_dO_q": p90([c["abs_dO_q"] for _, _, c in cells]),
           "disagreements": [{"block": r["block"], "concept": q, "owned_test": c["test"]["owned"], "owned_valid": bool(c["valid"]["owned"]),
                              "O_q_test": c["test"]["O_q"], "O_q_valid": c["valid"]["O_q"], "verdict_test": c["test"]["verdict"],
                              "verdict_valid": c["valid"]["verdict"], "reference_test": c["test"]["steering_reference"],
                              "reference_valid": c["valid"]["steering_reference"]} for r, q, c in cells if not c["owned_agree"]],
           "supported_cells": len(sup),
           "readable_agree_supported": sum(bool(c["readable_agree"]) for c in sup if c["readable_agree"] is not None),
           "readable_compared_supported": sum(c["readable_agree"] is not None for c in sup),
           "answer_capable_agree_supported": sum(bool(c["answer_capable_agree"]) for c in sup if c["answer_capable_agree"] is not None),
           "answer_capable_compared_supported": sum(c["answer_capable_agree"] is not None for c in sup),
           "unsupported_cells": [f"{r['block']}:{q}" for r, q, c in cells if not c["supported_10_10"]]}
    return {"blocks": rows, "aggregate": agg, "skipped": skipped}


# ------------------------------------------------------------------------------------------------ ALTDIRD
def altdird_section(blocks: dict) -> dict:
    rows, skipped = [], []
    for (mk, ds), b in blocks.items():
        a = b["summary"].get("altdird")
        if not a:
            continue
        if not completed(b, "ALTDIRD"):
            skipped.append({"block": f"{mk}/{ds}", "reason": "ALTDIRD not in run.json completed_modules"}); continue
        short = {f: a.get(f, {}).get("n_scored_rows_min") for f in ALTDIRD_FAMILIES}
        if any(short[f] != a["n_rows"] for f in ALTDIRD_FAMILIES):
            skipped.append({"block": f"{mk}/{ds}", "reason": f"families scored on {short} of {a['n_rows']} rows"}); continue
        core = b["summary"]["core"]["per_question"]
        per_q = {}
        for q in CONCEPTS[ds]:
            cell = {"logistic": {"owned": owned(core[q]), "O_q": core[q]["O_q"]}}
            for f in ALTDIRD_FAMILIES:
                c = a[f]["per_question"][q]
                cos = a["cosines"][q]["logistic"][f]
                if abs(cos - c["cos_to_logistic_model"]) > 1e-9:
                    raise RuntimeError(f"{mk}/{ds} {q} {f}: cosines[logistic] {cos} != cell cos_to_logistic_model {c['cos_to_logistic_model']}")
                cell[f] = {"owned": owned(c), "O_q": c["O_q"], "W_qq": c["W_qq"], "verdict": c["verdict"],
                           "steering_reference": c["steering_reference"], "cos_to_logistic_model": cos}
            per_q[q] = cell
        rows.append({"block": f"{mk}/{ds}", "model": mk, "dataset": ds, "n_rows": a["n_rows"], "gram_spectrum": a["gram_spectrum"],
                     "owned": {f: sum(c[f]["owned"] for c in per_q.values()) for f in ("logistic",) + tuple(ALTDIRD_FAMILIES)},
                     "per_question": per_q})
    per = {}
    for g, dss in GROUPS.items():
        bs = [r for r in rows if r["dataset"] in dss]
        cells = [c for r in bs for c in r["per_question"].values()]
        n = len(cells)
        d = {"blocks": len(bs), "cells": n}
        for f in ("logistic",) + tuple(ALTDIRD_FAMILIES):
            k = sum(c[f]["owned"] for c in cells)
            d[f] = {"owned": k, "share": k / n if n else None, "median_O_q": med([c[f]["O_q"] for c in cells])}
            if f != "logistic":
                d[f]["median_cos_to_logistic"] = med([c[f]["cos_to_logistic_model"] for c in cells])
        gs = [r["gram_spectrum"] for r in bs]
        d["gram_spectrum"] = ({"min_eigenvalue": min(x["min"] for x in gs), "max_eigenvalue": max(x["max"] for x in gs),
                               "condition_min": min(x["condition"] for x in gs), "condition_max": max(x["condition"] for x in gs),
                               "condition_median": med([x["condition"] for x in gs]), "all_full_rank": all(x["full_rank"] for x in gs),
                               "rank_min": min(x["rank"] for x in gs), "n": sorted({x["n"] for x in gs})} if gs else None)
        per[g] = d
    return {"families": list(ALTDIRD_FAMILIES), "blocks": [{k: v for k, v in r.items()} for r in rows], "per_dataset": per, "skipped": skipped}


# ------------------------------------------------------------------------------------------------ ANSDIRT
def ansdirt_section(blocks: dict) -> dict:
    rows, partial, skipped = [], [], []
    for (mk, ds), b in blocks.items():
        a = b["summary"].get("ansdirt")
        if not a:
            continue
        rec = {"block": f"{mk}/{ds}", "model": mk, "dataset": ds, "n_rows": a["n_rows"], "module_completed": completed(b, "ANSDIRT"),
               "ineligible_templates": a.get("ineligible_templates") or [], "not_started_templates": a.get("not_started_templates") or []}
        short = {t: a["per_template"][t]["n_scored_rows_min"] for t in a.get("templates", []) if a["per_template"][t]["n_scored_rows_min"] != a["n_rows"]}
        eligible = [t for t in ANSDIRT_TEMPLATES if t not in rec["ineligible_templates"]]
        missing = [t for t in eligible if t not in a.get("templates", [])]
        if a.get("iy_owned") is None:
            reason = "IY answer-direction grade (ansdir) missing"
        elif missing or short:
            reason = (f"templates not scored: {missing}; " if missing else "") + (f"scored on fewer than {a['n_rows']} rows: {short}" if short else "")
            if rec["module_completed"] and not missing:
                reason = "summary.json ansdirt is older than the completed module (" + reason + "); rerun python -m cftransfer.analysis --what ansdirt"
        else:
            reason = None
        if reason:
            skipped.append({"block": rec["block"], "reason": reason})
            if a.get("iy_owned") is None or missing:
                continue
        rec["status"] = "OK" if reason is None else ("STALE_SUMMARY" if rec["module_completed"] else "INCOMPLETE")
        rec["n_scored_rows_min"] = {t: a["per_template"][t]["n_scored_rows_min"] for t in eligible}
        iy = set(a["iy_owned"])
        ans = (b["summary"].get("ansdir") or {}).get("per_question") or {}
        iy_summary = {q for q, c in ans.items() if owned(c)}
        rec["iy_owned"] = sorted(iy); rec["n_iy_owned"] = len(iy)
        rec["iy_owned_matches_summary_ansdir"] = (iy == iy_summary) if ans else None
        rec["templates"] = {}
        for t in eligible:
            pt, tr = a["per_template"][t], a["transfer"][t]
            ow = {q for q, c in pt["per_question"].items() if c["steering_reference"] and c["verdict"] == "fixed_family_advantage"}
            if ow != set(pt["owned"]) or len(iy & ow) != tr["n_stay_owned"]:
                raise RuntimeError(f"{rec['block']} {t}: owned {sorted(ow)} vs recorded {pt['owned']} / stay {tr['n_stay_owned']}")
            rec["templates"][t] = {"n_owned": len(ow), "owned": sorted(ow), "kept_of_iy": len(iy & ow), "lost": sorted(iy - ow), "gained": sorted(ow - iy)}
        rec["kept_pairs"] = sum(v["kept_of_iy"] for v in rec["templates"].values())
        rec["pairs"] = len(iy) * len(rec["templates"])
        (rows if rec["status"] == "OK" else partial).append(rec)
    agg = {}
    for g, dss in (("chest", CHEST), ("coco", ("coco",)), ("all", tuple(DATASETS))):
        bs = [r for r in rows if r["dataset"] in dss]
        agg[g] = {"blocks": len(bs), "cells": sum(len(CONCEPTS[r["dataset"]]) for r in bs), "iy_owned": sum(r["n_iy_owned"] for r in bs),
                  "per_template": {t: {"owned": sum(r["templates"][t]["n_owned"] for r in bs if t in r["templates"]),
                                       "kept_of_iy": sum(r["templates"][t]["kept_of_iy"] for r in bs if t in r["templates"]),
                                       "iy_owned": sum(r["n_iy_owned"] for r in bs if t in r["templates"])} for t in ANSDIRT_TEMPLATES},
                  "kept_pairs": sum(r["kept_pairs"] for r in bs), "pairs": sum(r["pairs"] for r in bs),
                  # unconditional: every (cell, held-out template) pair, not only the pairs whose cell is owned under IY
                  "owned_pairs": sum(v["n_owned"] for r in bs for v in r["templates"].values()),
                  "all_pairs": sum(len(r["templates"]) * len(CONCEPTS[r["dataset"]]) for r in bs),
                  "blocks_list": [r["block"] for r in bs]}
    return {"templates": list(ANSDIRT_TEMPLATES), "blocks": rows, "aggregate": agg, "skipped": skipped,
            "partial_blocks": partial}          # scored on fewer rows than the cohort: listed for tracking, never aggregated


# ------------------------------------------------------------------------------------------------ ATTR
ATTR_FIELDS = ("auroc_real", "selectivity", "selectivity_lower95_one_sided", "answer_auroc", "answer_auroc_lower95_one_sided",
               "readable", "readable_status", "answer_capable", "n_pos", "n_neg", "n_train_pos", "n_train_neg")


# --- the attribute-versus-finding comparison, three ways (protocol.json attr_comparison) -------------------
def cell_auroc(cell: dict):
    """The clean-answer AUROC of one graded cell, as analysis.attr recorded it (None when the block carries none)."""
    return (cell.get("answerability") or {}).get("answer_auroc")


def cell_capable(cell: dict):
    return (cell.get("answerability") or {}).get("answer_capable")


def draw_flags(cell: dict, draws):
    """Per-draw ownership of one cell over the campaign's shared unit (patient) bootstrap, or None."""
    h = cell.get("owned_draws_hex")
    if not h or not draws:
        return None
    try:
        return unpack_draw_flags(h, int(draws))
    except (ValueError, TypeError):
        return None


BLOCK_BOOT_DRAWS = 2000


def block_bootstrap(attr_items: list, clin_items: list, blocks: list, draws: int = BLOCK_BOOT_DRAWS,
                    seed: int = BOOT_CORE_SEED) -> dict:
    """Cluster bootstrap over BLOCKS of the two ownership rates and their difference.

    The patient bootstrap resamples the rows inside a block, and at these effect sizes a cell's ownership almost never
    flips when it does, so that interval is near-degenerate: it says the per-cell decisions are stable, not that the
    RATE over cells is precise. The uncertainty that matters for "are attribute cells owned as often as finding cells"
    is which (model, dataset) blocks happen to be in the grid, so the same rates are also resampled over blocks, paired
    (one resample of blocks used for both kinds in every draw)."""
    if not blocks or not attr_items or not clin_items:
        return {"available": False, "reason": "no blocks or no cells on one side"}
    order = {b: i for i, b in enumerate(blocks)}
    na = np.zeros(len(blocks)); oa = np.zeros(len(blocks))
    nc = np.zeros(len(blocks)); oc = np.zeros(len(blocks))
    for b, c in attr_items:
        na[order[b]] += 1; oa[order[b]] += bool(c["owned"])
    for b, c in clin_items:
        nc[order[b]] += 1; oc[order[b]] += bool(c["owned"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(draws, len(blocks)))
    W = np.zeros((draws, len(blocks)))
    np.add.at(W, (np.repeat(np.arange(draws), len(blocks)), idx.ravel()), 1.0)
    dna, dnc = W @ na, W @ nc
    ok = (dna > 0) & (dnc > 0)
    if not ok.any():
        return {"available": False, "reason": "every draw is missing one kind of cell"}
    ra = (W @ oa)[ok] / dna[ok]
    rc = (W @ oc)[ok] / dnc[ok]
    d = ra - rc
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"available": True, "draws": int(draws), "valid_draws": int(ok.sum()), "n_blocks": len(blocks), "seed": int(seed),
            "attr_owned_share_ci95": [float(np.percentile(ra, 2.5)), float(np.percentile(ra, 97.5))],
            "clin_owned_share_ci95": [float(np.percentile(rc, 2.5)), float(np.percentile(rc, 97.5))],
            "difference_mean": float(d.mean()), "difference_ci95_percentile": [lo, hi],
            "difference_excludes_zero": bool(lo > 0 or hi < 0),
            "unit": "one (model, dataset) block = one cluster; the same resample of blocks is used for both kinds in a draw"}


def rate_difference(attr_cells: list, clin_cells: list, draws) -> dict:
    """Ownership rate of each kind over two LISTS of graded cells (a cell may repeat: the matched mode lists one entry
    per pair), their difference, and the percentile interval of the difference over the shared patient-bootstrap draws.
    The point rates use the campaign's owned() rule; the interval re-decides ownership inside every draw as
    steering_reference and O_q > 0, which is what analysis.attr stores per cell, so the per-draw rates are reported
    beside the point rates rather than instead of them."""
    na, nc = len(attr_cells), len(clin_cells)
    oa = sum(bool(c["owned"]) for c in attr_cells)
    oc = sum(bool(c["owned"]) for c in clin_cells)
    ra = (oa / na) if na else None
    rc = (oc / nc) if nc else None
    out = {"attr_cells": na, "attr_owned": oa, "attr_owned_share": ra,
           "clin_cells": nc, "clin_owned": oc, "clin_owned_share": rc,
           "difference": (ra - rc) if (ra is not None and rc is not None) else None}
    fa = [draw_flags(c, draws) for c in attr_cells]
    fc = [draw_flags(c, draws) for c in clin_cells]
    missing = sum(f is None for f in fa) + sum(f is None for f in fc)
    if not (na and nc) or missing:
        out["bootstrap"] = {"available": False,
                            "reason": ("no cells on one side" if not (na and nc) else
                                       f"{missing} of {na + nc} listed cells carry no per-draw ownership "
                                       f"(summary.json attr predates owned_draws_hex; rerun "
                                       f"python -m cftransfer.analysis --what attr for those blocks)")}
        return out
    A = np.stack(fa).mean(axis=0)
    C = np.stack(fc).mean(axis=0)
    D = A - C
    lo, hi = float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5))
    out["bootstrap"] = {"available": True, "draws": int(draws),
                        "attr_owned_share_per_draw": float(A.mean()), "clin_owned_share_per_draw": float(C.mean()),
                        "attr_owned_share_ci95": [float(np.percentile(A, 2.5)), float(np.percentile(A, 97.5))],
                        "clin_owned_share_ci95": [float(np.percentile(C, 2.5)), float(np.percentile(C, 97.5))],
                        "difference_per_draw": float(D.mean()), "difference_ci95_percentile": [lo, hi],
                        "difference_excludes_zero": bool(lo > 0 or hi < 0),
                        "unit": "test row = one unit (patient on NIH / CheXpert); the campaign's shared draws, "
                                "combined across datasets by draw number",
                        "per_draw_rule": "steering_reference (point estimate) and O_q > 0 inside the draw"}
    return out


def match_block(row: dict, window: float = ATTR_MATCH_WINDOW) -> tuple[list, list]:
    """Pair every attribute cell of one block with the clinical cells of the SAME block whose clean-answer AUROC is
    within `window` of it. Returns (pairs, unmatched); an attribute cell with no answer AUROC, or none within the
    window, is unmatched and named with its nearest clinical cell."""
    pairs, unmatched = [], []
    clin = [(q, c, cell_auroc(c)) for q, c in row["clinical"].items()]
    for q, ca in row["attributes"].items():
        aa = cell_auroc(ca)
        if aa is None:
            unmatched.append({"block": row["block"], "attribute": q, "answer_auroc": None, "nearest_clinical": None,
                              "reason": "no clean-answer AUROC recorded for this attribute cell"})
            continue
        near = [(qc, cc, abs(aa - ac)) for qc, cc, ac in clin if ac is not None]
        inside = [t for t in near if t[2] <= window]
        if not inside:
            n0 = min(near, key=lambda t: t[2], default=None)
            unmatched.append({"block": row["block"], "attribute": q, "answer_auroc": aa,
                              "nearest_clinical": ({"concept": n0[0], "answer_auroc": cell_auroc(n0[1]), "gap": n0[2]}
                                                   if n0 else None),
                              "reason": ("no clinical cell of this block within the window" if near else
                                         "no clinical cell of this block carries a clean-answer AUROC")})
            continue
        for qc, cc, gap in inside:
            pairs.append({"block": row["block"], "attribute": q, "clinical": qc, "abs_answer_auroc_gap": gap,
                          "attr_answer_auroc": aa, "clin_answer_auroc": cell_auroc(cc),
                          "attr_owned": bool(ca["owned"]), "clin_owned": bool(cc["owned"])})
    return pairs, unmatched


def _print_comparison(label: str, cmp: dict) -> None:
    """One stdout line per comparison mode: the two ownership rates, their difference and the bootstrap interval."""
    for mode in ("all_cells", "answer_capable_cells", "answerability_matched"):
        m = cmp[mode]
        b = m.get("bootstrap") or {}
        k = m.get("block_bootstrap") or {}
        ci = (f"patient [{b['difference_ci95_percentile'][0]:+.3f}, {b['difference_ci95_percentile'][1]:+.3f}]"
              if b.get("available") else "patient n/a")
        cib = (f"block [{k['difference_ci95_percentile'][0]:+.3f}, {k['difference_ci95_percentile'][1]:+.3f}]"
               if k.get("available") else "block n/a")
        extra = (f" pairs={m['n_pairs']} unmatched={m['n_attr_cells_unmatched']}/{m['n_attr_cells']}"
                 if mode == "answerability_matched" else "")
        print(f"  attr-compare {label:18s} {mode:22s} attr {m['attr_owned']}/{m['attr_cells']} "
              f"clinical {m['clin_owned']}/{m['clin_cells']} "
              f"d={m['difference'] if m['difference'] is None else round(m['difference'], 4)} {ci} {cib}{extra}")


def attr_comparison(rows: list, window: float = ATTR_MATCH_WINDOW) -> dict:
    """The three prespecified attribute-versus-finding comparisons over a set of block rows."""
    A = [(r, q, c) for r in rows for q, c in r["attributes"].items()]
    C = [(r, q, c) for r in rows for q, c in r["clinical"].items()]
    dset = sorted({r.get("draws") for r in rows if r.get("draws")})
    draws = dset[0] if len(dset) == 1 else None
    res = {"window": window, "blocks": len(rows), "draws": draws,
           "draws_note": None if draws else f"blocks disagree on the bootstrap draw count {dset}; no pooled interval",
           "answer_auroc_rows": {"attribute": sorted({(c.get("answerability") or {}).get("rows") for _r, _q, c in A} - {None}),
                                 "clinical": sorted({(c.get("answerability") or {}).get("rows") for _r, _q, c in C} - {None})},
           "median_answer_n_known": {
               "attribute": med([sum(x for x in ((c.get("answerability") or {}).get("n_pos"),
                                                 (c.get("answerability") or {}).get("n_neg")) if x is not None) or None
                                 for _r, _q, c in A]),
               "clinical": med([sum(x for x in ((c.get("answerability") or {}).get("n_pos"),
                                                (c.get("answerability") or {}).get("n_neg")) if x is not None) or None
                                for _r, _q, c in C])}}
    blist = [r["block"] for r in rows]
    res["all_cells"] = rate_difference([c for _r, _q, c in A], [c for _r, _q, c in C], draws)
    res["all_cells"]["block_bootstrap"] = block_bootstrap([(r["block"], c) for r, _q, c in A],
                                                          [(r["block"], c) for r, _q, c in C], blist)
    Ak = [t for t in A if cell_capable(t[2]) is True]
    Ck = [t for t in C if cell_capable(t[2]) is True]
    cap = rate_difference([c for _r, _q, c in Ak], [c for _r, _q, c in Ck], draws)
    cap["block_bootstrap"] = block_bootstrap([(r["block"], c) for r, _q, c in Ak],
                                             [(r["block"], c) for r, _q, c in Ck], blist)
    cap.update({"attr_cells_excluded": len(A) - len(Ak), "clin_cells_excluded": len(C) - len(Ck),
                "attr_cells_without_answer_number": sum(cell_capable(c) is None for _r, _q, c in A),
                "clin_cells_without_answer_number": sum(cell_capable(c) is None for _r, _q, c in C),
                "rule": "answer_capable as analysis.calibration defines it, applied to attributes and findings alike"})
    res["answer_capable_cells"] = cap
    pairs, unmatched = [], []
    for r in rows:
        pp, uu = match_block(r, window)
        pairs += pp
        unmatched += uu
    by_block = {r["block"]: r for r in rows}
    by_pair_attr = [by_block[p["block"]]["attributes"][p["attribute"]] for p in pairs]
    by_pair_clin = [by_block[p["block"]]["clinical"][p["clinical"]] for p in pairs]
    m = rate_difference(by_pair_attr, by_pair_clin, draws)
    m["block_bootstrap"] = block_bootstrap([(p["block"], a) for p, a in zip(pairs, by_pair_attr)],
                                           [(p["block"], c) for p, c in zip(pairs, by_pair_clin)], blist)
    m.update({"n_pairs": len(pairs), "n_attr_cells": len(A),
              "n_attr_cells_matched": len({(p["block"], p["attribute"]) for p in pairs}),
              "n_attr_cells_unmatched": len(unmatched), "unmatched": unmatched,
              "median_abs_answer_auroc_gap": med([p["abs_answer_auroc_gap"] for p in pairs]),
              "max_abs_answer_auroc_gap": max([p["abs_answer_auroc_gap"] for p in pairs], default=None),
              "n_distinct_clinical_cells": len({(p["block"], p["clinical"]) for p in pairs}),
              "rule": (f"an attribute cell pairs with every clinical cell of the SAME block whose clean-answer AUROC is "
                       f"within {window} of it; the rates are over PAIRS, so a cell that matches several counts several times")})
    res["answerability_matched"] = m
    return res


def attr_section(blocks: dict) -> dict:
    """Non-clinical attribute directions written inside one nine-direction family, beside the clinical directions of the same family.

    Picks up whatever blocks carry the module at run time: a block enters when it is included, ATTR is in run.json
    completed_modules, and summary.json attr was scored on every test row."""
    rows, skipped = [], []
    for (mk, ds), b in blocks.items():
        a = b["summary"].get("attr")
        if not a:
            if completed(b, "ATTR"):
                skipped.append({"block": f"{mk}/{ds}", "reason": "ATTR completed but summary.json carries no attr; rerun python -m cftransfer.analysis --what attr"})
            continue
        if not completed(b, "ATTR"):
            skipped.append({"block": f"{mk}/{ds}", "reason": "ATTR not in run.json completed_modules"}); continue
        if a.get("n_scored_rows_min") != a.get("n_rows"):
            skipped.append({"block": f"{mk}/{ds}", "reason": f"attr scored on {a.get('n_scored_rows_min')} of {a.get('n_rows')} rows"}); continue
        expect_q = list(ATTR_CONCEPTS) + list(CONCEPTS[ds])
        if list(a["questions"]) != expect_q or list(a["attributes"]) != list(ATTR_CONCEPTS):
            raise RuntimeError(f"{mk}/{ds} attr: questions {a['questions']} / attributes {list(a['attributes'])} are not {expect_q}")
        attrs, clin, cos_pairs = {}, {}, []
        for q in expect_q:
            c = a["per_question"][q]
            kind = "attribute" if q in ATTR_CONCEPTS else "clinical"
            if c["question_kind"] != kind or c["own_direction"] != (f"attr:{q}" if kind == "attribute" else f"concept:{q}"):
                raise RuntimeError(f"{mk}/{ds} attr {q}: question_kind {c['question_kind']} / own_direction {c['own_direction']} unexpected for a {kind} question")
            cell = {"owned": owned(c), "verdict": c["verdict"], "steering_reference": c["steering_reference"], "W_qq": c["W_qq"],
                    "O_q": c["O_q"], "max_other": c["max_other"], "argmax_other": c["argmax_other"],
                    "random_reference": c["random_reference"], "abs_sham": c["abs_sham"],
                    # the clean-answer number this cell is compared and matched on, exactly as analysis.attr read it
                    # from the block's own summary (calibration for a finding, attrq / ATTR test rows for an attribute)
                    "answerability": c.get("answerability") or {},
                    "owned_draws_hex": c.get("owned_draws_hex"), "owned_draw_rate": c.get("owned_draw_rate")}
            if kind == "attribute":
                at = a["attributes"][q]
                cos = {k: float(v) for k, v in at["cos_model_to_clinical"].items()}
                if set(cos) != set(CONCEPTS[ds]):
                    raise RuntimeError(f"{mk}/{ds} attr {q}: cos_model_to_clinical keys {sorted(cos)} are not the six clinical concepts")
                cell.update({k: at.get(k) for k in ATTR_FIELDS})
                cell["cos_model_to_clinical"] = cos
                cell["median_abs_cos_to_clinical"] = med([abs(v) for v in cos.values()])
                cos_pairs += [(q, k, abs(v)) for k, v in cos.items()]
                attrs[q] = cell
            else:
                clin[q] = cell
        rows.append({"block": f"{mk}/{ds}", "model": mk, "dataset": ds, "n_rows": a["n_rows"], "template_id": a.get("template_id"),
                     "alpha": a.get("alpha"), "draws": a.get("draws"), "max_t_critical": a.get("max_t_critical"),
                     "attr_cells": len(attrs), "attr_owned": sum(c["owned"] for c in attrs.values()),
                     "clin_cells": len(clin), "clin_owned": sum(c["owned"] for c in clin.values()),
                     "attr_readable": sum(bool(c["readable"]) for c in attrs.values()),
                     "attr_answer_capable": sum(bool(c["answer_capable"]) for c in attrs.values()),
                     "median_abs_W_qq_attr": med([abs(c["W_qq"]) for c in attrs.values()]),
                     "median_abs_W_qq_clinical": med([abs(c["W_qq"]) for c in clin.values()]),
                     "median_abs_cos_attr_clinical": med([x for _, _, x in cos_pairs]),
                     "attributes": attrs, "clinical": clin})
        rows[-1]["matched_pairs"], rows[-1]["matched_unmatched"] = match_block(rows[-1])
        rows[-1]["comparison"] = attr_comparison([rows[-1]])
    per = {}
    for g, dss in GROUPS.items():
        bs = [r for r in rows if r["dataset"] in dss]
        ac = [c for r in bs for c in r["attributes"].values()]
        cc = [c for r in bs for c in r["clinical"].values()]
        # blocks whose ATTRIBUTE cells carry the protocol's 119-direction random family (module ATTRRAND scored).
        # Without it an attribute cell is referenced against its sham alone while a clinical cell has the random p95,
        # so the two kinds are not on one steering reference and an ownership rate pooled over both kinds of block
        # would mix two rules. The comparison to read is the one restricted to these blocks.
        bs_ref = [r for r in bs if r["attributes"] and all(c["random_reference"] for c in r["attributes"].values())]
        pair = {}
        for r in bs:
            for q, c in r["attributes"].items():
                for k, v in c["cos_model_to_clinical"].items():
                    pair.setdefault((q, k), []).append(abs(v))
        pm = {f"{q}~{k}": {"median_abs_cos": med(v), "n_blocks": len(v)} for (q, k), v in sorted(pair.items())}
        top = max(pm.items(), key=lambda kv: kv[1]["median_abs_cos"], default=None) if pm else None
        d = {"blocks": len(bs), "blocks_list": [r["block"] for r in bs],
             "attr_cells": len(ac), "attr_owned": sum(c["owned"] for c in ac), "attr_owned_share": (sum(c["owned"] for c in ac) / len(ac)) if ac else None,
             "clin_cells": len(cc), "clin_owned": sum(c["owned"] for c in cc), "clin_owned_share": (sum(c["owned"] for c in cc) / len(cc)) if cc else None,
             "attr_readable": sum(bool(c["readable"]) for c in ac), "attr_answer_capable": sum(bool(c["answer_capable"]) for c in ac),
             "median_abs_W_qq_attr": med([abs(c["W_qq"]) for c in ac]), "median_abs_W_qq_clinical": med([abs(c["W_qq"]) for c in cc]),
             "median_abs_cos_attr_clinical": med([abs(v) for c in ac for v in c["cos_model_to_clinical"].values()]),
             "per_pair_median_abs_cos": pm,
             "max_median_abs_cos_pair": ({"pair": top[0], **top[1]} if top else None),
             "attribute_random_reference": any(c["random_reference"] for c in ac),
             "clinical_random_reference": all(c["random_reference"] for c in cc) if cc else None,
             "comparison": attr_comparison(bs),
             "blocks_matched_reference": [r["block"] for r in bs_ref],
             "comparison_matched_reference": attr_comparison(bs_ref),
             "per_attribute": {}}
        for q in ATTR_CONCEPTS:
            xs = [r["attributes"][q] for r in bs if q in r["attributes"]]
            d["per_attribute"][q] = {
                "blocks": len(xs), "owned": sum(c["owned"] for c in xs), "share": (sum(c["owned"] for c in xs) / len(xs)) if xs else None,
                "readable": sum(bool(c["readable"]) for c in xs), "answer_capable": sum(bool(c["answer_capable"]) for c in xs),
                "median_auroc_real": med([c["auroc_real"] for c in xs]), "median_selectivity": med([c["selectivity"] for c in xs]),
                "median_answer_auroc": med([c["answer_auroc"] for c in xs]),
                "median_abs_W_qq": med([abs(c["W_qq"]) for c in xs]),
                "median_abs_cos_to_clinical": med([abs(v) for c in xs for v in c["cos_model_to_clinical"].values()]),
                "median_n_pos": med([c["n_pos"] for c in xs]), "median_n_neg": med([c["n_neg"] for c in xs])}
        per[g] = d
    return {"attributes": list(ATTR_CONCEPTS), "blocks": rows, "per_dataset": per, "skipped": skipped,
            "comparison_modes": {"all_cells": "every attribute cell against every clinical cell of the same blocks",
                                 "answer_capable_cells": "only cells whose clean answer clears the campaign's answer-capable rule",
                                 "answerability_matched": f"attribute cells paired with clinical cells of the same block "
                                                          f"within {ATTR_MATCH_WINDOW} clean-answer AUROC"},
            "match_window": ATTR_MATCH_WINDOW}


# ------------------------------------------------------------------------------------------------ FGOBJ
# The difficulty-matched comparison. PRESPECIFICATION: the two matching windows below -- 0.05 on clean-answer AUROC and
# 0.05 on probe selectivity -- were fixed in the module brief BEFORE the FGOBJ grid was built and before any FGOBJ
# outcome existed. They are the windows the chest-versus-easy-COCO matched comparison already used, they were not tuned
# to any result here, and nothing in this section chooses a window, a group or a category by an ownership outcome. The
# six fine-grained categories were likewise chosen by cohort support and instance size alone (protocol.FGOBJ_CONCEPTS).
FGOBJ_WINDOW_ANSWER_AUROC = 0.05
FGOBJ_WINDOW_SELECTIVITY = 0.05
FGOBJ_BOOT_DRAWS = 5000
FGOBJ_BOOT_SEED = 2026091601
# The interval is a CLUSTER bootstrap over MODELS, not over patients, and it is named that way everywhere it is
# reported. A cell's ownership flag is already a 600-patient statistic (the max-T verdict and the steering reference are
# computed inside the block from the block's own unit bootstrap); what varies above it, and what the matched difference
# is a mean over, is the cell. Cells are clustered by model -- a model contributes its nih, its chexpert and its coco
# cells, and those are not independent -- so the model is the resampling unit. Recomputing a per-patient interval for a
# cell-level rate would require re-deriving every cell's verdict inside every draw from every block's outcomes parquet;
# this section reads block summaries only and does not claim to do that.
FGOBJ_GROUPS = ("chest", "coco_easy", "coco_fine")


def _cell_group(dataset: str, kind: str) -> str:
    return "chest" if dataset in CHEST else ("coco_fine" if kind == "fgobj" else "coco_easy")


def fgobj_cells(blocks: dict) -> tuple[list, list]:
    """One row per pooled cell: every chest cell, every easy COCO cell (both from summary.json core + calibration) and
    every fine-grained COCO cell (summary.json fgobj). A cell carries its ownership flag and the two matching
    variables -- clean-answer AUROC and probe selectivity -- measured on the block's own calibration rows by the
    campaign's rule for every group alike."""
    cells, skipped = [], []
    for (mk, ds), b in blocks.items():
        s = b["summary"]
        co, cal = s.get("core"), s.get("calibration")
        if not co or not cal:
            skipped.append({"block": f"{mk}/{ds}", "reason": "summary.json carries no core or no calibration"}); continue
        for q in CONCEPTS[ds]:
            c, g = co["per_question"].get(q), cal.get(q)
            if c is None or g is None:
                skipped.append({"block": f"{mk}/{ds}", "reason": f"{q}: missing from core or calibration"}); continue
            cells.append({"block": f"{mk}/{ds}", "model": mk, "dataset": ds, "concept": q, "kind": "core",
                          "group": _cell_group(ds, "core"), "owned": owned(c), "verdict": c.get("verdict"),
                          "steering_reference": c.get("steering_reference"), "W_qq": c.get("W_qq"), "O_q": c.get("O_q"),
                          "answer_auroc": g.get("answer_auroc"), "selectivity": g.get("selectivity"),
                          "readable": g.get("readable"), "answer_capable": g.get("answer_capable"),
                          "n_pos": g.get("n_pos"), "n_neg": g.get("n_neg")})
        if ds != "coco":
            continue
        f = s.get("fgobj")
        if not f:
            if completed(b, "FGOBJ"):
                skipped.append({"block": f"{mk}/{ds}", "reason": "FGOBJ completed but summary.json carries no fgobj; "
                                                                 "rerun python -m cftransfer.analysis --what fgobj"})
            continue
        if not completed(b, "FGOBJ"):
            skipped.append({"block": f"{mk}/{ds}", "reason": "FGOBJ not in run.json completed_modules"}); continue
        if f.get("n_scored_rows_min") != f.get("n_rows"):
            skipped.append({"block": f"{mk}/{ds}", "reason": f"fgobj scored on {f.get('n_scored_rows_min')} of {f.get('n_rows')} rows"}); continue
        if list(f["concepts"]) != list(FGOBJ_CONCEPTS):
            raise RuntimeError(f"{mk}/{ds} fgobj: concepts {f['concepts']} are not {list(FGOBJ_CONCEPTS)}")
        if not f["probe_grade_source"]["answer_available"]:
            skipped.append({"block": f"{mk}/{ds}", "reason": "FGOBJ_CALIBRATION not scored: the fine-grained cells carry "
                                                             "no clean-answer AUROC and cannot be pooled"}); continue
        for q in FGOBJ_CONCEPTS:
            c, p = f["per_question"][q], f["probes"][q]
            cells.append({"block": f"{mk}/{ds}", "model": mk, "dataset": ds, "concept": q, "kind": "fgobj",
                          "group": _cell_group(ds, "fgobj"), "owned": bool(c["owned"]), "verdict": c.get("verdict"),
                          "steering_reference": c.get("steering_reference"), "W_qq": c.get("W_qq"), "O_q": c.get("O_q"),
                          "answer_auroc": p.get("answer_auroc"), "selectivity": p.get("selectivity"),
                          "readable": p.get("readable"), "answer_capable": p.get("answer_capable"),
                          "n_pos": p.get("n_pos"), "n_neg": p.get("n_neg")})
    return cells, skipped


def _matchable(cells: list) -> list:
    return [c for c in cells if c["answer_auroc"] is not None and c["selectivity"] is not None
            and np.isfinite(c["answer_auroc"]) and np.isfinite(c["selectivity"])]


def match_matrix(chest: list, natural: list, window_auroc: float | None, window_selectivity: float | None) -> np.ndarray:
    """(n_chest, n_natural) boolean: a natural-image cell matches a chest cell when every ACTIVE matching variable is
    within its window. A window of None switches that variable off (the single-variable matchings)."""
    ca = np.array([c["answer_auroc"] for c in chest], float)[:, None]
    cs = np.array([c["selectivity"] for c in chest], float)[:, None]
    na = np.array([c["answer_auroc"] for c in natural], float)[None, :]
    ns = np.array([c["selectivity"] for c in natural], float)[None, :]
    M = np.ones((len(chest), len(natural)), bool)
    if window_auroc is not None:
        M &= np.abs(ca - na) <= window_auroc
    if window_selectivity is not None:
        M &= np.abs(cs - ns) <= window_selectivity
    return M


def _matched_estimate(M: np.ndarray, owned_c: np.ndarray, owned_n: np.ndarray, w_c: np.ndarray, w_n: np.ndarray):
    """(matched difference, pooled difference, weight of the matched chest cells). The matched difference is the mean
    over matched chest cells of (its own ownership minus the ownership RATE OF ITS OWN PARTNER SET), which is the
    estimator a variable-sized match set calls for; the pooled difference is the ownership rate among matched chest
    cells minus the rate among the natural-image cells that served as a partner at least once."""
    den = M @ w_n
    ok = den > 0
    if not ok.any() or w_c[ok].sum() <= 0:
        return None, None, 0.0
    num = M @ (w_n * owned_n)
    p = num[ok] / den[ok]
    wc = w_c[ok]
    matched = float((wc * (owned_c[ok] - p)).sum() / wc.sum())
    used = (M[ok].T @ wc) > 0
    pooled = None
    if used.any() and (w_n * used).sum() > 0:
        pooled = float((wc * owned_c[ok]).sum() / wc.sum() - (w_n * used * owned_n).sum() / (w_n * used).sum())
    return matched, pooled, float(wc.sum())


def matched_comparison(chest: list, natural: list, window_auroc, window_selectivity, name: str, draws: int = FGOBJ_BOOT_DRAWS) -> dict:
    """The difficulty-matched chest-versus-natural-image ownership difference under one matching, with a cluster
    bootstrap over models."""
    if not chest or not natural:
        return {"matching": name, "n_chest_cells": len(chest), "n_natural_cells": len(natural),
                "status": "NO_CELLS", "matched_ownership_difference": None}
    M = match_matrix(chest, natural, window_auroc, window_selectivity)
    owned_c = np.array([float(c["owned"]) for c in chest])
    owned_n = np.array([float(c["owned"]) for c in natural])
    one_c, one_n = np.ones(len(chest)), np.ones(len(natural))
    est, pooled, _ = _matched_estimate(M, owned_c, owned_n, one_c, one_n)
    matched = M.any(axis=1)
    used = M[matched].any(axis=0) if matched.any() else np.zeros(len(natural), bool)
    models = sorted({c["model"] for c in chest} | {c["model"] for c in natural})
    mi_c = np.array([models.index(c["model"]) for c in chest])
    mi_n = np.array([models.index(c["model"]) for c in natural])
    rng = np.random.Generator(np.random.PCG64(FGOBJ_BOOT_SEED))
    draws_est, draws_pooled = [], []
    for _ in range(draws):
        mult = np.bincount(rng.integers(0, len(models), len(models)), minlength=len(models)).astype(float)
        e, p, wsum = _matched_estimate(M, owned_c, owned_n, mult[mi_c], mult[mi_n])
        if e is not None and wsum > 0:
            draws_est.append(e)
        if p is not None:
            draws_pooled.append(p)
    partners = M.sum(axis=1)
    comp = {g: int(sum(1 for j, c in enumerate(natural) if used[j] and c["group"] == g)) for g in ("coco_easy", "coco_fine")}
    return {
        "matching": name, "window_answer_auroc": window_auroc, "window_selectivity": window_selectivity,
        "n_chest_cells": len(chest), "n_natural_cells": len(natural),
        "n_matched_chest_cells": int(matched.sum()), "n_unmatched_chest_cells": int((~matched).sum()),
        "unmatched_chest_cells": [f"{c['block']} {c['concept']}" for c, mm in zip(chest, matched) if not mm][:40],
        "n_natural_cells_used": int(used.sum()), "partner_composition_used": comp,
        "median_partners_per_matched_chest_cell": float(np.median(partners[matched])) if matched.any() else None,
        "chest_owned_rate_matched": float(owned_c[matched].mean()) if matched.any() else None,
        "natural_owned_rate_used": float(owned_n[used].mean()) if used.any() else None,
        "matched_ownership_difference": est,
        "matched_ownership_difference_ci95": [float(np.percentile(draws_est, 2.5)), float(np.percentile(draws_est, 97.5))] if draws_est else None,
        "pooled_ownership_difference": pooled,
        "pooled_ownership_difference_ci95": [float(np.percentile(draws_pooled, 2.5)), float(np.percentile(draws_pooled, 97.5))] if draws_pooled else None,
        "bootstrap_valid_draws": len(draws_est), "bootstrap_draws": draws,
        "bootstrap_unit": "model (cluster bootstrap; a model carries its nih, chexpert and coco cells)"}


def fgobj_section(blocks: dict) -> dict:
    """Fine-grained COCO object concepts, and the difficulty-matched chest-versus-natural-image comparison they make
    possible.

    Per block with FGOBJ completed and the six-direction write matrix scored on every test row: the fine-grained cells'
    ownership inside their own family, their probe readability and clean-answer capability from the block's calibration
    rows, and the block's own easy-COCO CORE cells beside them. Pooled: ownership rate, median clean-answer AUROC and
    median probe selectivity for chest cells, easy COCO cells and fine-grained COCO cells; then the matched comparison
    under three matchings (both variables, clean-answer AUROC alone, probe selectivity alone), plus the same matching
    restricted to the easy COCO cells so the starvation FGOBJ is meant to remove is visible next to the result."""
    cells, skipped = fgobj_cells(blocks)
    rows = []
    for (mk, ds), b in blocks.items():
        f = b["summary"].get("fgobj")
        if ds != "coco" or not f or not any(c["block"] == f"{mk}/{ds}" and c["kind"] == "fgobj" for c in cells):
            continue
        fine = [c for c in cells if c["block"] == f"{mk}/{ds}" and c["kind"] == "fgobj"]
        easy = [c for c in cells if c["block"] == f"{mk}/{ds}" and c["kind"] == "core"]
        rows.append({"block": f"{mk}/{ds}", "model": mk, "dataset": ds, "n_rows": f["n_rows"], "template_id": f.get("template_id"),
                     "alpha": f.get("alpha"), "draws": f.get("draws"), "max_t_critical": f.get("max_t_critical"),
                     "random_seed": f.get("random_seed"), "n_random": f.get("n_random"),
                     "fine_cells": len(fine), "fine_owned": sum(c["owned"] for c in fine),
                     "easy_cells": len(easy), "easy_owned": sum(c["owned"] for c in easy),
                     "fine_readable": sum(bool(c["readable"]) for c in fine),
                     "fine_answer_capable": sum(bool(c["answer_capable"]) for c in fine),
                     "median_answer_auroc_fine": med([c["answer_auroc"] for c in fine]),
                     "median_answer_auroc_easy": med([c["answer_auroc"] for c in easy]),
                     "median_selectivity_fine": med([c["selectivity"] for c in fine]),
                     "median_selectivity_easy": med([c["selectivity"] for c in easy]),
                     "median_abs_cos_to_easy": f.get("median_abs_cos_to_easy"),
                     "per_concept": {c["concept"]: {k: c[k] for k in ("owned", "verdict", "steering_reference", "W_qq", "O_q",
                                                                      "answer_auroc", "selectivity", "readable",
                                                                      "answer_capable", "n_pos", "n_neg")} for c in fine}})
    pool = {}
    for g in FGOBJ_GROUPS:
        gc = [c for c in cells if c["group"] == g]
        pool[g] = {"cells": len(gc), "blocks": len({c["block"] for c in gc}), "owned": sum(c["owned"] for c in gc),
                   "owned_rate": (sum(c["owned"] for c in gc) / len(gc)) if gc else None,
                   "median_answer_auroc": med([c["answer_auroc"] for c in gc]),
                   "median_selectivity": med([c["selectivity"] for c in gc]),
                   "readable": sum(bool(c["readable"]) for c in gc),
                   "answer_capable": sum(bool(c["answer_capable"]) for c in gc),
                   "with_both_matching_variables": len(_matchable(gc))}
    chest = _matchable([c for c in cells if c["group"] == "chest"])
    natural = _matchable([c for c in cells if c["group"] in ("coco_easy", "coco_fine")])
    easy_only = [c for c in natural if c["group"] == "coco_easy"]
    W, S = FGOBJ_WINDOW_ANSWER_AUROC, FGOBJ_WINDOW_SELECTIVITY
    matched = {
        "both": matched_comparison(chest, natural, W, S, "clean-answer AUROC and probe selectivity"),
        "answer_auroc": matched_comparison(chest, natural, W, None, "clean-answer AUROC alone"),
        "selectivity": matched_comparison(chest, natural, None, S, "probe selectivity alone"),
        # the pre-FGOBJ state, for reference only: the same matching with the six EASY COCO concepts as the only partners
        "both_easy_partners_only": matched_comparison(chest, easy_only, W, S, "both variables, easy COCO partners only")}
    return {"concepts": list(FGOBJ_CONCEPTS), "blocks": rows, "pool": pool, "matched": matched,
            "cells": cells, "skipped": skipped,
            "prespecification": f"the matching windows ({W} clean-answer AUROC, {S} probe selectivity) and the six "
                                f"fine-grained categories were fixed before the FGOBJ grid was built; no ownership "
                                f"outcome enters either choice",
            "matching_variables": {"clean_answer_auroc": "summary.json calibration answer_auroc for chest and easy COCO "
                                                         "cells, summary.json fgobj probes answer_auroc (module "
                                                         "FGOBJ_CALIBRATION) for fine-grained cells -- the same rows, rule "
                                                         "and estimator in both cases",
                                   "probe_selectivity": "probe AUROC minus the mean of the 20 type->random-label control "
                                                        "AUROCs on the calibration rows, the campaign's definition for "
                                                        "every group"}}


# ------------------------------------------------------------------------------------------------ PRECISION
def precision_section(blocks: dict) -> dict:
    rows, skipped = [], []
    for (mk, ds), b in blocks.items():
        p = b["summary"].get("precision")
        if not p:
            continue
        rec = {"block": f"{mk}/{ds}", "model": mk, "dataset": ds, "n_rows": p["n_rows"], "draws": p.get("draws"), "settings": {}}
        for st in PRECISION_SETTINGS:
            r = p.get(st) or {}
            if r.get("status") != "COMPLETE" or not r.get("grade"):
                skipped.append({"block": rec["block"], "setting": st, "reason": f"status {r.get('status')}, grade {'present' if r.get('grade') else 'missing'}"}); continue
            g = r["grade"]; gq = g["per_question"]
            own_ch = sum(owned({"steering_reference": c["steering_reference"], "verdict": c["verdict"]})
                         != owned({"steering_reference": c["steering_reference_core"], "verdict": c["verdict_core"]}) for c in gq.values())
            rec["settings"][st] = {"n_cells": len(gq), "n_verdict_changes": g["n_verdict_changes"],
                                   "n_steering_reference_changes": g["n_steering_reference_changes"], "n_owned_changes": int(own_ch),
                                   "n_grade_changes": int(sum(bool(c["verdict_changed"] or c["steering_reference_changed"]) for c in gq.values())),
                                   "owned_core": sum(owned({"steering_reference": c["steering_reference_core"], "verdict": c["verdict_core"]}) for c in gq.values()),
                                   "owned_setting": sum(owned(c) for c in gq.values()),
                                   "max_abs_dW_grid": g["max_abs_dW_grid"], "n_grid_cells_compared": g["n_grid_cells_compared"],
                                   "max_abs_dcontrast": g["max_abs_dcontrast"], "max_abs_dO_q_grade": max(abs(c["dO_q"]) for c in gq.values() if c.get("dO_q") is not None),
                                   "max_abs_dW_clinical": r["max_abs_dW"], "max_abs_dO_point": r["max_abs_dO"],
                                   "n_agree_competitor": r["n_agree_competitor"], "n_agree_random_p95": r["n_agree_random_p95"]}
        if rec["settings"]:
            rows.append(rec)
    graded = [(r, st, v) for r in rows for st, v in r["settings"].items()]
    agg = {"blocks": len(rows), "block_settings": len(graded), "graded_cells": sum(v["n_cells"] for _, _, v in graded),
           "verdict_changes": sum(v["n_verdict_changes"] for _, _, v in graded),
           "steering_reference_changes": sum(v["n_steering_reference_changes"] for _, _, v in graded),
           "owned_changes": sum(v["n_owned_changes"] for _, _, v in graded),
           "grade_changes": sum(v["n_grade_changes"] for _, _, v in graded),
           "max_abs_dW_grid": max((v["max_abs_dW_grid"] for _, _, v in graded), default=None),
           "max_abs_dW_grid_at": max(((f"{r['block']}/{st}", v["max_abs_dW_grid"]) for r, st, v in graded), key=lambda x: x[1], default=(None,))[0],
           "max_abs_dcontrast": max((v["max_abs_dcontrast"] for _, _, v in graded), default=None),
           "blocks_list": [r["block"] for r in rows]}
    return {"settings": list(PRECISION_SETTINGS), "blocks": rows, "aggregate": agg, "skipped": skipped}


# ------------------------------------------------------------------------------------------------ report
def write_md(R: dict, path: Path) -> None:
    m, V, A, T, P, B = R["meta"], R["valid"], R["altdird"], R["ansdirt"], R["precision"], R["attr"]
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
    L = ["# Round-2 robustness: VALID, ALTDIRD, ANSDIRT, ATTR, PRECISION", "",
         f"Generated {m['generated_utc']} from `{m['run_root']}`; {m['n_included_blocks']} blocks under the inclusion rule. "
         "Owned = steering reference and fixed-family advantage.", ""]
    a = V["aggregate"]
    L += ["## VALID: radiologist-labelled CheXpert valid rows vs labeler-labelled test rows", "",
          f"{a['blocks']} blocks, {a['cells']} cells, valid rows {a['valid_rows']}. Owned agreement {a['owned_agree']}/{a['cells']}; verdict agreement "
          f"{a['verdict_agree']}/{a['cells']}; owned on test {a['owned_test']} (paper grade {a['owned_paper']}), on valid {a['owned_valid']}; "
          f"median |dO_q| {f3(a['median_abs_dO_q'])}, p90 {f3(a['p90_abs_dO_q'])}. With >=10/10 support on both cohorts ({a['supported_cells']} cells): "
          f"readable agreement {a['readable_agree_supported']}/{a['readable_compared_supported']}, answer-capable agreement "
          f"{a['answer_capable_agree_supported']}/{a['answer_capable_compared_supported']}.", "",
          "| block | owned test | owned valid | verdict agree | owned agree | median abs dO |", "|---|---|---|---|---|---|"]
    for r in V["blocks"]:
        L.append(f"| {r['block']} | {r['owned_test']} | {r['owned_valid']} | {r['verdict_agree']}/{r['n_cells']} | {r['owned_agree']}/{r['n_cells']} | {f3(r['median_abs_dO_q'])} |")
    L += ["", "Owned disagreements:", ""] + [f"- {d['block']} {d['concept']}: test owned {d['owned_test']} (O {d['O_q_test']:+.4f}, {d['verdict_test']}, ref {d['reference_test']}) "
                                            f"-> valid owned {d['owned_valid']} (O {d['O_q_valid']:+.4f}, {d['verdict_valid']}, ref {d['reference_valid']})" for d in a["disagreements"]]
    L += ["", "## ALTDIRD: displacement-lifted difference of means and Haufe pattern", "",
          "| group | blocks | cells | logistic | dom_disp | pattern_disp | med cos dom | med cos pattern | cond. min-max | full rank |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g, d in A["per_dataset"].items():
        gs = d["gram_spectrum"] or {}
        L.append(f"| {g} | {d['blocks']} | {d['cells']} | {d['logistic']['owned']} | {d['dom_disp']['owned']} | {d['pattern_disp']['owned']} | "
                 f"{f3(d['dom_disp']['median_cos_to_logistic'])} | {f3(d['pattern_disp']['median_cos_to_logistic'])} | "
                 f"{f3(gs.get('condition_min'))}-{f3(gs.get('condition_max'))} | {gs.get('all_full_rank')} |")
    L += ["", "## ANSDIRT: the IY answer direction under the held-out templates", "",
          "| block | IY owned | " + " | ".join(f"{t} owned (kept)" for t in ANSDIRT_TEMPLATES) + " | kept pairs |", "|---|---|" + "---|" * (len(ANSDIRT_TEMPLATES) + 1)]
    for r in T["blocks"]:
        L.append(f"| {r['block']} | {r['n_iy_owned']} | " + " | ".join(f"{r['templates'][t]['n_owned']} ({r['templates'][t]['kept_of_iy']})" if t in r["templates"] else "--" for t in ANSDIRT_TEMPLATES)
                 + f" | {r['kept_pairs']}/{r['pairs']} |")
    for r in T["partial_blocks"]:
        L.append(f"| {r['block']} ({r['status']}, rows {min(r['n_scored_rows_min'].values())}/{r['n_rows']}; not aggregated) | {r['n_iy_owned']} | "
                 + " | ".join(f"{r['templates'][t]['n_owned']} ({r['templates'][t]['kept_of_iy']})" if t in r["templates"] else "--" for t in ANSDIRT_TEMPLATES)
                 + f" | {r['kept_pairs']}/{r['pairs']} |")
    for g, d in T["aggregate"].items():
        L.append(f"| *{g}* | {d['iy_owned']} | " + " | ".join(f"{d['per_template'][t]['owned']} ({d['per_template'][t]['kept_of_iy']})" for t in ANSDIRT_TEMPLATES) + f" | {d['kept_pairs']}/{d['pairs']} |")
    L += ["", "## ATTR: non-clinical attribute directions inside one nine-direction family", "",
          "| block | attr owned | clinical owned | " + " | ".join(ATTR_CONCEPTS) + " | med abs W attr | med abs W clin | med abs cos |", "|---|---|---|" + "---|" * (len(ATTR_CONCEPTS) + 3)]
    for r in B["blocks"]:
        L.append(f"| {r['block']} | {r['attr_owned']}/{r['attr_cells']} | {r['clin_owned']}/{r['clin_cells']} | "
                 + " | ".join(("owned" if r["attributes"][q]["owned"] else "--") if q in r["attributes"] else "n/a" for q in ATTR_CONCEPTS)
                 + f" | {f3(r['median_abs_W_qq_attr'])} | {f3(r['median_abs_W_qq_clinical'])} | {f3(r['median_abs_cos_attr_clinical'])} |")
    for g, d in B["per_dataset"].items():
        if not d["blocks"]:
            continue
        L.append(f"| *{g}* ({d['blocks']} blocks) | {d['attr_owned']}/{d['attr_cells']} | {d['clin_owned']}/{d['clin_cells']} | "
                 + " | ".join(f"{d['per_attribute'][q]['owned']}/{d['per_attribute'][q]['blocks']}" for q in ATTR_CONCEPTS)
                 + f" | {f3(d['median_abs_W_qq_attr'])} | {f3(d['median_abs_W_qq_clinical'])} | {f3(d['median_abs_cos_attr_clinical'])} |")
    L += ["", "| group | attribute | blocks | owned | probe AUROC | selectivity | answer AUROC | med abs cos to clinical |", "|---|---|---|---|---|---|---|---|"]
    for g, d in B["per_dataset"].items():
        if not d["blocks"]:
            continue
        for q in ATTR_CONCEPTS:
            p = d["per_attribute"][q]
            L.append(f"| {g} | {q} | {p['blocks']} | {p['owned']} | {f3(p['median_auroc_real'])} | {f3(p['median_selectivity'])} | "
                     f"{f3(p['median_answer_auroc'])} | {f3(p['median_abs_cos_to_clinical'])} |")
    ch = B["per_dataset"]["chest"]
    if ch["blocks"]:
        mm = ch["max_median_abs_cos_pair"] or {}
        L += ["", f"Pooled chest: {ch['blocks']} blocks; attributes owned {ch['attr_owned']}/{ch['attr_cells']}, clinical owned "
              f"{ch['clin_owned']}/{ch['clin_cells']} in the same family; attribute probes readable {ch['attr_readable']}/{ch['attr_cells']}, "
              f"answer-capable {ch['attr_answer_capable']}/{ch['attr_cells']}; median |cos| attribute-to-clinical {f3(ch['median_abs_cos_attr_clinical'])} "
              f"(largest pair median {f3(mm.get('median_abs_cos'))} for {mm.get('pair')} over {mm.get('n_blocks')} blocks); "
              f"attribute questions carry a random family: {ch['attribute_random_reference']}."]
    L += ["", "### The attribute-versus-finding comparison, three ways", "",
          f"Window for the answerability match: {B['match_window']} of clean-answer AUROC, inside one block.", "",
          "Two intervals: the patient bootstrap resamples the rows inside a block (it is near-degenerate here, because a "
          "cell's ownership almost never flips when the rows are resampled); the block bootstrap resamples the "
          "(model, dataset) blocks, which is the variation that decides whether attribute and finding cells are owned "
          "at the same rate.", "",
          "| group | mode | attr owned | clinical owned | difference | patient CI | block CI | block CI excludes 0 |",
          "|---|---|---|---|---|---|---|---|"]
    for g, d in B["per_dataset"].items():
        if not d["blocks"]:
            continue
        for key, tag in (("comparison", ""), ("comparison_matched_reference", " (ATTRRAND blocks)")):
            if key == "comparison_matched_reference" and not d["blocks_matched_reference"]:
                continue
            for mode in ("all_cells", "answer_capable_cells", "answerability_matched"):
                m = d[key][mode]
                bt = m.get("bootstrap") or {}
                bk = m.get("block_bootstrap") or {}
                ci = (f"[{bt['difference_ci95_percentile'][0]:+.3f}, {bt['difference_ci95_percentile'][1]:+.3f}]"
                      if bt.get("available") else "n/a")
                cib = (f"[{bk['difference_ci95_percentile'][0]:+.3f}, {bk['difference_ci95_percentile'][1]:+.3f}]"
                       if bk.get("available") else "n/a")
                L.append(f"| {g}{tag} | {mode} | {m['attr_owned']}/{m['attr_cells']} ({f3(m['attr_owned_share'])}) | "
                         f"{m['clin_owned']}/{m['clin_cells']} ({f3(m['clin_owned_share'])}) | {f3(m['difference'])} | {ci} | "
                         f"{cib} | {bk.get('difference_excludes_zero') if bk.get('available') else 'n/a'} |")
    if ch["blocks"] and ch["blocks_matched_reference"]:
        L += ["", f"Attribute and clinical cells are graded against one steering reference only where ATTRRAND has been "
              f"scored: {len(ch['blocks_matched_reference'])} of {ch['blocks']} chest blocks "
              f"({', '.join(ch['blocks_matched_reference'])}). Elsewhere the attribute cells still carry the sham-only "
              f"reference, so the rows without the (ATTRRAND blocks) tag mix two rules."]
    cm = ch["comparison_matched_reference"] if (ch["blocks"] and ch["blocks_matched_reference"]) else \
        (ch["comparison"] if ch["blocks"] else None)
    if cm:
        mt = cm["answerability_matched"]
        L += ["", f"Pooled chest, answerability-matched: {mt['n_pairs']} pairs from {mt['n_attr_cells_matched']} of "
              f"{mt['n_attr_cells']} attribute cells over {cm['blocks']} blocks; {mt['n_attr_cells_unmatched']} attribute "
              f"cells find no clinical cell within the window. Median |AUROC gap| inside a pair "
              f"{f3(mt['median_abs_answer_auroc_gap'])}.",
              f"Answer AUROC rows: attribute {cm['answer_auroc_rows']['attribute']} (median known labels "
              f"{f3(cm['median_answer_n_known']['attribute'])}), clinical {cm['answer_auroc_rows']['clinical']} "
              f"(median known labels {f3(cm['median_answer_n_known']['clinical'])})."]
        for u in mt["unmatched"]:
            n0 = u.get("nearest_clinical") or {}
            L.append(f"- unmatched: {u['block']} {u['attribute']} (answer AUROC {f3(u['answer_auroc'])}); nearest clinical "
                     f"{n0.get('concept')} at {f3(n0.get('answer_auroc'))} (gap {f3(n0.get('gap'))}) -- {u['reason']}")
    L += ["", "## PRECISION: full grade under fp32 and batch size one (200 rows)", "",
          "| block | setting | verdict changes | reference changes | owned changes | max abs dW grid | max abs d contrast |", "|---|---|---|---|---|---|---|"]
    for r in P["blocks"]:
        for st, v in r["settings"].items():
            L.append(f"| {r['block']} | {st} | {v['n_verdict_changes']}/{v['n_cells']} | {v['n_steering_reference_changes']}/{v['n_cells']} | {v['n_owned_changes']}/{v['n_cells']} | "
                     f"{v['max_abs_dW_grid']:.4f} | {v['max_abs_dcontrast']:.4f} |")
    pa = P["aggregate"]
    L += ["", f"Pooled: {pa['blocks']} blocks, {pa['block_settings']} block-settings, {pa['graded_cells']} graded cells; verdict changes {pa['verdict_changes']}, "
          f"reference changes {pa['steering_reference_changes']}, owned changes {pa['owned_changes']}; max |dW| grid {f3(pa['max_abs_dW_grid'])} ({pa['max_abs_dW_grid_at']}); "
          f"max |d contrast| {f3(pa['max_abs_dcontrast'])}."]
    G = R.get("fgobj")
    if G:
        L += ["", "## FGOBJ: fine-grained COCO object concepts, and the difficulty-matched comparison", "",
              "Six COCO categories that are small, often occluded or fine-grained ("
              + ", ".join(G["concepts"]) + "), built from the same instance annotations by the same rule, fitted with the "
              "same projection, scaler and probe settings on the same training rows, and written as a self-contained "
              "127-condition family (own baseline, six directions, own 119 random directions, own sham) at the primary "
              "template and dose. " + G["prespecification"][0].upper() + G["prespecification"][1:] + ".", "",
              "| block | fine owned | easy owned | fine readable | fine answerable | med answer AUROC fine / easy | med selectivity fine / easy |",
              "|---|---|---|---|---|---|---|"]
        for r in G["blocks"]:
            L.append(f"| {r['block']} | {r['fine_owned']}/{r['fine_cells']} | {r['easy_owned']}/{r['easy_cells']} | "
                     f"{r['fine_readable']}/{r['fine_cells']} | {r['fine_answer_capable']}/{r['fine_cells']} | "
                     f"{f3(r['median_answer_auroc_fine'])} / {f3(r['median_answer_auroc_easy'])} | "
                     f"{f3(r['median_selectivity_fine'])} / {f3(r['median_selectivity_easy'])} |")
        L += ["", "| group | blocks | cells | owned | rate | median clean-answer AUROC | median probe selectivity |",
              "|---|---|---|---|---|---|---|"]
        for g in FGOBJ_GROUPS:
            d = G["pool"][g]
            L.append(f"| {g} | {d['blocks']} | {d['cells']} | {d['owned']} | {f3(d['owned_rate'])} | "
                     f"{f3(d['median_answer_auroc'])} | {f3(d['median_selectivity'])} |")
        L += ["", "Difficulty-matched chest-versus-natural-image ownership difference. Every chest cell is paired with "
              "the natural-image cells of ANY block inside the prespecified window; the difference is the mean over "
              "matched chest cells of its own ownership minus the ownership rate of its own partner set. The interval "
              "is a cluster bootstrap over models (a cell's ownership is already a 600-patient statistic; the model is "
              "the unit that clusters cells).", "",
              "| matching | matched chest cells | unmatched | partners used (easy / fine) | chest rate | partner rate | difference | 95% CI |",
              "|---|---|---|---|---|---|---|---|"]
        for key in ("both", "answer_auroc", "selectivity", "both_easy_partners_only"):
            m = G["matched"][key]
            ci = m.get("matched_ownership_difference_ci95")
            comp = m.get("partner_composition_used") or {}
            L.append(f"| {m['matching']} | {m.get('n_matched_chest_cells')}/{m.get('n_chest_cells')} | "
                     f"{m.get('n_unmatched_chest_cells')} | {comp.get('coco_easy')} / {comp.get('coco_fine')} | "
                     f"{f3(m.get('chest_owned_rate_matched'))} | {f3(m.get('natural_owned_rate_used'))} | "
                     f"{f3(m.get('matched_ownership_difference'))} | "
                     + (f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |" if ci else "n/a |"))
    L += ["", "## Skipped", ""]
    for sec in ("valid", "altdird", "ansdirt", "attr", "precision", "fgobj"):
        if sec not in R:
            continue
        for s in R[sec]["skipped"]:
            L.append(f"- {sec}: {s['block']}{' ' + s['setting'] if s.get('setting') else ''}: {s['reason']}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(RUN_ROOT))
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    t0 = time.time()
    blocks, not_included = discover(Path(a.run_root))
    R = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_root": a.run_root,
                  "n_included_blocks": len(blocks), "included_blocks": [f"{mk}/{ds}" for mk, ds in blocks], "not_included": not_included,
                  "rules": {"owned": "steering_reference and verdict == fixed_family_advantage (cftransfer.manifest.owned)",
                            "support": f">= {SUPPORT_MIN} positives and >= {SUPPORT_MIN} negatives on the valid rows and on the calibration rows",
                            "valid": "CheXpert, VALID in run.json completed_modules",
                            "altdird": "ALTDIRD completed; dom_disp and pattern_disp scored on every test row",
                            "ansdirt": "every eligible held-out template scored on every test row; IY grade present",
                            "attr": "ATTR completed; the nine-direction write matrix scored on every test row",
                            "precision": "setting status COMPLETE with the full grade"},
                  "script": str(Path(__file__).resolve())},
         "valid": valid_section(blocks), "altdird": altdird_section(blocks), "ansdirt": ansdirt_section(blocks),
         "attr": attr_section(blocks), "precision": precision_section(blocks), "fgobj": fgobj_section(blocks)}
    R["meta"]["rules"]["fgobj"] = ("FGOBJ completed; the six-direction write matrix scored on every test row; "
                                   "FGOBJ_CALIBRATION scored, so the fine-grained cells carry a clean-answer AUROC "
                                   "measured on the same calibration rows and by the same rule as every other cell")
    R["meta"]["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "round2.json").write_text(json.dumps(clean(R), indent=1), encoding="utf-8")
    write_md(R, out / "round2.md")
    v, d, t, p, bt = R["valid"]["aggregate"], R["altdird"]["per_dataset"], R["ansdirt"]["aggregate"], R["precision"]["aggregate"], R["attr"]["per_dataset"]
    print(f"valid: {v['blocks']} blocks, owned agree {v['owned_agree']}/{v['cells']}, verdict agree {v['verdict_agree']}/{v['cells']}, "
          f"owned test {v['owned_test']} valid {v['owned_valid']}, median|dO| {v['median_abs_dO_q']:.4f} p90 {v['p90_abs_dO_q']:.4f}")
    print("altdird: " + "; ".join(f"{g} {x['blocks']}b dom {x['dom_disp']['owned']}/{x['cells']} pattern {x['pattern_disp']['owned']}/{x['cells']} logistic {x['logistic']['owned']}"
                                  for g, x in d.items() if g in DATASETS))
    print("ansdirt: " + "; ".join(f"{g} {x['blocks']}b kept {x['kept_pairs']}/{x['pairs']}" for g, x in t.items()))
    print("attr: " + "; ".join(f"{g} {x['blocks']}b attributes {x['attr_owned']}/{x['attr_cells']} clinical {x['clin_owned']}/{x['clin_cells']}"
                               for g, x in bt.items() if x["blocks"]))
    for g, x in bt.items():
        if not x["blocks"]:
            continue
        for key, tag in (("comparison", ""), ("comparison_matched_reference", "+ATTRRAND")):
            if key == "comparison_matched_reference" and not x["blocks_matched_reference"]:
                continue
            _print_comparison(g + tag, x[key])
    print(f"precision: {p['blocks']} blocks, {p['block_settings']} settings, verdict changes {p['verdict_changes']}, reference changes {p['steering_reference_changes']}, max|dW| grid {p['max_abs_dW_grid']}")
    G = R["fgobj"]
    print("fgobj: " + "; ".join(f"{g} {G['pool'][g]['cells']} cells {G['pool'][g]['owned']} owned "
                                f"(rate {G['pool'][g]['owned_rate'] if G['pool'][g]['owned_rate'] is None else round(G['pool'][g]['owned_rate'], 3)}, "
                                f"median answer AUROC {G['pool'][g]['median_answer_auroc'] if G['pool'][g]['median_answer_auroc'] is None else round(G['pool'][g]['median_answer_auroc'], 3)})"
                                for g in FGOBJ_GROUPS))
    for key in ("both", "answer_auroc", "selectivity", "both_easy_partners_only"):
        m = G["matched"][key]
        ci = m.get("matched_ownership_difference_ci95")
        d = m.get("matched_ownership_difference")
        print(f"  fgobj-matched {key:26s} matched {m.get('n_matched_chest_cells')}/{m.get('n_chest_cells')} chest cells "
              f"(unmatched {m.get('n_unmatched_chest_cells')}), partners used {m.get('partner_composition_used')}, "
              f"d={None if d is None else round(d, 4)} "
              + (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "no interval"))
    for sec in ("valid", "altdird", "ansdirt", "attr", "precision", "fgobj"):
        for s in R[sec]["skipped"]:
            print(f"  skipped {sec}: {s['block']} {s.get('setting', '')} {s['reason']}")
    print(f"wrote {out / 'round2.json'} and round2.md in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
