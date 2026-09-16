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
             answer AUROC / |W_qq|, and the median |cosine| between attribute and clinical directions. Attribute questions carry no
             random family (random_reference false): their steering reference is the sham alone.
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
from cftransfer.manifest import block_included, owned                                          # noqa: E402
from cftransfer.protocol import (ALTDIRD_FAMILIES, ANSDIRT_TEMPLATES, ATTR_CONCEPTS, CONCEPTS, DATASETS, MODEL_ORDER,  # noqa: E402
                                 PRECISION_SETTINGS)
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
                    "random_reference": c["random_reference"], "abs_sham": c["abs_sham"]}
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
    per = {}
    for g, dss in GROUPS.items():
        bs = [r for r in rows if r["dataset"] in dss]
        ac = [c for r in bs for c in r["attributes"].values()]
        cc = [c for r in bs for c in r["clinical"].values()]
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
    return {"attributes": list(ATTR_CONCEPTS), "blocks": rows, "per_dataset": per, "skipped": skipped}


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
    L += ["", "## PRECISION: full grade under fp32 and batch size one (200 rows)", "",
          "| block | setting | verdict changes | reference changes | owned changes | max abs dW grid | max abs d contrast |", "|---|---|---|---|---|---|---|"]
    for r in P["blocks"]:
        for st, v in r["settings"].items():
            L.append(f"| {r['block']} | {st} | {v['n_verdict_changes']}/{v['n_cells']} | {v['n_steering_reference_changes']}/{v['n_cells']} | {v['n_owned_changes']}/{v['n_cells']} | "
                     f"{v['max_abs_dW_grid']:.4f} | {v['max_abs_dcontrast']:.4f} |")
    pa = P["aggregate"]
    L += ["", f"Pooled: {pa['blocks']} blocks, {pa['block_settings']} block-settings, {pa['graded_cells']} graded cells; verdict changes {pa['verdict_changes']}, "
          f"reference changes {pa['steering_reference_changes']}, owned changes {pa['owned_changes']}; max |dW| grid {f3(pa['max_abs_dW_grid'])} ({pa['max_abs_dW_grid_at']}); "
          f"max |d contrast| {f3(pa['max_abs_dcontrast'])}.", "", "## Skipped", ""]
    for sec in ("valid", "altdird", "ansdirt", "attr", "precision"):
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
         "attr": attr_section(blocks), "precision": precision_section(blocks)}
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
    print(f"precision: {p['blocks']} blocks, {p['block_settings']} settings, verdict changes {p['verdict_changes']}, reference changes {p['steering_reference_changes']}, max|dW| grid {p['max_abs_dW_grid']}")
    for sec in ("valid", "altdird", "ansdirt", "attr", "precision"):
        for s in R[sec]["skipped"]:
            print(f"  skipped {sec}: {s['block']} {s.get('setting', '')} {s['reason']}")
    print(f"wrote {out / 'round2.json'} and round2.md in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
