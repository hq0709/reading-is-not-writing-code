"""Fill main-tables.csv rows from per-run summaries (T1/T2 from CALIBRATION + CORE; T3 from PROMPT/DOSE/REFIT/LOCUS).

Writes <RUN_ROOT>/main-tables.filled.csv: the package template with every cell the executor can compute from
its own returned artefacts; cells still TBD keep execution_status NOT_STARTED/RUNNING. The project owner
recomputes everything from the raw files; this copy is the executor's cross-check.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .protocol import CONCEPTS, PKG, PROMPT_CONCEPTS
from .runpaths import RUN_ROOT, run_dir


def fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fill() -> Path:
    rows = list(csv.DictReader((PKG / "main-tables.csv").open(newline="", encoding="utf-8")))
    cache: dict[tuple[str, str], dict] = {}

    def summary(m, d):
        if (m, d) not in cache:
            p = run_dir(m, d) / "summary.json"
            cache[(m, d)] = json.loads(p.read_text()) if p.exists() else {}
        return cache[(m, d)]

    for r in rows:
        m, d, c, metric = r["model_key"], r["dataset_id"], r["concept"], r["metric"]
        s = summary(m, d)
        cal, core = s.get("calibration"), s.get("core")
        primary = s.get("primary_template", "IY")        # template the block's single-template statistics refer to
        if r["table_id"] == "T1":
            if metric == "calibration_selectivity" and cal and c in cal:
                cell = cal[c]
                r["estimate"] = fmt(cell.get("selectivity"))
                ci = cell.get("selectivity_ci95")
                r["ci_low"], r["ci_high"] = (fmt(ci[0]), fmt(ci[1])) if ci else ("", "")
                r["execution_status"] = "COMPLETE" if cell.get("readable_status") != "insufficient_support" else "INELIGIBLE"
            elif metric == "test_ownership_O" and core and c in core.get("per_question", {}):
                cell = core["per_question"][c]
                r["estimate"] = fmt(cell.get("O_q"))
                ci = cell.get("O_q_ci95_percentile")
                r["ci_low"], r["ci_high"] = (fmt(ci[0]), fmt(ci[1])) if ci else ("", "")
                r["execution_status"] = "COMPLETE"
        elif r["table_id"] == "T2" and (cal or core):
            concepts = CONCEPTS[d]
            if metric == "readable_count" and cal:
                n = sum(1 for k in concepts if cal.get(k, {}).get("readable"))
                r["estimate"], r["numerator"], r["denominator"], r["execution_status"] = f"{n}/6", n, 6, "COMPLETE"
            elif metric == "answer_capable_count" and cal and all("answer_capable" in cal.get(k, {}) for k in concepts):
                n = sum(1 for k in concepts if cal[k].get("answer_capable"))
                r["estimate"], r["numerator"], r["denominator"], r["execution_status"] = f"{n}/6", n, 6, "COMPLETE"
            elif metric == "steering_reference_count" and core:
                n = sum(1 for k in concepts if core["per_question"].get(k, {}).get("steering_reference"))
                r["estimate"], r["numerator"], r["denominator"], r["execution_status"] = f"{n}/6", n, 6, "COMPLETE"
            elif metric == "competitor_wins_among_readable" and cal and core:
                readable = [k for k in concepts if cal.get(k, {}).get("readable")]
                n = sum(1 for k in readable if core["per_question"].get(k, {}).get("verdict") == "stronger_competitor")
                r["estimate"] = f"{n}/{len(readable)}" if readable else "NA"
                r["numerator"], r["denominator"], r["execution_status"] = n, len(readable), "COMPLETE"
        elif r["table_id"] == "T3":
            t3 = s.get("t3", {})
            key = f"{c}|{metric}"
            if key in t3:
                cell = t3[key]
                r["estimate"] = fmt(cell.get("estimate"))
                r["ci_low"], r["ci_high"] = fmt(cell.get("ci_low")), fmt(cell.get("ci_high"))
                r["execution_status"] = cell.get("status", "COMPLETE")
            elif metric in ("wording_IY_minus_WY_O", "mapping_IA_minus_IB_O") and primary != "IY" and (cal or core):
                # the PROMPT contrasts are defined against IY; a block scored on another primary template has no such cell
                r["execution_status"] = "INELIGIBLE"
    out = RUN_ROOT / "main-tables.filled.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return out


if __name__ == "__main__":
    p = fill()
    done = [r for r in csv.DictReader(p.open()) if r["execution_status"] == "COMPLETE"]
    print(p, f"{len(done)} cells filled")
    for r in done[:20]:
        print(" ", r["table_id"], r["model_key"], r["dataset_id"], r["concept"], r["metric"], r["estimate"], r["ci_low"], r["ci_high"])
