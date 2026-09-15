"""Benchmark view of the cf-transfer-v1 results: one row per (checkpoint, dataset) with the three graded
capabilities the protocol measures, computed from the packaged statistics with the protocol's own rules.

  readable        known-label probe AUROC exceeds the type->random-label controls: one-sided 95% lower bound of the
                  selectivity S = AUROC_real - mean(AUROC_control) is > 0, with >= 10 positives and negatives and
                  >= 1900 valid bootstrap draws (analysis.calibration).
  answer-capable  the model's clean answer on the block's primary template (IY, or IB when IY failed the image-free
                  preflight; summary.json "primary_template") separates positives from negatives: one-sided 95%
                  lower bound of the answer AUROC > 0.5 (analysis.calibration).
  owned           the concept write is specific: W_qq > 0, above the 95th percentile of the 119 random-direction
                  writes and above |sham| (steering reference), and every max-T simultaneous lower bound of
                  W_qq - W_qd over the five competitors is > 0 (verdict "fixed_family_advantage"; analysis.core).

Blocks whose templates are INELIGIBLE by preflight check E are reported with the disposition instead of a count; a
block scored on a non-IY primary template is graded on that template and the template is shown in an extra column.
Blocks with ALTDIR statistics (summary.json "altdir") add one CSV column, altdir_owned_dom_pattern_orth_resid: the
owned count per alternative direction family under the same rule (steering reference + fixed_family_advantage within
the family); the column is absent when no block has the key, and the markdown table ignores it.
Outputs runs/leaderboard.csv and runs/leaderboard.md.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .protocol import CONCEPTS, MODELS
from .runpaths import RUN_ROOT as RUNS_ROOT

DATASETS = ("nih", "chexpert", "coco")
COLUMNS = ["model_key", "dataset_id", "status", "modules_complete", "modules_ineligible", "n_concepts",
           "readable", "answer_capable", "owned", "owned_concepts", "readable_concepts", "capable_concepts",
           "eligible_templates", "primary_template", "gpu_hours"]
ALTDIR_COL = "altdir_owned_dom_pattern_orth_resid"          # only written when some block carries summary["altdir"]
ALTDIR_FAMILIES = ("dom", "pattern", "orth", "resid")


def grade_block(summary: dict, run: dict | None, eligibility: dict | None) -> dict:
    """Apply the protocol's rules to one packaged block; every count is over the concepts present in the summary."""
    cal = {c: v for c, v in (summary.get("calibration") or {}).items() if isinstance(v, dict)}
    core = (summary.get("core") or {}).get("per_question") or {}
    readable = sorted(c for c, v in cal.items() if v.get("readable"))
    capable = sorted(c for c, v in cal.items() if v.get("answer_capable"))
    owned = sorted(c for c, v in core.items()
                   if v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage")
    elig = "" if not eligibility else "".join(t for t, v in eligibility.items() if v.get("eligible"))
    ineligible = (run or {}).get("ineligible_modules") or []
    primary = summary.get("primary_template") or (run or {}).get("primary_template") or "IY"
    row = {
        "status": (run or {}).get("status", "?"),
        "modules_complete": "+".join((run or {}).get("completed_modules") or []),
        "modules_ineligible": "+".join(ineligible),
        "n_concepts": len(cal) if cal else len(core),
        "readable": len(readable), "answer_capable": len(capable) if any("answer_capable" in v for v in cal.values()) else None,
        "owned": None if "CORE" in ineligible else len(owned),
        "owned_concepts": ", ".join(owned), "readable_concepts": ", ".join(readable), "capable_concepts": ", ".join(capable),
        "eligible_templates": elig, "primary_template": primary, "gpu_hours": (run or {}).get("gpu_hours"),
    }
    alt = summary.get("altdir")
    if alt:
        counts = []
        for fam in alt.get("families") or ALTDIR_FAMILIES:
            pq_ = (alt.get(fam) or {}).get("per_question") or {}
            counts.append(str(sum(1 for v in pq_.values() if v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage")))
        row[ALTDIR_COL] = "/".join(counts)
    return row


def collect(runs_root: Path = RUNS_ROOT) -> list[dict]:
    rows = []
    for mk in MODELS:
        for ds in DATASETS:
            d = runs_root / mk / ds
            s = d / "summary.json"
            if not s.exists():
                continue
            summary = json.loads(s.read_text())
            run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else None
            elig = json.loads((d / "template_eligibility.json").read_text()) if (d / "template_eligibility.json").exists() else None
            if not summary.get("calibration") and not summary.get("core"):
                continue
            rows.append({"model_key": mk, "dataset_id": ds, **grade_block(summary, run, elig)})
    return rows


def _cell(r: dict) -> str:
    if r["owned"] is None:
        return f"{r['readable']}/-/INELIGIBLE"
    cap = "-" if r["answer_capable"] is None else r["answer_capable"]
    return f"{r['readable']}/{cap}/{r['owned']}"


def render_markdown(rows: list[dict]) -> str:
    by = {(r["model_key"], r["dataset_id"]): r for r in rows}
    models = [m for m in MODELS if any((m, ds) in by for ds in DATASETS)]
    # the primary-template column appears only when some block was scored on a template other than IY
    show_tpl = any(r.get("primary_template", "IY") != "IY" for r in rows)
    out = ["| checkpoint | NIH read/answer/own | CheXpert read/answer/own | COCO read/answer/own | owned clinical concepts |"
           + (" primary template |" if show_tpl else ""),
           "|---|---|---|---|---|" + ("---|" if show_tpl else "")]
    for m in models:
        cells, owned, tpl = [], [], []
        for ds in DATASETS:
            r = by.get((m, ds))
            cells.append(_cell(r) if r else "-")
            if r and ds != "coco" and r["owned_concepts"]:
                owned.append(f"{ds}: {r['owned_concepts']}")
            if r and r.get("primary_template", "IY") != "IY":
                tpl.append(f"{ds}: {r['primary_template']}")
        out.append(f"| {m} | {cells[0]} | {cells[1]} | {cells[2]} | {'; '.join(owned)} |"
                   + (f" {'; '.join(tpl)} |" if show_tpl else ""))
    out.append("")
    out.append(f"Cells are counts over the {len(CONCEPTS['nih'])} concepts of a dataset: readable at the consumed visual block / "
               "answer-capable on the clean yes/no question / owned by the concept write (protocol rules in "
               "`cftransfer.leaderboard`). INELIGIBLE marks blocks whose yes/no template fails the image-free "
               "semantic-mapping preflight, where ownership is not defined."
               + (" The primary-template column lists blocks whose IY template failed that preflight and which were "
                  "scored on the eligible B-positive A/B template instead (IB); answer capability and ownership "
                  "there refer to that template." if show_tpl else ""))
    return "\n".join(out)


def main(runs_root: Path = RUNS_ROOT) -> Path:
    rows = collect(runs_root)
    csv_path = runs_root / "leaderboard.csv"
    fields = COLUMNS + ([ALTDIR_COL] if any(ALTDIR_COL in r for r in rows) else [])
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (runs_root / "leaderboard.md").write_text(render_markdown(rows) + "\n")
    print(f"{csv_path} {len(rows)} blocks")
    return csv_path


if __name__ == "__main__":
    main()
