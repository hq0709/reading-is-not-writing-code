"""Campaign manifest: one row per coverage cell of every block, joined with the block's eligibility, status,
primary template, calibration grades, and core verdicts, plus THE inclusion rule of the paper.

    python -m cftransfer.manifest [--runs RUN_ROOT] [--out RUN_ROOT/manifest.csv]

Writes <RUN_ROOT>/manifest.csv (the only file this module writes under the run root) with one row per
(model_key, dataset, module, concept, template, fit_seed) of every block's coverage.csv, and prints a
per-dataset summary of the counts the paper reports.

Inclusion rule (ONE rule, used by every paper script through scripts/cf_inclusion.py in the paper repository):

    block_included  =  "CORE" in run.json completed_modules  and  "CALIBRATION" in run.json completed_modules

i.e. the write matrix was scored on the primary template and the calibration module completed. Only included
blocks contribute write-matrix counts (answerable, owned, stronger competitor, steering reference, effect floor,
median W_qq). A second, explicitly named flag

    probe_graded    =  "CALIBRATION" in run.json completed_modules

marks blocks whose probes were graded (readability) even when CORE is ineligible or incomplete; the paper uses
it only where it says "probe-graded" (the Read column of Table 1 and the readable count of Figure 2a). Blocks
whose CORE is ineligible or incomplete carry no core verdict fields and no answer grades in the manifest, so no
consumer can count them.

Cell rows: the concept cells of a block are its CORE rows with fit_seed 0 (one per concept; CORE scores one
template per block, the primary one). Calibration grades attach to those rows for probe-graded blocks; core
verdict fields and answer grades attach only for included blocks. ALTDIR rows carry the per-family ownership
flags of the alternative-direction module for included blocks whose ALTDIR is COMPLETE for that concept. VALID rows
carry the grade of the same concept on the radiologist-labelled CheXpert valid rows (summary.json `valid`: readable,
answer-capable, owned, O_q, verdict) for included blocks whose VALID is COMPLETE for that concept; block_included
never depends on VALID.

The three control modules added for the reviewer's remaining requests attach the same way, and block_included depends on
none of them: ATTRRAND rows (one per attribute question) carry the attribute cell's matched-reference grade from
summary.json `attr` (the random p95 that ATTRRAND supplies, the sham-only verdict it replaces, and which reference the
cell used); VALIDFIT rows carry the expert-label refit's ownership and its cosine / held-out AUROC against the
report-label direction (summary.json `validfit`); PROJSEED rows carry one ownership flag and O_q per further projection
seed (summary.json `projseed`).

The three modules added for the two reviewer objections about the reader and about the endpoints attach the same way,
and block_included depends on none of them. TOWERSWAP rows carry the CROSSED arm's grade (this block's reader with the
partner checkpoint's vision tower) next to the native arm's ownership on the same rows and the paired dW_qq, from
summary.json `towerswap`. REPLAY rows carry the grade obtained when the consumed block is the shared-tower group's
stored tensor, next to the block's own CORE grade on those rows and the drift the replacement removed, from
summary.json `replay`. SEMEND rows are per (question, endpoint) and carry that endpoint's signed effect for both the label and the answer
family, the own random / sham reference, the five-competitor contrast and verdict, and the spillover of the direction onto
the other five concepts' endpoints (intended, unintended, selectivity), from summary.json `semend`.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from .protocol import ALTDIR_FAMILIES, ATTR_CONCEPTS, DATASETS, PROJSEED_SEEDS, SEMEND_TEMPLATES, TOWERSWAP_PAIRS
from .protocol import primary_template as protocol_primary_template
from .runpaths import RUN_ROOT

INCLUSION_MODULES = ("CORE", "CALIBRATION")
PROBE_MODULE = "CALIBRATION"
EFFECT_FLOOR = 0.05          # effect floor applied to W_qq and O_q for the "effect-floored owned" count
CHEST = ("nih", "chexpert")

CAL_FIELDS = ("readable", "n_pos", "n_neg", "selectivity")
ANS_FIELDS = ("answer_capable", "answer_auroc")
CORE_FIELDS = ("W_qq", "O_q", "verdict", "steering_reference")
VALID_FIELDS = ("readable", "answer_capable", "owned", "O_q", "verdict")
ATTRRAND_FIELDS = ("random_p95", "steering_reference", "steering_reference_sham_only", "steering_reference_rule", "O_q", "verdict")
VALIDFIT_FIELDS = ("owned", "O_q", "verdict", "steering_reference", "cos_to_report_model", "cos_to_report_whitened",
                   "auroc_expert_heldout", "auroc_report_labels_heldout")
TOWERSWAP_FIELDS = ("tower", "reader", "W_qq", "O_q", "verdict", "steering_reference", "owned", "owned_native",
                    "owned_changed", "dW_qq", "reader_over_tower", "crossover_complete")
REPLAY_FIELDS = ("source_block", "W_qq", "O_q", "verdict", "steering_reference", "owned", "owned_core", "owned_changed",
                 "dW_qq", "drift_max_abs")
SEMEND_FIELDS = ("sign", "value", "competitor", "E_own", "E_max_other", "E_sham", "random_max", "rank_in_random_family",
                 "O_q", "verdict", "steering_reference", "owned", "owned_core", "intended", "unintended_mean",
                 "unintended_max", "selectivity", "answer_E_own", "answer_owned", "answer_selectivity")
COVERAGE_FIELDS = ("expected_rows", "actual_unique_rows", "failed_rows", "execution_status", "reason")

COLUMNS = (["model_key", "dataset", "module", "locus_id", "fit_seed", "template", "concept"]
           + list(COVERAGE_FIELDS)
           + ["template_eligible", "template_reason", "primary_template", "is_primary_template",
              "run_status", "completed_modules", "ineligible_modules", "requested_modules",
              "block_included", "probe_graded", "cell"]
           + list(CAL_FIELDS) + list(ANS_FIELDS) + list(CORE_FIELDS) + ["owned"]
           + [f"altdir_owned_{f}" for f in ALTDIR_FAMILIES] + [f"altdir_O_q_{f}" for f in ALTDIR_FAMILIES]
           + [f"valid_{f}" for f in VALID_FIELDS]
           + [f"attrrand_{f}" for f in ATTRRAND_FIELDS] + [f"validfit_{f}" for f in VALIDFIT_FIELDS]
           + [f"projseed_owned_seed{k}" for k in PROJSEED_SEEDS] + [f"projseed_O_q_seed{k}" for k in PROJSEED_SEEDS]
           + [f"towerswap_{f}" for f in TOWERSWAP_FIELDS] + [f"replay_{f}" for f in REPLAY_FIELDS]
           + [f"semend_{f}" for f in SEMEND_FIELDS])


# ------------------------------------------------------------------------------------------------ the rule
def block_included(run: dict) -> bool:
    """THE inclusion rule: run.json completed_modules contains both CORE and CALIBRATION."""
    done = set(run.get("completed_modules") or [])
    return all(m in done for m in INCLUSION_MODULES)


def probe_graded(run: dict) -> bool:
    """The block's probes were graded (CALIBRATION completed), whether or not its write matrix was scored."""
    return PROBE_MODULE in set(run.get("completed_modules") or [])


def owned(cell: dict) -> bool:
    """The paper's ownership verdict: steering reference met and fixed-family advantage under max-T bounds."""
    return bool(cell.get("steering_reference") and cell.get("verdict") == "fixed_family_advantage")


# ------------------------------------------------------------------------------------------------ loading
def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def resolve_primary_template(run: dict, summary: dict, run_dir: Path) -> str:
    """run.json > summary.json > template_eligibility.json (protocol fallback order) > IY."""
    return run.get("primary_template") or summary.get("primary_template") or protocol_primary_template(run_dir)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "" if v != v else repr(v)
    return str(v)


def block_rows(run_dir: Path) -> list[dict]:
    """Manifest rows of one block: every coverage.csv row, joined with the block-level facts and the grades."""
    run = _json(run_dir / "run.json")
    summary = _json(run_dir / "summary.json")
    elig = _json(run_dir / "template_eligibility.json")
    cov_path = run_dir / "coverage.csv"
    if not run or not cov_path.exists():
        return []
    mk, ds = run["model_key"], run["dataset_id"]
    included, probe = block_included(run), probe_graded(run)
    primary = resolve_primary_template(run, summary, run_dir)
    cal = {c: v for c, v in (summary.get("calibration") or {}).items() if isinstance(v, dict)}
    core = ((summary.get("core") or {}).get("per_question") or {}) if included else {}
    alt = summary.get("altdir") or {}
    alt_ok = included and "ALTDIR" in set(run.get("completed_modules") or []) and all(f in alt for f in ALTDIR_FAMILIES)
    val = (summary.get("valid") or {}).get("per_question") or {}
    done = set(run.get("completed_modules") or [])
    val_ok = included and "VALID" in done
    # attribute cells of summary.json `attr` (ATTRRAND supplies their random p95), the expert-label refit grade, and the
    # per-projection-seed grades; each attaches only to its own module's rows and never enters block_included
    att = ((summary.get("attr") or {}).get("per_question") or {})
    att_ok = included and "ATTRRAND" in done and ((summary.get("attr") or {}).get("attrrand") or {}).get("available") is True
    vfit = ((summary.get("validfit") or {}).get("per_question") or {})
    vfit_ok = included and "VALIDFIT" in done
    pseed = summary.get("projseed") or {}
    pseed_ok = included and "PROJSEED" in done
    # the crossed TOWERSWAP arm of this block, the REPLAY arm and the SEMEND endpoints; none enters block_included
    tsw = summary.get("towerswap") or {}
    tsw_cross = (tsw.get("combinations") or {}).get(f"tower={TOWERSWAP_PAIRS.get(mk)}|reader={mk}") or {}
    tsw_ok = included and "TOWERSWAP" in done and tsw_cross.get("status") == "COMPLETE"
    rep = summary.get("replay") or {}
    rep_ok = included and "REPLAY" in done and bool(rep.get("replay"))
    sem = summary.get("semend") or {}
    sem_ok = included and "SEMEND" in done and bool(sem.get("per_endpoint"))
    block = {"primary_template": primary, "run_status": run.get("status", ""),
             "completed_modules": "|".join(run.get("completed_modules") or []),
             "ineligible_modules": "|".join(run.get("ineligible_modules") or []),
             "requested_modules": "|".join(run.get("requested_modules") or []),
             "block_included": included, "probe_graded": probe}
    rows = []
    with cov_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            template, concept, seed = r.get("template_id", ""), r.get("concept", ""), r.get("fit_seed", "")
            e = elig.get(template) if template else None
            row = {"model_key": mk, "dataset": ds, "module": r["module"], "locus_id": r.get("locus_id", ""),
                   "fit_seed": seed, "template": template, "concept": concept}
            row.update({k: r.get(k, "") for k in COVERAGE_FIELDS})
            row["template_eligible"] = (e.get("eligible", True) if isinstance(e, dict) else True) if template else None
            row["template_reason"] = (e.get("reason", "") if isinstance(e, dict) else "") if template else ""
            row.update(block)
            row["is_primary_template"] = (template == primary) if template else None
            cell = r["module"] == "CORE" and seed == "0" and bool(concept)
            row["cell"] = cell
            for k in CAL_FIELDS + ANS_FIELDS + CORE_FIELDS + ("owned",):
                row[k] = None
            for fam in ALTDIR_FAMILIES:
                row[f"altdir_owned_{fam}"] = None; row[f"altdir_O_q_{fam}"] = None
            for k in VALID_FIELDS:
                row[f"valid_{k}"] = None
            for k in ATTRRAND_FIELDS:
                row[f"attrrand_{k}"] = None
            for k in VALIDFIT_FIELDS:
                row[f"validfit_{k}"] = None
            for k in PROJSEED_SEEDS:
                row[f"projseed_owned_seed{k}"] = None; row[f"projseed_O_q_seed{k}"] = None
            for k in TOWERSWAP_FIELDS:
                row[f"towerswap_{k}"] = None
            for k in REPLAY_FIELDS:
                row[f"replay_{k}"] = None
            for k in SEMEND_FIELDS:
                row[f"semend_{k}"] = None
            if cell and probe and concept in cal:
                for k in CAL_FIELDS:
                    row[k] = cal[concept].get(k)
                if included:
                    for k in ANS_FIELDS:
                        row[k] = cal[concept].get(k)
            if cell and included and concept in core:
                for k in CORE_FIELDS:
                    row[k] = core[concept].get(k)
                row["owned"] = owned(core[concept])
            if r["module"] == "ALTDIR" and seed == "0" and concept and alt_ok and r.get("execution_status") == "COMPLETE":
                for fam in ALTDIR_FAMILIES:
                    c = (alt[fam].get("per_question") or {}).get(concept)
                    if c:
                        row[f"altdir_owned_{fam}"] = owned(c); row[f"altdir_O_q_{fam}"] = c.get("O_q")
            if r["module"] == "VALID" and seed == "0" and concept in val and val_ok and r.get("execution_status") == "COMPLETE":
                for k in VALID_FIELDS:
                    row[f"valid_{k}"] = val[concept].get(k)
            if r["module"] == "ATTRRAND" and concept in ATTR_CONCEPTS and concept in att and att_ok \
                    and r.get("execution_status") == "COMPLETE":
                for k in ATTRRAND_FIELDS:
                    row[f"attrrand_{k}"] = att[concept].get(k)
            if r["module"] == "VALIDFIT" and seed == "0" and concept in vfit and vfit_ok and r.get("execution_status") == "COMPLETE":
                for k in VALIDFIT_FIELDS:
                    row[f"validfit_{k}"] = vfit[concept].get(k)
            if r["module"] == "PROJSEED" and pseed_ok and r.get("execution_status") == "COMPLETE":
                for k in PROJSEED_SEEDS:
                    c = ((pseed.get(f"seed{k}") or {}).get("per_question") or {}).get(concept)
                    if c and str(seed) == str(k):
                        row[f"projseed_owned_seed{k}"] = c.get("owned"); row[f"projseed_O_q_seed{k}"] = c.get("O_q")
            if r["module"] == "TOWERSWAP" and seed == "0" and concept and tsw_ok and r.get("execution_status") == "COMPLETE":
                g = (tsw_cross.get("grade") or {}).get(concept) or {}
                d = (tsw.get("swapped_minus_native") or {}).get(concept) or {}
                row.update({"towerswap_tower": tsw_cross.get("tower"), "towerswap_reader": tsw_cross.get("reader"),
                            "towerswap_W_qq": g.get("W_qq"), "towerswap_O_q": g.get("O_q"),
                            "towerswap_verdict": g.get("verdict"), "towerswap_steering_reference": g.get("steering_reference"),
                            "towerswap_owned": g.get("owned"), "towerswap_owned_native": d.get("owned_native"),
                            "towerswap_owned_changed": d.get("owned_changed"), "towerswap_dW_qq": d.get("estimate"),
                            "towerswap_reader_over_tower": (tsw.get("crossover") or {}).get("reader_over_tower"),
                            "towerswap_crossover_complete": tsw.get("crossover_complete")})
            if r["module"] == "REPLAY" and seed == "0" and concept and rep_ok and r.get("execution_status") == "COMPLETE":
                g = ((rep.get("replay") or {}).get("grade") or {}).get(concept) or {}
                c = ((rep.get("core") or {}).get("grade") or {}).get(concept) or {}
                d = (rep.get("replay_minus_core") or {}).get(concept) or {}
                row.update({"replay_source_block": rep.get("source_block"), "replay_W_qq": g.get("W_qq"),
                            "replay_O_q": g.get("O_q"), "replay_verdict": g.get("verdict"),
                            "replay_steering_reference": g.get("steering_reference"), "replay_owned": g.get("owned"),
                            "replay_owned_core": c.get("owned"), "replay_owned_changed": d.get("owned_changed"),
                            "replay_dW_qq": d.get("estimate"), "replay_drift_max_abs": (rep.get("drift") or {}).get("max_abs")})
            if r["module"] == "SEMEND" and concept and template in SEMEND_TEMPLATES and sem_ok \
                    and r.get("execution_status") == "COMPLETE":
                blk = (sem.get("per_endpoint") or {}).get(template) or {}
                fams = blk.get("families") or {}
                c = ((fams.get("label") or {}).get("per_question") or {}).get(concept) or {}
                a = ((fams.get("answer") or {}).get("per_question") or {}).get(concept) or {}
                if c:
                    row.update({"semend_sign": blk.get("sign"), "semend_value": blk.get("value"),
                                "semend_competitor": c.get("competitor"), "semend_E_own": c.get("W_qq"),
                                "semend_E_max_other": c.get("max_other"), "semend_E_sham": c.get("E_sham"),
                                "semend_random_max": c.get("random_max"),
                                "semend_rank_in_random_family": c.get("rank_in_random_family"),
                                "semend_O_q": c.get("O_q"), "semend_verdict": c.get("verdict"),
                                "semend_steering_reference": c.get("steering_reference"), "semend_owned": c.get("owned"),
                                "semend_owned_core": c.get("owned_core"), "semend_intended": c.get("intended"),
                                "semend_unintended_mean": c.get("unintended_mean"),
                                "semend_unintended_max": c.get("unintended_max"),
                                "semend_selectivity": c.get("selectivity"), "semend_answer_E_own": a.get("W_qq"),
                                "semend_answer_owned": a.get("owned"), "semend_answer_selectivity": a.get("selectivity")})
            rows.append(row)
    return rows


def build(root: Path = RUN_ROOT) -> list[dict]:
    rows = []
    for run_json in sorted(root.glob("*/*/run.json")):
        rows += block_rows(run_json.parent)
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in COLUMNS})


# ------------------------------------------------------------------------------------------------ summary
def _truth(v) -> bool:
    return v is True or v == "true"


def _num(v):
    return None if v in (None, "") else float(v)


def summarize(rows: list[dict]) -> dict:
    """The counts the paper reports, per dataset, plus 'chest' (NIH + CheXpert) and 'all'. Works on rows as
    built (Python values) or as read back from manifest.csv (strings)."""
    cells = [r for r in rows if _truth(r["cell"])]
    out = {}
    groups = {ds: [ds] for ds in DATASETS}; groups["chest"] = list(CHEST); groups["all"] = list(DATASETS)
    for name, dss in groups.items():
        c = [r for r in cells if r["dataset"] in dss]
        blocks = {(r["model_key"], r["dataset"]) for r in rows if r["dataset"] in dss}
        probe = [r for r in c if _truth(r["probe_graded"]) and r["readable"] not in (None, "")]
        wm = [r for r in c if _truth(r["block_included"])]
        ra = [r for r in wm if _truth(r["readable"]) and _truth(r["answer_capable"])]
        own = [r for r in wm if _truth(r["owned"])]
        ref = [r for r in wm if _truth(r["steering_reference"])]
        s = {"blocks": len(blocks),
             "probe_blocks": len({(r["model_key"], r["dataset"]) for r in probe}),
             "write_blocks": len({(r["model_key"], r["dataset"]) for r in wm}),
             "probe_cells": len(probe), "readable": sum(_truth(r["readable"]) for r in probe),
             "write_cells": len(wm), "answerable": sum(_truth(r["answer_capable"]) for r in wm),
             "owned": len(own), "stronger_competitor": sum(r["verdict"] == "stronger_competitor" for r in wm),
             "reference_met": len(ref), "reference_met_not_owned": len(ref) - len(own),
             "readable_and_answerable": len(ra), "owned_among_readable_and_answerable": sum(_truth(r["owned"]) for r in ra),
             "rest": len(wm) - len(ra), "owned_among_rest": len(own) - sum(_truth(r["owned"]) for r in ra),
             "effect_floored_owned": sum(_num(r["W_qq"]) >= EFFECT_FLOOR and _num(r["O_q"]) >= EFFECT_FLOOR for r in own),
             "median_W_qq": statistics.median(_num(r["W_qq"]) for r in wm) if wm else None}
        out[name] = s
    return out


def print_summary(summary: dict, file=sys.stdout) -> None:
    keys = ["blocks", "probe_blocks", "write_blocks", "probe_cells", "readable", "write_cells", "answerable", "owned",
            "stronger_competitor", "reference_met", "reference_met_not_owned", "readable_and_answerable",
            "owned_among_readable_and_answerable", "rest", "owned_among_rest", "effect_floored_owned", "median_W_qq"]
    print("block_included = CORE and CALIBRATION in run.json completed_modules; probe_graded = CALIBRATION completed", file=file)
    print(f"{'count':38s}" + "".join(f"{n:>10s}" for n in summary), file=file)
    for k in keys:
        vals = []
        for n in summary:
            v = summary[n][k]
            vals.append(f"{v:10.3f}" if isinstance(v, float) else f"{v:10d}" if v is not None else f"{'--':>10s}")
        print(f"{k:38s}" + "".join(vals), file=file)


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=RUN_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="default: <runs>/manifest.csv")
    a = ap.parse_args(argv)
    rows = build(a.runs)
    out = a.out or (a.runs / "manifest.csv")
    write_csv(rows, out)
    s = summarize(rows)
    print(f"{out}: {len(rows)} rows, {s['all']['blocks']} blocks, {s['all']['write_blocks']} included, "
          f"{s['all']['write_cells']} write-matrix cells, {s['all']['probe_cells']} probe-graded cells")
    print_summary(s)
    return out


if __name__ == "__main__":
    main()
