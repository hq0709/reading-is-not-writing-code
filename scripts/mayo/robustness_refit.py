#!/usr/bin/env python
"""Full ownership grade under the refit seeds: does a cell's grade survive refitting the six probes?

For every block under the paper's inclusion rule (cftransfer.manifest.block_included) that carries a merged
outcomes/REFIT.parquet, the complete ownership grade is recomputed under fit seeds 1 and 2 with the campaign rules:

  W^k_{q,d}   mean over the 600 test rows of p(q | write of the seed-k direction d) - p(q | CORE clean baseline)
  reference   steering reference under seed k: W^k_{q,q} > 0, above the CORE random-family 95th percentile and above
              the CORE |sham| of the same block, question and rows (the summary's random_p95 / abs_sham, random_n = 119)
  verdict     6 x 5 contrasts C_{q,d} = W^k_{q,q} - W^k_{q,d} with simultaneous max-T bounds over the campaign's shared
              unit-bootstrap draws (core_bootstrap_indices, seed BOOT_CORE_SEED, 2,000 draws): stronger_competitor if
              any upper bound < 0, fixed_family_advantage if every lower bound > 0, else unresolved
  grade       owned = reference and fixed_family_advantage; advantage_no_reference = fixed_family_advantage without
              the reference; stronger_competitor; unresolved

The seed-0 grade is the paper's (summary.json core.per_question); the same code path also regrades seed 0 with 2,000
draws as a cross-check (the campaign scored it with BOOT_CORE_DRAWS draws), and the agreement is reported.

Writes <RUN_ROOT>/robustness/refit.json (per block / seed / question values, per-dataset transition tables
grade(seed 0) x grade(seed k) for k = 1, 2 and pooled, and the survival counts) and refit.md. Nothing under the
block directories is modified.

Usage (from the concept-flow repo):
  PYTHONPATH=src python scripts/mayo/robustness_refit.py [--draws 2000] [--jobs 4] [--out <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from cftransfer.analysis import _load_module, _matrix, core_bootstrap_indices, max_t          # noqa: E402
from cftransfer.manifest import block_included                                                # noqa: E402
from cftransfer.manifest import owned as summary_owned                                        # noqa: E402
from cftransfer.protocol import BOOT_CORE_SEED, CONCEPTS, DATASETS, MODEL_ORDER, N_RANDOM, PRIMARY_ALPHA, primary_template  # noqa: E402
from cftransfer.runpaths import RUN_ROOT, run_dir                                             # noqa: E402

OUT_DIR = RUN_ROOT / "robustness"
GRADES = ("owned", "advantage_no_reference", "stronger_competitor", "unresolved")
SEEDS = (1, 2)


def test_order(rd: Path) -> dict[str, int]:
    rows = [r for r in csv.DictReader((rd / "manifests" / "cohort.csv").open(newline="", encoding="utf-8")) if r["role"] == "test"]
    rows.sort(key=lambda r: int(r["order"]))
    return {r["row_id"]: i for i, r in enumerate(rows)}


def grade_of(reference: bool, verdict: str) -> str:
    if verdict == "fixed_family_advantage":
        return "owned" if reference else "advantage_no_reference"
    return verdict


def regrade(delta: dict, concepts: list[str], ref: dict, idx: np.ndarray) -> dict:
    """Campaign grade of one 6x6 delta surface against the block's CORE references (random p95, |sham| per question)."""
    W = {q: {d: float(np.nanmean(delta[(q, f"concept:{d}")])) for d in concepts} for q in concepts}
    names, est, mat = [], [], []
    for q in concepts:
        dq = delta[(q, f"concept:{q}")]
        for d in concepts:
            if d == q:
                continue
            c = dq - delta[(q, f"concept:{d}")]
            names.append(f"{q}-{d}"); est.append(float(np.nanmean(c))); mat.append(c)
    mat = np.stack(mat)                                                                    # (30, n)
    boot = np.stack([np.nanmean(mat[:, idx[b]], axis=1) for b in range(idx.shape[0])])     # (B, 30)
    mt = max_t(np.array(est), boot)
    out = {}
    for qi, q in enumerate(concepts):
        others = {d: W[q][d] for d in concepts if d != q}
        w = W[q][q]
        r = ref[q]
        reference = bool(r["random_n"] == N_RANDOM and w > 0 and w > r["random_p95"] and w > r["abs_sham"])
        lows, highs = mt["lower"][qi * 5:(qi + 1) * 5], mt["upper"][qi * 5:(qi + 1) * 5]
        verdict = "stronger_competitor" if (highs < 0).any() else ("fixed_family_advantage" if (lows > 0).all() else "unresolved")
        Oq = boot[:, qi * 5:(qi + 1) * 5].min(axis=1)
        out[q] = {"W_qq": w, "O_q": w - max(others.values()), "max_other": max(others.values()), "argmax_other": max(others, key=others.get),
                  "O_q_ci95_percentile": [float(np.percentile(Oq, 2.5)), float(np.percentile(Oq, 97.5))],
                  "steering_reference": reference, "verdict": verdict, "grade": grade_of(reference, verdict), "W": W[q]}
    return out


def analyse_block(model_key: str, dataset_id: str, draws: int) -> dict:
    t0 = time.time()
    rd = run_dir(model_key, dataset_id)
    rec = {"model": model_key, "dataset": dataset_id}
    run = json.loads((rd / "run.json").read_text())
    if not block_included(run):
        rec["status"] = "NOT_INCLUDED"; return rec
    refit = _load_module(model_key, dataset_id, "REFIT")
    if refit is None or refit.empty:
        rec["status"] = "NO_REFIT"; return rec
    sm = json.loads((rd / "summary.json").read_text())
    primary = sm.get("primary_template") or primary_template(rd)
    concepts = CONCEPTS[dataset_id]
    order = test_order(rd); n = len(order)
    core = _load_module(model_key, dataset_id, "CORE")
    core = core[(core.template_id == primary)]
    base = core[(core.direction_id == "baseline") & (core.fit_seed == 0)]
    refit = refit[refit.template_id == primary]
    pq0 = sm["core"]["per_question"]
    ref = {q: {"random_p95": pq0[q]["random_p95"], "abs_sham": pq0[q]["abs_sham"], "random_n": pq0[q]["random_n"]} for q in concepts}
    idx = core_bootstrap_indices(dataset_id, n, draws)
    rec.update({"primary_template": primary, "n_rows": n, "draws": draws, "seeds": [], "per_seed": {}})
    # seed 0: the paper's grade, plus a regrade with the same draws as a cross-check
    d0 = _matrix(core, concepts, order, PRIMARY_ALPHA, template_id=primary, fit_seed=0, baseline=base)
    g0 = regrade(d0, concepts, ref, idx)
    paper = {q: {"grade": grade_of(bool(pq0[q].get("steering_reference")), pq0[q]["verdict"]), "verdict": pq0[q]["verdict"],
                 "steering_reference": bool(pq0[q].get("steering_reference")), "O_q": pq0[q]["O_q"], "W_qq": pq0[q]["W_qq"]} for q in concepts}
    rec["seed0_paper"] = paper
    rec["seed0_regrade"] = {q: {k: v for k, v in g0[q].items() if k != "W"} for q in concepts}
    rec["seed0_regrade_agrees"] = {q: g0[q]["grade"] == paper[q]["grade"] for q in concepts}
    for k in SEEDS:
        dk = _matrix(refit, concepts, order, PRIMARY_ALPHA, template_id=primary, fit_seed=k, baseline=base)
        if any((q, f"concept:{d}") not in dk for q in concepts for d in concepts):
            continue
        n_min = int(min((~np.isnan(dk[(q, f"concept:{d}")])).sum() for q in concepts for d in concepts))
        if n_min < n:                              # the refit module is still running: every cell needs all rows
            rec["incomplete_seed"] = {"seed": k, "n_scored_rows_min": n_min}; continue
        gk = regrade(dk, concepts, ref, idx)
        rec["seeds"].append(k)
        rec["per_seed"][str(k)] = {"n_scored_rows_min": n_min, "per_question": gk}
    rec["status"] = ("OK" if len(rec["seeds"]) == len(SEEDS) else
                     "INCOMPLETE_REFIT" if rec.get("incomplete_seed") else ("PARTIAL" if rec["seeds"] else "NO_REFIT"))
    if rec["status"] == "INCOMPLETE_REFIT":
        rec["reason"] = f"REFIT seed {rec['incomplete_seed']['seed']} scored on {rec['incomplete_seed']['n_scored_rows_min']} of {n} rows"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def _job(args):
    try:
        return analyse_block(*args)
    except Exception as e:                       # noqa: BLE001
        return {"model": args[0], "dataset": args[1], "status": "ERROR", "reason": repr(e)}


def transitions(blocks: list[dict]) -> dict:
    ok = [b for b in blocks if b.get("status") == "OK"]
    out = {}
    groups = {ds: [b for b in ok if b["dataset"] == ds] for ds in DATASETS}
    groups["chest"] = [b for b in ok if b["dataset"] in ("nih", "chexpert")]; groups["all"] = ok
    for name, bs in groups.items():
        cells = [(b, q) for b in bs for q in CONCEPTS[b["dataset"]]]
        T = {str(k): {g0: {g1: 0 for g1 in GRADES} for g0 in GRADES} for k in SEEDS}; T["pooled"] = {g0: {g1: 0 for g1 in GRADES} for g0 in GRADES}
        surv = {g: {"n": 0, "both": 0, "any": 0, "none": 0} for g in GRADES}
        for b, q in cells:
            g0 = b["seed0_paper"][q]["grade"]
            gk = {k: b["per_seed"][str(k)]["per_question"][q]["grade"] for k in SEEDS}
            for k in SEEDS:
                T[str(k)][g0][gk[k]] += 1; T["pooled"][g0][gk[k]] += 1
            same = [gk[k] == g0 for k in SEEDS]
            surv[g0]["n"] += 1; surv[g0]["both"] += all(same); surv[g0]["any"] += any(same); surv[g0]["none"] += not any(same)
        owned_pos = sum(all(b["per_seed"][str(k)]["per_question"][q]["O_q"] > 0 for k in SEEDS) for b, q in cells if b["seed0_paper"][q]["grade"] == "owned")
        out[name] = {"n_blocks": len(bs), "n_cells": len(cells), "transitions": T, "survival": surv,
                     "owned_O_pos_both_seeds": owned_pos,
                     "seed0_regrade_agrees": sum(b["seed0_regrade_agrees"][q] for b, q in cells),
                     "blocks": [f"{b['model']}/{b['dataset']}" for b in bs]}
    return out


def write_md(R: dict, path: Path) -> None:
    m = R["meta"]
    L = ["# Full ownership grade under the refit seeds", "",
         f"Generated {m['generated_utc']} from `{m['run_root']}`; {m['n_blocks']} blocks with REFIT under the inclusion rule "
         f"({m['n_blocks_by_dataset']}); {m['draws']} shared draws; seed-0 references (random p95, |sham|) from each block's summary.",
         "", "Grades: owned = steering reference and fixed-family advantage; advantage_no_reference = fixed-family advantage without the "
         "reference; stronger_competitor; unresolved.", "",
         "## Survival of the seed-0 grade", "",
         "| group | blocks | cells | owned | both | >=1 | none | competitor | both | >=1 | unresolved | both | >=1 | adv. no ref. | both | >=1 | owned O>0 both seeds | seed-0 regrade agrees |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, g in R["transitions"].items():
        s = g["survival"]
        L.append(f"| {name} | {g['n_blocks']} | {g['n_cells']} | {s['owned']['n']} | {s['owned']['both']} | {s['owned']['any']} | {s['owned']['none']} | "
                 f"{s['stronger_competitor']['n']} | {s['stronger_competitor']['both']} | {s['stronger_competitor']['any']} | "
                 f"{s['unresolved']['n']} | {s['unresolved']['both']} | {s['unresolved']['any']} | "
                 f"{s['advantage_no_reference']['n']} | {s['advantage_no_reference']['both']} | {s['advantage_no_reference']['any']} | "
                 f"{g['owned_O_pos_both_seeds']} | {g['seed0_regrade_agrees']}/{g['n_cells']} |")
    for name, g in R["transitions"].items():
        L += ["", f"## Transitions, {name} (rows: seed-0 grade; columns: seed-k grade; pooled over seeds 1 and 2)", "",
              "| seed 0 \\ seed k | " + " | ".join(GRADES) + " |", "|---|" + "---|" * len(GRADES)]
        for g0 in GRADES:
            L.append(f"| {g0} | " + " | ".join(str(g["transitions"]["pooled"][g0][g1]) for g1 in GRADES) + " |")
    L += ["", "## Owned cells that lose the grade under a refit", "", "| block | q | seed-0 O_q | seed-1 grade (O_q) | seed-2 grade (O_q) |", "|---|---|---|---|---|"]
    for b in R["blocks"]:
        if b.get("status") != "OK":
            continue
        for q in CONCEPTS[b["dataset"]]:
            if b["seed0_paper"][q]["grade"] == "owned" and any(b["per_seed"][str(k)]["per_question"][q]["grade"] != "owned" for k in SEEDS):
                L.append(f"| {b['model']}/{b['dataset']} | {q} | {b['seed0_paper'][q]['O_q']:+.3f} | "
                         + " | ".join(f"{b['per_seed'][str(k)]['per_question'][q]['grade']} ({b['per_seed'][str(k)]['per_question'][q]['O_q']:+.3f})" for k in SEEDS) + " |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    models = sorted([p.name for p in RUN_ROOT.iterdir() if p.is_dir() and any((p / d).is_dir() for d in DATASETS)],
                    key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 999, m))
    jobs = [(m, d, a.draws) for m in models for d in DATASETS if (run_dir(m, d) / "run.json").exists()]
    t0 = time.time()
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            blocks = list(ex.map(_job, jobs))
    else:
        blocks = [_job(j) for j in jobs]
    for b in blocks:
        print(f"{b['model']:12s} {b['dataset']:9s} {b['status']:14s} {b.get('reason', '')} {b.get('seconds', '')}", flush=True)
    ok = [b for b in blocks if b.get("status") == "OK"]
    R = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_root": str(RUN_ROOT), "alpha": PRIMARY_ALPHA,
                  "draws": a.draws, "bootstrap_seed": BOOT_CORE_SEED, "seeds": list(SEEDS), "grades": list(GRADES),
                  "n_blocks": len(ok), "n_blocks_by_dataset": {ds: sum(b["dataset"] == ds for b in ok) for ds in DATASETS},
                  "blocks": [f"{b['model']}/{b['dataset']}" for b in ok],
                  "skipped": [{"block": f"{b['model']}/{b['dataset']}", "status": b["status"], "reason": b.get("reason")} for b in blocks if b.get("status") != "OK"],
                  "rules": {"reference": "W^k_qq > 0 and > CORE random p95 and > CORE |sham| of the same block/question (summary values), random_n = 119",
                            "verdict": "6x5 max-T simultaneous bounds over the shared unit-bootstrap draws (cftransfer.analysis.max_t)",
                            "baseline": "CORE clean baseline (fit seed 0) reused, as in the REFIT module"},
                  "seconds": round(time.time() - t0, 1), "script": str(Path(__file__).resolve())},
         "transitions": transitions(blocks), "blocks": blocks}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "refit.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else o), encoding="utf-8")
    write_md(R, out / "refit.md")
    print(f"wrote {out / 'refit.json'} and refit.md in {time.time() - t0:.0f}s ({len(ok)} blocks)")


if __name__ == "__main__":
    main()
