"""Summarise D1 from whatever has finished, and refuse to over-report what has not.

Section 5 of docs/10_RESULTS.md is regenerated from this. Three rules are enforced here rather than left
to the reader, because a partial CSV looks exactly like a complete one:

  1. `src/intervene.py` rewrites `intervention.csv` after every locus, so the CSV's existence says nothing
     about whether the job finished. Only the `DONE` marker does, and it is reported per run.
  2. A locus is summarised only when its grid is complete *against the protocol*, not merely rectangular.
     An earlier version of this file checked only that the rows present formed a full
     directions x magnitudes rectangle. That check passes on a locus truncated to a rectangular subset:
     5 directions x 9 magnitudes out of 27 x 9 was reported as complete, and a locus cut down to 2 random
     directions was reported as complete *and* as meeting the selectivity criterion, because a 5th-to-95th
     percentile band over 2 samples is not a band. The grid is now checked against the expected number of
     random directions, the expected unrelated-concept directions, the sham, and the declared magnitude
     set, taken from the run's `meta.json` when it exists and from the protocol constants otherwise.
  3. The selectivity statistic is the pre-registered one. `docs/00_PAPER_PLAN.md` §5.4 and the
     `statistic` field that `src/intervene.py` writes into `meta.json` both say the effect of `v_c` must
     exceed the **95th percentile** of the random-direction effects. An earlier version of this file
     compared against the maximum instead, which is the 100th percentile and a strictly harder test; it
     reported the one qualifying cell as failing.

Two readings of "exceeds the 95th percentile of the random-direction effects" are reported separately,
because the plan's wording does not disambiguate them and they do not always agree:

  effect-size reading   max_alpha |P(yes | c, alpha) - base| against the 95th percentile of the same
                        quantity computed for each of the random directions. Sign-blind.
  per-magnitude reading at each alpha, whether P(yes) under `v_c` falls outside the 5th-to-95th percentile
                        of the 20 random P(yes) values at that same alpha. Keeps the sign, and records
                        which side, because exceeding on the quiet side is the opposite of selective.

    python src/summarize_interventions.py
    python src/summarize_interventions.py --runs runs --markdown
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The protocol, from src/intervene.py's defaults and docs/00_PAPER_PLAN.md 5.4. Used only when a run has
# no meta.json, which is the case for every run that crashed or is still going.
PROTOCOL_N_RANDOM = 20
PROTOCOL_ALPHAS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
# src/intervene.py draws unrelated directions from this list minus the target concept, keeping those with
# more than 50 training positives. Every concept in the manifest clears 50, so the count is 5, or 4 when
# the target is itself in the list.
PROTOCOL_UNRELATED_POOL = ("Atelectasis", "Pneumothorax", "Cardiomegaly", "Mass", "Nodule")


def locus_key(name: str):
    if name.startswith("vis"):
        return (0, 0)
    if name == "connector":
        return (1, 0)
    if ".L" in name:
        return (2, int(name.split(".L")[1].split(".")[0]))
    return (3, 0)


def concept_of(run_dir: str, meta: dict) -> str | None:
    """The target concept, from meta.json if the run got far enough to write one, else the directory."""
    if meta.get("concept"):
        return meta["concept"]
    m = re.match(r"int_.+_([a-z]+)$", os.path.basename(run_dir))
    return m.group(1).capitalize() if m else None


def expected_grid(meta: dict, concept: str | None):
    """How many directions and which magnitudes a complete locus must carry."""
    n_random = int(meta.get("n_random", PROTOCOL_N_RANDOM))
    alphas = [float(a) for a in meta.get("alphas", PROTOCOL_ALPHAS)]
    if concept is None:
        n_unrelated = None                      # unknown target: accept 4 or 5, checked by the caller
    else:
        n_unrelated = sum(1 for c in PROTOCOL_UNRELATED_POOL if c.lower() != concept.lower())
    return n_random, n_unrelated, sorted(alphas)


def read_run(d: str):
    """Returns (done, per-locus rows, meta) with no assumption that any of them exists."""
    path = os.path.join(d, "intervention.csv")
    done = os.path.exists(os.path.join(d, "DONE"))
    meta = {}
    mp = os.path.join(d, "meta.json")
    if os.path.exists(mp):
        try:
            meta = json.load(open(mp))
        except json.JSONDecodeError:
            meta = {}
    if not os.path.exists(path):
        return done, {}, meta
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["alpha"] = float(r["alpha"])
        r["mean_p_yes"] = float(r["mean_p_yes"])
        r["sd"] = float(r["sd"])
        r["n"] = int(r["n"])
    by = defaultdict(list)
    for r in rows:
        by[r["locus"]].append(r)
    return done, by, meta


def grid_complete(rows, n_random_exp, n_unrelated_exp, alphas_exp):
    """None if the locus is complete, otherwise the reason it is not.

    Rectangularity alone is not sufficient and is not sufficient here: the counts of each kind of
    direction and the magnitude set are checked against the protocol.
    """
    dirs = sorted({r["direction"] for r in rows})
    alphas = sorted({r["alpha"] for r in rows})
    n_random = sum(1 for d in dirs if d.startswith("random"))
    n_unrel = sum(1 for d in dirs if d.startswith("unrelated"))
    if alphas != alphas_exp:
        missing = [a for a in alphas_exp if a not in alphas]
        extra = [a for a in alphas if a not in alphas_exp]
        return (f"magnitudes {alphas} != protocol {alphas_exp}"
                + (f", missing {missing}" if missing else "") + (f", unexpected {extra}" if extra else ""))
    if "concept" not in dirs:
        return "no concept direction"
    if "sham" not in dirs:
        return "no sham direction"
    if n_random < n_random_exp:
        return f"{n_random} random directions, protocol requires {n_random_exp}"
    if n_unrelated_exp is None:
        if n_unrel not in (len(PROTOCOL_UNRELATED_POOL), len(PROTOCOL_UNRELATED_POOL) - 1):
            return f"{n_unrel} unrelated directions, protocol requires 4 or 5"
    elif n_unrel != n_unrelated_exp:
        return f"{n_unrel} unrelated directions, protocol requires {n_unrelated_exp}"
    if len(rows) != len(dirs) * len(alphas):
        return f"{len(rows)} rows for a {len(dirs)} x {len(alphas)} grid, {len(dirs) * len(alphas)} expected"
    if (0.0 not in alphas) or ("concept", 0.0) not in {(r["direction"], r["alpha"]) for r in rows}:
        return "no alpha = 0 baseline"
    return None


def summarise_locus(rows):
    """The numbers section 5 quotes. Callers must check `grid_complete` first."""
    dirs = sorted({r["direction"] for r in rows})
    alphas = sorted({r["alpha"] for r in rows})
    g = {(r["direction"], r["alpha"]): r["mean_p_yes"] for r in rows}
    base = g[("concept", 0.0)]
    rand = [d for d in dirs if d.startswith("random")]

    def eff(d):
        return max(abs(g[(d, a)] - base) for a in alphas)

    # Per-magnitude reading: keeps the sign, records which side of the band.
    outside = []
    for a in alphas:
        if a == 0:
            continue
        rd = [g[(d, a)] for d in rand]
        c = g[("concept", a)]
        if c > np.percentile(rd, 95):
            outside.append((a, "above"))
        elif c < np.percentile(rd, 5):
            outside.append((a, "below"))

    # Effect-size reading: the plan's literal statistic, 95th percentile and not the maximum.
    rand_effects = np.array([eff(d) for d in rand])
    rand_p95 = float(np.percentile(rand_effects, 95))
    concept_eff = eff("concept")

    cvals = [g[("concept", a)] for a in alphas]
    diffs = np.diff(cvals)
    monotone = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    return {
        "n_eval": rows[0]["n"], "n_directions": len(dirs), "n_random": len(rand), "alphas": alphas,
        "base": base,
        "concept_max": concept_eff,
        "sham_max": eff("sham") if "sham" in dirs else float("nan"),
        "any_max": max(eff(d) for d in dirs),
        "random_max": float(rand_effects.max()),
        "random_p95": rand_p95,
        "outside_band": outside,
        "concept_monotone": monotone,
        "selective_effect_size": bool(concept_eff > rand_p95),
        "selective_per_magnitude": bool(outside),
    }


def planned_jobs(runs_dir: str):
    """The queue's own plan, so the denominator is read rather than asserted."""
    log = os.path.join(runs_dir, "launcher.log")
    plan = {}
    if not os.path.exists(log):
        return plan
    for line in open(log):
        m = re.match(r"\s+(int_\S+)\s+loci\s+(\S+)\s*$", line)
        if m:
            plan[m.group(1)] = [s for s in m.group(2).split(",") if s]
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.runs, "int_*")))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print(f"no int_* run directories under {args.runs}. Nothing to summarise.")
        return
    plan = planned_jobs(args.runs)

    n_done = n_complete_loci = n_partial_loci = 0
    complete_by_run: dict[str, list[str]] = {}
    if args.markdown:
        print("| run | rows | complete loci | partial or absent | `DONE` |")
        print("|---|---|---|---|---|")
    for d in dirs:
        done, by, meta = read_run(d)
        n_done += int(done)
        concept = concept_of(d, meta)
        n_random_exp, n_unrel_exp, alphas_exp = expected_grid(meta, concept)
        complete, partial = [], {}
        for lo, rows in by.items():
            why = grid_complete(rows, n_random_exp, n_unrel_exp, alphas_exp)
            if why is None:
                complete.append(lo)
            else:
                partial[lo] = why
        complete.sort(key=locus_key)
        complete_by_run[os.path.basename(d)] = complete
        n_complete_loci += len(complete)
        n_partial_loci += len(partial)
        nrows = sum(len(v) for v in by.values())
        name = os.path.basename(d)
        planned = plan.get(name, [])
        missing = [lo for lo in planned if lo not in by]
        if args.markdown:
            gap = ", ".join(f"`{c}` ({partial[c]})" for c in sorted(partial, key=locus_key))
            if missing:
                gap = (gap + "; " if gap else "") + f"not started: {', '.join(f'`{m}`' for m in missing)}"
            print(f"| `{name}` | {nrows} | {', '.join(f'`{c}`' for c in complete) or 'none'} | "
                  f"{gap or 'none'} | {'yes' if done else '**no**'} |")
            continue
        print(f"\n### {name}  rows={nrows}  DONE={'yes' if done else 'NO, treat as preliminary'}")
        if planned:
            print(f"    planned {len(planned)} loci: {', '.join(planned)}")
        if not by:
            print("    no intervention.csv yet: job running, queued, or died before its first locus")
        for lo in sorted(partial, key=locus_key):
            print(f"    {lo:14s} INCOMPLETE, not summarised: {partial[lo]}")
        if missing:
            print(f"    {'':14s} not started: {', '.join(missing)}")
        for lo in complete:
            s = summarise_locus(by[lo])
            band = (", ".join(f"{a:+.0f}({side})" for a, side in s["outside_band"])
                    or "inside the random band at every magnitude")
            print(f"    {lo:14s} base P(yes) {s['base']:.4f}  n_eval {s['n_eval']}  "
                  f"dirs {s['n_directions']} ({s['n_random']} random)")
            print(f"    {'':14s} max|effect|  concept {s['concept_max']:.4f}  sham {s['sham_max']:.4f}  "
                  f"random 95th pct {s['random_p95']:.4f}  random max {s['random_max']:.4f}  "
                  f"any direction {s['any_max']:.4f}")
            print(f"    {'':14s} per-magnitude: concept outside the 5th-95th random band at alpha {band}")
            print(f"    {'':14s} concept dose-response monotone: {s['concept_monotone']}")
            print(f"    {'':14s} pre-registered selectivity, effect-size reading: "
                  f"{s['selective_effect_size']}  |  per-magnitude reading: "
                  f"{s['selective_per_magnitude']}")

    planned_loci = sum(len(v) for v in plan.values())
    # Only loci that the queue actually planned count against the plan. A run directory that predates the
    # queue, or a locus outside its job's planned list, is real data but is not progress against 105, and
    # crediting it there would understate how much is outstanding.
    in_plan = sum(1 for run, loci in complete_by_run.items() for lo in loci if lo in plan.get(run, []))
    off_plan = n_complete_loci - in_plan
    print(f"\n{n_done} of {len(dirs)} run directories carry a DONE marker.")
    print(f"{n_complete_loci} loci have a complete grid; {n_partial_loci} present but incomplete.")
    if planned_loci:
        print(f"Against the queue's own plan ({planned_loci} loci over {len(plan)} jobs, "
              f"from runs/launcher.log): {in_plan} complete, {planned_loci - in_plan} outstanding"
              + (f"; {off_plan} further complete loci belong to runs outside the queue and do not "
                 f"count towards it." if off_plan else "."))
    if n_done < len(dirs):
        print("Runs without DONE are unfinished or crashed. Their complete loci are usable; their")
        print("incomplete and absent loci are not evidence of anything.")


if __name__ == "__main__":
    main()
