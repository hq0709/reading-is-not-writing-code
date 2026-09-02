"""D1 analysis: at which locus does steering the concept direction move the output more than
matched controls do?

Reads every `runs/int_<arch>_<concept>/intervention.csv`, computes the concept effect against the
null band of 20 equal-norm random directions, the sham and the unrelated-concept directions, and then
joins the result against the D2 probe results so that the paper's central table can be written:

    the locus of PEAK DECODABILITY against the locus of PEAK STEERABILITY, per (model, concept).

Three rules this file enforces rather than assumes.

1. A CSV is not a finished job. `src/intervene.py` rewrites `intervention.csv` after every locus so a
   crash keeps partial results, and writes `DONE` only at the very end. Everything here carries a
   `job_done` column, and the console output separates finished jobs from partial ones. Nothing is
   summarised as complete unless `DONE` sits beside it.

2. Extreme magnitudes are excluded from the effect, and the exclusion is empirical rather than
   stylistic. At |alpha| = 8 the random control directions themselves collapse the output: in
   runs/int_lingshu_effusion/intervention.csv at locus vis.last the mean effect over the 20 random
   directions is -0.2672 at alpha = -8 and -0.2938 at alpha = +8, against -0.0117 at alpha = -1 and
   -0.0056 at alpha = +1. A perturbation that moves P(yes) by 0.29 no matter which direction it points
   in is measuring off-manifold disruption of the forward pass, not the concept. Including it would let
   a large concept effect at alpha = 8 be scored against a null band that is itself saturated, which
   inflates nothing and hides everything: the concept and the controls both sit at the floor. The
   analysis therefore restricts to |alpha| <= ALPHA_CORE_MAX and, inside that, to the sub-range where
   the concept response is still monotone in alpha, which is what a dose-response claim requires. The
   alpha = 8 numbers are still emitted, as `concept_effect_extreme` and `rand_mean_extreme`, so the
   reader can see the degradation that motivated the exclusion.

3. A locus can be causally inert by construction. Both LLaVA-family models set
   `vision_feature_layer = -2` (verified from their configs), so the output of the last vision block,
   which is the `vis.last` locus, is discarded by the model and never reaches the connector. Steering
   it changes nothing at all, for any direction at any magnitude, to the last digit of float64. That is
   not "not selective", it is "not connected", and the two must not be reported as the same thing.
   Such loci are flagged DEAD and excluded from the peak-steerability search.

4. A partly-flushed locus is not a locus. `intervene.py` rewrites the whole CSV after every locus, so a
   reader arriving mid-write can see a torn file. A locus whose directions do not all carry the full
   alpha grid is skipped with a printed reason, and `selective` -- the paper's pre-specified "beats at
   least 19 of 20 randoms" -- may only be True when the random band is COMPLETE. A short band still
   gets its numbers reported, with `full_random_band=False` saying why the boolean is withheld, because
   "beats the 95th percentile of the 13 randoms that happened to be on disk" is a different and weaker
   test that the boolean alone would silently pass off as the pre-specified one.

5. Two run directories for the same (model, concept) are not automatically duplicates. The pre-launcher
   `int_lingshu_effusion` steers llm.L13/L22 while the launcher's `int_lingshu7b_effusion` steers
   llm.L7/L14/L21/L27: different points on the computation. They are pooled only when they demonstrably
   measured the same thing (same n_eval and a bit-identical alpha = 0 baseline P(yes), which is the
   model's answer with the hook a no-op); otherwise both are reported separately and flagged. Neither
   is ever dropped, because dropping one would delete evidence under the label "duplicate".

    python src/analyze_intervention.py
    python src/analyze_intervention.py --runs runs --out runs
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from registry import REGISTRY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Magnitudes above this are dropped from the effect. See rule 2 in the module docstring.
ALPHA_CORE_MAX = 4.0
# The fixed cut above is not enough on its own: at the vision-tower locus the controls are already
# collapsing the output at |alpha| = 4 (mean random effect -0.2057 in runs/int_lingshu_effusion at
# vis.last). So a second, per-locus, data-driven cut is applied: a magnitude is admissible only if the
# MEAN effect over the 20 random directions at that magnitude is smaller than this. A magnitude at
# which a random vector already moves P(yes) by more than 2 points is a magnitude at which the forward
# pass is being disrupted, and a concept effect measured there is not a concept effect.
DEGRADE_TOL = 0.02
# Monotonicity tolerance, in units of P(yes). Movements smaller than this do not break a run.
MONO_TOL = 2e-3
# A locus whose entire response surface spans less than this is causally inert, not weakly steered.
DEAD_RANGE = 1e-9
# Fewer random directions than this and the 95th percentile is not worth quoting at all.
MIN_RANDOM = 10
# `selective` is the paper's pre-specified statistic, and it was specified against the FULL random
# band (20 directions), where "exceeds the 95th percentile" means "beats at least 19 of 20". Computing
# the same boolean from a partly-flushed band of 12 directions is a different and much weaker test, and
# the boolean would not say so. So the band has to be complete before `selective` may be True: loci
# with a short band are still measured and reported, but `selective` is forced False and
# `full_random_band` records why. MIN_RANDOM stays as the floor for reporting a number at all.
EXPECTED_N_RANDOM_FALLBACK = 20

# `runs/int_lingshu_effusion` predates the launcher and was invoked through the old `--model` CLI, which
# is why it crashed on `args.model` after finishing all five of its loci. Its loci (vis.last, connector,
# llm.L0.vis, llm.L13.vis, llm.L22.vis) are only valid for a 28-layer model and its log reports
# 18212 train rows, matching runs/probe_lingshu7b/probe_meta.json exactly. It is lingshu7b.
DIR_ALIAS = {"lingshu": "lingshu7b"}


# ----------------------------------------------------------------------------- loading

def parse_run_dir(name: str) -> tuple[str, str] | None:
    """`int_<arch>_<concept>` -> (arch_key, Concept). Returns None if the arch cannot be identified."""
    if not name.startswith("int_"):
        return None
    body = name[4:]
    for key in sorted(REGISTRY, key=len, reverse=True):
        if body.startswith(key + "_"):
            return key, body[len(key) + 1:]
    for alias, key in DIR_ALIAS.items():
        if body.startswith(alias + "_"):
            return key, body[len(alias) + 1:]
    return None


def load_runs(runs_dir: str) -> list[dict]:
    """One entry per intervention run directory that has a CSV, with its DONE state recorded."""
    out = []
    for csv_path in sorted(glob.glob(os.path.join(runs_dir, "int_*", "intervention.csv"))):
        d = os.path.dirname(csv_path)
        name = os.path.basename(d)
        parsed = parse_run_dir(name)
        if parsed is None:
            raise SystemExit(f"cannot identify the architecture of {d}. Add it to DIR_ALIAS in "
                             f"{__file__} rather than guessing at analysis time.")
        arch_key, concept_raw = parsed
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            continue
        meta_path = os.path.join(d, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        # meta.json, when it exists, is authoritative: it was written by the job itself.
        concept = meta.get("concept") or concept_raw.capitalize()
        # The band size the job was configured with. meta.json is only written at the very end, so a
        # partial run has none; fall back to the largest band actually observed in the file, which is
        # the size of any locus that finished flushing.
        obs_rand = max((sum(1 for d2 in {r2["direction"] for r2 in rows if r2["locus"] == lo}
                            if d2.startswith("random"))
                        for lo in {r2["locus"] for r2 in rows}), default=0)
        out.append({
            "run": name, "dir": d, "arch": meta.get("arch", arch_key), "concept": concept,
            "done": os.path.exists(os.path.join(d, "DONE")), "rows": rows,
            "n_eval": int(rows[0]["n"]),
            "expected_n_random": int(meta.get("n_random") or obs_rand or EXPECTED_N_RANDOM_FALLBACK),
            # The control directions are run on a subset of the concept's magnitudes: a null band only
            # has to exist where a comparison is made, and running 21 controls at every magnitude spent
            # most of the budget on points no statistic reads. Every comparison below therefore happens
            # on this grid, not on the concept's full sweep.
            "control_alphas": sorted(float(a) for a in (meta.get("control_alphas") or [])),
        })
    # Empty run directories exist for jobs that have started but not yet flushed a locus. Record them so
    # the console inventory can say "started, nothing written" instead of silently omitting them.
    for d in sorted(glob.glob(os.path.join(runs_dir, "int_*"))):
        if os.path.isdir(d) and not os.path.exists(os.path.join(d, "intervention.csv")):
            parsed = parse_run_dir(os.path.basename(d))
            if parsed:
                out.append({"run": os.path.basename(d), "dir": d, "arch": parsed[0],
                            "concept": parsed[1].capitalize(), "done": False, "rows": [], "n_eval": 0,
                            "expected_n_random": 0, "control_alphas": []})
    return out


def index_rows(rows: list[dict]) -> dict:
    """(locus, direction) -> {alpha: mean_p_yes}, plus the sd and n alongside."""
    idx: dict = defaultdict(dict)
    sds: dict = defaultdict(dict)
    for r in rows:
        k = (r["locus"], r["direction"])
        idx[k][float(r["alpha"])] = float(r["mean_p_yes"])
        sds[k][float(r["alpha"])] = float(r["sd"])
    return idx, sds


# ----------------------------------------------------------------------------- effects

def admissible_alphas(idx, locus: str, arm: str, alphas: list[float], base: float) -> list[float]:
    """Magnitudes on one arm that pass both the fixed cut and the per-locus degradation guard."""
    rand = sorted(d for (lo, d) in idx if lo == locus and d.startswith("random"))
    out = []
    for a in sorted([x for x in alphas if 0 < abs(x) <= ALPHA_CORE_MAX and (x > 0) == (arm == "pos")],
                    key=abs):
        eff = [idx[(locus, d)][a] - base for d in rand if a in idx[(locus, d)]]
        if eff and abs(float(np.mean(eff))) > DEGRADE_TOL:
            break                      # degradation is monotone in |alpha|, so stop rather than skip
        out.append(a)
    return out


def monotone_arm(curve: dict, arm: str, alphas: list[float], core: list[float]) -> list[float]:
    """The alphas on one arm over which the concept response is still monotone, walking out from 0.

    A dose-response slope is only a dose-response slope where the response is ordered by dose. Walking
    outward from alpha = 0 and stopping at the first reversal larger than MONO_TOL gives that range
    without any hand-picking of which points to keep.
    """
    if not core or 0.0 not in curve:
        return []
    core = sorted(core, key=abs)
    kept, direction = [], 0
    prev = curve[0.0]
    for a in core:
        if a not in curve:
            break
        step = curve[a] - prev
        if abs(step) > MONO_TOL:
            s = 1 if step > 0 else -1
            if direction == 0:
                direction = s
            elif s != direction:
                break
        kept.append(a)
        prev = curve[a]
    return kept


def slope(curve: dict, kept: list[float]) -> float:
    """Signed OLS slope of mean P(yes) on alpha over the monotone points plus the alpha = 0 anchor."""
    xs = [0.0] + list(kept)
    ys = [curve[a] for a in xs]
    if len(xs) < 2:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def analyse_locus(idx, locus: str, alphas: list[float], ctrl_alphas: list[float],
                  expected_n_random: int) -> list[dict]:
    """Both arms of one locus: concept effect, null band, sham, unrelated, selectivity, effect size."""
    if (locus, "concept") not in idx:
        return []
    curve = idx[(locus, "concept")]
    if 0.0 not in curve:
        return []
    base = curve[0.0]

    rand_names = sorted(d for (lo, d) in idx if lo == locus and d.startswith("random"))
    unrel_names = sorted(d for (lo, d) in idx if lo == locus and d.startswith("unrelated_"))
    has_sham = (locus, "sham") in idx

    # Causal inertia check. If nothing at this locus moves the output at any magnitude for any
    # direction, the locus is not weakly steered, it is disconnected from the forward pass.
    everything = [v for (lo, _), c in idx.items() if lo == locus for v in c.values()]
    dead = (max(everything) - min(everything)) < DEAD_RANGE

    out = []
    for arm in ("neg", "pos"):
        # Only magnitudes where the null band exists can carry the statistic.
        adm = admissible_alphas(idx, locus, arm, ctrl_alphas, base)
        degraded = not adm          # even the smallest magnitude already disrupts the forward pass
        if degraded:
            adm = sorted([a for a in ctrl_alphas
                          if 0 < abs(a) <= ALPHA_CORE_MAX and (a > 0) == (arm == "pos")], key=abs)[:1]
        kept = monotone_arm(curve, arm, ctrl_alphas, adm)
        # A dead or flat locus has no monotone range at all. Fall back to the smallest admissible
        # magnitude so the row still exists and carries an effect of ~0 rather than being dropped.
        if not kept:
            kept = adm[:1]
        if not kept:
            continue
        a_used = kept[-1]
        c_eff = curve[a_used] - base

        r_eff = np.array([idx[(locus, d)][a_used] - base for d in rand_names
                          if a_used in idx[(locus, d)]])
        r_mean = float(r_eff.mean()) if r_eff.size else float("nan")
        r_sd = float(r_eff.std(ddof=1)) if r_eff.size > 1 else float("nan")
        r_p95 = float(np.percentile(np.abs(r_eff), 95)) if r_eff.size else float("nan")

        sham_eff = (idx[(locus, "sham")].get(a_used, float("nan")) - base) if has_sham else float("nan")
        u_eff = np.array([idx[(locus, d)][a_used] - base for d in unrel_names
                          if a_used in idx[(locus, d)]])

        # Pre-specified statistic (docs/00_PAPER_PLAN.md 5.4): the concept effect must exceed the 95th
        # percentile of the random-direction effects at the same magnitude, on held-out images.
        # With 20 random directions the 95th percentile sits between the 19th and 20th order statistic,
        # so the test is "beats at least 19 of 20 randoms". `rand_rank` records how many it actually
        # beat, because 20/20 and 19/20 are different pieces of evidence and the boolean hides that.
        # The band must also be COMPLETE, not merely large enough to quote. "Exceeds the 95th
        # percentile of 20 randoms" and "exceeds the 95th percentile of the 13 randoms that had been
        # flushed when the analysis ran" are different tests, and only the first is the one the paper
        # pre-specified. A short band still gets its numbers reported, but not the boolean.
        full_band = bool(expected_n_random and r_eff.size >= expected_n_random)
        selective = bool(r_eff.size >= MIN_RANDOM and full_band and not dead and abs(c_eff) > r_p95)
        rand_rank = int((np.abs(r_eff) < abs(c_eff)).sum()) if r_eff.size else 0
        z = float((c_eff - r_mean) / r_sd) if r_sd and np.isfinite(r_sd) and r_sd > 0 else float("nan")

        # The magnitudes the analysis refuses to score on, kept visible so the exclusion is auditable.
        ext = max([a for a in ctrl_alphas if (a > 0) == (arm == "pos")], key=abs, default=None)
        c_ext = curve.get(ext, float("nan")) - base if ext is not None else float("nan")
        r_ext = np.array([idx[(locus, d)][ext] - base for d in rand_names
                          if ext is not None and ext in idx[(locus, d)]])

        out.append({
            "locus": locus, "arm": arm,
            "alphas_admissible": "|".join(f"{a:g}" for a in adm),
            "alphas_monotone": "|".join(f"{a:g}" for a in kept), "alpha_used": a_used,
            "degraded_at_every_alpha": degraded,
            "baseline_p_yes": base, "concept_effect": c_eff, "concept_slope": slope(curve, kept),
            "n_random": int(r_eff.size), "rand_mean": r_mean, "rand_sd": r_sd, "rand_p95_abs": r_p95,
            "sham_effect": sham_eff,
            "n_unrelated": int(u_eff.size),
            "unrel_mean": float(u_eff.mean()) if u_eff.size else float("nan"),
            "unrel_max_abs": float(np.abs(u_eff).max()) if u_eff.size else float("nan"),
            "beats_sham": bool(np.isfinite(sham_eff) and abs(c_eff) > abs(sham_eff)),
            "beats_all_unrelated": bool(u_eff.size and abs(c_eff) > float(np.abs(u_eff).max())),
            "rand_rank": rand_rank,
            "n_random_expected": int(expected_n_random), "full_random_band": full_band,
            # A concept direction that means what it is named must raise P(yes) when it is added and
            # lower it when it is subtracted. An effect that beats the random band with the wrong sign
            # is evidence of a direction that perturbs the answer, not one that encodes the concept, and
            # the paper must not count it as a use of the concept. Recorded per arm and used in the
            # headline so a sign-inverted "selective" locus cannot be read as a positive result.
            "sign_as_expected": bool(c_eff > 0) if arm == "pos" else bool(c_eff < 0),
            "selective": selective, "effect_size_z": z, "dead_locus": dead,
            "alpha_extreme": ext, "concept_effect_extreme": c_ext,
            "rand_mean_extreme": float(r_ext.mean()) if r_ext.size else float("nan"),
        })
    return out


def locus_is_analysable(idx, locus: str, alphas: list[float],
                        ctrl_alphas: list[float]) -> tuple[bool, str]:
    """Guard against a locus that was half written when the job was interrupted.

    `src/intervene.py` rewrites the whole CSV after every locus, so a reader that arrives mid-write can
    see a torn file: a locus whose last few directions, or whose last few magnitudes, are simply not
    there yet. Checking only the concept row would let such a locus through and score it against a
    short random band. Every direction present at the locus must therefore carry the complete alpha
    grid before the locus is analysed at all.
    """
    if (locus, "concept") not in idx:
        return False, "no concept direction"
    missing = [a for a in alphas if a not in idx[(locus, "concept")]]
    if missing:
        return False, f"concept missing alphas {missing}"
    # Controls are run on their own grid by design, so completeness is checked against that grid.
    # Requiring the concept's full sweep of them rejected every locus in the sweep.
    ragged = sorted(d for (lo, d) in idx if lo == locus and d != "concept"
                    and any(a not in idx[(lo, d)] for a in ctrl_alphas))
    if ragged:
        return False, (f"half-written locus: {len(ragged)} direction(s) missing magnitudes "
                       f"({', '.join(ragged[:4])}{'...' if len(ragged) > 4 else ''})")
    n_rand = sum(1 for (lo, d) in idx if lo == locus and d.startswith("random"))
    if n_rand < MIN_RANDOM:
        return False, f"only {n_rand} random directions"
    return True, ""


# ----------------------------------------------------------------------------- probe join

def load_probe(runs_dir: str, arch: str) -> dict:
    """(concept, locus) -> probe row, from runs/probe_<arch>/probe_results.csv."""
    path = os.path.join(runs_dir, f"probe_{arch}", "probe_results.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["concept"], r["locus"])] = {k: (float(v) if k not in ("concept", "locus") else v)
                                           for k, v in r.items()}
    return out


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out", default=None, help="defaults to --runs")
    args = ap.parse_args()
    out_dir = args.out or args.runs
    os.makedirs(out_dir, exist_ok=True)

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no intervention runs under {args.runs}")

    print("=" * 100)
    print("D1 INTERVENTION RUNS")
    print("=" * 100)
    print(f"{'run':32s} {'arch':13s} {'concept':14s} {'DONE':5s} {'loci':5s} {'rows':6s} {'n_eval':6s}")
    for r in runs:
        loci = sorted({x["locus"] for x in r["rows"]})
        print(f"{r['run']:32s} {r['arch']:13s} {r['concept']:14s} "
              f"{'yes' if r['done'] else 'NO':5s} {len(loci):<5d} {len(r['rows']):<6d} {r['n_eval']:<6d}")
    n_done = sum(1 for r in runs if r["done"])
    print(f"\n{n_done} of {len(runs)} run directories carry a DONE marker. "
          f"Everything else is partial and is labelled as such in every table below.")

    # ---- the empirical justification for dropping |alpha| = 8, recomputed from this data every run
    print("\n" + "=" * 100)
    print("WHY THE EXTREME MAGNITUDES ARE EXCLUDED: the random control directions degrade the output too")
    print("=" * 100)
    print(f"{'run':30s} {'locus':14s} {'mean random effect':>19s} {'|a|<=2':>9s} {'|a|=4':>9s} {'|a|=8':>9s}")
    for r in runs:
        if not r["rows"]:
            continue
        idx, _ = index_rows(r["rows"])
        alphas = sorted({float(x["alpha"]) for x in r["rows"]})
        ctrl = r.get("control_alphas") or [a for a in alphas if a != 0.0]
        for locus in sorted({x["locus"] for x in r["rows"]}, key=lambda s: (len(s), s)):
            if (locus, "concept") not in idx or 0.0 not in idx[(locus, "concept")]:
                continue
            base = idx[(locus, "concept")][0.0]
            rn = [d for (lo, d) in idx if lo == locus and d.startswith("random")]
            def band(pred):
                v = [idx[(locus, d)][a] - base for d in rn for a in alphas
                     if pred(a) and a in idx[(locus, d)]]
                return float(np.mean(v)) if v else float("nan")
            print(f"{r['run']:30s} {locus:14s} {'':19s} "
                  f"{band(lambda a: 0 < abs(a) <= 2):+9.4f} {band(lambda a: abs(a) == 4):+9.4f} "
                  f"{band(lambda a: abs(a) == 8):+9.4f}")

    # ---- per-locus effects
    summary_rows = []
    for r in runs:
        if not r["rows"]:
            continue
        idx, _ = index_rows(r["rows"])
        alphas = sorted({float(x["alpha"]) for x in r["rows"]})
        ctrl = r.get("control_alphas") or [a for a in alphas if a != 0.0]
        for locus in sorted({x["locus"] for x in r["rows"]}):
            ok, why = locus_is_analysable(idx, locus, alphas, ctrl)
            if not ok:
                print(f"  skip {r['run']}/{locus}: {why}")
                continue
            for row in analyse_locus(idx, locus, alphas, ctrl, r["expected_n_random"]):
                summary_rows.append({"model": r["arch"], "domain": REGISTRY[r["arch"]].domain,
                                     "concept": r["concept"], "run": r["run"],
                                     "job_done": r["done"], "n_eval": r["n_eval"], **row})

    if not summary_rows:
        raise SystemExit("no analysable loci yet")

    cols = ["model", "domain", "concept", "run", "job_done", "n_eval", "locus", "arm",
            "alphas_admissible", "alphas_monotone", "alpha_used", "degraded_at_every_alpha",
            "baseline_p_yes", "concept_effect", "concept_slope",
            "n_random", "rand_mean", "rand_sd", "rand_p95_abs", "sham_effect", "beats_sham",
            "n_unrelated", "unrel_mean", "unrel_max_abs", "beats_all_unrelated",
            "rand_rank", "n_random_expected", "full_random_band",
            "sign_as_expected", "selective", "effect_size_z", "dead_locus",
            "alpha_extreme", "concept_effect_extreme", "rand_mean_extreme"]
    sum_path = os.path.join(out_dir, "intervention_summary.csv")
    with open(sum_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nwrote {len(summary_rows)} rows to {sum_path}")

    # ---- compact console table
    print("\n" + "=" * 100)
    print("SELECTIVITY BY LOCUS.  effect = concept P(yes) change at the monotone edge magnitude;")
    print("p95 = 95th percentile of |effect| over the random directions at the same magnitude;")
    print("SEL = |effect| > p95, which with 20 randoms means beating at least 19 of them;")
    print("rank = how many of the 20 randoms the concept actually beat, which the boolean hides;")
    print("SEL-WRONGSIGN = beats the band but with the sign inverted, which is not a use of the concept.")
    print("* marks a partial job, DEAD marks a locus disconnected from the forward pass.")
    print("SHORT-BAND = the random band was incomplete when this ran, so the SEL boolean is withheld.")
    print("=" * 100)
    hdr = (f"{'model':13s} {'concept':13s} {'locus':13s} {'arm':4s} {'a':>5s} {'effect':>8s} "
           f"{'slope':>8s} {'rand mu':>8s} {'rand sd':>8s} {'p95':>7s} {'z':>7s} {'sham':>8s} "
           f"{'unrel':>8s} {'rank':>6s}  SEL")
    print(hdr)
    last = None
    for row in summary_rows:
        key = (row["model"], row["concept"])
        if last is not None and key != last:
            print("-" * len(hdr))
        last = key
        flag = ("DEAD" if row["dead_locus"] else
                ("SEL" if row["selective"] and row["sign_as_expected"] else
                 ("SEL-WRONGSIGN" if row["selective"] else
                  ("SHORT-BAND" if not row["full_random_band"] else "."))))
        star = "" if row["job_done"] else "*"
        print(f"{row['model'] + star:13s} {row['concept']:13s} {row['locus']:13s} {row['arm']:4s} "
              f"{row['alpha_used']:+5.0f} {row['concept_effect']:+8.4f} {row['concept_slope']:+8.4f} "
              f"{row['rand_mean']:+8.4f} {row['rand_sd']:8.4f} {row['rand_p95_abs']:7.4f} "
              f"{row['effect_size_z']:+7.2f} {row['sham_effect']:+8.4f} {row['unrel_mean']:+8.4f} "
              f"{row['rand_rank']:>3d}/{row['n_random']:<2d}  {flag}")

    # ---- headline: decodability against steerability
    print("\n" + "=" * 100)
    print("HEADLINE: PEAK DECODABILITY against PEAK STEERABILITY")
    print("=" * 100)

    # More than one run directory can cover the same (model, concept): `int_lingshu_effusion` predates
    # the launcher and steered the loci vis.last/connector/llm.L0/L13/L22, while the launcher's
    # `int_lingshu7b_effusion` steers vis.last/connector/llm.L0/L7/L14/L21/L27. Those runs are NOT
    # duplicates of one another -- they cover different points on the computation -- so picking one and
    # discarding the other would silently delete real evidence and label the deletion "duplicate".
    #
    # Two runs of the same model and concept may be pooled only if they demonstrably measured the same
    # thing: the same number of held-out images, and a bit-identical alpha = 0 baseline P(yes), which is
    # the model's answer on the eval set with the hook a no-op. If those agree the runs are two halves
    # of one locus sweep and their loci are pooled. If they disagree the runs are not comparable and
    # each gets its own headline row, flagged, rather than one being thrown away.
    by_mc: dict = defaultdict(list)
    for row in summary_rows:
        by_mc[(row["model"], row["concept"])].append(row)

    groups = []          # (model, concept, rows, runs_pooled, incompatible_note)
    for (model, concept), rows_mc in sorted(by_mc.items()):
        runs_here = sorted({r["run"] for r in rows_mc})
        if len(runs_here) == 1:
            groups.append((model, concept, rows_mc, runs_here, ""))
            continue
        sig = {}
        for run in runs_here:
            rr = [r for r in rows_mc if r["run"] == run]
            sig[run] = (rr[0]["n_eval"], round(float(rr[0]["baseline_p_yes"]), 12))
        if len(set(sig.values())) == 1:
            # Same eval set and same no-op baseline: pool the loci. Where two runs both carry a locus,
            # prefer the finished one, then the alphabetically first run, so the choice is deterministic
            # and never silently mixes two measurements of the same point.
            pooled, provenance = {}, {}
            for r in sorted(rows_mc, key=lambda r: (not r["job_done"], r["run"])):
                if (r["locus"], r["arm"]) not in pooled:
                    pooled[(r["locus"], r["arm"])] = r
                    provenance[r["locus"]] = r["run"]
            print(f"note: ({model}, {concept}) is covered by {len(runs_here)} runs with an identical "
                  f"eval set and baseline; pooling their loci: "
                  + ", ".join(f"{lo}<-{rn}" for lo, rn in sorted(provenance.items())))
            groups.append((model, concept, list(pooled.values()), runs_here, ""))
        else:
            note = ("runs disagree on eval set or baseline P(yes) -- "
                    + "; ".join(f"{rn}: n_eval={v[0]}, baseline={v[1]}" for rn, v in sorted(sig.items()))
                    + " -- so they are NOT pooled and each is reported separately")
            print(f"warning: ({model}, {concept}): {note}")
            for run in runs_here:
                groups.append((model, concept, [r for r in rows_mc if r["run"] == run], [run], note))

    head_rows, join_failures, not_yet = [], [], []
    for model, concept, rows_mc, runs_pooled, incompat in groups:
        run = "|".join(runs_pooled)
        probe = load_probe(args.runs, model)
        if not probe:
            join_failures.append(f"no probe_results.csv for {model} "
                                 f"(expected {args.runs}/probe_{model}/probe_results.csv)")
            continue

        # The join must not drop rows. Every locus that was steered has to exist in the probe table for
        # the same concept, or the two halves of the paper's central claim are not talking about the
        # same points on the computation.
        tested = sorted({r["locus"] for r in rows_mc})
        missing = [lo for lo in tested if (concept, lo) not in probe]
        if missing:
            join_failures.append(f"{model}/{concept}: steered loci with no probe row: {missing}. "
                                 f"probe file has {len({lo for (c, lo) in probe if c == concept})} "
                                 f"loci for this concept")
            continue

        probe_concept = {lo: v for (c, lo), v in probe.items() if c == concept}
        peak_all = max(probe_concept.items(), key=lambda kv: kv[1]["selectivity"])
        peak_all_auroc = max(probe_concept.items(), key=lambda kv: kv[1]["real"])
        peak_tested = max(((lo, probe_concept[lo]) for lo in tested),
                          key=lambda kv: kv[1]["selectivity"])

        live = [r for r in rows_mc if not r["dead_locus"]]
        n_dead = len({r["locus"] for r in rows_mc if r["dead_locus"]})
        if not live:
            # Not a join failure. Every locus this job has reached so far is disconnected from the
            # forward pass by construction, so there is no steerability number to report yet.
            not_yet.append(f"{model}/{concept} ({run}): the only loci written so far "
                           f"({', '.join(tested)}) are causally dead; waiting on the live loci")
            continue

        # PEAK STEERABILITY is ranked by the pre-specified statistic, not by raw P(yes) movement.
        # Ranking by raw movement picks the vision tower every time, because a large-magnitude
        # perturbation there collapses the output no matter which direction it points in, and that is a
        # disruption result rather than a steering result. The key is (selective, |z|): a locus where
        # the concept beats the random 95th percentile outranks one where it does not, and among
        # equals the larger control-normalised effect wins. The raw-movement peak is kept as a separate
        # column so the difference between the two rankings stays visible.
        def rank(r):
            z = abs(r["effect_size_z"]) if np.isfinite(r["effect_size_z"]) else -1.0
            return (r["selective"], z)
        peak_steer = max(live, key=rank)
        peak_raw = max(live, key=lambda r: abs(r["concept_effect"]))

        steer_at_peak_decod = [abs(r["concept_effect"]) for r in rows_mc
                               if r["locus"] == peak_tested[0]]
        head_rows.append({
            "model": model, "domain": REGISTRY[model].domain, "concept": concept, "run": run,
            "job_done": all(r["job_done"] for r in rows_mc),
            "n_loci_tested": len(tested), "n_loci_dead": n_dead,
            "loci_tested": "|".join(tested),
            "peak_decod_locus_all_loci": peak_all[0],
            "peak_decod_selectivity_all_loci": peak_all[1]["selectivity"],
            "peak_decod_auroc_locus_all_loci": peak_all_auroc[0],
            "peak_decod_auroc_all_loci": peak_all_auroc[1]["real"],
            "peak_decod_locus_tested": peak_tested[0],
            "peak_decod_selectivity_tested": peak_tested[1]["selectivity"],
            "peak_decod_auroc_tested": peak_tested[1]["real"],
            "peak_steer_locus": peak_steer["locus"], "peak_steer_arm": peak_steer["arm"],
            "peak_steer_alpha": peak_steer["alpha_used"],
            "peak_steer_effect": peak_steer["concept_effect"],
            "peak_steer_z": peak_steer["effect_size_z"],
            "peak_steer_selective": peak_steer["selective"],
            "peak_steer_sign_as_expected": peak_steer["sign_as_expected"],
            # A peak that sits at a magnitude where even the random controls are already disrupting the
            # forward pass is not a steering result, and the headline must not hide that it is one.
            "peak_steer_degraded_at_every_alpha": peak_steer["degraded_at_every_alpha"],
            "peak_steer_full_random_band": peak_steer["full_random_band"],
            "any_locus_selective": any(r["selective"] for r in rows_mc),
            "any_locus_selective_and_signed": any(r["selective"] and r["sign_as_expected"]
                                                  for r in rows_mc),
            "peak_raw_movement_locus": peak_raw["locus"],
            "peak_raw_movement_effect": peak_raw["concept_effect"],
            "peak_raw_movement_selective": peak_raw["selective"],
            "n_live_loci": len({r["locus"] for r in live}),
            # With one live locus the two peaks coincide by arithmetic, not by evidence. The flag says
            # so rather than letting a 1-of-1 job look like a confirmed coincidence.
            "coincidence_is_informative": len({r["locus"] for r in live}) >= 2,
            "coincide_within_tested": peak_tested[0] == peak_steer["locus"],
            "coincide_with_global_peak": peak_all[0] == peak_steer["locus"],
            "selectivity_at_peak_steer_locus": probe_concept[peak_steer["locus"]]["selectivity"],
            "steer_effect_at_peak_decod_locus": max(steer_at_peak_decod) if steer_at_peak_decod else float("nan"),
            "n_selective_loci": len({r["locus"] for r in rows_mc if r["selective"]}),
            "n_probe_loci": len(probe_concept),
            "n_runs_pooled": len(runs_pooled), "runs_pooled": "|".join(runs_pooled),
            "runs_incompatible_note": incompat,
            "any_locus_short_random_band": any(not r["full_random_band"] for r in rows_mc),
        })

    for m in not_yet:
        print(f"not in the headline table yet: {m}")
    if join_failures:
        # Fail loudly. A locus that was steered but has no probe row means the two halves of the central
        # claim are not indexed the same way, and a table built on a short join would be wrong in a way
        # no reader could see. Stop rather than emit it.
        print("\nJOIN FAILURE:")
        for f in join_failures:
            print(f"  {f}")
        raise SystemExit("locus naming does not match between the intervention runs and the probe "
                         "results; refusing to write decodability_vs_steerability.csv")

    head_path = os.path.join(out_dir, "decodability_vs_steerability.csv")
    if head_rows:
        with open(head_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(head_rows[0]))
            w.writeheader()
            w.writerows(head_rows)
        print(f"\nwrote {len(head_rows)} rows to {head_path}\n")
        h = (f"{'model':13s} {'dom':8s} {'concept':13s} {'DONE':5s} "
             f"{'peak decod (tested)':22s} {'sel':>7s} | {'peak steer':16s} {'effect':>8s} {'z':>7s} "
             f"{'SEL':9s} {'same?':6s} {'live':5s}")
        print(h)
        print("-" * len(h))
        for r in head_rows:
            print(f"{r['model']:13s} {r['domain']:8s} {r['concept']:13s} "
                  f"{'yes' if r['job_done'] else 'PART':5s} "
                  f"{r['peak_decod_locus_tested']:22s} {r['peak_decod_selectivity_tested']:+7.4f} | "
                  f"{r['peak_steer_locus'] + '.' + r['peak_steer_arm']:16s} "
                  f"{r['peak_steer_effect']:+8.4f} {r['peak_steer_z']:+7.2f} "
                  f"{('yes' if r['peak_steer_sign_as_expected'] else 'WRONGSIGN') if r['peak_steer_selective'] else 'no':9s} "
                  f"{('YES' if r['coincide_within_tested'] else 'no') if r['coincidence_is_informative'] else 'n/a':6s} "
                  f"{r['n_loci_tested'] - r['n_loci_dead']}/{r['n_loci_tested']}")
        print("\n'same?' is n/a where only one live locus has been steered so far, because with one "
              "locus\nthe two peaks coincide by arithmetic rather than by evidence.")
        print("\nAcross all probe loci, not only the ones steered:")
        for r in head_rows:
            print(f"  {r['model']:13s} {r['concept']:13s} peak selectivity {r['peak_decod_locus_all_loci']:12s} "
                  f"{r['peak_decod_selectivity_all_loci']:+.4f}   peak AUROC {r['peak_decod_auroc_locus_all_loci']:12s} "
                  f"{r['peak_decod_auroc_all_loci']:.4f}   "
                  f"({r['n_probe_loci']} loci)   coincides with peak steering: "
                  f"{'YES' if r['coincide_with_global_peak'] else 'no'}")

    # ---- the caveat that travels with the numbers
    print("\n" + "=" * 100)
    print("HOW MAGNITUDES ARE SCALED")
    print("=" * 100)
    print("""src/intervene.py perturbs each token by alpha times THAT TOKEN's activation norm, measured by a
calibration forward pass at the steered module. alpha therefore means the same relative perturbation
at every token and every locus, and raw effect sizes are comparable across loci.

Earlier runs, archived under runs/_pooled_scale_v1/, scaled instead by the norm of the POOLED
activation cached by src/extract.py. Pooled and per-token norms stand in a ratio that varies with
depth, from 0.42x at vis.last to 1.64x at the connector on Lingshu-7B, so a nominal magnitude was a
different physical perturbation at each stage. Those runs still support the control-normalised
statistic, which compares a direction against random directions at the same locus and magnitude and
is unaffected by a per-locus rescaling, but not any comparison of raw effect size across loci.

One locus in the sweep is a STRUCTURAL ZERO and is there on purpose. Perturbing the visual positions
at the final decoder layer cannot reach the answer position, because no attention layer follows:
measured on Lingshu-7B, llm.L27.vis moves the logits by exactly 0.00000, where llm.L26.vis moves
them by 1.03 and llm.L27.ans by 11.11. A non-zero effect there would indict the measurement rather
than describe the model.

The control directions are run on a subset of the concept's magnitudes, recorded as control_alphas
in each run's meta.json. Every statistic here is evaluated on that subset, because a null band has
to exist at the magnitude where the comparison is made.""")

    missing_jobs = [r["run"] for r in runs if not r["rows"]]
    partial_jobs = [r["run"] for r in runs if r["rows"] and not r["done"]]
    print("\n" + "=" * 100)
    print("WHAT IS STILL MISSING")
    print("=" * 100)
    print(f"started, no locus written yet ({len(missing_jobs)}): {', '.join(missing_jobs) or 'none'}")
    print(f"partial, no DONE marker ({len(partial_jobs)}): {', '.join(partial_jobs) or 'none'}")
    print(f"finished ({n_done}): {', '.join(r['run'] for r in runs if r['done']) or 'none'}")


if __name__ == "__main__":
    main()
