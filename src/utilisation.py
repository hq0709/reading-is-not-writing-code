"""The paper's central quantity, and the table behind every figure.

A probe fitted on 18,212 labelled radiographs beats a zero-shot prompt for reasons that have nothing to do
with what the model learned. The comparison is only well posed against a floor: the same probe family, at
the same locus, on the same architecture with randomly initialised weights. In the language of predictive
V-information (Xu et al. 2020; Ethayarajh et al. 2022), the floor is the V-usable information a linear
probe family can extract from an untrained network, the trained probe is the same quantity after training,
and their difference is the usable information training created.

    utilisation = (answer - floor) / (trained - floor)

0 when the model's own output is no better than a probe on an untrained network.
1 when it is as good as any linear readout of its own activations.
Negative when it is worse than the untrained floor, which is not a pathology of the measure: it happens.

Architectures that share weights-shape share a floor, so Lingshu and Qwen use one untrained model and
LLaVA-Med and LLaVA-1.5 the other.
"""
from __future__ import annotations
import csv
import os
import numpy as np

DOMAIN = {"lingshu7b": "medical", "llavamed7b": "medical", "qwen7b": "general",
          "llava15_7b": "general", "internvl3_8b": "general"}
NICE = {"lingshu7b": "Lingshu-7B", "llavamed7b": "LLaVA-Med-7B", "qwen7b": "Qwen2.5-VL-7B",
        "llava15_7b": "LLaVA-1.5-7B", "internvl3_8b": "InternVL3-8B"}
PAIRS = [("lingshu7b", "qwen7b"), ("llavamed7b", "llava15_7b")]
LOCUS = "connector"                       # fixed in advance, and upstream of the language model


def probe_at(path, locus=LOCUS):
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        if r["locus"] != locus:
            continue
        try:
            out[r["concept"]] = {"auroc": float(r["real"]), "sel": float(r["selectivity"]),
                                 "view": float(r.get("nuis_view_AP", "nan"))}
        except (TypeError, ValueError):
            pass
    return out


def build():
    rows = []
    for arch in DOMAIN:
        floor = probe_at(f"runs/rndprobe_{arch}/probe_results.csv")
        trained = probe_at(f"runs/probe_{arch}/probe_results.csv")
        beh_path = f"runs/behav_{arch}/behaviour_auroc.csv"
        if not (floor and trained and os.path.exists(beh_path)):
            continue
        beh = {r["concept"]: r for r in csv.DictReader(open(beh_path))}
        el = {}
        if os.path.exists(f"runs/elicit_{arch}/elicitation.csv"):
            el = {r["concept"]: float(r["auroc_best"])
                  for r in csv.DictReader(open(f"runs/elicit_{arch}/elicitation.csv"))}
        for c, br in beh.items():
            if c not in floor or c not in trained:
                continue
            f, t = floor[c]["auroc"], trained[c]["auroc"]
            b_raw = float(br["behavioural_auroc"])
            b = el.get(c, b_raw)
            add = t - f
            rows.append({
                "arch": arch, "model": NICE[arch], "domain": DOMAIN[arch], "concept": c,
                "floor": round(f, 4), "trained": round(t, 4), "adds": round(add, 4),
                "answer_raw": round(b_raw, 4), "answer": round(b, 4),
                "utilisation": round((b - f) / add, 4) if add > 0.02 else float("nan"),
                "below_floor": int(b < f),
                "floor_selectivity": round(floor[c]["sel"], 4),
                "trained_selectivity": round(trained[c]["sel"], 4),
                "floor_view_auroc": round(floor[c]["view"], 4),
                "trained_view_auroc": round(trained[c]["view"], 4),
                "elicitation_measured": int(c in el),
                "yes_rate": float(br["yes_rate"]), "frac_pinned": float(br["frac_pinned"]),
            })
    return rows


if __name__ == "__main__":
    rows = build()
    # Refuse to touch the output unless there is something to write. Opening it first truncates it, so a
    # run in an environment without the cached activations used to destroy the released result table
    # before failing -- which is exactly the environment someone cloning the repository is in.
    if not rows:
        raise SystemExit(
            "no cells could be built: runs/probe_<arch>/ and runs/rndprobe_<arch>/ are missing.\n"
            "runs/utilisation_final.csv in this repository is the released table; regenerating it needs\n"
            "the cached activations, which are not redistributed. See the README.")
    tmp = "runs/utilisation_final.csv.tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(tmp, "runs/utilisation_final.csv")   # atomic: a crash cannot leave a half-written table

    print(f"{'model':15s} {'domain':8s} {'cells':>6s} {'med util':>9s} {'below floor':>12s} "
          f"{'med floor':>10s} {'med adds':>9s} {'med answer':>11s}")
    for a in sorted(DOMAIN, key=lambda a: -np.nanmedian(
            [r["utilisation"] for r in rows if r["arch"] == a] or [np.nan])):
        s = [r for r in rows if r["arch"] == a]
        if not s:
            continue
        u = [r["utilisation"] for r in s if np.isfinite(r["utilisation"])]
        print(f"{NICE[a]:15s} {s[0]['domain']:8s} {len(u):6d} {np.median(u)*100:8.0f}% "
              f"{sum(r['below_floor'] for r in s):8d}/{len(s):<3d} "
              f"{np.median([r['floor'] for r in s]):10.3f} "
              f"{np.median([r['adds'] for r in s]):+9.3f} "
              f"{np.median([r['answer'] for r in s]):11.3f}")
    u = np.array([r["utilisation"] for r in rows if np.isfinite(r["utilisation"])])
    print(f"\nall: {len(rows)} cells, {len(u)} measurable, median utilisation {np.median(u)*100:.0f}%, "
          f"answer below the untrained floor in {sum(r['below_floor'] for r in rows)}/{len(rows)}")
    print("\nmatched architecture pairs (same untrained floor, so the difference is training alone):")
    for med, gen in PAIRS:
        a = [r["utilisation"] for r in rows if r["arch"] == med and np.isfinite(r["utilisation"])]
        b = [r["utilisation"] for r in rows if r["arch"] == gen and np.isfinite(r["utilisation"])]
        if a and b:
            print(f"  {NICE[med]:15s} {np.median(a)*100:+6.0f}%   vs   {NICE[gen]:15s} {np.median(b)*100:+6.0f}%")
    print(f"\nprobe AUROC spread across models  {np.ptp([np.median([r['trained'] for r in rows if r['arch']==a]) for a in DOMAIN if any(r['arch']==a for r in rows)]):.3f}")
    print(f"answer AUROC spread across models {np.ptp([np.median([r['answer'] for r in rows if r['arch']==a]) for a in DOMAIN if any(r['arch']==a for r in rows)]):.3f}")
    print("\nwrote runs/utilisation_final.csv")
