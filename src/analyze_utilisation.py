"""Utilisation: how much of what training adds to the representation does the model's own output use?

    utilisation = (behaviour - untrained floor) / (trained probe - untrained floor)

The untrained floor is the same probe, at the same loci, on the same architecture with randomly
initialised weights. It is what makes the comparison well-posed: a probe fitted on 18,212 labelled
radiographs beats a zero-shot prompt for reasons that have nothing to do with what the model learned, and
the floor carries exactly that advantage. Subtracting it from numerator and denominator alike leaves the
part attributable to training.

Zero for an untrained network by construction. One for a model whose own answer is as good as any linear
readout of its own activations.
"""
from __future__ import annotations
import csv, glob, os
import numpy as np

DOM = {"lingshu7b":"medical","llavamed7b":"medical","qwen7b":"general",
       "llava15_7b":"general","internvl3_8b":"general"}


def probe_table(path, concept, prompt_concept="Effusion"):
    """Best and connector AUROC over prompt-independent loci."""
    if not os.path.exists(path):
        return {}
    best, conn = -1.0, float("nan")
    for r in csv.DictReader(open(path)):
        if r["concept"] != concept:
            continue
        # answer-position activations were cached under one question and are valid only for it
        if r["locus"].endswith(".ans") and concept != prompt_concept:
            continue
        try:
            v = float(r["real"])
        except (TypeError, ValueError):
            continue
        best = max(best, v)
        if r["locus"] == "connector":
            conn = v
    return {"best": best if best >= 0 else float("nan"), "connector": conn}


rows = []
for d in sorted(glob.glob("runs/rndprobe_*")):
    if not os.path.isdir(d):
        continue
    arch = os.path.basename(d)[9:]
    el = f"runs/elicit_{arch}/elicitation.csv"
    beh = f"runs/behav_{arch}/behaviour_auroc.csv"
    behav = {r["concept"]: float(r["behavioural_auroc"]) for r in csv.DictReader(open(beh))} \
        if os.path.exists(beh) else {}
    best_el = {r["concept"]: float(r["auroc_best"]) for r in csv.DictReader(open(el))} \
        if os.path.exists(el) else {}
    for concept in sorted(behav):
        rnd = probe_table(os.path.join(d, "probe_results.csv"), concept)
        trn = probe_table(f"runs/probe_{arch}/probe_results.csv", concept)
        if not rnd or not trn or not np.isfinite(rnd["best"]) or not np.isfinite(trn["best"]):
            continue
        b_raw = behav[concept]
        b_best = best_el.get(concept, b_raw)          # best readout where the elicitation sweep ran
        for tag, p_t, p_r in (("best_locus", trn["best"], rnd["best"]),
                              ("connector", trn["connector"], rnd["connector"])):
            if not np.isfinite(p_t) or not np.isfinite(p_r):
                continue
            head = p_t - p_r
            rows.append({
                "arch": arch, "domain": DOM.get(arch, "?"), "concept": concept, "variant": tag,
                "floor_untrained": round(p_r, 4), "probe_trained": round(p_t, 4),
                "training_adds": round(head, 4),
                "behaviour_raw": round(b_raw, 4), "behaviour_best_readout": round(b_best, 4),
                "used_raw": round(b_raw - p_r, 4), "used_best": round(b_best - p_r, 4),
                "utilisation_raw": round((b_raw - p_r) / head, 4) if head > 0.01 else float("nan"),
                "utilisation_best": round((b_best - p_r) / head, 4) if head > 0.01 else float("nan"),
                "elicitation_measured": int(concept in best_el),
            })

if not rows:
    raise SystemExit("no cells with all three numbers yet")

with open("runs/utilisation.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

for variant in ("connector", "best_locus"):
    sub = [r for r in rows if r["variant"] == variant]
    if not sub:
        continue
    print("=" * 108)
    print(f"UTILISATION at {variant}")
    print("=" * 108)
    print(f"{'model':13s} {'concept':14s} {'untrained':>10s} {'trained':>8s} {'adds':>7s} "
          f"{'answer':>8s} {'used':>7s} {'UTILISATION':>12s}")
    for r in sorted(sub, key=lambda r: (r["arch"], -r["training_adds"])):
        u = r["utilisation_best"]
        print(f"{r['arch']:13s} {r['concept']:14s} {r['floor_untrained']:10.3f} "
              f"{r['probe_trained']:8.3f} {r['training_adds']:+7.3f} {r['behaviour_best_readout']:8.3f} "
              f"{r['used_best']:+7.3f} {(f'{u*100:.0f}%' if np.isfinite(u) else 'n/a'):>12s}")
    u = np.array([r["utilisation_best"] for r in sub if np.isfinite(r["utilisation_best"])])
    if u.size:
        print(f"\n  cells {u.size}   median utilisation {np.median(u)*100:.0f}%   "
              f"range {u.min()*100:.0f}% to {u.max()*100:.0f}%   "
              f"below 50% in {int((u < 0.5).sum())}/{u.size}")
        print(f"  median amount training adds  {np.median([r['training_adds'] for r in sub]):+.3f} AUROC")
    print()
print("wrote runs/utilisation.csv")
