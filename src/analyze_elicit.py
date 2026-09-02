"""Does a better readout recover what a probe reads? The elicitation test."""
from __future__ import annotations
import csv, glob, os
import numpy as np

DOM = {"lingshu7b":"medical","llavamed7b":"medical","qwen7b":"general",
       "llava15_7b":"general","internvl3_8b":"general"}
METHODS = ["yesno","yesno_alt","statement","negation"]

def probe_best(arch, concept):
    p=f"runs/probe_{arch}/probe_results.csv"
    if not os.path.exists(p): return float("nan"), float("nan")
    best, conn = -1, float("nan")
    for r in csv.DictReader(open(p)):
        if r["concept"]!=concept: continue
        try: v=float(r["real"])
        except: continue
        # answer-position loci are only valid for the finding the activations were cached under
        if r["locus"].endswith(".ans") and concept!="Effusion": continue
        best=max(best,v)
        if r["locus"]=="connector": conn=v
    return (best if best>=0 else float("nan")), conn

rows=[]
for d in sorted(glob.glob("runs/elicit_*")):
    if not os.path.isdir(d): continue
    arch=os.path.basename(d)[7:]
    f=os.path.join(d,"elicitation.csv")
    if not os.path.exists(f): continue
    for r in csv.DictReader(open(f)):
        pb, pc = probe_best(arch, r["concept"])
        rows.append({"arch":arch,"domain":DOM.get(arch,"?"),"concept":r["concept"],
                     **{m: float(r[f"auroc_{m}"]) for m in METHODS},
                     "best": float(r["auroc_best"]), "best_method": r["best_method"],
                     "gain": float(r["gain_over_yesno"]),
                     "probe_best": round(pb,4), "probe_connector": round(pc,4),
                     "gap_before": round(pb-float(r["auroc_yesno"]),4),
                     "gap_after":  round(pb-float(r["auroc_best"]),4),
                     "gap_closed_frac": round((float(r["auroc_best"])-float(r["auroc_yesno"]))
                                              /max(pb-float(r["auroc_yesno"]),1e-9),4)})

print("="*116)
print("ELICITATION: can a better readout recover what the probe reads?")
print("="*116)
print(f"{'model':13s} {'concept':14s} {'yesno':>7s} {'alt':>7s} {'stmt':>7s} {'neg':>7s} "
      f"{'BEST':>7s} {'probe':>7s} {'gap before':>11s} {'gap after':>10s} {'closed':>7s}")
for r in sorted(rows, key=lambda r:(r["arch"],r["concept"])):
    print(f"{r['arch']:13s} {r['concept']:14s} {r['yesno']:7.3f} {r['yesno_alt']:7.3f} "
          f"{r['statement']:7.3f} {r['negation']:7.3f} {r['best']:7.3f} {r['probe_best']:7.3f} "
          f"{r['gap_before']:+11.3f} {r['gap_after']:+10.3f} {r['gap_closed_frac']*100:6.0f}%")

g0=np.array([r["gap_before"] for r in rows]); g1=np.array([r["gap_after"] for r in rows])
gain=np.array([r["gain"] for r in rows])
print("\n"+"="*116)
print(f"cells                                        {len(rows)}")
print(f"median gain from the best elicitation        {np.median(gain):+.4f}")
print(f"largest gain in any cell                     {gain.max():+.4f}")
print(f"median probe-minus-answer gap, original      {np.median(g0):+.4f}")
print(f"median probe-minus-answer gap, best readout  {np.median(g1):+.4f}")
print(f"fraction of the gap closed (median)          {np.median([r['gap_closed_frac'] for r in rows])*100:.0f}%")
print(f"cells where the ORIGINAL question is best    {sum(r['best_method']=='yesno' for r in rows)}/{len(rows)}")
with open("runs/elicitation_summary.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nwrote runs/elicitation_summary.csv")
