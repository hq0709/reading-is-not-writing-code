"""Put the two modalities side by side under the identical validity battery.

The second modality is not there for breadth. It is there to find out which of the chest-radiograph
findings are findings about medical vision-language models and which are findings about chest radiographs,
and the only way to tell is to run the same instrument twice and let it disagree with itself.
"""
from __future__ import annotations
import csv, glob, os
import numpy as np

# The nuisance is the STRONGEST measured one, not one picked by hand. Hard-coding site_Trunk hid
# site_Headandneck, which reaches 0.9465 on LLaVA-Med and there exceeds that model's best diagnosis
# (0.9442). Reading only the trunk indicator turned a 1-of-5 result into a 0-of-5 result.
MODS = {"CXR": ("runs/probe_{a}", ("nuis_view_AP",), "view position"),
        "DERM": ("runs/isicprobe_{a}", ("nuis_site_Trunk", "nuis_site_Headandneck"), "anatomic site")}
ARCHS = ["lingshu7b", "qwen7b", "llavamed7b", "llava15_7b", "internvl3_8b"]
DOMAIN = {"lingshu7b": "medical", "llavamed7b": "medical", "qwen7b": "general",
          "llava15_7b": "general", "internvl3_8b": "general"}


def load(path):
    if not os.path.exists(os.path.join(path, "probe_results.csv")):
        return []
    rows = []
    for r in csv.DictReader(open(os.path.join(path, "probe_results.csv"))):
        for k, v in r.items():
            if k not in ("concept", "locus"):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = float("nan")
        rows.append(r)
    return rows


print(f"{'modality':9s} {'model':13s} {'dom':8s} {'cells':>6s} {'med real':>9s} {'med sel':>8s} "
      f"{'max sel':>8s} {'max nuis':>9s} {'nuis>best concept?':>19s}")
out = []
for mod, (fmt, nuis_keys, nuis_name) in MODS.items():
    for a in ARCHS:
        rows = load(fmt.format(a=a))
        if not rows:
            continue
        real = np.array([r["real"] for r in rows])
        sel = np.array([r["selectivity"] for r in rows])
        nz = np.array([max((r.get(k, np.nan) for k in nuis_keys),
                              key=lambda v: -np.inf if v != v else v) for r in rows])
        beats = np.nanmax(nz) > np.nanmax(real)
        print(f"{mod:9s} {a:13s} {DOMAIN[a]:8s} {len(rows):6d} {np.nanmedian(real):9.3f} "
              f"{np.nanmedian(sel):+8.3f} {np.nanmax(sel):+8.3f} {np.nanmax(nz):9.3f} "
              f"{('YES ' + nuis_name) if beats else 'no':>19s}")
        out.append({"modality": mod, "model": a, "domain": DOMAIN[a], "cells": len(rows),
                    "median_real": round(float(np.nanmedian(real)), 4),
                    "median_selectivity": round(float(np.nanmedian(sel)), 4),
                    "max_selectivity": round(float(np.nanmax(sel)), 4),
                    "max_nuisance": round(float(np.nanmax(nz)), 4),
                    "max_real": round(float(np.nanmax(real)), 4),
                    "nuisance_beats_best_concept": int(beats),
                    "nuisance_probes_read": "|".join(nuis_keys)})

with open("runs/modality_comparison.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

print("\n--- the claim that does not transfer ---")
for mod in MODS:
    sub = [r for r in out if r["modality"] == mod]
    if not sub:
        continue
    print(f"{mod:5s}  median selectivity {np.median([r['median_selectivity'] for r in sub]):+.3f}  "
          f"| best clinical concept {np.max([r['max_real'] for r in sub]):.3f}  "
          f"| best nuisance {np.max([r['max_nuisance'] for r in sub]):.3f}  "
          f"| nuisance wins in {sum(r['nuisance_beats_best_concept'] for r in sub)}/{len(sub)} models")

print("\n--- medical against general, within modality ---")
for mod in MODS:
    for dom in ("medical", "general"):
        sub = [r for r in out if r["modality"] == mod and r["domain"] == dom]
        if sub:
            print(f"  {mod:5s} {dom:8s} median selectivity "
                  f"{np.median([r['median_selectivity'] for r in sub]):+.3f}  "
                  f"({', '.join(r['model'] for r in sub)})")
print("\nwrote runs/modality_comparison.csv")
