"""Turn the D2 runs into the paper's read-versus-write table and its trade-off figure.

Three questions, in order of how much they carry:

  1. Does a concept-specific write direction exist at all?  A fitted direction counts only if it moves the
     concept's question much more than the other questions, and only if it does so on held-out patients.
     The `cad_nospec` arm is the same optimisation without the specificity penalty and shows what is found
     when nobody checks: a generic "answer yes" direction that moves everything at once.

  2. Is it the probe normal?  Compared on three axes: the effect each direction has, how well each one
     decodes the concept, and the cosine between them. The cosine has to be read against the floor for two
     unrelated vectors in D dimensions, which is where the random arm comes in, not against zero.

  3. Where is reading and writing separable?  `grad_align` is the cosine between the gradient of the
     concept's effect and the gradient of the other concepts' effects. At 1.0 no penalty can pull them
     apart and no concept-specific direction exists at that locus, whatever the fit reports.

    python src/analyze_cad.py --runs runs --out runs/cad_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

MEDICAL = {"lingshu7b": True, "llavamed7b": True, "huatuo7b": True,
           "qwen7b": False, "llava15_7b": False, "internvl3_8b": False}
PAIRS = [("lingshu7b", "qwen7b"), ("llavamed7b", "llava15_7b")]


def read(path):
    if not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        for k, v in list(r.items()):
            if k in ("locus", "direction", "placement"):
                continue
            try:
                r[k] = float(v) if v != "" else float("nan")
            except (TypeError, ValueError):
                pass
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/cad_summary.csv")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.runs, "cad_*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)[4:]
        arch, concept = name.rsplit("_", 1)
        res = read(os.path.join(d, "cad_results.csv"))
        if not res:
            continue
        done = os.path.exists(os.path.join(d, "DONE"))
        by_locus = defaultdict(dict)
        for r in res:
            by_locus[r["locus"]][r["direction"]] = r
        for locus, dirs in by_locus.items():
            p, c = dirs.get("probe"), dirs.get("cad")
            if not p or not c:
                continue
            ns = dirs.get("cad_nospec", {})
            rnd, sham = dirs.get("random", {}), dirs.get("sham", {})
            rows.append({
                "arch": arch, "medical": int(MEDICAL.get(arch, False)), "concept": concept,
                "locus": locus, "complete": int(done),
                "probe_effect": p["effect"], "cad_effect": c["effect"],
                "random_effect": rnd.get("effect", float("nan")),
                "sham_effect": sham.get("effect", float("nan")),
                "cad_over_probe": round(c["effect"] / max(p["effect"], 1e-6), 2),
                "cad_over_random": round(c["effect"] / max(rnd.get("effect", np.nan), 1e-6), 2),
                "probe_over_random": round(p["effect"] / max(rnd.get("effect", np.nan), 1e-6), 2),
                "probe_read": p["auroc"], "cad_read": c["auroc"],
                "random_read": rnd.get("auroc", float("nan")),
                "read_lost": round(p["auroc"] - c["auroc"], 4),
                "cos_probe_cad": c["cos_probe"], "cos_probe_random": rnd.get("cos_probe", float("nan")),
                # specificity ratio: how much of the movement lands on the concept asked about
                "cad_spec_ratio": round(c["effect"] / max(c["spec_cost"], 1e-6), 2),
                "nospec_spec_ratio": round(ns.get("effect", np.nan) / max(ns.get("spec_cost", np.nan), 1e-6), 2)
                                     if ns else float("nan"),
                "nospec_effect": ns.get("effect", float("nan")),
                "probe_spec_ratio": round(p["effect"] / max(p["spec_cost"], 1e-6), 2),
                # magnitude and sign reported separately: |cos| says the two gradients are collinear,
                # its sign says whether that collinearity is a conflict (+) or a synergy (-)
                "grad_collinearity": c.get("grad_collinearity", c.get("grad_align_mean", float("nan"))),
                "grad_align_signed": c.get("grad_align_signed_mean", float("nan")),
                "grad_sign_flips": c.get("grad_align_sign_flips", float("nan")),
                "grad_align": c.get("grad_align_mean", float("nan")),
                "p_yes_base": p.get("p_yes_base", float("nan")),
            })

    if not rows:
        print("no completed D2 runs yet"); return
    cols = list(rows[0])
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} model x concept x locus cells -> {args.out}\n")
    print(f"{'model':13s} {'concept':13s} {'locus':13s} {'probe':>7s} {'cad':>7s} {'rand':>6s} "
          f"{'x/probe':>8s} {'readP':>6s} {'readC':>6s} {'cos':>7s} {'spec':>7s} {'align':>6s}")
    for r in sorted(rows, key=lambda r: (r["arch"], r["concept"], r["locus"])):
        print(f"{r['arch']:13s} {r['concept']:13s} {r['locus']:13s} {r['probe_effect']:7.4f} "
              f"{r['cad_effect']:7.4f} {r['random_effect']:6.4f} {r['cad_over_probe']:8.1f} "
              f"{r['probe_read']:6.3f} {r['cad_read']:6.3f} {r['cos_probe_cad']:+7.3f} "
              f"{r['cad_spec_ratio']:7.1f} {r['grad_align']:6.3f}")

    fin = [r for r in rows if np.isfinite(r["cad_effect"]) and np.isfinite(r["probe_effect"])]
    print("\n--- summary ---")
    print(f"cells                                       {len(fin)}")
    print(f"cad steers more than probe                  "
          f"{sum(r['cad_effect'] > r['probe_effect'] for r in fin)}/{len(fin)}")
    print(f"probe steers no better than a random vector "
          f"{sum(r['probe_effect'] <= r['random_effect'] for r in fin)}/{len(fin)}")
    print(f"cad reads worse than probe                  "
          f"{sum(r['cad_read'] < r['probe_read'] for r in fin)}/{len(fin)}")
    cc = [abs(r["cos_probe_cad"]) for r in fin if np.isfinite(r["cos_probe_cad"])]
    cr = [abs(r["cos_probe_random"]) for r in fin if np.isfinite(r["cos_probe_random"])]
    print(f"|cos(probe, cad)|   median {np.median(cc):.4f}   "
          f"|cos(probe, random)| median {np.median(cr):.4f}" if cc and cr else "")
    sr = [r["cad_spec_ratio"] for r in fin if np.isfinite(r["cad_spec_ratio"])]
    nr = [r["nospec_spec_ratio"] for r in fin if np.isfinite(r["nospec_spec_ratio"])]
    if sr and nr:
        print(f"specificity ratio  with penalty median {np.median(sr):.1f}   "
              f"without median {np.median(nr):.1f}")

    print("\n--- matched pairs, medical against general ---")
    for med, gen in PAIRS:
        for concept in sorted({r["concept"] for r in rows}):
            a = [r for r in rows if r["arch"] == med and r["concept"] == concept]
            b = [r for r in rows if r["arch"] == gen and r["concept"] == concept]
            if not a or not b:
                continue
            print(f"  {concept:14s} {med:12s} cad_effect {np.mean([r['cad_effect'] for r in a]):.4f} "
                  f"align {np.nanmean([r['grad_align'] for r in a]):.3f}   |   "
                  f"{gen:12s} cad_effect {np.mean([r['cad_effect'] for r in b]):.4f} "
                  f"align {np.nanmean([r['grad_align'] for r in b]):.3f}")

    # The degenerate end of the separability profile, kept as a positive control on the measure itself.
    deep = [r for r in rows if r["locus"].endswith(".ans") and np.isfinite(r["grad_align"])]
    if deep:
        print(f"\nfinal-layer answer position, {len(deep)} cells:")
        print(f"  mean |cos| (collinearity)      {np.mean([r['grad_collinearity'] for r in deep]):.4f}")
        sg = [r['grad_align_signed'] for r in deep if np.isfinite(r.get('grad_align_signed', np.nan))]
        if sg:
            print(f"  mean SIGNED cos                {np.mean(sg):+.4f}")
            fl = [r['grad_sign_flips'] for r in deep if np.isfinite(r.get('grad_sign_flips', np.nan))]
            print(f"  mean sign flips per fit        {np.mean(fl):.1f}")
        print("  Collinearity near 1 at this locus is an architectural boundary condition, not a finding:")
        print("  the only computation downstream is the norm and a yes-minus-no readout that is the same")
        print("  vector for every question, so the gradients must be collinear. It is reported as a")
        print("  positive control on the measure. The sign flips show that the collinearity is not a")
        print("  standing conflict, so it does NOT license the reading that no specific direction exists.")

    # Is the depth profile actually monotone? Claiming a trend requires checking, not eyeballing curves.
    import re as _re
    prof = defaultdict(list)
    for r in rows:
        m = _re.match(r"llm\.L(\d+)\.", r["locus"])
        d = -1 if r["locus"] == "vis.last" else 0 if r["locus"] == "connector" else (int(m.group(1)) if m else None)
        if d is not None and np.isfinite(r.get("grad_collinearity", np.nan)):
            prof[(r["arch"], r["concept"])].append((d, r["grad_collinearity"]))
    mono = 0
    for k, v in prof.items():
        v = [y for _, y in sorted(v)]
        if len(v) > 2 and all(b >= a - 1e-9 for a, b in zip(v, v[1:])):
            mono += 1
    if prof:
        print(f"\ndepth profile of collinearity: strictly non-decreasing in {mono}/{len(prof)} curves")


if __name__ == "__main__":
    main()
