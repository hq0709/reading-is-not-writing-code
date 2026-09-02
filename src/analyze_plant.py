"""D3: which candidate direction points where the representation actually moves?

Planting a lesion of known strength gives what no other experiment here has: a ground-truth causal
displacement. `dh(beta) = h(image + beta*lesion) - h(image)` is the direction the representation travels
when the finding is genuinely put into the input. Every candidate direction can then be scored against it.

The gate comes first and is not negotiable. If the model's answer does not respond monotonically to the
planted lesion at the anatomical site, or if it responds just as much to the same blob dropped outside the
thorax, then the lesion is not being read as the finding and no alignment number below means anything.
"""
from __future__ import annotations
import csv, glob, os
import numpy as np

DOMAIN = {"lingshu7b": "medical", "llavamed7b": "medical", "qwen7b": "general",
          "llava15_7b": "general", "internvl3_8b": "general"}


def load(p):
    if not os.path.exists(p):
        return []
    rows = []
    for r in csv.DictReader(open(p)):
        for k, v in r.items():
            if k in ("placement", "locus", "question"):
                continue
            try:
                r[k] = float(v) if v != "" else float("nan")
            except (TypeError, ValueError):
                r[k] = float("nan")
        rows.append(r)
    return rows


def monotone(xs, ys, tol=0.005):
    """Non-decreasing in the intended direction, allowing tol-sized wobble."""
    o = np.argsort(xs); y = np.array(ys)[o]
    return bool(np.all(np.diff(y) > -tol))


gate, align_rows = [], []
for d in sorted(glob.glob("runs/plant_*")):
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "DONE")):
        continue
    name = os.path.basename(d)[6:]
    arch, concept = name.rsplit("_", 1)
    beh = load(os.path.join(d, "behaviour.csv"))
    ali = load(os.path.join(d, "alignment.csv"))
    if not beh:
        continue
    q0 = beh[0]["question"] if beh else concept
    qs = sorted({r["question"] for r in beh})
    target = [q for q in qs if q.lower() == concept.lower()]
    target = target[0] if target else q0

    def curve(pl, q):
        s = sorted([r for r in beh if r["placement"] == pl and r["question"] == q],
                   key=lambda r: r["beta"])
        return [r["beta"] for r in s], [r["p_yes"] for r in s]

    ab, ay = curve("anatomical", target)
    cb, cy = curve("control_location", target)
    if not ab:
        continue
    a_rise = ay[-1] - ay[0]
    c_rise = (cy[-1] - cy[0]) if cy else float("nan")
    # how much of the movement is specific to the concept's own question
    other_rise = []
    for q in qs:
        if q == target:
            continue
        _, oy = curve("anatomical", q)
        if oy:
            other_rise.append(oy[-1] - oy[0])
    gate.append({
        "arch": arch, "domain": DOMAIN.get(arch, "?"), "concept": concept,
        "p_yes_base": round(ay[0], 4), "anatomical_rise": round(a_rise, 4),
        "control_rise": round(c_rise, 4),
        # Magnitudes, not signed values: a location control that drifts the other way is still a control
        # whose movement has to be small compared with the anatomical one, and dividing by a negative
        # number turned "the control barely moved" into a negative ratio that read like a failure.
        "site_ratio": round(abs(a_rise) / max(abs(c_rise), 1e-6), 1),
        "other_rise_mean": round(float(np.mean(other_rise)), 4) if other_rise else float("nan"),
        "concept_ratio": round(abs(a_rise) / max(float(np.mean(np.abs(other_rise))), 1e-6), 1) if other_rise else float("nan"),
        "monotone": int(monotone(ab, ay)),
        # The direction of the response has to match the sign of the lesion, so a_rise must be positive
        # for an added opacity. A cell that moves the answer the wrong way has not planted the finding.
        "PASS": int(monotone(ab, ay) and a_rise > 0.02 and abs(a_rise) > 3 * max(abs(c_rise), 1e-6)),
    })
    for r in ali:
        if r["placement"] != "anatomical":
            continue
        align_rows.append({"arch": arch, "domain": DOMAIN.get(arch, "?"), "concept": concept, **r})

print("=" * 104)
print("D3 GATE: does the planted lesion behave like the finding?")
print("=" * 104)
print(f"{'model':13s} {'concept':13s} {'base':>6s} {'anat rise':>10s} {'ctrl rise':>10s} {'site x':>8s} "
      f"{'other rise':>11s} {'concept x':>10s} {'mono':>5s} {'PASS':>5s}")
for g in gate:
    print(f"{g['arch']:13s} {g['concept']:13s} {g['p_yes_base']:6.3f} {g['anatomical_rise']:+10.4f} "
          f"{g['control_rise']:+10.4f} {g['site_ratio']:8.1f} {g['other_rise_mean']:+11.4f} "
          f"{g['concept_ratio']:10.1f} {g['monotone']:5d} {'yes' if g['PASS'] else 'NO':>5s}")
npass = sum(g["PASS"] for g in gate)
print(f"\n{npass}/{len(gate)} cells pass the gate "
      f"(monotone, rise > 0.02, and more than 3x the location control)")

print("\n" + "=" * 104)
print("D3 ALIGNMENT: cosine between the displacement the lesion CAUSES and each candidate direction")
print("=" * 104)
ok = {(g["arch"], g["concept"]) for g in gate if g["PASS"]}
A = [r for r in align_rows if (r["arch"], r["concept"]) in ok]
if not A:
    print("no cells passed the gate; alignment is not interpretable")
else:
    loci = sorted({r["locus"] for r in A}, key=lambda s: (s != "vis.last", s != "connector", s))
    print(f"{'locus':14s} {'n':>4s} {'cos(dh,probe)':>14s} {'cos(dh,cad)':>12s} {'cos(dh,sham)':>13s} "
          f"{'cos(dh,random)':>15s} {'cad/probe':>10s}")
    for lo in loci:
        s = [r for r in A if r["locus"] == lo]
        def m(k):
            v = [abs(r[k]) for r in s if k in r and np.isfinite(r.get(k, np.nan))]
            return float(np.mean(v)) if v else float("nan")
        p, c, sh, rd = m("cos_probe"), m("cos_cad"), m("cos_sham"), m("cos_random")
        print(f"{lo:14s} {len(s):4d} {p:14.4f} {c:12.4f} {sh:13.4f} {rd:15.4f} "
              f"{(c / p if p else float('nan')):10.1f}")

    allp = [abs(r["cos_probe"]) for r in A if np.isfinite(r.get("cos_probe", np.nan))]
    allc = [abs(r["cos_cad"]) for r in A if np.isfinite(r.get("cos_cad", np.nan))]
    allr = [abs(r["cos_random"]) for r in A if np.isfinite(r.get("cos_random", np.nan))]
    print(f"\noverall mean |cos|:  probe {np.mean(allp):.4f}   cad {np.mean(allc):.4f}   "
          f"random {np.mean(allr):.4f}   (n={len(allp)} probe cells, {len(allc)} cad cells)")
    pairs = [(abs(r["cos_probe"]), abs(r["cos_cad"])) for r in A
             if np.isfinite(r.get("cos_probe", np.nan)) and np.isfinite(r.get("cos_cad", np.nan))]
    if pairs:
        wins = sum(c > p for p, c in pairs)
        print(f"cad closer to the true displacement than the probe normal in {wins}/{len(pairs)} cells")
    pr = [(abs(r["cos_probe"]), abs(r["cos_random"])) for r in A
          if np.isfinite(r.get("cos_probe", np.nan)) and np.isfinite(r.get("cos_random", np.nan))]
    if pr:
        print(f"probe normal closer than a RANDOM vector in {sum(p > q for p, q in pr)}/{len(pr)} cells")

for name, data in (("runs/plant_gate.csv", gate), ("runs/plant_alignment.csv", align_rows)):
    if data:
        cols, seen = [], set()
        for d in data:
            for k in d:
                if k not in seen:
                    seen.add(k); cols.append(k)
        with open(name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval=""); w.writeheader(); w.writerows(data)
        print(f"wrote {name}")
