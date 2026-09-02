"""The D2 figures: reading against writing, and how causal separability decays with depth."""
from __future__ import annotations
import csv, glob, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__))))
from registry import REGISTRY

DOMAIN = {"lingshu7b": "medical", "llavamed7b": "medical", "qwen7b": "general",
          "llava15_7b": "general", "internvl3_8b": "general"}
MARK = {"lingshu7b": "o", "qwen7b": "o", "llavamed7b": "s", "llava15_7b": "s", "internvl3_8b": "^"}
COL = {"probe": "#c0392b", "cad": "#1f77b4", "cad_nospec": "#7f7f7f",
       "random": "#bdbdbd", "sham": "#d9a441"}


def read(p):
    if not os.path.exists(p):
        return []
    out = []
    for r in csv.DictReader(open(p)):
        for k, v in r.items():
            if k in ("locus", "direction"):
                continue
            try:
                r[k] = float(v) if v != "" else float("nan")
            except (TypeError, ValueError):
                r[k] = float("nan")
        out.append(r)
    return out


def rel_depth(arch, locus):
    """Position of a locus on a 0-1 axis, so 28- and 32-layer models are comparable."""
    n = REGISTRY[arch].n_llm_layers
    if locus == "vis.last":
        return -0.12
    if locus == "connector":
        return 0.0
    m = re.match(r"llm\.L(\d+)\.", locus)
    return (int(m.group(1)) + 1) / n if m else np.nan


cells, frontier, dose = [], [], []
for d in sorted(glob.glob("runs/cad_*")):
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "DONE")):
        continue
    arch, concept = os.path.basename(d)[4:].rsplit("_", 1)
    for r in read(os.path.join(d, "cad_results.csv")):
        cells.append({"arch": arch, "concept": concept, **r})
    for r in read(os.path.join(d, "cad_frontier.csv")):
        frontier.append({"arch": arch, "concept": concept, **r})
    for r in read(os.path.join(d, "cad_dose.csv")):
        dose.append({"arch": arch, "concept": concept, **r})
os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------- 1. reading against writing
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for tag in ("random", "sham", "probe", "cad"):
    s = [c for c in cells if c["direction"] == tag]
    if not s:
        continue
    ax.scatter([c["auroc"] for c in s], [c["effect"] for c in s], s=44, alpha=0.8,
               c=COL[tag], edgecolor="white", linewidth=0.6,
               label=f"{tag}  (n={len(s)})", zorder=3 if tag in ("probe", "cad") else 2)
# join each cell's probe and cad so the trade-off reads as a movement, not two clouds
by = defaultdict(dict)
for c in cells:
    by[(c["arch"], c["concept"], c["locus"])][c["direction"]] = c
for k, v in by.items():
    if "probe" in v and "cad" in v:
        ax.annotate("", xy=(v["cad"]["auroc"], v["cad"]["effect"]),
                    xytext=(v["probe"]["auroc"], v["probe"]["effect"]),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=0.6, alpha=0.55), zorder=1)
ax.axhline(0, color="k", lw=0.6, alpha=0.4)
ax.axvline(0.5, color="k", lw=0.6, alpha=0.4, ls=":")
ax.set_xlabel("how well the direction READS the concept   (held-out AUROC of $x\\cdot v$)")
ax.set_ylabel("how much the direction WRITES it   (|$\\Delta$P(yes)| at $\\alpha$=0.25)")
ax.set_title("Reading and writing are different directions", fontsize=12)
ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
fig.tight_layout(); fig.savefig("figures/d2_read_vs_write.png", dpi=175); plt.close(fig)
print("wrote figures/d2_read_vs_write.png")

# ---------------------------------------------------------------- 2. separability against depth
fig, ax = plt.subplots(figsize=(7.6, 4.6))
cad = [c for c in cells if c["direction"] == "cad" and np.isfinite(c.get("grad_align_mean", np.nan))]
for arch in sorted({c["arch"] for c in cad}):
    for concept in sorted({c["concept"] for c in cad if c["arch"] == arch}):
        s = [c for c in cad if c["arch"] == arch and c["concept"] == concept]
        xy = sorted(((rel_depth(arch, c["locus"]), c["grad_align_mean"]) for c in s),
                    key=lambda t: t[0])
        ax.plot([p[0] for p in xy], [p[1] for p in xy], marker=MARK.get(arch, "o"), ms=5, lw=1.5,
                alpha=0.85, ls="-" if DOMAIN[arch] == "medical" else "--",
                label=f"{arch} {concept}")
ax.axhline(1.0, color="#c0392b", lw=1.2, ls=":")
ax.text(0.02, 1.005, "no concept-specific direction can exist above this line",
        color="#c0392b", fontsize=8.5)
ax.set_xlabel("relative depth   (vis.last, connector = 0, then fraction of the language model)")
ax.set_ylabel("|cos($\\nabla$ concept effect, $\\nabla$ other-concept effect)|")
ax.set_title("Causal separability decays with depth and vanishes at the readout", fontsize=12)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.95)
fig.tight_layout(); fig.savefig("figures/d2_separability_depth.png", dpi=175); plt.close(fig)
print("wrote figures/d2_separability_depth.png")

# ---------------------------------------------------------------- 3. the frontier
if frontier:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for arch in sorted({f["arch"] for f in frontier}):
        for concept in sorted({f["concept"] for f in frontier if f["arch"] == arch}):
            for locus in sorted({f["locus"] for f in frontier
                                 if f["arch"] == arch and f["concept"] == concept}):
                s = sorted([f for f in frontier if f["arch"] == arch and f["concept"] == concept
                            and f["locus"] == locus], key=lambda r: r["t"])
                if len(s) < 3:
                    continue
                ax.plot([r["auroc"] for r in s], [r["effect"] for r in s], lw=1.2, alpha=0.6,
                        marker="o", ms=3, color="#1f77b4")
    ax.set_xlabel("reads the concept   (AUROC)")
    ax.set_ylabel("writes the concept   (|$\\Delta$P(yes)|)")
    ax.set_title("Interpolating from the probe normal to the fitted write direction", fontsize=12)
    fig.tight_layout(); fig.savefig("figures/d2_frontier.png", dpi=175); plt.close(fig)
    print("wrote figures/d2_frontier.png")

# ---------------------------------------------------------------- 4. dose response
if dose:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for tag in ("probe", "cad"):
        s = [d for d in dose if d["direction"] == tag]
        alphas = sorted({d["alpha"] for d in s})
        m = [np.mean([d["effect"] for d in s if d["alpha"] == a]) for a in alphas]
        sd = [np.std([d["effect"] for d in s if d["alpha"] == a]) for a in alphas]
        ax.errorbar(alphas, m, yerr=sd, marker="o", lw=2, capsize=3, color=COL[tag], label=tag)
    ax.set_xlabel("$\\alpha$   (fraction of the per-token activation norm)")
    ax.set_ylabel("|$\\Delta$P(yes)|")
    ax.set_title("Dose response, averaged over models, concepts and loci", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig("figures/d2_dose.png", dpi=175); plt.close(fig)
    print("wrote figures/d2_dose.png")
