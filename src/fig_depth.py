"""Figure 3: the conclusion does not depend on which locus we report.

An earlier version of this figure drew the three ingredients -- floor, trained probe, answer -- as raw
AUROC at every depth, in five narrow panels. It was the wrong figure twice over. Half of each panel was
empty, because AUROC has to span 0.5 to 1.0 while the data occupies 0.65 to 0.78; and the reader had to
do the subtraction themselves to see the claim. What the section actually asserts is a statement about
the DERIVED quantity, so that is what is plotted here: utilisation against depth, with zero drawn, and
the claim is that no curve crosses it.

    (a)  utilisation at every prompt-independent locus. Two models above zero, three below, 0 crossings
         in 190 model-by-locus points. Choosing a different locus would not have changed any conclusion.
    (b)  what the probe is reading instead. The acquisition variable outranks the best of the nine
         clinical findings at every depth in every model, which is the fact the floor is made of.

Curves are labelled at their right-hand end rather than in a legend, after Liu et al. (2022): with five
series whose names are the point, a legend makes the reader look up something the plot can say.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import figstyle; figstyle.use()
import figcheck

FIND = ["Effusion", "Pneumothorax", "Cardiomegaly", "Edema", "Atelectasis",
        "Consolidation", "Infiltration", "Mass", "Nodule"]
MODELS = [("lingshu7b", "Lingshu", figstyle.SKY),
          ("internvl3_8b", "InternVL3", "#2E86AB"),
          ("qwen7b", "Qwen2.5-VL", figstyle.ORANGE),
          ("llavamed7b", "LLaVA-Med", "#B5651D"),
          ("llava15_7b", "LLaVA-1.5", "#8a3d05")]
YLIM = (-215, 305)


def order(n):
    if n.startswith("vis.block"): return (0, int(n[9:]))
    if n == "vis.last":           return (0, 10_000)
    if n == "connector":          return (1, 0)
    if n.endswith(".vis"):        return (2, int(n.split(".L")[1].split(".")[0]))
    return None


def read(path, cols=("real",)):
    d = {}
    for r in csv.DictReader(open(path)):
        if order(r["locus"]) is None:
            continue
        try:
            d.setdefault(r["concept"], {})[r["locus"]] = tuple(float(r[c] or "nan") for c in cols)
        except (ValueError, KeyError):
            pass
    return d


def main():
    ans = {}
    for r in csv.DictReader(open("runs/utilisation_final.csv")):
        ans.setdefault(r["arch"], {})[r["concept"]] = float(r["answer"])

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.98), gridspec_kw={"wspace": 0.55,
                                                                    "width_ratios": [1.16, 1.0]})
    flips = tot = 0

    ax = axes[0]
    ax.axhspan(0, YLIM[1], color=figstyle.BG_GOOD, zorder=0)
    ax.axhspan(YLIM[0], 0, color=figstyle.BG_BAD, zorder=0)
    ax.axhline(0, color=figstyle.INK, lw=0.9, zorder=5)
    ax.text(0.012, 3, "the answer is at the untrained floor", transform=ax.get_yaxis_transform(),
            fontsize=figstyle.ANNOT - 1.1, color=figstyle.INK, va="bottom", ha="left")

    ends = []
    for arch, name, col in MODELS:
        T = read(f"runs/probe_{arch}/probe_results.csv")
        F = read(f"runs/rndprobe_{arch}/probe_results.csv")
        loci = sorted({l for c in T.values() for l in c} & {l for c in F.values() for l in c},
                      key=order)
        u = []
        for l in loci:
            v = [(ans[arch][c] - F[c][l][0]) / (T[c][l][0] - F[c][l][0]) * 100
                 for c in FIND if c in T and c in F and l in T[c] and l in F[c]
                 and c in ans.get(arch, {}) and T[c][l][0] - F[c][l][0] > 0.01]
            u.append(np.median(v) if v else np.nan)
        u = np.array(u)
        g = np.isfinite(u)
        tot += int(g.sum()); flips += int(np.sum(np.diff(np.sign(u[g])) != 0))
        x = np.linspace(0, 1, len(loci))
        ax.plot(x, np.clip(u, *YLIM), color=col, lw=1.15, zorder=4, clip_on=True)
        ends.append((np.clip(u[g][-1], YLIM[0] + 12, YLIM[1] - 12), name, col))

    # direct labels at the right end, pushed apart so none can collide
    ends.sort(key=lambda z: -z[0])
    minsep = (YLIM[1] - YLIM[0]) * 0.085
    for i in range(1, len(ends)):
        if ends[i - 1][0] - ends[i][0] < minsep:
            ends[i] = (ends[i - 1][0] - minsep, ends[i][1], ends[i][2])
    for y, name, col in ends:
        ax.text(1.02, y, name, fontsize=figstyle.ANNOT - 0.6, color=col, va="center", ha="left")

    ax.set_xlim(0, 1.0); ax.set_ylim(*YLIM)
    ax.set_yticks([-200, -100, 0, 100, 200, 300])
    ax.set_yticklabels(["$-200$", "$-100$", "0", "$+100$", "$+200$", "$+300$"])
    ax.set_ylabel("utilisation (\\%)")
    ax.set_xticks([])
    ax.set_xlabel("vision blocks $\\rightarrow$ connector $\\rightarrow$ LLM layers")
    figstyle.panel(ax, "a", "no curve crosses zero")

    ax = axes[1]
    for arch, name, col in MODELS:
        T = read(f"runs/probe_{arch}/probe_results.csv", ("real", "nuis_view_AP"))
        loci = sorted({l for c in T.values() for l in c}, key=order)
        best = np.array([np.nanmax([T[c][l][0] for c in FIND if c in T and l in T[c]] or [np.nan])
                         for l in loci])
        nui = np.array([np.nanmedian([T[c][l][1] for c in FIND if c in T and l in T[c]] or [np.nan])
                        for l in loci])
        x = np.linspace(0, 1, len(loci))
        ax.plot(x, nui, color=figstyle.MUTE, lw=1.0, zorder=3)
        ax.plot(x, best, color=col, lw=1.1, zorder=4)
    ax.set_ylim(0.735, 1.025); ax.set_xlim(0, 1.0)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xticks([])
    ax.set_ylabel("AUROC")
    ax.set_xlabel("vision blocks $\\rightarrow$ connector $\\rightarrow$ LLM layers")
    ax.text(0.03, 0.9885, "view position", fontsize=figstyle.ANNOT - 0.8,
            color=figstyle.MUTE, va="top", ha="left")
    # the band between the acquisition line and the findings is the one empty region in this panel
    ax.text(0.03, 0.947, "best of the nine clinical findings,\none line per model, colours as in (a)",
            fontsize=figstyle.ANNOT - 1.0, color=figstyle.INK, va="top", ha="left", linespacing=1.3)
    figstyle.panel(ax, "b", "what the probe reads instead")

    figcheck.audit(fig, "fig_depth")
    os.makedirs("figures/paper", exist_ok=True)
    fig.savefig("figures/paper/fig_depth.png", bbox_inches="tight"); plt.close(fig)
    print(f"fig_depth.png   sign flips {flips} in {tot} model-by-locus points")


if __name__ == "__main__":
    main()
