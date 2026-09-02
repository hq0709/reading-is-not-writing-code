"""The floor is a property of the data, not of the network.

This figure exists because the account makes a prediction and the prediction can be wrong. If $F$ is what
any network of that shape provides -- if it is the image statistics being read through an arbitrary map --
then three architecturally unrelated untrained networks, sharing no weights, no width, no depth and no
connector, must nonetheless agree about WHICH findings are decodable. If instead $F$ were an accident of a
particular random initialisation, they would agree no better than chance.

    (a)  the prediction, drawn. Three unrelated random maps of the same images; the account says the
         ranking they induce over findings is a fact about the images and therefore common to all three.
    (b)  the test. Every pair of architectures, nine findings, rank against rank, with the identity line
         the prediction and the shaded band what random ranking produces.
    (c)  the null. Mean pairwise Spearman under 20,000 random rankings, against the observed value.

Colour encodes the architecture pair and shape encodes the finding, so a reader can follow one finding
across all three panels without a second legend.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np
from scipy import stats
import figstyle; figstyle.use()
import figcheck

FIND = ["Effusion", "Pneumothorax", "Cardiomegaly", "Edema", "Atelectasis",
        "Consolidation", "Infiltration", "Mass", "Nodule"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*", "<"]
REP = [("lingshu7b", "Lingshu / Qwen2.5-VL"), ("llavamed7b", "LLaVA-Med / LLaVA-1.5"),
       ("internvl3_8b", "InternVL3")]
PAIRCOL = [figstyle.ORANGE, figstyle.SKY, figstyle.VIOLET]
NPERM = 20_000


def main():
    U = {}
    for r in csv.DictReader(open("runs/utilisation_final.csv")):
        U[(r["arch"], r["concept"])] = float(r["floor"])
    V = {n: np.array([U[(a, c)] for c in FIND]) for a, n in REP}
    R = {k: stats.rankdata(v) for k, v in V.items()}
    names = [n for _, n in REP]

    def meanrho(d):
        return float(np.mean([stats.spearmanr(d[names[i]], d[names[j]])[0]
                              for i in range(3) for j in range(i + 1, 3)]))
    obs = meanrho(V)
    rng = np.random.default_rng(0)
    null = np.array([meanrho({k: rng.permutation(v) for k, v in V.items()}) for _ in range(NPERM)])
    pval = float((null >= obs).mean())

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.72),
                             gridspec_kw={"wspace": 0.40, "width_ratios": [1.02, 1.0, 0.88]})

    # ------------------------------------------------------ (a) the prediction, as a mechanism sketch
    ax = axes[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.40), 0.20, 0.30, fc="#eceff3", ec=figstyle.HAIR, lw=0.6))
    ax.text(0.12, 0.55, "the same\nimages", ha="center", va="center",
            fontsize=figstyle.ANNOT - 0.8, linespacing=1.25)
    for k, c in enumerate(PAIRCOL):
        yy = 0.82 - 0.30 * k
        ax.add_patch(Circle((0.47, yy), 0.055, fc="white", ec=c, lw=1.0))
        ax.text(0.47, yy, "rnd", ha="center", va="center", fontsize=figstyle.ANNOT - 1.8, color=c)
        ax.add_patch(FancyArrowPatch((0.235, 0.55), (0.405, yy), arrowstyle="->",
                                     mutation_scale=6, lw=0.7, color=figstyle.MUTE,
                                     connectionstyle="arc3,rad=0.18"))
        ax.add_patch(FancyArrowPatch((0.535, yy), (0.70, 0.55), arrowstyle="->",
                                     mutation_scale=6, lw=0.7, color=c,
                                     connectionstyle="arc3,rad=0.18"))
    ax.text(0.47, 0.06, "three unrelated\nrandom networks", ha="center", va="center",
            fontsize=figstyle.ANNOT - 1.4, color=figstyle.MUTE, linespacing=1.25)
    ax.add_patch(plt.Rectangle((0.72, 0.40), 0.26, 0.30, fc=figstyle.BG_GOOD,
                               ec=figstyle.SKY, lw=0.8))
    ax.text(0.85, 0.55, "one ranking\nof findings", ha="center", va="center",
            fontsize=figstyle.ANNOT - 0.8, linespacing=1.25)
    figstyle.panel(ax, "a", "the prediction", y=1.005)

    # ------------------------------------------------------ (b) rank against rank
    ax = axes[1]
    ax.plot([0.4, 9.6], [0.4, 9.6], color=figstyle.INK, lw=0.8, zorder=4)
    ax.text(0.8, 9.4, "identical\nranking", fontsize=figstyle.ANNOT - 1.4, color=figstyle.INK,
            ha="left", va="top", linespacing=1.25)
    k = 0
    for i in range(3):
        for j in range(i + 1, 3):
            for f in range(len(FIND)):
                ax.scatter(R[names[i]][f], R[names[j]][f], marker=MK[f], s=11,
                           facecolor=PAIRCOL[k], edgecolor="white", linewidths=0.35, zorder=3)
            k += 1
    ax.set_xlim(0.3, 9.7); ax.set_ylim(0.3, 9.7)
    ax.set_xticks([1, 5, 9]); ax.set_yticks([1, 5, 9])
    ax.set_xlabel("rank, one architecture"); ax.set_ylabel("rank, another")
    figstyle.panel(ax, "b", "every pair, nine findings")

    # ------------------------------------------------------ (c) against the null
    ax = axes[2]
    ax.hist(null, bins=np.linspace(-1, 1, 46), color=figstyle.HAIR, edgecolor="white",
            linewidth=0.3, zorder=2)
    ax.axvline(obs, color=figstyle.ORANGE, lw=1.5, zorder=5)
    ax.text(obs - 0.06, ax.get_ylim()[1] * 0.93, f"observed\n{obs:+.2f}", fontsize=figstyle.ANNOT - 1.0,
            color=figstyle.ORANGE, ha="right", va="top", linespacing=1.25)
    ax.text(-0.95, ax.get_ylim()[1] * 0.93, "random\nranking", fontsize=figstyle.ANNOT - 1.0,
            color=figstyle.MUTE, ha="left", va="top", linespacing=1.25)
    ax.set_xlim(-1.05, 1.05); ax.set_xticks([-1, 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("mean pairwise Spearman $\\rho$")
    figstyle.panel(ax, "c", f"$p = {pval:.4f}$")

    figcheck.audit(fig, "fig_floor")
    os.makedirs("figures/paper", exist_ok=True)
    fig.savefig("figures/paper/fig_floor.png", bbox_inches="tight"); plt.close(fig)
    print(f"fig_floor.png   rho={obs:.3f}  null mean={null.mean():+.3f}  95th={np.percentile(null,95):+.3f}  p={pval:.5f}")


if __name__ == "__main__":
    main()
