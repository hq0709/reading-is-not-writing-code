"""Figure 2: what each model reads and what it answers, finding by finding.

The form is taken from the results figures of CVPR-school papers: three panels sharing a y axis, a full
boxed frame with a pale grid, one series per quantity distinguished by both colour and marker shape, the
series carrying the argument drawn thicker and brighter, a grey dashed reference line with a small inline
label for the bound, and a framed legend in the corner the data leaves empty.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle
figstyle.use()

PANELS = [("lingshu7b", "Lingshu-7B"), ("qwen7b", "Qwen2.5-VL-7B"), ("llavamed7b", "LLaVA-Med-7B")]
FIND = ["Effusion", "Pneumothorax", "Cardiomegaly", "Edema", "Atelectasis",
        "Consolidation", "Infiltration", "Mass", "Nodule"]
ABB = {"Effusion": "Eff", "Pneumothorax": "Pnx", "Cardiomegaly": "Cmg", "Edema": "Edm",
       "Atelectasis": "Atl", "Consolidation": "Con", "Infiltration": "Inf", "Mass": "Mas",
       "Nodule": "Nod"}


def load():
    d = {}
    for r in csv.DictReader(open("runs/utilisation_final.csv")):
        for k, v in r.items():
            if k not in ("arch", "model", "domain", "concept"):
                try: r[k] = float(v)
                except (TypeError, ValueError): r[k] = float("nan")
        d[(r["arch"], r["concept"])] = r
    return d


def main():
    D = load()
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.30), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    x = np.arange(len(FIND))
    for k, (arch, name) in enumerate(PANELS):
        ax = axes[k]
        F = [D[(arch, c)]["floor"] for c in FIND]
        T = [D[(arch, c)]["trained"] for c in FIND]
        B = [D[(arch, c)]["answer"] for c in FIND]
        figstyle.series(ax, x, F, "untrained floor $F$", 2, color=figstyle.VIOLET)
        figstyle.series(ax, x, T, "probe, trained $T$", 0, color=figstyle.ORANGE)
        figstyle.series(ax, x, B, "the model's answer $B$", 3, emphasis=True, color=figstyle.SKY)
        ax.set_title(name, fontsize=figstyle.TITLE)
        ax.set_xticks(x)
        ax.set_xticklabels([ABB[c] for c in FIND], rotation=90, ha="center",
                           fontsize=figstyle.BASE - 1.2)
        ax.set_ylim(0.465, 0.908)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        if k == 0:
            ax.set_ylabel("AUROC")
    # One legend for three panels, ABOVE them, in a single frameless row. This is the arrangement
    # Wei et al. (2022) use for the eight-panel emergence figure and Schaeffer et al. (2023) reproduce:
    # a key that governs every panel reads as a key when it sits over all of them, and once it is outside
    # the axes the frame has nothing left to separate it from. Below the panels it collided with the
    # rotated tick labels; inside panel 1 it cost a fifth of the plotted range in all three.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
               columnspacing=2.2, handletextpad=0.5, borderaxespad=0.0)
    # one x-axis label for the whole grid, centred under it, as in both of those figures
    fig.supxlabel("finding", fontsize=figstyle.LABEL, y=-0.055)
    os.makedirs("figures/paper", exist_ok=True)
    fig.savefig("figures/paper/fig_profile.png"); plt.close(fig)
    print("fig_profile.png")


if __name__ == "__main__":
    main()
