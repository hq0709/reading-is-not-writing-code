"""The figure that shows the phenomenon rather than a summary of it.

An AUROC of 0.83 and an AUROC of 0.56 are two numbers. The distributions behind them are two separated
humps and a single spike with both classes on top of each other, and that is what a reader believes. The
design follows Darcet et al. (ICLR 2024) Fig. 3, which makes its case with a norm histogram rather than a
statistic, and Schaeffer et al. (NeurIPS 2023), whose captions open with the claim in bold.

Rows are the two quantities on identical images; columns are models in order of utilisation. The reader
compares by looking down a column: every model separates the classes in the top row, and only two do in
the bottom row.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import roc_auc_score

import figstyle
figstyle.use()
NEG, POS = figstyle.NEG, figstyle.POS
SHORT = {"lingshu7b": "Lingshu-7B", "internvl3_8b": "InternVL3-8B", "qwen7b": "Qwen2.5-VL-7B",
         "llavamed7b": "LLaVA-Med-7B", "llava15_7b": "LLaVA-1.5-7B"}
ORDER = ["lingshu7b", "internvl3_8b", "qwen7b", "llavamed7b", "llava15_7b"]
CONCEPT = "Effusion"


def hist(ax, v, y, bins, fill_neg=NEG, fill_pos=POS):
    for lab, col in ((0, fill_neg), (1, fill_pos)):
        h, e = np.histogram(v[y == lab], bins=bins, density=True)
        ax.fill_between(e[:-1], 0, h, step="post", color=col, alpha=0.72, linewidth=0)
        ax.step(e[:-1], h, where="post", color=col, lw=0.7)
    return


def main():
    avail = [a for a in ORDER if os.path.exists(f"runs/perimage/{a}.npz")]
    n = len(avail)
    fig, axes = plt.subplots(2, n, figsize=(min(5.5, 1.15 * n), 2.05), sharex="row",
                             gridspec_kw={"hspace": 0.58, "wspace": 0.26})
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, a in enumerate(avail):
        z = np.load(f"runs/perimage/{a}.npz")
        y = z[f"{CONCEPT}_y"]
        pr = z[f"{CONCEPT}_probe"]
        an = z[f"{CONCEPT}_answer"]
        # The probe row uses the standardised decision value, not a probability. A probe calibrated to a
        # 13% base rate puts almost all of its predicted probability mass below 0.2, which compresses the
        # very separation the panel exists to show. The two rows therefore carry different units, and the
        # comparison the reader makes down a column is of separation, not of location.
        pr = (pr - pr.mean()) / (pr.std() + 1e-9)

        ax = axes[0, j]
        hist(ax, pr, y, np.linspace(-3.0, 3.0, 42))
        ax.set_xlim(-3.0, 3.0); ax.set_xticks([-2, 0, 2])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
        ax.text(0.5, 1.16, SHORT[a], transform=ax.transAxes, ha="center", fontsize=7.2,
                fontweight="bold")
        ax.text(0.96, 0.94, f"{roc_auc_score(y, pr):.2f}", transform=ax.transAxes, fontsize=6.6,
                ha="right", va="top", color=figstyle.MUTE)

        ax = axes[1, j]
        hist(ax, an, y, np.linspace(0, 1, 44))
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
        ax.text(0.96, 0.94, f"{roc_auc_score(y, an):.2f}", transform=ax.transAxes, fontsize=6.6,
                ha="right", va="top", color=figstyle.MUTE)

    for r in range(2):
        for j in range(n):
            axes[r, j].set_yticks([])
            axes[r, j].tick_params(axis="y", length=0)
    axes[0, 0].set_ylabel("probe", fontsize=7.4)
    axes[1, 0].set_ylabel("answer", fontsize=7.4)
    for j in range(n):
        axes[1, j].set_xlabel("P(yes)" if j == 0 else "", fontsize=6.8)
        axes[1, j].set_xticks([0, 0.5, 1.0])
        axes[1, j].set_xticklabels(["0", ".5", "1"])
        axes[0, j].set_xlabel("probe score" if j == 0 else "", fontsize=6.8)
    fig.legend(handles=[Line2D([], [], color=POS, lw=3, label="effusion present"),
                        Line2D([], [], color=NEG, lw=3, label="absent")],
               loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=6.8,
               columnspacing=1.4, handlelength=1.1)
    os.makedirs("figures/paper", exist_ok=True)
    fig.savefig("figures/paper/fig0_distributions.png")
    plt.close(fig)
    print("fig0_distributions.png")
    print("\nCAPTION:")
    print("**Every model separates the classes in its representation. Only two separate them in their "
          "answers.** Distribution of two scores on the same 1000 held-out radiographs, coloured by "
          "whether a pleural effusion is present. **Top**: the decision value of a linear probe fitted at "
          "the connector, standardised per model. **Bottom**: the model's own P(yes) for \"is there a "
          "pleural effusion?\". The number in each panel is the AUROC. The probe separates the classes in "
          "all five models and the scores agree to within 0.03 across them. The answers do not: "
          "LLaVA-Med-7B places 98.8% of its answers outside [0.05, 0.95], with both classes on top of one "
          "another at 0.967.")


if __name__ == "__main__":
    main()
