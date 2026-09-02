"""The main result as a dense annotated grid, in the form top-conference papers actually use.

Modelled on Figure 6 of Laban et al. (ICLR 2026 outstanding paper), which puts 45 box plots and about 135
numbers into one figure: rows are conditions and carry a colour and a vertical label, columns are models
with one shared set of rotated headers, and every mark is annotated with its value. A narrow left panel
teaches the encoding on synthetic examples before the reader meets the data.

Our figures until now showed medians of 45 cells. The 45 cells are the result, and each carries four
numbers that a reader should be able to look up.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import figstyle
figstyle.use()

GOOD, BAD, INK, MUTE = figstyle.GOOD, figstyle.BAD, figstyle.INK, figstyle.MUTE
BG_GOOD, BG_BAD, HAIR = figstyle.BG_GOOD, figstyle.BG_BAD, figstyle.HAIR
BAND = "#dbe2e8"
SHORT = {"lingshu7b": "Lingshu", "internvl3_8b": "InternVL3", "qwen7b": "Qwen2.5-VL",
         "llavamed7b": "LLaVA-Med", "llava15_7b": "LLaVA-1.5"}
ORDER = ["lingshu7b", "internvl3_8b", "qwen7b", "llavamed7b", "llava15_7b"]
FIND = ["Effusion", "Pneumothorax", "Cardiomegaly", "Edema", "Atelectasis",
        "Consolidation", "Infiltration", "Mass", "Nodule"]
LO, HI = 0.46, 0.90


def load():
    d = {}
    for r in csv.DictReader(open("runs/utilisation_final.csv")):
        for k, v in r.items():
            if k not in ("arch", "model", "domain", "concept"):
                try: r[k] = float(v)
                except (TypeError, ValueError): r[k] = float("nan")
        d[(r["arch"], r["concept"])] = r
    return d


def cell(ax, x0, y, w, f, t, b, col, show_scale=False):
    """One bullet glyph on the shared scale: tick at F, band F..T, fill F..B, dot at B."""
    def X(v):
        return x0 + w * (np.clip(v, LO, HI) - LO) / (HI - LO)
    ax.plot([X(f), X(t)], [y, y], color=BAND, lw=3.4, solid_capstyle="butt", zorder=2)
    if b > f:
        ax.plot([X(f), X(min(b, t))], [y, y], color=col, lw=3.4, solid_capstyle="butt", zorder=3)
    ax.plot([X(f)], [y], "|", ms=4.6, color=INK, mew=1.0, zorder=4)
    ax.plot([X(b)], [y], "o", ms=2.9, color=col, mec="white", mew=0.5, zorder=5)


def main():
    D = load()
    fig = plt.figure(figsize=(5.5, 2.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.34, 1.0], wspace=0.13,
                          left=0.005, right=0.995, top=0.79, bottom=0.06)

    # ---------- (a) how to read one cell ----------
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.04, "a  how to read a cell", fontsize=figstyle.TITLE, fontweight="bold",
            va="bottom", transform=ax.transAxes)
    # the three demonstrations are stacked in the upper part of the column and every label is placed
    # to the LEFT of the glyph, so nothing can reach across into the grid beside it
    demo = [(0.80, GOOD, "+100%", "the answer reaches\nwhat training added"),
            (0.55, BAD, "0%", "the answer sits at\nthe untrained floor"),
            (0.30, BAD, "-100%", "the answer is below\nthe floor")]
    W = 0.72
    for y, col, pct, lab in demo:
        f, t = 0.55, 0.80
        # the "below the floor" case needs a value inside the drawn range, not off the left edge
        b = 0.80 if pct == "+100%" else (0.55 if pct == "0%" else 0.48)
        cell(ax, 0.0, y, W, f, t, b, col)
        # percentage and description are two separate texts at two separate x positions, never one
        # string drawn twice
        ax.text(0.0, y - 0.075, pct, fontsize=6.2, va="top", color=col,
                fontweight="bold")
        ax.text(0.27, y - 0.075, lab, fontsize=6.0, va="top", color=MUTE,
                linespacing=1.3)
    xF = W * (0.55 - LO) / (HI - LO)
    xT = W * (0.80 - LO) / (HI - LO)
    ax.text(xF, 0.90, "$F$", fontsize=figstyle.ANNOT - 0.4, ha="center", va="bottom")
    ax.text(xT, 0.90, "$T$", fontsize=figstyle.ANNOT - 0.4, ha="center", va="bottom")
    ax.text(0.0, 0.12, "every cell uses the same\nscale, AUROC 0.46 to 0.90",
            fontsize=6.0, va="top", color=MUTE, linespacing=1.3)

    # ---------- (b) the grid ----------
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_xlim(0, len(FIND)); ax.set_ylim(-0.85, len(ORDER))
    ax.text(0.0, len(ORDER) + 1.02, "b  every model and every finding",
            fontsize=figstyle.TITLE, fontweight="bold", va="bottom")
    for j, c in enumerate(FIND):
        ax.text(j + 0.5, len(ORDER) - 0.28, c, rotation=45, ha="left", va="bottom",
                fontsize=6.2, color=INK)
    for i, a in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        us = [D[(a, c)]["utilisation"] for c in FIND if (a, c) in D]
        good = np.nanmedian(us) > 0
        ax.add_patch(Rectangle((0, y - 0.44), len(FIND), 0.88,
                               facecolor=BG_GOOD if good else BG_BAD, zorder=0, lw=0))
        ax.text(-0.12, y, SHORT[a], ha="right", va="center", fontsize=figstyle.ANNOT,
                fontweight="bold", color=GOOD if good else BAD)
        ax.text(-0.12, y - 0.30, f"median {np.nanmedian(us)*100:+.0f}%", ha="right", va="center",
                fontsize=6.0, color=MUTE)
        for j, c in enumerate(FIND):
            r = D.get((a, c))
            if r is None:
                continue
            u = r["utilisation"]
            col = GOOD if (np.isfinite(u) and u > 0) else BAD
            cell(ax, j + 0.10, y + 0.10, 0.80, r["floor"], r["trained"], r["answer"], col)
            txt = f"{u*100:+.0f}" if np.isfinite(u) else "n/a"
            ax.text(j + 0.50, y - 0.26, txt, ha="center", va="center",
                    fontsize=6.0, color=col, fontweight="bold")
        if i:
            ax.plot([0, len(FIND)], [y + 0.5] * 2, color="white", lw=1.1, zorder=1)
    ax.text(0.0, -0.62, "tick: untrained floor $F$      bar: to the probe on the trained model $T$"
                        "      dot: the model's own answer $B$      number: utilisation",
            ha="left", va="center", fontsize=6.0, color=MUTE)
    os.makedirs("figures/paper", exist_ok=True)
    fig.savefig("figures/paper/fig_grid.png"); plt.close(fig)
    print("fig_grid.png")


if __name__ == "__main__":
    main()
