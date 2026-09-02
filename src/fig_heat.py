"""All 45 cells as an annotated heatmap with a diverging scale centred on zero.

This replaces a bullet-glyph grid of our own invention. The glyph carried more per cell -- floor, trained
probe and answer all at once -- but it was a form no reader has seen, so every cell had to be decoded
before the figure could be read, and a figure that needs a tutorial panel to be legible is the wrong
figure for a headline claim. A heatmap over a diverging scale is the form the field already reads: the
sign of utilisation is the hue, its size is the saturation, and the bimodality is two blue rows above
three orange ones with nothing between. The decomposition it drops is carried by Figure 1a and Figure 5.

Zero is pinned to white by construction, so "the answer is exactly at the untrained floor" is the colour
the eye takes as absent. The scale is clipped at +-150% because three cells run past -200 and would
otherwise flatten every other cell to white.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import figstyle; figstyle.use()

FIND = ["Effusion", "Pneumothorax", "Cardiomegaly", "Edema", "Atelectasis",
        "Consolidation", "Infiltration", "Mass", "Nodule"]
# abbreviated because the full names collide at any rotation that fits the text block; the caption
# spells them out once
SHORT = {"Effusion": "Effusion", "Pneumothorax": "Pneumothx.", "Cardiomegaly": "Cardiomeg.",
         "Edema": "Edema", "Atelectasis": "Atelectasis", "Consolidation": "Consolid.",
         "Infiltration": "Infiltration", "Mass": "Mass", "Nodule": "Nodule"}
ROWS = [("lingshu7b", "Lingshu-7B", "medical"), ("internvl3_8b", "InternVL3-8B", "general"),
        ("qwen7b", "Qwen2.5-VL-7B", "general"), ("llavamed7b", "LLaVA-Med-7B", "medical"),
        ("llava15_7b", "LLaVA-1.5-7B", "general")]
LIM = 150.0


def main():
    U = {}
    for r in csv.DictReader(open("runs/utilisation_final.csv")):
        U[(r["arch"], r["concept"])] = float(r["utilisation"]) * 100.0

    M = np.full((len(ROWS), len(FIND)), np.nan)
    for i, (a, _, _) in enumerate(ROWS):
        for j, c in enumerate(FIND):
            if (a, c) in U:
                M[i, j] = U[(a, c)]

    # orange for below the floor, blue for above it, white exactly at it: the same two hues the rest of
    # the paper uses for the same distinction, so the reader does not learn a second colour language
    cmap = LinearSegmentedColormap.from_list(
        "util", ["#8a3d05", figstyle.ORANGE, "#fdf0e0", "#ffffff", "#e2eff8", figstyle.SKY, "#0b4f77"])
    norm = Normalize(-LIM, LIM)

    fig, ax = plt.subplots(figsize=(5.5, 1.85))
    ax.imshow(np.clip(M, -LIM, LIM), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for i in range(len(ROWS)):
        for j in range(len(FIND)):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.0, color=figstyle.MUTE)
                continue
            # white type only where the cell is dark enough to carry it
            dark = abs(np.clip(v, -LIM, LIM)) > 0.62 * LIM
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6.2,
                    color="white" if dark else figstyle.INK)

    ax.set_xticks(range(len(FIND)))
    ax.set_xticklabels([SHORT[c] for c in FIND], rotation=45, ha="left", rotation_mode="anchor", fontsize=6.1)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([n for _, n, _ in ROWS], fontsize=6.8)
    ax.tick_params(axis="y", length=0, pad=2)
    for i, (_, _, dom) in enumerate(ROWS):      # the medical models, named where the row is named
        ax.text(-0.52, i + 0.30, dom, ha="right", va="center", fontsize=5.6, color=figstyle.MUTE,
                transform=ax.get_yaxis_transform(which="grid") if False else ax.transData)
    ax.set_xticks(np.arange(-0.5, len(FIND), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.grid(which="major", visible=False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # the divider between the two groups is the result, so it is drawn rather than left to the reader
    ax.axhline(1.5, color=figstyle.INK, lw=1.0)
    ax.text(len(FIND) - 0.42, 0.5, "answer reaches\nwhat training\nadded", fontsize=5.9, va="center",
            ha="left", color=figstyle.SKY, linespacing=1.25)
    ax.text(len(FIND) - 0.42, 3.0, "answer below\na probe on\nuntrained weights", fontsize=5.9,
            va="center", ha="left", color=figstyle.ORANGE, linespacing=1.25)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.021, pad=0.155,
                      ticks=[-LIM, -75, 0, 75, LIM])
    cb.ax.set_yticklabels(["$\\leq-150$", "$-75$", "0", "$+75$", "$\\geq+150$"], fontsize=5.8)
    cb.set_label("utilisation (%)", fontsize=6.2, labelpad=2)
    cb.outline.set_linewidth(0.4)

    os.makedirs("figures/paper", exist_ok=True)
    fig.savefig("figures/paper/fig_heat.png", bbox_inches="tight"); plt.close(fig)
    print("fig_heat.png")


if __name__ == "__main__":
    main()
