#!/usr/bin/env python3
"""Generate the paper's evidence-ladder hero figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_style import BLUE, GRAY, GREEN, LIGHT_GRAY, ORANGE, save


PAPER = Path(__file__).resolve().parents[1]
data = json.loads((PAPER / "data" / "accepted_results.json").read_text(encoding="utf-8"))

fig = plt.figure(figsize=(10.2, 3.0))
grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.1], wspace=0.5)

# (a) Same-locus measurement and intervention.
ax = fig.add_subplot(grid[0, 0])
ax.set_axis_off()
boxes = [
    (0.02, 0.58, 0.26, 0.20, "Image\n$x$"),
    (0.37, 0.58, 0.28, 0.20, "Final visual\nblock $h$"),
    (0.75, 0.58, 0.23, 0.20, "Answer\n$P(yes)$"),
    (0.37, 0.12, 0.28, 0.20, "Fixed linear\nprobe $w$"),
]
for x, y, w, h, label in boxes:
    color = BLUE if "visual" in label else LIGHT_GRAY
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.025", facecolor=color,
            edgecolor="black", linewidth=0.8, alpha=0.22 if color == BLUE else 0.6
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")
for start, end in [((0.28, 0.68), (0.37, 0.68)), ((0.65, 0.68), (0.75, 0.68)), ((0.51, 0.58), (0.51, 0.32))]:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.0))
ax.annotate(
    r"edit $h_t+\alpha\|h_t\|\hat w$",
    xy=(0.68, 0.68), xytext=(0.55, 0.90), ha="center", color=ORANGE,
    arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.0},
)
ax.text(0.0, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

# (b) Controlled decodability.
ax = fig.add_subplot(grid[0, 1])
labels = ["LLaVA\nEffusion", "LLaVA\nEdema", "Qwen\nEffusion"]
y = list(range(3))[::-1]
for yi, item in zip(y, data["cells"]):
    lo, hi = item["selectivity_ci95"]
    ax.errorbar(
        item["selectivity"], yi,
        xerr=[[item["selectivity"] - lo], [hi - item["selectivity"]]],
        fmt="o", color=BLUE, ecolor=BLUE, capsize=3, markersize=5,
    )
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y, labels)
ax.set_xlabel("Controlled selectivity\n(real AUROC $-$ control mean)")
ax.set_xlim(-0.01, 0.18)
ax.text(0.0, 1.02, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)

# (c) The evidentiary ladder.
ax = fig.add_subplot(grid[0, 2])
states = [[1, -1, 0], [1, -1, 0], [1, 1, -1]]
for row, values in enumerate(states):
    yy = 2 - row
    for col, value in enumerate(values):
        if value == 1:
            ax.scatter(col, yy, s=105, marker="o", facecolor=GREEN, edgecolor="black", linewidth=0.5)
            ax.text(col, yy, "+", color="white", ha="center", va="center", fontweight="bold")
        elif value == -1:
            ax.scatter(col, yy, s=105, marker="X", facecolor=ORANGE, edgecolor="black", linewidth=0.5)
        else:
            ax.scatter(col, yy, s=45, marker="o", facecolor="white", edgecolor=GRAY, linewidth=0.8)
ax.plot([0, 2], [2, 2], color=LIGHT_GRAY, zorder=0)
ax.plot([0, 2], [1, 1], color=LIGHT_GRAY, zorder=0)
ax.plot([0, 2], [0, 0], color=LIGHT_GRAY, zorder=0)
ax.set_xticks([0, 1, 2], ["Decodable", "Above\nrandom/sham", "Above fixed\nclinical dirs."])
ax.set_yticks([2, 1, 0], labels)
ax.set_xlim(-0.35, 2.35)
ax.set_ylim(-0.55, 2.55)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
ax.text(0.0, 1.02, "c", transform=ax.transAxes, fontweight="bold", fontsize=11)

save(fig, "fig1_evidence_ladder")
