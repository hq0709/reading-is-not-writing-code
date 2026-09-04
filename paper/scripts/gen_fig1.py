#!/usr/bin/env python3
"""Generate the paper's reader-writer mismatch hero figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_style import BLUE, GRAY, GREEN, LIGHT_GRAY, ORANGE, save


PAPER = Path(__file__).resolve().parents[1]
data = json.loads((PAPER / "data" / "accepted_results.json").read_text(encoding="utf-8"))
qwen = next(cell for cell in data["cells"] if cell["key"] == "qwen_effusion")
specificity = data["specificity"]

fig = plt.figure(figsize=(10.2, 2.75))
grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 0.9], wspace=0.48)


def box(ax, xy, width, height, label, *, color=LIGHT_GRAY, alpha=0.55):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025",
        facecolor=color,
        edgecolor="black",
        linewidth=0.8,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center")


# (a) The probe is trained as a reader.
ax = fig.add_subplot(grid[0, 0])
ax.set_axis_off()
box(ax, (0.00, 0.54), 0.25, 0.18, "Chest\nX-ray")
box(ax, (0.37, 0.54), 0.27, 0.18, "Consumed\nvisual block", color=BLUE, alpha=0.20)
box(ax, (0.76, 0.54), 0.22, 0.18, "Effusion\nreader $w_E$", color=GREEN, alpha=0.22)
for start, end in [((0.25, 0.63), (0.37, 0.63)), ((0.64, 0.63), (0.76, 0.63))]:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.0))
lo, hi = qwen["selectivity_ci95"]
ax.text(0.5, 0.30, f"AUROC = {qwen['auroc']:.3f}", ha="center", fontweight="bold")
ax.text(0.5, 0.18, f"controlled $S$ = {qwen['selectivity']:.3f}", ha="center")
ax.text(0.5, 0.08, f"95% CI [{lo:.3f}, {hi:.3f}]", ha="center", color=GRAY)
ax.text(0.0, 1.02, "a  Train to read", transform=ax.transAxes, fontweight="bold", fontsize=10)


# (b) Reusing the normal as a writer creates semantic competition.
ax = fig.add_subplot(grid[0, 1])
effects = specificity["direction_effects"]
labels = ["Sham", "Random p95", "Effusion $w_E$", "Nodule $w_N$"]
values = [
    specificity["absolute_sham_effect"],
    specificity["random_effect_p95"],
    effects["concept"],
    effects["unrelated_Nodule"],
]
colors = [GRAY, GRAY, BLUE, ORANGE]
y = [3, 2, 1, 0]
ax.axvline(0, color="black", linewidth=0.8)
for yi, value, color in zip(y, values, colors):
    ax.plot([0, value], [yi, yi], color=color, linewidth=2.2, alpha=0.75)
    ax.scatter(value, yi, s=45, color=color, edgecolor="black", linewidth=0.5, zorder=3)
    ax.text(value + 0.009, yi, f"{value:.3f}", va="center", color=color)
ax.set_yticks(y, labels)
ax.set_xlim(-0.005, 0.30)
ax.set_ylim(-0.6, 3.6)
ax.set_xlabel(r"Change in Effusion-answer mean $P(yes)$")
ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
ax.text(0.0, 1.02, "b  Reuse to write", transform=ax.transAxes, fontweight="bold", fontsize=10)


# (c) The familywise clinical comparison breaks Effusion-specific attribution.
ax = fig.add_subplot(grid[0, 2])
margin = specificity["primary"]["margin"]
ci_lo, ci_hi = specificity["primary"]["ci95"]
ax.axvline(0, color="black", linewidth=0.9)
ax.errorbar(
    margin,
    0.58,
    xerr=[[margin - ci_lo], [ci_hi - margin]],
    fmt="o",
    color=ORANGE,
    ecolor=ORANGE,
    capsize=4,
    markersize=6,
)
ax.set_xlim(-0.082, 0.018)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel(r"$\Delta_E-\max_d\Delta_d$")
ax.text(0.5, 0.83, "Familywise clinical margin", transform=ax.transAxes, ha="center")
label_box = {"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.9}
ax.text(0.5, 0.29, "Non-generic writer: yes", transform=ax.transAxes, ha="center", color=GREEN, fontweight="bold", fontsize=8.5, bbox=label_box)
ax.text(0.5, 0.15, "Effusion-specific: no", transform=ax.transAxes, ha="center", color=ORANGE, fontweight="bold", fontsize=8.5, bbox=label_box)
ax.text(0.0, 1.02, "c  Test attribution", transform=ax.transAxes, fontweight="bold", fontsize=10)

save(fig, "fig1_evidence_ladder")
