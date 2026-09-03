#!/usr/bin/env python3
"""Generate registered dose-response curves and matched control bands."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_style import BLUE, GRAY, ORANGE, save


PAPER = Path(__file__).resolve().parents[1]
data = json.loads((PAPER / "data" / "accepted_results.json").read_text(encoding="utf-8"))
fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.75), sharex=True, sharey=True)

for index, (ax, item) in enumerate(zip(axes, data["cells"])):
    alphas = [row["alpha"] for row in item["dose"]]
    effects = [row["raw_change"] for row in item["dose"]]
    ax.plot(alphas, effects, color=BLUE, marker="o", markersize=3.5, linewidth=1.3, label="Probe normal")
    ctrl_alphas = sorted(float(value) for value in item["controls"])
    lows = [item["controls"][str(value)]["random_p05"] for value in ctrl_alphas]
    highs = [item["controls"][str(value)]["random_p95"] for value in ctrl_alphas]
    shams = [item["controls"][str(value)]["sham_effect"] for value in ctrl_alphas]
    ax.fill_between(ctrl_alphas, lows, highs, color=GRAY, alpha=0.2, label="Random 5--95%")
    ax.scatter(ctrl_alphas, shams, color=ORANGE, marker="x", s=22, label="Sham")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel(r"Relative-token dose $\alpha$")
    ax.text(0.04, 0.94, f"{item['model'].replace('-7B', '')}\n{item['concept']}", transform=ax.transAxes, va="top")
    if index == 0:
        ax.set_ylabel(r"Change in mean $P(yes)$")
        ax.legend(frameon=False, loc="lower right")
    ax.text(0.0, 1.02, chr(ord("a") + index), transform=ax.transAxes, fontweight="bold", fontsize=11)

axes[0].set_ylim(-0.30, 0.34)
save(fig, "fig2_dose_responses")
