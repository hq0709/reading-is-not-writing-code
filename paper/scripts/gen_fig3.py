#!/usr/bin/env python3
"""Generate the independent-patient clinical direction-specificity result."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import BLUE, GRAY, GREEN, ORANGE, PURPLE, save


PAPER = Path(__file__).resolve().parents[1]
data = json.loads((PAPER / "data" / "accepted_results.json").read_text(encoding="utf-8"))["specificity"]
effects = data["direction_effects"]
random_values = [effects[f"random{i}"] for i in range(20)]
clinical_names = ["Atelectasis", "Cardiomegaly", "Mass", "Nodule", "Pneumothorax"]
clinical_values = [effects[f"unrelated_{name}"] for name in clinical_names]

fig, (ax, margin_ax) = plt.subplots(1, 2, figsize=(8.0, 3.0), gridspec_kw={"width_ratios": [3.6, 1.25]})
rng = np.random.default_rng(0)
ax.scatter(rng.normal(0, 0.055, len(random_values)), random_values, color=GRAY, s=18, alpha=0.8, label="20 random")
ax.scatter(1, effects["concept"], color=BLUE, s=58, marker="o", edgecolor="black", linewidth=0.5, label="Effusion")
ax.scatter(2, effects["sham"], color=PURPLE, s=55, marker="X", edgecolor="black", linewidth=0.5, label="Sham")
for position, (name, value) in enumerate(zip(clinical_names, clinical_values), start=3):
    color = ORANGE if name == "Nodule" else GREEN
    ax.scatter(position, value, color=color, s=48, marker="D", edgecolor="black", linewidth=0.5)
ax.hlines(data["random_effect_p95"], -0.25, 0.25, color=GRAY, linestyle="--", linewidth=1.0)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xticks(range(8), ["Random", "Effusion", "Sham", "Atel.", "Card.", "Mass", "Nodule", "Pneumo."], rotation=35, ha="right")
ax.set_ylabel(r"Change in mean $P(yes)$")
ax.set_ylim(-0.045, 0.285)
ax.text(0.0, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

primary = data["primary"]
estimate = primary["margin"]
lo, hi = primary["ci95"]
margin_ax.errorbar(estimate, 0, xerr=[[estimate - lo], [hi - estimate]], fmt="o", color=ORANGE, capsize=4)
margin_ax.axvline(0, color="black", linewidth=0.8)
margin_ax.set_yticks([0], [r"$\Delta_{Eff}-\max_d\Delta_d$"])
margin_ax.set_xlabel("Familywise margin\n(95% patient bootstrap)")
margin_ax.set_xlim(-0.08, 0.02)
margin_ax.set_ylim(-0.7, 0.7)
margin_ax.text(0.0, 1.02, "b", transform=margin_ax.transAxes, fontweight="bold", fontsize=11)

save(fig, "fig3_direction_specificity")
