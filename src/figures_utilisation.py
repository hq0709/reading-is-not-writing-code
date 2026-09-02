"""The paper's central figure: what training adds, and how much of it the model's own answer uses."""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
src = sys.argv[1] if len(sys.argv) > 1 else "runs/utilisation_connector.csv"
rows = list(csv.DictReader(open(src)))
for r in rows:
    for k in ("floor", "trained", "answer", "adds", "util"):
        try: r[k] = float(r[k])
        except (TypeError, ValueError): r[k] = float("nan")
rows = [r for r in rows if np.isfinite(r["adds"]) and r["adds"] > 0.02 and np.isfinite(r["util"])]
order = sorted(rows, key=lambda r: (-np.median([x["util"] for x in rows if x["arch"] == r["arch"]]),
                                    r["arch"], -r["util"]))
os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.34 * len(order))))
y = np.arange(len(order))
for i, r in enumerate(order):
    f, t, b = r["floor"], r["trained"], r["answer"]
    ax.plot([f, t], [i, i], color="#c9d6e3", lw=7, solid_capstyle="butt", zorder=1)
    if b > f:
        ax.plot([f, min(b, t)], [i, i], color="#2b6cb0", lw=7, solid_capstyle="butt", zorder=2)
    ax.plot([f], [i], marker="|", ms=13, color="#444444", mew=2, zorder=4)
    ax.plot([t], [i], marker="|", ms=13, color="#444444", mew=2, zorder=4)
    ax.plot([b], [i], marker="o", ms=6.5, color="#c0392b",
            markeredgecolor="white", mew=1.0, zorder=5)
ax.set_yticks(y)
ax.set_yticklabels([f"{r['arch']}  {r['concept']}" for r in order], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("AUROC on held-out radiographs")
ax.set_title("Grey bar: what a probe reads from an UNTRAINED network, and what training adds.\n"
             "Red dot: how well the model's own answer separates the same labels.", fontsize=10.5)
ax.axvline(0.5, color="k", lw=0.7, alpha=0.4, ls=":")
h = [plt.Line2D([], [], color="#c9d6e3", lw=7, label="what training adds (floor → trained probe)"),
     plt.Line2D([], [], color="#2b6cb0", lw=7, label="the part the model's own answer reaches"),
     plt.Line2D([], [], marker="o", ls="", color="#c0392b", ms=6.5, label="model's own answer"),
     plt.Line2D([], [], marker="|", ls="", color="#444444", ms=13, mew=2,
                label="untrained floor  |  trained probe")]
ax.legend(handles=h, fontsize=8, loc="lower right", framealpha=0.95)
fig.tight_layout(); fig.savefig("figures/utilisation.png", dpi=175); plt.close(fig)
print("wrote figures/utilisation.png")

by = {}
for r in rows:
    by.setdefault(r["arch"], []).append(r["util"])
fig, ax = plt.subplots(figsize=(7.2, 4.2))
archs = sorted(by, key=lambda a: -np.median(by[a]))
ax.bxp([{"med": np.median(by[a]), "q1": np.percentile(by[a], 25), "q3": np.percentile(by[a], 75),
         "whislo": min(by[a]), "whishi": max(by[a]), "fliers": [], "label": a} for a in archs],
       showfliers=False, patch_artist=True,
       boxprops=dict(facecolor="#cfe0f0", edgecolor="#2b6cb0"),
       medianprops=dict(color="#c0392b", lw=2))
ax.axhline(0, color="k", lw=1.0)
ax.axhline(1, color="#2e7d32", lw=1.0, ls="--")
ax.text(0.52, 1.02, "answer as good as any linear readout of the model's own activations",
        transform=ax.get_yaxis_transform(), fontsize=7.5, color="#2e7d32")
ax.text(0.52, 0.02, "answer no better than a probe on an untrained network",
        transform=ax.get_yaxis_transform(), fontsize=7.5)
ax.set_ylabel("utilisation")
ax.set_ylim(-3.0, 1.6)
ax.set_title("How much of what training adds does each model's own answer use?", fontsize=11)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
fig.tight_layout(); fig.savefig("figures/utilisation_by_model.png", dpi=175); plt.close(fig)
print("wrote figures/utilisation_by_model.png")
