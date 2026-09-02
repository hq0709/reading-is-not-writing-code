"""Figure 4: four ways of asking, and the gap does not move.

The account this figure has to eliminate is that the gap between what a probe reads and what the model
answers is a readout problem -- that the model knows, and we asked badly. The previous version of this
figure was a scatter of gap-before against gap-after on the identity line, which makes the reader verify
twenty points against a diagonal one at a time. The claim is about MOVEMENT, so movement is what is
drawn: one row per cell, a dot where the gap started, a dot where the best of four readouts left it, and
a line between them. Twenty near-zero lines is the claim, seen at once.

    (a)  the gap before and after, one row per model-by-finding cell, sorted. Zero is the gap closed.
    (b)  where each of the four readouts actually lands, against the probe it would have to reach. The
         probe is tight around 0.80; all four readouts sit far below it with a wide spread, and which
         one wins is close to uniform across the twenty cells -- 7, 3, 3 and 7 -- so the winner is a
         draw among noisy estimates, not a method that works.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import figstyle; figstyle.use()
import figcheck

SHORT = {"lingshu7b": "Lingshu", "internvl3_8b": "InternVL3", "qwen7b": "Qwen2.5-VL",
         "llavamed7b": "LLaVA-Med", "llava15_7b": "LLaVA-1.5"}
METHOD = [("yesno", "original"), ("yesno_alt", "rephrased"),
          ("statement", "statement"), ("negation", "negated")]


def main():
    E = list(csv.DictReader(open("runs/elicitation_summary.csv")))
    for r in E:
        for k in ("gap_before", "gap_after", "probe_connector", "yesno", "yesno_alt",
                  "statement", "negation"):
            if k in r:
                r[k] = float(r[k])
    E.sort(key=lambda r: r["gap_before"])

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.28),
                             gridspec_kw={"wspace": 0.36, "width_ratios": [1.28, 1.0]})

    # ---------------------------------------------------------------- (a) how far the gap moved
    ax = axes[0]
    y = np.arange(len(E))
    ax.axvline(0, color=figstyle.INK, lw=0.9, zorder=5)
    ax.text(0.004, len(E) - 0.4, "gap fully closed", fontsize=figstyle.ANNOT - 1.0,
            color=figstyle.INK, va="top", ha="left")
    for i, r in enumerate(E):
        ax.plot([r["gap_after"], r["gap_before"]], [i, i], color=figstyle.HAIR, lw=2.0,
                solid_capstyle="round", zorder=2)
        ax.plot([r["gap_before"]], [i], marker="o", ms=3.1, color=figstyle.ORANGE,
                mec="white", mew=0.4, zorder=4)
        ax.plot([r["gap_after"]], [i], marker="D", ms=2.9, color=figstyle.SKY,
                mec="white", mew=0.4, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{SHORT[r['arch']]} {r['concept'][:4].lower()}" for r in E],
                       fontsize=figstyle.ANNOT - 1.7)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_ylim(-0.8, len(E) - 0.2)
    ax.set_xlim(-0.045, 0.40)
    ax.set_xlabel("probe $-$ answer, in AUROC")
    ax.text(0.395, 0.6, "before: the original question", fontsize=figstyle.ANNOT - 1.0,
            color=figstyle.ORANGE, ha="right", va="center")
    ax.text(0.395, -0.35, "after: the best of four readouts", fontsize=figstyle.ANNOT - 1.0,
            color=figstyle.SKY, ha="right", va="center")
    figstyle.panel(ax, "a", "twenty cells, and nothing moves")

    # ---------------------------------------------------------------- (b) where the readouts land
    ax = axes[1]
    rng = np.random.default_rng(3)
    for j, (key, lab) in enumerate(METHOD):
        v = np.array([r[key] for r in E])
        ax.scatter(j + rng.uniform(-0.16, 0.16, v.size), v, s=6.5, color=figstyle.SKY,
                   alpha=0.60, linewidths=0, zorder=3)
        ax.plot([j - 0.30, j + 0.30], [np.median(v)] * 2, color=figstyle.SKY, lw=2.0,
                solid_capstyle="butt", zorder=4)
    p = np.array([r["probe_connector"] for r in E])
    ax.scatter(np.full(p.size, -0.85) + rng.uniform(-0.14, 0.14, p.size), p, s=6.5,
               color=figstyle.ORANGE, alpha=0.60, linewidths=0, zorder=3)
    ax.plot([-1.18, -0.52], [np.median(p)] * 2, color=figstyle.ORANGE, lw=2.0,
            solid_capstyle="butt", zorder=4)
    ax.axhline(np.median(p), color=figstyle.ORANGE, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    ax.axvline(-0.35, color=figstyle.HAIR, lw=0.7, zorder=1)
    ax.set_xticks([-0.85] + list(range(len(METHOD))))
    ax.set_xticklabels(["the probe"] + [l for _, l in METHOD], fontsize=figstyle.ANNOT - 1.2,
                       rotation=32, ha="right")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-1.35, len(METHOD) - 0.45)
    ax.set_ylim(0.425, 0.90)
    ax.set_ylabel("AUROC")
    figstyle.panel(ax, "b", "all four land far below the probe")

    figcheck.audit(fig, "fig_elicit")
    os.makedirs("figures/paper", exist_ok=True)
    fig.savefig("figures/paper/fig_elicit.png", bbox_inches="tight"); plt.close(fig)
    med = np.nanmedian([(r["gap_before"] - r["gap_after"]) / r["gap_before"] * 100
                        for r in E if r["gap_before"] > 0.01])
    print(f"fig_elicit.png   median gap closed {med:.1f}%")


if __name__ == "__main__":
    main()
