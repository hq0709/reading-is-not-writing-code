"""Figure 1: what is measured, what comes out, and what it looks like.

Modelled on Laban et al. (ICLR 2026 outstanding paper), whose first figure carries the whole argument:
a schematic of the measurement on the left, the quantitative result in the centre with every model named
on the plot, and a concrete instance on the right, with coloured regions carrying the semantic split
across the whole figure.

Layout rules this file obeys, learned by rendering that figure and measuring it:
  every element is placed on an explicit grid, never by default
  the figure is graphical; symbol definitions live in the caption, not in the axes
  panel titles sit above their own panel and are given room so they cannot collide
  a legend goes below the panel it belongs to, never over an axis label
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import roc_auc_score

import figstyle
figstyle.use()
USES, FAILS = figstyle.GOOD, figstyle.BAD
BG_USES, BG_FAILS = figstyle.BG_GOOD, figstyle.BG_BAD
INK, MUTE, PALE = figstyle.INK, figstyle.MUTE, figstyle.HAIR
POS_C, NEG_C = figstyle.POS, figstyle.NEG
NICE = {"lingshu7b": "Lingshu-7B", "internvl3_8b": "InternVL3-8B", "qwen7b": "Qwen2.5-VL-7B",
        "llavamed7b": "LLaVA-Med-7B", "llava15_7b": "LLaVA-1.5-7B"}
ORDER = ["lingshu7b", "internvl3_8b", "qwen7b", "llavamed7b", "llava15_7b"]
TITLE_FS = figstyle.TITLE


def load():
    rows = []
    for r in csv.DictReader(open("runs/utilisation_final.csv")):
        for k, v in r.items():
            if k in ("arch", "model", "domain", "concept"):
                continue
            try: r[k] = float(v)
            except (TypeError, ValueError): r[k] = float("nan")
        rows.append(r)
    return rows


def node(ax, cx, cy, w, h, text, fc="#ffffff", ec=INK, fs=5.6, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.025",
                                lw=0.55, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def link(ax, x0, y0, x1, y1, color=INK, lw=0.6, rad=0.0, label=None, lx=None, ly=None, fs=5.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=4.5,
                                 lw=lw, color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text(lx, ly, label, fontsize=fs, color=color, ha="center", va="center", zorder=4)


def main():
    rows = load()
    fig = plt.figure(figsize=(5.5, 1.98))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.06, 0.80], wspace=0.40,
                          left=0.012, right=0.988, top=0.865, bottom=0.185)

    # ------------------------------------------------ a  the measurement
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0, 1.055, "a   what is measured", fontsize=TITLE_FS,
            fontweight="bold", va="bottom", transform=ax.transAxes)
    # explicit geometry: three columns with a guaranteed gap between every box and every label
    XL, XM, XR = 0.115, 0.505, 0.885
    WL, WM, WR = 0.21, 0.30, 0.135
    YU, YT, YB = 0.80, 0.475, 0.135          # untrained row, trained row, answer row
    node(ax, XL, 0.64, WL, 0.15, "chest\nradiograph", fc="#f4f6f8", ec=MUTE)
    node(ax, XM, YU, WM, 0.15, "untrained\nnetwork", fc="#f7f8f9", ec=MUTE, tc=MUTE)
    node(ax, XM, YT, WM, 0.15, "trained\nmodel", fc="#ffffff", ec=INK)
    # radiograph feeds both networks; arrows start clear of the box edge and stop clear of the target
    link(ax, XL + WL / 2 + 0.012, 0.675, XM - WM / 2 - 0.012, YU - 0.02, color=MUTE, rad=0.22)
    link(ax, XL + WL / 2 + 0.012, 0.605, XM - WM / 2 - 0.012, YT + 0.02, color=INK, rad=-0.22)
    node(ax, XR, YU, WR, 0.15, "$F$", fc="#eef1f4", ec=MUTE, fs=7.4, tc=MUTE)
    node(ax, XR, YT, WR, 0.15, "$T$", fc="#ffffff", ec=INK, fs=7.4)
    node(ax, XR, YB, WR, 0.15, "$B$", fc="#ffffff", ec=FAILS, fs=7.4, tc=FAILS)
    gap_l, gap_r = XM + WM / 2 + 0.012, XR - WR / 2 - 0.012
    mid = (gap_l + gap_r) / 2
    link(ax, gap_l, YU, gap_r, YU, color=MUTE)
    ax.text(mid, YU + 0.055, "probe", fontsize=5.1, color=MUTE, ha="center", va="bottom")
    link(ax, gap_l, YT, gap_r, YT, color=INK)
    ax.text(mid, YT + 0.055, "probe", fontsize=5.1, color=INK, ha="center", va="bottom")
    link(ax, XM, YT - 0.085, XM, YB, color=FAILS, rad=0.0)
    link(ax, XM + 0.012, YB, gap_r, YB, color=FAILS)
    ax.text(XM + 0.02, (YT + YB) / 2 - 0.02, "answer", fontsize=5.1, color=FAILS, ha="left",
            va="center")

    # ------------------------------------------------ b  the result
    ax = fig.add_subplot(gs[0, 1])
    med = {a: {k: np.median([r[k] for r in rows if r["arch"] == a])
               for k in ("floor", "trained", "answer", "utilisation")} for a in ORDER}
    lo, hi = 0.525, 0.775
    # bands span the whole axis, so no band edge reads as a data boundary
    ax.add_patch(Rectangle((-0.42, 0.655), 2.64, hi - 0.655, facecolor=BG_USES, zorder=0))
    ax.add_patch(Rectangle((-0.42, lo), 2.64, 0.655 - lo, facecolor=BG_FAILS, zorder=0))
    # labels are nudged apart when two answers fall within a label height of each other
    placed = []
    for a in sorted(ORDER, key=lambda k: -med[k]["answer"]):
        m = med[a]
        col = USES if m["utilisation"] > 0 else FAILS
        ax.plot([0, 1], [m["trained"], m["answer"]], color=col, lw=1.3, zorder=2, alpha=0.9)
        ax.plot([0], [m["trained"]], "o", ms=5.2, color=INK, zorder=4, mec="none")
        ax.plot([1], [m["answer"]], "o", ms=5.2, color=col, zorder=4, mec="none")
        ly = m["answer"]
        while any(abs(ly - q) < 0.0135 for q in placed):
            ly -= 0.0135
        placed.append(ly)
        ax.annotate(NICE[a], xy=(1, m["answer"]), xytext=(26, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=figstyle.ANNOT, color=col,
                    zorder=6, annotation_clip=False)
        if abs(ly - m["answer"]) > 1e-9:
            ax.texts[-1].set_position((1, ly))
            ax.texts[-1].xy = (1, ly)
    tr = [med[a]["trained"] for a in ORDER]; an = [med[a]["answer"] for a in ORDER]
    ax.annotate("", xy=(-0.25, min(tr)), xytext=(-0.25, max(tr)),
                arrowprops=dict(arrowstyle="<->", lw=0.5, color=INK, shrinkA=0, shrinkB=0))
    ax.text(-0.325, np.mean(tr), "0.033", rotation=90, fontsize=figstyle.ANNOT, ha="center",
            va="center", fontweight="bold")
    ax.annotate("", xy=(1.10, min(an)), xytext=(1.10, max(an)),
                arrowprops=dict(arrowstyle="<->", lw=0.5, color=INK, shrinkA=0, shrinkB=0))
    ax.text(1.055, np.mean(an), "0.179", rotation=90, fontsize=figstyle.ANNOT, ha="center",
            va="center", fontweight="bold")
    ax.set_xlim(-0.50, 2.40); ax.set_ylim(lo, hi)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["probe $T$", "answer $B$"])
    ax.set_yticks([0.55, 0.65, 0.75])
    ax.set_ylabel("AUROC")
    ax.set_title("b   probing cannot separate them", fontsize=TITLE_FS,
                 fontweight="bold", loc="left", pad=5)

    # ------------------------------------------------ c  the instance
    inner = gs[0, 2].subgridspec(2, 1, hspace=0.70)
    for i, (a, col, bg) in enumerate((("lingshu7b", USES, BG_USES), ("llavamed7b", FAILS, BG_FAILS))):
        ax = fig.add_subplot(inner[i, 0])
        z = np.load(f"runs/perimage/{a}.npz")
        y, an_ = z["Effusion_y"], z["Effusion_answer"]
        ax.set_facecolor(bg)
        for lab, c in ((0, NEG_C), (1, POS_C)):
            h, e = np.histogram(an_[y == lab], bins=np.linspace(0, 1, 38), density=True)
            ax.fill_between(e[:-1], 0, h, step="post", color=c, alpha=0.8, linewidth=0)
            ax.step(e[:-1], h, where="post", color=c, lw=0.55)
        ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", ".5", "1"])
        ax.spines["left"].set_visible(False)
        ax.text(0.04, 0.94, NICE[a], transform=ax.transAxes, fontsize=figstyle.ANNOT, color=col,
                va="top")
        ax.text(0.04, 0.68, f"AUROC {roc_auc_score(y, an_):.2f}", transform=ax.transAxes,
                fontsize=figstyle.ANNOT, color=col, va="top")
        if i == 1:
            # named inside the lower panel, whose left half is empty for this model
            figstyle.mixed(ax, 0.04, 0.30, [("effusion present", POS_C, "bold"),
                                            ("   absent", "#7c8b9a", "bold")],
                           size=figstyle.ANNOT - 0.5, ha="left")
        if i == 0:
            ax.set_title("c   the answers", fontsize=TITLE_FS, fontweight="bold",
                         loc="left", pad=5)
        else:
            ax.set_xlabel("the model's own P(yes)", fontsize=figstyle.BASE + 0.6, labelpad=2.0)

    os.makedirs("figures/paper", exist_ok=True)
    fig.savefig("figures/paper/fig_teaser.png"); plt.close(fig)
    print("fig_teaser.png")


if __name__ == "__main__":
    main()
