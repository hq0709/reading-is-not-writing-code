"""Figures at conference scale.

The style follows the CVPR/ICLR line-chart school: a serif face matching the body text, a full boxed
axis over a pale grid, series separated by marker shape as well as by colour, and a framed legend inside
the panel. The earlier version of this file followed the minimal school of Darcet et al. (ICLR 2024) --
no titles, no grid, no legend, claims only in the caption -- which is a defensible style but leaves a
reader who skims the figures with nothing. The rules now are:

    a figure is 5.5 inches wide, the ICLR text block, or about 2.7 for a half-width one
    every panel carries a bold letter and a short title stating that panel's claim
    colour is Okabe and Ito, never red against green, and shape repeats every colour distinction
    7 to 8 point type throughout, sparse ticks, pale grid inside a full box

An earlier version of this module broke all five rules. It is kept in git history as a reminder.
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

from utilisation import NICE, DOMAIN, PAIRS

import figstyle
figstyle.use()

COL = {"floor": figstyle.MUTE, "trained": figstyle.INK, "answer": figstyle.BAD,
       "used": figstyle.BLUE, "band": "#e3e9ee", "good": figstyle.GOOD, "mute": figstyle.MUTE}
SHORT = {"lingshu7b": "Lingshu", "internvl3_8b": "InternVL3", "qwen7b": "Qwen2.5-VL",
         "llavamed7b": "LLaVA-Med", "llava15_7b": "LLaVA-1.5"}
ORDER = ["lingshu7b", "internvl3_8b", "qwen7b", "llavamed7b", "llava15_7b"]
os.makedirs("figures/paper", exist_ok=True)
CAPTIONS = {}


def load(p="runs/utilisation_final.csv"):
    rows = []
    for r in csv.DictReader(open(p)):
        for k, v in r.items():
            if k in ("arch", "model", "domain", "concept"):
                continue
            try: r[k] = float(v)
            except (TypeError, ValueError): r[k] = float("nan")
        rows.append(r)
    return rows


def save(fig, name, caption):
    fig.savefig(f"figures/paper/{name}.png")
    plt.close(fig)
    CAPTIONS[name] = caption
    print(f"{name}.png")


# ---------------------------------------------------------------- Fig 1  the main result
def fig1(rows):
    fin = [r for r in rows if np.isfinite(r["utilisation"])]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.42),
                             gridspec_kw={"width_ratios": [1.35, 0.72, 0.72], "wspace": 0.42})

    ax = axes[0]
    rng = np.random.default_rng(0)
    for k, a in enumerate(ORDER):
        u = np.array([r["utilisation"] for r in fin if r["arch"] == a])
        c = COL["good"] if np.median(u) > 0 else COL["answer"]
        ax.scatter(k + rng.uniform(-0.16, 0.16, u.size), u, s=13, color=c, alpha=0.7, linewidths=0, zorder=3)
        ax.plot([k - 0.33, k + 0.33], [np.median(u)] * 2, color=c, lw=2.4, zorder=4,
                solid_capstyle="butt")
    ax.axhspan(0, 2.2, color=figstyle.BG_GOOD, zorder=0)
    ax.axhspan(-3.6, 0, color=figstyle.BG_BAD, zorder=0)
    ax.axhline(0, color=figstyle.INK, lw=1.0, zorder=2)
    ax.axhline(1, color=figstyle.GOOD, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[a] for a in ORDER], rotation=32, ha="right")
    ax.set_ylim(-3.3, 1.9); ax.set_yticks([-3, -2, -1, 0, 1])
    ax.set_ylabel("utilisation")

    ax = axes[1]
    # Widening the range pushes the two columns TOGETHER in figure coordinates, which is how the tick
    # labels came to touch. The range is therefore tight, the labels are the symbols defined in Figure 1a,
    # and the span bars are gone: Figure 1b already carries that comparison and repeating it here only
    # crowded a panel 1.4 inches wide.
    all_rows = load()
    pr = [np.median([r["trained"] for r in all_rows if r["arch"] == a]) for a in ORDER]
    an = [np.median([r["answer"] for r in all_rows if r["arch"] == a]) for a in ORDER]
    for k in range(5):
        ax.plot([0, 1], [pr[k], an[k]], color="#c9d1d9", lw=0.7, zorder=1)
    ax.scatter(np.zeros(5), pr, s=26, color=COL["trained"], zorder=3, linewidths=0)
    ax.scatter(np.ones(5), an, s=26, color=COL["answer"], zorder=3, linewidths=0)
    ax.set_xlim(-0.28, 1.28); ax.set_xticks([0, 1])
    ax.set_xticklabels(["probe $T$", "answer $B$"], fontsize=6.2)
    ax.set_ylabel("AUROC")

    ax = axes[2]
    for k, (med, gen) in enumerate(PAIRS):
        um = np.median([r["utilisation"] for r in fin if r["arch"] == med])
        ug = np.median([r["utilisation"] for r in fin if r["arch"] == gen])
        ax.plot([k, k], [ug, um], color="#c9d1d9", lw=0.9, zorder=1)
        ax.scatter([k], [ug], s=40, marker="s", color=COL["mute"], zorder=3, linewidths=0)
        ax.scatter([k], [um], s=44, marker="o", color=COL["good"], zorder=4, linewidths=0)
        # LLaVA-Med and LLaVA-1.5 sit 0.005 apart, so the medical marker would vanish under the general
        # one. A short leader separates them without moving either value.
        if abs(um - ug) < 0.05:
            ax.annotate("", xy=(k, um), xytext=(k + 0.17, um + 0.30),
                        arrowprops=dict(arrowstyle="-", lw=0.7, color=figstyle.MUTE))
            ax.scatter([k + 0.17], [um + 0.30], s=44, marker="o", color=COL["good"], zorder=5,
                       linewidths=0)
            ax.text(k + 0.24, um + 0.30, "medical", fontsize=figstyle.ANNOT - 0.6, va="center",
                    color=COL["good"], fontweight="bold")
    ax.axhline(0, color="#000000", lw=0.7)
    ax.set_xlim(-0.45, 1.45); ax.set_xticks([0, 1])
    ax.set_xticklabels(["Qwen2.5-VL", "LLaVA"], rotation=32, ha="right")
    ax.set_ylim(-1.25, 1.6); ax.set_yticks([-1, 0, 1])
    ax.set_ylabel("utilisation")
    save(fig, "fig1_main",
         "The model's own answer uses little of what training added to its representation. "
         "**(a)**: utilisation, one point per finding, bar at the median. Zero is an answer no better "
         "than a probe on an untrained network of the same architecture; the dashed line is an answer as "
         "good as any linear readout of the model's own activations. Two models sit above it and three "
         "sit below zero. **(b)**: median AUROC over nine findings, probe against answer, one line per "
         "model. Probe scores span 0.033 and answers span 0.179. **(c)**: the two architecture-matched "
         "pairs, medical (circle) against general (square). Each pair shares an untrained floor, so the "
         "difference within a pair is training alone.")


# ---------------------------------------------------------------- Fig 2  the decomposition
def fig2(rows):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.70),
                             gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.30})
    for ax, arch in zip(axes, ["lingshu7b", "llavamed7b"]):
        s = sorted([r for r in rows if r["arch"] == arch], key=lambda r: r["floor"])
        y = np.arange(len(s))
        for i, r in enumerate(s):
            ax.plot([r["floor"], r["trained"]], [i, i], color=COL["band"], lw=4.2,
                    solid_capstyle="butt", zorder=1)
            if r["answer"] > r["floor"]:
                ax.plot([r["floor"], min(r["answer"], r["trained"])], [i, i], color=COL["used"],
                        lw=4.2, solid_capstyle="butt", zorder=2)
            ax.plot([r["floor"]], [i], "|", ms=5, color="#000000", mew=1.0, zorder=3)
            ax.plot([r["answer"]], [i], "o", ms=2.8, color=COL["answer"], zorder=4)
        ax.set_yticks(y); ax.set_yticklabels([r["concept"] for r in s], fontsize=6.2)
        ax.set_xlim(0.44, 0.92); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_xlabel("AUROC")
        ax.invert_yaxis()
        ax.text(0.02, 1.04, SHORT[arch], transform=ax.transAxes, fontsize=7.6, fontweight="bold",
                color=COL["good"] if DOMAIN[arch] == "medical" else "#000000")
    save(fig, "fig2_decomposition",
         "What training adds, and how much of it the answer reaches, for the two medical models. "
         "The tick is a probe on an untrained network of the same architecture, the bar runs from there "
         "to the probe on the trained network, the shaded part is what the model's own answer reaches, "
         "and the dot is that answer. **Left**: Lingshu-7B reaches most of the interval on every "
         "finding. **Right**: LLaVA-Med-7B falls to the left of the interval on eight of nine.")


# ---------------------------------------------------------------- Fig 3  the two controls
def fig3(rows):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.85), gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    el = list(csv.DictReader(open("runs/elicitation_summary.csv")))
    before = np.array([float(r["gap_before"]) for r in el])
    after = np.array([float(r["gap_after"]) for r in el])
    lim = [-0.02, 0.38]
    ax.plot(lim, lim, color=figstyle.MUTE, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    # on the line, in the empty upper-left triangle, so it labels the diagonal without sitting on data
    ax.text(0.085, 0.098, "no change", fontsize=figstyle.ANNOT - 0.4, color=figstyle.MUTE,
            ha="left", va="bottom", rotation=44, rotation_mode="anchor")
    ax.scatter(before, after, s=17, marker="o", facecolor=figstyle.ORANGE, edgecolor="white",
               linewidths=0.4, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("gap, original question"); ax.set_ylabel("gap, best of four")
    figstyle.panel(ax, "a", "four ways of asking, one answer")

    ax = axes[1]
    man = [m for m in csv.DictReader(open("data/manifest.csv")) if m["split"] == "test"]
    v = np.array([int(m["view_AP"]) for m in man])
    sub = [r for r in rows if r["arch"] == "llavamed7b"]
    xs = np.array([abs(np.corrcoef(np.array([int(m[r["concept"]]) for m in man]), v)[0, 1])
                   for r in sub])
    ys = np.array([r["floor"] for r in sub])
    from scipy import stats as _st
    m, b = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 10)
    ax.plot(xx, m * xx + b, color=figstyle.MUTE, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
    ax.scatter(xs, ys, s=21, marker="D", facecolor=figstyle.VIOLET, edgecolor="white",
               linewidths=0.4, zorder=3)
    _r, _p = _st.pearsonr(xs, ys)
    # not significant at n = 9; the figure says so rather than letting the line imply otherwise
    ax.text(0.965, 0.055, f"$r$ = {_r:.2f}, $p$ = {_p:.2f} (n.s.)", transform=ax.transAxes,
            ha="right", fontsize=figstyle.ANNOT - 0.2, color=figstyle.MUTE)
    ax.set_xlabel("|corr(label, view position)|"); ax.set_ylabel("AUROC, untrained")
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.set_xlim(0.0, 0.178)
    figstyle.panel(ax, "b", "an untrained network, and acquisition")
    save(fig, "fig3_controls",
         "Neither explanation that avoids our account survives. **(a)**: the probe-minus-answer gap under "
         "the original yes-or-no question against the best of four readouts, one point per model and "
         "finding; points on the diagonal moved not at all. The best readout closes a median of 7% of the "
         "gap. **(b)**: what a probe reads from an untrained network, against how strongly that finding's "
         "label correlates with view position. An untrained network of the same architecture decodes view "
         "position at 0.997, and what it appears to know about a finding follows from that.")


# ---------------------------------------------------------------- Fig 4  the probe direction
def fig4():
    def rd(p):
        out = []
        for r in csv.DictReader(open(p)):
            for k, v in r.items():
                try: r[k] = float(v)
                except (TypeError, ValueError): pass
            out.append(r)
        return out

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.85), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    C2 = rd("runs/cad_summary_v1.csv")
    pe = np.array([r["probe_effect"] for r in C2])
    re_ = np.array([r["random_effect"] for r in C2])
    # the ratio, on a log axis, so "no better than chance" is a line rather than a diagonal the reader
    # has to count points against
    ratio = np.log2(np.maximum(pe, 1e-6) / np.maximum(re_, 1e-6))
    below = int((pe <= re_).sum())
    # the two sides are the claim, so they are coloured rather than left as one grey block
    bins = np.linspace(-3.5, 3.5, 22)
    ax.hist(ratio[ratio <= 0], bins=bins, color=figstyle.ORANGE, edgecolor="white", linewidth=0.5,
            zorder=3)
    ax.hist(ratio[ratio > 0], bins=bins, color=figstyle.SKY, edgecolor="white", linewidth=0.5,
            zorder=3)
    ax.axvline(0, color=figstyle.INK, lw=0.9, zorder=4)
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(0, 12.4)
    ax.set_yticks([0, 4, 8, 12])
    top = ax.get_ylim()[1]
    # annotations are anchored at the outer edges, not near the line they must not touch
    ax.text(-3.5, top * 0.93, f"{below} cells", fontsize=figstyle.ANNOT, ha="left",
            color=figstyle.ORANGE, fontweight="bold")
    ax.text(3.5, top * 0.93, f"{len(pe) - below} cells", fontsize=figstyle.ANNOT, ha="right",
            color=figstyle.SKY, fontweight="bold")
    ax.text(-3.5, top * 0.79, "worse than random", fontsize=figstyle.ANNOT - 0.6, ha="left",
            color=figstyle.MUTE)
    ax.text(3.5, top * 0.79, "better", fontsize=figstyle.ANNOT - 0.6, ha="right",
            color=figstyle.MUTE)
    ax.set_xlabel("log$_2$(probe effect / random effect)")
    ax.set_ylabel("cells")
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    figstyle.panel(ax, "a", "steering with the probe normal")

    ax = axes[1]
    G = rd("runs/plant_gate.csv"); A = rd("runs/plant_alignment.csv")
    ok = {(r["arch"], r["concept"]) for r in G if r["PASS"]}
    Aok = [r for r in A if (r["arch"], r["concept"]) in ok and r["placement"] == "anatomical"]
    keys = [("cos_probe", "probe\nnormal"), ("cos_sham", "sham"), ("cos_random", "random\nvector"),
            ("cos_cad", "fitted\ndirection")]
    vals = [[abs(r[k]) for r in Aok if isinstance(r.get(k), float) and np.isfinite(r[k])]
            for k, _ in keys]
    rng = np.random.default_rng(1)
    # the probe normal is the claim and the fitted direction is the ceiling; the two controls between
    # them are the scale, so they are drawn in grey and the two ends in colour
    bar = [figstyle.ORANGE, figstyle.MUTE, figstyle.MUTE, figstyle.SKY]
    for i, v in enumerate(vals):
        v = np.array(v)
        ax.scatter(i + rng.uniform(-0.17, 0.17, v.size), v, s=3.2, color=figstyle.MUTE,
                   alpha=0.35, linewidths=0, zorder=2)
        ax.plot([i - 0.32, i + 0.32], [np.mean(v)] * 2, color=bar[i], lw=2.2,
                solid_capstyle="butt", zorder=4)
    ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n in keys], fontsize=figstyle.ANNOT)
    ax.set_ylim(-0.004, 0.16)
    ax.set_ylabel("|cos| with the true displacement")
    figstyle.panel(ax, "b", "where the representation actually moves")
    save(fig, "fig4_direction",
         "The probe normal does not point where the representation moves. **(a)**: the change in the "
         "model's answer produced by adding the probe normal, divided by the change produced by a random "
         "direction of the same norm at the same locus and magnitude, one cell per model, finding and "
         "locus. Left of the line the probe normal does less than a random vector, in 26 of 50 cells. "
         "**(b)**: the cosine between each candidate direction and the activation "
         "displacement that a planted lesion of known intensity actually causes, over the nine "
         "model-by-finding cells that pass the dose-response gate. Bars are means. The probe normal is "
         "less aligned with the true displacement than a random vector.")


# ---------------------------------------------------------------- Fig 5  the second modality
def fig5():
    M = []
    for r in csv.DictReader(open("runs/modality_comparison.csv")):
        for k, v in r.items():
            try: r[k] = float(v)
            except (TypeError, ValueError): pass
        M.append(r)
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    for k, mod in enumerate(("CXR", "DERM")):
        sub = [r for r in M if r["modality"] == mod]
        best = [r["max_real"] for r in sub]
        nuis = [r["max_nuisance"] for r in sub]
        x = np.array([k - 0.16] * len(sub)), np.array([k + 0.16] * len(sub))
        for b, n in zip(best, nuis):
            ax.plot([k - 0.16, k + 0.16], [b, n], color="#c9d1d9", lw=0.5, zorder=1)
        ax.scatter(x[0], best, s=24, marker="o", facecolor=figstyle.SKY, edgecolor="white",
                   linewidths=0.45, zorder=3)
        ax.scatter(x[1], nuis, s=26, marker="D", facecolor=figstyle.ORANGE, edgecolor="white",
                   linewidths=0.45, zorder=3)
    ax.axvline(0.5, color=figstyle.HAIR, lw=0.7, zorder=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["chest radiograph", "dermoscopy"]); ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("")
    ax.text(0.012, 0.975, "AUROC", transform=ax.transAxes, ha="left", va="top",
            fontsize=figstyle.ANNOT, color=figstyle.MUTE)
    ax.set_ylim(0.828, 1.014); ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    # Direct labels instead of a legend, as in Liu et al. (2022, Fig. 1): with two series whose names
    # are the point, a legend makes the reader look something up that the plot can just say. Labelling
    # the radiograph side is enough; the dermoscopy side inherits the encoding.
    ax.annotate("acquisition variable", xy=(0.16, 0.9993), xytext=(0.30, 0.9935),
                fontsize=figstyle.ANNOT, color=figstyle.ORANGE, ha="left", va="top")
    ax.annotate("best clinical finding", xy=(-0.16, 0.8414), xytext=(-0.02, 0.8480),
                fontsize=figstyle.ANNOT, color=figstyle.SKY, ha="left", va="bottom")
    save(fig, "fig5_modality",
         "Acquisition dominance is a chest-radiograph result, not a property of medical "
         "vision-language models. For each of the five models, the best AUROC over the clinical findings "
         "and the AUROC of the acquisition variable at the same locus set. On chest radiographs the "
         "acquisition variable, view position, outranks every finding in all five models. On dermoscopy "
         "the acquisition variable, anatomic site, outranks the best finding in one of five.")


if __name__ == "__main__":
    rows = load()
    fig1(rows); fig2(rows); fig3(rows); fig4(); fig5()
    with open("figures/paper/captions.md", "w") as fh:
        for k in ("fig1_main", "fig2_decomposition", "fig3_controls", "fig4_direction",
                  "fig5_modality"):
            fh.write(f"**{k}.** {CAPTIONS[k]}\n\n")
    print("captions.md")
