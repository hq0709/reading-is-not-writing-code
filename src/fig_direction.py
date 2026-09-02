"""Figure 5: the probe normal is not the direction the representation uses.

Two independent lines of evidence, each in the form its claim needs.

    (a)  ground truth. A lesion of known intensity is planted and the displacement it causes is recorded,
         so there is a direction to be right about. Plotted against the dose, because a quantity that
         does not move with the dose was never measuring the thing: a direction fitted on behaviour
         climbs and plateaus, an arbitrary direction sits flat, and the probe normal sits flat BELOW the
         arbitrary one at every dose.

    (b)  intervention. Adding the probe normal to the residual stream against adding a random direction
         of the same norm at the same locus, one point per model, finding and locus. On the diagonal the
         two do the same thing. The cloud straddles it -- 26 of 50 below -- which is what a direction
         with no privileged relationship to the output looks like.

Panel (b) replaces a histogram of the log ratio. The histogram was not wrong, but it threw away the
pairing: each cell has a matched random control, and a paired comparison should be drawn as pairs.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import figstyle; figstyle.use()
import figcheck


def num(path):
    out = []
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
        out.append(r)
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.88), gridspec_kw={"wspace": 0.34})

    # ---------------------------------------------------------------- (a) against the dose
    ax = axes[0]
    A = num("runs/plant_alignment.csv")
    gate = {(r["arch"], r["concept"]) for r in num("runs/plant_gate.csv")
            if str(r.get("PASS", "")).strip() in ("True", "TRUE", "1", "1.0")}
    A = [r for r in A if (r["arch"], r["concept"]) in gate]
    betas = sorted({r["beta"] for r in A})
    series = [("cos_cad", "fitted on behaviour", figstyle.SKY, "D"),
              ("cos_random", "an arbitrary direction", figstyle.MUTE, "^"),
              ("cos_probe", "the probe normal", figstyle.ORANGE, "o")]
    for key, lab, c, mk in series:
        m, lo, hi = [], [], []
        for b in betas:
            v = np.array([abs(float(r[key])) for r in A
                          if r["beta"] == b and isinstance(r.get(key), float)], dtype=float)
            v = v[np.isfinite(v)]
            mu = float(np.nanmean(v)); se = float(np.nanstd(v)) / max(np.sqrt(len(v)), 1)
            m.append(mu); lo.append(mu - 1.96 * se); hi.append(mu + 1.96 * se)
        ax.fill_between(betas, lo, hi, color=c, alpha=0.15, lw=0, zorder=2)
        ax.plot(betas, m, color=c, lw=1.25, marker=mk, ms=3.3, mec="white", mew=0.5, zorder=4)
        ax.text(betas[-1] + 0.018, m[-1], lab, color=c, fontsize=figstyle.ANNOT - 1.0,
                va="center", ha="left")
    ax.set_xlim(betas[0] - 0.03, betas[-1] + 0.30)
    ax.set_ylim(0, 0.062)
    ax.set_xticks([0.1, 0.3, 0.6])
    ax.set_xlabel("planted lesion intensity $\\beta$")
    ax.set_ylabel("$|\\cos|$ with the true displacement")
    figstyle.panel(ax, "a", "against ground truth")

    # ---------------------------------------------------------------- (b) paired against random
    ax = axes[1]
    C = num("runs/cad_summary_v1.csv")
    pe = np.array([r["probe_effect"] for r in C])
    re_ = np.array([r["random_effect"] for r in C])
    g = np.isfinite(pe) & np.isfinite(re_) & (pe > 0) & (re_ > 0)
    pe, re_ = pe[g], re_[g]
    lim = (min(pe.min(), re_.min()) * 0.6, max(pe.max(), re_.max()) * 1.7)
    ax.plot(lim, lim, color=figstyle.MUTE, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    below = pe <= re_
    ax.scatter(re_[~below], pe[~below], s=9, color=figstyle.SKY, alpha=0.75, linewidths=0, zorder=3)
    ax.scatter(re_[below], pe[below], s=9, color=figstyle.ORANGE, alpha=0.75, linewidths=0, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("effect of a random direction")
    ax.set_ylabel("effect of the probe normal")
    ax.text(0.045, 0.955, f"{int((~below).sum())} cells: the probe normal does more",
            transform=ax.transAxes, fontsize=figstyle.ANNOT - 1.0, color=figstyle.SKY,
            va="top", ha="left")
    ax.text(0.955, 0.055, f"{int(below.sum())} cells: it does less",
            transform=ax.transAxes, fontsize=figstyle.ANNOT - 1.0, color=figstyle.ORANGE,
            va="bottom", ha="right")
    figstyle.panel(ax, "b", "against a random direction")

    figcheck.audit(fig, "fig_direction")
    os.makedirs("figures/paper", exist_ok=True)
    fig.savefig("figures/paper/fig_direction.png", bbox_inches="tight"); plt.close(fig)
    print(f"fig_direction.png   probe<=random in {int(below.sum())}/{len(pe)} cells")


if __name__ == "__main__":
    main()
