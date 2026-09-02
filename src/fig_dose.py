"""Dose-response for the planted lesion: the manipulation works, and the probe normal does not follow it.

A claim about a direction is only as good as the ground truth it is measured against, and the only ground
truth available here is one we construct: plant an opacity of known intensity beta and record the
displacement the representation actually undergoes. Plotting against the dose rather than at one dose is
what separates a real effect from a coincidence, and it is the form this genre uses -- Laban et al. vary
the number of shards, Schaeffer et al. vary the metric's sharpness -- because a quantity that does not
move with the dose was never measuring the thing.

Left: the manipulation check. The displacement grows monotonically with beta, so the lesion is doing
something to the representation at every dose.

Right: the claim. If the probe normal were the direction the representation travels when the finding is
added, its alignment would grow with the dose. It does not: it stays at the level of an arbitrary
direction across the whole range, while a direction fitted on behaviour rises above both.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import figstyle; figstyle.use()


def rows():
    out = []
    for r in csv.DictReader(open("runs/plant_alignment.csv")):
        try:
            r["beta"] = float(r["beta"]); r["dh_norm"] = float(r["dh_norm"])
        except (TypeError, ValueError):
            continue
        out.append(r)
    return out


def main():
    A = rows()
    gate = {(r["arch"], r["concept"]) for r in csv.DictReader(open("runs/plant_gate.csv"))
            if str(r.get("PASS", "")).strip() in ("True", "TRUE", "1")}
    A = [r for r in A if (r["arch"], r["concept"]) in gate]
    betas = sorted({r["beta"] for r in A})

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.78), gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    # per-cell curves behind the median, so the reader sees the spread and not only a summary
    for key in sorted({(r["arch"], r["concept"]) for r in A}):
        v = [np.nanmedian([r["dh_norm"] for r in A if r["beta"] == b
                           and (r["arch"], r["concept"]) == key]) for b in betas]
        ax.plot(betas, v, color=figstyle.HAIR, lw=0.7, zorder=2)
    med = [np.nanmedian([r["dh_norm"] for r in A if r["beta"] == b]) for b in betas]
    ax.plot(betas, med, color=figstyle.INK, lw=1.3, marker="o", ms=3.4, mec="white", mew=0.5, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("planted lesion intensity $\\beta$")
    ax.set_ylabel("$\\|\\Delta h\\|$")
    figstyle.panel(ax, "a", "the manipulation does something")

    ax = axes[1]
    series = [("cos_probe", "probe normal", figstyle.ORANGE, True),
              ("cos_random", "a random vector", figstyle.MUTE, False),
              ("cos_cad", "direction fitted on behaviour", figstyle.SKY, False)]
    for i, (col, lab, c, emph) in enumerate(series):
        m, lo, hi = [], [], []
        for b in betas:
            v = np.array([abs(float(r[col])) for r in A
                          if r["beta"] == b and r.get(col) not in (None, "")], dtype=float)
            v = v[np.isfinite(v)]
            m.append(np.nanmean(v))
            # standard error of the mean, drawn as a band: n varies by dose and column
            se = np.nanstd(v) / max(np.sqrt(len(v)), 1)
            lo.append(m[-1] - 1.96 * se); hi.append(m[-1] + 1.96 * se)
        ax.fill_between(betas, lo, hi, color=c, alpha=0.15, lw=0, zorder=2)
        ax.plot(betas, m, color=c, lw=1.35 if emph else 1.05, marker=["o", "^", "D"][i], ms=3.4,
                mec="white", mew=0.5, label=lab, zorder=4)
    ax.set_xlabel("planted lesion intensity $\\beta$")
    ax.set_ylabel("$|\\cos|$ with $\\Delta h$")
    ax.set_ylim(bottom=0)
    ax.legend(loc="center right", fontsize=figstyle.ANNOT - 0.6, handletextpad=0.5,
              borderaxespad=0.35, labelspacing=0.28)
    figstyle.panel(ax, "b", "the probe normal does not follow it")

    os.makedirs("figures/paper", exist_ok=True)
    fig.savefig("figures/paper/fig_dose.png", bbox_inches="tight"); plt.close(fig)
    print("fig_dose.png")


if __name__ == "__main__":
    main()
