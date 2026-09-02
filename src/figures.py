"""Turn probe results into the paper's survival-curve figures and its headline table.

One figure per model: AUROC against depth, clinical concepts as lines, with the permutation null and the
control-task band drawn underneath so a reader cannot mistake an uncontrolled number for a controlled one.
The nuisance concepts are drawn on the same axes, because the point of the paper is that they behave
differently from the clinical ones.

    python src/figures.py --probes runs/probe_qwen7b,runs/probe_lingshu7b --out figures/
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLINICAL = ["Effusion", "Atelectasis", "Pneumothorax", "Consolidation", "Cardiomegaly",
            "Infiltration", "Mass", "Nodule", "Edema",
            # dermoscopy
            "Nevus", "Melanoma", "BasalCellCarcinoma", "SeborrheicKeratosis",
            "SquamousCellCarcinoma", "ActinicKeratosis"]

# The acquisition nuisance is named per modality: view position for chest radiographs, anatomic site for
# dermoscopy. Hardcoding the chest-radiograph column made every dermoscopy panel draw an empty red line.
NUISANCE_BY_MODALITY = {"cxr": ("nuis_view_AP", "view position"),
                        "derm": ("nuis_site_Trunk", "anatomic site (trunk)")}
STAGE_COLOR = {"vision": "#7fb3d5", "connector": "#f5b041", "llm": "#ffffff"}


def locus_order(name):
    """Vision blocks, then the connector, then LLM layers. Within a layer, visual before answer."""
    if name.startswith("vis.block"):
        return (0, int(name[len("vis.block"):]), 0)
    if name == "vis.last":
        return (0, 10_000, 0)
    if name == "connector":
        return (1, 0, 0)
    if name.startswith("llm.L"):
        layer = int(name.split(".L")[1].split(".")[0])
        return (2, layer, 0 if name.endswith(".vis") else 1)
    return (3, 0, 0)


def read_probe(path):
    rows = list(csv.DictReader(open(os.path.join(path, "probe_results.csv"))))
    for r in rows:
        for k in list(r):
            if k in ("concept", "locus"):
                continue
            if True:
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = float("nan")
    return rows


def survival_figure(rows, title, out_png, positions="vis", nuis_key="nuis_view_AP",
                    nuis_label="view position"):
    """AUROC against depth for one model, at one position set."""
    keep = [r for r in rows if not r["locus"].startswith("llm.") or r["locus"].endswith("." + positions)]
    loci = sorted({r["locus"] for r in keep}, key=locus_order)
    x = np.arange(len(loci))
    idx = {l: i for i, l in enumerate(loci)}

    by_concept = defaultdict(lambda: np.full(len(loci), np.nan))
    perm = np.full(len(loci), np.nan)
    ctrl = np.full(len(loci), np.nan)
    view = np.full(len(loci), np.nan)
    sex = np.full(len(loci), np.nan)
    for r in keep:
        i = idx[r["locus"]]
        by_concept[r["concept"]][i] = r["real"]
        perm[i] = np.nanmax([perm[i], r["perm"]]) if not np.isnan(perm[i]) else r["perm"]
        ctrl[i] = np.nanmax([ctrl[i], r.get("control", np.nan)]) if not np.isnan(ctrl[i]) else r.get("control", np.nan)
        view[i] = r.get(nuis_key, np.nan)
        sex[i] = r.get("nuis_sex_M", np.nan)

    fig, ax = plt.subplots(figsize=(11, 4.6))

    # stage bands, so the connector is visible as a place rather than a label
    for i, l in enumerate(loci):
        stage = "vision" if l.startswith("vis") else "connector" if l == "connector" else "llm"
        if stage != "llm":
            ax.axvspan(i - 0.5, i + 0.5, color=STAGE_COLOR[stage], alpha=0.22, lw=0)
    if "connector" in idx:
        ax.axvline(idx["connector"], color="#d35400", lw=1.4, ls="--", zorder=1)
        ax.text(idx["connector"], 1.012, "connector", color="#d35400", ha="center", fontsize=9)

    ax.axhspan(0.5, np.nanmax(perm) if not np.isnan(perm).all() else 0.5,
               color="#cccccc", alpha=0.5, lw=0, label="permutation null")
    if not np.isnan(ctrl).all():
        ax.plot(x, ctrl, color="#7f8c8d", lw=1.6, ls=":", label="control task")

    cmap = plt.get_cmap("viridis")
    clin = [c for c in CLINICAL if c in by_concept]
    for j, c in enumerate(clin):
        ax.plot(x, by_concept[c], lw=1.7, color=cmap(j / max(len(clin) - 1, 1)), label=c)

    ax.plot(x, view, color="#c0392b", lw=2.4, label=f"{nuis_label} (nuisance)")
    ax.plot(x, sex, color="#8e44ad", lw=1.6, ls="-.", label="sex (nuisance)")

    ax.axhline(0.5, color="k", lw=0.7, alpha=0.5)
    ax.set_xlim(-0.5, len(loci) - 0.5)
    ax.set_ylim(0.44, 1.03)
    ax.set_ylabel("held-out AUROC")
    ax.set_title(f"{title}   (LLM positions: {positions})", fontsize=11)

    ticks = [i for i, l in enumerate(loci) if l.startswith("vis.block") or l in ("vis.last", "connector")
             or (l.startswith("llm.L") and int(l.split(".L")[1].split(".")[0]) % 4 == 0)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([loci[i].replace("llm.L", "L").replace(f".{positions}", "").replace("vis.block", "V")
                        for i in ticks], rotation=60, ha="right", fontsize=8)
    ax.legend(fontsize=7.5, ncol=4, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return out_png


def headline_table(all_rows, out_csv, nuis_key="nuis_view_AP"):
    """One row per model and concept: where the concept peaks, and what the nuisance does there."""
    out = []
    for model, rows in all_rows.items():
        for concept in sorted({r["concept"] for r in rows}):
            sub = [r for r in rows if r["concept"] == concept and r["locus"].endswith(".vis")
                   or r["concept"] == concept and r["locus"] in ("vis.last", "connector")]
            if not sub:
                continue
            best = max(sub, key=lambda r: (r["real"] if not np.isnan(r["real"]) else -1))
            conn = next((r for r in sub if r["locus"] == "connector"), None)
            last = max((r for r in sub if r["locus"].startswith("llm.")),
                       key=lambda r: locus_order(r["locus"]), default=None)
            out.append({
                "model": model, "concept": concept,
                "peak_locus": best["locus"], "peak_auroc": round(best["real"], 4),
                "peak_perm": round(best["perm"], 4),
                "peak_control": round(best.get("control", float("nan")), 4),
                "peak_selectivity": round(best.get("selectivity", float("nan")), 4),
                "connector_auroc": round(conn["real"], 4) if conn else "",
                "final_layer_auroc": round(last["real"], 4) if last else "",
                "view_at_peak": round(best.get(nuis_key, float("nan")), 4),
            })
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True, help="comma separated probe output directories")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--modality", default="cxr", choices=["cxr", "derm"],
                    help="selects which acquisition variable is drawn as the nuisance line")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    nuis_key, nuis_label = NUISANCE_BY_MODALITY[args.modality]
    all_rows = {}
    for path in [p.strip() for p in args.probes.split(",") if p.strip()]:
        name = os.path.basename(path).replace("probe_", "")
        if not os.path.exists(os.path.join(path, "probe_results.csv")):
            print(f"skip {name}: no results yet")
            continue
        rows = read_probe(path)
        all_rows[name] = rows
        for positions in ("vis", "ans"):
            png = survival_figure(rows, name, os.path.join(args.out, f"survival_{name}_{positions}.png"),
                                  positions, nuis_key, nuis_label)
            print(f"wrote {png}")

    if all_rows:
        csv_path = os.path.join(args.out, "headline.csv")
        rows = headline_table(all_rows, csv_path, nuis_key)
        print(f"\nwrote {csv_path}")
        print(f"\n{'model':14s} {'concept':14s} {'peak locus':14s} {'AUROC':>6s} {'perm':>6s} "
              f"{'ctrl':>6s} {'sel':>7s} {'conn':>6s} {'nuis':>6s}")
        for r in rows:
            print(f"{r['model']:14s} {r['concept']:14s} {r['peak_locus']:14s} {r['peak_auroc']:6.3f} "
                  f"{r['peak_perm']:6.3f} {r['peak_control']:6.3f} {r['peak_selectivity']:+7.3f} "
                  f"{str(r['connector_auroc']):>6s} {r['view_at_peak']:6.3f}")


if __name__ == "__main__":
    main()
