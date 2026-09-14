"""Figure pack for the cf-transfer-v1 campaign, drawn only from packaged statistics (summary.json, run.json,
leaderboard.csv) and the DOSE/CORE outcomes, in the house style of `cftransfer.figstyle`. Writes PDF (paper
format), PNG (300 dpi) and SVG to runs/figures/ and an index README.md.

  F1 leaderboard_heatmap      checkpoints x datasets, cell = owned concepts (readable / answer-capable annotated)
  F2 read_vs_write            per concept cell: readability selectivity S vs ownership O, chest vs COCO
  F3 effusion_across_blocks   ownership of Effusion with 95% intervals in every chest block
  F4 same_write_different_reader   Gemma 3 4B/12B/27B and MedGemma 4B/27B (shared tower): O per concept per size
  F5 reference_family         the 119 random-direction writes, the sham and the concept write for chosen blocks
  F6 dose_response            O as a function of alpha (DOSE module), chest vs COCO
  F7 size_ladders             Qwen2.5-VL 3B->72B, Qwen3-VL, InternVL3.5 on NIH, Lingshu on CheXpert: O per concept
  F8 capability_counts        readable / answer-capable / owned fractions per dataset
  F9 controls                 connector-locus O, refit SD and label-shift gap per block
  F10 prompt_dependence       wording and mapping contrasts per block
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import figstyle as H
from .figstyle import plt
from .protocol import CONCEPTS, MODELS
from .runpaths import RUN_ROOT

DATASETS = ("nih", "chexpert", "coco")
FIG = RUN_ROOT / "figures"
ORDER = [m for m in MODELS]
DS_COL, DS_MK, DS_LABEL = H.DATASET_COLOR, H.DATASET_MARKER, H.DATASET_LABEL


def load_blocks() -> dict[tuple[str, str], dict]:
    out = {}
    for mk in ORDER:
        for ds in DATASETS:
            d = RUN_ROOT / mk / ds
            if not (d / "summary.json").exists():
                continue
            s = json.loads((d / "summary.json").read_text())
            if not s.get("core") or "per_question" not in s["core"]:
                continue
            out[(mk, ds)] = {"summary": s, "run": json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}}
    return out


def save(fig, name: str):
    H.save(fig, FIG / name)


def owned_set(s: dict) -> set[str]:
    return {c for c, v in s["core"]["per_question"].items() if v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage"}


def _label(m: str, d: str) -> str:
    return f"{m} / {d}"


# ------------------------------------------------------------------------------------------------ F1
def f1_leaderboard(blocks):
    rows = [r for r in csv.DictReader((RUN_ROOT / "leaderboard.csv").open())]
    models = [m for m in ORDER if any(r["model_key"] == m for r in rows)]
    grid = np.full((len(models), 3), np.nan); ann = [["" for _ in DATASETS] for _ in models]
    for r in rows:
        i, j = models.index(r["model_key"]), DATASETS.index(r["dataset_id"])
        if r["owned"] == "":
            ann[i][j] = f"{r['readable']} / – / inel."
        else:
            grid[i, j] = float(r["owned"]); ann[i][j] = f"{r['readable']} / {r['answer_capable'] or '–'} / {r['owned']}"
    fig, ax = plt.subplots(figsize=(H.SINGLE, 0.235 * len(models) + 0.9))
    ax.grid(False)
    im = ax.imshow(grid, cmap=H.HEAT, vmin=0, vmax=6, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], fontsize=7)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=6.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(models)):
        for j in range(3):
            v = grid[i, j]
            ax.text(j, i, ann[i][j] or "–", ha="center", va="center", fontsize=6.2,
                    color="white" if (not np.isnan(v) and v >= 4) else "#2f2f2f")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, ticks=[0, 2, 4, 6])
    cb.set_label("owned concepts (of 6)", fontsize=7); cb.ax.tick_params(labelsize=6.5, length=0); cb.outline.set_linewidth(0.5)
    ax.set_xlabel("cell: readable / answer-capable / owned", fontsize=7, labelpad=4)
    save(fig, "F1_leaderboard_heatmap")


# ------------------------------------------------------------------------------------------------ F2
def f2_read_vs_write(blocks):
    fig, ax = plt.subplots(figsize=(H.DOUBLE * 0.72, 3.0))
    hs = []
    for ds in ("coco", "chexpert", "nih"):
        col, mk = DS_COL[ds], DS_MK[ds]
        xs, ys, own = [], [], []
        for (m, d), b in blocks.items():
            if d != ds:
                continue
            cal = b["summary"].get("calibration", {}); pq_ = b["summary"]["core"]["per_question"]
            for c, v in pq_.items():
                cc = cal.get(c, {})
                if not isinstance(cc, dict) or "selectivity" not in cc or v.get("O_q") is None:
                    continue
                xs.append(cc["selectivity"]); ys.append(v["O_q"]); own.append(c in owned_set(b["summary"]))
        xs, ys, own = np.array(xs), np.array(ys), np.array(own, dtype=bool)
        ax.scatter(xs[~own], ys[~own], s=20, marker=mk, facecolors="white", edgecolors=col, linewidths=0.9, zorder=3)
        ax.scatter(xs[own], ys[own], s=24, marker=mk, color=col, edgecolors="white", linewidths=0.7, zorder=4)
        hs.append(H.series_handle(col, mk, f"{DS_LABEL[ds]}: owned ({int(own.sum())})", ls="none"))
        hs.append(H.open_handle(col, mk, f"{DS_LABEL[ds]}: not owned ({int((~own).sum())})"))
    ax.set_xlim(-0.12, 0.38); ax.set_ylim(-0.92, 0.95)
    H.chance_line(ax, y=0, label=None); H.chance_line(ax, x=0)
    ax.set_xlabel("readability: probe selectivity $S$ = AUROC(real) $-$ mean AUROC(random-label controls)", fontsize=8)
    ax.set_ylabel("ownership $O = W_{qq} - \\max_{d\\neq q} W_{qd}$", fontsize=8)
    H.boxed_legend(ax, hs, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, **H.LEG)
    save(fig, "F2_read_vs_write")


# ------------------------------------------------------------------------------------------------ F3
def f3_effusion(blocks):
    rows = []
    for (m, d), b in blocks.items():
        if d == "coco":
            continue
        v = b["summary"]["core"]["per_question"].get("Effusion")
        if v and v.get("O_q") is not None:
            ci = v.get("O_q_ci95_percentile") or [v["O_q"], v["O_q"]]
            rows.append((_label(m, d), v["O_q"], ci[0], ci[1], d))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(H.SINGLE, 0.135 * len(rows) + 0.7))
    ax.grid(False, axis="y")
    for i, (lab, o, lo, hi, d) in enumerate(rows):
        col = DS_COL[d]
        ax.errorbar(o, i, xerr=[[o - lo], [hi - o]], fmt=DS_MK[d], ms=4.2, mfc=col, mec="white", mew=0.7,
                    ecolor=col, elinewidth=1.0, capsize=1.6, capthick=0.8, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("ownership of Effusion, $O$ with 95% interval", fontsize=8)
    H.ref_line(ax, x=0)
    ax.annotate("$O=0$", (0, -0.6), xytext=(3, 0), textcoords="offset points", ha="left", va="bottom", fontsize=7.5, color=H.RED)
    hs = [H.series_handle(DS_COL[d], DS_MK[d], DS_LABEL[d], ls="none") for d in ("nih", "chexpert")]
    H.boxed_legend(ax, hs, loc="upper left", **H.LEG)
    save(fig, "F3_effusion_across_blocks")


# ------------------------------------------------------------------------------------------------ F4
def _grouped_ownership(ax, blocks, mks, ds, colors):
    concepts = CONCEPTS[ds]; present = [m for m in mks if (m, ds) in blocks]; w = 0.8 / max(len(mks), 1); hs = []
    for k, mk in enumerate(mks):
        if mk not in present:
            continue
        b = blocks[(mk, ds)]; pq_ = b["summary"]["core"]["per_question"]; own = owned_set(b["summary"])
        vals = [pq_.get(c, {}).get("O_q", np.nan) for c in concepts]
        x = np.arange(len(concepts)) + (k - (len(mks) - 1) / 2) * w
        bars = ax.bar(x, vals, w * 0.92, color=colors[k], edgecolor="none", label=mk, zorder=3)
        hs.append(bars)
        for bar, c in zip(bars, concepts):
            if c in own:
                bar.set_edgecolor(H.CHARCOAL); bar.set_linewidth(1.0); bar.set_zorder(4)
    H.chance_line(ax, y=0, label=None)
    ax.set_xticks(range(len(concepts))); ax.set_xticklabels(concepts, rotation=30, ha="right", rotation_mode="anchor", fontsize=6.6)
    ax.set_xlim(-0.6, len(concepts) - 0.4); ax.grid(False, axis="x"); ax.margins(y=0.12)
    return hs


def f4_shared_tower(blocks):
    families = [("Gemma 3, one SigLIP tower", ["gemma3-4", "gemma3-12", "gemma3-27"]),
                ("MedGemma, one MedSigLIP tower", ["medgemma-4", "medgemma-27"])]
    fig, axes = plt.subplots(2, 3, figsize=(H.DOUBLE, 4.6), sharey="row")
    letters = "abcdef"
    for r, (title, mks) in enumerate(families):
        colors = [H.LILAC, H.SAGE, H.MUSTARD][:len(mks)]
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            hs = _grouped_ownership(ax, blocks, mks, ds, colors)
            H.panel_title(ax, letters[r * 3 + c], DS_LABEL[ds])
            if c == 0:
                ax.set_ylabel(f"{title}\nownership $O$ (outlined: owned)", fontsize=8)
            if hs:
                H.boxed_legend(ax, hs, loc="upper left" if ds != "coco" else "lower left", **H.LEG)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, "F4_same_write_different_reader")


# ------------------------------------------------------------------------------------------------ F5
def f5_reference_family(blocks):
    picks = [("q25-7", "nih", "Effusion"), ("q25-7", "coco", "person"), ("q25-72", "nih", "Atelectasis"), ("lingshu-32", "chexpert", "Edema")]
    fig, axes = plt.subplots(1, len(picks), figsize=(H.DOUBLE, 2.3))
    for k, (ax, (mk, ds, c)) in enumerate(zip(axes, picks)):
        b = blocks.get((mk, ds))
        if not b:
            ax.set_visible(False); continue
        W = b["summary"]["core"]["W"][c]
        rand = np.array([v for kk, v in W.items() if kk.startswith("random:")]); sham = W.get("sham", np.nan); conc = W.get(f"concept:{c}", np.nan)
        ax.hist(rand, bins=24, color=H.GREY, edgecolor="white", linewidth=0.4, zorder=3)
        ax.grid(False, axis="x")
        p95 = float(np.percentile(rand, 95))
        ax.axvline(p95, color=H.RED, ls="--", lw=1.0, zorder=4)
        ax.axvline(sham, color=H.CHARCOAL, ls=":", lw=1.2, zorder=4)
        ax.axvline(conc, color=DS_COL[ds], ls="-", lw=1.8, zorder=5)
        H.value_label(ax, conc, ax.get_ylim()[1] * 0.97, f"{conc:+.2f}", DS_COL[ds], dx=-3 if conc > p95 else 3,
                      ha="right" if conc > p95 else "left", va="top", size=6.4)
        H.panel_title(ax, "abcd"[k], f"{mk} / {ds} / {c}")
        ax.set_xlabel("$W$: mean change in $p$(yes)", fontsize=7.5)
        ax.tick_params(labelsize=6.8)
        if k == 0:
            ax.set_ylabel("random directions (count)", fontsize=7.5)
    hs = [Patch(facecolor=H.GREY, edgecolor="white", label="119 random directions"),
          Line2D([], [], color=H.RED, ls="--", lw=1.0, label="random p95"),
          Line2D([], [], color=H.CHARCOAL, ls=":", lw=1.2, label="sham (permuted direction)"),
          Line2D([], [], color=H.INK, ls="-", lw=1.8, label="concept write $W_{qq}$ (dataset colour)")]
    H.boxed_legend(fig, hs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), **H.LEG)
    fig.tight_layout(w_pad=0.9)
    save(fig, "F5_reference_family")


# ------------------------------------------------------------------------------------------------ F6
def dose_curves(blocks, cache=FIG / "dose_curves.json") -> dict:
    """Median ownership O over the six questions at each alpha of the DOSE module (plus the CORE alpha), using the
    clean baseline that the DOSE module shares with CORE (fixed batch composition; baseline rows live in CORE)."""
    if cache.exists():
        return json.loads(cache.read_text())
    cols = ["row_id", "concept", "template_id", "fit_seed", "direction_id", "direction_kind", "alpha", "p_present", "sample_status"]
    out = {}
    for (mk, ds), b in blocks.items():
        fd, fc = RUN_ROOT / mk / ds / "outcomes" / "DOSE.parquet", RUN_ROOT / mk / ds / "outcomes" / "CORE.parquet"
        if not fd.exists() or not fc.exists():
            continue
        tpl = b["summary"].get("primary_template") or b["run"].get("primary_template") or "IY"    # block's primary template
        core = pq.read_table(fc, columns=cols).to_pandas()
        core = core[(core.sample_status == "OK") & (core.template_id == tpl) & (core.fit_seed == 0)]
        base = core[core.direction_kind == "baseline"].set_index(["row_id", "concept"])["p_present"]
        dose = pq.read_table(fd, columns=cols).to_pandas()
        dose = dose[(dose.sample_status == "OK") & (dose.template_id == tpl) & (dose.fit_seed == 0)]
        df = pd.concat([dose[dose.direction_kind == "concept"], core[core.direction_kind == "concept"]], ignore_index=True)
        curves = {}
        for a, g in df.groupby("alpha"):
            g = g.copy()
            g["delta"] = g["p_present"].values - base.reindex(pd.MultiIndex.from_arrays([g.row_id.values, g.concept.values])).values
            W = g.groupby(["concept", "direction_id"])["delta"].mean().unstack()      # question x direction
            O = []
            for q in W.index:
                own = W.loc[q, f"concept:{q}"] if f"concept:{q}" in W.columns else np.nan
                others = [W.loc[q, cc] for cc in W.columns if cc != f"concept:{q}"]
                O.append(own - np.nanmax(others) if others else np.nan)
            if np.isfinite(O).any():
                curves[str(float(a))] = float(np.nanmedian(O))
        out[f"{mk}/{ds}"] = curves
    FIG.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(out, indent=1))
    return out


def f6_dose(blocks):
    curves = dose_curves(blocks)
    fig, ax = plt.subplots(figsize=(H.SINGLE, 2.7))
    present = []
    for key, cv in curves.items():
        ds = key.split("/")[1]; col = DS_COL[ds]; present.append(ds)
        a = sorted(float(x) for x in cv)
        ax.plot(a, [cv[str(x)] for x in a], ls="--", lw=0.9, color=col, alpha=0.75, marker=DS_MK[ds], ms=2.6, mec="white", mew=0.4, zorder=3)
    H.chance_line(ax, y=0, label=None); H.chance_line(ax, x=0)
    ax.set_xticks([-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]); ax.set_xticklabels(["$-$0.5", "$-$0.25", "$-$0.1", "0.1", "0.25", "0.5"], fontsize=7)
    ax.set_xlabel("relative dose $\\alpha$"); ax.set_ylabel("median ownership $O$ over six questions")
    hs = [H.series_handle(DS_COL[d], DS_MK[d], f"{DS_LABEL[d]} ({sum(1 for p in present if p == d)} blocks)") for d in DATASETS if d in present]
    H.boxed_legend(ax, hs, loc="upper left", **H.LEG)
    fig.tight_layout()
    save(fig, "F6_dose_response")


# ------------------------------------------------------------------------------------------------ F7
def f7_ladders(blocks):
    ladders = [("Qwen2.5-VL on NIH", ["q25-3", "q25-7", "q25-32", "q25-72"], "nih"), ("Qwen3-VL on NIH", ["q3-4", "q3-8", "q3-32"], "nih"),
               ("InternVL3.5 on NIH", ["iv35-8", "iv35-14", "iv35-38"], "nih"), ("Lingshu on CheXpert", ["lingshu-7", "lingshu-32"], "chexpert")]
    fig, axes = plt.subplots(1, len(ladders), figsize=(H.DOUBLE, 2.5), sharey=True)
    for k, (ax, (title, mks, ds)) in enumerate(zip(axes, ladders)):
        mks = [m for m in mks if (m, ds) in blocks]
        hs = _grouped_ownership(ax, blocks, mks, ds, [H.LILAC, H.SAGE, H.MUSTARD, H.HAZE][:len(mks)])
        H.panel_title(ax, "abcd"[k], title)
        if hs:
            H.boxed_legend(ax, hs, loc="upper right", **H.LEG)
    axes[0].set_ylabel("ownership $O$ (outlined: owned)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "F7_size_ladders")


# ------------------------------------------------------------------------------------------------ F8
def f8_counts(blocks):
    fig, ax = plt.subplots(figsize=(H.SINGLE, 2.5))
    cats = ["readable", "answer-capable", "owned"]; x = np.arange(3); w = 0.24; gap = 0.02; hs = []
    for k, ds in enumerate(DATASETS):
        tot = rd = cp = ow = 0
        for (m, d), b in blocks.items():
            if d != ds:
                continue
            cal = b["summary"].get("calibration", {}); pq_ = b["summary"]["core"]["per_question"]
            for c in pq_:
                tot += 1; cc = cal.get(c, {})
                rd += bool(isinstance(cc, dict) and cc.get("readable")); cp += bool(isinstance(cc, dict) and cc.get("answer_capable"))
            ow += len(owned_set(b["summary"]))
        vals = [100 * rd / tot, 100 * cp / tot, 100 * ow / tot] if tot else [0, 0, 0]
        xs = x + (k - 1) * (w + gap)
        hs.append(ax.bar(xs, vals, width=w, color=DS_COL[ds], edgecolor="none", label=f"{DS_LABEL[ds]} ({tot} cells)", zorder=3))
        for xi, v in zip(xs, vals):
            H.value_label(ax, xi, v, f"{v:.0f}", DS_COL[ds], dy=1.5, size=6.2)
    ax.grid(False, axis="x")
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("concept cells (%)"); ax.set_ylim(0, 128); ax.set_yticks([0, 25, 50, 75, 100])
    H.boxed_legend(ax, hs, loc="upper left", **H.LEG)
    fig.tight_layout()
    save(fig, "F8_capability_counts")


# ------------------------------------------------------------------------------------------------ F9 / F10
def _t3(b, key):
    v = b["summary"].get("t3", {}).get(key)
    return (v["estimate"], v.get("ci_low"), v.get("ci_high")) if isinstance(v, dict) and v.get("estimate") is not None else (np.nan, np.nan, np.nan)


def f9_controls(blocks):
    keys = [k for k in blocks if "t3" in blocks[k]["summary"] and blocks[k]["summary"]["t3"].get("all|connector_median_O")]
    keys.sort(key=lambda k: (DATASETS.index(k[1]), k[0]))
    fig, axes = plt.subplots(1, 3, figsize=(H.DOUBLE, 0.13 * len(keys) + 1.0), sharey=True)
    panels = [("all|connector_median_O", "connector-locus median $O$"), ("all|median_refit_O_sd", "refit SD of $O$ (3 seeds)"), ("all|median_label_gap", "label-shift gap")]
    for k, (ax, (metric, lab)) in enumerate(zip(axes, panels)):
        ax.grid(False, axis="y")
        for i, key in enumerate(keys):
            e, lo, hi = _t3(blocks[key], metric); col = DS_COL[key[1]]
            if np.isnan(e):
                continue
            xerr = None if lo is None or hi is None else [[e - lo], [hi - e]]
            ax.errorbar(e, i, xerr=xerr, fmt=DS_MK[key[1]], ms=3.2, mfc=col, mec="white", mew=0.5, ecolor=col, elinewidth=0.9, capsize=1.2, capthick=0.7, zorder=3)
        H.panel_title(ax, "abc"[k], lab)
        if k != 1:
            H.ref_line(ax, x=0)
        ax.tick_params(labelsize=6.5)
    axes[0].set_yticks(range(len(keys))); axes[0].set_yticklabels([_label(m, d) for m, d in keys], fontsize=5.4)
    axes[0].set_ylim(-0.7, len(keys) - 0.3)
    hs = [H.series_handle(DS_COL[d], DS_MK[d], DS_LABEL[d], ls="none") for d in DATASETS if any(k[1] == d for k in keys)]
    H.boxed_legend(axes[2], hs, loc="lower left", **H.LEG)
    fig.tight_layout(w_pad=0.6)
    save(fig, "F9_controls")


def _has_prompt_contrast(s: dict) -> bool:
    """A block contributes to F10 when at least one wording/mapping contrast has a number (INELIGIBLE cells carry None)."""
    return any(isinstance(v, dict) and v.get("estimate") is not None and ("|wording" in k or "|mapping" in k)
               for k, v in s.get("t3", {}).items())


def f10_prompt(blocks):
    keys = sorted([k for k in blocks if _has_prompt_contrast(blocks[k]["summary"])], key=lambda k: (DATASETS.index(k[1]), k[0]))
    fig, ax = plt.subplots(figsize=(H.SINGLE, 0.14 * len(keys) + 0.8))
    ax.grid(False, axis="y")
    for i, k in enumerate(keys):
        t3 = blocks[k]["summary"]["t3"]
        for kk, v in t3.items():
            if isinstance(v, dict) and v.get("estimate") is not None and ("|wording" in kk or "|mapping" in kk):
                wording = "wording" in kk
                ax.plot(v["estimate"], i, marker="o" if wording else "s", ms=3.6, ls="none", mfc=H.SAGE if wording else H.LILAC,
                        mec="white", mew=0.5, zorder=3)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([_label(m, d) for m, d in keys], fontsize=5.4)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    H.ref_line(ax, x=0)
    ax.set_xlabel("difference in ownership $O$ between templates", fontsize=8)
    hs = [H.series_handle(H.SAGE, "o", "wording contrast (IY $-$ WY)", ls="none"), H.series_handle(H.LILAC, "s", "mapping contrast (IA $-$ IB)", ls="none")]
    H.boxed_legend(ax, hs, loc="lower left", **H.LEG)
    save(fig, "F10_prompt_dependence")


def main():
    H.use_house_style()
    blocks = load_blocks()
    for fn in (f1_leaderboard, f2_read_vs_write, f3_effusion, f4_shared_tower, f5_reference_family, f6_dose, f7_ladders, f8_counts, f9_controls, f10_prompt):
        try:
            fn(blocks); print("ok", fn.__name__)
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAILED", fn.__name__, repr(e))
    (FIG / "README.md").write_text("# cf-transfer-v1 figure pack\n\nHouse style: `cftransfer.figstyle` (MedVIGIL journal figure standard, Morandi palette). PDF is the paper "
                                   "format; PNG (300 dpi) and SVG alongside.\n\n" + "\n".join(f"- `{p.name}`" for p in sorted(FIG.glob("*.pdf"))) +
                                   "\n\nGenerated by `python -m cftransfer.figures` from the packaged statistics.\n")
    print(FIG)


if __name__ == "__main__":
    main()
