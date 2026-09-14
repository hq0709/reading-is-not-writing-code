"""Figure pack for the cf-transfer-v1 campaign, drawn only from packaged statistics (summary.json, run.json,
leaderboard.csv) and the DOSE outcomes. Writes PNG + SVG to runs/figures/ and an index README.md.

  F1 leaderboard_heatmap      checkpoints x datasets, cell = owned concepts (readable / answer-capable annotated)
  F2 read_vs_write            per concept cell: readability selectivity S vs ownership O, chest vs COCO
  F3 effusion_across_blocks   ownership of Effusion with 95% intervals in every chest block
  F4 same_write_different_reader   Gemma 3 4B/12B/27B and MedGemma 4B/27B (shared tower): O per concept per size
  F5 reference_family         the 119 random-direction writes, the sham and the concept write for chosen blocks
  F6 dose_response            O as a function of alpha (DOSE module), chest vs COCO
  F7 size_ladders             Qwen2.5-VL 3B->72B and Qwen3-VL 4B->32B on NIH: O per concept
  F8 capability_counts        readable / answer-capable / owned counts per dataset
  F9 controls                 connector-locus O and refit SD per block; label-shift gap
  F10 prompt_dependence       wording and mapping contrasts per block
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .protocol import CONCEPTS, MODELS
from .runpaths import RUN_ROOT

DATASETS = ("nih", "chexpert", "coco")
DS_LABEL = {"nih": "NIH ChestX-ray14", "chexpert": "CheXpert Plus", "coco": "COCO (control)"}
FIG = RUN_ROOT / "figures"
ORDER = [m for m in MODELS]


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
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def owned_set(s: dict) -> set[str]:
    return {c for c, v in s["core"]["per_question"].items() if v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage"}


# ------------------------------------------------------------------------------------------------ F1
def f1_leaderboard(blocks):
    rows = [r for r in csv.DictReader((RUN_ROOT / "leaderboard.csv").open())]
    models = [m for m in ORDER if any(r["model_key"] == m for r in rows)]
    grid = np.full((len(models), 3), np.nan); ann = [["" for _ in DATASETS] for _ in models]
    for r in rows:
        i, j = models.index(r["model_key"]), DATASETS.index(r["dataset_id"])
        if r["owned"] == "":
            ann[i][j] = f"{r['readable']}/-/inel."
        else:
            grid[i, j] = float(r["owned"]); ann[i][j] = f"{r['readable']}/{r['answer_capable'] or '-'}/{r['owned']}"
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(models) + 1.2))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=6, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], fontsize=9)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
    for i in range(len(models)):
        for j in range(3):
            ax.text(j, i, ann[i][j] or "-", ha="center", va="center", fontsize=7, color="black" if np.isnan(grid[i, j]) or grid[i, j] < 4 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label("owned concepts (of 6)")
    ax.set_title("readable / answer-capable / owned concepts per checkpoint and dataset", fontsize=10)
    save(fig, "F1_leaderboard_heatmap")


# ------------------------------------------------------------------------------------------------ F2
def f2_read_vs_write(blocks):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for ds, col, mk in (("nih", "#c0392b", "o"), ("chexpert", "#e67e22", "s"), ("coco", "#2980b9", "^")):
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
        xs, ys, own = np.array(xs), np.array(ys), np.array(own)
        ax.scatter(xs[~own], ys[~own], s=18, marker=mk, facecolors="none", edgecolors=col, alpha=0.8, label=f"{DS_LABEL[ds]} (not owned, n={int((~own).sum())})")
        ax.scatter(xs[own], ys[own], s=26, marker=mk, color=col, alpha=0.9, label=f"{DS_LABEL[ds]} (owned, n={int(own.sum())})")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("readability: probe selectivity S = AUROC(real) - mean AUROC(random-label controls)")
    ax.set_ylabel("ownership O = W(q,q) - max_d W(q,d)")
    ax.set_title("reading is not writing: every concept cell of the campaign", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
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
            rows.append((f"{m} / {d}", v["O_q"], ci[0], ci[1], d))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(rows) + 1.2))
    y = np.arange(len(rows))
    for i, (lab, o, lo, hi, d) in enumerate(rows):
        col = "#c0392b" if d == "nih" else "#e67e22"
        ax.plot([lo, hi], [i, i], color=col, lw=1.2); ax.plot(o, i, "o", color=col, ms=4)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("ownership of Effusion, O with 95% percentile interval")
    ax.set_title("Effusion: the paper's concept across every chest block (red NIH, orange CheXpert)", fontsize=9)
    save(fig, "F3_effusion_across_blocks")


# ------------------------------------------------------------------------------------------------ F4
def f4_shared_tower(blocks):
    families = [("Gemma 3 (one SigLIP tower)", ["gemma3-4", "gemma3-12", "gemma3-27"]), ("MedGemma (one MedSigLIP tower)", ["medgemma-4", "medgemma-27"])]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharey="row", constrained_layout=True)
    for r, (title, mks) in enumerate(families):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]; concepts = CONCEPTS[ds]; w = 0.8 / len(mks)
            for k, mk in enumerate(mks):
                b = blocks.get((mk, ds))
                if not b:
                    continue
                pq_ = b["summary"]["core"]["per_question"]
                vals = [pq_.get(cc, {}).get("O_q", np.nan) for cc in concepts]
                own = [cc in owned_set(b["summary"]) for cc in concepts]
                x = np.arange(len(concepts)) + (k - (len(mks) - 1) / 2) * w
                bars = ax.bar(x, vals, w, label=mk, alpha=0.85)
                for bb, o in zip(bars, own):
                    if o:
                        bb.set_edgecolor("black"); bb.set_linewidth(1.4)
            ax.axhline(0, color="gray", lw=0.8); ax.set_xticks(range(len(concepts))); ax.set_xticklabels(concepts, rotation=30, fontsize=7)
            ax.set_title(f"{title}\n{DS_LABEL[ds]}", fontsize=8)
            if c == 0:
                ax.set_ylabel("ownership O")
            ax.legend(fontsize=6)
    fig.suptitle("same write vector, different reader: identical vis.last features and directions across sizes (black edge = owned)", fontsize=9)
    save(fig, "F4_same_write_different_reader")


# ------------------------------------------------------------------------------------------------ F5
def f5_reference_family(blocks):
    picks = [("q25-7", "nih", "Effusion"), ("q25-7", "coco", "person"), ("q25-72", "nih", "Atelectasis"), ("lingshu-32", "chexpert", "Edema")]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.2), constrained_layout=True)
    for ax, (mk, ds, c) in zip(axes, picks):
        b = blocks.get((mk, ds))
        if not b:
            ax.set_visible(False); continue
        W = b["summary"]["core"]["W"][c]
        rand = np.array([v for k, v in W.items() if k.startswith("random:")]); sham = W.get("sham", np.nan); conc = W.get(f"concept:{c}", np.nan)
        ax.hist(rand, bins=25, color="#bdc3c7", label="119 random directions")
        ax.axvline(np.percentile(rand, 95), color="gray", ls="--", lw=1, label="random p95")
        ax.axvline(sham, color="#8e44ad", lw=1.2, label="sham (permuted)")
        ax.axvline(conc, color="#c0392b", lw=1.6, label=f"concept write W(q,q)")
        ax.set_title(f"{mk} / {ds} / {c}", fontsize=8); ax.set_xlabel("W: mean change in p(yes)", fontsize=7); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6)
    fig.suptitle("the steering reference: a concept write must beat the random family and its own sham", fontsize=9)
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
        core = pq.read_table(fc, columns=cols).to_pandas()
        core = core[(core.sample_status == "OK") & (core.template_id == "IY") & (core.fit_seed == 0)]
        base = core[core.direction_kind == "baseline"].set_index(["row_id", "concept"])["p_present"]
        dose = pq.read_table(fd, columns=cols).to_pandas()
        dose = dose[(dose.sample_status == "OK") & (dose.template_id == "IY") & (dose.fit_seed == 0)]
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
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, cv in curves.items():
        ds = key.split("/")[1]; col = {"nih": "#c0392b", "chexpert": "#e67e22", "coco": "#2980b9"}[ds]
        a = sorted(float(x) for x in cv); ax.plot(a, [cv[str(x)] for x in a], color=col, alpha=0.5, lw=1)
    present = {k.split("/")[1] for k in curves}
    for ds, col in (("nih", "#c0392b"), ("chexpert", "#e67e22"), ("coco", "#2980b9")):
        if ds in present:
            ax.plot([], [], color=col, label=DS_LABEL[ds])
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("relative dose alpha (DOSE module: -0.5, -0.25, -0.1, +0.1, +0.5; CORE: +0.25)")
    ax.set_ylabel("median ownership O over the six questions")
    ax.set_title("dose response of the concept write, one line per block (NIH and COCO carry the DOSE module)", fontsize=9); ax.legend(fontsize=8)
    save(fig, "F6_dose_response")


# ------------------------------------------------------------------------------------------------ F7
def f7_ladders(blocks):
    ladders = [("Qwen2.5-VL on NIH", ["q25-3", "q25-7", "q25-32", "q25-72"], "nih"), ("Qwen3-VL on NIH", ["q3-4", "q3-8", "q3-32"], "nih"),
               ("InternVL3.5 on NIH", ["iv35-8", "iv35-14", "iv35-38"], "nih"), ("Lingshu on CheXpert", ["lingshu-7", "lingshu-32"], "chexpert")]
    fig, axes = plt.subplots(1, len(ladders), figsize=(3.4 * len(ladders), 3.6), sharey=True, constrained_layout=True)
    for ax, (title, mks, ds) in zip(axes, ladders):
        concepts = CONCEPTS[ds]; mks = [m for m in mks if (m, ds) in blocks]; w = 0.8 / max(len(mks), 1)
        for k, mk in enumerate(mks):
            pq_ = blocks[(mk, ds)]["summary"]["core"]["per_question"]; own = owned_set(blocks[(mk, ds)]["summary"])
            vals = [pq_.get(c, {}).get("O_q", np.nan) for c in concepts]
            x = np.arange(len(concepts)) + (k - (len(mks) - 1) / 2) * w
            bars = ax.bar(x, vals, w, label=mk, alpha=0.85)
            for bb, c in zip(bars, concepts):
                if c in own:
                    bb.set_edgecolor("black"); bb.set_linewidth(1.4)
        ax.axhline(0, color="gray", lw=0.8); ax.set_xticks(range(len(concepts))); ax.set_xticklabels(concepts, rotation=30, fontsize=7)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=6)
    axes[0].set_ylabel("ownership O (black edge = owned)")
    save(fig, "F7_size_ladders")


# ------------------------------------------------------------------------------------------------ F8
def f8_counts(blocks):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    cats = ["readable", "answer-capable", "owned"]; x = np.arange(3); w = 0.26
    for k, (ds, col) in enumerate((("nih", "#c0392b"), ("chexpert", "#e67e22"), ("coco", "#2980b9"))):
        tot = rd = cp = ow = 0
        for (m, d), b in blocks.items():
            if d != ds:
                continue
            cal = b["summary"].get("calibration", {}); pq_ = b["summary"]["core"]["per_question"]
            for c in pq_:
                tot += 1; cc = cal.get(c, {})
                rd += bool(isinstance(cc, dict) and cc.get("readable")); cp += bool(isinstance(cc, dict) and cc.get("answer_capable"))
            ow += len(owned_set(b["summary"]))
        vals = [rd / tot, cp / tot, ow / tot] if tot else [0, 0, 0]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=col, label=f"{DS_LABEL[ds]} ({tot} cells)")
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{100 * v:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("fraction of concept cells"); ax.set_ylim(0, 1.05)
    ax.set_title("what the campaign finds per concept cell", fontsize=10); ax.legend(fontsize=7)
    save(fig, "F8_capability_counts")


# ------------------------------------------------------------------------------------------------ F9 / F10
def _t3(b, key):
    v = b["summary"].get("t3", {}).get(key)
    return (v["estimate"], v.get("ci_low"), v.get("ci_high")) if isinstance(v, dict) and v.get("estimate") is not None else (np.nan, np.nan, np.nan)


def f9_controls(blocks):
    keys = [k for k in blocks if "t3" in blocks[k]["summary"] and blocks[k]["summary"]["t3"].get("all|connector_median_O")]
    keys.sort(key=lambda k: (k[1], k[0]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.22 * len(keys) + 1.6), sharey=True, constrained_layout=True)
    y = np.arange(len(keys))
    for ax, (metric, lab) in zip(axes, [("all|connector_median_O", "connector-locus median O"), ("all|median_refit_O_sd", "refit SD of O (3 seeds)"), ("all|median_label_gap", "label-shift gap")]):
        for i, k in enumerate(keys):
            e, lo, hi = _t3(blocks[k], metric); col = {"nih": "#c0392b", "chexpert": "#e67e22", "coco": "#2980b9"}[k[1]]
            if lo is not None and hi is not None and not np.isnan(e):
                ax.plot([lo, hi], [i, i], color=col, lw=1)
            ax.plot(e, i, "o", color=col, ms=3)
        ax.axvline(0, color="gray", lw=0.8); ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=7)
    axes[0].set_yticks(y); axes[0].set_yticklabels([f"{m}/{d}" for m, d in keys], fontsize=6)
    fig.suptitle("T3 controls per block: the connector write is inert, refits are stable, label shifts move answers more than concept writes", fontsize=9)
    save(fig, "F9_controls")


def f10_prompt(blocks):
    keys = sorted([k for k in blocks if any(kk.endswith("wording_IY_minus_WY_O") for kk in blocks[k]["summary"].get("t3", {}))], key=lambda k: (k[1], k[0]))
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(keys) + 1.4))
    y = np.arange(len(keys))
    for i, k in enumerate(keys):
        t3 = blocks[k]["summary"]["t3"]
        for kk, v in t3.items():
            if "|wording" in kk or "|mapping" in kk:
                mk = "o" if "wording" in kk else "s"; col = "#16a085" if "wording" in kk else "#8e44ad"
                if isinstance(v, dict) and v.get("estimate") is not None:
                    ax.plot(v["estimate"], i, mk, color=col, ms=4, alpha=0.8)
    ax.plot([], [], "o", color="#16a085", label="wording contrast (IY - WY)"); ax.plot([], [], "s", color="#8e44ad", label="mapping contrast (IA - IB)")
    ax.axvline(0, color="gray", lw=0.8); ax.set_yticks(y); ax.set_yticklabels([f"{m}/{d}" for m, d in keys], fontsize=6)
    ax.set_xlabel("difference in ownership O between templates"); ax.set_title("prompt-template dependence of the write effect", fontsize=10); ax.legend(fontsize=7)
    save(fig, "F10_prompt_dependence")


def main():
    blocks = load_blocks()
    for fn in (f1_leaderboard, f2_read_vs_write, f3_effusion, f4_shared_tower, f5_reference_family, f6_dose, f7_ladders, f8_counts, f9_controls, f10_prompt):
        try:
            fn(blocks); print("ok", fn.__name__)
        except Exception as e:
            print("FAILED", fn.__name__, repr(e))
    (FIG / "README.md").write_text("# cf-transfer-v1 figure pack\n\n" + "\n".join(f"- `{p.name}`" for p in sorted(FIG.glob("*.png"))) + "\n\nGenerated by `python -m cftransfer.figures` from the packaged statistics.\n")
    print(FIG)


if __name__ == "__main__":
    main()
