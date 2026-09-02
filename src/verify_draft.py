"""Recompute every number quoted in paper/00_DRAFT.md from the result tables and report disagreements.

Written after two numbers in an earlier draft turned out to disagree with each other because one was
computed over all cells and the other over the cells a figure happened to keep. A draft is a claim about
what the CSVs say, and that claim is checkable.
"""
from __future__ import annotations
import csv, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from utilisation import NICE, DOMAIN, PAIRS


ORDER = ["lingshu7b", "internvl3_8b", "qwen7b", "llavamed7b", "llava15_7b"]


def num(path, cast=float):
    rows = []
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            try: r[k] = cast(v)
            except (TypeError, ValueError): pass
        rows.append(r)
    return rows


checks = []
def check(name, computed, quoted, tol=5e-4, fmt="{:.4f}"):
    ok = (abs(computed - quoted) <= tol) if np.isfinite(computed) and np.isfinite(quoted) else False
    checks.append((ok, name, computed, quoted))


U = num("runs/utilisation_final.csv")
u_all = np.array([r["utilisation"] for r in U if np.isfinite(r["utilisation"])])
check("median utilisation over all cells (%)", float(np.median(u_all)) * 100, 9, tol=0.5)
check("cells below the untrained floor", sum(r["below_floor"] for r in U), 22, tol=0)
check("total cells", len(U), 45, tol=0)
check("median training adds (AUROC)", float(np.median([r["adds"] for r in U])), 0.11, tol=0.002)
for a, q in (("lingshu7b", 121), ("internvl3_8b", 83), ("qwen7b", -61),
             ("llavamed7b", -68), ("llava15_7b", -69)):
    v = [r["utilisation"] for r in U if r["arch"] == a and np.isfinite(r["utilisation"])]
    check(f"median utilisation {NICE[a]} (%)", float(np.median(v)) * 100, q, tol=0.5)
    below = sum(r["below_floor"] for r in U if r["arch"] == a)
    if a in ("lingshu7b", "internvl3_8b"):
        check(f"{NICE[a]} cells below floor", below, 0, tol=0)

# spreads: computed over ALL cells, which is what the text must quote
pr = [np.median([r["trained"] for r in U if r["arch"] == a]) for a in ORDER]
an = [np.median([r["answer"] for r in U if r["arch"] == a]) for a in ORDER]
check("probe AUROC spread across models", float(np.ptp(pr)), 0.033, tol=0.0015)
check("answer AUROC spread across models", float(np.ptp(an)), 0.179, tol=0.0015)
check("ratio of the two spreads", float(np.ptp(an) / np.ptp(pr)), 5.3, tol=0.08)

um = np.median([r["utilisation"] for r in U if r["arch"] == "lingshu7b"])
ug = np.median([r["utilisation"] for r in U if r["arch"] == "qwen7b"])
check("Lingshu minus Qwen utilisation (points)", (um - ug) * 100, 183, tol=1.0)
# finite only: one cell in each LLaVA model has training_adds below the threshold and no utilisation
um2 = np.median([r["utilisation"] for r in U if r["arch"] == "llavamed7b" and np.isfinite(r["utilisation"])])
ug2 = np.median([r["utilisation"] for r in U if r["arch"] == "llava15_7b" and np.isfinite(r["utilisation"])])
check("LLaVA-Med minus LLaVA-1.5 utilisation (points)", (um2 - ug2) * 100, 0.5, tol=0.6)

E = num("runs/elicitation_summary.csv")
gains = np.array([r["gain"] for r in E])
closed = np.array([r["gap_closed_frac"] for r in E])
check("elicitation cells", len(E), 20, tol=0)
check("median fraction of the gap closed (%)", float(np.median(closed)) * 100, 7, tol=0.6)
check("median gap before elicitation", float(np.median([r["gap_before"] for r in E])), 0.177, tol=0.0015)
check("median gap after elicitation", float(np.median([r["gap_after"] for r in E])), 0.170, tol=0.0015)
check("largest elicitation gain", float(gains.max()), 0.075, tol=0.0015)
check("cells where the original question is best", sum(r["best_method"] == "yesno" for r in E), 7, tol=0)

lm = num("runs/rndprobe_llavamed7b/probe_results.csv")
check("untrained LLaVA-Med view-position AUROC", max(r["nuis_view_AP"] for r in lm), 0.9974, tol=0.0006)
tm = num("runs/probe_llavamed7b/probe_results.csv")
check("trained LLaVA-Med view-position AUROC", max(r["nuis_view_AP"] for r in tm), 0.9992, tol=0.0006)
qz = num("runs/rndprobe_qwen7b/probe_results.csv")
check("untrained Qwen-arch view-position AUROC", max(r["nuis_view_AP"] for r in qz), 0.996, tol=0.0015)
check("untrained Qwen-arch selectivity, cardiomegaly",
      max(r["selectivity"] for r in qz if r["concept"] == "Cardiomegaly"), 0.174, tol=0.006)
check("untrained Qwen-arch selectivity, oedema",
      max(r["selectivity"] for r in qz if r["concept"] == "Edema"), 0.219, tol=0.002)

# the r between the untrained floor and the label's correlation with view position
man = [m for m in csv.DictReader(open("data/manifest.csv")) if m["split"] == "test"]
v = np.array([int(m["view_AP"]) for m in man])
sub = [r for r in U if r["arch"] == "llavamed7b"]
xs = [abs(np.corrcoef(np.array([int(m[r["concept"]]) for m in man]), v)[0, 1]) for r in sub]
ys = [r["floor"] for r in sub]
check("r(untrained floor, |corr(label, view)|)", float(np.corrcoef(xs, ys)[0, 1]), 0.65, tol=0.015)
from scipy import stats as _st
check("p of that correlation", float(_st.pearsonr(xs, ys)[1]), 0.06, tol=0.005)
check("Spearman rho of that correlation", float(_st.spearmanr(xs, ys)[0]), 0.50, tol=0.005)
_k = [i for i, r in enumerate(sub) if r["concept"] != "Edema"]
check("r with oedema removed", float(np.corrcoef(np.array(xs)[_k], np.array(ys)[_k])[0, 1]), 0.42, tol=0.005)
_flo = [r["floor"] for r in sub]
check("lowest untrained floor", min(_flo), 0.540, tol=0.002)
check("highest untrained floor", max(_flo), 0.793, tol=0.002)
check("untrained floor, oedema", [r["floor"] for r in sub if r["concept"] == "Edema"][0], 0.793, tol=0.002)
check("untrained floor, mass", [r["floor"] for r in sub if r["concept"] == "Mass"][0], 0.540, tol=0.002)

b = {r["concept"]: r for r in num("runs/behav_llavamed7b/behaviour_auroc.csv")}
check("LLaVA-Med effusion mean P(yes)", b["Effusion"]["mean_p_yes"], 0.967, tol=0.0006)
check("LLaVA-Med effusion sd P(yes)", b["Effusion"]["sd_p_yes"], 0.006, tol=0.0006)
check("LLaVA-Med effusion fraction pinned (%)", b["Effusion"]["frac_pinned"] * 100, 98.8, tol=0.06)
check("LLaVA-Med findings answered yes on every image",
      sum(r["yes_rate"] >= 0.999 for r in num("runs/behav_llavamed7b/behaviour_auroc.csv")), 9, tol=0)

C2 = num("runs/cad_summary_v1.csv")
check("D2 cells", len(C2), 50, tol=0)
check("probe steers no better than random", sum(r["probe_effect"] <= r["random_effect"] for r in C2), 26, tol=0)
check("median |cos(probe, cad)|", float(np.median(np.abs([r["cos_probe_cad"] for r in C2]))), 0.0135, tol=0.0004)
check("median |cos(probe, random)|", float(np.median(np.abs([r["cos_probe_random"] for r in C2]))), 0.0091, tol=0.0004)

G = num("runs/plant_gate.csv"); A = num("runs/plant_alignment.csv")
check("D3 cells", len(G), 15, tol=0)
check("D3 cells passing the gate", sum(r["PASS"] for r in G), 9, tol=0)
ok_cells = {(r["arch"], r["concept"]) for r in G if r["PASS"]}
Aok = [r for r in A if (r["arch"], r["concept"]) in ok_cells and r["placement"] == "anatomical"]
for key, q in (("cos_probe", 0.0041), ("cos_random", 0.0126), ("cos_cad", 0.0442)):
    vals = [abs(r[key]) for r in Aok if isinstance(r.get(key), float) and np.isfinite(r[key])]
    check(f"D3 mean |{key}|", float(np.mean(vals)), q, tol=0.0004)

M = num("runs/modality_comparison.csv")
for mod, q_sel, q_wins in (("CXR", 0.050, 5), ("DERM", 0.308, 1)):
    s_ = [r for r in M if r["modality"] == mod]
    check(f"{mod} median selectivity", float(np.median([r["median_selectivity"] for r in s_])), q_sel, tol=0.0015)
    check(f"{mod} models where the nuisance wins", sum(r["nuisance_beats_best_concept"] for r in s_), q_wins, tol=0)

# The bootstrap interval quoted beside the 0.033 spread. Recomputed from the released CI table rather
# than from the figure, so the figure and the sentence cannot drift apart.
CI = num("runs/probe_ci.csv")
_conn = [r for r in CI if r["locus"] == "connector"]
_per = {}
for r in (_conn or CI):
    _per.setdefault(r["model"], []).append(r)
_w = float(np.median([np.median([x["real_hi"] - x["real_lo"] for x in v]) for v in _per.values()]))
_pt = [float(np.median([x["real"] for x in v])) for v in _per.values()]
check("half-width of one probe interval", _w / 2, 0.033, tol=0.0010)
check("probe spread in units of that half-width", (max(_pt) - min(_pt)) / (_w / 2), 1.0, tol=0.06)
check("spread across the five probe estimates", max(_pt) - min(_pt), 0.033, tol=0.0015)

# The flat probe-normal alignment quoted in the dose figure's caption.
P = num("runs/plant_alignment.csv")
_gate = {(r["arch"], r["concept"]) for r in num("runs/plant_gate.csv")
         if str(r.get("PASS", "")).strip() in ("True", "TRUE", "1", "1.0")}
_ok = [r for r in P if (r["arch"], r["concept"]) in _gate
       and isinstance(r.get("cos_probe"), float) and np.isfinite(r["cos_probe"])]
check("probe-normal alignment, pooled over doses",
      float(np.mean([abs(r["cos_probe"]) for r in _ok])), 0.004, tol=0.0009)

bad = [c for c in checks if not c[0]]
print("=" * 92)
print(f"DRAFT VERIFICATION: {len(checks) - len(bad)}/{len(checks)} quoted numbers reproduce")
print("=" * 92)
for ok, name, comp, quoted in checks:
    mark = "  ok " if ok else "MISMATCH"
    print(f"{mark}  {name:58s} computed {comp:>10.4f}   draft {quoted:>10.4f}")
if bad:
    print(f"\n{len(bad)} MISMATCH(ES) — fix the draft or the analysis before submitting")
    sys.exit(1)
print("\nall quoted numbers reproduce from runs/*.csv")
