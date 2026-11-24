"""Difficulty control on common support.

Pools every chest cell, every easy COCO cell, and every fine-grained COCO cell (round2 fgobj), keeps the cells
whose clean-answer AUROC is at least LO (where both domains have cells), and estimates the chest-minus-natural
ownership difference adjusted for clean-answer AUROC and probe selectivity with a logistic model
(average marginal effect; bootstrap SD over checkpoints). Owned = steering reference met and O_q > 0.
Usage: python -I common_support.py RUNS two_way.csv OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

RUNS, TW, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tw = pd.read_csv(TW)
cells = tw[["model", "dataset", "concept", "owned", "answer_auroc", "selectivity"]].copy()
cells["kind"] = "core"
fg = json.load(open(RUNS / "robustness" / "round2.json"))["fgobj"]
extra = []
for b in fg["blocks"]:
    for c, v in b["per_concept"].items():
        extra.append(dict(model=b["model"], dataset="coco_fine", concept=c,
                          owned=bool(v["steering_reference"]) and v["O_q"] > 0,
                          answer_auroc=v["answer_auroc"], selectivity=v["selectivity"], kind="fine"))
cells = pd.concat([cells, pd.DataFrame(extra)], ignore_index=True).dropna(subset=["answer_auroc", "selectivity"])
cells["chest"] = cells.dataset.isin(["nih", "chexpert"]).astype(int)

res = {}
for lo in (0.80, 0.85):
    s = cells[cells.answer_auroc >= lo].copy()

    def ame(d):
        X = np.c_[d.chest, d.answer_auroc, d.selectivity]
        y = d.owned.astype(int).values
        lr = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
        X1, X0 = X.copy(), X.copy()
        X1[:, 0], X0[:, 0] = 1, 0
        return float(np.mean(lr.predict_proba(X1)[:, 1] - lr.predict_proba(X0)[:, 1]))

    est = ame(s)
    rng = np.random.default_rng(0)
    models = s.model.unique()
    boots = []
    for _ in range(2000):
        pick = rng.choice(models, len(models), replace=True)
        d = pd.concat([s[s.model == mm] for mm in pick])
        if d.chest.nunique() == 2 and d.owned.nunique() == 2:
            boots.append(ame(d))
    g = s.groupby("chest")
    res[f"auroc_ge_{lo:.2f}"] = dict(
        chest_cells=int((s.chest == 1).sum()), chest_owned=int(s[s.chest == 1].owned.sum()),
        chest_nih=int((s.dataset == "nih").sum()), chest_nih_owned=int(s[s.dataset == "nih"].owned.sum()),
        natural_cells=int((s.chest == 0).sum()), natural_owned=int(s[s.chest == 0].owned.sum()),
        median_auroc_chest=float(s[s.chest == 1].answer_auroc.median()),
        median_auroc_natural=float(s[s.chest == 0].answer_auroc.median()),
        adjusted_difference=est, bootstrap_sd=float(np.std(boots, ddof=1)),
        n_boot=len(boots))
top = cells[cells.answer_auroc >= 0.90]
def rawdiff(d):
    return float(d[d.chest == 1].owned.mean() - d[d.chest == 0].owned.mean())
rng = np.random.default_rng(0); models = top.model.unique(); bs = []
for _ in range(2000):
    d = pd.concat([top[top.model == mm] for mm in rng.choice(models, len(models), replace=True)])
    if d.chest.nunique() == 2: bs.append(rawdiff(d))
res["top_bin_ge_0.90"] = dict(chest_cells=int((top.chest == 1).sum()), chest_owned=int(top[top.chest == 1].owned.sum()),
    natural_cells=int((top.chest == 0).sum()), natural_owned=int(top[top.chest == 0].owned.sum()),
    difference=rawdiff(top), bootstrap_sd=float(np.std(bs, ddof=1)))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
