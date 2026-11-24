"""Two-way (row and column) analysis of the primary write matrices.

Reads every block's summary.json under runs/, rebuilds the 6x6 clinical write matrix W[q][d]
(question q, direction d), and grades each cell three ways:
  row ownership   : steering reference met and O_q = W_qq - max_{d!=q} W_qd > 0   (the paper's grade)
  column specific : C_d = W_dd - max_{q!=d} |W_qd| > 0  (the write moves its own answer more than it
                    moves any other answer, in either direction)
  two-way owned   : both of the above.
Also: competitor-subset sensitivity of the row grade, and the reference-only grade.
Writes analysis/two_way.json.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(sys.argv[1])
OUT = Path(sys.argv[2])

m = pd.read_csv(RUNS / "manifest.csv", low_memory=False)
core = m[(m.module == "CORE") & (m.locus_id == "vis.last") & (m.fit_seed == 0)
         & (m.is_primary_template == True)].copy()

rows = []
for (mk, ds), g in core.groupby(["model_key", "dataset"]):
    s = json.load(open(RUNS / mk / ds / "summary.json"))["core"]
    W = s["W"]
    pq = s["per_question"]
    concepts = list(g.concept)
    for c in concepts:
        assert f"concept:{c}" in W[c], (mk, ds, c)
    M = np.array([[W[q][f"concept:{d}"] for d in concepts] for q in concepts])
    for j, d in enumerate(concepts):
        r = g[g.concept == d].iloc[0]
        ref = bool(r.steering_reference)
        own_row = ref and (r.O_q > 0)
        off = np.delete(M[:, j], j)
        col_margin = M[j, j] - np.max(np.abs(off))
        kappa = M[j, j] / np.sum(np.abs(M[:, j])) if np.sum(np.abs(M[:, j])) > 0 else np.nan
        offtarget_ratio = np.sum(np.abs(off)) / abs(M[j, j]) if M[j, j] != 0 else np.inf
        # competitor-subset sensitivity: row grade against every subset of k of the 5 competitors
        others = [i for i in range(len(concepts)) if i != j]
        sub = {}
        for k in (1, 2, 3, 4):
            vals = [ref and (M[j, j] - max(M[j, i] for i in S) > 0)
                    for S in itertools.combinations(others, k)]
            sub[k] = float(np.mean(vals))
        rows.append(dict(model=mk, dataset=ds, concept=d, ref=ref, owned=own_row,
                         O_q=r.O_q, W_qq=M[j, j], col_margin=col_margin,
                         col_specific=bool(col_margin > 0) and ref, kappa=kappa,
                         offtarget_ratio=offtarget_ratio,
                         two_way=own_row and bool(col_margin > 0),
                         readable=bool(r.selectivity > 0) if pd.notna(r.selectivity) else None,
                         answerable=bool(r.answer_auroc > 0.5) if pd.notna(r.answer_auroc) else None,
                         answer_auroc=r.answer_auroc, selectivity=r.selectivity,
                         **{f"own_sub{k}": v for k, v in sub.items()}))

df = pd.DataFrame(rows)
df["domain"] = np.where(df.dataset == "coco", "coco", "chest")
df.to_csv(OUT.with_suffix(".csv"), index=False)

res = {}
for dom, g in list(df.groupby("domain")) + list(df.groupby("dataset")):
    n = len(g)
    res[dom] = dict(
        n=n, ref=int(g.ref.sum()), owned=int(g.owned.sum()),
        ref_lost_competitor=int((g.ref & ~g.owned).sum()),
        col_specific=int(g.col_specific.sum()), two_way=int(g.two_way.sum()),
        owned_not_colspec=int((g.owned & ~g.two_way).sum()),
        median_kappa=float(g.kappa.median()),
        median_kappa_owned=float(g[g.owned].kappa.median()) if g.owned.any() else None,
        median_offtarget_ratio_ref=float(g[g.ref].offtarget_ratio.median()) if g.ref.any() else None,
        **{f"own_sub{k}_mean": float(g[f"own_sub{k}"].sum()) for k in (1, 2, 3, 4)},
    )
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
