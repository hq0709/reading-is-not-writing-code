"""Per-image probe score and per-image answer score, on identical rows, for the distribution figure.

Summary statistics hide the thing this paper is about. An AUROC of 0.556 and an AUROC of 0.827 are two
numbers; the distributions behind them are a single spike with the classes on top of each other, and two
separated humps. The figure that shows those distributions is the one a reader believes, so this script
saves what it needs: for each model and finding, the probe's decision value and the model's own P(yes) on
exactly the rows the behavioural pass used, with the labels.
"""
from __future__ import annotations
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from probe import load_acts

ARCHS = ["lingshu7b", "internvl3_8b", "qwen7b", "llavamed7b", "llava15_7b"]
CONCEPTS = ["Effusion", "Cardiomegaly", "Pneumothorax", "Edema", "Atelectasis",
            "Consolidation", "Infiltration", "Mass", "Nodule"]
LOCUS = "connector"

os.makedirs("runs/perimage", exist_ok=True)
man = {r["row_id"]: r for r in csv.DictReader(open("data/manifest.csv"))}
for arch in ARCHS:
    out_path = f"runs/perimage/{arch}.npz"
    if os.path.exists(out_path):
        print(f"skip {arch}"); continue
    meta = json.load(open(f"runs/behav_{arch}/meta.json"))
    wanted = list(meta["row_ids"])
    ids, acts = load_acts(f"runs/act_{arch}")
    if LOCUS not in acts:
        print(f"{arch}: no {LOCUS}"); continue
    pos = {r: i for i, r in enumerate(ids)}
    te = np.array([pos[r] for r in wanted if r in pos])
    tr = np.array([i for i, r in enumerate(ids) if r in man and man[r]["split"] == "train"])
    X = acts[LOCUS].astype(np.float32)
    rng = np.random.default_rng(0)
    P = rng.standard_normal((X.shape[1], 512)) / np.sqrt(512) if X.shape[1] > 512 else None
    Xp = X @ P if P is not None else X
    sc = StandardScaler().fit(Xp[tr]); Xtr, Xte = sc.transform(Xp[tr]), sc.transform(Xp[te])

    ans = np.load(f"runs/behav_{arch}/per_image.npz")
    store = {}
    for c in CONCEPTS:
        y = np.array([int(man[ids[i]][c]) for i in range(len(ids))])
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(Xtr, y[tr])
        s = clf.decision_function(Xte)
        store[f"{c}_probe"] = s
        store[f"{c}_answer"] = ans[f"{c}_p"]
        store[f"{c}_y"] = ans[f"{c}_y"]
        print(f"  {arch:13s} {c:14s} probe {roc_auc_score(y[te], s):.4f}  "
              f"answer {roc_auc_score(ans[f'{c}_y'], ans[f'{c}_p']):.4f}", flush=True)
    np.savez_compressed(out_path, **store)
    del acts, X, Xp
print("done")
