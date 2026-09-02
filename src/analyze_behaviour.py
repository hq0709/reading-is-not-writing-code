"""D0: what the representation carries against what the answer carries, on THE SAME ROWS.

Three corrections over the first version of this analysis, each of which changed a number that had already
been written down:

1. SAME ROWS. The behavioural score is computed on 1000 sampled test radiographs; the probe tables are
   computed on all 5343. Joining the two compared a statistic on one sample against a statistic on another.
   The probe is now refitted on train and scored on exactly the rows the behavioural pass used, whose ids
   are recorded in runs/behav_<arch>/meta.json.

2. PROMPT-INDEPENDENT LOCI ONLY, for the selected variant. Every activation was cached under the pleural
   effusion question. Measured on Lingshu-7B, changing the question changes the answer-position activations
   by 6% to 34% in relative norm and the visual-position, connector and vision-tower activations by exactly
   0.000000, because visual tokens precede the question text and attention is causal. Answer-position loci
   are therefore valid only for effusion and are excluded from the other eight findings.

3. SELECTION REPORTED HONESTLY. A probe may be chosen from 64 to 72 loci; the model's answer has no such
   choice. The headline is a single locus fixed in advance. The maximum is reported beside it as an upper
   bound, never as the comparison.
"""
from __future__ import annotations
import csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from probe import load_acts

DOM = {"lingshu7b": "medical", "llavamed7b": "medical", "qwen7b": "general",
       "llava15_7b": "general", "internvl3_8b": "general"}
EXTRACT_PROMPT_CONCEPT = "Effusion"      # every activation was cached under this question
FIXED_LOCUS = "connector"                # pre-registered, and upstream of the language model


def prompt_independent(locus: str) -> bool:
    """True where the cached activation does not depend on which finding the prompt names."""
    return not locus.endswith(".ans")


def main():
    proj_dim, seed = 512, 0
    rows = []
    for d in sorted(glob.glob("runs/behav_*")):
        if not os.path.isdir(d):
            continue                      # runs/behav_<arch>.log matches this glob too
        arch = os.path.basename(d)[6:]
        meta = json.load(open(os.path.join(d, "meta.json")))
        wanted = list(meta["row_ids"])
        ids, acts = load_acts(f"runs/act_{arch}")
        man = {r["row_id"]: r for r in csv.DictReader(open("data/manifest.csv"))}
        pos = {r: i for i, r in enumerate(ids)}
        te = np.array([pos[r] for r in wanted if r in pos])
        tr = np.array([i for i, r in enumerate(ids)
                       if r in man and man[r]["split"] == "train"])
        print(f"{arch}: {len(tr)} train rows, scoring the {len(te)} rows the behavioural pass used")
        rng = np.random.default_rng(seed)
        proj = {k: (rng.standard_normal((v.shape[1], proj_dim)) / np.sqrt(proj_dim)
                    if v.shape[1] > proj_dim else np.eye(v.shape[1]))
                for k, v in acts.items()}

        beh = {r["concept"]: r for r in csv.DictReader(open(os.path.join(d, "behaviour_auroc.csv")))}
        for concept, br in beh.items():
            y = np.array([int(man[ids[i]][concept]) for i in range(len(ids))])
            usable = [k for k in acts if prompt_independent(k) or concept == EXTRACT_PROMPT_CONCEPT]
            per_locus = {}
            for k in usable:
                X = acts[k].astype(np.float32) @ proj[k]
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(sc.transform(X[tr]), y[tr])
                s = clf.decision_function(sc.transform(X[te]))
                if len(np.unique(y[te])) > 1:
                    per_locus[k] = float(roc_auc_score(y[te], s))
            if not per_locus:
                continue
            best = max(per_locus, key=per_locus.get)
            rows.append({
                "arch": arch, "domain": DOM.get(arch, "?"), "concept": concept,
                "n_rows": len(te), "prevalence": round(float(y[te].mean()), 4),
                "probe_fixed_locus": FIXED_LOCUS,
                "probe_fixed": round(per_locus.get(FIXED_LOCUS, float("nan")), 4),
                "probe_max": round(per_locus[best], 4), "probe_max_locus": best,
                "probe_median": round(float(np.median(list(per_locus.values()))), 4),
                "n_loci_used": len(per_locus),
                "answer_loci_excluded": int(concept != EXTRACT_PROMPT_CONCEPT),
                "behav_auroc": float(br["behavioural_auroc"]),
                "gap_fixed": round(per_locus.get(FIXED_LOCUS, float("nan")) - float(br["behavioural_auroc"]), 4),
                "gap_max": round(per_locus[best] - float(br["behavioural_auroc"]), 4),
                "mean_p_yes": float(br["mean_p_yes"]), "sd_p_yes": float(br["sd_p_yes"]),
                "frac_pinned": float(br["frac_pinned"]), "yes_rate": float(br["yes_rate"]),
            })

    with open("runs/behaviour_vs_probe_samerows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    pf = np.array([r["probe_fixed"] for r in rows]); pm = np.array([r["probe_max"] for r in rows])
    h = np.array([r["behav_auroc"] for r in rows])
    print("\n" + "=" * 96)
    print("D0 on identical rows, prompt-independent loci")
    print("=" * 96)
    print(f"{'variant':38s} {'median probe':>13s} {'median behav':>13s} {'median gap':>11s} {'probe>behav':>12s}")
    for name, p in ((f"fixed locus ({FIXED_LOCUS}), no selection", pf),
                    ("max over usable loci (upper bound)", pm)):
        ok = np.isfinite(p)
        print(f"{name:38s} {np.median(p[ok]):13.4f} {np.median(h[ok]):13.4f} "
              f"{np.median(p[ok]-h[ok]):+11.4f} {int((p[ok]>h[ok]).sum()):8d}/{ok.sum()}")
    print(f"\n  {'model':13s} {'dom':8s} {'probe@'+FIXED_LOCUS:>14s} {'behaviour':>10s} {'gap':>8s} "
          f"{'always-yes':>11s}")
    for a in sorted({r["arch"] for r in rows}):
        s = [r for r in rows if r["arch"] == a]
        print(f"  {a:13s} {s[0]['domain']:8s} {np.median([r['probe_fixed'] for r in s]):14.3f} "
              f"{np.median([r['behav_auroc'] for r in s]):10.3f} "
              f"{np.median([r['gap_fixed'] for r in s]):+8.3f} "
              f"{sum(r['yes_rate']>=0.999 for r in s):8d}/{len(s)}")
    print(f"\nwrote runs/behaviour_vs_probe_samerows.csv")


if __name__ == "__main__":
    main()
