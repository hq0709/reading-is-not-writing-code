#!/usr/bin/env python
"""Crossed tower and reader: the factor effects on OWNERSHIP, not only on the own-write magnitude.

The packaged TOWERSWAP crossover reports, for each factor held fixed, the mean absolute paired change in the six
own-question write magnitudes W_qq. The claim the experiment supports is about ownership, so this script recomputes
the same four contrasts for two further quantities, from the same stored per-image outcomes, the same 200 rows and
the same shared bootstrap draws:

  W_qq        the own-question write magnitude                              (reproduces the packaged numbers)
  O_q         the ownership contrast, W_qq - max over the five competing concept directions
  M_q         the margin over the strongest competitor in the full reference family,
              W_qq - max(the five competing concept directions, the 119-random 95th percentile, |sham|)

Each arm's statistic is recomputed inside every bootstrap draw, with the maximum recomputed there too, so the paired
difference between two arms carries a percentile interval and a simultaneous max-T interval over the six questions
exactly as the packaged W_qq contrast does.

The four contrasts are the campaign's:
  reader effect at own tower       (tower A, reader A) - (tower A, reader B)
  reader effect at partner tower   (tower B, reader A) - (tower B, reader B)
  tower effect at own reader       (tower B, reader A) - (tower A, reader A)
  tower effect at partner reader   (tower A, reader B) - (tower B, reader B)

Replacing a tower also replaces the directions fitted on it, so the tower factor moves the representation and the
written direction together; the reader factor moves neither. That asymmetry is a property of the design and is
recorded in the output as `tower_factor_carries`.

Read-only over runs/<model>/<dataset>/. Writes runs/robustness/crossover.json and crossover.md.

Usage (from the concept-flow repo root):
  PYTHONPATH=src python scripts/mayo/robustness_crossover.py [--draws 2000] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from cftransfer.analysis import _load_module, _matrix, _primary, core_bootstrap_indices, max_t   # noqa: E402
from cftransfer.manifest import block_included                                                   # noqa: E402
from cftransfer.protocol import (BOOT_CALIBRATION_DRAWS, CONCEPTS, MODULES, N_RANDOM, PRIMARY_ALPHA,  # noqa: E402
                                 TOWERSWAP_PAIRS)
from cftransfer.runpaths import RUN_ROOT, run_dir                                                # noqa: E402

OUT_DIR = RUN_ROOT / "robustness"
QUANTITIES = ("W_qq", "O_q", "M_q")
CONTRASTS = ("reader_effect_at_own_tower", "reader_effect_at_partner_tower",
             "tower_effect_at_own_reader", "tower_effect_at_partner_reader")


def _combo(tower: str, reader: str) -> str:
    return f"tower={tower}|reader={reader}"


def _arm_surface(reader: str, dataset: str, module: str, concepts: list[str], order: dict, n: int) -> dict | None:
    """Per-row deltas of one arm: the six concept directions, the random family and the question's sham."""
    df = _load_module(reader, dataset, module)
    if df is None or df.empty:
        return None
    base = df[df.direction_id == "baseline"]
    if base.empty:
        return None
    tpl = _primary(reader, dataset)
    d = _matrix(df, concepts, order, PRIMARY_ALPHA, template_id=tpl, baseline=base)
    if any((q, f"concept:{c}") not in d for q in concepts for c in concepts):
        return None
    out = {}
    for qi, q in enumerate(concepts):
        conc = np.stack([d[(q, f"concept:{c}")] for c in concepts]).astype(float)           # (6, n)
        rand = np.stack([d[(q, f"random:{i:03d}")] for i in range(N_RANDOM)
                         if (q, f"random:{i:03d}") in d]).astype(float)
        sham = d.get((q, f"sham:{q}"))
        if rand.shape[0] != N_RANDOM or sham is None:
            return None
        out[q] = {"concept": conc, "random": rand, "sham": np.asarray(sham, float), "index": qi}
    return out


def _statistics(surface: dict, concepts: list[str], rows: np.ndarray | None) -> np.ndarray:
    """(3, 6) array of W_qq, O_q and M_q over the questions, on the given rows."""
    out = np.empty((3, len(concepts)))
    for q, s in surface.items():
        qi = s["index"]
        m = np.nanmean(s["concept"] if rows is None else s["concept"][:, rows], axis=1)      # (6,)
        own = m[qi]
        comp = np.delete(m, qi).max()
        rnd = np.nanmean(s["random"] if rows is None else s["random"][:, rows], axis=1)
        sham = abs(float(np.nanmean(s["sham"] if rows is None else s["sham"][rows])))
        out[0, qi] = own
        out[1, qi] = own - comp
        out[2, qi] = own - max(comp, float(np.percentile(rnd, 95)), sham)
    return out


def _paired(a: np.ndarray, b: np.ndarray, boot_a: np.ndarray, boot_b: np.ndarray,
            concepts: list[str], names: list[str]) -> dict:
    """Question-wise a - b for each of the three quantities, with percentile and simultaneous max-T intervals."""
    out = {"contrast": names}
    for qi_stat, stat in enumerate(QUANTITIES):
        est = a[qi_stat] - b[qi_stat]
        boot = boot_a[:, qi_stat, :] - boot_b[:, qi_stat, :]                                  # (B, 6)
        mt = max_t(est, boot)
        per_q = {q: {"estimate": float(est[i]), "sd": float(mt["sd"][i]),
                     "ci95_percentile": [float(np.percentile(boot[:, i], 2.5)),
                                         float(np.percentile(boot[:, i], 97.5))],
                     "max_t_lower": float(mt["lower"][i]), "max_t_upper": float(mt["upper"][i]),
                     "nonzero_simultaneous": bool(mt["lower"][i] > 0 or mt["upper"][i] < 0)}
                 for i, q in enumerate(concepts)}
        mab = np.mean(np.abs(boot), axis=1)                                                   # (B,)
        out[stat] = per_q | {"max_t_critical": mt["critical"],
                             "mean_abs_effect": float(np.mean(np.abs(est))),
                             "mean_abs_effect_ci95": [float(np.percentile(mab, 2.5)), float(np.percentile(mab, 97.5))],
                             "mean_abs_effect_boot": mab,
                             "n_simultaneously_nonzero": int(sum(v["nonzero_simultaneous"] for v in per_q.values()))}
    return out


def crossover_block(model_key: str, dataset: str, draws: int) -> dict | None:
    partner = TOWERSWAP_PAIRS.get(model_key)
    if partner is None:
        return None
    from cftransfer.images import load_cohort
    concepts = CONCEPTS[dataset]
    n_rows = MODULES["TOWERSWAP"].row_limit
    rows = load_cohort(dataset, ("test",))[:n_rows]
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    arms = {}
    for reader, tower, module in ((model_key, model_key, "CORE"), (model_key, partner, "TOWERSWAP"),
                                  (partner, partner, "CORE"), (partner, model_key, "TOWERSWAP")):
        s = _arm_surface(reader, dataset, module, concepts, order, n)
        if s is None:
            return None
        arms[_combo(tower, reader)] = s
    idx = core_bootstrap_indices(dataset, n, draws)
    est = {k: _statistics(v, concepts, None) for k, v in arms.items()}
    boot = {k: np.stack([_statistics(v, concepts, idx[b]) for b in range(idx.shape[0])]) for k, v in arms.items()}
    a_self, a_cross = _combo(model_key, model_key), _combo(partner, model_key)
    b_self, b_cross = _combo(partner, partner), _combo(model_key, partner)
    pairs = {"reader_effect_at_own_tower": (a_self, b_cross), "reader_effect_at_partner_tower": (a_cross, b_self),
             "tower_effect_at_own_reader": (a_cross, a_self), "tower_effect_at_partner_reader": (b_cross, b_self)}
    res = {"block": f"{model_key}/{dataset}", "model": model_key, "dataset": dataset, "reader": model_key,
           "partner": partner, "n_rows": n, "alpha": PRIMARY_ALPHA, "draws": int(idx.shape[0]),
           "arms": sorted(arms), "quantities": list(QUANTITIES), "crossover": {}}
    for name, (x, y) in pairs.items():
        res["crossover"][name] = _paired(est[x], est[y], boot[x], boot[y], concepts, [x, y])
    for stat in QUANTITIES:
        r = float(np.mean([res["crossover"][c][stat]["mean_abs_effect"] for c in CONTRASTS[:2]]))
        t = float(np.mean([res["crossover"][c][stat]["mean_abs_effect"] for c in CONTRASTS[2:]]))
        rb = np.mean([res["crossover"][c][stat]["mean_abs_effect_boot"] for c in CONTRASTS[:2]], axis=0)
        tb = np.mean([res["crossover"][c][stat]["mean_abs_effect_boot"] for c in CONTRASTS[2:]], axis=0)
        res["crossover"][f"mean_abs_reader_effect_{stat}"] = r
        res["crossover"][f"mean_abs_tower_effect_{stat}"] = t
        res["crossover"][f"mean_abs_reader_effect_{stat}_ci95"] = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]
        res["crossover"][f"mean_abs_tower_effect_{stat}_ci95"] = [float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))]
        res["crossover"][f"reader_minus_tower_{stat}"] = float(r - t)
        res["crossover"][f"reader_minus_tower_{stat}_ci95"] = [float(np.percentile(rb - tb, 2.5)),
                                                               float(np.percentile(rb - tb, 97.5))]
        res["crossover"][f"reader_over_tower_{stat}"] = float(r / t) if t > 0 else None
    for c in CONTRASTS:                                     # the per-draw arrays are working state, not output
        for stat in QUANTITIES:
            res["crossover"][c][stat].pop("mean_abs_effect_boot", None)
    res["tower_factor_carries"] = ("the representation and the directions fitted on it; the reader factor changes "
                                   "neither, because a swapped arm writes the donor tower's own seed-0 fit")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=BOOT_CALIBRATION_DRAWS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    t0 = time.time()
    blocks = []
    for rj in sorted(RUN_ROOT.glob("*/*/run.json")):
        run = json.loads(rj.read_text())
        mk, ds = run.get("model_key", rj.parent.parent.name), run.get("dataset_id", rj.parent.name)
        if not block_included(run) or mk not in TOWERSWAP_PAIRS:
            continue
        sp = run_dir(mk, ds) / "summary.json"
        s = json.loads(sp.read_text()) if sp.exists() else {}
        if not (s.get("towerswap") or {}).get("crossover_complete"):
            continue
        r = crossover_block(mk, ds, args.draws)
        if r is None:
            continue
        packaged = s["towerswap"]["crossover"]
        r["packaged_mean_abs_reader_effect"] = packaged["mean_abs_reader_effect"]
        r["packaged_mean_abs_tower_effect"] = packaged["mean_abs_tower_effect"]
        r["reproduces_packaged_W_qq"] = bool(
            abs(r["crossover"]["mean_abs_reader_effect_W_qq"] - packaged["mean_abs_reader_effect"]) < 1e-9
            and abs(r["crossover"]["mean_abs_tower_effect_W_qq"] - packaged["mean_abs_tower_effect"]) < 1e-9)
        blocks.append(r)
        print(f"  {r['block']}: reader |dO| {r['crossover']['mean_abs_reader_effect_O_q']:.3f}, "
              f"tower |dO| {r['crossover']['mean_abs_tower_effect_O_q']:.3f}, "
              f"W_qq reproduced: {r['reproduces_packaged_W_qq']}")
    if not blocks:
        raise SystemExit("no block carries a complete tower-and-reader crossover")
    # the four arms of a dataset are the same four files whichever of the pair's two blocks indexes them, so the two
    # host blocks of a dataset must give the same crossover; any disagreement means one summary predates a rescored
    # arm, and the recomputation here -- one pass over the current outcomes -- is the one the tables use
    off = [b["block"] for b in blocks if not b["reproduces_packaged_W_qq"]]
    out = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_root": str(RUN_ROOT),
                    "draws": args.draws, "alpha": PRIMARY_ALPHA, "quantities": list(QUANTITIES),
                    "contrasts": list(CONTRASTS), "seconds": round(time.time() - t0, 1),
                    "script": "scripts/mayo/robustness_crossover.py",
                    "blocks_whose_packaged_W_qq_crossover_predates_the_current_outcomes": off},
           "blocks": blocks}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "crossover.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["# Crossed tower and reader: factor effects on ownership", "",
         f"{len(blocks)} block(s) with all four arms, {args.draws} shared draws of the "
         f"{blocks[0]['n_rows']} rows.", "",
         "| block | quantity | reader effect (own tower) | reader effect (partner tower) | tower effect (own reader) "
         "| tower effect (partner reader) | mean reader | mean tower |", "|---|---|---|---|---|---|---|---|"]
    for b in blocks:
        for stat in QUANTITIES:
            c = b["crossover"]
            L.append(f"| {b['block']} | {stat} | "
                     + " | ".join(f"{c[k][stat]['mean_abs_effect']:.3f} [{c[k][stat]['n_simultaneously_nonzero']}]"
                                  for k in CONTRASTS)
                     + f" | {c[f'mean_abs_reader_effect_{stat}']:.3f} | {c[f'mean_abs_tower_effect_{stat}']:.3f} |")
    (args.out / "crossover.md").write_text("\n".join(L) + "\n")
    print(f"wrote {args.out / 'crossover.json'} ({len(blocks)} blocks, {round(time.time() - t0, 1)}s)")


if __name__ == "__main__":
    main()
