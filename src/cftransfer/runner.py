"""GPU runner: score one (model, dataset, module) block, or a row shard of it, into outcomes parquet.

Every (row, question) is scored under the module's condition grid (protocol.conditions_for). Conditions
of one question share the encoded input; they are batched along the batch axis with one direction and
dose per batch element (the hook applies per-element deltas). Baseline rows are true clean forwards
with the hook passive. Output rows carry every return-format column; failed rows keep their key and an
error reason with empty scores. Parts are resumable: existing row_ids in this shard's parts are skipped.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .adapters import get_adapter
from .altdir import load_altdir
from .ansdir import load_ansdir
from .attr import load_attr
from .extcomp import load_extcomp
from .fit import load_fit, run_id_for
from .hooks import LocusHook
from .images import image_path, load_cohort, open_rgb
from .hooks import MODE_SOFTMAX, MODE_TOPQ
from .projseed import load_projseed
from .protocol import (ALTDIR_FAMILIES, ALTDIRD_FAMILIES, ATTR_CONCEPTS, CONCEPTS, LOCI, MODULE_SETTINGS, MODULES, NUMERICS_DEFAULT,
                       PROTOCOL_ID, TOKENW_VARIANTS, conditions_for, direction_kind, primary_template, question_list, render_question)
from .validfit import load_validfit
from .runpaths import outcomes_dir, run_dir
from .scoring import score_logits

SCHEMA = pa.schema([
    ("protocol_id", pa.string()), ("run_id", pa.string()), ("model_key", pa.string()), ("dataset_id", pa.string()),
    ("module", pa.string()), ("role", pa.string()), ("row_id", pa.string()), ("unit_id", pa.string()),
    ("concept", pa.string()), ("template_id", pa.string()), ("locus_id", pa.string()), ("fit_seed", pa.int32()),
    ("direction_id", pa.string()), ("direction_kind", pa.string()), ("alpha", pa.float64()),
    ("positive_token_ids", pa.list_(pa.int32())), ("negative_token_ids", pa.list_(pa.int32())),
    ("positive_logits", pa.list_(pa.float32())), ("negative_logits", pa.list_(pa.float32())),
    ("vocab_logsumexp", pa.float32()), ("semantic_margin", pa.float32()), ("p_present", pa.float64()),
    ("lse_margin", pa.float32()), ("raw_ab_margin", pa.float32()), ("candidate_mass", pa.float32()),
    ("valid_token_count", pa.int32()), ("input_token_count", pa.int32()),
    ("token_norm_mean", pa.float32()), ("token_norm_median", pa.float32()), ("delta_norm_mean", pa.float32()),
    ("sample_status", pa.string()), ("error_reason", pa.string()),
    ("numerics", pa.string()),          # NUMERICS_DEFAULT for every module but PRECISION (fp32 | batch1); added later,
])                                      # so package.merge_module fills it in for older part files
KEY = ["run_id", "module", "row_id", "concept", "template_id", "locus_id", "fit_seed", "direction_id", "alpha", "numerics"]
TOKENW_MODE = {"tokenw": MODE_SOFTMAX, "topq": MODE_TOPQ}


class DirectionBank:
    """direction_id -> unit vector for one (dataset, locus, seed); random family always from seed 0. The ALTDIR module
    (altdir=True) also loads the four alternative families from fits/<locus>/altdir_seed0.npz, written by the CPU prep
    step `python -m cftransfer.altdir`; a missing file fails here with that command, before any model is loaded."""

    def __init__(self, model_key, dataset_id, locus_id, seeds, altdir: bool = False, extcomp: bool = False, tokenw: bool = False,
                 ansdir: bool = False, altdird: bool = False, attr: bool = False, validfit: bool = False,
                 projseed: tuple[int, ...] | None = None):
        self.fits = {s: load_fit(model_key, dataset_id, locus_id, s) for s in seeds}
        if 0 not in self.fits:
            self.fits[0] = load_fit(model_key, dataset_id, locus_id, 0)
        self.concepts = list(self.fits[0]["concept_names"].astype(str))
        self.D = int(self.fits[0]["clinical_vectors"].shape[1])
        self.extcomp = load_extcomp(model_key, dataset_id, locus_id) if extcomp else None
        if self.extcomp is not None:
            self.extra = list(self.extcomp["extra_names"].astype(str))
            if self.extcomp["extra_vectors"].shape[1] != self.D:
                raise RuntimeError("extcomp_seed0.npz extra_vectors width differs from the fit")
        # TOKENW token scorer: h_t . (P w_c / s), the token's projected, scaled probe logit up to a constant
        self.scorers = {s: (f["projection"] @ (f["coefficients"] / np.maximum(f["scaler_scale"], 1e-8)).T).T.astype(np.float32)
                        for s, f in self.fits.items()} if tokenw else None
        self.ansdir = load_ansdir(model_key, dataset_id, locus_id) if ansdir else None
        if self.ansdir is not None and (list(self.ansdir["concept_names"].astype(str)) != self.concepts
                                        or self.ansdir["answer_vectors"].shape != (len(self.concepts), self.D)):
            raise RuntimeError("ansdir_seed0.npz concept order or width differs from seed0.npz")
        self.attr = load_attr(model_key, dataset_id, locus_id) if attr else None
        if self.attr is not None and (list(self.attr["attr_names"].astype(str)) != list(ATTR_CONCEPTS)
                                      or self.attr["attr_vectors"].shape != (len(ATTR_CONCEPTS), self.D)):
            raise RuntimeError("attr_seed0.npz attribute order or width differs from the protocol / fit")
        self.validfit = load_validfit(model_key, dataset_id, locus_id) if validfit else None
        if self.validfit is not None and (list(self.validfit["concept_names"].astype(str)) != self.concepts
                                          or self.validfit["expert_vectors"].shape != (len(self.concepts), self.D)):
            raise RuntimeError("validfit_seed0.npz concept order or width differs from seed0.npz")
        # PROJSEED: one refit file per PROJECTION seed; the seed travels in the condition's seed slot (the fit_seed column)
        self.projseed = {k: load_projseed(model_key, dataset_id, locus_id, k) for k in (projseed or ())}
        for k, arr in self.projseed.items():
            if list(arr["concept_names"].astype(str)) != self.concepts or arr["clinical_vectors"].shape != (len(self.concepts), self.D):
                raise RuntimeError(f"projseed_seed{k}.npz concept order or width differs from seed0.npz")
            if arr["random_vectors"].shape[1] != self.D or int(arr["projection_seed"]) != k:
                raise RuntimeError(f"projseed_seed{k}.npz random family width or projection seed is wrong")
        self.altdir = load_altdir(model_key, dataset_id, locus_id) if (altdir or altdird) else None
        if altdird and any(f"{fam}_vectors" not in self.altdir for fam in ALTDIRD_FAMILIES):
            raise FileNotFoundError(f"ALTDIRD needs the displacement families in altdir_seed0.npz; re-run "
                                    f"python -m cftransfer.altdir --model-key {model_key} --dataset {dataset_id} (appends them)")
        if self.altdir is not None:
            if list(self.altdir["concept_names"].astype(str)) != self.concepts:
                raise RuntimeError("altdir_seed0.npz concept order differs from seed0.npz")
            for fam in ALTDIR_FAMILIES:
                if self.altdir[f"{fam}_vectors"].shape != (len(self.concepts), self.D):
                    raise RuntimeError(f"altdir_seed0.npz {fam}_vectors has shape {self.altdir[f'{fam}_vectors'].shape}")

    def vector(self, direction_id: str, seed: int) -> np.ndarray | None:
        kind, _, name = direction_id.partition(":")
        if kind == "baseline":
            return None
        if kind == "concept":
            return self.fits[seed]["clinical_vectors"][self.concepts.index(name)]
        if kind == "random":
            return self.fits[0]["random_vectors"][int(name)]
        if kind == "sham":
            return self.fits[seed]["sham_vectors"][self.concepts.index(name)]
        if kind in ALTDIR_FAMILIES or kind in ALTDIRD_FAMILIES:
            if self.altdir is None or f"{kind}_vectors" not in self.altdir:
                raise KeyError(f"{direction_id}: alternative directions are only loaded for the ALTDIR / ALTDIRD modules")
            return self.altdir[f"{kind}_vectors"][self.concepts.index(name)]
        if kind in ("attr", "attrsham"):
            if self.attr is None:
                raise KeyError(f"{direction_id}: attribute directions are only loaded for the ATTR module")
            return self.attr["attr_vectors" if kind == "attr" else "attr_sham_vectors"][list(ATTR_CONCEPTS).index(name)]
        if kind == "extra":
            if self.extcomp is None:
                raise KeyError(f"{direction_id}: extra directions are only loaded for the EXTCOMP module")
            return self.extcomp["extra_vectors"][self.extra.index(name)]
        if kind in TOKENW_VARIANTS:
            return self.fits[seed]["clinical_vectors"][self.concepts.index(name)]      # weighted write of the logistic direction
        if kind in ("ans", "anssham"):
            if self.ansdir is None:
                raise KeyError(f"{direction_id}: answer directions are only loaded for the ANSDIR module")
            return self.ansdir["answer_vectors" if kind == "ans" else "answer_sham_vectors"][self.concepts.index(name)]
        if kind == "vfit":
            if self.validfit is None:
                raise KeyError(f"{direction_id}: expert-label directions are only loaded for the VALIDFIT module")
            return self.validfit["expert_vectors"][self.concepts.index(name)]
        if kind in ("proj", "projrand", "projsham"):
            if seed not in self.projseed:
                raise KeyError(f"{direction_id}: projection-seed directions are only loaded for the PROJSEED module (seed {seed})")
            arr = self.projseed[seed]
            if kind == "projrand":
                return arr["random_vectors"][int(name)]
            return arr["clinical_vectors" if kind == "proj" else "sham_vectors"][self.concepts.index(name)]
        raise KeyError(direction_id)

    def scorer(self, direction_id: str, seed: int) -> np.ndarray:
        kind, _, name = direction_id.partition(":")
        if self.scorers is None or kind not in TOKENW_VARIANTS:
            raise KeyError(f"{direction_id}: token scorers are only built for the TOKENW module")
        return self.scorers[seed][self.concepts.index(name)]


def completed_rows(part_dir: Path, shard_tag: str, expected_per_row: int | None = None) -> set[str]:
    """Row ids of this shard that are fully scored: when expected_per_row is given, a row counts only if it has that many
    OK outcomes across the shard's part files (a row whose failed outcomes were dropped is re-scored in full; the merge
    keeps the last copy of every KEY). Without it, any presence counts (legacy behaviour)."""
    counts: dict[str, int] = {}
    for p in part_dir.glob(f"part-{shard_tag}-*.parquet"):
        t = pq.read_table(p, columns=["row_id", "sample_status"])
        for rid, st in zip(t.column("row_id").to_pylist(), t.column("sample_status").to_pylist()):
            if expected_per_row is None or st == "OK":
                counts[rid] = counts.get(rid, 0) + 1
    if expected_per_row is None:
        return set(counts)
    return {rid for rid, n in counts.items() if n >= expected_per_row}


def run_block(model_key: str, dataset_id: str, module: str, shard: int, n_shards: int, batch: int, device_map: str,
              rows_per_part: int = 10, limit_rows: int | None = None, revision: str | None = None,
              numerics: str | None = None, fit_seed: int | None = None) -> dict:
    spec = MODULES[module]
    if dataset_id not in spec.datasets:
        raise SystemExit(f"{module} is NOT_REQUESTED for {dataset_id}")
    # numerics setting: required for modules scored per setting (PRECISION), fixed to the default elsewhere
    settings = MODULE_SETTINGS.get(module)
    if settings:
        if numerics not in settings:
            raise ValueError(f"{module} needs --numerics one of {settings}, got {numerics!r}")
    elif numerics not in (None, NUMERICS_DEFAULT):
        raise ValueError(f"{module} runs only under {NUMERICS_DEFAULT}; --numerics {numerics} is a PRECISION setting")
    else:
        numerics = NUMERICS_DEFAULT
    # fit-seed selection: a module whose seeds are independent grids (PROJSEED: one projection seed each) can be run one
    # seed per task, so a block costs one shard per seed; without the flag every seed of the spec is scored in one pass.
    if fit_seed is not None and fit_seed not in spec.fit_seeds:
        raise ValueError(f"{module} has fit seeds {spec.fit_seeds}; --fit-seed {fit_seed} is not one of them")
    seeds = spec.fit_seeds if fit_seed is None else (fit_seed,)
    locus_id = LOCI[spec.locus]
    # directions first: a missing fit or altdir/extcomp prep file fails before the outcomes directory or the model exist
    bank = DirectionBank(model_key, dataset_id, locus_id, (0,) if spec.directions == "projseed" else spec.fit_seeds,
                         altdir=spec.directions == "altdir",
                         extcomp=spec.directions == "extcomp", tokenw=spec.directions == "tokenw",
                         ansdir=spec.directions in ("ansdir", "ansdirt"), altdird=spec.directions == "altdird",
                         attr=spec.directions == "attr", validfit=spec.directions == "validfit",
                         projseed=seeds if spec.directions == "projseed" else None)
    rows = load_cohort(dataset_id, (spec.role,))
    if spec.row_limit:
        rows = rows[:spec.row_limit]
    if limit_rows:
        rows = rows[:limit_rows]
    rows = rows[shard::n_shards]
    part_dir = outcomes_dir(model_key, dataset_id) / module
    part_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f"{shard:03d}of{n_shards:03d}" if not settings else f"{numerics}-{shard:03d}of{n_shards:03d}"   # parts per setting
    if fit_seed is not None:
        shard_tag = f"s{fit_seed}-{shard_tag}"        # one part / meta stream per seed, so resume never mixes seeds
    # Questions: the block's primary template stands in for IY (protocol.primary_template: IB when preflight check E
    # marked IY INELIGIBLE). Templates still INELIGIBLE are dropped; a module left with no question closes as terminal
    # before the model is loaded.
    rd = run_dir(model_key, dataset_id)
    primary = primary_template(rd)
    questions = question_list(dataset_id, module, primary)
    if primary != "IY":
        print(f"[{model_key}/{dataset_id}/{module}] primary template {primary} (IY INELIGIBLE by preflight check E)", flush=True)
    skipped: list[str] = []
    elig_path = rd / "template_eligibility.json"
    if elig_path.exists():
        elig = json.loads(elig_path.read_text())
        skipped = sorted({t for _c, t in questions if not elig.get(t, {}).get("eligible", True)})
        if skipped:
            print(f"[{model_key}/{dataset_id}/{module}] templates INELIGIBLE by preflight, not scored: {skipped}", flush=True)
        questions = [(c, t) for c, t in questions if elig.get(t, {}).get("eligible", True)]
    if not questions:
        print(f"[{model_key}/{dataset_id}/{module}] no eligible questions; nothing to score", flush=True)
        meta = {"model_key": model_key, "dataset_id": dataset_id, "module": module, "shard": shard, "n_shards": n_shards,
                "rows": len(rows), "scored_rows": 0, "outcomes": 0, "seconds": 0.0, "throughput_per_s": 0.0, "batch": batch,
                "primary_template": primary, "ineligible_templates": skipped, "numerics": numerics, "fit_seed": fit_seed,
                "note": "every template of this module is INELIGIBLE by preflight check E; "
                "the shard is terminal with no outcomes (coverage records the disposition)",
                "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        part_dir.mkdir(parents=True, exist_ok=True)
        (part_dir / f"meta-{shard_tag}-{int(time.time())}.json").write_text(json.dumps(meta, indent=1))
        return meta
    expected_per_row = sum(len(conditions_for(module, dataset_id, c)) for c, _t in questions) * len(seeds)
    done = completed_rows(part_dir, shard_tag, expected_per_row)
    todo = [r for r in rows if r["row_id"] not in done]
    print(f"[{model_key}/{dataset_id}/{module} shard {shard_tag}] {len(rows)} rows, {len(done)} done, {len(todo)} todo "
          f"({expected_per_row} outcomes per row)", flush=True)
    if not todo:
        meta = {"model_key": model_key, "dataset_id": dataset_id, "module": module, "shard": shard, "n_shards": n_shards,
                "rows": len(rows), "todo": 0, "scored_rows": 0, "outcomes": 0, "seconds": 0.0, "throughput_per_s": 0.0, "batch": batch,
                "primary_template": primary, "fit_seed": fit_seed, "note": "every row of the shard was already fully scored; nothing to do",
                "ended_utc": time.strftime("%Y-%m-%dT%T", time.gmtime()) + "Z"}
        (part_dir / f"meta-{shard_tag}-{int(time.time())}.json").write_text(json.dumps(meta, indent=1))
        return meta

    # PRECISION settings: fp32 loads weights and runs the forward in float32 (CORE batch composition kept);
    # batch1 keeps bf16 and scores every condition, baseline included, as a batch of one
    dtype = torch.float32 if numerics == "fp32" else torch.bfloat16
    ad = get_adapter(model_key, revision).load(device_map=device_map, dtype=dtype)
    locus = ad.loci()[locus_id]
    hook = LocusHook(ad.module(locus.module_path), locus_id)
    run_id = run_id_for(model_key, dataset_id)
    if spec.directions == "clean" or numerics == "batch1":
        batch = 1          # clean-only modules score one forward per (row, question); no steered composition to match
    t0, n_out, part_idx = time.time(), 0, len(list(part_dir.glob(f"part-{shard_tag}-*.parquet")))
    buffer: list[dict] = []

    def flush():
        nonlocal buffer, part_idx
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=SCHEMA)
        tmp = part_dir / f"part-{shard_tag}-{part_idx:05d}.parquet.tmp"
        pq.write_table(table, tmp, compression="zstd")
        tmp.rename(part_dir / f"part-{shard_tag}-{part_idx:05d}.parquet")
        part_idx += 1
        buffer = []

    with hook:
        for ri, row in enumerate(todo):
            image = open_rgb(image_path(dataset_id, row))
            for concept, template_id in questions:
                question = render_question(dataset_id, concept, template_id)
                cands = ad.candidates[template_id]
                conds = [(d, a, s) for s in seeds for d, a in conditions_for(module, dataset_id, concept)]
                base = dict(protocol_id=PROTOCOL_ID, run_id=run_id, model_key=model_key, dataset_id=dataset_id,
                            module=module, role=row["role"], row_id=row["row_id"], unit_id=row["unit_id"],
                            concept=concept, template_id=template_id, locus_id=locus_id, numerics=numerics,
                            positive_token_ids=list(cands.positive_ids), negative_token_ids=list(cands.negative_ids))
                # Every forward of a question uses the SAME batch composition: `batch` replicas of one image and one
                # question. bf16 kernels are shape-dependent, so mixing a B=1 baseline with B=32 steered batches would
                # put a composition offset into every W_qd. The baseline is a clean replicated batch with the hook
                # passive (element 0 kept); steered batches are padded to `batch` by repeating conditions (discarded).
                clean = [c for c in conds if c[0] == "baseline"]
                steered = [c for c in conds if c[0] != "baseline"]
                groups = ([clean] if clean else []) + [steered[i:i + batch] for i in range(0, len(steered), batch)]
                enc = ad.expand(ad.encode([image], [question]), batch)
                lay = ad.layouts(enc, [image] * batch)[locus_id]
                for group in groups:
                    B = len(group)
                    padded = group + [group[-1]] * (batch - B) if group[0][0] != "baseline" else group
                    try:
                        if group[0][0] == "baseline":
                            hook.arm(lay)
                        else:
                            vecs = torch.from_numpy(np.stack([bank.vector(d, s) for d, _a, s in padded]))
                            alphas = torch.tensor([a for _d, a, _s in padded], dtype=torch.float32)
                            if spec.directions == "tokenw":
                                scorers = torch.from_numpy(np.stack([bank.scorer(d, s) for d, _a, s in padded]))
                                modes = torch.tensor([TOKENW_MODE[direction_kind(d)] for d, _a, _s in padded])
                                hook.arm(lay, vecs, alphas, scorers=scorers, modes=modes)
                            else:
                                hook.arm(lay, vecs, alphas)
                        logits = ad.forward_last_logits(enc)
                        if hook.calls != 1:
                            raise RuntimeError(f"hook fired {hook.calls} times")
                        sc = score_logits(logits, cands)
                        n_input = enc["attention_mask"].sum(dim=1).tolist()
                        counts = lay.counts()
                        for b, (d, a, s) in enumerate(group):
                            buffer.append({**base, "fit_seed": int(s), "direction_id": d, "direction_kind": direction_kind(d),
                                           "alpha": float(a), "positive_logits": sc.positive_logits[b].tolist(),
                                           "negative_logits": sc.negative_logits[b].tolist(),
                                           "vocab_logsumexp": float(sc.vocab_logsumexp[b]),
                                           "semantic_margin": float(sc.semantic_margin[b]), "p_present": float(sc.p_present[b]),
                                           "lse_margin": float(sc.lse_margin[b]), "raw_ab_margin": float(sc.raw_ab_margin[b]),
                                           "candidate_mass": float(sc.candidate_mass[b]),
                                           "valid_token_count": int(counts[b]), "input_token_count": int(n_input[b]),
                                           "token_norm_mean": hook.stats["token_norm_mean"][b],
                                           "token_norm_median": hook.stats["token_norm_median"][b],
                                           "delta_norm_mean": hook.stats["delta_norm_mean"][b],
                                           "sample_status": "OK", "error_reason": ""})
                    except Exception as e:                       # keep keys, empty scores, explicit reason
                        reason = f"{type(e).__name__}: {str(e)[:300]}"
                        print(f"FAILED {row['row_id']} {concept} {template_id} {group[0][0]}..: {reason}", flush=True)
                        for d, a, s in group:
                            buffer.append({**base, "fit_seed": int(s), "direction_id": d, "direction_kind": direction_kind(d),
                                           "alpha": float(a), "positive_logits": None, "negative_logits": None,
                                           "vocab_logsumexp": None, "semantic_margin": None, "p_present": None,
                                           "lse_margin": None, "raw_ab_margin": None, "candidate_mass": None,
                                           "valid_token_count": None, "input_token_count": None, "token_norm_mean": None,
                                           "token_norm_median": None, "delta_norm_mean": None,
                                           "sample_status": "FAILED", "error_reason": reason})
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                    n_out += B
            if (ri + 1) % rows_per_part == 0:
                flush()
                el = time.time() - t0
                print(f"[{model_key}/{dataset_id}/{module} {shard_tag}] {ri + 1}/{len(todo)} rows, {n_out} outcomes, "
                      f"{n_out / el:.1f}/s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        flush()
    el = time.time() - t0
    meta = {"model_key": model_key, "dataset_id": dataset_id, "module": module, "shard": shard, "n_shards": n_shards,
            "rows": len(rows), "scored_rows": len(todo), "outcomes": n_out, "seconds": round(el, 1),
            "throughput_per_s": round(n_out / max(el, 1e-9), 2), "batch": batch, "primary_template": primary, "numerics": numerics,
            "fit_seed": fit_seed,
            "batch_policy": "fixed composition per question: clean replicated baseline batch + padded steered batches",
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "gpu_count": torch.cuda.device_count(),
            "gpu_models": sorted({torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}),
            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (part_dir / f"meta-{shard_tag}-{int(time.time())}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--module", required=True, choices=list(MODULES))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32, help="conditions per forward (batch axis)")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--rows-per-part", type=int, default=10)
    ap.add_argument("--limit-rows", type=int, default=None, help="debug: first N rows only")
    ap.add_argument("--numerics", default=None, help="PRECISION only: fp32 | batch1")
    ap.add_argument("--fit-seed", type=int, default=None, help="score only this fit seed of the module (PROJSEED: the projection seed)")
    a = ap.parse_args()
    run_block(a.model_key, a.dataset, a.module, a.shard, a.n_shards, a.batch, a.device_map, a.rows_per_part, a.limit_rows,
              numerics=a.numerics, fit_seed=a.fit_seed)
