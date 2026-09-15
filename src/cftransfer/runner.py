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
from .fit import load_fit, run_id_for
from .hooks import LocusHook
from .images import image_path, load_cohort, open_rgb
from .protocol import (ALTDIR_FAMILIES, CONCEPTS, LOCI, MODULES, PROTOCOL_ID, conditions_for, direction_kind, primary_template,
                       question_list, render_question)
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
])
KEY = ["run_id", "module", "row_id", "concept", "template_id", "locus_id", "fit_seed", "direction_id", "alpha"]


class DirectionBank:
    """direction_id -> unit vector for one (dataset, locus, seed); random family always from seed 0. The ALTDIR module
    (altdir=True) also loads the four alternative families from fits/<locus>/altdir_seed0.npz, written by the CPU prep
    step `python -m cftransfer.altdir`; a missing file fails here with that command, before any model is loaded."""

    def __init__(self, model_key, dataset_id, locus_id, seeds, altdir: bool = False):
        self.fits = {s: load_fit(model_key, dataset_id, locus_id, s) for s in seeds}
        if 0 not in self.fits:
            self.fits[0] = load_fit(model_key, dataset_id, locus_id, 0)
        self.concepts = list(self.fits[0]["concept_names"].astype(str))
        self.D = int(self.fits[0]["clinical_vectors"].shape[1])
        self.altdir = load_altdir(model_key, dataset_id, locus_id) if altdir else None
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
        if kind in ALTDIR_FAMILIES:
            if self.altdir is None:
                raise KeyError(f"{direction_id}: alternative directions are only loaded for the ALTDIR module")
            return self.altdir[f"{kind}_vectors"][self.concepts.index(name)]
        raise KeyError(direction_id)


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
              rows_per_part: int = 10, limit_rows: int | None = None, revision: str | None = None) -> dict:
    spec = MODULES[module]
    if dataset_id not in spec.datasets:
        raise SystemExit(f"{module} is NOT_REQUESTED for {dataset_id}")
    locus_id = LOCI[spec.locus]
    # directions first: a missing fit or altdir prep file fails before the outcomes directory or the model exist
    bank = DirectionBank(model_key, dataset_id, locus_id, spec.fit_seeds, altdir=spec.directions == "altdir")
    rows = load_cohort(dataset_id, (spec.role,))
    if spec.row_limit:
        rows = rows[:spec.row_limit]
    if limit_rows:
        rows = rows[:limit_rows]
    rows = rows[shard::n_shards]
    part_dir = outcomes_dir(model_key, dataset_id) / module
    part_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f"{shard:03d}of{n_shards:03d}"
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
                "primary_template": primary, "ineligible_templates": skipped,
                "note": "every template of this module is INELIGIBLE by preflight check E; "
                "the shard is terminal with no outcomes (coverage records the disposition)",
                "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        part_dir.mkdir(parents=True, exist_ok=True)
        (part_dir / f"meta-{shard_tag}-{int(time.time())}.json").write_text(json.dumps(meta, indent=1))
        return meta
    expected_per_row = sum(len(conditions_for(module, dataset_id, c)) for c, _t in questions) * len(spec.fit_seeds)
    done = completed_rows(part_dir, shard_tag, expected_per_row)
    todo = [r for r in rows if r["row_id"] not in done]
    print(f"[{model_key}/{dataset_id}/{module} shard {shard_tag}] {len(rows)} rows, {len(done)} done, {len(todo)} todo "
          f"({expected_per_row} outcomes per row)", flush=True)
    if not todo:
        meta = {"model_key": model_key, "dataset_id": dataset_id, "module": module, "shard": shard, "n_shards": n_shards,
                "rows": len(rows), "todo": 0, "scored_rows": 0, "outcomes": 0, "seconds": 0.0, "throughput_per_s": 0.0, "batch": batch,
                "primary_template": primary, "note": "every row of the shard was already fully scored; nothing to do",
                "ended_utc": time.strftime("%Y-%m-%dT%T", time.gmtime()) + "Z"}
        (part_dir / f"meta-{shard_tag}-{int(time.time())}.json").write_text(json.dumps(meta, indent=1))
        return meta

    ad = get_adapter(model_key, revision).load(device_map=device_map)
    locus = ad.loci()[locus_id]
    hook = LocusHook(ad.module(locus.module_path), locus_id)
    run_id = run_id_for(model_key, dataset_id)
    if spec.directions == "clean":
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
                conds = [(d, a, s) for s in spec.fit_seeds for d, a in conditions_for(module, dataset_id, concept)]
                base = dict(protocol_id=PROTOCOL_ID, run_id=run_id, model_key=model_key, dataset_id=dataset_id,
                            module=module, role=row["role"], row_id=row["row_id"], unit_id=row["unit_id"],
                            concept=concept, template_id=template_id, locus_id=locus_id,
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
            "throughput_per_s": round(n_out / max(el, 1e-9), 2), "batch": batch, "primary_template": primary,
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
    a = ap.parse_args()
    run_block(a.model_key, a.dataset, a.module, a.shard, a.n_shards, a.batch, a.device_map, a.rows_per_part, a.limit_rows)
