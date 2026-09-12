"""Extract pooled, unprojected features at both loci for every manifest row (return-format features/<locus>.npz).

Both loci are prompt-independent for every family handled here, so only the vision tower and connector run.
Pooling is the mean over the consumed image tokens of the clean activation, in float32, stored as float16.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .adapters import get_adapter
from .hooks import LocusHook
from .images import image_path, load_cohort, open_rgb
from .runpaths import features_dir

ROLES_ALL = ("train", "preflight", "calibration", "test")


def extract(model_key: str, dataset_id: str, roles: tuple[str, ...], batch_size: int, device_map: str,
            out_dir: Path, shard: int = 0, n_shards: int = 1, revision: str | None = None) -> dict:
    rows = load_cohort(dataset_id, roles)
    rows = rows[shard::n_shards]
    ad = get_adapter(model_key, revision).load(device_map=device_map)
    loci = ad.loci()
    hooks = {lid: LocusHook(ad.module(info.module_path), lid) for lid, info in loci.items()}
    acc = {lid: [] for lid in loci}
    counts = {lid: [] for lid in loci}
    ids, fit_role = [], []
    t0 = time.time()
    with hooks["vis.last"], hooks["connector"]:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            images = [open_rgb(image_path(dataset_id, r)) for r in chunk]
            enc = ad.encode(images, ["Is there a finding in this image? Answer yes or no."] * len(chunk))
            lay = ad.layouts(enc, images)
            for lid, h in hooks.items():
                h.arm(lay[lid], capture=True)
            ad.vision_features(enc)
            for lid, h in hooks.items():
                if h.pooled is None or h.pooled.shape[0] != len(chunk):
                    raise RuntimeError(f"{lid}: captured {None if h.pooled is None else h.pooled.shape} for {len(chunk)} rows")
                acc[lid].append(h.pooled.to(torch.float16).numpy())
                counts[lid].extend(lay[lid].counts())
            ids.extend(r["row_id"] for r in chunk)
            fit_role.extend(r["role"] for r in chunk)
            if (i // batch_size) % 50 == 0:
                print(f"[{model_key}/{dataset_id}] {i + len(chunk)}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model_key": model_key, "dataset_id": dataset_id, "roles": roles, "rows": len(ids), "shard": shard,
            "n_shards": n_shards, "seconds": round(time.time() - t0, 1), "batch_size": batch_size,
            "loci": {lid: vars(info) for lid, info in loci.items()},
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}
    for lid in loci:
        x = np.concatenate(acc[lid])
        suffix = "" if n_shards == 1 else f".shard{shard:03d}of{n_shards:03d}"
        np.savez(out_dir / f"{lid}{suffix}.npz", row_id=np.array(ids, dtype="U32"), x=x,
                 valid_token_count=np.array(counts[lid], dtype=np.int32), fit_role=np.array(fit_role, dtype="U12"))
        meta[f"shape_{lid}"] = list(x.shape)
    (out_dir / f"features_meta{'' if n_shards == 1 else f'.shard{shard:03d}'}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "loci"}), flush=True)
    return meta


def merge_shards(out_dir: Path, locus_id: str, n_shards: int) -> None:
    parts = [np.load(out_dir / f"{locus_id}.shard{s:03d}of{n_shards:03d}.npz") for s in range(n_shards)]
    np.savez(out_dir / f"{locus_id}.npz", row_id=np.concatenate([p["row_id"] for p in parts]),
             x=np.concatenate([p["x"] for p in parts]),
             valid_token_count=np.concatenate([p["valid_token_count"] for p in parts]),
             fit_role=np.concatenate([p["fit_role"] for p in parts]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--roles", default="train,preflight,calibration,test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--merge", action="store_true", help="merge existing shards instead of extracting")
    a = ap.parse_args()
    out = Path(a.out_dir) if a.out_dir else features_dir(a.model_key, a.dataset)
    if a.merge:
        for lid in ("vis.last", "connector"):
            merge_shards(out, lid, a.n_shards)
        print("merged")
    else:
        extract(a.model_key, a.dataset, tuple(a.roles.split(",")), a.batch_size, a.device_map, out, a.shard, a.n_shards)
