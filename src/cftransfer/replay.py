"""REPLAY: write the SAME stored consumed-block tensor into two readers, so the comparison is exact.

The shared-tower pairs of the grid (Gemma 3 4B/12B/27B share one SigLIP tower, MedGemma 4B/27B share one
MedSigLIP tower, LLaVA-1.5 7B/13B share one CLIP tower -- each verified bitwise identical in the staged
checkpoints, 437/437 and 391/391 tensors; LLaVA-Med 7B carries the same CLIP tower as LLaVA-1.5 7B up to its
fp16 -> bf16 storage, max |difference| 0.047 over the 391 tensors) are today compared as "the same tower, two readers". That is only
true up to bf16 kernel noise: each block ran its own tower forward, and a bf16 matmul's reduction order
depends on the batch shape and the device, so the two readers never saw byte-identical activations.

REPLAY removes that gap. One block of a pair (REPLAY_SOURCE) runs its tower once over the module's rows and
stores the consumed block's output; every block of the pair then scores the CORE grid with that stored
tensor substituted for its own tower output, so the tensor entering the reader is byte-identical and the
directions written on it are the SAME block's seed-0 directions. Everything that then differs is the reader.

Mechanics. runner.py obtains the consumed block as the forward output of one module (adapters' `vis.last`
locus) and steers it with LocusHook, which registers with prepend=True. ReplayHook registers on the same
module, also with prepend=True, AFTER LocusHook, so it runs FIRST: it replaces the module's output with the
stored tensor and LocusHook then computes the token norms and the write on the replayed tensor. Nothing
downstream can tell the difference, and the outcome schema is unchanged.

The tower still runs (the hook fires on its output), so a replay block costs what the same CORE grid costs
on the same rows; what it buys is exactness, not speed.

Storage: runs/<source>/<dataset>/replay/<locus>.npz -- row_id, x (n, T, D) with the ORIGINAL bits of the
captured tensor (bfloat16 stored as int16 and reinterpreted on load, so nothing is rounded), stored_dtype,
token_count, plus replay_meta.json. Gemma 3: 200 x 4096 x 1152 x 2 B = 1.9 GB; LLaVA-1.5: 0.24 GB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .hooks import as_hidden, with_hidden
from .protocol import LOCI, MODULES, REPLAY_SOURCE
from .runpaths import run_dir

PACK = {torch.bfloat16: ("bfloat16", torch.int16), torch.float16: ("float16", torch.float16),
        torch.float32: ("float32", torch.float32)}
UNPACK = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def replay_dir(model_key: str, dataset_id: str) -> Path:
    return run_dir(model_key, dataset_id) / "replay"


def replay_source(model_key: str) -> str:
    """The block whose stored tensors this block replays (itself for the pair's source block)."""
    if model_key not in REPLAY_SOURCE:
        raise KeyError(f"{model_key}: no REPLAY source; the shared-tower groups are "
                       f"{sorted(set(REPLAY_SOURCE.values()))} (see protocol.REPLAY_SOURCE)")
    return REPLAY_SOURCE[model_key]


def pack(t: torch.Tensor) -> tuple[np.ndarray, str]:
    """Numpy view of a tensor that preserves its bits exactly (bfloat16 travels as int16)."""
    if t.dtype not in PACK:
        raise TypeError(f"replay cannot store {t.dtype}")
    name, view = PACK[t.dtype]
    return t.detach().contiguous().cpu().view(view).numpy(), name


def unpack(arr: np.ndarray, stored_dtype: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).view(UNPACK[stored_dtype])


# ------------------------------------------------------------------------------------------- capture (GPU)
class _Capture:
    """Record the consumed block's output per batch element, in the tensor's own dtype."""

    def __init__(self, module: torch.nn.Module, name: str):
        self.module, self.name, self.value, self._handle = module, name, None, None

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def _hook(self, _m, _i, output):
        h = as_hidden(output)
        if h is None:
            raise RuntimeError(f"{self.name}: module returned no hidden-state tensor")
        self.value = h.detach()
        return None


def build_replay(model_key: str, dataset_id: str, locus_id: str = "vis.last", n_rows: int | None = None,
                 batch_size: int = 8, device_map: str = "cuda:0", revision: str | None = None,
                 out_dir: Path | None = None, adapter=None) -> tuple[Path, dict]:
    """Run the block's tower over the REPLAY rows and store the consumed block's output tensors."""
    from .adapters import get_adapter
    from .images import image_path, load_cohort, open_rgb
    n_rows = MODULES["REPLAY"].row_limit if n_rows is None else n_rows
    rows = load_cohort(dataset_id, ("test",))[:n_rows]
    ad = adapter if adapter is not None else get_adapter(model_key, revision).load(device_map=device_map)
    cap = _Capture(ad.module(ad.loci()[locus_id].module_path), locus_id)
    chunks, counts, ids, t0 = [], [], [], time.time()
    stored_dtype = None
    with cap:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            images = [open_rgb(image_path(dataset_id, r)) for r in chunk]
            enc = ad.encode(images, ["Is there a finding in this image? Answer yes or no."] * len(chunk))
            lay = ad.layouts(enc, images)[locus_id]
            cap.value = None
            ad.vision_features(enc)
            h = cap.value
            if h is None:
                raise RuntimeError(f"{model_key}/{dataset_id}: the {locus_id} hook did not fire")
            per_row = _split(h, lay, len(chunk), locus_id)
            for t in per_row:
                arr, name = pack(t)
                stored_dtype = stored_dtype or name
                if name != stored_dtype:
                    raise RuntimeError(f"mixed capture dtypes {name} / {stored_dtype}")
                chunks.append(arr)
            counts.extend(lay.counts())
            ids.extend(r["row_id"] for r in chunk)
            if (i // batch_size) % 10 == 0:
                print(f"[replay {model_key}/{dataset_id}] {i + len(chunk)}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    x = np.stack(chunks)
    out = Path(out_dir) if out_dir else replay_dir(model_key, dataset_id)
    out.mkdir(parents=True, exist_ok=True)
    width = max(max(len(i) for i in ids) + 1, 32)
    np.savez(out / f"{locus_id}.npz", row_id=np.array(ids, dtype=f"U{width}"), x=x,
             stored_dtype=np.array(stored_dtype), valid_token_count=np.array(counts, dtype=np.int32))
    meta = {"model_key": model_key, "dataset_id": dataset_id, "locus_id": locus_id, "rows": len(ids),
            "shape": list(x.shape), "stored_dtype": stored_dtype, "capture_batch_size": batch_size,
            "sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
            "seconds": round(time.time() - t0, 1),
            "note": "consumed-block output of this block's own tower on the first N test rows; replayed into "
                    "every reader of the shared-tower group (protocol.REPLAY_SOURCE)",
            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / f"replay_meta.{locus_id}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return out / f"{locus_id}.npz", meta


def _split(h: torch.Tensor, lay, n: int, locus_id: str) -> list[torch.Tensor]:
    """Per-batch-element (T, D) slices of a captured block output, for either token layout."""
    if lay.flat:
        if h.dim() != 2:
            raise RuntimeError(f"{locus_id}: flat layout expects (N, D), got {tuple(h.shape)}")
        return [h[s:e].clone() for s, e in lay.slices[:n]]
    if h.dim() != 3 or h.shape[0] != n:
        raise RuntimeError(f"{locus_id}: batched layout expects ({n}, T, D), got {tuple(h.shape)}")
    return [h[b].clone() for b in range(n)]


# -------------------------------------------------------------------------------------------- replay (GPU)
class ReplayHook:
    """Overwrite the consumed block's output with a stored tensor, before the steering hook reads it.

    Register AFTER the LocusHook (both with prepend=True) so this hook runs first; PyTorch feeds a hook's
    return value to the next hook, so LocusHook sees the replayed tensor and writes the direction on it.
    """

    def __init__(self, module: torch.nn.Module, tensors: dict[str, torch.Tensor], name: str = "replay"):
        self.module, self.tensors, self.name = module, tensors, name
        self.row: torch.Tensor | None = None
        self.row_id: str | None = None
        self.calls = 0
        self.drift: dict[str, dict] = {}
        self._handle = None

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook, prepend=True)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def set_row(self, row_id: str) -> None:
        if row_id not in self.tensors:
            raise KeyError(f"{self.name}: no stored tensor for row {row_id}")
        self.row_id, self.row = row_id, self.tensors[row_id]

    def _hook(self, _m, _i, output):
        h = as_hidden(output)
        if h is None:
            raise RuntimeError(f"{self.name}: module returned no hidden-state tensor")
        if self.row is None:
            raise RuntimeError(f"{self.name}: hook fired before set_row()")
        self.calls += 1
        t = self.row.to(device=h.device, dtype=h.dtype)
        if h.dim() == 2:                                   # flat (B*T, D)
            if h.shape[0] % t.shape[0] or h.shape[1] != t.shape[1]:
                raise RuntimeError(f"{self.name}: stored {tuple(t.shape)} does not tile {tuple(h.shape)}")
            new = t.repeat(h.shape[0] // t.shape[0], 1)
        elif h.dim() == 3:
            if h.shape[1:] != t.shape:
                raise RuntimeError(f"{self.name}: stored {tuple(t.shape)} != block {tuple(h.shape[1:])}")
            new = t.unsqueeze(0).expand(h.shape[0], *t.shape).contiguous()
        else:
            raise RuntimeError(f"{self.name}: cannot replay a {h.dim()}-d block output {tuple(h.shape)}")
        if self.row_id not in self.drift:                  # how far this reader's own tower output was
            d = (h.float() - new.float()).abs()
            self.drift[self.row_id] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                       "block_mean_abs": float(h.float().abs().mean())}
        return with_hidden(output, new)


class ReplayBank:
    """Stored consumed-block tensors of one (source block, dataset, locus), keyed by row id."""

    def __init__(self, source_key: str, dataset_id: str, locus_id: str, tensors: dict[str, torch.Tensor], meta: dict):
        self.source_key, self.dataset_id, self.locus_id, self.tensors, self.meta = source_key, dataset_id, locus_id, tensors, meta

    @classmethod
    def open(cls, model_key: str, dataset_id: str, locus_id: str = "vis.last", source_key: str | None = None) -> "ReplayBank":
        src = source_key or replay_source(model_key)
        path = replay_dir(src, dataset_id) / f"{locus_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"REPLAY needs the stored tensors at {path}; run "
                                    f"python -m cftransfer.replay --model-key {src} --dataset {dataset_id}")
        z = np.load(path, allow_pickle=False)
        dtype = str(z["stored_dtype"])
        ids = list(z["row_id"].astype(str))
        x = z["x"]
        tensors = {rid: unpack(x[i], dtype) for i, rid in enumerate(ids)}
        mp = replay_dir(src, dataset_id) / f"replay_meta.{locus_id}.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {"model_key": src, "rows": len(ids)}
        return cls(src, dataset_id, locus_id, tensors, meta)

    def hook(self, module: torch.nn.Module) -> ReplayHook:
        return ReplayHook(module, self.tensors, f"replay<{self.source_key}>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-key", required=True, help="the SOURCE block whose tower output is stored")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--locus", default=LOCI["primary"])
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    build_replay(a.model_key, a.dataset, a.locus, a.rows, a.batch_size, a.device_map,
                 out_dir=Path(a.out_dir) if a.out_dir else None)
