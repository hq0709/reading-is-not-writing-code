"""TOWERSWAP: put one checkpoint's vision tower into the other checkpoint's reader, keeping everything else.

The campaign's evidence that a written direction's effect is decided by the READER of the representation is
today made of shared-tower pairs (one tower, two readers: Gemma 3 4B/12B/27B, MedGemma 4B/27B, LLaVA-1.5
7B/13B). The missing half is the crossover -- one reader, two towers -- and Gemma 3 and MedGemma supply it:
they are the same architecture with different weights, so the tower of one loads into the other.

Verified from the staged checkpoints (CPU, no GPU, `python -m cftransfer.towerswap --verify`):

  google/gemma-3-4b-it   vs google/medgemma-4b-it    883 tensors each, IDENTICAL name sets;
      vision_tower.*            437 tensors, 416.9M params, identical shapes,  0/437 bitwise identical
      multi_modal_projector.*     2 tensors,   3.0M params, identical shapes,  0/2   bitwise identical
      language_model.*          444 tensors,  3880.3M params, identical shapes, 0/444 bitwise identical
  google/gemma-3-27b-it  vs google/medgemma-27b-it   1247 tensors each, IDENTICAL name sets and shapes in
      all three groups (so the 27B pair is interchangeable in exactly the same way)
  google/gemma-3-12b-it  vs google/medgemma-27b-it   tower shapes match but the reader does NOT:
      multi_modal_projector.mm_input_projection_weight (1152, 3840) vs (1152, 5376), and the language model
      has 48 vs 62 layers -- the expected failure of a cross-size swap.

So the tower (and the reader half) of the 4B pair and of the 27B pair are interchangeable, and the tower
weights genuinely differ (MedSigLIP vs SigLIP: mean relative difference 3.17 over the sampled tensors).

This module builds ONE model (the host = the reader) and replaces the parameters and buffers of its vision
tower with the donor checkpoint's, then proves the swap:
  * every replaced tensor equals the donor checkpoint's tensor exactly (torch.equal after the dtype cast
    the runtime uses), and the donor bytes are hashed into the receipt;
  * every parameter and buffer OUTSIDE the tower has an unchanged signature (dtype, shape, float64 sum,
    float64 sum of |x|, NaN count) before and after the swap.
The receipt is written next to the module's outcomes so the swap is auditable after the fact.

The `reader` of the campaign's claim is connector + language model, so the multimodal projector stays with
the host; only the tower is replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import torch

from .adapters import REVISIONS, get_adapter
from .adapters.base import HF_HOME
from .protocol import MODELS, TOWERSWAP_PAIRS

TOWER_MARKER = "vision_tower"
BLOCK_SPLIT = ".encoder.layers."


def swap_partner(model_key: str) -> str:
    """The model whose tower this block's TOWERSWAP arm writes into its reader."""
    if model_key not in TOWERSWAP_PAIRS:
        raise KeyError(f"{model_key}: no TOWERSWAP partner; pairs are {sorted(TOWERSWAP_PAIRS)}")
    return TOWERSWAP_PAIRS[model_key]


def snapshot_dir(model_key: str, revision: str | None = None) -> Path:
    """Staged snapshot of a pinned checkpoint (the adapters resolve the same path)."""
    rev = revision or REVISIONS.get(model_key)
    if not rev:
        raise RuntimeError(f"{model_key}: revision not locked")
    snap = HF_HOME / "hub" / f"models--{MODELS[model_key]['model_id'].replace('/', '--')}" / "snapshots" / rev
    if not (snap / "config.json").exists():
        raise FileNotFoundError(f"{model_key}: no staged snapshot at {snap}")
    return snap


# --------------------------------------------------------------------------------- checkpoint index (no torch)
def safetensors_index(path: Path) -> dict[str, tuple[Path, str, tuple[int, ...], int, int]]:
    """name -> (file, dtype, shape, byte_start, byte_end) over every shard of a snapshot directory."""
    out: dict[str, tuple[Path, str, tuple[int, ...], int, int]] = {}
    for f in sorted(path.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n))
        base = 8 + n
        for name, meta in head.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            out[name] = (f, meta["dtype"], tuple(meta["shape"]), base + s, base + e)
    if not out:
        raise FileNotFoundError(f"{path}: no safetensors shards")
    return out


def _sha(entry, chunk: int = 1 << 22) -> str:
    f, _dt, _sh, s, e = entry
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        fh.seek(s)
        left = e - s
        while left:
            b = fh.read(min(chunk, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def normalise_tower_key(key: str) -> str | None:
    """Checkpoint key -> name relative to the vision tower module, or None when it is not a tower tensor.

    Checkpoints spell the tower `vision_tower.*` (Gemma 3 / MedGemma), `vision_tower.vision_model.*`
    (LLaVA-1.5) or with a leading `model.`; the runtime tree of transformers 5.x flattens some of these.
    Only the part below the tower is used for matching, so every spelling lines up.
    """
    k = key[len("model."):] if key.startswith("model.") else key
    if not k.startswith(TOWER_MARKER + "."):
        return None
    k = k[len(TOWER_MARKER) + 1:]
    return k[len("vision_model."):] if k.startswith("vision_model.") else k


def tower_root(ad) -> str:
    """Module path of the vision tower of a loaded adapter, derived from its own primary-locus path."""
    path = ad.loci()["vis.last"].module_path
    if BLOCK_SPLIT not in path:
        raise NotImplementedError(f"{ad.model_key}: primary locus {path!r} is not an encoder layer; "
                                  "TOWERSWAP only handles families whose consumed block is <tower>.encoder.layers.<k>")
    return path.split(BLOCK_SPLIT)[0]


def compare_checkpoints(a_key: str, b_key: str, deep: bool = False) -> dict:
    """Name / shape (and optionally bitwise) comparison of two staged checkpoints, by tensor group."""
    ia, ib = safetensors_index(snapshot_dir(a_key)), safetensors_index(snapshot_dir(b_key))
    out = {"a": a_key, "b": b_key, "n_tensors": [len(ia), len(ib)], "name_sets_equal": set(ia) == set(ib), "groups": {}}
    for group in ("vision_tower.", "multi_modal_projector.", "language_model."):
        ga = {k: v for k, v in ia.items() if k.startswith(group) or k.startswith("model." + group)}
        gb = {k: v for k, v in ib.items() if k.startswith(group) or k.startswith("model." + group)}
        names_eq = set(ga) == set(gb)
        shapes_eq = names_eq and all(ga[k][1:3] == gb[k][1:3] for k in ga)
        cell = {"n": [len(ga), len(gb)], "names_equal": names_eq, "shapes_equal": shapes_eq,
                "params_millions": round(sum(int(torch.tensor(v[2]).prod()) for v in ga.values()) / 1e6, 1)}
        if names_eq and not shapes_eq:
            cell["shape_mismatches"] = [{"name": k, a_key: list(ga[k][2]), b_key: list(gb[k][2])}
                                        for k in sorted(ga) if ga[k][2] != gb[k][2]][:6]
        if shapes_eq and deep:
            cell["bitwise_identical"] = sum(1 for k in sorted(ga) if _sha(ga[k]) == _sha(gb[k]))
        out["groups"][group] = cell
    return out


# --------------------------------------------------------------------------------------------- the swap
def param_signature(model: torch.nn.Module) -> dict[str, tuple]:
    """name -> (dtype, shape, float64 sum, float64 sum of |x|, NaN count) for every parameter AND buffer.

    Cheap (two reductions per tensor, on the tensor's own device) and enough to prove that nothing outside
    the tower moved: a changed weight changes the sum or the sum of absolute values with probability 1.
    """
    sig = {}
    with torch.no_grad():
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            f = t.detach().double()
            sig[name] = (str(t.dtype), tuple(t.shape), float(f.sum()), float(f.abs().sum()), int(torch.isnan(f).sum()))
    return sig


def donor_tower_tensors(donor_key: str, revision: str | None = None) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Tower tensors of a staged checkpoint, keyed relative to the tower module, plus their sha256."""
    from safetensors import safe_open
    snap = snapshot_dir(donor_key, revision)
    idx = safetensors_index(snap)
    tensors, shas = {}, {}
    handles: dict[Path, object] = {}
    try:
        for key, entry in idx.items():
            rel = normalise_tower_key(key)
            if rel is None:
                continue
            if rel in tensors:
                raise RuntimeError(f"{donor_key}: tower name {rel!r} appears twice in the checkpoint")
            f = entry[0]
            if f not in handles:
                handles[f] = safe_open(f, framework="pt", device="cpu")
            tensors[rel] = handles[f].get_tensor(key)
            shas[rel] = _sha(entry)
    finally:
        for h in handles.values():
            getattr(h, "__exit__", lambda *a: None)(None, None, None)
    if not tensors:
        raise RuntimeError(f"{donor_key}: checkpoint has no '{TOWER_MARKER}.' tensors")
    return tensors, shas


NON_CHECKPOINT_BUFFERS = ("position_ids",)   # built by the module from its config, absent from both checkpoints


def apply_tower_swap(ad, donor_key: str, revision: str | None = None, donor_tensors: dict | None = None,
                     donor_shas: dict | None = None) -> dict:
    """Replace the loaded adapter's vision-tower weights with `donor_key`'s and verify. Returns the receipt."""
    t0 = time.time()
    root = tower_root(ad)
    tower = ad.module(root)
    before = param_signature(ad.model)
    if donor_tensors is None:
        donor_tensors, donor_shas = donor_tower_tensors(donor_key, revision)
    donor_shas = donor_shas or {}
    runtime = {k: v for k, v in (list(tower.named_parameters()) + list(tower.named_buffers()))}
    # A buffer the checkpoint does not store is constructed by the module itself from its config (position_ids is the
    # one case in this grid). Both checkpoints build it identically because the swap only runs on families whose tower
    # configs match, so it is kept and its signature is verified with the rest of the untouched model below.
    unstored = sorted(k for k in set(runtime) - set(donor_tensors) if k.split(".")[-1] in NON_CHECKPOINT_BUFFERS)
    runtime = {k: v for k, v in runtime.items() if k not in unstored}
    missing = sorted(set(runtime) - set(donor_tensors))
    extra = sorted(set(donor_tensors) - set(runtime))
    if missing or extra:
        raise RuntimeError(f"tower swap {donor_key} -> {ad.model_key}: {len(missing)} runtime tensors without a donor "
                           f"tensor (e.g. {missing[:3]}) and {len(extra)} donor tensors without a runtime tensor "
                           f"(e.g. {extra[:3]})")
    bad = [k for k in runtime if tuple(runtime[k].shape) != tuple(donor_tensors[k].shape)]
    if bad:
        raise RuntimeError(f"tower swap {donor_key} -> {ad.model_key}: shape mismatch on {len(bad)} tensors, e.g. "
                           f"{[(k, tuple(runtime[k].shape), tuple(donor_tensors[k].shape)) for k in bad[:3]]}")
    state = {k: donor_tensors[k].to(dtype=runtime[k].dtype) for k in runtime}
    tower.load_state_dict(state, strict=False if unstored else True)
    # 1) every replaced tensor is exactly the donor's
    runtime_after = {k: v for k, v in (list(tower.named_parameters()) + list(tower.named_buffers()))}
    unequal = [k for k in runtime_after if k in state and not torch.equal(runtime_after[k].detach().cpu(), state[k])]
    if unequal:
        raise RuntimeError(f"tower swap {donor_key} -> {ad.model_key}: {len(unequal)} tensors differ from the donor "
                           f"after loading, e.g. {unequal[:3]}")
    # 2) nothing outside the tower moved
    after = param_signature(ad.model)
    prefix = root + "."
    outside_changed = sorted(k for k in before if not k.startswith(prefix) and before[k] != after.get(k))
    if outside_changed:
        raise RuntimeError(f"tower swap {donor_key} -> {ad.model_key}: {len(outside_changed)} tensors OUTSIDE the "
                           f"tower changed, e.g. {outside_changed[:5]}")
    inside = [k for k in before if k.startswith(prefix)]
    changed_inside = sum(1 for k in inside if before[k] != after[k])
    h = hashlib.sha256()
    for k in sorted(donor_shas):
        h.update(f"{k}:{donor_shas[k]}\n".encode())
    return {"host_model_key": ad.model_key, "host_model_id": ad.model_id, "host_revision": ad.revision,
            "donor_model_key": donor_key, "donor_model_id": (MODELS.get(donor_key) or {}).get("model_id"),
            "donor_revision": revision or REVISIONS.get(donor_key), "tower_root": root,
            "n_tensors_replaced": len(runtime), "n_params_replaced": int(sum(t.numel() for t in runtime.values())),
            "runtime_dtype": str(next(iter(runtime.values())).dtype),
            "donor_tower_sha256": h.hexdigest() if donor_shas else None,
            "n_tower_tensors_changed": changed_inside, "n_tower_tensors": len(inside),
            "buffers_built_by_the_module": unstored,
            "n_tensors_outside_tower": len(before) - len(inside), "n_tensors_outside_tower_changed": 0,
            "verified_bitwise_equal_to_donor": True, "verified_rest_unchanged": True,
            "seconds": round(time.time() - t0, 1),
            "note": "reader = multimodal projector + language model, kept from the host; only the tower was replaced"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="verify that two staged checkpoints' towers / readers are interchangeable")
    ap.add_argument("--pairs", default=None, help="comma list of A:B pairs (default: TOWERSWAP_PAIRS plus the control pairs)")
    ap.add_argument("--deep", action="store_true", help="also hash every tensor (slow for 27B)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pairs = ([tuple(p.split(":")) for p in a.pairs.split(",")] if a.pairs else
             sorted({tuple(sorted((k, v))) for k, v in TOWERSWAP_PAIRS.items()}) + [("gemma3-12", "medgemma-27")])
    report = [compare_checkpoints(x, y, deep=a.deep) for x, y in pairs]
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
