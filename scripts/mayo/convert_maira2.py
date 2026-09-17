"""Assemble microsoft/maira-2 into the transformers 5.x module layout, in bfloat16.

Nothing is merged and no tensor is altered: the snapshot is already one complete checkpoint. What it is not
is loadable, because its remote `Maira2ForConditionalGeneration` builds `vision_tower`,
`multi_modal_projector` and `language_model` directly on itself (the transformers 4.51 layout) while the
`LlavaForConditionalGeneration` it inherits from now keeps those three inside `self.model` and reaches them
through it. This script renames the four prefixes onto the 5.x layout, casts the float32 shards to the
campaign's bfloat16, and writes the result with a receipt; the model class that reads it is
cftransfer.adapters.maira2.Maira2ForConditionalGeneration, which is the snapshot's own tower, projector and
feature selection expressed in that layout.

Steps: verify the six shard SHA-256 against the Hub's LFS digests -> remap 522 keys -> build the class and
load it strictly -> write bf16 shards, config.json, the snapshot's processor, tokenizer and chat template,
and conversion.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, "/rodata/azradonc_dev/m253405/concept-flow/src")

SRC = Path("/rodata/azradonc_dev/m253405/cache/hub/models--microsoft--maira-2/snapshots/795a2b1cd4a310624b4e3d14b5a23e41fd273deb")
SRC_REVISION = "795a2b1cd4a310624b4e3d14b5a23e41fd273deb"
OUT = Path("/rodata/azradonc_dev/m253405/cache/converted/maira-2-hf")
PREFIX_MAP = [                                   # (source prefix, target prefix); order matters
    ("language_model.model.", "model.language_model."),
    ("language_model.lm_head.", "lm_head."),
    ("vision_tower.", "model.vision_tower."),
    ("multi_modal_projector.", "model.multi_modal_projector."),
]
N_TENSORS = 522
SHARD_BYTES = 5 * 1000 ** 3
COPY_FILES = ("preprocessor_config.json", "processor_config.json", "processing_maira2.py", "chat_template.json",
              "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
              "added_tokens.json", "generation_config.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def remap(key: str) -> str:
    for src, dst in PREFIX_MAP:
        if key.startswith(src):
            return dst + key[len(src):]
    raise KeyError(f"unmapped source key {key}")


def write_shards(state: dict, out: Path) -> dict:
    names, cur, cur_bytes, shards = [], {}, 0, []
    for k, v in state.items():
        n = v.numel() * v.element_size()
        if cur and cur_bytes + n > SHARD_BYTES:
            shards.append(cur); cur, cur_bytes = {}, 0
        cur[k] = v; cur_bytes += n
    if cur:
        shards.append(cur)
    weight_map, total = {}, 0
    for i, sh in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file({k: v.contiguous() for k, v in sh.items()}, out / name, metadata={"format": "pt"})
        names.append(name)
        for k, v in sh.items():
            weight_map[k] = name
            total += v.numel() * v.element_size()
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    return {"shards": names, "tensors": len(weight_map), "total_bytes": total}


def main() -> None:
    t0 = time.time()
    receipt = {"source_repo": "microsoft/maira-2", "source_revision": SRC_REVISION, "steps": {}}

    # (a) shard identities against the Hub's LFS digests
    from huggingface_hub import HfApi
    digests = {s.rfilename: (s.lfs.sha256 if s.lfs else None)
               for s in HfApi().model_info("microsoft/maira-2", revision=SRC_REVISION, files_metadata=True).siblings}
    index = json.loads((SRC / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    shards = {}
    for name in shard_names:
        got, want = sha256(SRC / name), digests.get(name)
        if want is not None and got != want:
            raise SystemExit(f"{name}: sha256 {got} != Hub {want}")
        shards[name] = {"sha256": got, "hub_sha256": want, "bytes": os.path.getsize(SRC / name)}
    receipt["steps"]["shards_verified"] = shards
    print(f"[{time.time()-t0:.0f}s] {len(shards)} shard identities verified against the Hub LFS digests", flush=True)

    # (b) stream-remap to the transformers 5.x layout, in bfloat16
    state, src_dtypes = {}, set()
    for name in shard_names:
        sd = load_file(str(SRC / name))
        for k, v in sd.items():
            src_dtypes.add(str(v.dtype))
            nk = remap(k)
            if nk in state:
                raise SystemExit(f"collision on target key {nk}")
            state[nk] = v.to(torch.bfloat16)
        del sd
    if len(state) != N_TENSORS:
        raise SystemExit(f"{len(state)} remapped tensors, expected {N_TENSORS}")
    receipt["steps"]["remap"] = {"tensors": len(state), "source_dtypes": sorted(src_dtypes), "stored_dtype": "bfloat16",
                                 "prefix_map": [list(p) for p in PREFIX_MAP]}

    # (c) config for the 5.x classes, from the snapshot's own config.json
    src_cfg = json.loads((SRC / "config.json").read_text())
    assert src_cfg["architectures"] == ["Maira2ForConditionalGeneration"] and src_cfg["model_type"] == "maira2"
    vis = dict(src_cfg["vision_config"])
    vis.pop("architectures", None); vis.pop("torch_dtype", None)
    text = dict(src_cfg["text_config"])
    text.pop("architectures", None); text.pop("_name_or_path", None); text.pop("torch_dtype", None)
    raw = {"model_type": "maira2",
           "image_token_index": src_cfg["image_token_index"],
           "image_seq_length": src_cfg["image_seq_length"],
           "projector_hidden_act": src_cfg["projector_hidden_act"],
           "projector_n_layers": src_cfg["projector_n_layers"],
           "multimodal_projector_bias": src_cfg["multimodal_projector_bias"],
           "vision_feature_layer": src_cfg["vision_feature_layer"],
           "vision_feature_select_strategy": src_cfg["vision_feature_select_strategy"],
           "pad_token_id": src_cfg["pad_token_id"],
           "text_config": text, "vision_config": vis}

    from cftransfer.adapters.maira2 import Maira2ForConditionalGeneration, build_config
    from transformers.initialization import no_init_weights
    cfg = build_config(raw)
    with no_init_weights(), torch.device("cpu"):
        model = Maira2ForConditionalGeneration(cfg)
    target = set(model.state_dict())
    if target != set(state):
        raise SystemExit(f"key mismatch: missing {sorted(target - set(state))[:5]} "
                         f"unexpected {sorted(set(state) - target)[:5]}")
    res = model.load_state_dict(state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    receipt["steps"]["strict_load"] = {"tensors": len(state), "missing": 0, "unexpected": 0, "dtype": "bfloat16"}
    print(f"[{time.time()-t0:.0f}s] strict load {len(state)}/{len(state)}", flush=True)
    del model

    # (d) write the assembly
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "config.json").write_text(json.dumps(raw, indent=1))
    receipt["steps"]["written"] = write_shards(state, OUT)
    copied = []
    for f in COPY_FILES:
        if (SRC / f).exists():
            shutil.copy(SRC / f, OUT / f)
            copied.append(f)
    receipt["steps"]["copied_from_snapshot"] = copied
    receipt["output_dir"] = str(OUT)
    receipt["output_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                              for p in sorted(OUT.iterdir()) if p.is_file()}
    receipt["seconds"] = round(time.time() - t0, 1)
    receipt["transformers_version"] = __import__("transformers").__version__
    (OUT / "conversion.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt["steps"].items() if k != "shards_verified"}, indent=1, default=str))
    print(f"ASSEMBLY_DONE in {receipt['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
