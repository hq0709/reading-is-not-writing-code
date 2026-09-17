"""Assemble microsoft/llava-rad into one runnable checkpoint.

The published repository holds no language-model weights: it is a LoRA adapter over lmsys/vicuna-7b-v1.5
(r=64, alpha=128, 448 tensors over q/k/v/o/gate/up/down of all 32 layers), the two `mm_projector` tensors in
`non_lora_trainables.bin`, and a BiomedCLIP-CXR-518 vision tower inside an open_clip training checkpoint
(`biomedclipcxr_518_checkpoint.pt`, keys `module.visual.trunk.*`, a timm `vit_base_patch14_dinov2` at 518 px).
`llava/model/builder.py` assembles those three at load time; this script does the same once, deterministically,
and writes a receipt.

Steps: verify every source file's SHA-256 against the Hub's LFS digests -> merge the LoRA into Vicuna in
float32 (W + (alpha/r) B A) -> map the projector and the timm trunk -> add `<image>` at 32000 with a zero row
in the embedding and the head (the LoRA trains neither, so the original 32,000 rows are Vicuna's and the
adapter scores that vocabulary) -> build cftransfer.adapters.llavarad.LlavaRadForConditionalGeneration and
load it strictly -> write bf16 shards, config.json, the Vicuna tokenizer with <image>, the open_clip
evaluation transform as a CLIPImageProcessor, and conversion.json.
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
from safetensors.torch import save_file

sys.path.insert(0, "/rodata/azradonc_dev/m253405/concept-flow/src")

VICUNA = Path("/rodata/azradonc_dev/m253405/cache/hub/models--lmsys--vicuna-7b-v1.5/snapshots")
LLAVARAD = Path("/rodata/azradonc_dev/m253405/cache/hub/models--microsoft--llava-rad/snapshots/dcdbc6caf6806c9acb66e12a827a39c86e00f6a7")
LLAVARAD_REVISION = "dcdbc6caf6806c9acb66e12a827a39c86e00f6a7"
OUT = Path("/rodata/azradonc_dev/m253405/cache/converted/llava-rad-hf")
IMAGE_TOKEN, IMAGE_TOKEN_ID, ORIGINAL_VOCAB = "<image>", 32000, 32000
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# open_clip's evaluation transform for a 518 px model with no preprocess_cfg in its model config: shortest-edge
# bicubic resize to 518, centre crop 518, to-tensor, normalise with the OpenAI CLIP statistics (open_clip
# transform.image_transform defaults mean/std to OPENAI_DATASET_MEAN/STD).
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
SHARD_BYTES = 5 * 1000 ** 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def hub_digests(repo: str, revision: str) -> dict:
    """SHA-256 the Hub records for each LFS object of a revision, so the local copies can be checked."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    return {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}


def write_shards(state: dict, out: Path) -> dict:
    """Write bf16 safetensors shards of at most SHARD_BYTES with the standard index, so the adapter's
    `load_sharded_checkpoint` can fill the model one shard at a time."""
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
    snaps = sorted(p for p in VICUNA.iterdir() if p.is_dir())
    if len(snaps) != 1:
        raise SystemExit(f"expected one staged vicuna snapshot, found {[p.name for p in snaps]}")
    vic, vic_rev = snaps[0], snaps[0].name
    receipt = {"sources": {"base_model": {"repo": "lmsys/vicuna-7b-v1.5", "revision": vic_rev},
                           "llava_rad": {"repo": "microsoft/llava-rad", "revision": LLAVARAD_REVISION}},
               "steps": {}}

    # (a) source identities against the Hub's LFS digests
    ids = {}
    for repo, rev, root, files in (("lmsys/vicuna-7b-v1.5", vic_rev, vic,
                                    sorted(p.name for p in vic.iterdir() if p.suffix in (".safetensors", ".bin", ".model"))),
                                   ("microsoft/llava-rad", LLAVARAD_REVISION, LLAVARAD,
                                    ["adapter_model.bin", "non_lora_trainables.bin", "biomedclipcxr_518_checkpoint.pt"])):
        digests = hub_digests(repo, rev)
        for name in files:
            got, want = sha256(root / name), digests.get(name)
            if want is not None and got != want:
                raise SystemExit(f"{repo}/{name}: sha256 {got} != Hub {want}")
            ids[f"{repo}/{name}"] = {"sha256": got, "hub_sha256": want, "bytes": os.path.getsize(root / name)}
    receipt["steps"]["sources_verified"] = ids
    print(f"[{time.time()-t0:.0f}s] {len(ids)} source files verified against the Hub LFS digests", flush=True)

    # (b) base weights (the Vicuna repository ships torch pickles, not safetensors)
    base = {}
    index = json.loads((vic / "pytorch_model.bin.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        base.update(torch.load(vic / shard, map_location="cpu", weights_only=True))
    receipt["steps"]["base_tensors"] = {"count": len(base), "dtypes": sorted({str(v.dtype) for v in base.values()})}

    # (c) merge the LoRA in float32
    ad_cfg = json.loads((LLAVARAD / "adapter_config.json").read_text())
    scale = ad_cfg["lora_alpha"] / ad_cfg["r"]
    assert set(ad_cfg["target_modules"]) == set(LORA_TARGETS) and ad_cfg["bias"] == "none"
    lora = torch.load(LLAVARAD / "adapter_model.bin", map_location="cpu", weights_only=True)
    lora = {k[len("base_model.model."):] if k.startswith("base_model.model.") else k: v for k, v in lora.items()}
    merged, deltas = 0, []
    for k in list(lora):
        if not k.endswith(".lora_A.weight"):
            continue
        stem = k[: -len(".lora_A.weight")]
        A, B = lora[k].float(), lora[stem + ".lora_B.weight"].float()
        w_key = stem + ".weight"
        if w_key not in base:
            raise SystemExit(f"LoRA target {w_key} is not in the base model")
        w = base[w_key].float()
        delta = scale * (B @ A)
        if delta.shape != w.shape:
            raise SystemExit(f"{w_key}: delta {tuple(delta.shape)} != weight {tuple(w.shape)}")
        new = w + delta
        deltas.append((delta.abs().max().item(), delta.norm().item() / max(w.norm().item(), 1e-9)))
        base[w_key] = new.to(torch.bfloat16)
        merged += 1
    if merged != len(LORA_TARGETS) * 32:
        raise SystemExit(f"merged {merged} LoRA targets, expected {len(LORA_TARGETS) * 32}")
    receipt["steps"]["lora_merge"] = {"targets_merged": merged, "scale_alpha_over_r": scale,
                                      "max_abs_delta": round(max(d for d, _ in deltas), 5),
                                      "max_relative_delta_norm": round(max(r for _, r in deltas), 5),
                                      "arithmetic": "W + (alpha/r) * B @ A in float32, stored bfloat16"}
    print(f"[{time.time()-t0:.0f}s] merged {merged} LoRA targets (scale {scale})", flush=True)

    # (d) assemble the target state dict
    # The 4.31-era Vicuna pickles store a DERIVED rotary buffer per layer (`self_attn.rotary_emb.inv_freq`),
    # which transformers 5.x computes in `LlamaRotaryEmbedding.__init__` and no longer keeps in the state dict.
    # They are dropped rather than mapped: they are a function of rope_theta and head_dim, not learned weights.
    dropped = sorted(k for k in base if k.endswith(".rotary_emb.inv_freq"))
    for k in dropped:
        base.pop(k)
    receipt["steps"]["derived_buffers_dropped"] = {
        "count": len(dropped), "example": dropped[:2],
        "reason": "derived rotary buffers stored by transformers 4.31; 5.x recomputes them from rope_theta"}
    state = {}
    for k, v in base.items():
        t = v.to(torch.bfloat16)
        if k == "lm_head.weight":
            state["lm_head.weight"] = t
        elif k.startswith("model."):
            state["model.language_model." + k[len("model."):]] = t
        else:
            raise SystemExit(f"unmapped base key {k}")
    proj = torch.load(LLAVARAD / "non_lora_trainables.bin", map_location="cpu", weights_only=True)
    proj = {k[len("base_model.model.model."):]: v for k, v in proj.items()}
    for src, dst in (("mm_projector.0.", "model.multi_modal_projector.linear_1."),
                     ("mm_projector.2.", "model.multi_modal_projector.linear_2.")):
        for suffix in ("weight", "bias"):
            state[dst + suffix] = proj[src + suffix].to(torch.bfloat16)
    tower_ck = torch.load(LLAVARAD / "biomedclipcxr_518_checkpoint.pt", map_location="cpu", weights_only=False)
    trunk = {k[len("module.visual.trunk."):]: v for k, v in tower_ck["state_dict"].items()
             if k.startswith("module.visual.trunk.")}
    for k, v in trunk.items():
        state["model.vision_tower.vit." + k] = v.to(torch.bfloat16)
    receipt["steps"]["tower"] = {"source": "biomedclipcxr_518_checkpoint.pt module.visual.trunk.*",
                                 "open_clip_run_name": tower_ck.get("name"), "epoch": tower_ck.get("epoch"),
                                 "tensors": len(trunk),
                                 "unused_from_that_checkpoint": "module.visual.head.proj (the CLIP projection, "
                                                                "which LLaVA-Rad's VisionTower never applies) and "
                                                                "the whole text encoder"}

    # (e) <image> at 32000: one zero row in the embedding and the head, exactly as the LLaVA-Med conversion
    emb = state["model.language_model.embed_tokens.weight"]
    head = state["lm_head.weight"]
    assert emb.shape[0] == head.shape[0] == ORIGINAL_VOCAB, (emb.shape, head.shape)
    state["model.language_model.embed_tokens.weight"] = torch.cat([emb, torch.zeros(1, emb.shape[1], dtype=emb.dtype)])
    state["lm_head.weight"] = torch.cat([head, torch.zeros(1, head.shape[1], dtype=head.dtype)])
    receipt["steps"]["image_token"] = {"image_token": IMAGE_TOKEN, "image_token_id": IMAGE_TOKEN_ID,
                                       "new_vocab": ORIGINAL_VOCAB + 1,
                                       "added_rows_init": "zeros (embedding row overwritten by image features; "
                                                          "head row excluded from scoring)"}

    # (f) config and a strict load into the real class
    src_cfg = json.loads((LLAVARAD / "config.json").read_text())
    tower_cfg = json.loads((LLAVARAD / "biomedclipcxr_518.json").read_text())
    assert src_cfg["mm_vision_select_layer"] == -2 and src_cfg["mm_vision_select_feature"] == "patch" \
        and src_cfg["mm_projector_type"] == "mlp2x_gelu" and src_cfg["image_aspect_ratio"] == "square" \
        and src_cfg["mm_hidden_size"] == 768 and src_cfg["mm_use_im_start_end"] is False \
        and src_cfg["mm_use_im_patch_token"] is False
    img_size = int(tower_cfg["vision_cfg"]["image_size"])
    timm_name = tower_cfg["vision_cfg"]["timm_model_name"].split(".")[0]
    raw = {
        "model_type": "llavarad",
        "timm_model_name": timm_name,
        "image_token_index": IMAGE_TOKEN_ID,
        "image_seq_length": (img_size // 14) ** 2,
        "projector_hidden_act": "gelu",
        "vision_feature_layer": src_cfg["mm_vision_select_layer"],
        "vision_feature_select_strategy": "default",
        "pad_token_id": src_cfg["pad_token_id"],
        "text_config": {"vocab_size": ORIGINAL_VOCAB + 1, "hidden_size": src_cfg["hidden_size"],
                        "intermediate_size": src_cfg["intermediate_size"],
                        "num_hidden_layers": src_cfg["num_hidden_layers"],
                        "num_attention_heads": src_cfg["num_attention_heads"],
                        "num_key_value_heads": src_cfg["num_key_value_heads"],
                        "max_position_embeddings": src_cfg["max_position_embeddings"],
                        "rms_norm_eps": src_cfg["rms_norm_eps"], "hidden_act": src_cfg["hidden_act"],
                        "tie_word_embeddings": False, "bos_token_id": src_cfg["bos_token_id"],
                        "eos_token_id": src_cfg["eos_token_id"], "pad_token_id": src_cfg["pad_token_id"],
                        "initializer_range": src_cfg["initializer_range"], "model_type": "llama"},
        "vision_config": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
                          "image_size": img_size, "patch_size": 14, "model_type": "dinov2",
                          "note": "the architecture of the timm trunk; the RUNNING tower is the timm module "
                                  f"{timm_name} at {img_size} px, not transformers' Dinov2Model"},
    }
    from cftransfer.adapters.llavarad import LlavaRadForConditionalGeneration, build_config
    from transformers.initialization import no_init_weights
    cfg = build_config(raw)
    with no_init_weights(), torch.device("cpu"):
        model = LlavaRadForConditionalGeneration(cfg)
    target = set(model.state_dict())
    if target != set(state):
        raise SystemExit(f"key mismatch: missing {sorted(target - set(state))[:5]} "
                         f"unexpected {sorted(set(state) - target)[:5]}")
    res = model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()}, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    receipt["steps"]["strict_load"] = {"tensors": len(state), "missing": 0, "unexpected": 0, "dtype": "bfloat16"}
    print(f"[{time.time()-t0:.0f}s] strict load {len(state)}/{len(state)}", flush=True)
    del model

    # (g) write the assembly
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "config.json").write_text(json.dumps(raw, indent=1))
    receipt["steps"]["written"] = write_shards({k: v.to(torch.bfloat16) for k, v in state.items()}, OUT)
    from transformers import AutoTokenizer, CLIPImageProcessor
    for f in ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        if (vic / f).exists():
            shutil.copy(vic / f, OUT / f)
    tok = AutoTokenizer.from_pretrained(vic, use_fast=False)
    assert len(tok) == ORIGINAL_VOCAB and tok.bos_token_id == 1, (len(tok), tok.bos_token_id)
    added = tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    assert added == 1 and tok.convert_tokens_to_ids(IMAGE_TOKEN) == IMAGE_TOKEN_ID
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token
    tok.save_pretrained(OUT)
    ip = CLIPImageProcessor(do_resize=True, size={"shortest_edge": img_size}, resample=3, do_center_crop=True,
                            crop_size={"height": img_size, "width": img_size}, do_rescale=True,
                            rescale_factor=1 / 255, do_normalize=True, image_mean=OPENAI_MEAN, image_std=OPENAI_STD,
                            do_convert_rgb=True)
    ip.save_pretrained(OUT)
    receipt["steps"]["image_processor"] = {
        "source": "open_clip.transform.image_transform for a 518 px model with no preprocess_cfg",
        "settings": {"shortest_edge": img_size, "center_crop": img_size, "resample": "bicubic",
                     "image_mean": OPENAI_MEAN, "image_std": OPENAI_STD}}
    receipt["output_dir"] = str(OUT)
    receipt["output_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                              for p in sorted(OUT.iterdir()) if p.is_file()}
    receipt["seconds"] = round(time.time() - t0, 1)
    receipt["transformers_version"] = __import__("transformers").__version__
    receipt["timm_version"] = __import__("timm").__version__
    (OUT / "conversion.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt["steps"], indent=1, default=str))
    print(f"ASSEMBLY_DONE in {receipt['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
