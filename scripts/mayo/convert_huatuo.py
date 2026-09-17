"""Convert FreedomIntelligence/HuatuoGPT-Vision-7B (model_type llava_qwen2, a class that lives only in the
HuatuoGPT-Vision GitHub repository) into a native LlavaForConditionalGeneration checkpoint.

The checkpoint is LLaVA-1.5 with a Qwen2-7B language model ("trained based on Qwen2-7B using the LLaVA-v1.5
architecture", model card): a CLIP ViT-L/14-336 tower (391 tensors, stored INSIDE the four shards under
`model.vision_tower.vision_tower.vision_model.*`, so the repository's redundant `vit/` copy is not used for
weights), an `mlp2x_gelu` projector, `mm_vision_select_layer=-2`, `mm_vision_select_feature="patch"` and
`image_aspect_ratio="pad"`. Every one of those is the LLaVA-1.5 setting the campaign's LLaVA adapter already
implements, so the same remap that produced the native LLaVA-Med checkpoint applies key for key.

One thing differs from the LLaVA-Med conversion and it is what keeps this one numerically exact. LLaVA-Med
had to ADD a row for `<image>` (32000 into a 32,000-row head), so its adapter scores the original vocabulary
to keep log-partitions identical. Qwen2's head here has 152,064 rows while its tokenizer has 151,646, i.e.
418 trained-but-unreachable padded rows. `<image>` is therefore mapped onto the FIRST of those, id 151646:
the tokenizer gains a name for a row that already exists, no tensor is added, resized or reordered, and the
answer logits are the source model's own 152,064-row head. The embedding row is never read (LLaVA overwrites
those positions with image features) and the head row is a row the source model already had.

Steps: verify the four shard SHA-256 against the Hub's LFS digests -> build the native config (Qwen2-7B text,
CLIP ViT-L/14-336 vision, mlp2x_gelu projector, vision_feature_layer=-2, strategy "default") -> remap all 734
keys with the LLaVA-Med prefix table -> strict load -> name <image> at the first unused row -> save
safetensors, config, the checkpoint's OWN CLIP-336 image processor, LlavaProcessor config and the Qwen2
tokenizer -> re-read every saved tensor and compare with the source -> write conversion.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open

SRC = Path("/rodata/azradonc_dev/m253405/cache/hub/models--FreedomIntelligence--HuatuoGPT-Vision-7B/snapshots/34dfcdbb7728ff38da865839f342b88c4cf6ef39")
SRC_REVISION = "34dfcdbb7728ff38da865839f342b88c4cf6ef39"
OUT = Path("/rodata/azradonc_dev/m253405/cache/converted/HuatuoGPT-Vision-7B-hf")
CLIP_REF = Path("/rodata/azradonc_dev/m253405/cache/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
VIT_DIR = SRC / "vit" / "clip_vit_large_patch14_336"      # the checkpoint's own CLIP-336 preprocessor_config.json
# SHA-256 as the Hub records them for the LFS objects of revision 34dfcdbb (HfApi.model_info(files_metadata=True))
EXPECTED_SHARDS = {
    "model-00001-of-00004.safetensors": "da1f9b7837a6c54e7803a8a36dca76ba2b09f116b044225f1bcf04c6ab18db4d",
    "model-00002-of-00004.safetensors": "5ab6f7191d5d35b13f22bd238155b83d15a683a56d9a6b46d038ef6e4d3b897b",
    "model-00003-of-00004.safetensors": "cf5442ead64542c60c0ecc647fee679b7f18b531d47d169cac08d496e8b78be3",
    "model-00004-of-00004.safetensors": "9df68cee325562ea25cba56f5bac5027000fd288031fbeb1f66000a4c8dafaf4",
}
PREFIX_MAP = [                                                   # (source prefix, target prefix); order matters
    ("model.vision_tower.vision_tower.vision_model.", "model.vision_tower."),
    ("model.mm_projector.0.", "model.multi_modal_projector.linear_1."),
    ("model.mm_projector.2.", "model.multi_modal_projector.linear_2."),
    ("model.layers.", "model.language_model.layers."),
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.norm.", "model.language_model.norm."),
    ("lm_head.", "lm_head."),
]
IMAGE_TOKEN = "<image>"
TOKENIZER_VOCAB = 151646            # ids the Qwen2 tokenizer can produce: 0..151645
IMAGE_TOKEN_ID = 151646             # first row of the head that no token maps to
HEAD_ROWS = 152064                  # config.vocab_size of the source checkpoint
N_TENSORS = 734


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


def main() -> None:
    t0 = time.time()
    receipt = {"source_repo": "FreedomIntelligence/HuatuoGPT-Vision-7B", "source_revision": SRC_REVISION, "steps": {}}
    # (a) shard identities against the Hub's LFS digests
    shards = {}
    for name, expected in EXPECTED_SHARDS.items():
        got = sha256(SRC / name)
        if got != expected:
            raise SystemExit(f"{name}: sha256 {got} != expected {expected}")
        shards[name] = {"sha256": got, "bytes": os.path.getsize(SRC / name)}
    receipt["steps"]["shards_verified"] = shards
    print(f"[{time.time()-t0:.0f}s] four shard identities verified against the Hub LFS digests", flush=True)

    # (b) native config
    from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPVisionConfig, LlavaConfig,
                              LlavaForConditionalGeneration, LlavaProcessor, Qwen2Config)
    from transformers.initialization import no_init_weights
    src_cfg = json.loads((SRC / "config.json").read_text())
    assert src_cfg["architectures"] == ["LlavaQwen2ForCausalLM"] and src_cfg["model_type"] == "llava_qwen2"
    assert src_cfg["mm_vision_select_layer"] == -2 and src_cfg["mm_vision_select_feature"] == "patch" \
        and src_cfg["mm_projector_type"] == "mlp2x_gelu" and src_cfg["image_aspect_ratio"] == "pad" \
        and src_cfg["mm_hidden_size"] == 1024 and src_cfg["vocab_size"] == HEAD_ROWS \
        and src_cfg["tie_word_embeddings"] is False
    text = Qwen2Config(vocab_size=src_cfg["vocab_size"], hidden_size=src_cfg["hidden_size"],
                       intermediate_size=src_cfg["intermediate_size"], num_hidden_layers=src_cfg["num_hidden_layers"],
                       num_attention_heads=src_cfg["num_attention_heads"], num_key_value_heads=src_cfg["num_key_value_heads"],
                       max_position_embeddings=src_cfg["max_position_embeddings"], rms_norm_eps=src_cfg["rms_norm_eps"],
                       rope_theta=src_cfg["rope_theta"], sliding_window=src_cfg["sliding_window"],
                       use_sliding_window=src_cfg["use_sliding_window"], max_window_layers=src_cfg["max_window_layers"],
                       hidden_act=src_cfg["hidden_act"], attention_dropout=src_cfg["attention_dropout"],
                       tie_word_embeddings=False, bos_token_id=src_cfg["bos_token_id"], eos_token_id=src_cfg["eos_token_id"],
                       initializer_range=src_cfg["initializer_range"])
    clip_ref = json.loads((CLIP_REF / "config.json").read_text())["vision_config"]
    vit_ref = json.loads((VIT_DIR / "config.json").read_text())["vision_config"]
    vision = CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16,
                              image_size=336, patch_size=14, projection_dim=768, hidden_act="quick_gelu", layer_norm_eps=1e-5)
    for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "image_size", "patch_size",
              "projection_dim", "hidden_act", "layer_norm_eps"):
        # the checkpoint's own tower config carries every field; llava-1.5-7b-hf's stores only the ones that differ
        # from the CLIP defaults, so it is compared on the fields it has
        assert getattr(vision, k) == vit_ref[k], (k, getattr(vision, k), vit_ref[k])
        assert k not in clip_ref or clip_ref[k] == vit_ref[k], (k, clip_ref[k], vit_ref[k])
    cfg = LlavaConfig(vision_config=vision, text_config=text, image_token_index=IMAGE_TOKEN_ID, projector_hidden_act="gelu",
                      vision_feature_select_strategy="default", vision_feature_layer=-2, multimodal_projector_bias=True,
                      pad_token_id=src_cfg["eos_token_id"], tie_word_embeddings=False)
    cfg.architectures = ["LlavaForConditionalGeneration"]

    # (c) stream-remap and strict load
    state = {}
    for name in EXPECTED_SHARDS:
        with safe_open(SRC / name, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if t.dtype != torch.bfloat16:
                    raise SystemExit(f"{k}: dtype {t.dtype} is not bf16")
                nk = remap(k)
                if nk in state:
                    raise SystemExit(f"collision on target key {nk}")
                state[nk] = t
    if len(state) != N_TENSORS:
        raise SystemExit(f"{len(state)} remapped tensors, expected {N_TENSORS}")
    with no_init_weights():
        model = LlavaForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
    target_keys = set(model.state_dict().keys())
    if target_keys != set(state):
        raise SystemExit(f"key mismatch: missing {sorted(target_keys - set(state))[:5]} unexpected {sorted(set(state) - target_keys)[:5]}")
    res = model.load_state_dict(state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    receipt["steps"]["strict_load"] = {"tensors": len(state), "missing": 0, "unexpected": 0, "dtype": "bfloat16"}
    print(f"[{time.time()-t0:.0f}s] strict load {N_TENSORS}/{N_TENSORS}", flush=True)

    # (d) name <image> at the first row no token maps to: nothing is added, resized or reordered
    emb = model.get_input_embeddings().weight
    head = model.lm_head.weight
    assert emb.shape[0] == head.shape[0] == HEAD_ROWS, (emb.shape, head.shape)
    receipt["steps"]["image_token"] = {
        "image_token": IMAGE_TOKEN, "image_token_id": IMAGE_TOKEN_ID, "head_rows": HEAD_ROWS,
        "tokenizer_vocab": TOKENIZER_VOCAB, "rows_added": 0, "embeddings_resized": False,
        "note": "Qwen2's head carries 152,064 rows for a 151,646-token tokenizer; <image> names row 151646, which "
                "no token could produce, so the answer logits are the source model's own head",
        "unused_row_norms": {"embedding": float(emb[IMAGE_TOKEN_ID].float().norm()),
                             "lm_head": float(head[IMAGE_TOKEN_ID].float().norm())}}

    # (e) save model, tokenizer, processor
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
              "added_tokens.json", "generation_config.json"):
        shutil.copy(SRC / f, OUT / f)
    tok = AutoTokenizer.from_pretrained(OUT)
    assert len(tok) == TOKENIZER_VOCAB and tok.bos_token_id is None and tok.eos_token_id == 151643, (len(tok), tok.bos_token_id)
    added = tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    assert added == 1 and tok.convert_tokens_to_ids(IMAGE_TOKEN) == IMAGE_TOKEN_ID, (added, tok.convert_tokens_to_ids(IMAGE_TOKEN))
    assert len(tok) == TOKENIZER_VOCAB + 1 <= HEAD_ROWS
    tok.save_pretrained(OUT)
    image_processor = CLIPImageProcessor.from_pretrained(VIT_DIR)           # the checkpoint's OWN CLIP-336 settings
    ref_processor = CLIPImageProcessor.from_pretrained(CLIP_REF)
    a, b = image_processor.to_dict(), ref_processor.to_dict()
    differing = {k: (a.get(k), b.get(k)) for k in set(a) | set(b)
                 if k not in ("_processor_class", "image_processor_type", "feature_extractor_type") and a.get(k) != b.get(k)}
    receipt["steps"]["image_processor"] = {"source": str(VIT_DIR.relative_to(SRC)),
                                           "differs_from_llava15_reference": differing}
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tok, patch_size=14,
                               vision_feature_select_strategy="default", num_additional_image_tokens=1,
                               image_token=IMAGE_TOKEN)
    processor.save_pretrained(OUT)
    print(f"[{time.time()-t0:.0f}s] saved to {OUT}", flush=True)

    # (f) re-read through the public loader and compare every tensor with the source
    del model
    reloaded = LlavaForConditionalGeneration.from_pretrained(OUT, dtype=torch.bfloat16, device_map="cpu")
    saved = reloaded.state_dict()
    max_diff, checked = 0.0, 0
    for nk, t in state.items():
        s = saved[nk]
        if s.shape != t.shape:
            raise SystemExit(f"{nk}: saved shape {tuple(s.shape)} != source {tuple(t.shape)}")
        d = (s.float() - t.float()).abs().max().item()
        max_diff = max(max_diff, d); checked += 1
    assert checked == N_TENSORS and max_diff == 0.0, (checked, max_diff)
    assert saved["model.language_model.embed_tokens.weight"].shape[0] == HEAD_ROWS
    assert saved["lm_head.weight"].shape[0] == HEAD_ROWS
    assert reloaded.config.image_token_id == IMAGE_TOKEN_ID and reloaded.config.vision_feature_layer == -2
    receipt["steps"]["reload_check"] = {"tensors_compared": checked, "max_abs_diff_vs_source": max_diff,
                                        "embedding_rows": int(saved["model.language_model.embed_tokens.weight"].shape[0]),
                                        "head_rows": int(saved["lm_head.weight"].shape[0]),
                                        "loader": "LlavaForConditionalGeneration.from_pretrained (runtime keys)"}
    receipt["output_dir"] = str(OUT)
    receipt["output_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(OUT.iterdir()) if p.is_file()}
    receipt["seconds"] = round(time.time() - t0, 1)
    receipt["transformers_version"] = __import__("transformers").__version__
    (OUT / "conversion.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt["steps"], indent=1))
    print(f"CONVERSION_DONE in {receipt['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
