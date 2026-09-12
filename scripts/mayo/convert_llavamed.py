"""Convert microsoft/llava-med-v1.5-mistral-7b (model_type llava_mistral, not loadable by transformers 5.x)
into a native LlavaForConditionalGeneration checkpoint, following docs/LLAVA_MED_RUNTIME_FEASIBILITY.md.

Steps: verify the four source shard SHA-256 values -> build the native config (Mistral-7B text, CLIP ViT-L/14-336
vision, mlp2x_gelu projector, vision_feature_layer=-2, strategy "default") -> remap all 686 keys with the doc's
prefix table -> strict load (686/686) -> add <image> at id 32000 and resize to 32,001 rows -> save safetensors,
config, CLIP-336 image processor, LlavaProcessor config and the pinned Mistral v0.2 tokenizer -> re-read every
saved tensor and compare with the source -> write conversion.json.
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

SRC = Path("/rodata/azradonc_dev/m253405/cache/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2")
SRC_REVISION = "91bb16c122001ddc9cf1fd36ce1dae09448943a2"
OUT = Path("/rodata/azradonc_dev/m253405/cache/converted/llava-med-v1.5-mistral-7b-hf")
PINNED_TOKENIZER_JSON = Path("/rodata/azradonc_dev/m253405/cache/converted/_pins/mistral-v0.2-tokenizer/tokenizer.json")
CLIP_REF = Path("/rodata/azradonc_dev/m253405/cache/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
EXPECTED_SHARDS = {
    "model-00001-of-00004.safetensors": "ef2190dc6c2a940e60f03f5fdb4dddb2320eb87801aeca5c40b0a28ce8aa420e",
    "model-00002-of-00004.safetensors": "2b229607fecd98b8111320178e5bf3e2c527b05a942c85d65b5b507c76c1ed00",
    "model-00003-of-00004.safetensors": "12b18ecdf8924d5fe28ada797fe6697fa60e62cba630759fbeb52975b261c4e2",
    "model-00004-of-00004.safetensors": "1d2063fcd429d3f0f0a8a091b0522f0e02f2d85fe0e5b0eeb4ae168183a603bc",
}
EXPECTED_TOKENIZER_JSON_SHA = "11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720"
EXPECTED_TOKENIZER_MODEL_SHA = "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055"
PREFIX_MAP = [                                                   # (source prefix, target prefix); order matters
    ("model.vision_tower.vision_tower.vision_model.", "model.vision_tower."),
    ("model.mm_projector.0.", "model.multi_modal_projector.linear_1."),
    ("model.mm_projector.2.", "model.multi_modal_projector.linear_2."),
    ("model.layers.", "model.language_model.layers."),
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.norm.", "model.language_model.norm."),
    ("lm_head.", "lm_head."),
]
IMAGE_TOKEN, IMAGE_TOKEN_ID, ORIGINAL_VOCAB = "<image>", 32000, 32000


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
    receipt = {"source_repo": "microsoft/llava-med-v1.5-mistral-7b", "source_revision": SRC_REVISION, "steps": {}}
    # (a) shard identities
    shards = {}
    for name, expected in EXPECTED_SHARDS.items():
        got = sha256(SRC / name)
        if got != expected:
            raise SystemExit(f"{name}: sha256 {got} != expected {expected}")
        shards[name] = {"sha256": got, "bytes": os.path.getsize(SRC / name)}
    receipt["steps"]["shards_verified"] = shards
    tm_sha = sha256(SRC / "tokenizer.model")
    if tm_sha != EXPECTED_TOKENIZER_MODEL_SHA:
        raise SystemExit(f"tokenizer.model sha256 {tm_sha} unexpected")
    tj_sha = sha256(PINNED_TOKENIZER_JSON)
    if tj_sha != EXPECTED_TOKENIZER_JSON_SHA:
        raise SystemExit(f"pinned tokenizer.json sha256 {tj_sha} unexpected")
    receipt["steps"]["tokenizer_files"] = {"tokenizer.model_sha256": tm_sha, "tokenizer.json_sha256": tj_sha,
                                           "tokenizer.json_source": "mistralai/Mistral-7B-Instruct-v0.2@63a8b081895390a26e140280378bc85ec8bce07a"}
    print(f"[{time.time()-t0:.0f}s] shard and tokenizer identities verified", flush=True)

    # (b) native config
    from transformers import CLIPImageProcessor, CLIPVisionConfig, LlavaConfig, LlavaForConditionalGeneration, LlavaProcessor, \
        MistralConfig, AutoTokenizer
    from transformers.initialization import no_init_weights
    src_cfg = json.loads((SRC / "config.json").read_text())
    assert src_cfg["mm_vision_select_layer"] == -2 and src_cfg["mm_vision_select_feature"] == "patch" \
        and src_cfg["mm_projector_type"] == "mlp2x_gelu" and src_cfg["mm_vision_tower"] == "openai/clip-vit-large-patch14-336"
    text = MistralConfig(vocab_size=src_cfg["vocab_size"], hidden_size=src_cfg["hidden_size"], intermediate_size=src_cfg["intermediate_size"],
                         num_hidden_layers=src_cfg["num_hidden_layers"], num_attention_heads=src_cfg["num_attention_heads"],
                         num_key_value_heads=src_cfg["num_key_value_heads"], max_position_embeddings=src_cfg["max_position_embeddings"],
                         rms_norm_eps=src_cfg["rms_norm_eps"], rope_theta=src_cfg["rope_theta"], sliding_window=src_cfg["sliding_window"],
                         hidden_act=src_cfg["hidden_act"], attention_dropout=src_cfg["attention_dropout"], tie_word_embeddings=False,
                         bos_token_id=src_cfg["bos_token_id"], eos_token_id=src_cfg["eos_token_id"], initializer_range=src_cfg["initializer_range"])
    clip_ref = json.loads((CLIP_REF / "config.json").read_text())["vision_config"]
    vision = CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16, image_size=336,
                              patch_size=14, projection_dim=768, hidden_act="quick_gelu", layer_norm_eps=1e-5)
    for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "image_size", "patch_size", "projection_dim"):
        assert getattr(vision, k) == clip_ref[k], k
    cfg = LlavaConfig(vision_config=vision, text_config=text, image_token_index=IMAGE_TOKEN_ID, projector_hidden_act="gelu",
                      vision_feature_select_strategy="default", vision_feature_layer=-2, multimodal_projector_bias=True,
                      pad_token_id=0, tie_word_embeddings=False)
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
    if len(state) != 686:
        raise SystemExit(f"{len(state)} remapped tensors, expected 686")
    with no_init_weights():
        model = LlavaForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
    target_keys = set(model.state_dict().keys())
    if target_keys != set(state):
        raise SystemExit(f"key mismatch: missing {sorted(target_keys - set(state))[:5]} unexpected {sorted(set(state) - target_keys)[:5]}")
    res = model.load_state_dict(state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    receipt["steps"]["strict_load"] = {"tensors": len(state), "missing": 0, "unexpected": 0, "dtype": "bfloat16"}
    print(f"[{time.time()-t0:.0f}s] strict load 686/686", flush=True)

    # expand: <image> at id 32000; both added rows zero-initialised (input row is always replaced by image features,
    # output row is excluded from answer logits by the adapter, which scores the original 32,000-token vocabulary)
    emb_before = model.get_input_embeddings().weight[:ORIGINAL_VOCAB].clone()
    head_before = model.lm_head.weight[:ORIGINAL_VOCAB].clone()
    model.resize_token_embeddings(ORIGINAL_VOCAB + 1, mean_resizing=False)
    with torch.no_grad():
        model.get_input_embeddings().weight[ORIGINAL_VOCAB].zero_()
        model.lm_head.weight[ORIGINAL_VOCAB].zero_()
    assert torch.equal(model.get_input_embeddings().weight[:ORIGINAL_VOCAB], emb_before)
    assert torch.equal(model.lm_head.weight[:ORIGINAL_VOCAB], head_before)
    model.config.text_config.vocab_size = ORIGINAL_VOCAB + 1
    model.config.image_token_index = IMAGE_TOKEN_ID
    receipt["steps"]["expand"] = {"image_token": IMAGE_TOKEN, "image_token_id": IMAGE_TOKEN_ID, "new_vocab": ORIGINAL_VOCAB + 1,
                                  "added_rows_init": "zeros (embedding row overwritten by image features; head row excluded from scoring)",
                                  "original_rows_preserved": True}

    # (d) save model, tokenizer, processor
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
    for f in ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(SRC / f, OUT / f)
    shutil.copy(PINNED_TOKENIZER_JSON, OUT / "tokenizer.json")
    tok = AutoTokenizer.from_pretrained(OUT)
    assert len(tok) == ORIGINAL_VOCAB and tok.bos_token_id == 1 and tok.eos_token_id == 2 and tok.pad_token_id == 0
    added = tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    assert added == 1 and tok.convert_tokens_to_ids(IMAGE_TOKEN) == IMAGE_TOKEN_ID
    tok.save_pretrained(OUT)
    image_processor = CLIPImageProcessor.from_pretrained(CLIP_REF)          # CLIP-336: resize 336, center crop 336, CLIP mean/std
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tok, patch_size=14, vision_feature_select_strategy="default",
                               num_additional_image_tokens=1, image_token=IMAGE_TOKEN)
    processor.save_pretrained(OUT)
    print(f"[{time.time()-t0:.0f}s] saved to {OUT}", flush=True)

    # re-read through the public loader (transformers 5 stores the legacy on-disk key layout and maps it back on
    # load, so the runtime state_dict is the faithful comparison) and compare every tensor with the source
    del model
    reloaded = LlavaForConditionalGeneration.from_pretrained(OUT, dtype=torch.bfloat16, device_map="cpu")
    saved = reloaded.state_dict()
    max_diff, checked = 0.0, 0
    for nk, t in state.items():
        s = saved[nk]
        if nk in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
            s = s[:ORIGINAL_VOCAB]
        if s.shape != t.shape:
            raise SystemExit(f"{nk}: saved shape {tuple(s.shape)} != source {tuple(t.shape)}")
        d = (s.float() - t.float()).abs().max().item()
        max_diff = max(max_diff, d); checked += 1
    assert checked == 686 and max_diff == 0.0, (checked, max_diff)
    assert saved["model.language_model.embed_tokens.weight"].shape[0] == ORIGINAL_VOCAB + 1
    assert reloaded.config.image_token_id == IMAGE_TOKEN_ID and reloaded.config.vision_feature_layer == -2
    receipt["steps"]["reload_check"] = {"tensors_compared": checked, "max_abs_diff_vs_source": max_diff,
                                        "embedding_rows": ORIGINAL_VOCAB + 1, "head_rows": int(saved["lm_head.weight"].shape[0]),
                                        "loader": "LlavaForConditionalGeneration.from_pretrained (runtime keys)",
                                        "added_rows_all_zero": bool((saved["model.language_model.embed_tokens.weight"][ORIGINAL_VOCAB] == 0).all()
                                                                     and (saved["lm_head.weight"][ORIGINAL_VOCAB] == 0).all())}
    receipt["output_dir"] = str(OUT)
    receipt["output_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(OUT.iterdir()) if p.is_file()}
    receipt["seconds"] = round(time.time() - t0, 1)
    receipt["transformers_version"] = __import__("transformers").__version__
    (OUT / "conversion.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt["steps"].items()}, indent=1))
    print(f"CONVERSION_DONE in {receipt['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
