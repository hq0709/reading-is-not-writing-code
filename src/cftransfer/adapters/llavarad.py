"""LLaVA-Rad (microsoft/llava-rad) via the assembly produced by scripts/mayo/convert_llavarad.py.

The published repository is NOT a checkpoint. It carries three parts and no language-model weights at all:
a LoRA adapter over `lmsys/vicuna-7b-v1.5` (`adapter_model.bin`, r=64, alpha=128, 448 tensors over q/k/v/o/
gate/up/down of all 32 layers), the two projector tensors (`non_lora_trainables.bin`), and a BiomedCLIP-CXR
vision tower in an open_clip training checkpoint (`biomedclipcxr_518_checkpoint.pt`, `module.visual.trunk.*`).
The conversion merges the LoRA into the base model, attaches the projector and the tower, and writes one
assembled checkpoint with a receipt; this adapter builds the model and fills it from that assembly.

Once assembled it is LLaVA-1.5 with a different tower, so the campaign's LLaVA loci are unchanged in kind:

  tower       the vision tower is `vit_base_patch14_dinov2` (DINOv2 ViT-B/14) at 518 px, trained on chest
              radiographs under the BiomedCLIP framework. 518/14 = 37, so one image is 1369 patch tokens plus
              a CLS token, at width 768 -- 2.4x LLaVA-1.5's 576 tokens, and the reason this block is the most
              expensive 7B member of the grid per row.
  consumption `OpenCLIPVisionTower.feature_select` takes `hidden_states[-2]` (the output of block 10 of 12;
              block 11 is computed and discarded) and drops index 0 (CLS) because `mm_vision_select_feature`
              is "patch". The tower's final `norm` is NEVER applied on the consumed path: LLaVA-Rad's
              `VisionTower.forward` runs patch_embed -> _pos_embed -> norm_pre -> blocks and collects the
              block outputs, and stops there. This adapter runs that same forward, so the primary locus is the
              output of `model.vision_tower.vit.blocks.10`, (B, 1370, 768), with index 0 not consumed.
  connector   `mm_projector` is `mlp2x_gelu`, which is exactly transformers' `LlavaMultiModalProjector`
              (Linear 768->4096, GELU, Linear 4096->4096), so the connector locus is its output,
              (B, 1369, 4096), all tokens consumed.
  prompt      the checkpoint's own evaluation recipe (scripts/eval.sh: conv_mode "v1"): the Vicuna v1
              conversation with `<image>\\n` before the question, tokenised by the official chunked
              `tokenizer_image_token`, with `image_aspect_ratio="square"` -- so, unlike LLaVA-Med and
              HuatuoGPT-Vision, non-square images are NOT padded to a square, they are resized on the shortest
              edge and centre-cropped.
  vocab       the LoRA leaves the embedding and the head at Vicuna's 32,000 rows, and the checkpoint has no
              image token (`mm_use_im_patch_token` false; the original code uses the -200 sentinel). The
              conversion therefore adds `<image>` at 32000 exactly as the LLaVA-Med conversion did, and the
              answer logits are scored over the original 32,000-token vocabulary so log-partitions match the
              source model.

The model class is built here rather than loaded with `from_pretrained` because the tower is a timm module:
`LlavaModel.__init__` would call `AutoModel.from_config(vision_config)` and build a transformers Dinov2, whose
weights are named differently and whose forward applies a final LayerNorm the consumed path does not. The
class below replaces only that one attribute; `get_image_features`, the placeholder scatter and the whole
language path are transformers' own, unmodified, and they reproduce LLaVA-Rad's feature selection exactly
(`hidden_states[vision_feature_layer]`, CLS dropped for strategy "default", then the projector).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.llava.modeling_llava import (LlavaForConditionalGeneration, LlavaModel,
                                                      LlavaMultiModalProjector, LlavaPreTrainedModel)

from ..hooks import TokenLayout
from .base import Adapter, LocusInfo
from .llava import LlavaAdapter

CONVERTED = Path("/rodata/azradonc_dev/m253405/cache/converted/llava-rad-hf")
IMAGE_TOKEN, IMAGE_TOKEN_ID, ORIGINAL_VOCAB = "<image>", 32000, 32000
TIMM_MODEL = "vit_base_patch14_dinov2"          # the trunk of BiomedCLIP-CXR-518 (biomedclipcxr_518.json)
# Vicuna v1 conversation (llava/conversation.py conv_vicuna_v1), which scripts/eval.sh selects with conv_mode "v1"
SYSTEM_PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.")


class TimmVisionTower(nn.Module):
    """LLaVA-Rad's `VisionTower`: the timm trunk, run block by block with every block output collected.

    Verbatim from llava/model/multimodal_encoder/open_clip_encoder/open_clip_encoder.py -- patch_embed,
    _pos_embed, norm_pre, then the blocks, with NO final `norm`. The return type carries `pooler_output` so
    transformers' `LlavaModel.get_image_features` can set its projected features on it as it does for every
    other family."""

    def __init__(self, model_name: str = TIMM_MODEL, img_size: int = 518):
        super().__init__()
        import timm
        self.vit = timm.create_model(model_name, pretrained=False, img_size=img_size, num_classes=0)
        self.model_name, self.img_size = model_name, img_size
        self.hidden_size = int(self.vit.embed_dim)
        self.patch_size = int(self.vit.patch_embed.patch_size[0])
        self.num_patches = int(self.vit.patch_embed.num_patches)

    def forward(self, pixel_values, output_hidden_states: bool = True, return_dict: bool = True, **kwargs):
        # the processor emits float32 pixels and timm's patch embedding does not cast, unlike every transformers
        # vision tower (CLIPVisionEmbeddings / Dinov2PatchEmbeddings cast to the projection's dtype); done here
        h = self.vit.patch_embed(pixel_values.to(dtype=self.vit.patch_embed.proj.weight.dtype))
        h = self.vit._pos_embed(h)
        h = self.vit.norm_pre(h)
        states = [h]
        for block in self.vit.blocks:
            h = block(h)
            states.append(h)
        return BaseModelOutputWithPooling(last_hidden_state=h, pooler_output=None, hidden_states=tuple(states))


class LlavaRadModel(LlavaModel):
    """`LlavaModel` with the timm tower in place of `AutoModel.from_config(vision_config)`; nothing else changes."""
    _no_split_modules = ["TimmVisionTower"]

    def __init__(self, config):
        LlavaPreTrainedModel.__init__(self, config)
        self.vision_tower = TimmVisionTower(getattr(config, "timm_model_name", TIMM_MODEL),
                                            int(config.vision_config.image_size))
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        from transformers import AutoModel
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()


class LlavaRadForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config):
        LlavaPreTrainedModel.__init__(self, config)
        self.model = LlavaRadModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


def build_config(raw: dict):
    """LlavaConfig for the assembly, from the JSON the conversion writes."""
    from transformers import Dinov2Config, LlamaConfig, LlavaConfig
    text = LlamaConfig(**raw["text_config"])
    vision = Dinov2Config(**raw["vision_config"])
    cfg = LlavaConfig(vision_config=vision, text_config=text, image_token_index=raw["image_token_index"],
                      projector_hidden_act=raw["projector_hidden_act"],
                      vision_feature_select_strategy=raw["vision_feature_select_strategy"],
                      vision_feature_layer=raw["vision_feature_layer"], multimodal_projector_bias=True,
                      pad_token_id=raw["pad_token_id"], tie_word_embeddings=False)
    cfg.timm_model_name = raw["timm_model_name"]
    cfg.image_seq_length = raw["image_seq_length"]
    cfg.architectures = ["LlavaRadForConditionalGeneration"]
    return cfg


class LlavaRadAdapter(LlavaAdapter):
    family = "llava-rad"

    def resolve_local_path(self) -> Path:
        if not (CONVERTED / "config.json").exists() or not (CONVERTED / "conversion.json").exists():
            raise FileNotFoundError(f"{self.model_key}: assembled checkpoint with receipt not found at {CONVERTED}")
        self.local_path = CONVERTED
        return CONVERTED

    # ------------------------------------------------------------------ loading
    def load(self, device_map="cuda:0", dtype=torch.bfloat16):
        """Construct the class and fill it from the assembly, because the tower is a timm module that
        `from_pretrained` cannot rebuild from a config (see the module docstring)."""
        from transformers import AutoTokenizer, CLIPImageProcessor, LlavaProcessor
        from .chexagent import default_dtype, load_sharded_checkpoint

        path = self.resolve_local_path()
        raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
        cfg = build_config(raw)
        cfg._attn_implementation = self.attn_implementation
        from transformers.initialization import no_init_weights
        # every parameter is overwritten by the strict load below, so the random initialisation of a 7B model
        # is skipped; buffers (rotary inv_freq) are still computed by __init__ and checked for finiteness
        with no_init_weights(), torch.device("cpu"), default_dtype(dtype):
            self.model = LlavaRadForConditionalGeneration(cfg)
        info = load_sharded_checkpoint(self.model, path)
        if info["missing_keys"] or info["unexpected_keys"]:
            raise RuntimeError(f"{self.model_key}: assembly does not match the model -- "
                               f"{len(info['missing_keys'])} missing (e.g. {info['missing_keys'][:3]}), "
                               f"{len(info['unexpected_keys'])} unexpected (e.g. {info['unexpected_keys'][:3]})")
        self._checkpoint_load = info
        tok = AutoTokenizer.from_pretrained(path)
        image_processor = CLIPImageProcessor.from_pretrained(path)
        self.processor = LlavaProcessor(image_processor=image_processor, tokenizer=tok,
                                        patch_size=int(cfg.vision_config.patch_size),
                                        vision_feature_select_strategy=cfg.vision_feature_select_strategy,
                                        num_additional_image_tokens=1, image_token=IMAGE_TOKEN)
        tok.padding_side = "left"
        self.device_map_requested = device_map
        self.device = "cuda:0" if device_map in ("auto", None) else device_map
        self.model.to(device=self.device, dtype=dtype)
        dtypes = {str(p.dtype) for p in self.model.parameters()}
        if dtypes != {str(dtype)}:
            raise RuntimeError(f"{self.model_key}: mixed parameter dtypes after the cast: {sorted(dtypes)}")
        bad = [n for n, b in self.model.named_buffers() if b.is_floating_point() and not torch.isfinite(b).all()]
        if bad:
            raise RuntimeError(f"{self.model_key}: {len(bad)} non-finite buffers after loading, e.g. {bad[:3]}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._modules = dict(self.model.named_modules())
        from ..scoring import candidate_sets
        self._cands = candidate_sets(tok)
        self._install_fp32_head()
        self.after_load()
        return self

    # ---- float32 head restricted to the original vocabulary (the added row is not part of the source model)
    def _install_fp32_head(self) -> None:
        Adapter._install_fp32_head(self)
        self._head_w32 = self._head_w32[:ORIGINAL_VOCAB]
        if self._head_b32 is not None:
            self._head_b32 = self._head_b32[:ORIGINAL_VOCAB]

    def after_load(self) -> None:
        cfg = self.model.config
        tower = self.model.model.vision_tower
        self.n_layers = len(tower.vit.blocks)
        vfl = cfg.vision_feature_layer
        if not isinstance(vfl, int):
            raise RuntimeError(f"{self.model_key}: multi-layer vision_feature_layer {vfl!r} is not a single consumed block")
        self.vfl = vfl
        self.layer_idx = self.n_layers + vfl if vfl < 0 else vfl - 1     # hidden_states[0] is the pre-block tensor
        if not 0 <= self.layer_idx < self.n_layers:
            raise RuntimeError(f"{self.model_key}: vision_feature_layer {vfl} does not select a block")
        self.strategy = cfg.vision_feature_select_strategy
        if self.strategy not in ("default", "full"):
            raise RuntimeError(f"{self.model_key}: unknown vision_feature_select_strategy {self.strategy!r}")
        self._layers_prefix = "model.vision_tower.vit.blocks"
        self._block_path = f"{self._layers_prefix}.{self.layer_idx}"
        self._proj_path = "model.multi_modal_projector"
        self.module(self._block_path); self.module(self._proj_path)
        self.grid = tower.img_size // tower.patch_size
        self.n_patches = self.grid * self.grid
        if self.n_patches != tower.num_patches:
            raise RuntimeError(f"{self.model_key}: {self.n_patches} != timm num_patches {tower.num_patches}")
        self.n_consumed = self.n_patches + (1 if self.strategy == "full" else 0)
        tok = self.tokenizer
        if tok.convert_tokens_to_ids(IMAGE_TOKEN) != IMAGE_TOKEN_ID or tok.bos_token_id != 1:
            raise RuntimeError(f"{self.model_key}: tokenizer does not carry <image>=32000 / BOS=1")

    def loci(self) -> dict[str, LocusInfo]:
        cfg = self.model.config
        tower = self.model.model.vision_tower
        return {
            "vis.last": LocusInfo(
                "vis.last", self._block_path,
                f"tensor output of timm block {self.layer_idx} of {self.n_layers} = hidden_states[{self.vfl}] "
                f"(the block the projector consumes; block {self.n_layers - 1} is computed and discarded), "
                f"(B, {self.n_patches + 1}, {tower.hidden_size})",
                tower.hidden_size,
                f"batched (B, 1+{self.n_patches}, D); index 0 is the CLS token and is NOT consumed "
                f"(mm_vision_select_feature 'patch'); tokens 1..{self.n_patches} pooled"
                if self.strategy == "default" else f"batched (B, 1+{self.n_patches}, D); all tokens incl. CLS consumed",
                "the single image-token stream selected by mm_vision_select_layer",
                "the trunk's final `norm` is never reached: LLaVA-Rad's VisionTower stops after the blocks",
                self._proj_path, True,
                f"tower {TIMM_MODEL} (DINOv2 ViT-B/14) at {tower.img_size} px, trained on chest radiographs as "
                f"BiomedCLIP-CXR-518"),
            "connector": LocusInfo(
                "connector", self._proj_path, "tensor output of the two-layer MLP projector (mlp2x_gelu)",
                int(cfg.text_config.hidden_size), f"batched (B, {self.n_consumed}, D_text); all tokens consumed",
                "projected image embeddings scattered into <image> placeholder positions", "none",
                "language model input embeddings", True),
        }

    # ---- prompt and tokens
    def prompt_text(self, question: str, with_image: bool = True) -> str:
        """Vicuna v1 conversation, SeparatorStyle.TWO, one user turn and an open assistant turn."""
        body = f"{IMAGE_TOKEN}\n{question}" if with_image else question
        return f"{SYSTEM_PROMPT} USER: {body} ASSISTANT:"

    def tokenize_image_prompt(self, prompt: str) -> list[int]:
        """Official `tokenizer_image_token` with the -200 sentinel replaced by the tower's consumed-token count."""
        tok = self.tokenizer
        chunks = [tok(c).input_ids for c in prompt.split(IMAGE_TOKEN)]
        ids, offset = [], 0
        if chunks and chunks[0] and chunks[0][0] == tok.bos_token_id:
            offset = 1
            ids.append(chunks[0][0])
        sep = [-200] * (offset + 1)
        seq = [x for pair in zip(chunks, [sep] * len(chunks)) for x in pair][:-1]
        for x in seq:
            ids.extend(x[offset:])
        out = []
        for t in ids:
            out.extend([IMAGE_TOKEN_ID] * self.n_consumed if t == -200 else [t])
        return out

    def encode(self, images, questions):
        """`image_aspect_ratio` is "square" for this checkpoint, so the image is resized on the shortest edge and
        centre-cropped -- the open_clip evaluation transform, reproduced by the saved CLIPImageProcessor."""
        tok = self.tokenizer
        rows = [self.tokenize_image_prompt(self.prompt_text(q, with_image=images is not None)) for q in questions]
        L = max(len(r) for r in rows)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        input_ids = torch.full((len(rows), L), pad, dtype=torch.long)
        attention = torch.zeros((len(rows), L), dtype=torch.long)
        for i, r in enumerate(rows):                       # left padding: answer position is always -1
            input_ids[i, L - len(r):] = torch.tensor(r)
            attention[i, L - len(r):] = 1
        enc = {"input_ids": input_ids, "attention_mask": attention}
        if images is not None:
            enc["pixel_values"] = self.processor.image_processor(images, return_tensors="pt")["pixel_values"]
        return enc

    def image_token_mask(self, enc) -> torch.Tensor:
        return enc["input_ids"] == IMAGE_TOKEN_ID

    def layouts(self, enc, images) -> dict[str, TokenLayout]:
        B = enc["pixel_values"].shape[0] if "pixel_values" in enc else len(images)
        if B != len(images):
            raise RuntimeError(f"{self.model_key}: {B} pixel tensors for {len(images)} images")
        per_row = (enc["input_ids"] == IMAGE_TOKEN_ID).sum(dim=1).tolist()
        if any(c != self.n_consumed for c in per_row):
            raise RuntimeError(f"{self.model_key}: image placeholder counts {per_row} != consumed tokens {self.n_consumed}")
        vis_mask = torch.ones(self.n_patches + 1, dtype=torch.bool)
        if self.strategy == "default":
            vis_mask[0] = False
        con_mask = torch.ones(self.n_consumed, dtype=torch.bool)
        return {"vis.last": TokenLayout(False, masks=[vis_mask.clone() for _ in range(B)]),
                "connector": TokenLayout(False, masks=[con_mask.clone() for _ in range(B)])}

    def processing_settings(self) -> dict:
        s = super().processing_settings()
        s["prompt_construction"] = (f"the checkpoint's own evaluation recipe (scripts/eval.sh conv_mode 'v1'): the "
                                    f"Vicuna v1 conversation '{{system}} USER: <image>\\n{{q}} ASSISTANT:'; chunked "
                                    f"tokenizer_image_token with {self.n_consumed} x <image>({IMAGE_TOKEN_ID}); "
                                    f"image_aspect_ratio 'square' (shortest-edge resize + centre crop, no padding); "
                                    f"answer logits over the original {ORIGINAL_VOCAB}-token vocabulary")
        s["vision_tower"] = {"timm_model": TIMM_MODEL, "image_size": self.model.model.vision_tower.img_size,
                             "patch_size": self.model.model.vision_tower.patch_size,
                             "patch_tokens": self.n_patches, "hidden_size": self.model.model.vision_tower.hidden_size,
                             "forward": "LLaVA-Rad VisionTower: patch_embed -> _pos_embed -> norm_pre -> blocks, "
                                        "block outputs collected, trunk `norm` not applied"}
        s["conversion_receipt"] = str(CONVERTED / "conversion.json")
        s["checkpoint_load"] = getattr(self, "_checkpoint_load", None)
        s["device_map"] = {"requested": getattr(self, "device_map_requested", None),
                           "device": getattr(self, "device", None),
                           "parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters()})}
        return s
