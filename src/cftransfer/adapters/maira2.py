"""MAIRA-2 (microsoft/maira-2) via the assembly produced by scripts/mayo/convert_maira2.py.

The snapshot is LLaVA-shaped and carries its own remote code, but that code cannot run under the campaign's
transformers. `Maira2ForConditionalGeneration` subclasses `LlavaForConditionalGeneration` and builds
`self.vision_tower`, `self.multi_modal_projector` and `self.language_model` directly on itself, which was the
4.51 layout. In transformers 5.x those three live on a `LlavaModel` at `self.model`, and the inherited
`forward` calls `self.model(...)` and `self.model.get_image_features(...)`. A model built by the snapshot's
`__init__` therefore has no `self.model` at all, and its `get_image_features` override -- the one that reads
the backbone's `feature_maps` instead of `hidden_states` -- is never reached, which is exactly the silent
failure its own `check_min_version("4.46.0.dev0")` was written to catch one refactor earlier. The classes
below are that same model expressed in the 5.x layout: the tower, the projector and the feature selection are
the snapshot's, everything else is transformers' own.

Architecture:

  tower       `Dinov2Backbone` (DINOv2 ViT-B/14, 12 layers, width 768) at 518 px with `apply_layernorm=True`
              and `out_features=["stage12"]`, so `feature_maps[-1]` is `layernorm(hidden_states[12])`, the
              output of block 11 with the backbone's final LayerNorm applied. 518/14 = 37, so one image is
              1369 patch tokens plus a CLS token.
  selection   `get_image_features` takes `feature_maps[vision_feature_layer]` (vision_feature_layer = -1, the
              only stage) and drops index 0 for strategy "default", which is why the primary locus is the
              output of `model.vision_tower.encoder.layer.11`, (B, 1370, 768), with index 0 not consumed.
  connector   a FOUR-layer projector (`projector_n_layers` 4: Linear 768->4096, GELU, Linear 4096->4096,
              GELU, Linear, GELU, Linear), deeper than every other block of the grid, whose output is
              (B, 1369, 4096), all tokens consumed and scattered into the `<image>` (32204) positions.
  language    a Vicuna-7B-shaped Llama with a 32,207-token vocabulary (the grounding tokens `<obj>`, `<box>`
              and the 200 coordinate bins are already in it, so nothing is added by the assembly) and linear
              rope scaling by 1.5.
  images      the checkpoint's own `Maira2Processor._normalize_image` converts the image to greyscale and
              min-max rescales it to [0, 255] before the BitImageProcessor's 518 px shortest-edge resize,
              centre crop and normalisation at mean/std 0.5307/0.2583. That per-image rescale is part of the
              checkpoint's pipeline, not of the campaign's, and it is applied here for every row.
  prompt      the snapshot's own chat template, through the processor: "You are an expert radiology assistant
              tasked with interpreting a chest X-ray study.  USER:  <image>{question}  ASSISTANT: ".
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.llava.modeling_llava import (LlavaForConditionalGeneration, LlavaModel,
                                                      LlavaPreTrainedModel)

from ..hooks import TokenLayout
from .base import Adapter, LocusInfo
from .llava import LlavaAdapter

CONVERTED = Path("/rodata/azradonc_dev/m253405/cache/converted/maira-2-hf")
IMAGE_TOKEN, IMAGE_TOKEN_ID = "<image>", 32204


class Maira2MultiModalProjector(nn.Module):
    """The snapshot's `Maira2MultiModalProjector`, verbatim: `projector_n_layers` linear layers with the
    projector activation between them, as `nn.Sequential` so the parameter names match the checkpoint."""

    def __init__(self, config):
        super().__init__()
        n_layers = config.projector_n_layers
        if n_layers < 1:
            raise ValueError(f"Number of layers should be at least 1, got {n_layers=}")
        text_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size
        layers = [nn.Linear(vision_hidden_size, text_hidden_size, bias=True)]
        for _ in range(n_layers - 1):
            layers.append(ACT2FN[config.projector_hidden_act])
            layers.append(nn.Linear(text_hidden_size, text_hidden_size, bias=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.layers(image_features)


class Maira2Model(LlavaModel):
    """`LlavaModel` with the snapshot's backbone tower, 4-layer projector and feature-map selection."""

    def __init__(self, config):
        LlavaPreTrainedModel.__init__(self, config)
        from transformers import AutoBackbone, AutoModel
        self.vision_tower = AutoBackbone.from_config(config.vision_config)
        self.multi_modal_projector = Maira2MultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    def get_image_features(self, pixel_values, vision_feature_layer=None, vision_feature_select_strategy=None,
                           **kwargs):
        """The snapshot's `get_image_features`: the backbone's `feature_maps` (hidden states with the
        backbone's LayerNorm applied), not `hidden_states`, which is what the default LLaVA path would take.

        The return object carries `pooler_output` as the per-image feature list, which is the contract
        `LlavaModel.forward` consumes in transformers 5.x."""
        cfg = self.config
        vfl = cfg.vision_feature_layer if vision_feature_layer is None else vision_feature_layer
        strategy = (cfg.vision_feature_select_strategy if vision_feature_select_strategy is None
                    else vision_feature_select_strategy)
        if isinstance(vfl, list):
            raise ValueError("MAIRA-2 does not support list values for vision_feature_layer.")
        if strategy not in ("default", "full"):
            raise ValueError(f"Unexpected select feature strategy: {strategy}")
        extra = {k: v for k, v in kwargs.items() if v is not None and k not in ("return_dict", "output_hidden_states")}
        if extra:
            raise ValueError(f"MAIRA-2 does not support passing extra kwargs to the vision tower, received: {extra}")
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected = image_outputs.feature_maps[vfl]
        if strategy == "default":
            selected = selected[:, 1:]
        image_features = self.multi_modal_projector(selected)
        return BaseModelOutputWithPooling(last_hidden_state=selected, pooler_output=list(image_features),
                                          hidden_states=image_outputs.hidden_states)


class Maira2ForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config):
        LlavaPreTrainedModel.__init__(self, config)
        self.model = Maira2Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


def build_config(raw: dict):
    """LlavaConfig for the assembly, from the snapshot's own config.json fields."""
    from transformers import Dinov2Config, LlamaConfig, LlavaConfig
    text = LlamaConfig(**raw["text_config"])
    vision = Dinov2Config(**raw["vision_config"])
    cfg = LlavaConfig(vision_config=vision, text_config=text, image_token_index=raw["image_token_index"],
                      projector_hidden_act=raw["projector_hidden_act"],
                      vision_feature_select_strategy=raw["vision_feature_select_strategy"],
                      vision_feature_layer=raw["vision_feature_layer"],
                      multimodal_projector_bias=raw["multimodal_projector_bias"],
                      pad_token_id=raw["pad_token_id"], tie_word_embeddings=False)
    cfg.projector_n_layers = raw["projector_n_layers"]
    cfg.image_seq_length = raw["image_seq_length"]
    cfg.architectures = ["Maira2ForConditionalGeneration"]
    return cfg


class Maira2Adapter(LlavaAdapter):
    family = "maira-2"

    def resolve_local_path(self) -> Path:
        if not (CONVERTED / "config.json").exists() or not (CONVERTED / "conversion.json").exists():
            raise FileNotFoundError(f"{self.model_key}: assembled checkpoint with receipt not found at {CONVERTED}")
        self.local_path = CONVERTED
        return CONVERTED

    def load(self, device_map="cuda:0", dtype=torch.bfloat16):
        from transformers import AutoProcessor
        from .chexagent import default_dtype, load_sharded_checkpoint

        path = self.resolve_local_path()
        raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
        cfg = build_config(raw)
        cfg._attn_implementation = self.attn_implementation
        from transformers.initialization import no_init_weights
        # every parameter is overwritten by the strict load below, so the random initialisation of a 7B model
        # is skipped; buffers (rotary inv_freq) are still computed by __init__ and checked for finiteness
        with no_init_weights(), torch.device("cpu"), default_dtype(dtype):
            self.model = Maira2ForConditionalGeneration(cfg)
        info = load_sharded_checkpoint(self.model, path)
        if info["missing_keys"] or info["unexpected_keys"]:
            raise RuntimeError(f"{self.model_key}: assembly does not match the model -- "
                               f"{len(info['missing_keys'])} missing (e.g. {info['missing_keys'][:3]}), "
                               f"{len(info['unexpected_keys'])} unexpected (e.g. {info['unexpected_keys'][:3]})")
        self._checkpoint_load = info
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, **self.processor_kwargs)
        tok = self.tokenizer
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

    def after_load(self) -> None:
        cfg = self.model.config
        tower = self.model.model.vision_tower
        self.n_layers = len(tower.encoder.layer)
        vfl = cfg.vision_feature_layer
        if not isinstance(vfl, int):
            raise RuntimeError(f"{self.model_key}: multi-layer vision_feature_layer {vfl!r} is not a single consumed block")
        stages = list(cfg.vision_config.out_features or [])
        if len(stages) != 1 or stages[0] != f"stage{self.n_layers}":
            raise RuntimeError(f"{self.model_key}: backbone out_features {stages} is not the single last stage")
        if vfl != -1:
            raise RuntimeError(f"{self.model_key}: vision_feature_layer {vfl} indexes a one-entry feature_maps tuple")
        self.vfl, self.layer_idx = vfl, self.n_layers - 1
        self.strategy = cfg.vision_feature_select_strategy
        if self.strategy not in ("default", "full"):
            raise RuntimeError(f"{self.model_key}: unknown vision_feature_select_strategy {self.strategy!r}")
        self._layers_prefix = "model.vision_tower.encoder.layer"
        self._block_path = f"{self._layers_prefix}.{self.layer_idx}"
        self._proj_path = "model.multi_modal_projector"
        self.module(self._block_path); self.module(self._proj_path)
        vc = cfg.vision_config
        self.grid = int(vc.image_size) // int(vc.patch_size)
        self.n_patches = self.grid * self.grid
        self.n_consumed = self.n_patches + (1 if self.strategy == "full" else 0)
        tok = self.tokenizer
        if tok.convert_tokens_to_ids(IMAGE_TOKEN) != IMAGE_TOKEN_ID:
            raise RuntimeError(f"{self.model_key}: tokenizer does not carry <image>={IMAGE_TOKEN_ID}")
        if int(cfg.image_token_id) != IMAGE_TOKEN_ID:
            raise RuntimeError(f"{self.model_key}: config image token {cfg.image_token_id} != {IMAGE_TOKEN_ID}")
        self._normalize_image = type(self.processor)._normalize_image

    def loci(self) -> dict[str, LocusInfo]:
        cfg = self.model.config
        return {
            "vis.last": LocusInfo(
                "vis.last", self._block_path,
                f"tensor output of Dinov2 block {self.layer_idx} of {self.n_layers}, (B, {self.n_patches + 1}, "
                f"{cfg.vision_config.hidden_size}); the backbone's LayerNorm is applied to it and the result is "
                f"feature_maps['stage{self.n_layers}'], which the projector consumes",
                int(cfg.vision_config.hidden_size),
                f"batched (B, 1+{self.n_patches}, D); index 0 is the CLS token and is NOT consumed (strategy "
                f"'default'); tokens 1..{self.n_patches} pooled" if self.strategy == "default"
                else f"batched (B, 1+{self.n_patches}, D); all tokens incl. CLS consumed",
                "the single image-token stream the backbone emits as its one output stage",
                "none: the backbone's final LayerNorm is ON the consumed path (apply_layernorm=True), so a write "
                "here reaches the projector through it",
                self._proj_path, True,
                f"tower DINOv2 ViT-B/14 at {cfg.vision_config.image_size} px, fine-tuned on chest radiographs by "
                f"the MAIRA-2 authors"),
            "connector": LocusInfo(
                "connector", self._proj_path,
                f"tensor output of the {cfg.projector_n_layers}-layer MLP projector (the deepest connector of the "
                f"grid), (B, {self.n_consumed}, {cfg.text_config.hidden_size})",
                int(cfg.text_config.hidden_size), f"batched (B, {self.n_consumed}, D_text); all tokens consumed",
                f"projected image embeddings scattered into <image>({IMAGE_TOKEN_ID}) placeholder positions",
                "none", "language model input embeddings", True),
        }

    def encode(self, images, questions):
        """The checkpoint's own processor, with its own per-image greyscale min-max rescale applied first."""
        texts = [self.prompt_text(q, with_image=images is not None) for q in questions]
        kwargs = dict(text=texts, return_tensors="pt", padding=True)
        if images is not None:
            kwargs["images"] = [self._normalize_image(im) for im in images]
        return self.processor(**kwargs)

    def image_token_mask(self, enc) -> torch.Tensor:
        return enc["input_ids"] == IMAGE_TOKEN_ID

    def layouts(self, enc, images) -> dict[str, TokenLayout]:
        B = enc["pixel_values"].shape[0]
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

    @torch.no_grad()
    def vision_features(self, enc):
        cfg = self.model.config
        batch = self.to_device(enc)
        return self.model.model.get_image_features(
            pixel_values=batch["pixel_values"], vision_feature_layer=cfg.vision_feature_layer,
            vision_feature_select_strategy=cfg.vision_feature_select_strategy)

    def processing_settings(self) -> dict:
        s = super().processing_settings()
        s["prompt_construction"] = ("the snapshot's own chat template through its processor: 'You are an expert "
                                    "radiology assistant tasked with interpreting a chest X-ray study.  USER:  "
                                    "<image>{q}  ASSISTANT: '")
        s["image_processing"] = {
            "pre_processor_step": "Maira2Processor._normalize_image: convert to greyscale, min-max rescale to "
                                  "[0, 255] per image, back to L; the checkpoint's own step, applied before the "
                                  "image processor",
            "image_processor": {k: v for k, v in self.processor.image_processor.to_dict().items()
                                if isinstance(v, (int, float, str, bool, list, dict, type(None)))},
            "patch_tokens": self.n_patches, "consumed_tokens": self.n_consumed}
        s["conversion_receipt"] = str(CONVERTED / "conversion.json")
        s["checkpoint_load"] = getattr(self, "_checkpoint_load", None)
        s["device_map"] = {"requested": getattr(self, "device_map_requested", None),
                           "device": getattr(self, "device", None),
                           "parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters()})}
        return s
