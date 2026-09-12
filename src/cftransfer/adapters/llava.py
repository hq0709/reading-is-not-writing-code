"""LLaVA-1.5 family (llava-hf/llava-1.5-7b-hf, llava-1.5-13b-hf).

Consumed boundary: `LlavaModel.get_image_features` runs the CLIP vision tower with hidden-state capture
and selects `hidden_states[vision_feature_layer]` (-2 for LLaVA-1.5, i.e. the output of encoder layer
n-2, NOT the last block, whose output is computed and discarded), drops the CLS token
(`vision_feature_select_strategy="default"`) and feeds the remaining 576 patch tokens to
`multi_modal_projector`. The primary locus is therefore `vision_tower.encoder.layers.{n + vfl}` with a
batched (B, 577, 1024) output whose index 0 (CLS) is not consumed; the connector locus is the projector
output, batched (B, 576, D_text), all tokens consumed.

transformers 5.x collects `hidden_states` through a forward hook installed on each encoder layer the
first time capture is requested; a steering hook must run BEFORE that recorder (LocusHook registers
with prepend=True), otherwise the projector reads the unsteered tensor.
"""
from __future__ import annotations

import torch

from ..hooks import TokenLayout
from .base import Adapter, LocusInfo


class LlavaAdapter(Adapter):
    family = "llava1.5"
    default_processor_kwargs: dict = {}      # native: shortest edge 336, center crop 336, CLIP normalisation

    def after_load(self) -> None:
        cfg = self.model.config
        for prefix in ("model.vision_tower.encoder.layers", "model.vision_tower.vision_model.encoder.layers"):
            if prefix in self._modules:
                self._layers_prefix = prefix
                break
        else:
            raise KeyError(f"{self.model_key}: cannot locate the CLIP encoder layers")
        n = len(self.module(self._layers_prefix))
        vfl = cfg.vision_feature_layer
        if not isinstance(vfl, int):
            raise RuntimeError(f"{self.model_key}: multi-layer vision_feature_layer {vfl!r} is not a single consumed block")
        # hidden_states[0] is the embedding output, hidden_states[k] the output of layer k-1
        layer_idx = n + vfl if vfl < 0 else vfl - 1
        if not 0 <= layer_idx < n:
            raise RuntimeError(f"{self.model_key}: vision_feature_layer {vfl} does not select an encoder layer")
        self.n_layers, self.vfl, self.layer_idx = n, vfl, layer_idx
        self.strategy = cfg.vision_feature_select_strategy
        if self.strategy not in ("default", "full"):
            raise RuntimeError(f"{self.model_key}: unknown vision_feature_select_strategy {self.strategy!r}")
        self._block_path = f"{self._layers_prefix}.{layer_idx}"
        self._proj_path = "model.multi_modal_projector"
        self.module(self._block_path); self.module(self._proj_path)
        vc = cfg.vision_config
        self.grid = int(vc.image_size) // int(vc.patch_size)          # 336 / 14 = 24
        self.n_patches = self.grid * self.grid                         # 576
        self.n_consumed = self.n_patches + (1 if self.strategy == "full" else 0)

    def loci(self) -> dict[str, LocusInfo]:
        cfg = self.model.config
        return {
            "vis.last": LocusInfo(
                "vis.last", self._block_path,
                f"tensor output of CLIP encoder layer {self.layer_idx} of {self.n_layers} = hidden_states[{self.vfl}] "
                f"(the block the projector consumes; the final block {self.n_layers - 1} is computed and discarded)",
                int(cfg.vision_config.hidden_size),
                "batched (B, 1+576, D); index 0 is the CLS token and is NOT consumed (strategy 'default'); tokens 1..576 pooled"
                if self.strategy == "default" else "batched (B, 1+576, D); all tokens incl. CLS consumed (strategy 'full')",
                "the single image-token stream selected by vision_feature_layer", "none", self._proj_path, True,
                "transformers 5.x records hidden_states via forward hooks on each encoder layer; the steering hook is "
                "prepended so the recorder and the projector observe the steered output"),
            "connector": LocusInfo(
                "connector", self._proj_path, "tensor output of the two-layer MLP projector",
                int(cfg.text_config.hidden_size), "batched (B, 576, D_text); all tokens consumed",
                "projected image embeddings scattered into <image> placeholder positions", "none",
                "language model input embeddings", True),
        }

    def layouts(self, enc, images) -> dict[str, TokenLayout]:
        B = enc["pixel_values"].shape[0]
        if B != len(images):
            raise RuntimeError(f"{self.model_key}: {B} pixel tensors for {len(images)} images")
        per_row = (enc["input_ids"] == self.model.config.image_token_id).sum(dim=1).tolist()
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
        return self.model.model.get_image_features(pixel_values=batch["pixel_values"],
                                                   vision_feature_layer=cfg.vision_feature_layer,
                                                   vision_feature_select_strategy=cfg.vision_feature_select_strategy)
