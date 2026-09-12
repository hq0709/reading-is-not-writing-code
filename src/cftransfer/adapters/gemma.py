"""Gemma 3 IT (4B/12B/27B) and MedGemma IT (4B/27B): SigLIP-400M tower at 896px -> post_layernorm ->
multi-modal projector (RMSNorm, 4x4 average pool to 256 soft tokens, linear) -> <image_soft_token> slots.

Primary locus: output of the last SigLIP encoder layer (B, 4096, 1152); every patch token is consumed (no
CLS). `post_layernorm` belongs to the tower head and sits between this block and the projector. Connector
locus: projector output (B, 256, D_text). Image strategy: native processor (896x896, no pan-and-scan).
"""
from __future__ import annotations

import torch

from ..hooks import TokenLayout
from .base import Adapter, LocusInfo


class Gemma3Adapter(Adapter):
    family = "gemma3"
    default_processor_kwargs = {}

    def after_load(self) -> None:
        # transformers 5.x flattens SiglipVisionModel (no `.vision_model` child); accept both spellings
        prefix = "model.vision_tower" if "model.vision_tower.encoder.layers.0" in self._modules else "model.vision_tower.vision_model"
        vm = self.module(prefix)
        self.n_layers = len(vm.encoder.layers)
        self._block_path = f"{prefix}.encoder.layers.{self.n_layers - 1}"
        self._proj_path = "model.multi_modal_projector"
        vc = self.model.config.vision_config
        self.patches = (int(vc.image_size) // int(vc.patch_size)) ** 2
        self.soft_tokens = int(self.model.config.mm_tokens_per_image)
        cfg = self.model.config
        self.image_token = getattr(cfg, "image_token_id", None) or getattr(cfg, "image_token_index", None)
        if self.image_token is None:
            raise RuntimeError("gemma3 config exposes neither image_token_id nor image_token_index")

    def encode(self, images, questions):
        """Gemma3Processor expects one image list per text (nested), not a flat list."""
        texts = [self.prompt_text(q, with_image=images is not None) for q in questions]
        kwargs = dict(text=texts, return_tensors="pt", padding=True)
        if images is not None:
            kwargs["images"] = [[im] for im in images]
        return self.processor(**kwargs)

    def image_token_mask(self, enc) -> torch.Tensor:
        return enc["input_ids"] == self.image_token

    def loci(self) -> dict[str, LocusInfo]:
        cfg = self.model.config
        return {
            "vis.last": LocusInfo("vis.last", self._block_path, "tensor output of the last SigLIP encoder layer",
                                  int(cfg.vision_config.hidden_size), f"all {self.patches} patch tokens (no CLS token)",
                                  "single visual stream", "none (post_layernorm follows, then projector)", self._proj_path),
            "connector": LocusInfo("connector", self._proj_path, "tensor output of Gemma3MultiModalProjector",
                                   int(cfg.text_config.hidden_size), f"all {self.soft_tokens} pooled soft tokens",
                                   "image soft tokens scattered into <image_soft_token> positions", "none",
                                   "language model input embeddings"),
        }

    def layouts(self, enc, images) -> dict[str, TokenLayout]:
        B = enc["pixel_values"].shape[0]
        if B != len(images):
            raise RuntimeError(f"{self.model_key}: {B} pixel tensors for {len(images)} images (pan-and-scan must be off)")
        n_tok = self.image_token_mask(enc).sum(dim=1).tolist()
        if any(n != self.soft_tokens for n in n_tok):
            raise RuntimeError(f"{self.model_key}: image token counts {n_tok} != {self.soft_tokens}")
        return {"vis.last": TokenLayout(False, masks=[torch.ones(self.patches, dtype=torch.bool) for _ in range(B)]),
                "connector": TokenLayout(False, masks=[torch.ones(self.soft_tokens, dtype=torch.bool) for _ in range(B)])}

    @torch.no_grad()
    def vision_features(self, enc):
        batch = self.to_device(enc)
        return self.model.model.get_image_features(pixel_values=batch["pixel_values"])
