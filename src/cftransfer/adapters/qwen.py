"""Qwen2.5-VL family (Qwen2.5-VL 3B/7B/32B/72B, Lingshu 7B/32B) and Qwen3-VL (4B/8B/32B).

Consumed boundary: the vision tower permutes patches into window order, runs `blocks`, then feeds the
LAST block's output straight into `merger` (LayerNorm + MLP, 2x2 spatial merge). The primary locus
is `blocks[-1]` output, a flat (sum_patches, D) tensor with each image occupying one contiguous slice
(window permutation stays inside an image). The connector locus is `merger` output, flat
(sum_patches/4, D_llm), also contiguous per image. Qwen3-VL additionally injects DeepStack features
from earlier blocks into early LLM layers; those bypasses are untouched by either locus.
"""
from __future__ import annotations

import torch

from ..hooks import TokenLayout
from .base import Adapter, LocusInfo


class QwenVLAdapter(Adapter):
    family = "qwen2.5-vl"
    default_processor_kwargs = {"min_pixels": 336 * 336, "max_pixels": 336 * 336}   # bridge configuration

    def after_load(self) -> None:
        vis = self.module("model.visual")
        self.n_blocks = len(vis.blocks)
        self.merge_unit = int(vis.spatial_merge_size) ** 2
        self._block_path = f"model.visual.blocks.{self.n_blocks - 1}"
        self._merger_path = "model.visual.merger"

    def loci(self) -> dict[str, LocusInfo]:
        cfg = self.model.config
        return {
            "vis.last": LocusInfo("vis.last", self._block_path, "tensor output of the last vision block (window order)",
                                  int(cfg.vision_config.hidden_size),
                                  "all patches of the image (flat contiguous slice); no CLS token exists",
                                  "main visual stream into merger",
                                  "Qwen3-VL DeepStack mergers read earlier blocks and are not modified" if self.family == "qwen3-vl" else "none",
                                  self._merger_path),
            "connector": LocusInfo("connector", self._merger_path, "tensor output of the patch merger",
                                   int(getattr(cfg.vision_config, "out_hidden_size", cfg.text_config.hidden_size)),
                                   "all merged tokens of the image (flat contiguous slice)",
                                   "merged image embeddings scattered into <|image_pad|> positions", "none",
                                   "language model input embeddings"),
        }

    def _counts(self, enc) -> list[int]:
        thw = enc["image_grid_thw"]
        return [int(t * h * w) for t, h, w in thw.tolist()]

    def layouts(self, enc, images) -> dict[str, TokenLayout]:
        counts = self._counts(enc)
        if len(counts) != len(images):
            raise RuntimeError(f"{self.model_key}: {len(counts)} grid entries for {len(images)} images")
        vis, con, s, sm = [], [], 0, 0
        for c in counts:
            vis.append((s, s + c)); s += c
            con.append((sm, sm + c // self.merge_unit)); sm += c // self.merge_unit
        return {"vis.last": TokenLayout(True, slices=vis), "connector": TokenLayout(True, slices=con)}

    @torch.no_grad()
    def vision_features(self, enc):
        batch = self.to_device(enc)
        return self.model.model.get_image_features(pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"])


class Qwen3VLAdapter(QwenVLAdapter):
    family = "qwen3-vl"
    # official processor; cap the longest edge so a 1024px radiograph does not expand to 1,024 image tokens.
    # 512x512 -> 32x32 patches of 16px -> 256 merged tokens. Recorded in processing_settings.
    default_processor_kwargs = {"max_pixels": 512 * 512}
