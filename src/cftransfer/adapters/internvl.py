"""InternVL3.5-HF family (8B/14B/38B): InternViT-300M/6B tower -> pixel shuffle + MLP projector -> Qwen3 LLM.

Consumed boundary (transformers 5.15 `InternVLModel.get_image_features`): the vision tower runs
`encoder.layer[0..n-1]`, then `vision_tower.layernorm` (an `nn.Identity` when `use_mean_pooling=True`,
which is the shipped configuration), then the CLS row is dropped (`vision_feature_select_strategy="default"`
-> `[:, 1:]`), the 32x32 patch grid is pixel-shuffled by `downsample_ratio` 0.5 to 16x16 and the
`multi_modal_projector` (LayerNorm + MLP) emits the 256 tokens scattered into <IMG_CONTEXT> positions.

Primary locus: `model.vision_tower.encoder.layer.{n-1}` output, batched (B, 1025, D); CLS (index 0) is not
consumed and is excluded from the mask. Connector locus: `model.multi_modal_projector` output (B, 256, D_llm).

Image strategy for the whole family: a single 448x448 tile with no dynamic cropping. The processor's
call-time defaults set `crop_to_patches=True` (up to 12 tiles + thumbnail), overriding the shipped
preprocessor config, so `crop_to_patches=False` is passed explicitly on every call.
"""
from __future__ import annotations

import torch

from ..hooks import TokenLayout
from .base import Adapter, LocusInfo


class InternVLAdapter(Adapter):
    family = "internvl3.5"
    default_processor_kwargs = {}
    IMAGE_KWARGS = {"crop_to_patches": False}

    def resolve_local_path(self):
        """Pattern-filtered snapshots (no README/examples) make `snapshot_download(local_files_only=True)` raise
        IncompleteSnapshotError; resolve the pinned snapshot directory directly and require config.json."""
        from .base import HF_HOME
        p = HF_HOME / "hub" / f"models--{self.model_id.replace('/', '--')}" / "snapshots" / self.revision
        if not (p / "config.json").exists():
            raise FileNotFoundError(f"{self.model_key}: snapshot {p} missing config.json")
        self.local_path = p
        return p

    def after_load(self) -> None:
        cfg = self.model.config
        if int(cfg.vision_feature_layer) != -1 or cfg.vision_feature_select_strategy != "default":
            raise RuntimeError(f"{self.model_key}: unexpected consumption rule "
                               f"vision_feature_layer={cfg.vision_feature_layer} strategy={cfg.vision_feature_select_strategy}")
        tower = self.module("model.vision_tower")
        self.n_layers = len(tower.encoder.layer)
        self._block_path = f"model.vision_tower.encoder.layer.{self.n_layers - 1}"
        self._proj_path = "model.multi_modal_projector"
        self.module(self._block_path); self.module(self._proj_path)
        self.tower_norm = type(tower.layernorm).__name__
        vc = cfg.vision_config
        img = vc.image_size if isinstance(vc.image_size, (list, tuple)) else (vc.image_size, vc.image_size)
        pat = vc.patch_size if isinstance(vc.patch_size, (list, tuple)) else (vc.patch_size, vc.patch_size)
        self.patches_per_tile = (img[0] // pat[0]) * (img[1] // pat[1])                    # 1024
        self.tokens_per_tile = int(self.patches_per_tile * float(cfg.downsample_ratio) ** 2)  # 256
        if self.tokens_per_tile != int(cfg.image_seq_length):
            raise RuntimeError(f"{self.model_key}: image_seq_length {cfg.image_seq_length} != {self.tokens_per_tile}")
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None:
            ip.crop_to_patches = False

    def encode(self, images, questions):
        texts = [self.prompt_text(q, with_image=images is not None) for q in questions]
        kwargs = dict(text=texts, return_tensors="pt", padding=True)
        if images is not None:
            kwargs["images"] = images
            kwargs.update(self.IMAGE_KWARGS)
        return self.processor(**kwargs)

    def loci(self) -> dict[str, LocusInfo]:
        cfg = self.model.config
        return {
            "vis.last": LocusInfo("vis.last", self._block_path,
                                  "tensor output of the last InternViT encoder layer (B, 1+patches, D)",
                                  int(cfg.vision_config.hidden_size),
                                  "all patch tokens of the single 448x448 tile; CLS at index 0 is dropped by the consumer and excluded",
                                  "single tile visual stream into pixel-shuffle + projector", "none",
                                  self._proj_path,
                                  notes=f"vision_tower.layernorm is {self.tower_norm} between this block and the consumer; "
                                        f"crop_to_patches=False (no dynamic tiling, no thumbnail)"),
            "connector": LocusInfo("connector", self._proj_path, "tensor output of the multimodal projector (B, 256, D_llm)",
                                   int(cfg.text_config.hidden_size), "all 256 projected tokens of the image",
                                   "image embeddings scattered into <IMG_CONTEXT> positions", "none",
                                   "language model input embeddings"),
        }

    def layouts(self, enc, images) -> dict[str, TokenLayout]:
        B = len(images)
        pv = enc["pixel_values"]
        if pv.shape[0] != B:
            raise RuntimeError(f"{self.model_key}: {pv.shape[0]} tiles for {B} images; dynamic cropping must be off")
        n_ctx = (enc["input_ids"] == self.model.config.image_token_id).sum(dim=1).tolist()
        if any(n != self.tokens_per_tile for n in n_ctx):
            raise RuntimeError(f"{self.model_key}: <IMG_CONTEXT> counts {n_ctx} != {self.tokens_per_tile}")
        vis_mask = torch.ones(self.patches_per_tile + 1, dtype=torch.bool); vis_mask[0] = False
        con_mask = torch.ones(self.tokens_per_tile, dtype=torch.bool)
        return {"vis.last": TokenLayout(False, masks=[vis_mask.clone() for _ in range(B)]),
                "connector": TokenLayout(False, masks=[con_mask.clone() for _ in range(B)])}

    @torch.no_grad()
    def vision_features(self, enc):
        cfg = self.model.config
        batch = self.to_device(enc)
        return self.model.model.get_image_features(pixel_values=batch["pixel_values"],
                                                   vision_feature_layer=cfg.vision_feature_layer,
                                                   vision_feature_select_strategy=cfg.vision_feature_select_strategy)
