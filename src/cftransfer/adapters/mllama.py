"""Llama 3.2 Vision-Instruct (11B/90B, `Mllama`): ViT-H/14 tower at 560 px -> linear projector -> cross-attention.

Consumed boundary (transformers 5.15 `MllamaVisionModel.forward` / `MllamaModel.forward`): the vision tower runs
a 32-layer local transformer over `num_tiles x 1608` rows per image (1 CLS + 1600 patches per 560x560 tile, zero-
padded from 1601 to a multiple of 8), applies `layernorm_post`, the tile positional embedding and an 8-layer
gated *global* transformer. The tensor the language model consumes is

    cat([global_transformer output, local layer 3, 7, 15, 23, 30 outputs], -1)   -> (B, 1, num_tiles, 1601, 7680)

with the 7 padding rows per tile sliced off, then `multi_modal_projector` (Linear 7680 -> 4096) applied to every
row and the result reshaped to (B*num_tiles, 1601, 4096) and read as keys/values by the eight cross-attention
decoder layers. `cross_attention_mask` admits all 1601 rows (CLS included) of every real tile of the image and
none of the padding tiles, so every row of a real tile is consumed.

Primary locus `vis.last`: the output tensor of the last global-transformer block,
`model.vision_model.global_transformer.layers.{n-1}`, shape (B, num_tiles*1608, 1280). It is dims [0:1280) of
the 7680-d concatenation the projector reads; the remaining 5x1280 dims are the local-transformer layer 3/7/15/
23/30 outputs, which bypass this block and are recorded as such (`bypasses`). Consumed rows at this locus: the
1601 rows (CLS + patches) of each real tile; the 7 padding rows per tile and the padding tiles are excluded.
Connector locus: `model.multi_modal_projector` output (B, 1, num_tiles, 1601, 4096), which the hooks view as
(B, num_tiles*1601, 4096); consumed rows: the 1601 rows of each real tile.

Image strategy (whole family): one 560x560 tile per image. Each image is resized in PIL so that its longer side
is 560 (bilinear), which makes the processor's canvas search pick the (1,1) canvas: no tiling, the processor pads
the image to 560x560 (bottom/right, zeros), aspect_ratio_id 1, tiles 2..4 zero-filled and masked out of the
vision attention and the cross-attention. Every image therefore contributes exactly 1601 consumed rows at both
loci. `max_image_tiles` stays at the checkpoint's 4 because the tile positional embeddings are shaped by it.

Prompt: the checkpoint's chat template, i.e.
`<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`;
the text is tokenised with `add_special_tokens=False` because the template already carries `<|begin_of_text|>`
(the official usage), so the input holds exactly one BOS. Image-free prompts (the semantic-mapping preflight)
get the template's system header, whose date is pinned to the template default so prompts are reproducible.
"""
from __future__ import annotations

import torch
from PIL import Image

from ..hooks import TokenLayout
from .base import Adapter, LocusInfo

TEMPLATE_DATE = "26 Jul 2024"     # the chat template's own fallback date; pinned so image-free prompts never change


class MllamaAdapter(Adapter):
    family = "llama3.2-vision"
    default_processor_kwargs: dict = {}

    def after_load(self) -> None:
        cfg = self.model.config
        vc = cfg.vision_config
        vm = self.module("model.vision_model")
        self.n_global = len(vm.global_transformer.layers)
        self._block_path = f"model.vision_model.global_transformer.layers.{self.n_global - 1}"
        self._proj_path = "model.multi_modal_projector"
        self.module(self._block_path); self.module(self._proj_path)
        self.tile = int(vc.image_size)                                                   # 560
        self.n_rows = (self.tile // int(vc.patch_size)) ** 2 + 1                          # 1601 = CLS + 1600
        self.n_padded = self.n_rows + (8 - self.n_rows % 8) % 8                           # 1608
        self.max_tiles = int(vc.max_num_tiles)                                            # 4
        self.intermediate = [int(i) for i in vc.intermediate_layers_indices]
        self.vision_dim = int(vc.hidden_size)                                             # 1280
        self.consumed_dim = int(vc.vision_output_dim)                                     # 7680
        if self.consumed_dim != self.vision_dim * (1 + len(self.intermediate)):
            raise RuntimeError(f"{self.model_key}: vision_output_dim {self.consumed_dim} != "
                               f"{self.vision_dim} x (1 + {len(self.intermediate)})")
        ip = self.processor.image_processor
        sz = ip.size
        size = sz["height"] if isinstance(sz, dict) else getattr(sz, "height", sz)     # SizeDict in transformers 5.x
        if int(size) != self.tile or int(ip.max_image_tiles) != self.max_tiles:
            raise RuntimeError(f"{self.model_key}: processor tile {size}/{ip.max_image_tiles} != model {self.tile}/{self.max_tiles}")
        self.image_token = int(cfg.image_token_id)

    # ------------------------------------------------------------------ prompts / encoding
    def prompt_text(self, question: str, with_image: bool = True) -> str:
        return self.processor.apply_chat_template(self.messages(question, with_image), add_generation_prompt=True,
                                                  tokenize=False, date_string=TEMPLATE_DATE)

    def fit_single_tile(self, im: Image.Image) -> Image.Image:
        """Longer side -> 560 so the processor's canvas search selects the (1,1) canvas (one tile, no upscaling)."""
        w, h = im.size
        if max(w, h) == self.tile:
            return im
        if w >= h:
            return im.resize((self.tile, max(1, round(h * self.tile / w))), Image.BILINEAR)
        return im.resize((max(1, round(w * self.tile / h)), self.tile), Image.BILINEAR)

    def encode(self, images, questions):
        texts = [self.prompt_text(q, with_image=images is not None) for q in questions]
        kwargs = dict(text=texts, return_tensors="pt", padding=True, add_special_tokens=False)
        if images is not None:
            kwargs["images"] = [[self.fit_single_tile(im)] for im in images]      # one image per sample, nested
        return self.processor(**kwargs)

    def image_token_mask(self, enc) -> torch.Tensor:
        return enc["input_ids"] == self.image_token

    # ------------------------------------------------------------------ loci
    def loci(self) -> dict[str, LocusInfo]:
        cfg = self.model.config
        return {
            "vis.last": LocusInfo(
                "vis.last", self._block_path,
                f"tensor output of global-transformer block {self.n_global - 1} of {self.n_global} "
                f"(B, {self.max_tiles}x{self.n_padded}, {self.vision_dim})",
                self.vision_dim,
                f"the {self.n_rows} rows (CLS + {self.n_rows - 1} patches) of each real 560x560 tile; the "
                f"{self.n_padded - self.n_rows} zero-padding rows per tile and the padding tiles are not consumed",
                f"global-transformer stream = dims [0:{self.vision_dim}) of the {self.consumed_dim}-d tensor the projector reads",
                f"local-transformer layer {self.intermediate} outputs ({len(self.intermediate)}x{self.vision_dim} dims of "
                f"the projector input) bypass this block",
                self._proj_path, True,
                "single tile per image (longer side resized to 560 before the processor); CLS is consumed by the "
                "cross-attention and is included; padding rows are sliced off before the projector"),
            "connector": LocusInfo(
                "connector", self._proj_path,
                f"tensor output of multi_modal_projector (Linear {self.consumed_dim}->{cfg.text_config.hidden_size}), "
                f"(B, 1, {self.max_tiles}, {self.n_rows}, D) viewed as (B, {self.max_tiles}x{self.n_rows}, D)",
                int(cfg.text_config.hidden_size),
                f"the {self.n_rows} rows of each real tile (cross_attention_mask admits them all)",
                "keys/values of the cross-attention decoder layers", "none",
                f"language-model cross-attention layers {list(cfg.text_config.cross_attention_layers)}", True),
        }

    def layouts(self, enc, images) -> dict[str, TokenLayout]:
        B = len(images)
        pv = enc["pixel_values"]
        if pv.dim() != 6 or pv.shape[0] != B or pv.shape[1] != 1 or pv.shape[2] != self.max_tiles:
            raise RuntimeError(f"{self.model_key}: pixel_values {tuple(pv.shape)} != (B={B}, 1 image, {self.max_tiles} tiles, ...)")
        n_img = self.image_token_mask(enc).sum(dim=1).tolist()
        if any(n != 1 for n in n_img):
            raise RuntimeError(f"{self.model_key}: <|image|> counts {n_img} != 1 per sample")
        arm = enc["aspect_ratio_mask"]                       # (B, 1, max_tiles)
        cam = enc["cross_attention_mask"]                    # (B, L, 1, max_tiles); last position is real (left padding)
        vis, con = [], []
        for b in range(B):
            tiles = arm[b, 0].bool()
            if int(tiles.sum()) != 1 or not bool(tiles[0]):
                raise RuntimeError(f"{self.model_key}: sample {b} has tiles {tiles.tolist()}; the single-tile strategy expects [1,0,0,0]")
            if not torch.equal(cam[b, -1, 0].bool(), tiles):
                raise RuntimeError(f"{self.model_key}: cross_attention_mask {cam[b, -1, 0].tolist()} != aspect_ratio_mask {tiles.tolist()}")
            m_vis = torch.zeros(self.max_tiles * self.n_padded, dtype=torch.bool)
            m_con = torch.zeros(self.max_tiles * self.n_rows, dtype=torch.bool)
            for t in range(self.max_tiles):
                if bool(tiles[t]):
                    m_vis[t * self.n_padded: t * self.n_padded + self.n_rows] = True
                    m_con[t * self.n_rows: (t + 1) * self.n_rows] = True
            vis.append(m_vis); con.append(m_con)
        return {"vis.last": TokenLayout(False, masks=vis), "connector": TokenLayout(False, masks=con)}

    @torch.no_grad()
    def vision_features(self, enc):
        batch = self.to_device(enc)
        out = self.model.model.vision_model(pixel_values=batch["pixel_values"], aspect_ratio_ids=batch["aspect_ratio_ids"],
                                            aspect_ratio_mask=batch["aspect_ratio_mask"])
        return self.model.model.multi_modal_projector(out.last_hidden_state)
