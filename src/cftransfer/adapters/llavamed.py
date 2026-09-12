"""LLaVA-Med v1.5 (Mistral-7B) via the native conversion produced by scripts/mayo/convert_llavamed.py
(docs/LLAVA_MED_RUNTIME_FEASIBILITY.md). Architecture and loci are those of LLaVA-1.5 (CLIP ViT-L/14-336,
hidden_states[-2], CLS dropped, two-layer GELU projector), so the LLaVA adapter is reused; what differs is the
input construction, which follows the official LLaVA-Med code rather than a chat template:

  prompt    "[INST] <image>\\n{question} [/INST]"   (mistral_instruct conversation; BOS added by the tokenizer)
  tokens    official tokenizer_image_token: each chunk around <image> is tokenized separately (chunk BOS stripped
            after the first), the single sentinel is replaced by 576 x id 32000 -> 605 positions for the Effusion
            question, preserving the space token immediately after the image
  image     image_aspect_ratio="pad": non-square images are padded to a centred square with the CLIP-mean
            background before the 336 resize/crop/normalisation
  vocab     answer logits are scored over the original 32,000-token vocabulary; the added <image> row (32000)
            is excluded from the fp32 head so log-partitions match the source model exactly
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .llava import LlavaAdapter

CONVERTED = Path("/rodata/azradonc_dev/m253405/cache/converted/llava-med-v1.5-mistral-7b-hf")
IMAGE_TOKEN, IMAGE_TOKEN_ID, ORIGINAL_VOCAB, N_IMAGE_TOKENS = "<image>", 32000, 32000, 576


def expand2square(im: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """Official LLaVA `expand2square`: centred paste on a square canvas filled with the processor mean."""
    w, h = im.size
    if w == h:
        return im
    if w > h:
        canvas = Image.new(im.mode, (w, w), background)
        canvas.paste(im, (0, (w - h) // 2))
    else:
        canvas = Image.new(im.mode, (h, h), background)
        canvas.paste(im, ((h - w) // 2, 0))
    return canvas


class LlavaMedAdapter(LlavaAdapter):
    family = "llava-med"

    def resolve_local_path(self) -> Path:
        if not (CONVERTED / "config.json").exists() or not (CONVERTED / "conversion.json").exists():
            raise FileNotFoundError(f"{self.model_key}: converted checkpoint with receipt not found at {CONVERTED}")
        self.local_path = CONVERTED
        return CONVERTED

    def after_load(self) -> None:
        super().after_load()
        tok = self.tokenizer
        if tok.convert_tokens_to_ids(IMAGE_TOKEN) != IMAGE_TOKEN_ID or tok.bos_token_id != 1:
            raise RuntimeError(f"{self.model_key}: tokenizer does not carry <image>=32000 / BOS=1")
        ip = self.processor.image_processor
        self._background = tuple(int(x * 255) for x in ip.image_mean)

    # ---- float32 head restricted to the original vocabulary (the added row is not part of the source model)
    def _install_fp32_head(self) -> None:
        super()._install_fp32_head()
        self._head_w32 = self._head_w32[:ORIGINAL_VOCAB]
        if self._head_b32 is not None:
            self._head_b32 = self._head_b32[:ORIGINAL_VOCAB]

    # ---- prompt and tokens
    def prompt_text(self, question: str, with_image: bool = True) -> str:
        return f"[INST] {IMAGE_TOKEN}\n{question} [/INST]" if with_image else f"[INST] {question} [/INST]"

    def tokenize_image_prompt(self, prompt: str) -> list[int]:
        """Official `tokenizer_image_token` with the -200 sentinel replaced by 576 native placeholders."""
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
            out.extend([IMAGE_TOKEN_ID] * N_IMAGE_TOKENS if t == -200 else [t])
        return out

    def encode(self, images, questions):
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
            padded = [expand2square(im, self._background) for im in images]
            enc["pixel_values"] = self.processor.image_processor(padded, return_tensors="pt")["pixel_values"]
        return enc

    def image_token_mask(self, enc) -> torch.Tensor:
        return enc["input_ids"] == IMAGE_TOKEN_ID

    def processing_settings(self) -> dict:
        s = super().processing_settings()
        s["prompt_construction"] = ("official LLaVA-Med mistral_instruct conversation '[INST] <image>\\n{q} [/INST]'; chunked "
                                    "tokenizer_image_token with 576 x <image>(32000); expand2square padding with CLIP mean; "
                                    "answer logits over the original 32,000-token vocabulary")
        s["conversion_receipt"] = str(CONVERTED / "conversion.json")
        return s
