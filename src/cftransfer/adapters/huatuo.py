"""HuatuoGPT-Vision-7B (FreedomIntelligence/HuatuoGPT-Vision-7B) via the native conversion produced by
scripts/mayo/convert_huatuo.py.

The checkpoint is LLaVA-1.5 with a Qwen2-7B language model: a CLIP ViT-L/14-336 tower, `hidden_states[-2]`,
CLS dropped, a two-layer GELU projector, 576 image tokens. Architecture and loci are therefore the LLaVA
adapter's and it is reused unchanged; what differs is the input construction, which follows the checkpoint's
own inference code (HuatuoGPT-Vision `cli.py`) rather than a chat template:

  prompt    "<|user|>\\n<image>\\n{question}\\n<|assistant|>\\n"   (cli.HuatuoChatbot.preprocess_huatuo with
            insert_image_placeholder; `<|user|>` and `<|assistant|>` are PLAIN TEXT for this tokenizer, which
            is what the checkpoint's own code tokenises, so they are not added as special tokens)
  tokens    official tokenizer_image_token: each chunk around <image> is tokenised separately with
            add_special_tokens=False; Qwen2 has no BOS, so no chunk offset applies, and the single sentinel is
            replaced by 576 x id 151646 -> 596 positions for the Effusion question
  image     image_aspect_ratio="pad": non-square images are padded to a centred square with the CLIP-mean
            background before the 336 resize / crop / normalisation
  vocab     answer logits are the source model's own 152,064-row head. Qwen2's head has 418 more rows than its
            tokenizer has tokens, so the conversion named <image> at the first unreachable row (151646) instead
            of adding one; no row was added, resized or reordered, and no row has to be excluded from scoring.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .llava import LlavaAdapter
from .llavamed import expand2square

CONVERTED = Path("/rodata/azradonc_dev/m253405/cache/converted/HuatuoGPT-Vision-7B-hf")
IMAGE_TOKEN, IMAGE_TOKEN_ID, HEAD_ROWS, N_IMAGE_TOKENS = "<image>", 151646, 152064, 576


class HuatuoAdapter(LlavaAdapter):
    family = "huatuo-vision"

    def resolve_local_path(self) -> Path:
        if not (CONVERTED / "config.json").exists() or not (CONVERTED / "conversion.json").exists():
            raise FileNotFoundError(f"{self.model_key}: converted checkpoint with receipt not found at {CONVERTED}")
        self.local_path = CONVERTED
        return CONVERTED

    def after_load(self) -> None:
        super().after_load()
        tok = self.tokenizer
        if tok.convert_tokens_to_ids(IMAGE_TOKEN) != IMAGE_TOKEN_ID or tok.bos_token_id is not None:
            raise RuntimeError(f"{self.model_key}: tokenizer does not carry <image>={IMAGE_TOKEN_ID} with no BOS")
        if int(self.model.lm_head.weight.shape[0]) != HEAD_ROWS:
            raise RuntimeError(f"{self.model_key}: head has {self.model.lm_head.weight.shape[0]} rows, expected {HEAD_ROWS}")
        ip = self.processor.image_processor
        self._background = tuple(int(x * 255) for x in ip.image_mean)

    # ---- prompt and tokens
    def prompt_text(self, question: str, with_image: bool = True) -> str:
        body = f"{IMAGE_TOKEN}\n{question}" if with_image else question
        return f"<|user|>\n{body}\n<|assistant|>\n"

    def tokenize_image_prompt(self, prompt: str) -> list[int]:
        """Official `tokenizer_image_token` (cli.HuatuoChatbot) with the -200 sentinel replaced by 576 native
        placeholders. The chunk-offset branch of the original is dead here: Qwen2 has no BOS token."""
        tok = self.tokenizer
        chunks = [tok(c, add_special_tokens=False).input_ids for c in prompt.split(IMAGE_TOKEN)]
        ids, offset = [], 0
        if chunks and chunks[0] and tok.bos_token_id is not None and chunks[0][0] == tok.bos_token_id:
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
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
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
        s["prompt_construction"] = ("the checkpoint's own recipe (HuatuoGPT-Vision cli.py): insert_image_placeholder puts "
                                    "'<image>\\n' before the question and preprocess_huatuo wraps it as "
                                    "'<|user|>\\n<image>\\n{q}\\n<|assistant|>\\n'; chunked tokenizer_image_token with "
                                    f"{N_IMAGE_TOKENS} x <image>({IMAGE_TOKEN_ID}); expand2square padding with the CLIP mean; "
                                    f"answer logits over the source model's own {HEAD_ROWS}-row head")
        s["conversion_receipt"] = str(CONVERTED / "conversion.json")
        return s
