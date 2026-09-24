"""ECHO (Midea-AIRC), a discrete-diffusion chest-report model built on the Qwen2.5-VL 7B stack.

The vision side is a rename of Qwen2.5-VL: `model.visual.blocks` feed `model.visual.merger`, with the same
window permutation, the same 2x2 spatial merge and the same flat per-image slices, so both loci and the token
layout are inherited unchanged from QwenVLAdapter.

The reader is not. ECHO decodes by discrete diffusion: it fills masked positions rather than extending a
sequence, so the answer distribution does not sit at a next-token position. Appending one `<|MASK|>` after the
generation prompt puts it there instead -- the mask is the last input token, the model predicts what fills it,
and `forward_last_logits` reads position -1 exactly as it does for an autoregressive family. One forward pass,
no sampling, so the determinism gate reads the same quantity it reads everywhere else.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..scoring import candidate_sets
from .qwen import QwenVLAdapter

# the published modelling file imports an unused fused RMSNorm from flash_attn at module scope, which alone
# blocks loading; scripts/mayo/convert_echo.py stages a copy with that one line guarded and writes the receipt
CONVERTED = Path("/rodata/azradonc_dev/m253405/cache/converted/echo-block4-hf")


class EchoAdapter(QwenVLAdapter):
    family = "echo"
    # ECHO ships fp32 weights and custom modelling code; the processor is the stock Qwen2.5-VL one
    default_processor_kwargs = {"min_pixels": 336 * 336, "max_pixels": 336 * 336}
    mask_token = "<|MASK|>"

    def load(self, device_map="cuda:0", dtype=torch.bfloat16):
        """Same as the base load, with two checkpoint facts. The repository registers its classes under
        AutoModelForCausalLM and not AutoModelForImageTextToText, and its modelling file is remote code, which
        transformers will not run unattended without trust_remote_code. The processor is stock Qwen2.5-VL."""
        from transformers import AutoModelForCausalLM, AutoProcessor
        path = self.resolve_local_path()
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, **self.processor_kwargs)
        tok = getattr(self.processor, "tokenizer", self.processor)
        tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, dtype=dtype, device_map=device_map,
            attn_implementation=self.attn_implementation)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._modules = dict(self.model.named_modules())
        self._cands = candidate_sets(tok)
        self._install_fp32_head()
        self.after_load()
        return self

    def resolve_local_path(self) -> Path:
        if not (CONVERTED / "config.json").exists() or not (CONVERTED / "conversion.json").exists():
            raise FileNotFoundError(f"{self.model_key}: staged checkpoint with receipt not found at {CONVERTED}")
        self.local_path = CONVERTED
        return CONVERTED

    def prompt_text(self, question: str, with_image: bool = True) -> str:
        return super().prompt_text(question, with_image) + self.mask_token

    def after_load(self) -> None:
        super().after_load()
        mid = getattr(self.model.config, "mask_token_id", None)
        got = self.tokenizer.convert_tokens_to_ids(self.mask_token)
        if mid is None or got != mid:
            raise RuntimeError(f"{self.model_key}: {self.mask_token} tokenises to {got}, config says {mid}; "
                               f"the answer position would not be the masked one")
