"""Family adapter interface: model loading, prompt/batch construction, loci and token layouts.

An adapter answers four questions the protocol leaves to the family (README 7.2):
  1. how to load the pinned checkpoint and its official processor/chat template,
  2. which module output is the consumed final visual block (primary) and the connector output (second),
  3. which entries of those tensors are the consumed image tokens of each batch element,
  4. how to run one forward and read next-token logits at the fixed answer position.
Everything scientific (steering rule, probes, scoring) lives outside the adapter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from ..hooks import LocusHook, TokenLayout
from ..scoring import CandidateSet, candidate_sets

HF_HOME = Path("/rodata/azradonc_dev/m253405/cache")


@dataclass
class LocusInfo:
    locus_id: str                   # "vis.last" | "connector"
    module_path: str                # dotted path under the loaded model
    output_selection: str           # how the hidden state is read from the module output
    hidden_dim: int
    token_rule: str                 # which tokens are consumed and pooled
    affected_branch: str
    bypasses: str
    consumer: str
    prompt_independent: bool = True
    notes: str = ""


@dataclass
class ForwardResult:
    last_logits: torch.Tensor                       # (B, V) float32 on CPU
    stats: dict[str, list[float]]                   # token_norm_mean/median, delta_norm_mean per row
    input_token_count: list[int]
    valid_token_count: list[int]


class Adapter:
    family = "base"
    default_processor_kwargs: dict = {}
    attn_implementation = "sdpa"

    def __init__(self, model_key: str, model_id: str, revision: str, processor_kwargs: dict | None = None):
        self.model_key, self.model_id, self.revision = model_key, model_id, revision
        self.processor_kwargs = dict(self.default_processor_kwargs)
        if processor_kwargs:
            self.processor_kwargs.update(processor_kwargs)
        self.model = None
        self.processor = None
        self.local_path: Path | None = None
        self._cands: dict[str, CandidateSet] | None = None
        self._modules: dict[str, torch.nn.Module] = {}

    # ------------------------------------------------------------------ loading
    def resolve_local_path(self) -> Path:
        """Snapshot directory for the pinned revision. Resolved directly because checkpoints staged with
        allow_patterns (weights + configs only) are 'incomplete' to huggingface_hub's offline resolver."""
        snap = HF_HOME / "hub" / f"models--{self.model_id.replace('/', '--')}" / "snapshots" / self.revision
        if not (snap / "config.json").exists():
            raise FileNotFoundError(f"{self.model_key}: no staged snapshot at {snap}")
        self.local_path = snap
        return snap

    def load(self, device_map="cuda:0", dtype=torch.bfloat16):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        path = self.resolve_local_path()
        self.processor = AutoProcessor.from_pretrained(path, **self.processor_kwargs)
        tok = getattr(self.processor, "tokenizer", self.processor)
        tok.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=dtype, device_map=device_map, attn_implementation=self.attn_implementation)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._modules = dict(self.model.named_modules())
        self._cands = candidate_sets(tok)
        self._install_fp32_head()
        self.after_load()
        return self

    # ------------------------------------------------------------------ float32 answer logits
    def _install_fp32_head(self) -> None:
        """The model's own `logits` are bf16 (0.125 resolution at |logit| ~ 25), which would put every margin on a
        lattice. The protocol stores float32 logits, so the answer-position logits are recomputed in float32 from
        the final hidden state that feeds `lm_head` (captured by a hook) against an fp32 copy of the head weight."""
        head = self.model.lm_head
        self._head_input = None

        def grab(_m, inputs, _out):
            self._head_input = inputs[0]
        head.register_forward_hook(grab)
        self._head_w32 = head.weight.detach().float()
        self._head_b32 = head.bias.detach().float() if getattr(head, "bias", None) is not None else None
        tc = getattr(self.model.config, "text_config", self.model.config)
        self._softcap = getattr(tc, "final_logit_softcapping", None)

    def fp32_logits_from_hidden(self, h_last: torch.Tensor) -> torch.Tensor:
        lg = h_last.float().to(self._head_w32.device) @ self._head_w32.T
        if self._head_b32 is not None:
            lg = lg + self._head_b32
        if self._softcap:
            lg = torch.tanh(lg / self._softcap) * self._softcap
        return lg

    def after_load(self) -> None:
        pass

    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    @property
    def candidates(self) -> dict[str, CandidateSet]:
        assert self._cands is not None
        return self._cands

    def module(self, path: str) -> torch.nn.Module:
        if path not in self._modules:
            raise KeyError(f"{self.model_key}: module {path!r} not found; prefixes {sorted({k.split('.')[0] for k in self._modules})[:10]}")
        return self._modules[path]

    # ------------------------------------------------------------------ prompts
    def messages(self, question: str, with_image: bool = True) -> list[dict]:
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": question}]
        return [{"role": "user", "content": content}]

    def prompt_text(self, question: str, with_image: bool = True) -> str:
        return self.processor.apply_chat_template(self.messages(question, with_image), add_generation_prompt=True,
                                                  tokenize=False)

    def encode(self, images: list[Image.Image] | None, questions: list[str]):
        texts = [self.prompt_text(q, with_image=images is not None) for q in questions]
        kwargs = dict(text=texts, return_tensors="pt", padding=True)
        if images is not None:
            kwargs["images"] = images
        enc = self.processor(**kwargs)
        return enc

    def expand(self, enc, B: int):
        """Replicate a single-example encoding B times along the batch axis (identical rows, no padding)."""
        out = {}
        for k, v in enc.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
                out[k] = v.expand(B, *v.shape[1:]).contiguous()
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------ loci (family-specific)
    def loci(self) -> dict[str, LocusInfo]:
        raise NotImplementedError

    def layouts(self, enc, images: list[Image.Image]) -> dict[str, TokenLayout]:
        """Token layout at each locus for this batch (derived from processor outputs, never guessed)."""
        raise NotImplementedError

    def image_token_mask(self, enc) -> torch.Tensor:
        """(B, L) bool: positions of image placeholder tokens in the language-model input."""
        return enc["input_ids"] == self.model.config.image_token_id

    def vision_features(self, enc):
        """Run only the vision tower and connector on the encoded batch (prompt-independent loci)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ forward
    def to_device(self, enc):
        dev = next(self.model.parameters()).device
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}

    @torch.no_grad()
    def forward_last_logits(self, enc, return_model_logits: bool = False):
        """(B, V) float32 answer-position logits computed from the final hidden state (see _install_fp32_head).
        With `return_model_logits` also return the model's own (bf16-quantised) logits for consistency checks."""
        batch = self.to_device(enc)
        self._head_input = None
        try:
            out = self.model(**batch, logits_to_keep=1, use_cache=False)
        except TypeError:
            out = self.model(**batch, use_cache=False)
        if self._head_input is None:
            raise RuntimeError("lm_head hook did not fire")
        lg32 = self.fp32_logits_from_hidden(self._head_input[:, -1, :]).cpu()
        if return_model_logits:
            return lg32, out.logits[:, -1, :].float().cpu()
        return lg32

    # ------------------------------------------------------------------ metadata for run.json
    def processing_settings(self) -> dict:
        tok = self.tokenizer
        cfg = self.model.config
        ip = getattr(self.processor, "image_processor", None)
        return {
            "processor_kwargs": self.processor_kwargs,
            "image_processor": {k: v for k, v in (ip.to_dict().items() if ip is not None else [])
                                if isinstance(v, (int, float, str, bool, list, dict, type(None)))},
            "chat_template": getattr(self.processor, "chat_template", None) or getattr(tok, "chat_template", None),
            "example_prompt_IY": self.prompt_text("Is there X in this image? Answer yes or no."),
            "special_token_ids": {"image_token_id": getattr(cfg, "image_token_id", None),
                                  "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id},
            "attn_implementation": self.attn_implementation,
            "candidate_tokens": {t: {"positive_ids": list(c.positive_ids), "negative_ids": list(c.negative_ids),
                                     "detail": c.detail} for t, c in self.candidates.items()},
            "model_config": json.loads(cfg.to_json_string()),
        }
