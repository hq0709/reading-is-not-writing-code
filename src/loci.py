"""Locus definitions and activation capture for Qwen2.5-VL-family models.

A locus is a named point along the computation where a representation can be read. The whole paper rests
on these being comparable across models and across depth, so the pooling is fixed here once and every
experiment imports it rather than choosing its own.

Verified module names for Qwen2.5-VL (7B, 28 LLM layers) and Lingshu-7B, which shares the architecture:

    model.visual                     Qwen2_5_VisionTransformerPretrainedModel
    model.visual.blocks[i]           vision transformer blocks
    model.visual.merger              the connector. Output is the visual tokens the LLM consumes
    model.language_model.layers[i]   decoder layers, residual stream
    model.language_model.norm        final norm
    lm_head                          unembedding

The merger is the connector for this family: it consumes vision-block output and emits the token
sequence that is spliced into the language model's input embeddings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch


def as_hidden(out):
    """Pull the hidden-state tensor out of whatever a module returned.

    Modules in this study return three different things and the difference is silent: a bare tensor
    (Qwen blocks), a tuple whose first element is the tensor (decoder layers), and a ModelOutput
    (CLIPVisionTransformer returns BaseModelOutputWithPooling). Treating a ModelOutput as "not a tensor"
    makes the locus vanish from the results with no error, which is how the LLaVA vision-tower locus was
    silently dropped until an architecture check caught it.
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        return out[0] if out and torch.is_tensor(out[0]) else None
    for attr in ("last_hidden_state", "hidden_states", "logits"):
        v = getattr(out, attr, None)
        if torch.is_tensor(v):
            return v
        if isinstance(v, (tuple, list)) and v and torch.is_tensor(v[-1]):
            return v[-1]
    try:
        v = out[0]
        return v if torch.is_tensor(v) else None
    except Exception:
        return None


@dataclass
class Locus:
    """One readable point. `name` is what appears in every table and figure."""
    name: str
    module: str          # dotted module path to hook
    stage: str           # vision | connector | llm
    depth: int           # ordering along the computation, for the x axis of a survival curve
    positions: str       # visual | answer | all


def qwen_loci(n_vision_blocks: int, n_llm_layers: int, vision_every: int = 4) -> list[Locus]:
    """The standard locus set for a Qwen2.5-VL-family model.

    Vision blocks are subsampled because there are 32 of them and they are not the object of study;
    LLM layers are taken in full because the survival curve's shape across LLM depth is the result.
    """
    loci: list[Locus] = []
    d = 0
    for i in range(0, n_vision_blocks, vision_every):
        loci.append(Locus(f"vis.block{i}", f"model.visual.blocks.{i}", "vision", d, "all"))
        d += 1
    loci.append(Locus("vis.last", f"model.visual.blocks.{n_vision_blocks - 1}", "vision", d, "all")); d += 1
    # the connector. This is the locus the field has measured exactly once.
    loci.append(Locus("connector", "model.visual.merger", "connector", d, "all")); d += 1
    for i in range(n_llm_layers):
        loci.append(Locus(f"llm.L{i}.vis", f"model.language_model.layers.{i}", "llm", d, "visual"))
        loci.append(Locus(f"llm.L{i}.ans", f"model.language_model.layers.{i}", "llm", d, "answer"))
        d += 1
    return loci


def loci_for(arch, vision_every: int = 4) -> list[Locus]:
    """The standard locus set for any registered architecture.

    Same shape for every model so the survival curves are comparable: a subsample of vision blocks, the
    last vision block, the connector, then every LLM layer at both the visual positions and the answer
    position. Only the module paths differ, and those come from the registry, which is checked against a
    loaded model by src/verify_arch.py.
    """
    loci: list[Locus] = []
    d = 0
    for i in range(0, arch.n_vision_blocks, vision_every):
        loci.append(Locus(f"vis.block{i}", arch.vision_block_fmt.format(i=i), "vision", d, "all"))
        d += 1
    # The block the connector actually consumes. LLaVA discards the final block (vision_feature_layer=-2),
    # so hooking n-1 there reads a tensor that never reaches the language model.
    last = arch.n_vision_blocks + getattr(arch, "vision_feature_layer", -1)
    loci.append(Locus("vis.last", arch.vision_block_fmt.format(i=last), "vision", d, "all"))
    d += 1
    loci.append(Locus("connector", arch.connector, "connector", d, "all"))
    d += 1
    for i in range(arch.n_llm_layers):
        loci.append(Locus(f"llm.L{i}.vis", arch.llm_layer_fmt.format(i=i), "llm", d, "visual"))
        loci.append(Locus(f"llm.L{i}.ans", arch.llm_layer_fmt.format(i=i), "llm", d, "answer"))
        d += 1
    return loci


class ActivationCache:
    """Registers forward hooks and pools each locus to one vector per example.

    Pooling is fixed: mean over the selected positions. A single forward pass fills every locus, which is
    what makes the full-depth curve cheap.
    """

    def __init__(self, model, loci: list[Locus], image_token_id: int | None = None):
        self.model = model
        self.loci = loci
        self.image_token_id = image_token_id
        self.raw: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._visual_mask: torch.Tensor | None = None
        self._modules = dict(model.named_modules())
        self.batch_size = 1

    def _resolve(self, path: str):
        m = self._modules.get(path)
        if m is None:
            raise KeyError(f"module {path!r} not found. Available prefixes: "
                           f"{sorted({k.split('.')[0] for k in self._modules})[:12]}")
        return m

    def set_visual_mask(self, input_ids: torch.Tensor):
        """Which sequence positions hold visual tokens, and how many examples are in the batch.

        The batch size is needed because some vision towers do not keep a batch axis. Qwen2.5-VL flattens
        every image's patches into one long sequence, so its blocks emit (sum_of_patches, D) rather than
        (B, T, D). Pooling that as if it had no batch axis collapses the whole batch to a single row,
        which is silent: the array simply comes out B times too short.
        """
        self.batch_size = int(input_ids.shape[0])
        if self.image_token_id is None:
            self._visual_mask = None
            return
        self._visual_mask = input_ids == self.image_token_id

    def __enter__(self):
        by_module: dict[str, list[Locus]] = {}
        for lo in self.loci:
            by_module.setdefault(lo.module, []).append(lo)

        for path, group in by_module.items():
            mod = self._resolve(path)

            def make_hook(group=group):
                def hook(_m, _inp, out):
                    h = as_hidden(out)
                    if h is None:
                        return
                    for lo in group:
                        self.raw[lo.name] = self._pool(h, lo).detach().float().cpu()
                return hook

            self._handles.append(mod.register_forward_hook(make_hook()))
        return self

    def _pool(self, h: torch.Tensor, lo: Locus) -> torch.Tensor:
        """(B, T, D) or (B*T, D) -> (B, D). Mean pooling, fixed across every locus."""
        if h.dim() == 2:
            # No batch axis: the tokens of every example are concatenated. Split them back out. Every
            # image in a batch contributes the same number of tokens here because the processor pins the
            # resolution, so an even split is exact; if it ever is not, fail loudly rather than average
            # across examples.
            B = max(self.batch_size, 1)
            n = h.shape[0]
            if B == 1:
                return h.mean(dim=0, keepdim=True)
            if n % B != 0:
                raise RuntimeError(
                    f"locus {lo.name}: {n} tokens do not divide evenly into {B} examples. Pin the "
                    f"processor resolution, or pass per-example token counts.")
            return h.view(B, n // B, h.shape[-1]).mean(dim=1)
        if lo.positions == "all":
            return h.mean(dim=1)
        if lo.positions == "answer":
            return h[:, -1, :]
        if lo.positions == "visual":
            if self._visual_mask is None or self._visual_mask.shape[1] != h.shape[1]:
                return h.mean(dim=1)           # fall back, and the caller must record that it happened
            mask = self._visual_mask.to(h.device).unsqueeze(-1)
            return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        raise ValueError(lo.positions)

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def pop(self) -> dict[str, torch.Tensor]:
        out, self.raw = self.raw, {}
        return out


def find_image_token_id(processor, config) -> int | None:
    """The token id that marks a visual position in the LLM input sequence."""
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(config, attr, None)
        if isinstance(v, int):
            return v
    tok = getattr(processor, "tokenizer", processor)
    for s in ("<|image_pad|>", "<image>", "<img>"):
        try:
            i = tok.convert_tokens_to_ids(s)
            if i is not None and i >= 0 and i != getattr(tok, "unk_token_id", -1):
                return i
        except Exception:
            continue
    return None
