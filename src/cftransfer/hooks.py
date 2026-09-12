"""Forward hooks: relative-token steering and pooled capture at a consumed visual locus.

The steering rule is fixed by the protocol (README 5.2):  h'_t = h_t + alpha * ||h_t||_2 * v,
with the norm taken from that token's own clean activation at this forward. The hook never falls
back silently: the adapter must supply, for every batch element, the exact token indices the
connector consumes at this locus. Anything else raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def as_hidden(out):
    """The hidden-state tensor of whatever a module returned (tensor, tuple, or ModelOutput)."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        return out[0] if out and torch.is_tensor(out[0]) else None
    for attr in ("last_hidden_state", "hidden_states"):
        v = getattr(out, attr, None)
        if torch.is_tensor(v):
            return v
    return None


def with_hidden(out, new):
    """Rebuild the module's return value with the hidden state replaced."""
    if torch.is_tensor(out):
        return new
    if isinstance(out, tuple):
        return (new,) + tuple(out[1:])
    if isinstance(out, list):
        return [new] + list(out[1:])
    if hasattr(out, "last_hidden_state"):
        out.last_hidden_state = new
        return out
    raise TypeError(f"cannot replace hidden state in {type(out)}")


@dataclass
class TokenLayout:
    """Which entries of the hooked tensor belong to which batch element.

    `flat=True`: the tensor is (N_tokens, D) with images concatenated; `slices[b] = (start, end)`.
    `flat=False`: the tensor is (B, T, D); `masks[b]` is a bool tensor of length T marking consumed tokens.
    """
    flat: bool
    slices: list[tuple[int, int]] = field(default_factory=list)
    masks: list[torch.Tensor] = field(default_factory=list)

    @property
    def batch(self) -> int:
        return len(self.slices) if self.flat else len(self.masks)

    def counts(self) -> list[int]:
        return [e - s for s, e in self.slices] if self.flat else [int(m.sum()) for m in self.masks]


class LocusHook:
    """One forward hook on one module; used both for steering and for pooled capture.

    Set `layout` before every forward (the adapter derives it from the processor outputs).
    Set `vectors` (B, D) and `alphas` (B,) to steer; leave `vectors=None` for a clean pass.
    After the forward, `stats` holds per-element token-norm mean/median and mean delta norm, and
    `pooled` holds the mean over consumed tokens of the *clean* activation when `capture=True`.
    """

    def __init__(self, module: torch.nn.Module, name: str, prepend: bool = True):
        self.module, self.name, self.prepend = module, name, prepend
        self.layout: TokenLayout | None = None
        self.vectors: torch.Tensor | None = None
        self.alphas: torch.Tensor | None = None
        self.capture = False
        self.pooled: torch.Tensor | None = None
        self.stats: dict[str, list[float]] = {}
        self.calls = 0
        self._handle = None

    # ---------------------------------------------------------------- lifecycle
    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook, prepend=self.prepend)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def arm(self, layout: TokenLayout, vectors: torch.Tensor | None = None, alphas: torch.Tensor | None = None,
            capture: bool = False):
        if vectors is not None:
            if vectors.shape[0] != layout.batch or alphas is None or alphas.shape[0] != layout.batch:
                raise ValueError("vectors/alphas must have one row per batch element in the layout")
        self.layout, self.vectors, self.alphas, self.capture = layout, vectors, alphas, capture
        self.pooled, self.stats, self.calls = None, {}, 0

    # ---------------------------------------------------------------- the hook
    def _hook(self, _module, _inputs, output):
        h = as_hidden(output)
        if h is None:
            raise RuntimeError(f"{self.name}: module returned no hidden-state tensor")
        if self.layout is None:
            raise RuntimeError(f"{self.name}: hook fired without a token layout")
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError(f"{self.name}: hooked module fired {self.calls} times in one forward; layout is ambiguous")
        lay = self.layout
        if lay.flat:
            if h.dim() != 2:
                raise RuntimeError(f"{self.name}: flat layout expects (N, D), got {tuple(h.shape)}")
            total = lay.slices[-1][1] if lay.slices else 0
            if total != h.shape[0]:
                raise RuntimeError(f"{self.name}: layout covers {total} tokens but tensor has {h.shape[0]}")
        else:
            if h.dim() != 3 or h.shape[0] != lay.batch:
                raise RuntimeError(f"{self.name}: batched layout expects ({lay.batch}, T, D), got {tuple(h.shape)}")
            for m in lay.masks:
                if m.shape[0] != h.shape[1]:
                    raise RuntimeError(f"{self.name}: mask length {m.shape[0]} != token axis {h.shape[1]}")

        tok_mean, tok_med, delta_mean, pooled = [], [], [], []
        new = h if self.vectors is None else h.clone()
        for b in range(lay.batch):
            if lay.flat:
                s, e = lay.slices[b]
                hb = h[s:e]
            else:
                hb = h[b][lay.masks[b].to(h.device)]
            if hb.shape[0] == 0:
                raise RuntimeError(f"{self.name}: batch element {b} has no consumed tokens")
            norms = hb.float().norm(dim=-1)                      # (T_b,), clean norms
            tok_mean.append(float(norms.mean()))
            tok_med.append(float(norms.median()))
            if self.capture:
                pooled.append(hb.float().mean(dim=0).cpu())
            if self.vectors is None:
                delta_mean.append(0.0)
                continue
            v = self.vectors[b].to(device=h.device, dtype=torch.float32)
            a = float(self.alphas[b])
            delta = (a * norms).unsqueeze(-1) * v.unsqueeze(0)      # (T_b, D) in fp32
            delta_mean.append(float(delta.norm(dim=-1).mean()))
            steered = (hb.float() + delta).to(h.dtype)
            if lay.flat:
                new[s:e] = steered
            else:
                idx = lay.masks[b].to(h.device)
                row = new[b]
                row[idx] = steered
                new[b] = row
        self.stats = {"token_norm_mean": tok_mean, "token_norm_median": tok_med, "delta_norm_mean": delta_mean}
        if self.capture:
            self.pooled = torch.stack(pooled)
        return with_hidden(output, new) if self.vectors is not None else None


class CaptureHook:
    """Record a module's hidden-state output (used for preflight consumer-change checks)."""

    def __init__(self, module: torch.nn.Module, name: str):
        self.module, self.name = module, name
        self.value: torch.Tensor | None = None
        self._handle = None

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
        return False

    def _hook(self, _m, _i, output):
        h = as_hidden(output)
        self.value = None if h is None else h.detach().float().cpu()
        return None
