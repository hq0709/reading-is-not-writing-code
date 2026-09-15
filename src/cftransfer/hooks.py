"""Forward hooks: relative-token steering and pooled capture at a consumed visual locus.

The steering rule is fixed by the protocol (README 5.2):  h'_t = h_t + alpha * ||h_t||_2 * v,
with the norm taken from that token's own clean activation at this forward. The hook never falls
back silently: the adapter must supply, for every batch element, the exact token indices the
connector consumes at this locus. Anything else raises.

TOKENW (per-token weights): with `scorers` (B, D) and `modes` (B,) armed, token t of element b gets
h'_t = h_t + alpha * w_t * ||h_t|| * v with w from token_weights(h_t . scorer_b, mode): mode 1 ("tokenw")
w = softmax over the consumed tokens of the score (temperature 1) times T (mean 1); mode 2 ("topq") the
top TOPQ_FRACTION of tokens by score get 1/fraction, the rest 0 (mean 1); mode 0 is the uniform write.
The score h_t . (P w_c / s) is the token's projected, scaled probe logit up to an additive constant, which
neither the softmax nor the top-quarter selection sees.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

TOPQ_FRACTION = 0.25
MODE_UNIFORM, MODE_SOFTMAX, MODE_TOPQ = 0, 1, 2


def token_weights(scores: torch.Tensor, modes: torch.Tensor, fraction: float = TOPQ_FRACTION) -> torch.Tensor:
    """(B, T) per-token weights with mean 1 over T for every row: uniform (mode 0), softmax x T (mode 1), or the
    top round(fraction * T) tokens (at least one) at T / k and the rest 0 (mode 2). `scores`: (B, T) float32."""
    B, T = scores.shape
    k = max(1, int(round(fraction * T)))
    soft = torch.softmax(scores.float(), dim=1) * T
    top = torch.zeros_like(soft).scatter_(1, scores.topk(k, dim=1).indices, T / k)
    modes = modes.to(scores.device).view(B, 1)
    return torch.where(modes == MODE_SOFTMAX, soft, torch.where(modes == MODE_TOPQ, top, torch.ones_like(soft)))


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
        self.scorers: torch.Tensor | None = None            # (B, D) per-element token scorer for weighted writes
        self.modes: torch.Tensor | None = None              # (B,) token_weights modes
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
            capture: bool = False, scorers: torch.Tensor | None = None, modes: torch.Tensor | None = None):
        if vectors is not None:
            if vectors.shape[0] != layout.batch or alphas is None or alphas.shape[0] != layout.batch:
                raise ValueError("vectors/alphas must have one row per batch element in the layout")
        if scorers is not None:
            if vectors is None or modes is None or scorers.shape != vectors.shape or modes.shape[0] != layout.batch:
                raise ValueError("scorers (B, D) and modes (B,) must match the steered batch")
        self.layout, self.vectors, self.alphas, self.capture = layout, vectors, alphas, capture
        self.scorers, self.modes = scorers, modes
        self.pooled, self.stats, self.calls = None, {}, 0

    def _weights(self, hf: torch.Tensor, b: int | None) -> torch.Tensor | None:
        """Per-token weights for one element (hf: (T, D), b given) or the batch (hf: (B, T, D)); None when uniform."""
        if self.scorers is None:
            return None
        sc = self.scorers.to(device=hf.device, dtype=torch.float32)
        if b is not None:
            return token_weights((hf @ sc[b]).unsqueeze(0), self.modes[b:b + 1])[0]
        return token_weights(torch.einsum("btd,bd->bt", hf, sc), self.modes)

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
        restore = None
        if not lay.flat and h.dim() > 3:
            # (B, ..., D) tensors (e.g. Mllama's (B, images, tiles, rows, D)) are steered as (B, T, D) with
            # T = product of the middle axes, in the tensor's own row order; the original shape is restored
            if h.shape[0] != lay.batch:
                raise RuntimeError(f"{self.name}: batched layout expects leading axis {lay.batch}, got {tuple(h.shape)}")
            orig_shape = h.shape
            h = h.reshape(h.shape[0], -1, h.shape[-1])
            restore = lambda t: t.reshape(orig_shape)
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

        counts = lay.counts()
        if len(set(counts)) == 1 and (lay.flat or all(bool(m.all()) for m in lay.masks) or
                                      all(torch.equal(m, lay.masks[0]) for m in lay.masks)):
            return self._vectorised(output, h, lay, counts[0], restore)
        tok_mean, tok_med, delta_mean, pooled, w_max = [], [], [], [], []
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
            wt = self._weights(hb.float(), b)                          # (T_b,) or None
            scale = a * norms if wt is None else a * norms * wt
            w_max.append(1.0 if wt is None else float(wt.max()))
            delta = scale.unsqueeze(-1) * v.unsqueeze(0)              # (T_b, D) in fp32
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
        if self.vectors is not None:
            self.stats["token_weight_max"] = w_max
        if self.capture:
            self.pooled = torch.stack(pooled)
        if self.vectors is None:
            return None
        return with_hidden(output, restore(new) if restore else new)


    def _vectorised(self, output, h, lay: TokenLayout, T: int, restore=None):
        """Same arithmetic as the loop, for layouts where every element has the same consumed-token pattern."""
        B = lay.batch
        if lay.flat:
            hb = h.view(B, T, -1)
        else:
            idx = lay.masks[0].to(h.device)
            hb = h[:, idx, :]                                          # (B, T, D)
        hf = hb.float()
        norms = hf.norm(dim=-1)                                        # (B, T)
        self.stats = {"token_norm_mean": norms.mean(dim=1).tolist(),
                      "token_norm_median": norms.median(dim=1).values.tolist(),
                      "delta_norm_mean": [0.0] * B}
        if self.capture:
            self.pooled = hf.mean(dim=1).cpu()
        if self.vectors is None:
            return None
        v = self.vectors.to(device=h.device, dtype=torch.float32)     # (B, D)
        a = self.alphas.to(device=h.device, dtype=torch.float32)      # (B,)
        wt = self._weights(hf, None)                                   # (B, T) or None
        scale = a[:, None] * norms if wt is None else a[:, None] * norms * wt
        delta = scale[:, :, None] * v[:, None, :]                     # (B, T, D)
        self.stats["delta_norm_mean"] = delta.norm(dim=-1).mean(dim=1).tolist()
        self.stats["token_weight_max"] = [1.0] * B if wt is None else wt.max(dim=1).values.tolist()
        steered = (hf + delta).to(h.dtype)
        if lay.flat:
            new = steered.reshape(h.shape)
        else:
            new = h.clone()
            new[:, idx, :] = steered
        return with_hidden(output, restore(new) if restore else new)


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
        if h is not None and h.dim() > 3:
            h = h.reshape(h.shape[0], -1, h.shape[-1])      # same (B, T, D) view the steering hook uses
        self.value = None if h is None else h.detach().float().cpu()
        return None
