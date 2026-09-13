"""LocusHook/CaptureHook on a (B, images, tiles, rows, D) tensor (Mllama projector output): steered as (B, T, D)
in row order, shape restored, padding rows untouched, pooled capture over the masked rows only."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cftransfer.hooks import CaptureHook, LocusHook, TokenLayout   # noqa: E402


def test_hook_on_5d_tensor_restores_shape_and_masks_rows():
    torch.manual_seed(0)
    B, tiles, R, D = 2, 3, 4, 6
    lin = torch.nn.Identity()
    h = torch.randn(B, 1, tiles, R, D)
    v = torch.randn(B, D); v = v / v.norm(dim=1, keepdim=True)
    al = torch.tensor([0.25, -0.5])
    mask = torch.zeros(tiles * R, dtype=torch.bool); mask[:R] = True           # only tile 0 consumed
    hook, cap = LocusHook(lin, "conn"), CaptureHook(lin, "cap")
    with hook, cap:
        hook.arm(TokenLayout(False, masks=[mask, mask]), v, al, capture=True)
        out = lin(h.clone())
    assert out.shape == h.shape
    exp = h.clone()
    t0 = h[:, 0, 0]                                                          # (B, R, D)
    exp[:, 0, 0] = t0 + al[:, None, None] * t0.norm(dim=-1, keepdim=True) * v[:, None, :]
    assert torch.allclose(out, exp, atol=1e-6)
    assert torch.equal(out[:, 0, 1:], h[:, 0, 1:])                            # padding tiles untouched
    assert torch.allclose(hook.pooled, t0.mean(1), atol=1e-6)
    assert cap.value.shape == (B, tiles * R, D)                               # capture flattened to (B, T, D)
    # unequal masks fall back to the per-element loop with the same result
    mask2 = mask.clone(); mask2[R:2 * R] = True
    with hook:
        hook.arm(TokenLayout(False, masks=[mask, mask2]), v, al)
        out2 = lin(h.clone())
    assert torch.allclose(out2[0], exp[0], atol=1e-6)
    t1 = h[1, 0, :2].reshape(2 * R, D)
    exp1 = t1 + al[1] * t1.norm(dim=-1, keepdim=True) * v[1][None, :]
    assert torch.allclose(out2[1, 0, :2].reshape(2 * R, D), exp1, atol=1e-6)
