"""
Minimal working example: plain torch forward hooks on a HuggingFace VLM.

Proves the primitives needed for:
  D2 (concept survival curve): read hidden states at vision-encoder output,
      connector output, and every LM layer, at chosen token positions.
  D1 (steering):               add alpha * v to a residual stream at a locus
      and show the output logits move.

Uses ONLY torch.nn.Module.register_forward_hook. No interpretability library.

Run:
  python src/hook_demo.py --model qwen3b
  python src/hook_demo.py --model llava7b
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image

CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
CACHE_ALT = "${HF_HOME:-$HOME/.cache/huggingface}/hub"

MODELS = {
    "qwen3b":  ("Qwen/Qwen2.5-VL-3B-Instruct", CACHE),
    "qwen7b":  ("Qwen/Qwen2.5-VL-7B-Instruct", CACHE),
    "llava7b": ("llava-hf/llava-1.5-7b-hf",     CACHE_ALT),
    "qwen32b": ("Qwen/Qwen2.5-VL-32B-Instruct", CACHE),
}


def pick_gpu(need_gb=20):
    """Pick the GPU with the most free memory. The box is shared; do not assume idle."""
    best, best_free = None, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        free_gb = free / 1e9
        if free_gb > best_free:
            best, best_free = i, free_gb
    print(f"[gpu] selected cuda:{best} with {best_free:.1f} GB free")
    if best_free < need_gb:
        raise RuntimeError(f"no GPU with {need_gb} GB free; best was {best_free:.1f} GB")
    return best


def make_image():
    """Deterministic synthetic grayscale image. No network, reproducible."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:336, 0:336]
    base = 90 + 50 * np.exp(-(((x - 168) ** 2 + (y - 190) ** 2) / (2 * 90.0 ** 2)))
    base += 25 * np.sin(y / 22.0)                      # rib-like banding
    base[140:210, 120:220] += 45                       # bright blob
    base += rng.normal(0, 6, base.shape)
    arr = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def unwrap(out):
    """Module outputs come in three shapes across HF VLMs:
    a bare Tensor (Qwen vision blocks, decoder layers in transformers 5.x),
    a tuple whose [0] is the hidden state (older decoder layers),
    a ModelOutput with .last_hidden_state (LLaVA's CLIP vision_tower).
    Normalize all three."""
    if isinstance(out, torch.Tensor):
        return out, "tensor"
    if isinstance(out, tuple):
        return out[0], "tuple"
    lhs = getattr(out, "last_hidden_state", None)
    if isinstance(lhs, torch.Tensor):
        return lhs, "modeloutput"
    return None, "unknown"


def rewrap(new_t, out, kind):
    if kind == "tensor":
        return new_t
    if kind == "tuple":
        return (new_t,) + tuple(out[1:])
    if kind == "modeloutput":
        out.last_hidden_state = new_t
        return out
    raise TypeError(f"cannot rewrap output of kind {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3b", choices=list(MODELS))
    ap.add_argument("--layer", type=int, default=None, help="LM layer to steer (default: middle)")
    ap.add_argument("--alpha", type=float, default=30.0)
    args = ap.parse_args()

    model_id, cache = MODELS[args.model]
    dev = f"cuda:{pick_gpu()}"
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"\n=== loading {model_id} ===")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, cache_dir=cache, device_map=dev)
    model.eval()
    torch.cuda.synchronize()
    load_s = time.time() - t0
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[load] {load_s:.1f} s | params {n_param/1e9:.2f} B | type {type(model).__name__}")
    print(f"[load] weights VRAM {torch.cuda.memory_allocated(dev)/1e9:.2f} GB")

    # ---------------- (a) one image + prompt -> forward pass ----------------
    img = make_image()
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Is there an opacity in the left lung field? Answer yes or no."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = proc(images=[img], text=[text], return_tensors="pt").to(dev)
    ids = inputs["input_ids"]
    print(f"\n[input] input_ids {tuple(ids.shape)}  keys={sorted(inputs.keys())}")

    # locate the visual token positions in the LM sequence (needed by D2)
    cfg = model.config
    img_tok = None
    for attr in ("image_token_id", "image_token_index"):
        if getattr(cfg, attr, None) is not None:
            img_tok = getattr(cfg, attr); break
    vis_pos = (ids[0] == img_tok).nonzero().flatten()
    print(f"[input] image_token_id={img_tok}  n_visual_tokens={vis_pos.numel()}  "
          f"span=[{vis_pos.min().item()}..{vis_pos.max().item()}]  answer_pos={ids.shape[1]-1}")

    # ---------------- (b,c) capture hidden states at named modules ----------------
    named = dict(model.named_modules())
    if args.model.startswith("qwen"):
        n_lm = len(model.model.language_model.layers)
        n_vis = len(model.model.visual.blocks)
        loci = [f"model.visual.blocks.{n_vis-1}", "model.visual.merger",
                "model.language_model.layers.0",
                f"model.language_model.layers.{n_lm//2}",
                f"model.language_model.layers.{n_lm-1}"]
    else:
        n_lm = len(model.model.language_model.layers)
        loci = ["model.vision_tower", "model.multi_modal_projector",
                "model.language_model.layers.0",
                f"model.language_model.layers.{n_lm//2}",
                f"model.language_model.layers.{n_lm-1}"]
    for L in loci:
        assert L in named, f"module not found: {L}"

    caught = {}

    def mk_reader(name):
        def hook(mod, inp, out):
            t, _ = unwrap(out)
            if isinstance(t, torch.Tensor):
                caught[name] = t.detach()
        return hook

    handles = [named[L].register_forward_hook(mk_reader(L)) for L in loci]

    torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        base_out = model(**inputs)          # warmup 1 (cuBLAS/kernel autotune)
        model(**inputs)                     # warmup 2
        torch.cuda.synchronize()
        reps, t0 = 5, time.time()
        for _ in range(reps):
            model(**inputs)
        torch.cuda.synchronize()
        fwd_s = (time.time() - t0) / reps
    peak_gb = torch.cuda.max_memory_allocated(dev) / 1e9
    for h in handles:
        h.remove()

    print(f"\n=== (c) captured hidden states ===")
    for L in loci:
        t = caught[L]
        print(f"  {L:<42} shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}")
    print(f"\n[cost] forward {fwd_s*1000:.0f} ms (mean of 5, post-warmup) | peak VRAM {peak_gb:.2f} GB | logits {tuple(base_out.logits.shape)}")

    base_logits = base_out.logits[0, -1].float().clone()

    # ---------------- (d) steering: add a fixed vector, show logits move ----------------
    L_steer = f"model.language_model.layers.{args.layer if args.layer is not None else n_lm//2}"
    d_model = caught[L_steer].shape[-1]
    resid_rms = caught[L_steer][0, -1].float().norm().item()

    g = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(d_model, generator=g).to(dev, torch.bfloat16)
    v = v / v.float().norm()      # unit vector; a real run uses the probe normal

    def mk_steer(vec, alpha, pos=-1):
        def hook(mod, inp, out):
            t, wt = unwrap(out)
            t = t.clone()
            t[:, pos, :] = t[:, pos, :] + alpha * vec.to(t.dtype)
            return rewrap(t, out, wt)
        return hook

    print(f"\n=== (d) steering at {L_steer} (answer position) ===")
    print(f"  d_model={d_model}  ||resid@answer||={resid_rms:.1f}  steering vector is unit-norm")

    rows = []
    # scale alpha relative to the residual norm at this locus: that is the only
    # scale that means the same thing across layers and models.
    fracs = [0.0, 0.25, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0]
    for alpha in [f * resid_rms for f in fracs]:
        h = named[L_steer].register_forward_hook(mk_steer(v, alpha))
        with torch.no_grad():
            lg = model(**inputs).logits[0, -1].float()
        h.remove()
        d = (lg - base_logits)
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(lg, -1), torch.log_softmax(base_logits, -1),
            log_target=True, reduction="sum").item()
        top_b = base_logits.argmax().item()
        top_s = lg.argmax().item()
        rows.append((alpha, d.abs().max().item(), kl, top_s,
                     proc.tokenizer.decode([top_s]), top_s != top_b))
        print(f"  alpha={alpha:+8.2f} ({alpha/resid_rms:+.2f}x|r|)  max|dlogit|={d.abs().max().item():8.4f}  "
              f"KL={kl:9.5f}  top1={top_s:6d} {proc.tokenizer.decode([top_s])!r:12} "
              f"changed={top_s != top_b}")

    assert rows[0][1] == 0.0, "alpha=0 must be a no-op; hook plumbing is wrong"
    assert rows[3][1] > 0.0, "alpha!=0 produced no logit change; steering did not take effect"
    print("\n[check] alpha=0 is exactly a no-op, and alpha!=0 moves the logits. Hooks work.")

    # ---------------- (e) steering during GENERATION (what D1 actually needs) ----------------
    # With a KV cache the hook fires once per decode step: on prefill seq_len is the
    # full prompt (pos -1 = last prompt token), on each decode step seq_len == 1
    # (pos -1 = the token being produced). So pos=-1 steers every generated position.
    print(f"\n=== (e) steering during generation at {L_steer} ===")
    gen_rows = []
    for frac in [0.0, 0.25, 1.0]:
        alpha = frac * resid_rms
        h = named[L_steer].register_forward_hook(mk_steer(v, alpha))
        with torch.no_grad():
            g_out = model.generate(**inputs, max_new_tokens=12, do_sample=False,
                                   pad_token_id=proc.tokenizer.eos_token_id)
        h.remove()
        txt = proc.tokenizer.decode(g_out[0, ids.shape[1]:], skip_special_tokens=True)
        n_steps = int(g_out.shape[1] - ids.shape[1])
        gen_rows.append((frac, txt))
        print(f"  frac={frac:+.2f} alpha={alpha:+8.2f} steps={n_steps:2d} -> {txt!r}")
    print(f"  [check] generation with alpha=0 vs alpha!=0 differ: "
          f"{gen_rows[0][1] != gen_rows[2][1]}")

    out = dict(model=model_id, load_s=round(load_s, 2), n_param_B=round(n_param/1e9, 3),
               fwd_ms=round(fwd_s*1000, 1), peak_vram_gb=round(peak_gb, 2),
               weights_vram_gb=round(torch.cuda.memory_allocated(dev)/1e9, 2),
               n_visual_tokens=int(vis_pos.numel()), seq_len=int(ids.shape[1]),
               d_model=int(d_model), n_lm_layers=int(n_lm),
               shapes={L: list(caught[L].shape) for L in loci},
               gen={f'{f:+.2f}': t for f, t in gen_rows})
    os.makedirs("runs", exist_ok=True)
    p = f"runs/hook_demo_{args.model}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
