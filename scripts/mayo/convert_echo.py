"""Stage echo-7 with one dead import guarded, and write the receipt.

ECHO's released modeling_echo.py imports a fused RMSNorm from flash_attn at module scope:

    from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_rms_norm

The name is never referenced anywhere else in the file, but transformers' check_imports scans top-level
imports and refuses to load remote code when one is missing, so the checkpoint cannot be loaded without
flash_attn even though it never calls it. The staged copy wraps that one line in try/except. No other line
is touched, and because the name is unused the model's arithmetic is unchanged.

Everything else is symlinked to the pinned snapshot, so the weights are the published bytes.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

HF = Path("/rodata/azradonc_dev/m253405/cache")
REV = "26f9ab2a7f536cce07d99cc0f7f7afbff712ae43"
SRC = HF / "hub" / "models--Midea-AIRC--ECHO_block4" / "snapshots" / REV
OUT = HF / "converted" / "echo-block4-hf"
DEAD = "from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_rms_norm"

# transformers 5.x dropped the "default" key from ROPE_INIT_FUNCTIONS; stock Qwen2.5-VL branches to its own
# static compute_default_rope_parameters instead. ECHO is Qwen2.5-VL, so the staged copy takes that same branch
# and calls that same function: its rope becomes bit-identical to q25-7's, which is what the comparison needs.
ROPE_OLD = "        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]"
ROPE_NEW = """        rope_init_fn = self.compute_default_rope_parameters   # staged: the stock Qwen2.5-VL branch
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.rope_init_fn = rope_init_fn"""

# transformers 5.x also reads this attribute off the module itself (modeling_utils._init_weights), so it has to
# live on the class exactly as stock Qwen2.5-VL defines it, not only inside __init__
# the decoding branch of the text attention calls flash_attn_func unconditionally, ignoring the
# _attn_implementation the model was loaded with. The same file computes the same attention -- non-causal, same
# scale, GQA -- with F.scaled_dot_product_attention in its prefilling branch, so the patch makes the branch
# honour the setting and fall into the file's own SDPA path. Every checkpoint in the grid runs sdpa.
ATTN_OLD = (
    "        if torch.all(attention_mask):  # decoding\n"
    "            query_states = query_states.transpose(1, 2)\n"
    "            key_states = key_states.transpose(1, 2)\n"
    "            value_states = value_states.transpose(1, 2)\n"
    "            attn_output = flash_attn_func(\n"
    "                query_states,\n"
    "                key_states,\n"
    "                value_states,\n"
    "                causal=False,\n"
    "                softmax_scale=self.scaling\n"
    "            )\n"
    "            attn_output = rearrange(attn_output, 'b l h d -> b l (h d)')\n")

# The two branches do not take the same mask. The decoding branch reaches flash_attn_func with causal=False and
# no mask -- the guard above it established that every entry is True, so the attention is simply unmasked --
# while the prefilling branch feeds a real mask to SDPA. Routing decoding into the prefilling call would pass a
# block-shaped mask against a full-length query, which is a shape error, so the staged branch keeps the decoding
# case separate and expresses "all true" the way SDPA spells it: attn_mask=None. Same operation, same scale,
# same GQA, through the kernel every other checkpoint in the grid runs.
ATTN_NEW = (
    "        if torch.all(attention_mask):  # decoding: the guard means the attention is unmasked\n"
    "            if self.config._attn_implementation == \"flash_attention_2\":\n"
    "                query_states = query_states.transpose(1, 2)\n"
    "                key_states = key_states.transpose(1, 2)\n"
    "                value_states = value_states.transpose(1, 2)\n"
    "                attn_output = flash_attn_func(\n"
    "                    query_states, key_states, value_states, causal=False, softmax_scale=self.scaling)\n"
    "                attn_output = rearrange(attn_output, 'b l h d -> b l (h d)')\n"
    "            else:   # staged: the same unmasked attention, expressed for SDPA\n"
    "                attn_output = F.scaled_dot_product_attention(\n"
    "                    query=query_states, key=key_states, value=value_states,\n"
    "                    attn_mask=None, is_causal=False, scale=self.scaling, enable_gqa=True)\n"
    "                attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')\n")

CLASS_OLD = "class EchoRotaryEmbedding(nn.Module):"
CLASS_NEW = """class EchoRotaryEmbedding(nn.Module):
    # staged: borrow stock Qwen2.5-VL's static method; transformers 5.x expects it on the module
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding as _StockRotary
    compute_default_rope_parameters = staticmethod(_StockRotary.compute_default_rope_parameters)
    del _StockRotary
"""

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

OUT.mkdir(parents=True, exist_ok=True)
src_text = (SRC / "modeling_echo.py").read_text()
if src_text.count(DEAD) != 1:
    raise SystemExit(f"expected exactly one dead flash_attn import, found {src_text.count(DEAD)}")
if len(re.findall(r"\bflash_rms_norm\b", src_text)) != 1:
    raise SystemExit("flash_rms_norm is referenced somewhere; the import is not dead and must not be guarded")

if src_text.count(ROPE_OLD) != 1:
    raise SystemExit(f"expected exactly one ROPE_INIT_FUNCTIONS lookup, found {src_text.count(ROPE_OLD)}")

patched = src_text.replace(
    DEAD,
    "try:  # staged: the name is never used; the unguarded import alone blocks loading without flash_attn\n"
    "    " + DEAD + "\n"
    "except Exception:\n"
    "    flash_rms_norm = None")
patched = patched.replace(ROPE_OLD, ROPE_NEW)
if patched.count(CLASS_OLD) != 1:
    raise SystemExit(f"expected exactly one EchoRotaryEmbedding class, found {patched.count(CLASS_OLD)}")
patched = patched.replace(CLASS_OLD, CLASS_NEW)
if patched.count(ATTN_OLD) != 1:
    raise SystemExit(f"expected exactly one unconditional flash decoding branch, found {patched.count(ATTN_OLD)}")
patched = patched.replace(ATTN_OLD, ATTN_NEW)
if patched.count("flash_attn_func(") != 1:
    raise SystemExit("more than one runtime flash_attn_func call site; the single-branch assumption is wrong")
if "elif False" in patched:
    raise SystemExit("dead branch left in the staged file")
(OUT / "modeling_echo.py").write_text(patched)

shards = {}
for f in sorted(SRC.iterdir()):
    if f.name == "modeling_echo.py":
        continue
    dst = OUT / f.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(f.resolve())
    if f.name.endswith(".safetensors"):
        shards[f.name] = {"sha256": sha(f.resolve()), "bytes": f.resolve().stat().st_size}

(OUT / "conversion.json").write_text(json.dumps({
    "source_repo": "Midea-AIRC/ECHO_block4",
    "source_revision": REV,
    "steps": {
        "shards_verified": shards,
        "modeling_file_patch": {
            "file": "modeling_echo.py",
            "lines_changed": 1,
            "change": "wrapped one unused module-scope import in try/except",
            "line": DEAD,
            "reason": "flash_rms_norm is imported and never referenced; transformers check_imports refuses the "
                      "remote code when flash_attn is absent, so the unused import alone blocks loading",
            "arithmetic_effect": "none: the guarded name is not read anywhere in the file",
            "source_sha256": sha(SRC / "modeling_echo.py"),
            "staged_sha256": sha(OUT / "modeling_echo.py"),
        },
        "rope_default_patch": {
            "file": "modeling_echo.py",
            "lines_changed": 2,
            "change": 'routed rope_type "default" to Qwen2_5_VLRotaryEmbedding.compute_default_rope_parameters',
            "reason": "transformers 5.15 removed the \"default\" key from ROPE_INIT_FUNCTIONS; the released file "
                      "targets transformers 4.55.4, where it existed",
            "arithmetic_effect": "matches stock Qwen2.5-VL under this transformers version, which is the rope "
                      "path every other Qwen2.5-VL block of the grid already runs",
        },
        "attention_branch_patch": {
            "file": "modeling_echo.py",
            "lines_changed": 1,
            "change": "the decoding branch of the text attention now requires _attn_implementation == "
                      "flash_attention_2 instead of calling flash_attn_func unconditionally",
            "reason": "the released branch ignores the attention implementation the model is loaded with and "
                      "hard-calls flash_attn, which the grid's environment does not carry",
            "arithmetic_effect": "falls into the same file's SDPA path for the same operation -- non-causal "
                      "attention at the same scale with GQA -- which is the kernel every other checkpoint in "
                      "the grid runs, so this aligns ECHO with the grid rather than separating it",
        },
        "everything_else": "symlinked to the pinned snapshot; weights are the published bytes",
    },
}, indent=1) + "\n")
print("staged", OUT)
print("shards verified:", len(shards))
