"""CheXagent-2 (StanfordAIMI/CheXagent-2-3b): a chest-specialised XraySigLIP ViT-L/16 tower at 512 px ->
per-token resampler -> a Phi-2-shaped 32-layer language model. The only medical checkpoint of the grid whose
vision tower was trained on chest radiographs rather than on natural images.

Consumed boundary (the checkpoint's own remote code, `modeling_visual.CLIPModel.forward`):

    x = self.model(pixel_values, output_hidden_states=True).hidden_states[-1]      # (B, 1024, 1024)
    x = forward_resampler(x) = ln_post(attn_pool(x + pos_embed)) @ proj            # (B, 1024, 2560)

and `modeling_chexagent.CheXagentModel.forward` splices that tensor into the 1024 positions between
`<|img|>` and `<|/img|>` of the prompt (`hidden_states[i][a + 1: b] = images[idx]`), so every one of the
1024 rows is read by the language model at both loci. The resampler is per-token (a two-layer MLP, a
LayerNorm and a linear projection), not a perceiver: it neither pools nor changes the token count.

Primary locus `vis.last`: the output of `model.visual.model.encoder.layers.23`, (B, 1024, 1024). That tensor
IS `hidden_states[-1]`: the encoder's hidden-state tuple is taken before the tower's `post_layernorm`, and
the tower's attention-pooling `head` is computed and discarded, so the resampler consumes the raw last
encoder block. Connector locus `connector`: the output of `model.visual`, (B, 1024, 2560). `forward_resampler`
ends in `x @ self.proj`, a bare `nn.Parameter`, so the `CLIPModel` module itself is the only module boundary
whose output is the tensor the language model consumes.

Three things the checkpoint does differently from every other family of the grid, and what this adapter does
about them:

  remote code   `modeling_chexagent.py` and `modeling_visual.py` assert `transformers.__version__ == "4.40.0"`
                at import time; `modeling_visual.py` imports albumentations, cv2, einops and pyarrow without
                using any of them; and `tokenization_chexagent.py` imports `tensorflow` and
                `transformers.utils.is_tf_available` inside `if TYPE_CHECKING:`. Every transformers symbol the
                three files actually CALL still exists in the campaign's transformers 5.15
                (`_prepare_4d_causal_attention_mask`, `ACT2FN`, `BaseModelOutputWithPast`, ...), and the only
                removed API among them, `DynamicCache.from_legacy_cache`, sits behind `if use_cache:` -- a
                branch the campaign never takes, because every forward runs `use_cache=False` with
                `past_key_values=None`. `remote_code_shim()` therefore stubs the unused imports, restores the
                one deleted symbol and reports the asserted version for the duration of the import. Nothing in
                the pinned snapshot is edited or copied, and every shim is recorded in
                `processing_settings()["remote_code_shim"]`.
  image input   `CheXagentModel.forward` decodes an image PATH out of the prompt tokens and calls
                `visual.encode(paths)`, which re-opens the file and calls `visual.forward` DIRECTLY -- not
                through `__call__` -- so no forward hook on `model.visual` can fire and the connector locus
                would be unhookable. `_install_image_bridge` replaces those two bound methods with the same
                computation on the pixel tensor this adapter builds with the checkpoint's OWN
                `image_transform` (bicubic resize to 512, ToTensor, SigLIP mean/std), invoked through
                `__call__`. The arithmetic is the checkpoint's, the image is the campaign's manifest row, and
                the path in the prompt is inert.
  construction  `CheXagentModel.__init__` loads its vision tower with a nested `AutoModel.from_pretrained`
                and derives 97 buffers that the checkpoint does not store, neither of which survives the
                meta-device context transformers 5.x builds every `from_pretrained` model in. `load()`
                therefore constructs the checkpoint's own class directly, as 4.40 did, and fills it with
                `load_sharded_checkpoint`; see its docstring for what the context breaks and how silently.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import sys
import types

import torch

from ..hooks import TokenLayout
from .base import Adapter, LocusInfo

IMG_TOKEN_SPAN = 1024                       # tokenization_chexagent.IMG_TOKEN_SPAN: the fixed <|img|>..<|/img|> span
SYSTEM_PROMPT = "You are a helpful assistant."          # the checkpoint's own inference recipe (README of the snapshot)
IMAGE_SLOT = "cf-transfer-image"            # inert path string: the visual bridge supplies the pixels
REMOTE_CODE_TF_VERSION = "4.40.0"           # the version the snapshot's remote code asserts
# The two packages transformers' `check_imports` demands for the snapshot's remote code and that the remote code
# never executes against: `albumentations` (imported by modeling_visual.py, used nowhere in it, alongside the
# equally unused cv2 / einops / pyarrow, which the environment happens to have) and `tensorflow` (reached by
# tokenization_chexagent.py only inside `if TYPE_CHECKING:`). Neither has a symbol on any code path this campaign
# runs; installing either to satisfy an unused import would change the environment of every other block.
STUB_MODULES = ("albumentations", "albumentations.pytorch", "tensorflow")
# The one transformers symbol the remote code imports that 5.x no longer has. `is_tf_available` disappeared with
# TensorFlow support; `tokenization_chexagent.py` imports it and calls it only inside `if TYPE_CHECKING:`, which
# never runs. Every other transformers symbol the three files import exists unchanged in 5.15, and the only
# removed API they CALL (`DynamicCache.from_legacy_cache`) sits behind `if use_cache:`, a branch no campaign
# forward takes (`forward_last_logits` passes use_cache=False with past_key_values=None).
MISSING_SYMBOLS = {("transformers.utils", "is_tf_available"): (lambda: False)}
# Config fields that `PretrainedConfig.__init__` defined unconditionally in 4.40 and no longer defines in 5.x when
# the checkpoint's config.json does not carry them. `CheXagentModel.__init__` reads `config.pad_token_id` straight
# into `nn.Embedding(..., padding_idx=...)`; this checkpoint's config.json has no `pad_token_id`, so its value
# under the asserted transformers version was None, and None is what is restored.
CONFIG_DEFAULTS_4_40 = {"pad_token_id": None}


def _restore_4_40_config(cfg, raw: dict) -> list[str]:
    """Give the config the two readings the remote code was written against, and record both.

    `pad_token_id`: `PretrainedConfig.__init__` defined it unconditionally in 4.40; 5.x does not, and
    `CheXagentModel.__init__` reads it straight into `nn.Embedding(..., padding_idx=...)`. This checkpoint's
    config.json carries no `pad_token_id`, so its 4.40 value was None.

    `rope_scaling`: in 5.x it is a property over `rope_parameters`, and a config whose JSON says
    `"rope_scaling": null` gets a SYNTHESISED `{"rope_theta": ..., "partial_rotary_factor": ...,
    "rope_type": "default"}`. `PhiAttention._init_rope` branches on `if self.config.rope_scaling is None`
    and reads `rope_scaling["type"]` otherwise, so the synthesised dict sends it down the scaled-rope branch
    and raises `KeyError: 'type'`. `rope_type: "default"` IS the unscaled rotary embedding the null asks for,
    so the 4.40 reading is restored by emptying `rope_parameters` -- and only when the checkpoint's own JSON
    stored nothing there, so a revision that really did specify a scaled rope would not be silently unscaled.
    """
    restored = []
    for k, v in CONFIG_DEFAULTS_4_40.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
            restored.append(f"{k}={v!r} (absent in transformers 5.x, defined as None by 4.40)")
    if raw.get("rope_scaling") is None and raw.get("rope_parameters") is None and cfg.rope_scaling is not None:
        synthesised = dict(cfg.rope_scaling)
        cfg.rope_parameters = None
        restored.append(f"rope_scaling=None (config.json says null; 5.x synthesised {synthesised})")
    return restored


@contextlib.contextmanager
def remote_code_shim():
    """Import the snapshot's remote code under the campaign's transformers without editing the snapshot.

    Three shims, each narrow, each undone on exit, each recorded in `processing_settings()`:
    empty modules for the imports the remote code declares and never uses (`STUB_MODULES`, which
    transformers' own `check_imports` demands before it will load the file), the one deleted transformers
    symbol (`MISSING_SYMBOLS`), and `transformers.__version__` reported as the asserted 4.40.0 for the
    duration of the import. The pinned snapshot is read as it is and nothing is copied or edited.
    """
    import transformers

    created = []
    for name in STUB_MODULES:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []                       # package, so `from albumentations.pytorch import ...` resolves
            # a real ModuleSpec, because torch._dynamo's trace rules call importlib.util.find_spec on every
            # name in sys.modules and raise on `__spec__ is None`
            mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
            mod.ToTensorV2 = None
            sys.modules[name] = mod
            created.append(name)
    if "albumentations" in sys.modules and "albumentations.pytorch" in sys.modules:
        sys.modules["albumentations"].pytorch = sys.modules["albumentations.pytorch"]
    patched = []
    for (mod_name, attr), value in MISSING_SYMBOLS.items():
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, attr):
            setattr(mod, attr, value)
            patched.append(f"{mod_name}.{attr}")
    installed = transformers.__version__
    transformers.__version__ = REMOTE_CODE_TF_VERSION
    try:
        yield {"stubbed_modules": created, "patched_symbols": patched,
               "transformers_version_installed": installed,
               "transformers_version_reported_during_import": REMOTE_CODE_TF_VERSION}
    finally:
        transformers.__version__ = installed
        for name in created:
            sys.modules.pop(name, None)
        for full in patched:
            mod_name, attr = full.rsplit(".", 1)
            delattr(importlib.import_module(mod_name), attr)


@contextlib.contextmanager
def default_dtype(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def load_sharded_checkpoint(model, path, index_name="model.safetensors.index.json") -> dict:
    """Fill `model` from the snapshot's safetensors shards, one shard at a time, and account for every key.

    `nn.Module.load_state_dict` copies in place and casts, so the model's own dtype (set at construction)
    survives and only one shard is resident on top of it. The return value names every key of the checkpoint
    that the model did not want and every key of the model the checkpoint did not carry; `load()` refuses
    both, so no parameter can keep the value it was constructed with.
    """
    from safetensors.torch import load_file

    index = json.loads((path / index_name).read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    expected, loaded, unexpected = set(model.state_dict()), set(), []
    for shard in shards:
        sd = load_file(str(path / shard))
        info = model.load_state_dict(sd, strict=False)
        unexpected += list(info.unexpected_keys)
        loaded |= set(sd) - set(info.unexpected_keys)
        del sd
    return {"shards": shards, "checkpoint_tensors": len(index["weight_map"]),
            "missing_keys": sorted(expected - loaded), "unexpected_keys": sorted(unexpected)}


class CheXagentAdapter(Adapter):
    family = "chexagent-2"
    default_processor_kwargs: dict = {}
    # `PHI_ATTENTION_CLASSES` of the remote code offers eager and flash_attention_2 only, and the class does not
    # declare SDPA support, so the campaign's default "sdpa" is not available for this family.
    attn_implementation = "eager"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._pixels: torch.Tensor | None = None
        self._shim: dict = {}

    # ------------------------------------------------------------------ loading
    def load(self, device_map="cuda:0", dtype=torch.bfloat16):
        """Construct the checkpoint's own class on CPU and then fill it from the shards, instead of calling
        `from_pretrained`.

        `from_pretrained` runs every construction under a meta-device context in transformers 5.x, and this
        checkpoint does three things in `__init__` that the context breaks, two of them silently:

          1. it builds its vision tower with a NESTED `AutoModel.from_pretrained(<XraySigLIP repo>)`, which
             the context refuses outright ("anti-pattern ... wants to load existing weights");
          2. that nested call leaves the default dtype at the tower repo's float32, so `pos_embed`,
             `attn_pool`, `ln_post` and `proj` -- everything `CLIPModel.__init__` builds after it -- come
             back in float32 while the rest of the model is bf16;
          3. the 97 buffers the remote code DERIVES in `__init__` and never stores in the checkpoint (the
             32 rotary embeddings' `inv_freq` / `cos_cached` / `sin_cached`, and the tower's `position_ids`)
             are materialised as uninitialised memory, because they are not in the state dict and
             `_init_weights` does not touch buffers. Measured on this checkpoint: `inv_freq` came back as
             2.4e+38 / 0.0 / 5.3e-33 per layer and every `cos_cached` was non-finite, which makes the first
             decoder layer emit NaN and the whole logit tensor NaN -- in float32 as well, so it does not look
             like an overflow.

        Building the class directly is what transformers 4.40 did, which is the version the remote code
        asserts: `__init__` runs for real, so the tower loads its own weights (immediately overwritten by the
        checkpoint's 400 `model.visual.model.*` tensors), the derived buffers are computed by the
        checkpoint's own code, and `load_sharded_checkpoint` then accounts for every tensor in both
        directions.
        """
        from transformers import AutoConfig, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from .base import HF_HOME  # noqa: F401  (documents where the snapshot is resolved from)

        path = self.resolve_local_path()
        with remote_code_shim() as shim:
            self._shim = shim
            self.processor = AutoTokenizer.from_pretrained(path, trust_remote_code=True, **self.processor_kwargs)
            cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
            raw_cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
            shim["config_defaults_restored"] = _restore_4_40_config(cfg, raw_cfg)
            cfg._attn_implementation = self.attn_implementation
            cls = get_class_from_dynamic_module(cfg.auto_map["AutoModelForCausalLM"], str(path))
            with torch.device("cpu"), default_dtype(dtype):
                self.model = cls(cfg)
            info = load_sharded_checkpoint(self.model, path)
        if info["missing_keys"] or info["unexpected_keys"]:
            raise RuntimeError(f"{self.model_key}: checkpoint does not match the model -- "
                               f"{len(info['missing_keys'])} missing (e.g. {info['missing_keys'][:3]}), "
                               f"{len(info['unexpected_keys'])} unexpected (e.g. {info['unexpected_keys'][:3]})")
        shim["checkpoint_load"] = info
        tok = self.tokenizer
        tok.padding_side = "left"
        self.device_map_requested = device_map
        self.device = "cuda:0" if device_map in ("auto", None) else device_map
        # the nested tower load brings its own float32 weights; one cast puts the whole model in the
        # campaign's dtype, and the check refuses a model that is still mixed
        self._dtypes_after_construction = sorted({str(p.dtype) for p in self.model.parameters()})
        self.model.to(device=self.device, dtype=dtype)
        dtypes = {str(p.dtype) for p in self.model.parameters()}
        if dtypes != {str(dtype)}:
            raise RuntimeError(f"{self.model_key}: mixed parameter dtypes after the cast: {sorted(dtypes)}")
        bad = [n for n, b in self.model.named_buffers() if b.is_floating_point() and not torch.isfinite(b).all()]
        if bad:
            raise RuntimeError(f"{self.model_key}: {len(bad)} non-finite buffers after loading, e.g. {bad[:3]}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._modules = dict(self.model.named_modules())
        from ..scoring import candidate_sets
        self._cands = candidate_sets(tok)
        self._install_fp32_head()
        self.after_load()
        return self

    def after_load(self) -> None:
        visual = self.module("model.visual")                      # modeling_visual.CLIPModel: tower + resampler
        tower = self.module("model.visual.model")                 # SiglipVisionTransformer
        self.n_tower_layers = len(tower.encoder.layers)
        self._block_path = f"model.visual.model.encoder.layers.{self.n_tower_layers - 1}"
        self._connector_path = "model.visual"
        self.module(self._block_path)
        vc = tower.config
        vcfg = dict(self.model.config.visual)
        self.image_size = int(vcfg["image_size"])
        self.patch_size = int(vc.patch_size)
        self.n_patches = (self.image_size // self.patch_size) ** 2
        self.tower_dim = int(vc.hidden_size)
        self.connector_dim = int(vcfg["output_dim"])
        self.vision_model_id = vcfg["vision_model_name_or_path"]
        if int(vc.image_size) != self.image_size:
            raise RuntimeError(f"{self.model_key}: tower config image_size {vc.image_size} != visual {self.image_size}")
        if self.n_patches != IMG_TOKEN_SPAN:
            raise RuntimeError(f"{self.model_key}: {self.n_patches} patch tokens != <|img|> span {IMG_TOKEN_SPAN}")
        if self.connector_dim != int(self.model.config.hidden_size):
            raise RuntimeError(f"{self.model_key}: resampler output {self.connector_dim} != language model width "
                               f"{self.model.config.hidden_size}")
        tok = self.tokenizer
        self.img_start_id, self.img_end_id, self.img_pad_id = tok.img_start_id, tok.img_end_id, tok.img_pad_id
        self._transform = visual.image_transform
        self._install_image_bridge(visual)

    def _install_image_bridge(self, visual) -> None:
        """Replace the checkpoint's read-the-image-from-a-path step with the campaign's pixel batch.

        `CheXagentModel.forward` reconstructs a file path from the prompt tokens
        (`self.tokenizer.decode(image)`) and hands it to `visual.encode(paths)`, which re-opens the file with
        PIL, applies `self.image_transform` and calls `self.forward(images)` DIRECTLY -- not through
        `__call__`. Two consequences: no forward hook on `model.visual` can fire, so the connector locus would
        be unobservable; and the image would be read from disk rather than from the campaign's manifest row.
        Two bound methods are replaced, on this model instance only:

          tokenizer.decode  returns the inert `IMAGE_SLOT` the prompt carries. (It is also the one method of
                            `CheXagentTokenizer` that 4.40 semantics leave broken under transformers 5.x: its
                            `_decode` forwards four positional arguments to a `_decode` that now takes two.
                            No campaign code path calls it; this one is the model's own.)
          visual.encode     runs the checkpoint's own `CLIPModel.forward` on the staged pixel batch, through
                            `__call__`. The transform is the checkpoint's own, applied in `encode()` to the
                            campaign's already-RGB manifest image, so the arithmetic is unchanged.
        """
        ad = self

        def encode(image_paths, training=False):
            if ad._pixels is None:
                raise RuntimeError(f"{ad.model_key}: no pixel batch staged for this forward")
            if ad._pixels.shape[0] != len(image_paths):
                raise RuntimeError(f"{ad.model_key}: {ad._pixels.shape[0]} images staged for "
                                   f"{len(image_paths)} image spans in the prompt")
            return ad._resampler_forward()

        visual.encode = encode
        self.module("model").tokenizer.decode = lambda token_ids, *a, **kw: IMAGE_SLOT

    def _resampler_forward(self) -> torch.Tensor:
        visual = self.module(self._connector_path)
        p = next(visual.parameters())
        return visual(self._pixels.to(device=p.device, dtype=p.dtype))

    # ------------------------------------------------------------------ prompts
    def prompt_text(self, question: str, with_image: bool = True) -> str:
        """The checkpoint's own inference recipe: `tokenizer.from_list_format` renders one image as
        `Picture 1:<|img|>{path}<|/img|>\\n`, and the conversation template wraps system and user turns.
        The path is the inert `IMAGE_SLOT` because the visual bridge supplies the pixels; it keeps the
        prompt byte-identical for every row, so a question's token length never depends on the image."""
        tok = self.tokenizer
        query = (f"Picture 1:{tok.image_start_tag}{IMAGE_SLOT}{tok.image_end_tag}\n" if with_image else "") + question
        conv = [{"from": "system", "value": SYSTEM_PROMPT}, {"from": "human", "value": query}]
        return tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    def encode(self, images, questions):
        """Tokenise with the checkpoint's own tokenizer (which expands `<|img|>..<|/img|>` to the fixed
        1024-token span) and build the pixel batch with the checkpoint's own `image_transform`."""
        tok = self.tokenizer
        texts = [self.prompt_text(q, with_image=images is not None) for q in questions]
        enc = tok(texts, return_tensors="pt", padding=True)
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if images is not None:
            out["pixel_values"] = torch.stack([self._transform(im) for im in images])
        return out

    def to_device(self, enc):
        """`CheXagentForCausalLM.forward` takes no pixel argument, so the pixel batch is staged for the visual
        bridge and removed from the keyword arguments of the model call."""
        dev = next(self.model.parameters()).device
        self._pixels = None
        out = {}
        for k, v in enc.items():
            t = v.to(dev) if torch.is_tensor(v) else v
            if k == "pixel_values":
                self._pixels = t
            else:
                out[k] = t
        return out

    # ------------------------------------------------------------------ loci
    def loci(self) -> dict[str, LocusInfo]:
        return {
            "vis.last": LocusInfo(
                "vis.last", self._block_path,
                f"tensor output of XraySigLIP encoder layer {self.n_tower_layers - 1} of {self.n_tower_layers} "
                f"(B, {self.n_patches}, {self.tower_dim}); this is the `hidden_states[-1]` that "
                f"CLIPModel.forward passes to forward_resampler",
                self.tower_dim,
                f"all {self.n_patches} patch tokens of the {self.image_size}x{self.image_size} image "
                f"(SigLIP has no CLS token, and the resampler reads every row)",
                "the single visual stream into the resampler",
                "none on the consumed path: the tower's post_layernorm and its attention-pooling head are "
                "computed from this tensor and discarded, because the resampler reads the encoder's "
                "hidden-state tuple rather than the tower's last_hidden_state",
                self._connector_path, True,
                f"tower {self.vision_model_id} (ViT-L/16 at {self.image_size} px, {self.patch_size} px patches), "
                f"trained on chest radiographs"),
            "connector": LocusInfo(
                "connector", self._connector_path,
                f"tensor output of modeling_visual.CLIPModel (the resampler: + 2-D sincos pos_embed, attn_pool "
                f"MLP, ln_post, @ proj), (B, {self.n_patches}, {self.connector_dim})",
                self.connector_dim,
                f"all {self.n_patches} resampled tokens of the image; the resampler is per-token and pools nothing",
                f"the image embeddings spliced into the {IMG_TOKEN_SPAN} positions between <|img|> and <|/img|>",
                "none",
                "language model input embeddings", True,
                "hooked on the CLIPModel module itself because forward_resampler ends in `x @ self.proj`, a bare "
                "Parameter, so no child module's output is the consumed tensor"),
        }

    def image_token_mask(self, enc) -> torch.Tensor:
        """(B, L) bool: the 1024 positions strictly between `<|img|>` and `<|/img|>`, which are the positions
        `CheXagentModel.forward` overwrites with the resampler output."""
        ids = enc["input_ids"]
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for b in range(ids.shape[0]):
            a, e = self._image_span(ids[b])
            mask[b, a + 1:e] = True
        return mask

    def _image_span(self, row: torch.Tensor) -> tuple[int, int]:
        start = (row == self.img_start_id).nonzero().flatten().tolist()
        end = (row == self.img_end_id).nonzero().flatten().tolist()
        if len(start) != 1 or len(end) != 1:
            raise RuntimeError(f"{self.model_key}: {len(start)} <|img|> / {len(end)} <|/img|> tokens in one prompt")
        return int(start[0]), int(end[0])

    def layouts(self, enc, images) -> dict[str, TokenLayout]:
        B = len(images)
        ids = enc["input_ids"]
        if ids.shape[0] != B:
            raise RuntimeError(f"{self.model_key}: {ids.shape[0]} prompts for {B} images")
        pv = enc.get("pixel_values")
        if pv is not None and pv.shape[0] != B:
            raise RuntimeError(f"{self.model_key}: {pv.shape[0]} pixel tensors for {B} images")
        for b in range(B):
            a, e = self._image_span(ids[b])
            if e - a - 1 != self.n_patches:
                raise RuntimeError(f"{self.model_key}: image span {e - a - 1} != {self.n_patches} resampled tokens")
        mask = torch.ones(self.n_patches, dtype=torch.bool)
        return {"vis.last": TokenLayout(False, masks=[mask.clone() for _ in range(B)]),
                "connector": TokenLayout(False, masks=[mask.clone() for _ in range(B)])}

    @torch.no_grad()
    def vision_features(self, enc):
        """Tower and resampler only; both loci are prompt-independent."""
        self.to_device(enc)                       # stages the pixel batch for _resampler_forward
        if self._pixels is None:
            raise RuntimeError(f"{self.model_key}: vision_features called without pixel_values")
        return self._resampler_forward()

    # ------------------------------------------------------------------ metadata
    def processing_settings(self) -> dict:
        s = super().processing_settings()
        ip = getattr(self._transform, "transforms", [])
        s["image_processing"] = {
            "source": "modeling_visual.CLIPModel.image_transform (the checkpoint's own transform)",
            "steps": [repr(t) for t in ip],
            "image_size": self.image_size, "patch_size": self.patch_size, "patch_tokens": self.n_patches,
            "vision_model_name_or_path": self.vision_model_id,
        }
        s["prompt_construction"] = (
            "the checkpoint's own recipe: tokenizer.from_list_format renders the image as "
            f"'Picture 1:<|img|>{IMAGE_SLOT}<|/img|>\\n' before the question, the conversation is "
            f"[system: '{SYSTEM_PROMPT}', human: <query>] and the chat template adds the assistant header; "
            f"the tokenizer expands the image tag to a fixed {IMG_TOKEN_SPAN}-token span"
        )
        s["special_token_ids"].update({"img_start_id": self.img_start_id, "img_end_id": self.img_end_id,
                                       "img_pad_id": self.img_pad_id, "img_token_span": IMG_TOKEN_SPAN})
        s["remote_code_shim"] = self._shim
        s["image_bridge"] = (
            "CheXagentModel.forward reads an image PATH out of the prompt: its tokenizer.decode is replaced with "
            "the inert prompt slot and CLIPModel.encode with the same forward on the staged pixel batch, called "
            "through __call__ so the connector locus is hookable; the transform and the forward are the "
            "checkpoint's own"
        )
        s["device_map"] = {"requested": getattr(self, "device_map_requested", None),
                           "device": getattr(self, "device", None),
                           "parameter_dtypes_after_construction": getattr(self, "_dtypes_after_construction", None),
                           "parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters()})}
        return s
