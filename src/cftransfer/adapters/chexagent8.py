"""CheXagent (StanfordAIMI/CheXagent-8b): the checkpoint of the CheXagent paper (arXiv:2401.12208).

Architecture, from the snapshot's own remote code (`modeling_chexagent.CheXagentForConditionalGeneration`),
is BLIP-2 rather than LLaVA and it is NOT the architecture of CheXagent-2-3b, which the campaign already
carries: a 40-layer EVA-CLIP-shaped ViT at 448 px with 14 px patches (hidden 1408), a 12-layer BERT-shaped
Q-Former with 128 learned query tokens that cross-attend to the tower output, a linear projection to the
language width, and a Mistral-7B language model. The visual tokens are CONCATENATED in front of the prompt
embeddings, not scattered into placeholder positions:

    vision_outputs = self.vision_model(pixel_values[image_mask])                 # (B, 1025, 1408)
    query_outputs  = self.qformer(query_embeds=self.query_tokens.expand(B, -1, -1),
                                  encoder_hidden_states=image_embeds,
                                  encoder_attention_mask=image_attention_mask)   # (B, 128, 768)
    input_vis      = self.language_projection(query_outputs[0])                  # (B, 128, 4096)
    inputs_embeds  = torch.cat([input_vis, self.language_model.get_input_embeddings()(input_ids)], dim=1)

Primary locus `vis.last`: the output of `vision_model.encoder.layers.39`, (B, 1025, 1408). Every one of the
1025 rows is consumed -- `image_attention_mask` is all ones, so the Q-Former's cross-attention reads the CLS
token and all 1024 patch tokens. The tower's `post_layernorm` sits BETWEEN this tensor and the Q-Former
(`last_hidden_state = self.post_layernorm(encoder_outputs[0])`), which is the one structural difference from
LLaVA and from CheXagent-2: there the tower's final LayerNorm is computed and discarded, here it is on the
consumed path, so a write at this locus reaches the reader through a LayerNorm. It is still the last block
the connector consumes, which is what the locus is defined to be, and preflight check C measures the reach
that survives.

Connector locus `connector`: the output of `language_projection`, (B, 128, 4096), all 128 tokens consumed.
The Q-Former takes NO text input (`query_embeds` only, no `qformer_input_ids`), so both loci are
prompt-independent exactly as the protocol assumes.

Three things this checkpoint does differently from the campaign's other families:

  remote code   `modeling_chexagent.py` imports `find_pruneable_heads_and_indices` from
                `transformers.pytorch_utils`, which 5.x no longer has. It is used only by
                `CheXagentQFormerAttention.prune_heads`, which no campaign forward calls, so
                `remote_code_shim()` restores it as a raising stub for the duration of the import and
                records the shim in `processing_settings()`. Every other symbol the file imports exists
                unchanged in 5.15, the file asserts no transformers version, and nothing in the pinned
                snapshot is edited or copied.
  forward       `CheXagentForConditionalGeneration.forward` takes neither `use_cache` nor `logits_to_keep`,
                and its `lm_head` lives at `language_model.lm_head`, so both the float32 answer head and the
                forward are overridden here. `use_cache` is turned off on the language model's config
                instead of being passed.
  pixel batch   the checkpoint's own `CheXagentProcessor` returns pixel_values as (1, n_images, 3, 448, 448)
                -- n images of ONE sample. The campaign scores one image per row, so `encode` builds
                (B, 1, 3, 448, 448) with the checkpoint's own BlipImageProcessor and asserts the tower sees
                exactly B images, which is what makes the batched token layout well defined.
"""
from __future__ import annotations

import contextlib
import importlib
import json

import torch

from ..hooks import TokenLayout
from .base import Adapter, LocusInfo

# The checkpoint's own inference recipe (model card "Get Started"): the prompt is wrapped in this exact
# USER / ASSISTANT frame and tokenised with add_special_tokens=True, so the leading BOS is the tokenizer's.
PROMPT_TEMPLATE = " USER: <s>{question} ASSISTANT: <s>"
N_QUERY_TOKENS = 128


def _removed_find_pruneable_heads_and_indices(*args, **kwargs):
    raise NotImplementedError(
        "transformers 5.x removed transformers.pytorch_utils.find_pruneable_heads_and_indices; the snapshot's "
        "remote code imports it only for CheXagentQFormerAttention.prune_heads, which no campaign forward calls")


def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    """`PreTrainedModel.get_head_mask` as transformers 4.35 defined it, for the `head_mask is None` branch.

    `CheXagentQFormerModel.forward` calls it once per forward and nothing in this campaign ever passes a head
    mask, so the 4.35 return for None -- one None per layer -- is restored and the other branch refuses rather
    than guessing at `_convert_head_mask_to_5d`, which 5.x removed as well."""
    if head_mask is not None:
        raise NotImplementedError("transformers 5.x removed PreTrainedModel.get_head_mask; only the head_mask=None "
                                  "path the campaign uses is restored")
    return [None] * num_hidden_layers


# The one transformers symbol the remote code imports that 5.x no longer has.
MISSING_SYMBOLS = {("transformers.pytorch_utils", "find_pruneable_heads_and_indices"):
                   _removed_find_pruneable_heads_and_indices}
# Methods `PreTrainedModel` no longer provides that the remote code calls at RUN time (so an import-time shim
# would not reach them). They are installed on the dynamically-loaded remote class itself, which exists only for
# this checkpoint, so no other family's model class is touched. `get_extended_attention_mask` and
# `invert_attention_mask`, the other two helpers the Q-Former calls, still exist in 5.15.
RESTORED_METHODS = {"get_head_mask": _get_head_mask}


@contextlib.contextmanager
def remote_code_shim():
    """Import the snapshot's remote code under the campaign's transformers without editing the snapshot."""
    patched = []
    for (mod_name, attr), value in MISSING_SYMBOLS.items():
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, attr):
            setattr(mod, attr, value)
            patched.append(f"{mod_name}.{attr}")
    import transformers
    try:
        yield {"patched_symbols": patched, "stubbed_modules": [],
               "transformers_version_installed": transformers.__version__,
               "transformers_version_asserted_by_remote_code": None}
    finally:
        for full in patched:
            mod_name, attr = full.rsplit(".", 1)
            delattr(importlib.import_module(mod_name), attr)


class CheXagent8Adapter(Adapter):
    family = "chexagent-1"
    default_processor_kwargs: dict = {}
    # The remote code's own attention is hand-written and the class declares no SDPA support, so the campaign's
    # default "sdpa" is not available for the vision tower or the Q-Former; "eager" is what the checkpoint runs.
    attn_implementation = "eager"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._shim: dict = {}

    # ------------------------------------------------------------------ loading
    def load(self, device_map="cuda:0", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoProcessor
        path = self.resolve_local_path()
        with remote_code_shim() as shim:
            self._shim = shim
            self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, **self.processor_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code=True, dtype=dtype, device_map=device_map,
                attn_implementation=self.attn_implementation)
        tok = self.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            # the checkpoint's LlamaTokenizer carries no pad token (tokenizer_config.json: "pad_token": null).
            # Every campaign composition is a batch of equal-length rows -- the runner replicates one image and one
            # question, and preflight scores one question over 16 images -- so no row is ever padded; the choice is
            # the standard EOS and is recorded in processing_settings().
            tok.pad_token = tok.eos_token
            self._pad_token_set = f"pad_token set to eos_token ({tok.eos_token!r}, id {tok.eos_token_id})"
        else:
            self._pad_token_set = None
        self.device_map_requested = device_map
        dtypes = {str(p.dtype) for p in self.model.parameters()}
        if dtypes != {str(dtype)}:
            raise RuntimeError(f"{self.model_key}: mixed parameter dtypes after loading: {sorted(dtypes)}")
        bad = [n for n, b in self.model.named_buffers() if b.is_floating_point() and not torch.isfinite(b).all()]
        if bad:
            raise RuntimeError(f"{self.model_key}: {len(bad)} non-finite buffers after loading, e.g. {bad[:3]}")
        qcls = type(self.model.qformer)
        restored = []
        for name, fn in RESTORED_METHODS.items():
            if not hasattr(qcls, name):
                setattr(qcls, name, fn)
                restored.append(f"{qcls.__name__}.{name}")
        self._shim["restored_methods"] = restored
        # the forward takes no use_cache argument; turning it off on the language model's config keeps every
        # campaign forward cache-free, as every other family's does
        self.model.language_model.config.use_cache = False
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._modules = dict(self.model.named_modules())
        from ..scoring import candidate_sets
        self._cands = candidate_sets(tok)
        self._install_fp32_head()
        self.after_load()
        return self

    # ------------------------------------------------------------------ float32 answer logits
    def _install_fp32_head(self) -> None:
        """Same construction as the base adapter, on `language_model.lm_head` (this model has no top-level head)."""
        head = self.model.language_model.lm_head
        self._head_input = None

        def grab(_m, inputs, _out):
            self._head_input = inputs[0]
        head.register_forward_hook(grab)
        self._head_w32 = head.weight.detach().float()
        self._head_b32 = head.bias.detach().float() if getattr(head, "bias", None) is not None else None
        self._softcap = getattr(self.model.config.text_config, "final_logit_softcapping", None)

    def after_load(self) -> None:
        vcfg = self.model.config.vision_config
        tower = self.module("vision_model")
        self.n_tower_layers = len(tower.encoder.layers)
        self._block_path = f"vision_model.encoder.layers.{self.n_tower_layers - 1}"
        self._connector_path = "language_projection"
        self.module(self._block_path); self.module(self._connector_path)
        self.image_size = int(vcfg.image_size)
        self.patch_size = int(vcfg.patch_size)
        self.n_patches = (self.image_size // self.patch_size) ** 2
        self.n_vision_tokens = self.n_patches + 1                    # CLS + patches, all cross-attended
        self.tower_dim = int(vcfg.hidden_size)
        self.n_query_tokens = int(self.model.config.num_query_tokens)
        self.connector_dim = int(self.model.config.text_config.hidden_size)
        if self.n_query_tokens != N_QUERY_TOKENS:
            raise RuntimeError(f"{self.model_key}: {self.n_query_tokens} query tokens, expected {N_QUERY_TOKENS}")
        if int(self.module(self._connector_path).out_features) != self.connector_dim:
            raise RuntimeError(f"{self.model_key}: language_projection width "
                               f"{self.module(self._connector_path).out_features} != language model hidden "
                               f"{self.connector_dim}")
        self._image_processor = self.processor.image_processor
        # The connector preflight captures the consumer of the connector locus as the module named `lm_head`
        # (preflight.run_preflight). This family keeps its head inside the language model, so the name is
        # registered as an alias of `language_model.lm_head` -- the same module object, no wrapper.
        self._modules["lm_head"] = self.model.language_model.lm_head

    # ------------------------------------------------------------------ prompts
    def prompt_text(self, question: str, with_image: bool = True) -> str:
        """The checkpoint's own recipe. The frame carries no image placeholder: the 128 visual embeddings are
        concatenated in front of the prompt, so the text is identical with and without an image."""
        return PROMPT_TEMPLATE.format(question=question)

    def encode(self, images, questions):
        tok = self.tokenizer
        texts = [self.prompt_text(q, with_image=images is not None) for q in questions]
        enc = tok(texts, return_tensors="pt", padding=True)
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if images is not None:
            pv = self._image_processor(images, return_tensors="pt")["pixel_values"]
            if pv.shape[0] != len(images):
                raise RuntimeError(f"{self.model_key}: {pv.shape[0]} pixel tensors for {len(images)} images")
            out["pixel_values"] = pv.unsqueeze(1)                    # (B, 1 image, 3, H, W): one image per row
        return out

    # ------------------------------------------------------------------ loci
    def loci(self) -> dict[str, LocusInfo]:
        return {
            "vis.last": LocusInfo(
                "vis.last", self._block_path,
                f"tensor output of vision encoder layer {self.n_tower_layers - 1} of {self.n_tower_layers} "
                f"(B, {self.n_vision_tokens}, {self.tower_dim}); the tower's post_layernorm is applied to it and the "
                f"result is what the Q-Former cross-attends to",
                self.tower_dim,
                f"all {self.n_vision_tokens} tokens (CLS + {self.n_patches} patches of the {self.image_size}x"
                f"{self.image_size} image); the Q-Former's encoder_attention_mask is all ones, so every row is read",
                "the single visual stream into the Q-Former's cross-attention",
                "none: the tower's post_layernorm is ON the consumed path (unlike LLaVA and CheXagent-2, where the "
                "tower's final LayerNorm is computed and discarded), so a write here reaches the reader through it",
                self._connector_path, True,
                f"EVA-CLIP-shaped ViT, {self.n_tower_layers} layers at {self.image_size} px with {self.patch_size} px "
                f"patches, trained on chest radiographs by the CheXagent authors"),
            "connector": LocusInfo(
                "connector", self._connector_path,
                f"tensor output of the linear language_projection over the Q-Former's {self.n_query_tokens} query "
                f"tokens, (B, {self.n_query_tokens}, {self.connector_dim})",
                self.connector_dim,
                f"all {self.n_query_tokens} query tokens; the Q-Former is a perceiver-style resampler, so the token "
                f"count is fixed and independent of the image",
                f"the {self.n_query_tokens} visual embeddings concatenated in front of the prompt embeddings",
                "none", "language model input embeddings", True,
                "the Q-Former takes no text input (query_embeds only), so the connector is prompt-independent"),
        }

    def image_token_mask(self, enc) -> torch.Tensor:
        """(B, L) bool over the LANGUAGE-MODEL input positions. This family has no image placeholder token: the
        visual embeddings are the first `n_query_tokens` positions of `inputs_embeds`, ahead of every prompt token,
        so the mask is built over the concatenated length rather than over `input_ids`."""
        B, L = enc["input_ids"].shape
        mask = torch.zeros((B, self.n_query_tokens + L), dtype=torch.bool)
        mask[:, :self.n_query_tokens] = True
        return mask

    def layouts(self, enc, images) -> dict[str, TokenLayout]:
        B = len(images)
        pv = enc.get("pixel_values")
        if pv is not None:
            if pv.shape[0] != B or pv.shape[1] != 1:
                raise RuntimeError(f"{self.model_key}: pixel batch {tuple(pv.shape)} is not (B={B}, 1 image, 3, H, W)")
            if bool((pv.flatten(2).sum(dim=2) == 0).any()):
                raise RuntimeError(f"{self.model_key}: an all-zero image would be dropped by the model's image_mask")
        if enc["input_ids"].shape[0] != B:
            raise RuntimeError(f"{self.model_key}: {enc['input_ids'].shape[0]} prompts for {B} images")
        vis = torch.ones(self.n_vision_tokens, dtype=torch.bool)
        con = torch.ones(self.n_query_tokens, dtype=torch.bool)
        return {"vis.last": TokenLayout(False, masks=[vis.clone() for _ in range(B)]),
                "connector": TokenLayout(False, masks=[con.clone() for _ in range(B)])}

    @torch.no_grad()
    def vision_features(self, enc):
        """Tower, Q-Former and projection only -- the checkpoint's own steps 1-3, on one image per row."""
        from einops import rearrange
        batch = self.to_device(enc)
        pv = batch.get("pixel_values")
        if pv is None:
            raise RuntimeError(f"{self.model_key}: vision_features called without pixel_values")
        image_mask = pv.sum(dim=(2, 3, 4)) != 0
        if not bool(image_mask.all()):
            raise RuntimeError(f"{self.model_key}: {int((~image_mask).sum())} images would be dropped as all-zero")
        vision_outputs = self.model.vision_model(pixel_values=pv[image_mask], return_dict=True)
        tmp = vision_outputs[0]
        image_embeds = tmp.new_zeros((*image_mask.shape, *tmp.shape[1:]))
        image_embeds[image_mask] = tmp
        image_attention_mask = torch.zeros(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        image_attention_mask[image_mask] = 1
        image_embeds = rearrange(image_embeds, "b i n d -> b (i n) d")
        image_attention_mask = rearrange(image_attention_mask, "b i n -> b (i n)")
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_attention_mask, return_dict=True)
        return self.model.language_projection(query_outputs[0])

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward_last_logits(self, enc, return_model_logits: bool = False):
        """The model's forward takes neither `logits_to_keep` nor `use_cache` (the latter is off on the language
        model's config), so it is called with the encoded batch alone; everything else is the base adapter's."""
        batch = self.to_device(enc)
        self._head_input = None
        out = self.model(**batch)
        if self._head_input is None:
            raise RuntimeError("lm_head hook did not fire")
        lg32 = self.fp32_logits_from_hidden(self._head_input[:, -1, :]).cpu()
        if return_model_logits:
            return lg32, out.logits[:, -1, :].float().cpu()
        return lg32

    # ------------------------------------------------------------------ metadata
    def processing_settings(self) -> dict:
        s = super().processing_settings()
        ip = self._image_processor
        s["image_processing"] = {"source": "the checkpoint's own BlipImageProcessor (preprocessor_config.json)",
                                 "settings": {k: v for k, v in ip.to_dict().items()
                                              if isinstance(v, (int, float, str, bool, list, dict, type(None)))},
                                 "image_size": self.image_size, "patch_size": self.patch_size,
                                 "vision_tokens": self.n_vision_tokens, "query_tokens": self.n_query_tokens}
        s["prompt_construction"] = (f"the checkpoint's own recipe (model card): {PROMPT_TEMPLATE!r} with "
                                    "add_special_tokens=True; the 128 projected query tokens are CONCATENATED in "
                                    "front of the prompt embeddings, so the prompt carries no image placeholder")
        s["special_token_ids"].update({"image_token_id": None, "visual_positions": f"0..{self.n_query_tokens - 1} of "
                                                                                   f"inputs_embeds (prepended)"})
        s["remote_code_shim"] = self._shim
        s["pad_token"] = getattr(self, "_pad_token_set", None)
        s["device_map"] = {"requested": getattr(self, "device_map_requested", None),
                           "parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters()})}
        return s
