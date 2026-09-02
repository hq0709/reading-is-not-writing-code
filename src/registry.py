"""Model registry: per-architecture hook paths, so one instrument serves every model in the study.

Module paths were read off `named_modules()` of each model after actually loading it, not inferred from
the class name. They differ enough between families that a generic guess fails: Qwen puts the connector
inside the vision tower as `visual.merger`, LLaVA has it as a sibling `multi_modal_projector`, and
InternVL calls its vision blocks `encoder.layer` rather than `blocks` or `layers`.

Every model here is 7B or larger by decision. CheXagent-2-3b is recorded but disabled, because 3B models
are excluded from the study; it stays in the table so nobody re-discovers it and wonders why it is absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Arch:
    key: str
    hf_id: str
    family: str                 # qwen | llava | internvl
    domain: str                 # medical | general
    params_b: float
    vision_root: str            # module whose output is the encoder representation
    vision_block_fmt: str       # format string for one vision block, {i}
    n_vision_blocks: int
    connector: str              # module whose output is the visual tokens the LLM consumes
    llm_layer_fmt: str          # format string for one LLM decoder layer, {i}
    n_llm_layers: int
    image_token: str | None     # token whose positions mark visual tokens
    image_token_id: int | None
    visual_tokens: int          # per image, at the pinned resolution
    notes: str = ""
    enabled: bool = True
    processor_kwargs: dict = field(default_factory=dict)
    pair: str | None = None     # architectural twin, for a matched domain comparison
    vision_feature_layer: int = -1
    """Which vision block the connector actually consumes, as a negative index.

    LLaVA sets this to -2, so the LAST vision block's output is computed and then discarded. Hooking
    block n-1 there reads and steers a tensor the model never uses: measured on llava15_7b and
    llavamed7b, all 27 steering directions at all 9 magnitudes returned P(yes) identical to 16 decimal
    places. Any locus called 'vis.last' must therefore point at n + vision_feature_layer, not n-1."""


REGISTRY: dict[str, Arch] = {
    # --- the matched pair. Identical architecture, so a difference between them is domain training ---
    "qwen7b": Arch(
        key="qwen7b", hf_id="Qwen/Qwen2.5-VL-7B-Instruct", family="qwen", domain="general",
        params_b=8.29, vision_root="model.visual", vision_block_fmt="model.visual.blocks.{i}",
        n_vision_blocks=32, connector="model.visual.merger",
        llm_layer_fmt="model.language_model.layers.{i}", n_llm_layers=28,
        image_token="<|image_pad|>", image_token_id=151655, visual_tokens=144,
        processor_kwargs={"min_pixels": 336 * 336, "max_pixels": 336 * 336},
        notes="PatchMerger connector, 2x2 spatial merge. Token count is variable unless pixels are pinned.",
        pair="lingshu7b"),
    "lingshu7b": Arch(
        key="lingshu7b", hf_id="lingshu-medical-mllm/Lingshu-7B", family="qwen", domain="medical",
        params_b=8.29, vision_root="model.visual", vision_block_fmt="model.visual.blocks.{i}",
        n_vision_blocks=32, connector="model.visual.merger",
        llm_layer_fmt="model.language_model.layers.{i}", n_llm_layers=28,
        image_token="<|image_pad|>", image_token_id=151655, visual_tokens=144,
        processor_kwargs={"min_pixels": 336 * 336, "max_pixels": 336 * 336},
        notes="Architecturally identical to qwen7b: 28 layers, hidden 3584, same connector.",
        pair="qwen7b"),

    # --- the second matched pair, LLaVA framework ---
    "llavamed7b": Arch(
        key="llavamed7b", hf_id="models/llava-med-7b-hf", family="llava", domain="medical",
        params_b=7.57, vision_root="model.vision_tower.vision_model",
        vision_block_fmt="model.vision_tower.vision_model.encoder.layers.{i}", n_vision_blocks=24,
        connector="model.multi_modal_projector",
        llm_layer_fmt="model.language_model.layers.{i}", n_llm_layers=32,
        image_token="<image>", image_token_id=32000, visual_tokens=576,
        notes="Converted from microsoft/llava-med-v1.5-mistral-7b, whose model_type llava_mistral does "
              "not exist in transformers 5.2. The remap is numerically exact: 686/686 tensors, max abs "
              "diff 4.2e-5, cosine 0.99999994. Its config advertises convnext and img_size 640; both are "
              "vestigial, the weights are CLIP ViT-L/14-336.",
        pair="llava15_7b", vision_feature_layer=-2),
    "llava15_7b": Arch(
        key="llava15_7b", hf_id="llava-hf/llava-1.5-7b-hf", family="llava", domain="general",
        params_b=7.06, vision_root="model.vision_tower.vision_model",
        vision_block_fmt="model.vision_tower.vision_model.encoder.layers.{i}", n_vision_blocks=24,
        connector="model.multi_modal_projector",
        llm_layer_fmt="model.language_model.layers.{i}", n_llm_layers=32,
        image_token="<image>", image_token_id=32000, visual_tokens=576,
        notes="Domain control for llavamed7b. Base LLM differs (Vicuna against Mistral), so this pair is "
              "less clean than qwen7b/lingshu7b and is reported as a secondary comparison.",
        pair="llavamed7b", vision_feature_layer=-2),

    # --- additional architectures, for the connector-type axis ---
    "huatuo7b": Arch(
        key="huatuo7b", hf_id="FreedomIntelligence/HuatuoGPT-Vision-7B", family="llava", domain="medical",
        params_b=7.0, vision_root="model.vision_tower.vision_model",
        vision_block_fmt="model.vision_tower.vision_model.encoder.layers.{i}", n_vision_blocks=24,
        connector="model.multi_modal_projector",
        llm_layer_fmt="model.language_model.layers.{i}", n_llm_layers=28,
        image_token="<image>", image_token_id=None, visual_tokens=576,
        notes="UNVERIFIED. Paths assumed from the LLaVA family; confirm with src/verify_arch.py before "
              "trusting any number from it."),
    "internvl3_8b": Arch(
        key="internvl3_8b", hf_id="OpenGVLab/InternVL3-8B-hf", family="internvl", domain="general",
        params_b=8.08, vision_root="model.vision_tower",
        vision_block_fmt="model.vision_tower.encoder.layer.{i}", n_vision_blocks=24,
        connector="model.multi_modal_projector",
        llm_layer_fmt="model.language_model.layers.{i}", n_llm_layers=28,
        image_token="<IMG_CONTEXT>", image_token_id=151667, visual_tokens=256,
        processor_kwargs={"crop_to_patches": False},
        notes="Pixel-shuffle 0.5 then LayerNorm then MLP. Vision blocks are 'encoder.layer', singular."),
    "llavaov7b": Arch(
        key="llavaov7b", hf_id="llava-hf/llava-onevision-qwen2-7b-ov-hf", family="llava", domain="general",
        params_b=8.03, vision_root="model.vision_tower.vision_model",
        vision_block_fmt="model.vision_tower.vision_model.encoder.layers.{i}", n_vision_blocks=27,
        connector="model.multi_modal_projector",
        llm_layer_fmt="model.language_model.layers.{i}", n_llm_layers=28,
        image_token="<image>", image_token_id=151646, visual_tokens=1485,
        notes="Emits far more visual tokens than the others (1485 at 336px, 3699 at 512px), so its "
              "extraction is slower and its pooled representation averages over more positions. "
              "Isolates the Qwen2 backbone from the Qwen connector."),

    # --- recorded but excluded ---
    "chexagent3b": Arch(
        key="chexagent3b", hf_id="StanfordAIMI/CheXagent-2-3b", family="chexagent", domain="medical",
        params_b=3.14, vision_root="model.visual.model", vision_block_fmt="model.visual.model.layers.{i}",
        n_vision_blocks=24, connector="model.visual.ln_post",
        llm_layer_fmt="model.layers.{i}", n_llm_layers=32,
        image_token=None, image_token_id=None, visual_tokens=1024, enabled=False,
        notes="EXCLUDED: 3B, and the study is 7B and above by decision. Also needs conda env flow_grpo, "
              "because its remote code asserts transformers==4.40.0. Verified to load and hook there, so "
              "it can be revived as an appendix if a chest-radiograph specialist is wanted."),
}


def enabled_archs() -> list[Arch]:
    return [a for a in REGISTRY.values() if a.enabled]


def matched_pairs() -> list[tuple[Arch, Arch]]:
    """Pairs that differ in domain but not architecture. These carry the domain claim."""
    seen, out = set(), []
    for a in REGISTRY.values():
        if a.pair and a.enabled and REGISTRY.get(a.pair, Arch("", "", "", "", 0, "", "", 0, "", "", 0, None, None, 0)).enabled:
            k = tuple(sorted((a.key, a.pair)))
            if k not in seen:
                seen.add(k)
                out.append((a, REGISTRY[a.pair]))
    return out


if __name__ == "__main__":
    print(f"{'key':14s} {'domain':8s} {'family':10s} {'B':>5s} {'vis':>4s} {'llm':>4s} {'vtok':>5s}  pair")
    for a in REGISTRY.values():
        flag = "" if a.enabled else "  [EXCLUDED]"
        print(f"{a.key:14s} {a.domain:8s} {a.family:10s} {a.params_b:5.2f} {a.n_vision_blocks:4d} "
              f"{a.n_llm_layers:4d} {a.visual_tokens:5d}  {a.pair or '-'}{flag}")
    print("\nmatched pairs (domain differs, architecture does not):")
    for a, b in matched_pairs():
        same = a.n_llm_layers == b.n_llm_layers and a.family == b.family
        print(f"  {a.key} ({a.domain}) vs {b.key} ({b.domain})  "
              f"{'identical architecture' if same else 'architecture differs, secondary'}")
