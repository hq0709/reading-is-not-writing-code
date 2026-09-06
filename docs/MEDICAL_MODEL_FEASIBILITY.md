# Medical-domain behavioral opportunity: asset feasibility

## Run

Read-only assessment on 2026-09-06 UTC of the accepted evidence, official model metadata and implementation sources, and the fixed server runtime. No model weights or patient outcomes were generated.

## Observation

The accepted Mass study has calibration eligibility but near-chance confirmation rankings and larger random-direction effects. The LLaVA independent image study leaves image opportunity unestablished. These results make an independently qualified behavioral condition the useful prerequisite for the next direction-control comparison. Existing matrix geometry and answer-encoding results already constrain broad-response explanations; an additional label-conditioned interaction analysis would distinguish only an additive shared-response model, not the remaining multiplicative or nonlinear alternatives.

The first medical-domain candidate is `microsoft/llava-med-v1.5-mistral-7b`. The official Hub API reports public, ungated revision `91bb16c122001ddc9cf1fd36ce1dae09448943a2`, four safetensors shards and 15,132,438,528 tensor bytes. Its index contains 686 keys, including 391 vision-tower and four projector keys. The official implementation is pinned for assessment at `30697ca50b5c29a8e955c99330b259776aef27b9`. [Model and file metadata](https://huggingface.co/api/models/microsoft/llava-med-v1.5-mistral-7b?blobs=true), [weight index](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/91bb16c122001ddc9cf1fd36ce1dae09448943a2/model.safetensors.index.json).

Its configuration declares `LlavaMistralForCausalLM` / `llava_mistral`, a Mistral language backbone, `openai/clip-vit-large-patch14-336`, layer `-2`, patch-only features and square-padding preprocessing. The official vision implementation selects that hidden state and removes CLS before the projector. These settings are useful for locating the consumed visual representation; matching the existing LLaVA encoder name does not establish matching checkpoint weights or activations. [Pinned configuration](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/91bb16c122001ddc9cf1fd36ce1dae09448943a2/config.json), [vision implementation](https://github.com/microsoft/LLaVA-Med/blob/30697ca50b5c29a8e955c99330b259776aef27b9/llava/model/multimodal_encoder/clip_encoder.py).

The server runtime reports Transformers 5.16.1, PyTorch 2.7.0+cu126, Accelerate 1.14.0 and Tokenizers 0.23.1. SentencePiece distribution metadata is absent. The native configuration registry contains `mistral` and `llava`, but not `llava_mistral`. Upstream instead pins Transformers 4.36.2 and Accelerate 0.21.0 and supplies its own model registration. Direct native loading is therefore not yet an established path. [Official dependencies](https://github.com/microsoft/LLaVA-Med/blob/30697ca50b5c29a8e955c99330b259776aef27b9/pyproject.toml), [model registration](https://github.com/microsoft/LLaVA-Med/blob/30697ca50b5c29a8e955c99330b259776aef27b9/llava/model/language_model/llava_mistral.py).

The model card describes biomedical figure-caption training from PubMed Central and research-only intended use. That provenance supplies a domain-adaptation rationale, rather than patient-level decontamination evidence for NIH. [Official model card](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b), [data summary](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/91bb16c122001ddc9cf1fd36ce1dae09448943a2/data_summary_card.md).

## Gate decision

`ASSET_METADATA_READY`; runtime compatibility remains `UNRESOLVED`. Select this candidate for one bounded compatibility audit before weight staging. Its architecture keeps a consumed CLIP locus available for the project's fixed-capacity readout and equal-norm intervention design, while changing the downstream model package. Its clinical capability must be measured independently.

## Next step

Can the pinned checkpoint's tokenizer, image preparation, feature selection, projector and first-answer logits be faithfully represented in the fixed environment? The bounded audit is specified in `docs/RESEARCH_PLAN.md#medical-domain-runtime-feasibility`.

## Limitations

This is metadata and source feasibility, not a model-loading or scientific result. Training overlap with the NIH evaluation patients is unresolved. Comparisons with the accepted LLaVA model would change the language backbone and biomedical training together, so they would concern model packages rather than isolate a training effect. Existing activation archives require a demonstrated model/input match before reuse. Any dependency change needs an explicit reviewed lockfile change. MedGemma remains an alternative whose public card requires accepting access terms; account entitlement was not checked. [MedGemma access conditions](https://huggingface.co/google/medgemma-4b-it#access-medgemma-on-hugging-face).
