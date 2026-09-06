# LLaVA-Med runtime feasibility

## Run

The bounded CPU-only preparation inspected the public checkpoint
`microsoft/llava-med-v1.5-mistral-7b` at
`91bb16c122001ddc9cf1fd36ce1dae09448943a2`, the official LLaVA-Med source at
`30697ca50b5c29a8e955c99330b259776aef27b9`, and the fixed Concept Flow runtime at clean pushed
source `322bdeeb85b9978e9ef777f73aec68ba5b340d1d`. The runtime was Python 3.13.15,
PyTorch 2.7.0+cu126, Transformers 5.16.1, Accelerate 1.14.0 and Tokenizers 0.23.1.
The project health gate passed with 1,334,893,256 KiB free, 38,998 accumulated GPU-seconds,
one consecutive failed dispatcher run, zero API spend and an absent stop sentinel. The audit read
small public source, configuration, tokenizer and safetensors-header metadata; it did not load a
model or evaluate an image.

## Observation

### Complete parameter path

The [checkpoint index](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/91bb16c122001ddc9cf1fd36ce1dae09448943a2/model.safetensors.index.json)
contains 686 tensors and 15,132,438,528 tensor bytes. Range reads of the four safetensors headers
show that all 686 tensors are BF16. The learned families and representative shapes are:

| family | tensors | representative shape |
|---|---:|---|
| Mistral language model | 290 | embeddings `[32000,4096]`; 32 layers with Q `[4096,4096]`, K/V `[1024,4096]` and MLP `[14336,4096]` |
| language-model head | 1 | `[32000,4096]` |
| CLIP vision tower | 391 | patch kernel `[1024,3,14,14]`; positions `[577,1024]`; 24 encoder layers |
| two-layer projector | 4 | `[4096,1024]`, `[4096]`, `[4096,4096]`, `[4096]` |

The smallest fixed-runtime route is a native `LlavaForConditionalGeneration` conversion with a
Mistral text configuration and CLIP ViT-L/14-336 vision configuration. The source keys have a
one-to-one, collision-free mapping:

| checkpoint prefix | native Transformers 5.16.1 prefix |
|---|---|
| `model.layers.`, `model.embed_tokens.`, `model.norm.` | `model.language_model.layers.`, `model.language_model.embed_tokens.`, `model.language_model.norm.` |
| `model.vision_tower.vision_tower.vision_model.` | `model.vision_tower.` |
| `model.mm_projector.0.`, `model.mm_projector.2.` | `model.multi_modal_projector.linear_1.`, `model.multi_modal_projector.linear_2.` |
| `lm_head.` | `lm_head.` |

This transforms 686 source keys into 686 unique target keys: 290 language, 391 vision, four
projector and one head tensor. Instantiate the native model first with vocabulary size 32,000,
load this mapped state strictly, and only then add the special `<image>` token at ID 32,000 and
resize to 32,001 rows. Strict loading before expansion preserves every learned tensor and every
original embedding/head row. The added input row is overwritten by image features. First-answer
log partitions and token mass are computed over original IDs 0--31,999 so the added output row does
not alter the original vocabulary distribution. The official Transformers
[conversion utility](https://github.com/huggingface/transformers/blob/c93057d4835cd31752bb56f59989dd27696eb45b/src/transformers/models/llava/convert_llava_weights_to_hf.py)
establishes the same meta-device, strict-load-then-expand pattern; the prefix table above binds it
to the nested module layout observed in the fixed runtime.

The four source shard identities are:

| shard | bytes | SHA-256 |
|---|---:|---|
| `model-00001-of-00004.safetensors` | 4,943,162,336 | `ef2190dc6c2a940e60f03f5fdb4dddb2320eb87801aeca5c40b0a28ce8aa420e` |
| `model-00002-of-00004.safetensors` | 4,999,819,336 | `2b229607fecd98b8111320178e5bf3e2c527b05a942c85d65b5b507c76c1ed00` |
| `model-00003-of-00004.safetensors` | 4,927,408,360 | `12b18ecdf8924d5fe28ada797fe6697fa60e62cba630759fbeb52975b261c4e2` |
| `model-00004-of-00004.safetensors` | 262,144,128 | `1d2063fcd429d3f0f0a8a091b0522f0e02f2d85fe0e5b0eeb4ae168183a603bc` |

### Tokenizer and conversation

The checkpoint's `tokenizer.model` has SHA-256
`dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055`, identical to the file at
public Mistral revision `63a8b081895390a26e140280378bc85ec8bce07a`. Pin that revision's
[`tokenizer.json`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/63a8b081895390a26e140280378bc85ec8bce07a/tokenizer.json),
SHA-256 `11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720`, beside the
checkpoint's own tokenizer configuration. The fixed runtime then loads the tokenizer without a
new dependency and retains length 32,000, BOS 1, EOS 2, PAD/UNK 0 and model length 2,048.

The official [`mistral_instruct` conversation](https://github.com/microsoft/LLaVA-Med/blob/30697ca50b5c29a8e955c99330b259776aef27b9/llava/conversation.py)
renders a single-image question as
`[INST] <image>\n{question} [/INST]`; BOS is added by the tokenizer. The official
[`tokenizer_image_token`](https://github.com/microsoft/LLaVA-Med/blob/30697ca50b5c29a8e955c99330b259776aef27b9/llava/mm_utils.py)
uses sentinel `-200`. For the prospective Effusion question it produces 30 positions. The native
input constructor must preserve those IDs rather than tokenize the complete prompt after adding
`<image>`: replace the single `-200` position with 576 repetitions of native special-token ID
32,000. This produces 605 positions, preserves all 29 non-image IDs including the space token
immediately after the image, and supplies one placeholder per projected patch. Directly tokenizing
the complete prompt with `<image>` as an added token would incorrectly drop that post-image space
and produce only 604 positions. The available single-token answer IDs are yes/Yes `5081/5592` and
no/No `708/1770`.

### Image and first-answer path

The checkpoint [configuration](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/91bb16c122001ddc9cf1fd36ce1dae09448943a2/config.json)
selects square padding, `openai/clip-vit-large-patch14-336`, hidden layer `-2`, patch-only features
and an `mlp2x_gelu` projector. Pin the CLIP configuration and processor at public revision
`ce19dc912ca5cd21c8a653c79e251e808ccabcd1`. Its processor resizes and center-crops to 336,
then normalizes with mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation
`(0.26862954, 0.26130258, 0.27577711)`.

The exact image path is: preserve the source image mode for the padding/background decision; pad a
non-square image to a square with the processor-mean background; let the pinned CLIP processor
convert it to RGB, resize, crop and normalize it; run the embedded 391-tensor CLIP tower; take
hidden state `-2`; discard CLS; project the resulting 576 patch vectors through the two-layer GELU
projector; and replace the 576 explicitly expanded image placeholders in the Mistral input
embedding sequence. Patch size 14, one additional CLIP token and `default` feature selection imply
exactly 576 projected patch vectors. The embedded vision weights from the LLaVA-Med shards, rather
than separately fetched base-CLIP weights, own the vision computation. The official [vision tower](https://github.com/microsoft/LLaVA-Med/blob/30697ca50b5c29a8e955c99330b259776aef27b9/llava/model/multimodal_encoder/clip_encoder.py)
and [multimodal insertion](https://github.com/microsoft/LLaVA-Med/blob/30697ca50b5c29a8e955c99330b259776aef27b9/llava/model/llava_arch.py)
define the reference order.

Transformers 5.16.1 constructs the native LLaVA language component through `AutoModel` from the
Mistral subconfiguration and accepts current `Cache`, attention-mask and position-ID semantics.
The measurement path uses one direct full-prompt forward with `use_cache=False` and
`logits_to_keep=1`. The final prompt-position logits supply the registered answer candidates; this
avoids the legacy custom generation wrapper while preserving the first-answer computation.

### Executable verification design and staging estimate

The next gate can verify this path with a small, fixed suite:

1. Stage the four source shards and the pinned model, tokenizer and CLIP metadata into a partial
   directory; verify the four published shard hashes and required identities once before atomic
   promotion.
2. Stream-remap all 686 tensors, require 686 unique target keys and strict native loading with no
   missing or unexpected learned tensors, then confirm the first 32,000 embedding and head rows
   match their source tensors exactly after image-token expansion.
3. Replay the official prompt builder and deterministic native expansion; require identical 29
   non-image token IDs, exactly one source sentinel versus 576 native placeholders, 605 total
   positions, and identical post-insertion attention masks, position IDs and multimodal embedding
   sequence.
4. On fixed synthetic square and rectangular images, compare padded pixels, normalized tensors,
   layer-22 patch features, projector output and the 576 inserted vectors. Bind Python's padding RNG
   for the rectangular case because the official helper permits the two central one-pixel offsets.
5. On the registered validation-only images, compare the official custom path and native path for
   finite layer-22 features, projector outputs and the four final yes/Yes/no/No logits. Require exact
   token identities and a prospectively fixed BF16 tolerance before accepting first-answer scoring.
6. Record load time and memory, run an exact clean repeat, and preserve the mapping, prompt,
   processor, forward-path and score receipts before any scientific cohort is allocated.

The source shards total 15,132,534,160 file bytes. A shard-streaming conversion retains about
15.14 GB of source and 15.14 GB of converted weights, so 31 GB covers durable assets and 40 GB is a
conservative staging reserve. At 10--50 MB/s, source transfer is approximately 5--26 minutes;
conversion performs about 30.3 GB of sequential tensor I/O. Reserve 45 minutes for the atomic asset
stage and account GPU validation separately under its own registered cap.

## Gate decision

`PASS`; preparation disposition `READY`. The native conversion supplies a supported current-runtime
model interface, a complete learned-parameter mapping, a dependency-preserving tokenizer path, an
exact conversation/image construction, a bounded staging estimate and a finite validation design.

## Next step

Can the accepted native conversion and input-construction path pass a prospectively specified asset
identity and validation-only measurement gate? Register the atomic staging receipt, strict mapping,
prompt/processor equivalence, first-answer score checks, timing, memory and stop conditions before
execution.
