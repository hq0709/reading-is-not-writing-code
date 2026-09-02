# Literature survey, 2026-08-29

Run to answer two questions before submission: **can this be scooped**, and **what must be cited**.

## The finding that changes the positioning

The claim "a model does not use what it encodes" is **no longer novel on its own**. Three independent
groups state a version of it, one of them two months ago and on two of our five models:

| paper | date | setting | what they show | what they lack |
|---|---|---|---|---|
| Ravichander, Belinkov & Hovy, *Probing the Probing Paradigm* (EACL 2021) | 2021 | synthetic NLP tasks | a property provably irrelevant to the task is still encoded above chance, so probe accuracy does not entail task relevance | synthetic only; no floor, no normalisation, no real model comparison |
| Kumar et al., *Encoded but Not Routed* (arXiv 2606.01679) | Jun 2026 | scientific charts vs tables, **Qwen2.5-VL-7B/32B, InternVL3-8B** | chart evidence is encoded in intermediate layers but does not reach the prediction position | no untrained baseline; no normalised quantity; localisation, not quantification |
| Goel, Bandyopadhyay & Shenk, *The Knowing-Saying Gap* (arXiv 2608.07528) | Jul 2026 | text LLMs, synthetic arithmetic | probes detect corrupted context at 0.98 AUROC while the model's confidence predicts the resulting error at 0.50 | no untrained floor; no share-of-what-training-added; not vision, not medical |

**Consequence.** The paper must not be sold on the phenomenon. It must be sold on the **measurement**:
the architecture-matched untrained floor $F$, the normalisation $(B-F)/(T-F)$, and the benchmarking
consequence. Nobody else has any of the three. Done: a new paragraph before Contributions says this
outright, and Related Work cites all three.

This is a *strengthening*, not a retreat. Three groups finding the same dissociation in text, charts and
chest radiographs is the argument that a measurement of it is worth having, and it is the answer to the
"one domain, five models" reviewer.

## Citations that were missing and are now in

1. **Ravichander, Belinkov & Hovy (EACL 2021)** — the direct ancestor of the thesis. Its absence was a
   real gap a reviewer would have found.
2. **Hewitt, Ethayarajh, Liang & Manning, *Conditional probing* (EMNLP 2021)** — the formal machinery for
   $T-F$. They condition on a baseline representation instead of comparing against it, and their baseline
   is the non-contextual word embedding. Images have no word identity; an untrained network of the same
   architecture is the substitute this setting admits. Saying so makes the $T-F$ half a principled
   instance of an existing framework and isolates the novelty in the third term $B$.
3. **Heap, Lawson, Farnik & Aitchison (arXiv 2501.17727)** — sparse autoencoders score a *randomly
   initialised* transformer as interpretably as a trained one. The SAE analogue of $F$, and independent
   evidence that the untrained control is the one that bites.
4. **Pedersen, Sydendal, Cheplygina & Sourget (arXiv 2608.12086, MICCAI-W 2026)** — layer-wise probes on
   the **same ChestX-ray14 images** find chest drains and scanner identity as the decodable content. Our
   acquisition-dominance result, arrived at independently, on the same dataset.

## Checked and not a threat

- **Mirage Probes** (2606.13870, Jun 2026) — VLMs answer image questions correctly with no image. About
  language priors inflating benchmarks, not probe-vs-behaviour. Adjacent, not competing.
- **SAE steering of medical VLMs** (2605.24977, May 2026) — RadVLM/LLaVA-Rad/CheXOne, steers to reduce
  hallucination. No random-direction control, which is the gap Section 8 measures.
- **HalluCXR** (2605.20469) — yes-bias in medical VLMs under our exact prompt format. Corroborates the
  LLaVA-Med extreme case.
- **Visual Grounding in Zero-Shot VL Control** (2608.06154, Aug 2026) — finds VLMs that are
  "image-invariant or nearly constant". The control-task analogue of a model answering yes to everything.

## Terminology collision, noted

*Model Utilization Index* (arXiv 2504.07440) uses "utilization" for the fraction of **activated neurons**
during inference. Unrelated quantity. Our definition is given explicitly at first use, so no change made,
but do not let a reviewer conflate them.

## Style base, re-checked

The ICLR 2026 outstanding papers are *Transformers are Inherently Succinct* and *LLMs Get Lost in
Multi-Turn Conversation*; CVPR 2026's are 4D reconstruction and 3D generation papers whose figures are
qualitative result grids, not applicable here. The teaser is already modelled on *LLMs Get Lost*, and the
line-chart and table styles come from the two artefacts supplied directly. No change warranted.
