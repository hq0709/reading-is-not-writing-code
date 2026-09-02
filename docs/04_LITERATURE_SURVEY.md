# Literature survey: positive taxonomy

Survey date: 2026-08-29. The literature supports a measurement paper about availability, behavioural use
and causal anchoring rather than a priority claim for the general encoded-but-unused phenomenon.

## Availability and behavioural use

| Work | Setting | Evidence relevant to Concept Flow |
|---|---|---|
| Ravichander, Belinkov & Hovy, *Probing the Probing Paradigm* (EACL 2021) | synthetic NLP tasks | a task-irrelevant property remains decodable, establishing that probe accuracy and task relevance are different quantities |
| Kumar et al., *Encoded but Not Routed* (arXiv:2606.01679) | charts versus tables; Qwen2.5-VL-7B/32B and InternVL3-8B | chart evidence is available in intermediate layers but does not reach the prediction position |
| Goel, Bandyopadhyay & Shenk, *The Knowing-Saying Gap* (arXiv:2608.07528) | text LLM arithmetic | probes decode corrupted context at 0.98 AUROC while model confidence predicts the resulting error at 0.50 |

Together these studies establish the broad dissociation across synthetic language, multimodal charts and
text reasoning. Concept Flow contributes the architecture-matched untrained floor `F`, trained readout
`T`, behaviour `B`, and the normalised utilisation `(B-F)/(T-F)` in medical VLMs.

## Controls and baselines

| Work | Contribution to the method |
|---|---|
| Hewitt, Ethayarajh, Liang & Manning, *Conditional probing* (EMNLP 2021) | formalises the incremental information in `T-F` by conditioning on a baseline representation; the same-architecture untrained network is the image-model baseline used here |
| Hewitt & Liang, control tasks (EMNLP-IJCNLP 2019) | defines selectivity against a matched random-label task; Concept Flow adapts recurring word types to recurring image metadata types |
| Heap, Lawson, Farnik & Aitchison (arXiv:2501.17727) | a randomly initialised transformer can score as interpretably as a trained one under sparse-autoencoder evaluation, motivating an explicit untrained floor |
| Wu, Zhao & Chen, *When Is a Steerable Concept Representation Real?* (arXiv:2608.08159) | shows that norm calibration, readout choice and operating-point selection determine cross-model steering comparisons |

## Medical imaging and medical VLM neighbours

| Work | Relationship |
|---|---|
| Zhu et al., *Lost in the Hype* (arXiv:2604.08333) | traces class information through vision, connector and language layers in 14 medical MLLMs; establishes the depth-curve object that Concept Flow controls |
| Pedersen et al., *Look What the Probes Dragged In!* (arXiv:2608.12086; MICCAI-W 2026) | probes 17 MedCLIP layers on NIH-CXR14 and PadChest and identifies chest drains, scanner identity and data-quality effects; direct support for acquisition-nuisance measurement |
| Nooralahzadeh et al., *Universal Boosts, Specific Suppressors* (arXiv:2605.24977) | steers sparse-autoencoder features in medical VLM report generation; nearest medical neighbour for residual-stream intervention mechanics |
| HalluCXR (arXiv:2605.20469) | documents yes-bias in medical VLMs under a closely matched prompt format, motivating AUROC and paired-readout checks |

## VLM probing and steering

| Work | Relationship |
|---|---|
| Theodoridis et al., *Probing Visual Concepts in Lightweight VLMs for Automated Driving* (arXiv:2603.06054; TMLR 2026) | probes every vision, projector and language layer and steers with probe weights; supplies the closest cross-domain protocol ancestor |
| Rajaram et al., *Line of Sight* (arXiv:2506.04706) | shows linearly available ImageNet features in LLaVA-Next and tests them with targeted edits |
| TRAPSBench (arXiv:2608.13167) | VLM answerability reaches 0.91 probe AUROC while spontaneous epistemic restraint is weak; single-layer steering changes abstention |
| CARD (arXiv:2608.20763) | diagnoses represented-but-unused belief information through cross-axis interventions in VLMs |

## Terminology

*Model Utilization Index* (arXiv:2504.07440) uses “utilization” for the fraction of activated neurons.
Concept Flow defines utilisation explicitly as `(B-F)/(T-F)` at first use.

## Positioning carried into the paper

The literature establishes that decodability and behavioural expression can diverge. Concept Flow asks
where that divergence appears along the medical VLM pipeline, how it changes after an untrained floor and
matched controls, and which internal direction follows a planted clinical finding. `docs/11_RELATED_WORK.md`
contains the protocol-level evidence and reproducible search record.
