# Related work: literature evidence

This file is the protocol-level evidence base for the paper's related-work section. Scientific run status
and project results are recorded elsewhere.

## 1. Layer-wise medical VLM probing

### Zhu et al., *Lost in the Hype*

Xun Zhu, Fanbin Mo, Xi Chen, Kaili Zheng, Shaoshuai Yang, Yiming Shi, Jian Gao, Miao Li and Ji Wu.
*Lost in the Hype: Revealing and Dissecting the Performance Degradation of Medical Multimodal Large
Language Models in Image Classification.* arXiv:2604.08333, 2026-04-09.
https://arxiv.org/abs/2604.08333

| Protocol element | Evidence |
|---|---|
| Models | 14 medical MLLMs from 2.21B to 34.75B, plus 23 traditional reference models: 8 CNNs, 4 MLPs, 4 ViTs and 7 hybrids |
| Data | BUSI, 780 images/600 patients; COVID19-CT, 746 images; Chest-Xray, 5,856 paediatric images |
| Loci | every vision-tower layer, connector, every language layer and final semantic output; 84 curves across model, dataset and tuning conditions |
| Probe | two-layer MLP with ReLU and dropout 0.1; AdamW, learning rate 1e-4, 20 epochs, batch size 4 |
| Comparisons | frozen versus LoRA-tuned tower; probing versus end behaviour; medical models versus general bases |
| Controls | no permutation null, randomised control task, random-encoder floor or matched intervention direction |

The paper identifies four failure modes: vision-tower quality limitation, connector fidelity loss, LLM
comprehension deficit and semantic-output misalignment. It reports that the connector mainly aligns
dimensions and that final generated accuracy may lie above or below the last-layer probe. Medical/general
comparisons include HuatuoGPT-Vision-7B, Lingshu-7B and ShizhenGPT-7B-VL against Qwen2.5-VL-7B, and
MedGemma variants against Gemma-3-4B.

Concept Flow therefore contributes controlled base-model curves and matched causal tests, not the first
medical VLM depth trace or the first Lingshu/Qwen comparison.

## 2. VLM probing plus steering

### Theodoridis et al., *Probing Visual Concepts in Lightweight VLMs for Automated Driving*

Nikos Theodoridis, Reenu Mohandas, Ganesh Sistu, Anthony Scanlan, Ciaran Eising and Tim Brophy.
arXiv:2603.06054, v2 2026-08-06; Transactions on Machine Learning Research, 2026.
https://arxiv.org/abs/2603.06054

| Protocol element | Evidence |
|---|---|
| Models | six variants of five architectures under 4B: Ovis2.5-2B, InternVL3.5-2B, Qwen3-VL-2B, VST-3B variants and DriveFusionQA |
| Data | counterfactual CARLA pairs over presence, count, spatial and orientation concepts; 400 train, 50 validation and 50 test samples per class and distance, with task-specific town exceptions; nuScenes OOD check |
| Loci | every vision block, projector output, every language block and post-layernorm output |
| Pooling | patch average; left/right region concatenation for spatial tasks; visual-token mean concatenated with last-token activation in the language model |
| Probe | standardised linear readout; learning-rate sweep 1e-4–5e-1; 10 seeds; chance-corrected accuracy and Wilson intervals |
| Steering | unit probe direction mapped back from standardised space; edit visual and last-token halves; threshold alpha selected from five values; one early high-probe-accuracy language locus |
| Controls | task chance and OOD evaluation; no permutation null, control task, random-encoder floor or matched steering directions |

Steering succeeds broadly for presence concepts and is absent for orientation. DriveFusionQA Count-1
provides a direct availability/use contrast: 38 samples produced valid base responses and three steered
successfully despite strong probe accuracy. The authors attribute no cause because steering
hyperparameters and loci were not exhaustively tested.

Concept Flow extends this protocol with models at 7B and above, clinical concepts, matched loci across the
vision-to-language path, full dose responses and matched control directions.

## 3. Contribution map

| Measurement object | Prior evidence | Concept Flow scope |
|---|---|---|
| medical VLM depth curve | Zhu et al., 14 medical MLLMs | fixed linear capacity, patient splits, repeated controls and nuisance probes |
| VLM probe-vector steering | Theodoridis et al., lightweight driving VLMs | clinical concepts, matched loci, both signs and control bands |
| randomised control task | Hewitt & Liang; later text-LLM applications | recurring medical-image metadata types and control-map uncertainty |
| equal probe capacity across width | no matched-width VLM depth study found | common 512-dimensional projection and fixed logistic readout |
| acquisition nuisance profile | Pedersen et al. across MedCLIP vision layers | same-locus nuisance profiles through complete VLMs |
| random-architecture floor | sparse prior use; Heap et al. provide an SAE analogue | same architecture, same labels and same held-out rows |
| decodability/steerability relation | established in text LLMs and non-medical VLMs | localisation along the medical vision-to-language pipeline |
| image-space versus representation-space dose response | no matching study found | planted clinical finding, location control and matched internal directions |

## 4. Decodability and steerability

The general dissociation is established. The literature leaves the locus-resolved medical-image question
open.

1. **Nadaf, *Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens*,
   arXiv:2604.02608.** Twelve tasks, six models and 4,032 directed cross-template pairs separate linear
   decodability from steerability. The decodable-but-not-steerable cell is rare (3/72) under its logit
   lens, tuned lens and controlled MLP readouts. https://arxiv.org/abs/2604.02608

2. **Liu, *Decodable but Not Corrected by Fixed Residual-Stream Linear Steering*,
   arXiv:2605.05715.** Medical text QA overthinking is decodable at 71.6% balanced accuracy, while five
   steering families across 29 configurations and `n=1,273` yield effects near zero; the per-instance
   probe/steering relation is `r=-0.002`, `p=0.97`. LEACE lowers accuracy by 3.6 percentage points while
   ten random erasures average +0.3 points. https://arxiv.org/abs/2605.05715

3. **Liang, Cheng & Wajid, *Encoded but Not Actionable*, arXiv:2608.17843.** Six frozen language models
   show divergence among decodability, forced-choice generation, activation influence and behavioural
   steerability. Shuffled-order and random-initialisation controls show that sketch-level degrees of
   freedom can be decodable before training. https://arxiv.org/abs/2608.17843

4. **Pramono, Cai & Kulkarni, *TRAPSBench*, arXiv:2608.13167.** Across 16 VLMs, answerability reaches
   0.91 probe AUROC while spontaneous epistemic restraint peaks at PECS 0.292; single-layer void-direction
   steering changes abstention. https://arxiv.org/abs/2608.13167

5. **Majumdar, Kogel & Bulling, *CARD*, arXiv:2608.20763.** Cross-axis interventions show VLMs retaining
   partner-belief information without routing it into the next action. https://arxiv.org/abs/2608.20763

6. **Wu, Zhao & Chen, *When Is a Steerable Concept Representation Real?*, arXiv:2608.08159.** A
   17-model, five-family, 0.6B–72B audit shows that raw units, readout metric and operating-point selection
   shape apparent steering trends. With residual-norm-comparable interventions and held-out selection,
   steering remains significant at every tested scale while the Qwen3-series slope is not significant;
   its interval still admits a moderate positive slope. https://arxiv.org/abs/2608.08159

7. **Fan, Cheng, Li, Feizi & Zhou, *When is Your LLM Steerable?*, arXiv:2606.11599.** ASTEER contains
   1.4M generations across 150 concepts and predicts steerability from early states at about 0.7
   macro-F1. https://arxiv.org/abs/2606.11599

8. **Torop, Masoomi & Dy, *Inverted Detection and Control in Steering Vectors*,
   arXiv:2608.02957.** Highly discriminative directions can consistently steer behaviour in the opposite
   direction, establishing sign inversion as a distinct outcome. https://arxiv.org/abs/2608.02957

9. **Liu, Wang, Wang, Xiao & Lin, *Broken Symmetry in LLM Refusal*, arXiv:2608.15772.** Bidirectional
   patching finds linearly recoverable answers under refusal and different intervention locality for
   answer release versus renewed suppression. https://arxiv.org/abs/2608.15772

## 5. Adjacent medical and multimodal work

- **Nooralahzadeh et al., *Universal Boosts, Specific Suppressors*, arXiv:2605.24977.** Per-token SAE
  steering at late medical VLM layers improves clinical-composite report scores by 5.4%, 7.2% and 17.0%
  on MIMIC-CXR and transfers to IU-Xray at +7.7% GREEN. The method uses SAE features for report quality;
  Concept Flow uses supervised concept and behaviour-fitted directions for measurement.
  https://arxiv.org/abs/2605.24977

- **Pedersen, Sydendal, Cheplygina & Sourget, *Look What the Probes Dragged In!*,
  arXiv:2608.12086.** Seventeen MedCLIP vision-layer probes on NIH-CXR14 and PadChest identify localised
  drains, diffuse scanner noise, calibration weaknesses and dataset-quality problems.
  https://arxiv.org/abs/2608.12086

- **Rajaram, Schwettmann, Andreas & Conmy, *Line of Sight*, arXiv:2506.04706.** LLaVA-Next contains
  linearly available ImageNet-class features in the residual stream; targeted edits provide causal
  evidence. https://arxiv.org/abs/2506.04706

- **Takeda & Sakai, *Does medical specialization of VLMs enhance discriminative power?*,
  arXiv:2601.14774.** Feature-distribution comparisons find that medical specialisation alone does not
  determine discriminative representation quality. https://arxiv.org/abs/2601.14774

## 6. Methodological ancestors

| Work | Reference | Role |
|---|---|---|
| Alain & Bengio, linear probes | arXiv:1610.01644 | linear readouts across depth |
| Hewitt & Liang, control tasks | arXiv:1909.03368; EMNLP-IJCNLP 2019, https://aclanthology.org/D19-1275/ | selectivity against matched random labels |
| Belinkov, probing survey | arXiv:2102.12452; *Computational Linguistics* 48(1) | availability/use distinction and probe limits |
| Elazar et al., amnesic probing | arXiv:2006.00995; TACL 2021, https://aclanthology.org/2021.tacl-1.10/ | intervention test for behavioural use |
| Ravfogel et al., INLP | arXiv:2004.07667; ACL 2020, https://aclanthology.org/2020.acl-main.647/ | iterative linear concept erasure |
| Belrose et al., LEACE | arXiv:2306.03819 | closed-form linear concept erasure |
| Turner et al., activation addition | arXiv:2308.10248 | residual edit `h + alpha*v` |
| Panickssery et al., contrastive activation addition | arXiv:2312.06681; ACL 2024, https://aclanthology.org/2024.acl-long.828/ | difference-of-means direction; cite under Panickssery et al. |
| Zou et al., representation engineering | arXiv:2310.01405 | representation reading and control framework |
| nostalgebraist, logit lens | https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens | intermediate-layer unembedding |
| Belrose et al., tuned lens | arXiv:2303.08112 | calibrated intermediate readout |

Original submission years remain the bibliography year when later arXiv revisions exist. Published venue
versions are primary for Hewitt & Liang, Ravfogel et al., Elazar et al. and Panickssery et al.

## 7. Reproducible search record

Full-text protocol checks used arXiv HTML for `2604.08333v1` and `2603.06054v2`, with local searches for
`steer`, `random`, `control`, `baseline`, `sham`, `shuffle`, `permutation`, `alpha`, `pool`, `split`,
`patient`, `Hewitt` and `selectivity`.

The August 2026 arXiv search used:

1. `(abs:probing OR abs:"linear probe" OR abs:probes) AND (abs:medical OR abs:clinical OR
   abs:radiology) AND (abs:"vision-language" OR abs:multimodal OR abs:VLM)` with
   `submittedDate:[202608010000 TO 202609010000]`: 9 `abs:` results, 10 under `all:`; 2608.12086 was the
   relevant medical-image layer-probing result.
2. `(steering OR "activation addition" OR steerability) AND (probe OR probing OR decodability)` over the
   same date window: 47 results under both `abs:` and `all:`; 2608.02957 and 2608.15772 were additional
   relevant results.
3. `(connector OR projector OR "vision encoder") AND ("layer-wise" OR "across layers" OR
   "information flow" OR depth) AND (probe OR probing)`, widened to July 2026 onward: 2 results, both
   already represented above.

Within that search window, the closest clinical-concept depth study was Pedersen et al., which stops at
the MedCLIP vision encoder. The exact queries and date window should be rerun immediately before
submission because this literature was adding relevant papers weekly in August 2026.
