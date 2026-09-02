# Concept Flow: paper plan

## Paper objective

Concept Flow measures two different properties of internal representations in medical vision-language
models: whether a clinical concept can be read from a locus, and whether changing a direction at that
locus changes the model's behaviour. The paper's contribution is a controlled, causally anchored
measurement of the gap between those properties.

Zhu et al. (arXiv:2604.08333) already trace diagnostic information through medical VLM vision towers,
connectors and language layers. Theodoridis et al. (arXiv:2603.06054) combine layer-wise VLM probing with
probe-vector steering in automated driving. Concept Flow builds on that foundation with matched probe
capacity, patient-level splits, nuisance measurements, randomised control tasks, matched intervention
controls and planted image-space ground truth.

The positive question for the paper is:

> Under matched controls, where does a clinical concept remain linearly available, where can it change
> behaviour, and how closely do learned read and write directions follow the displacement caused by the
> finding itself?

## Claims and evidence path

| Claim | Measurement | Decision rule |
|---|---|---|
| Clinical concepts have a measurable availability profile through the model | patient-level linear probes at registered loci | real-label AUROC, permutation null, type-to-label controls and nuisance probes reported together |
| Availability and behavioural use can diverge | matched-locus interventions along probe and behaviour-fitted directions | held-out dose response exceeds equal-norm random, sham and unrelated-concept controls |
| The divergence has a location along the vision-to-language path | decodability and steerability compared at the same loci | locus-wise effect with bootstrap uncertainty and control spread |
| A named direction corresponds to image content | planted findings of known location and intensity | image-space and representation-space dose responses agree, with an anatomical location control |
| Domain training changes the profile | architecture-matched medical/general models | patient-, concept-, locus- and readout-matched comparison |

Claim strength follows the completed gate. A curve supports availability; an intervention supports use;
the planted comparison anchors the semantic interpretation.

## Study design

### Models

The main study uses models around the 7B scale and varies domain training and connector architecture.

| Model | Domain | Connector | Role |
|---|---|---|---|
| LLaVA-Med-7B (`microsoft/llava-med-v1.5-mistral-7b`) | medical | linear/MLP projector | medical LLaVA-family model |
| LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`) | general | linear/MLP projector | general LLaVA-family comparison |
| Qwen2.5-VL-7B (`Qwen/Qwen2.5-VL-7B-Instruct`) | general | PatchMerger | architecture match for Lingshu |
| Lingshu-7B (`lingshu-medical-mllm/Lingshu-7B`) | medical | PatchMerger | architecture match for Qwen2.5-VL |
| InternVL3-8B (`OpenGVLab/InternVL3-8B-hf`) | general | pixel shuffle + MLP | third connector type |

CheXagent-2-3B is a specialist comparison for report-generation endpoints rather than a headline model.

### Data and concepts

NIH ChestX-ray14 supplies the primary patient-level study. The clinical concepts are Effusion,
Pneumothorax, Cardiomegaly, Consolidation, Atelectasis, Edema, Infiltration, Mass and Nodule. View
position and patient sex are measured as nuisance variables. The randomised control task assigns fixed
random labels to recurring image types built from acquisition and demographic metadata.

Dermoscopy provides the cross-modality replication. Diagnosis is the clinical target; anatomical site,
sex, age and acquisition type provide the matched nuisance and control-task structure.

### Loci and readout

| Locus | Representation | Pooling |
|---|---|---|
| vision blocks, including the actual consumed final block | vision-tower hidden state | mean over patch tokens; localised/max sensitivity where registered |
| connector | visual tokens entering the language model | mean over visual tokens |
| language layer, visual positions | residual stream at image-token positions | mean over visual positions |
| language layer, answer position | residual stream at the final prompt position | no pooling |

The hook test must show that an intervention reaches the model output. For LLaVA models configured with
`vision_feature_layer = -2`, the consumed CLIP block is `encoder.layers.22`; this is the registered
`vis.last` locus.

Every probe uses the same 512-dimensional Gaussian projection and logistic readout (`C=1`, seed 0), with
patient-disjoint train and test partitions. Patient-level bootstrap intervals and repeated type-to-label
control draws quantify sampling and control-map uncertainty.

## Experiments

### D0: utilisation

Compare three AUROCs on the same held-out rows: a probe on a randomly initialised architecture (`F`), a
probe on the trained model (`T`) and the model's behavioural answer (`B`). Report
`(B - F) / (T - F)` with its denominator and uncertainty. This measures how much of the decodable gain
introduced by training appears in behaviour.

### D1: controlled intervention

At each registered locus, unit-normalise the concept direction and apply it at visual or answer positions.
Scale each token by its own activation norm so alpha has the same relative meaning across loci. Sweep both
signs and report the full dose response.

Controls are equal-norm random directions, unrelated-concept directions and a coordinate-permutation
sham. A result passes when the held-out concept effect exceeds the registered random-direction band,
retains the expected sign and remains specific to the target concept.

### D2: availability and write directions

Fit the capacity-matched probe at every locus. Alongside its normal `v_probe`, fit a unit direction
`v_cad` on behaviour with a specificity penalty:

`target effect - lambda * mean absolute off-target effect`.

The behavioural objective uses no clinical labels. Report steering effect, specificity cost, readout
AUROC, cosine with `v_probe` and the interpolation frontier between the two directions.

### D3: planted anchor

Plant a synthetic finding of known location and intensity into an image the model reads as normal.

| Stage | Measurement |
|---|---|
| D3a | representation displacement and availability across lesion intensity |
| D3b | localisation against an outside-thorax placement control |
| D3c | image-space dose response compared with steering along `v_probe` and `v_cad` |

The image-space response must be monotone before representation-space equivalence is evaluated.

## Validity battery

| Alternative explanation | Registered control |
|---|---|
| probe capacity | fixed 512-dimensional projection, fixed linear readout, repeated type-to-label control |
| pipeline artefact | within-train label permutation |
| input statistics rather than training | randomly initialised architecture |
| acquisition or demographic shortcut | view-position and sex probes; nuisance-controlled sensitivity |
| dimensionality or pooling choice | common projection dimension; registered pooling sensitivity |
| split leakage | deterministic patient-level split and zero cross-split patients |
| intervention scale | token-relative scaling, both signs and dose response |
| generic or off-manifold change | random, sham and unrelated-concept controls; activation-norm and subspace checks |
| semantic misidentification | planted finding with anatomical location control |
| architecture-specific effect | matched model pairs and multiple connector types |

## Work order and gates

| Phase | Run | Observation | Gate decision | Next step |
|---|---|---|---|---|
| P0 | architecture and hook preflight | registered locus changes downstream logits | hook reaches the real forward path | build immutable activation run |
| P1 | one model, one concept, one locus | probe, bootstrap and 20 control draws complete | uncertainty and control spread are valid | run the registered intervention grid |
| P2 | primary-model locus profile | availability and use measured at matched loci | control-qualified profile complete | expand across concepts and models |
| P3 | matched model pairs | domain contrast replicated under the same protocol | architecture-qualified comparison complete | cross-modality replication |
| P4 | planted anchor | image- and representation-space dose responses available | semantic anchor passes its monotonicity and location controls | write the anchored result |

The immediate gate is `llava-effusion-vislast-reltoken`: LLaVA-1.5-7B, Effusion, the consumed
`encoder.layers.22` representation exposed as `vis.last`, a 512-dimensional `C=1` probe with 2,000
patient bootstraps and control seeds 0–19, followed by the registered token-relative intervention on 200
held-out images with 20 random directions.

## Paper structure

1. Problem: availability is routinely used as a proxy for behavioural use.
2. Measurement: matched readouts, patient-level data, nuisance controls and intervention controls.
3. Availability: controlled depth profiles and architecture-matched comparisons.
4. Use: matched-locus dose responses and behaviour-fitted directions.
5. Anchor: planted image-space displacement against representation-space steering.
6. Implications: what each evidence level licenses for medical VLM interpretation.

This project remains separate from `research/medvlm-agentic-rl`: Concept Flow is the measurement study;
the other project is a training method and shares no code, data or compute.
