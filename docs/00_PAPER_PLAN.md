# Concept Flow: paper plan

Status: planning, 2026-08-26. Local experiments, this machine.
Separate project from `research/medvlm-agentic-rl` (that one is agentic RL training and moved to another
server). Nothing here is shared with it except the survey knowledge behind both.

---

## 0. Novelty gate, run 2026-08-26: FAILED as originally stated, repositioned

Before building the argument, the plan was checked against the current literature. Two papers, both
verified to exist (HTTP 200 on arXiv, titles match), already occupy the ground the first draft of this
plan claimed:

| Paper | What it already does |
|---|---|
| Zhu et al., **"Lost in the Hype: Revealing and Dissecting the Performance Degradation of Medical VLMs"**, arXiv:2604.08333 (2026-04-09) | Probes vision-tower layers, connector layers, every LLM layer and the semantic output, across 14 medical MLLMs and 3 datasets. This is D2's skeleton, in medicine |
| Theodoridis et al., **"Probing Visual Concepts in Lightweight Vision-Language Models for Automated Driving"**, arXiv:2603.06054 (2026-03-06, rev 2026-08-06) | Linear probes at every vision-encoder layer, the projector output and every LLM layer, pooled at both visual-token positions and the last token, and then uses the probe weight vectors as steering vectors with a scale factor. This is D1 and D2 combined |

The medical survey behind this project had a cutoff of 16 August 2026 and recorded exactly one study
following a concept from encoder through connector (statement F8b). That was accurate then. It is not a
claim we can make now.

**Any framing of the form "first to trace a concept through the pipeline" is dead and would be desk
rejected.** The repositioning below is what survives, and it is arguably a better fit for this author.

### What survives, and why it is still a strong paper

The two prior papers measure decodability curves and read them as maps of where the information is.
Neither controls the measurement. That is the opening.

| Surviving contribution | Why the prior work does not have it |
|---|---|
| **The validity battery on the curve.** Permutation null, random-encoder floor, capacity-matched readout, nuisance probes, and a randomised control task, at every locus | The survey found a control task in **0 of 135** medical studies, a permutation null in 3 of 69, a random encoder in 5 of 121. Neither prior paper reports any of them. An uncontrolled curve cannot separate information from probe capacity |
| **D3c, the planted anchor.** Compare an intervention in image space against an intervention in representation space and ask whether they produce the same behavioural effect | Neither paper plants ground truth, and no surveyed medical record reports this comparison. It is the strongest available evidence that a concept vector means what it is named |
| **The decodability-steerability dissociation, quantified.** Is the locus where a concept is most decodable the locus where intervening on it works best? | Theodoridis steers with probe weights but, on the current reading, does not test whether peak decodability predicts peak steerability. **Verify this by reading the paper in full before committing.** If they do test it, fall back to the battery and D3c |
| **A matched medical-versus-general pair.** Lingshu-7B and Qwen2.5-VL-7B are architecturally identical, 28 layers, hidden 3584, same connector, differing only in medical training | Zhu et al. compare 14 medical models to each other. A matched twin isolates domain training from architecture, which their design cannot |

### The repositioned claim

Not "we are the first to trace a concept through a medical VLM". Instead:

> The field now has decodability curves through medical VLMs and reads them as evidence about where
> information lives. We show what happens when the controls the field never runs are applied to those
> curves, and we anchor the whole measurement against a planted ground truth. The controlled curve is
> not the uncontrolled curve, and the locus where a concept is most decodable is not the locus where
> intervening on it works.

That is a correction paper with a positive result attached, and the systematic survey behind it is what
licenses the claim that the controls are missing field-wide rather than in two papers we happened to pick.

### Decision on how hard to gate

Taken 2026-08-26 by the principal investigator: **do not gate the project on novelty.** State the
differences from the two prior papers clearly and move on. The deliverable is a complete piece of work,
and the narrative angle is a choice we make rather than a constraint the literature imposes.

The angle chosen, and every section is written to it:

> **The controlled measurement.** Others have drawn the curve. We are the ones who checked whether the
> curve means what it appears to mean, and anchored it against a ground truth we planted ourselves.

Practically this changes three things and nothing else:

1. Related work states the delta in a table rather than arguing for priority. Both papers are cited in
   the first paragraph, generously, and the contribution is stated as "controlled and anchored", never
   as "first".
2. We still read both papers in full, but to extract their protocol so ours is a superset, not to look
   for a gap to squeeze into. Where they made a reasonable choice, we copy it, and say so.
3. Effort moves from positioning to completeness. Every model, every locus, every concept, every
   control, bootstrap intervals, multiple seeds. A complete result table is the contribution that
   survives a reviewer who disagrees with the framing.

---

## 1. What the paper claims

Medical multimodal LLMs are trained and deployed on the assumption that what their vision encoder
represents is what their language model uses. Nobody has checked. This paper checks it, twice.

> **Framing, fixed.** This is a controlled-measurement paper. Every claim below is stated as "under the
> full control battery" and none is stated as "first". See §0 for why.

**Claim A, where the concept goes.** For a named clinical finding, we measure whether it is linearly
decodable at every stage of the computation, from the vision encoder through the connector into every
layer of the language model. The result is a survival curve, and it is the first controlled version of a
measurement that has been made before but never with a permutation null, a random-encoder floor, a
capacity-matched readout, or a control task.

**Claim B, whether the model uses it.** At the loci where the concept survives, we intervene on the
direction that encodes it and measure whether the model's output moves, against the full battery of
matched controls. This answers the question the field has been substituting decodability for.

**Claim C, the dissociation.** Our expectation, stated in advance so the paper is falsifiable either way:
decodability at a locus will not predict use at that locus, and the divergence will be largest after the
connector. If the two track each other instead, that is also a publishable result and it makes probing
much more useful than the field currently has grounds to assume.

**Claim D, the anchor.** We validate both measurements against a planted ground truth. Take a normal
radiograph, insert a synthetic finding at a known location with a known intensity, and we now know that
the concept is present, where it is, and how strong it is. Two things follow that nothing in the
literature currently has:

- the survival curve can be checked against a concept we *know* is there, so a flat curve means the
  information was lost rather than that the probe was weak;
- and, the part that matters most, we can compare **an intervention in image space** (plant the lesion at
  strength `s`) against **an intervention in representation space** (add `alpha * v_c`). If adding the
  direction reproduces the behavioural effect of actually putting the finding in the image, then `v_c`
  encodes the concept in the only sense that matters. That equivalence is the strongest available evidence
  that a concept vector means what it is named, and no medical study in the surveyed 135 reports it.

**Claim E, the method.** We run the complete validity battery that a systematic review of 135 medical
interpretability studies found is almost never run, including a randomised control task that zero of
those 135 studies reported. The battery is the contribution that makes A through D trustworthy rather
than just new.

---

## 2. Why this is a landmark rather than an increment

The first author surveyed 135 studies and 167 individual results in this exact area. That survey produced
a quantitative map of what the field's evidence is missing. This paper is designed against that map.

### 2.1 What the survey found missing, and what we do about it

| Survey statement | Its evidence base | What we do |
|---|---|---|
| **F8b** "Labels less decodable after the connector" | **1 record, 1 study, narrow, uncontrolled** | Claim A turns the field's single uncontrolled observation into a controlled measurement across several models and connector types |
| **F4** "Sex redundant for a refitted probe" | **1 record** | Claim B generalises the availability-versus-use test from one demographic attribute to clinical findings, with interventions rather than refitting |
| **F9a** "Targeted interventions report changed outputs" | 8 records, only 2 to 3 with a matched control | Claim B runs the matched-control battery on every intervention |
| **F1a** "Clinical findings and anatomy decodable" | 18 studies, broad, but almost no probe controls | Claim A treats decodability as the thing to be validated, not assumed |

### 2.2 The validity battery

The survey coded eleven classes of validity test. Across 135 studies the coverage was:

| Class | Field coverage | Our plan |
|---|---|---|
| Sample-level permutation control | 3 of 69 applicable records | every probe, every locus |
| Reference-distribution calibration | 5 of 110 | every probe |
| **Randomised control task** | **0 of 135 studies** | **yes, see §5.3. This is the methodological first** |
| Matched-object control | 1 of 80 | yes |
| Planted ground truth | 4 of 80 | yes, and extended into D3c, which no surveyed record reports |
| Learned-representation attribution (random encoder) | 5 of 121 | every locus |
| Comparison model | 16 of 30 | medical against general, matched architecture |
| Input-space corroboration | 6 of 11 | occlusion at the localised region |
| Internal-component perturbation | 2 of 24 | yes, that is Claim B |
| Intervention specificity (matched controls) | 3 of 8 | every intervention |

Running the battery is not thoroughness for its own sake. Each class rules out one specific alternative
explanation, and the survey showed that when they are absent the resulting claims are not interpretable.
A paper that runs all of them can say things the rest of the literature cannot.

### 2.3 Design lessons taken from the surveyed literature

Things the field does badly, that we avoid by construction:

1. **Probes are fitted with no capacity control**, so a strong probe reads the decoder's power rather than
   the representation. We fix decoder class and capacity across every locus, and report selectivity, not
   accuracy.
2. **Loci are not comparable.** Different studies pool differently and then compare numbers. We fix the
   readout across loci and state the pooling in every figure.
3. **Interventions report a change and stop.** We pre-specify magnitudes, run both signs, and require the
   effect to beat matched controls before we call it selective.
4. **Splits leak.** Chest radiograph datasets have multiple studies per patient. Every split here is
   patient-level, and the split is stated in every table.
5. **Single model, single dataset.** Every headline claim is replicated across models with different
   connector types and across at least two datasets.
6. **Nuisance concepts are ignored.** We probe for view position, sex and acquisition source alongside the
   clinical findings, because the survey's most consistent finding was that acquisition variables are the
   most recoverable thing in medical representations. If a "finding" probe is really reading view
   position, the nuisance probes expose it.

---

## 3. Experimental design

### 3.1 Models

All at the 7B scale by decision. Chosen so that connector type and domain are both varied, and so that at
least one pair differs only in domain.

| Model | Domain | Connector | Role |
|---|---|---|---|
| LLaVA-Med-7B (`microsoft/llava-med-v1.5-mistral-7b`) | medical | linear projector | primary. Simplest connector, so the survival curve is cleanest |
| LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`) | general | linear projector | domain control against LLaVA-Med, same framework |
| Qwen2.5-VL-7B (`Qwen/Qwen2.5-VL-7B-Instruct`) | general | MLP merger, dynamic resolution | second connector type |
| LLaVA-OneVision-7B (`llava-hf/llava-onevision-qwen2-7b-ov-hf`) | general | MLP, Qwen2 backbone | isolates backbone from connector |
| InternVL3-8B (`OpenGVLab/InternVL3-8B-hf`) | general | pixel shuffle + MLP | third connector type |
| Lingshu-7B or HuatuoGPT-Vision-7B | medical | MLP | second medical model. Needs download |

CheXagent-2-3b is cached and is a chest-radiograph specialist that generates reports, which makes it
valuable for Claim B's end-point. It is 3B, so it appears as a specialist comparison and never as a
headline number.

### 3.2 Concepts

| Group | Concepts | Why |
|---|---|---|
| Clinical findings | pleural effusion, pneumothorax, cardiomegaly, consolidation, atelectasis, pulmonary edema | the target |
| Anatomy | presence of a support device, lung field laterality | a concept the model must represent to answer anything |
| **Nuisance** | view position (PA / AP / lateral), patient sex, source dataset | the survey's most consistent finding is that these are the most recoverable properties. If a finding probe tracks these instead, we detect it |
| **Control task** | a random label assigned per input *type*, see §5.3 | the selectivity denominator |

### 3.3 Loci

The measurement is only meaningful if the readout is comparable across loci, so this is fixed in advance:

| Locus | What is read | Pooling |
|---|---|---|
| Vision encoder, patch tokens | last hidden state | mean over patches, and separately max |
| Vision encoder, pooled | the pooled or class token | as is |
| **Connector output** | the visual tokens entering the LLM | mean over visual tokens |
| LLM layer `l`, visual positions | residual stream at the visual token positions | mean over those positions |
| LLM layer `l`, answer position | residual stream at the final prompt position | as is |

Every layer of the LLM, not a subsample. For a 7B model that is 32 layers, two position sets, so 64
readouts plus 3 pre-LLM ones per model per concept. That is cheap: one forward pass caches all of them.

### 3.4 The two experiments

**D2, the survival curve.** For each model, concept and locus: fit a linear probe on a patient-level
training split, evaluate AUROC on held-out patients, and report the curve against depth, with the full
control battery at every point. Deliverable: a figure per model, concepts as lines, depth as the x axis,
with the permutation null and the random-encoder floor drawn as shaded bands.

**D1, the use test.** At the loci where D2 says the concept is present, take the probe normal `v_c`, add
`alpha * v_c` to the residual stream during generation, and measure the behavioural change. Controls,
magnitudes and end-points in §5.4. Deliverable: a dose-response figure per locus, with the matched-control
band, and a table of which loci show selective sensitivity.

**D3, the planted anchor.** Synthesise findings into normal radiographs at known location and strength,
then run D2 and D1 on them. Three sub-experiments:

| Sub | What it does | What it establishes |
|---|---|---|
| D3a | Survival curve on planted images | The curve measures information loss, not probe weakness, because the concept is known to be present |
| D3b | Localisation check: does the probe's attention concentrate on the planted region? | The probe reads the finding rather than a global correlate |
| D3c | **Image-space against representation-space intervention.** Plant at strength `s`, sweep `s`; separately steer with `alpha * v_c`, sweep `alpha`. Compare the two dose-response curves | If they match, `v_c` reproduces the effect of the finding itself. This is the paper's strongest single result |

Planting is the validity class the survey found in 4 of 80 applicable records. D3c is not in any of them.
Deliverable: two dose-response curves on the same axes, and the alpha that matches a given `s`.

---

## 4. What would make this wrong

Stated in advance. Each has a check that catches it.

| Failure mode | Check |
|---|---|
| The probe reads decoder capacity, not the representation | control task, selectivity rather than accuracy |
| The probe reads a nuisance variable correlated with the finding | nuisance probes on the same features; a finding probe that tracks view position is reported as such |
| The survival curve reflects dimensionality, not information | capacity-matched readout, and a random projection to a common dimension as a sensitivity |
| The steering effect is a norm artefact | equal-norm random directions, both signs, dose-response |
| The steering effect is off-manifold garbage | check the edited activation stays within the layer's empirical norm range and principal subspace; report fluency |
| The effect is one model's quirk | replicate across connector types |
| Split leakage inflates everything | patient-level splits everywhere, stated in every table |
| The concept never appears in the output vocabulary | pick concepts the model demonstrably mentions, verified before the intervention arm |

---

## 5. Open design decisions

Being resolved by the reconnaissance workflow running now. Recorded so they are not silently defaulted.

1. **Data.** No credentialed data: MIMIC-CXR and CheXpert are out. Candidates are VinDr-CXR, NIH
   ChestX-ray14 with `BBox_List_2017`, PadChest, Open-i. The blocking question is whether any open set has
   free-text reports, because that decides D1's end-point.
2. **Hooking.** Plain torch forward hooks against a library. Preference is plain hooks: the base env has
   torch 2.7 and transformers 5.2 and we do not want to disturb it.
3. **The control task construction.** §5.3 below. This is the methodological novelty and it needs to be
   right.
4. **The intervention end-point.** Logit of the answer token for VQA, or mention of the finding in a
   generated report. The second is better and needs a report-generating model.

### 5.3 The randomised control task, adapted to medical images

Hewitt and Liang's control task assigns each input *type* a fixed random label, then refits the probe. A
probe that scores as well on random labels as on real ones is reading decoder capacity, not the
representation. Zero of the 135 surveyed studies ran one, because language has recurring types (word
identity) and images do not.

Medical images do have types that recur and that cross a patient-level split: **view position, source
dataset, scanner or acquisition protocol, and annotated region class.** Assign each type a fixed random
label, refit the same probe class with the same capacity, and report selectivity as real minus control.

If this works it is a reusable contribution independent of our findings, and it is the first time the
field has a selectivity number at all.

### 5.4 Intervention protocol

- `v_c` from the probe normal, unit-normalised. Fallback: difference of class means.
- Applied at the visual token positions and separately at the answer position, at one locus at a time.
- Magnitudes: pre-specified multiples of the layer's median activation norm, both signs, at least five
  points, so the result is a curve and not a single number.
- Controls, all equal-norm: 20 random directions, 5 unrelated-concept directions, 1 sham (a permutation of
  `v_c`'s coordinates).
- Statistic: the effect of `v_c` must exceed the 95th percentile of the random-direction effects, on
  held-out inputs disjoint from those that fitted `v_c`.
- End-point: change in the probability of the finding's token, and for report generators whether the
  finding is mentioned, scored by exact phrase matching plus a checked LLM judge.

---

## 6. Phases

| Phase | Content | Exit criterion |
|---|---|---|
| P0 | Environment, hooking, one model loading and caching activations | `src/hook_demo.py` prints a changed logit after a hook writes to the residual stream |
| P1 | Data: open CXR set with labels, patient-level splits, concept counts | every concept has a stated positive count in train and test |
| P2 | D2 on the primary model, one concept, full control battery | a survival curve with permutation and random-encoder bands |
| P3 | D2 across all models and concepts | the full figure set |
| P4 | D1 at the loci P3 identifies, full control battery | dose-response curves with control bands |
| P5 | D3, the planted anchor, including the image-space against representation-space comparison | the two dose-response curves on the same axes |
| P5b | Replication and nuisance analysis | every headline claim holds on a second model and a second dataset |
| P6 | Writing | draft with every number traceable to a run directory |

---

## 7. Relationship to the survey and to the other project

The survey (Neurocomputing, submitted) identified these two gaps and this paper fills them. The
introduction can say that the gap was identified by a systematic review of 135 studies rather than
asserted, and the survey is citable for every "nobody has done X" claim here. That is a real advantage
and it should be used explicitly.

The other project, `research/medvlm-agentic-rl`, extends MedVR with tool-agnostic counterfactual credit
for agentic RL. It shares the intellectual ancestry, since its CATA component generalises the
probe-then-intervene design of Claim B, but it shares no code, no data and no compute with this paper. Do
not merge them. If both succeed, this paper is the measurement result and that one is the training
method, and each cites the other.
