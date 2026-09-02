# Related work: the evidence base

Compiled 2026-08-27. This file is the source material for the paper's related-work section. It is not
the section itself. Every claim about another paper here was checked against that paper's full text,
downloaded and searched locally, not against its abstract. Where a paper is quoted, the quotation is
verbatim from the arXiv HTML of the version named.

**How to reproduce the full-text checks.** The extracted text lived in a session scratch directory and is
gone. Regenerate it rather than trusting this file:

```
curl -sSL -o t.html https://arxiv.org/html/2603.06054v2
curl -sSL -o z.html https://arxiv.org/html/2604.08333v1
# strip <script|style|svg>, then all tags, unescape entities, collapse whitespace
```

Stripped sizes depend on the stripping rule, so no character count is quoted here; an earlier revision of
this file quoted 119,844 and 67,503 chars, which a second independent strip did not reproduce (119,829 and
66,827). Every quotation below was re-verified verbatim against a fresh download on 2026-08-27 by an
independent auditor, with two exceptions noted under "Quote hygiene" at the end of §1.2.

## 0. Status of our own evidence at the time of writing

The delta table in §2 marks four rows as depending on D1. D1 is not finished, and **no D1 number in this
document or the paper may be quoted yet**.

Regenerate the status with `python src/analyze_intervention.py`, which reads the `DONE` marker for every
run directory and prints finished and partial jobs separately.

What is stable as of 2026-08-27 15:00 EDT, re-verified by an independent auditor: `find runs/ -name "DONE*"`
returns **nothing**, and `src/supervise.sh` reports `running=8 done=0 pending=7` against a queue of 15 jobs
(5 models x 3 concepts). Nine `runs/int_*` directories exist and several hold a partial `intervention.csv`.

**No per-directory row counts are recorded here, on purpose.** An earlier revision listed three directories
as "1216 / 244 / 235 rows"; those were `wc -l` line counts including the CSV header, and they described a
smaller, earlier queue state. A replacement table written at 14:57 was stale again by 15:00, when
`int_llavamed7b_cardiomegaly` went from 468 to 702 data rows and `int_llavamed7b_effusion` went from empty
to 243. A frozen count of a live queue decays into a false claim, so this file does not carry one.

The derived files `runs/intervention_summary.csv` and `runs/decodability_vs_steerability.csv` exist despite
no job being finished. Both carry `job_done=False` on **every** row, which the auditor confirmed directly
and which must stay that way. A CSV beside no `DONE` marker is a live job, not a result.

D2 numbers used below come from the five current probe runs, 3060 rows total:

| File | Rows |
|---|---|
| `runs/probe_lingshu7b/probe_results.csv` | 594 |
| `runs/probe_qwen7b/probe_results.csv` | 594 |
| `runs/probe_llavamed7b/probe_results.csv` | 648 |
| `runs/probe_llava15_7b/probe_results.csv` | 648 |
| `runs/probe_internvl3_8b/probe_results.csv` | 576 |

Pooled over all 3060 rows: median `real` 0.717, median `control` 0.671, median `selectivity` 0.062.
Median `selectivity` by concept: Nodule -0.110, Mass -0.021, Atelectasis 0.029, Pneumothorax 0.048,
Effusion 0.062, Cardiomegaly 0.068, Consolidation 0.087, Infiltration 0.101, Edema 0.197.
`nuis_view_AP` spans 0.9964 to 0.9994 across all rows (exactly 0.996415 to 0.999367). `n_types` is 39 in
every row. `selectivity` equals `real - control` in all 3060 rows. 74 distinct loci: 64 `llm.*`, 9 `vis.*`,
and `connector`. The permutation null runs 0.4046 to 0.5869 over all 3060 rows.

Independently recomputed on 2026-08-27 by an auditor reading the CSVs directly. Every figure in this
subsection reproduced to the digit except where the blockquote below says otherwise.

> **RESOLVED. The brief's headline number is the peak statistic, and it reproduces.** An earlier revision
> of this file claimed the brief's "median AUROC about 0.75, control about 0.69, median selectivity +0.084"
> did not reproduce "under any of four aggregations", and quoted **0.726 / 0.681 / 0.069** for the "per
> model-concept peak". That triple is wrong: an independent audit swept 108 aggregation variants (9 locus
> subsets x 4 grouping keys x 3 ranking keys x 3 aggregators) and **no variant produces it**. It is
> withdrawn and marked UNKNOWN — nobody has been able to say what computation it came from, and it must
> not be quoted.
>
> The peak statistic, recomputed directly from the five CSVs, is the brief's number. Take the argmax row
> per (model, concept) — 45 cells — and report the median:
>
> | Peak search ranges over | median `real` | median `control` | median `selectivity` |
> |---|---|---|---|
> | all 74 loci, argmax `real` | 0.7526 | 0.6895 | **+0.0881** |
> | LLM loci only, argmax `real` | 0.7526 | 0.6772 | **+0.0844** |
> | visual-position loci only (`*.vis`), argmax `real` | 0.7507 | 0.6832 | **+0.0764** |
>
> The brief's 0.75 / 0.69 / +0.084 is the peak-per-(model, concept) statistic. The only open question is
> which locus set the peak search ranges over, which moves selectivity between +0.076 and +0.088.
>
> `docs/10_RESULTS.md` §2.2, written independently after this file, lands in the same place: it reports
> +0.0764 for the pre-registered visual-position track, notes the earlier +0.084 explicitly, and gives
> +0.0881 for the search widened to all loci. Its own 45 tabulated peak rows median to 0.751 / 0.685 /
> +0.076, agreeing with the table above on `real` and `selectivity` but differing by 0.002 on `control`,
> because its peak search admits `vis.last` and `connector` in 2 of the 45 cells while the row above is
> `*.vis` only. That 0.002 is a locus-set difference, not a disagreement about the data, and the two files
> should say which set they mean rather than being reconciled to the third decimal.
>
> **The results section should quote the visual-position track and state its scope in the same sentence**,
> because that is the pre-registered locus set; the other two rows exist here so nobody re-derives them by
> hand.
>
> The other three aggregations in the earlier revision were correct and are retained: pooled over all 3060
> rows 0.7168 / 0.6707 / +0.0623; LLM loci only, pooled, 0.7160 / 0.6690 / +0.0620; mean per (model,
> concept) then median 0.7165 / 0.6674 / +0.0622.
>
> The two older directories `runs/probe_act_lingshu7b/` and `runs/probe_act_qwen7b/` are superseded: a
> leaking control task, median `control` 0.9983 and 0.9981, median `selectivity` **-0.3029 and -0.3062**.
> An earlier revision gave "-0.303" for both; that is the lingshu figure only.
>
> The qualitative findings are unaffected and all reproduce: Nodule is negative on the pooled median, Mass
> is near zero, view position decodes above 0.996 everywhere, and selectivity is far below surface AUROC.

---

## 1. The two papers the positioning depends on

Both are cited generously in the first paragraph of the section. Neither is minimised. The paper's claim
is that ours is the controlled version of a measurement these two made first, and that claim is only
credible if their contribution is stated accurately.

### 1.1 Zhu et al., "Lost in the Hype"

Xun Zhu, Fanbin Mo, Xi Chen, Kaili Zheng, Shaoshuai Yang, Yiming Shi, Jian Gao, Miao Li, Ji Wu.
*Lost in the Hype: Revealing and Dissecting the Performance Degradation of Medical Multimodal Large
Language Models in Image Classification.* arXiv:2604.08333, submitted 2026-04-09.
https://arxiv.org/abs/2604.08333 . Full text read at https://arxiv.org/html/2604.08333v1 .

| Protocol element | What they did |
|---|---|
| Models | 14 medical MLLMs, 2.21B to 34.75B: MedVLM-R1-2B, HuatuoGPT-Vision-7B/34B, Lingshu-7B/32B, ShizhenGPT-7B/32B-VL, Hulu-Med-4B/7B/14B/32B, MedGemma-4B/27B, MedGemma-1.5-4B. Plus 23 traditional models as reference: 8 CNN, 4 MLP, 4 ViT, 7 hybrid |
| Datasets | BUSI breast ultrasound, 780 images, 3 classes, 600 patients; COVID19-CT, 746 images, 2 classes, 349 positive from 216 patients; Chest-Xray (Kermany), 5856 images, 2 classes, paediatric cohort aged one to five |
| Concepts | Only the dataset's own class label. One concept per dataset, all diagnostic. No anatomy, no nuisance variables |
| Loci | Every layer of the vision tower, the connector, every LLM layer, and the final semantic output. 84 probing accuracy curves, being 14 models x 3 datasets x 2 conditions (before and after dataset-specific LoRA SFT) |
| Pooling | Not specified for the general case. Two named cases only: comprehension is measured by "probing averaged image token embeddings", utilisation by probing "the final input token". The paper gives no pooling rule for vision tower or connector layers |
| Probe class | **Not linear.** "a lightweight probing head consisting of a two-layer MLP with ReLU activation and dropout rate 0.1". AdamW, lr 1e-4, cosine schedule, warmup 0.05, 20 epochs, batch size 4, backbone frozen, trained twice, on 8 A800 GPUs and 8 Ascend 910B NPUs |
| Controls | **None of the kind this paper runs.** No permutation null, no control task, no random-encoder floor, no chance line, no capacity matching across loci of different width. Their comparisons are: vision tower frozen against LoRA fine-tuned; MLLM SFT against probing against vision-tower SFT; medical model against its general base |
| Splits | Not stated. No patient-level split, no leakage discussion, although BUSI and COVID19-CT are described as multi-image-per-patient cohorts |
| Statistic | Accuracy is the primary metric. F1, precision, recall and AUC appear in supplementary. Plus a composite **Feature Health Score**, `FHS_M = P(M,end) * (1 + GF_M) * VP_M`, with growth factor `GF_M = ln(P(M,end) / sqrt(P(M,start) * P(M,min)))` and volatility penalty `VP_M = exp(-lambda * sum_i |P(M,i+1) - P(M,i)| / P(M,end))`, lambda 0.2. Reported as a four-part profile `FHS_V -> FHS_C -> FHS_L -> P_final` |

**Headline findings.**

1. Four failure modes: quality limitation in the vision tower, fidelity loss in the connector,
   comprehension deficit in LLM reasoning, misalignment of semantic mapping.
2. The connector is close to inert. Verbatim: "the connector primarily performs dimensional alignment
   between the vision tower's output space and the LLM's input embedding space, rather than functioning
   as a sophisticated feature refiner", and "the connector does not substantially enhance or impair the
   discriminative information relevant to classification."
3. Medical adaptation buys very little. Against the same base model, HuatuoGPT-Vision-7B, Lingshu-7B and
   ShizhenGPT-7B-VL gain 1.37%, 1.27% and 0.59% average accuracy. MedGemma-4B is 1.66% **below**
   Gemma-3-4B; MedGemma-1.5-4B is 1.17% above.
4. Final generated accuracy departs from last-layer probe accuracy in both directions: "sometimes lower,
   occasionally higher, but rarely equal".
5. Vision-tower SFT with cross-entropy beats MLLM SFT across datasets.

**Two things worth noting for our framing.** First, the paper is probing-only. It performs no
intervention, no steering, no patching, no editing. Its causal language is inference from curve shape.
Second, it cites Hewitt and Liang (2019) once, at line 176 of the extracted text, in the sentence
"Probing originates in natural language processing, where it reveals that language models encode
hierarchical linguistic information in a layer-wise fashion (Van Aken et al., 2019; Hewitt and Liang,
2019)." That is a citation of the control-task paper for a claim the control-task paper does not make.
The control-task construction is never used. This is a clean, checkable instance of the pattern our
survey quantified as 0 of 135, and it can be stated in the paper without editorialising, because the
citation and the absence of the method are both verifiable facts.

### 1.2 Theodoridis et al., "Probing Visual Concepts in Lightweight VLMs for Automated Driving"

Nikos Theodoridis, Reenu Mohandas, Ganesh Sistu, Anthony Scanlan, Ciaran Eising, Tim Brophy.
*Probing Visual Concepts in Lightweight Vision-Language Models for Automated Driving.* arXiv:2603.06054,
submitted 2026-03-06, v2 2026-08-06. Published in Transactions on Machine Learning Research, 2026.
https://arxiv.org/abs/2603.06054 . Full text read at https://arxiv.org/html/2603.06054v2 .

| Protocol element | What they did |
|---|---|
| Models | Six variants of five architectures, **all under 4B**: Ovis2.5-2B (2.57B), InternVL3.5-2B (2.35B), Qwen3-VL-2B (2.13B), VST-3B (3.75B, spatial-reasoning fine-tune, in SFT and RL variants), DriveFusionQA (3.75B, driving fine-tune) |
| Datasets | Counterfactual image pairs generated in CARLA. Towns 01 to 07 and Town10HD for training (400 samples per class per distance), Town12 validation (50), Town15 test (50), **with two exceptions the paper states and an earlier revision of this row omitted: Spatial-1 uses Town10HD as validation instead, and Spatial-2 uses Town01 and Town02 for both training and validation and Town07 for testing**. Object distances 5, 10, 20, 30, 40, 50 m; Spatial-2 has no 5 m version. 1920x1080, 90 degree FOV. Out-of-distribution check on real nuScenes data, for presence and count only |
| Concepts | Four, two instantiations each. Presence-1 pedestrian, Presence-2 traffic barrel; Count-1 and Count-2 (0 to 4 objects); Spatial-1 truck blinker left/right, Spatial-2 pedestrian side of road; Orientation-1 pedestrian direction, Orientation-2 bicycle direction. No nuisance concepts |
| Loci | Every vision encoder block, the projector output, every LLM block, and the post-layernorm output |
| Pooling | Three named strategies, stated explicitly. (a) average-pooling: element-wise mean over all patch vectors. (b) region-pooling, spatial and orientation tasks only: split the image, mean-pool left and right regions independently, concatenate, which "preserves minimal spatial structure". (c) in the LLM: mean over visual token positions **concatenated with** the last token activation. The probe weight vector therefore has two halves, and they use that structure later |
| Probe class | Linear. Activations standardised per dimension using training-set mean and std, then `z = W f_tilde + b`. They justify standardisation as making cross-layer comparison better conditioned, and note it cannot change linear separability because it is affine and the probe has a bias. lr swept 1e-4 to 5e-1, best chosen on validation, AdamW, 10 seeds, mean and std reported |
| Controls | Chance level per task (50% binary, 20% five-way count), and chance-corrected accuracy. Wilson score 95% intervals, Appendix B. Cosine similarity between probe weight vectors for same against different concepts. OOD evaluation on nuScenes. **No permutation null, no control task, no random-encoder floor, and, in the steering experiment, no control directions of any kind** |
| Statistic | Chance-corrected accuracy `a' = (a_o - a_c) / (1 - a_c)`, stated as mathematically equivalent to Cohen's kappa, bounded at zero. Mean over 10 runs |

**The steering experiment, in detail, because our positioning turns on it.**

Section 5.2.2 and Appendix C. This is a real experiment with a quantitative result, not a demonstration,
and the paper should be described that way.

- The direction is the element-wise mean of the probe weights over the 10 best runs, excluding bias
  (Eq. 8). Because probes were fitted on standardised activations, they divide by the standardisation
  std to map back to raw activation space, then normalise: `s = (w / sigma) / ||w / sigma||_2` (Eq. 9).
  So the steering vector **is unit-norm**.
- The edit is `l_visual + alpha * s_first` at visual token positions (Eq. 10) and `l_{t-1} + alpha *
  s_second` at the last token (Eq. 11), using the two halves of the probe vector that correspond to the
  two pooling positions.
- Magnitude: "We performed a small grid search over [alpha], testing five different values per sample. In
  Table 3, we present the results obtained using the smallest absolute value of [alpha] that caused a
  semantic change in the model's description of the scene." They report a threshold alpha, not a
  dose-response curve.
- Locus: LLM layers only. Choice rule, verbatim: "we applied steering at the earliest LLM layer where
  probe accuracy was high."
- Endpoint: free-text description under the prompt "Describe the image briefly.", with task-specific
  additions for the spatial settings. 50 samples per model per category. Success requires the base
  description (alpha = 0) to mention the original concept state and at least one steered description to
  change to the target state while the rest stays accurate. Samples where the base never mentions the
  concept are "undefined" and excluded rather than counted as failures. Success rates in Table 6 are near
  100% for Presence across all models and high elsewhere.
- Orientation is excluded from the steering results: steering produced no effect there.

**On whether they already ran our dissociation test.** They did not, and they say so. Verbatim: "we did
not perform an extensive exploration of other steering-related hyperparameters, such as which model
component or layer is most effective for steering, or whether steering should be applied only to the
visual tokens, only to the last token, or to both within the LLM. Determining the optimal steering
strategy is outside the scope of this experiment." Since they steer at one locus per concept, chosen by
probe accuracy, they cannot compare steerability against decodability across loci. The open design
question recorded in `docs/00_PAPER_PLAN.md` §0 is now resolved: **verified, they do not test it.**

**But they do observe the dissociation once, and we must credit it.** Appendix C, on DriveFusionQA
Count-1: 38 samples gave valid responses and only three steered successfully, while "Figure 4 shows that
Count-1 information is well encoded in the model's activations." Their reading: "although the
fine-tuning undergone by DriveFusionQA preserved the linear accessibility of count information, it may
have weakened the causal influence of this information on the generated output." They then decline the
claim: "because the steering configuration was not exhaustively optimised, the present experiment cannot
isolate fine-tuning as the cause." So the phenomenon our Claim C names appears in their paper as a
single uncontrolled model-by-concept cell, flagged as anomalous and explicitly not attributed. The
correct sentence for our related-work section is that they observed it and declined to claim it, and
that we test it as a designed comparison across loci with matched controls. Not that they missed it.

**Their stated limitations**, which define open ground we occupy: only counterfactual synthetic CARLA
data; only four concepts; only linear probes; and, in their own words, "It would also be interesting to
repeat this study for larger VLMs (i.e., more than 4 billion parameters) to see to what degree the
patterns observed for small models persist." Every model in our study is 7B or larger.

**Quote hygiene.** An independent auditor re-downloaded both papers on 2026-08-27 and re-checked every
quotation in §1.1 and §1.2 verbatim against the fresh text. All of them are present, and every protocol
number in the two tables above was confirmed against the source: 780 / 600 patients, 349 + 397 = 746,
5,856, 2.21B to 34.75B, 8 CNN + 4 MLP + 4 ViT + 7 hybrid = 23, 84 curves, lambda 0.2, AdamW lr 1e-4 with
warmup 0.05 and 20 epochs and batch size 4 on 8 A800 and 8 Ascend 910B for Zhu; 2.57B / 2.35B / 2.13B /
3.75B / 3.75B, learning rates 1e-4 to 5e-1, ten repeats, 400 / 50 / 50 samples per class per distance,
and "38 samples produced valid responses, of which only three were successfully steered" for Theodoridis.
Two mechanical caveats, both introduced by this file and neither affecting meaning: the source renders the
steering coefficient as the Greek letter, which this ASCII-only document transliterates to "alpha" (shown
bracketed above where it falls inside a quotation), and one quotation was previously truncated mid-sentence
without an ellipsis, now restored in full.

**Verified absences** (this is what the delta table rests on, so it was re-run rather than trusted). In the
stripped text of 2603.06054v2 the strings `random direction`, `random vector`, `control direction`, `sham`,
`permut`, `shuffl`, `control task`, `selectivity`, `Hewitt` and `randomly initial` occur **zero** times; the
only two uses of "random" are "random chance" and "50 randomly selected samples". In 2604.08333v1 the string
`control task` occurs exactly once, in the bibliography, as the title of Hewitt and Liang; `permut`,
`shuffl`, `selectivity`, `chance`, `random label`, `randomly initial` and `steer` occur **zero** times, and
`intervention` occurs once, in a conference name. Both absence claims hold.

### 1.3 Two corrections to `docs/00_PAPER_PLAN.md` §0

The plan was written before either paper was read in full. Two of its statements are now wrong and must
not reach the draft.

**Correction 1. The matched medical-versus-general pair is not ours alone.** The plan says "Zhu et al.
compare 14 medical models to each other. A matched twin isolates domain training from architecture,
which their design cannot." Zhu et al. Figure 4 and §5 compare HuatuoGPT-Vision-7B, Lingshu-7B and
ShizhenGPT-7B-VL against Qwen2.5-VL-7B, and MedGemma-4B and MedGemma-1.5-4B against Gemma-3-4B. Lingshu
against Qwen is exactly our pair. What survives is narrower and must be stated narrowly: their
comparison is of accuracy after dataset-specific LoRA SFT, on one label per dataset, with an MLP probe
and no control task. Ours is of controlled probe curves on the unmodified base models across 74 loci and
9 concepts, with selectivity against a 39-type control task, plus nuisance probes. The novel object is
the **locus of peak selectivity** in each twin, not the existence of the twin comparison.

**Correction 2. "The field never runs these controls" is true of medical VLM probing and false of
2026 interpretability generally.** See §3. Within medicine the survey's counts stand and neither focal
paper runs a control task, a permutation null or a random-encoder floor. Outside medicine, several
papers from the last five months run norm-matched interventions, random-initialisation baselines and
Hewitt-Liang control tasks. The battery must be justified as the right instrument for this measurement,
not as an unprecedented one.

---

## 2. Delta table

Status values are `new`, `extension`, or `covered`. "Covered" means someone has published the claim in a
form a reviewer would call the same claim, even in another domain.

| # | Our contribution | Status | Justification |
|---|---|---|---|
| 1 | Concept decodability traced across vision encoder, connector and every LLM layer, in medical VLMs | **covered** | 2604.08333 does exactly this, 14 medical MLLMs, 3 datasets, 84 curves. Do not claim it. Cite it as the measurement we are controlling |
| 2 | The same trace in VLMs generally, with steering attached | **covered** | 2603.06054 does this in driving, probes plus steering, all loci |
| 3 | A randomised control task at every locus, and selectivity rather than accuracy as the reported statistic | **extension** | Not run by either focal paper, and 0 of 135 medical studies in our survey. But Hewitt-Liang controls are used in 2604.02608 (with a 2-layer MLP probe) and shuffled-order controls in 2608.17843, both text LLMs. New for medical imaging and for VLM depth curves; the 39-type construction over image types (view position, source, protocol, region class) is genuinely ours. Claim the construction, not the idea |
| 4 | Permutation null at every locus (implemented), random-encoder floor (**NOT IMPLEMENTED**) | **extension**, **half unbuilt** | The permutation null is real: `src/probe.py` fits every locus on within-split shuffled labels and writes the `perm` column, which runs 0.4046 to 0.5869 over all 3060 rows. **The random-encoder floor does not exist.** An audit on 2026-08-27 found no random-encoder code in `src/` and no such column in any `probe_results.csv`; it is listed in `docs/00_PAPER_PLAN.md` (§0 and the battery table, "every locus") but was never built. Claim the permutation null now and either build the floor or drop it from the battery, exactly as row 11 is handled. Absent from both focal papers. 2608.17843 uses randomly initialised representations as a baseline and finds sketch-level DOF "already highly decodable from randomly initialized representations", which is the same logic in another domain |
| 5 | Capacity-matched readout, random projection to 512 dims so loci of different width are comparable | **new** | 2604.08333 uses an MLP probe of unstated input handling across loci of very different width. 2603.06054 standardises per dimension, which fixes conditioning but not width. No paper found equalises probe capacity across loci. This is a small, defensible methodological first |
| 6 | Nuisance probes (view position, sex) run alongside clinical concepts at the same loci | **extension** | Neither focal paper probes any nuisance variable. 2608.12086 (2026-08-12) probes real-world CXR shortcuts across 17 layers of MedCLIP's ResNet-50 and finds shortcut patterns emerging at different depths, which is the same concern one architecture short of a VLM. Our view-position result (0.9964 to 0.9994, `runs/probe_*/probe_results.csv`) is a stronger version of a finding now in circulation |
| 7 | Nodule negative and Mass near zero once selectivity is applied, i.e. published decodability that does not survive its control | **new** | This is the paper's sharpest empirical result and nothing found reports it. It is the concrete payoff of row 3 and is fully supported by current data |
| 8 | Peak decodability sits deep in the LLM for medical twins and at the vision tower or connector for general twins | **extension** | 2604.08333 characterises LLM curve shapes qualitatively (drop-then-recover, flat, declining) but does not localise a peak per model or tie it to domain training. Their medical-vs-base comparison is on final accuracy. Locus-of-peak as a domain signature is ours |
| 9 | Steering with matched controls: 20 equal-norm random directions, up to 5 unrelated-concept directions, a coordinate-permutation sham, 9 magnitudes both signs | **extension**, **D1 incomplete** | 2603.06054 runs zero control directions and reports a threshold alpha, not a dose-response. 2608.08159 (2026-08-08) does run residual-norm-comparable interventions with held-out operating-point selection and shows an apparent scale trend in steerability dissolves under them, which is our argument in text LLMs. The full control set at every locus is not published; the principle is |
| 10 | Decodability at a locus does not predict steerability at that locus (Claim C) | **covered in general, new in our setting**, **D1 incomplete** | See §3. Published at least four times since April 2026, including in VLMs and in medical text QA. What is unpublished is the comparison across the vision-encoder-to-LLM depth axis, for clinical image concepts, in medical VLMs, at matched loci. Reposition Claim C from discovery to localisation |
| 11 | D3c, image-space planting against representation-space steering, matched dose-response | **new** | Not written yet. Nothing found compares an intervention on the image against an intervention on the residual stream on the same axes. This is the strongest remaining originality in the plan and it is the part not yet implemented |
| 12 | Models all 7B or larger, five of them, three connector types | **extension** | 2603.06054 is entirely under 4B and names larger VLMs as future work. 2604.08333 spans 2.21B to 34.75B and already covers scale in medicine |

Rows 9, 10 and 11 are the paper's remaining differentiators, and rows 9 and 10 cannot be written until
D1 finishes. Row 11 is not implemented, and half of row 4 is not implemented either. If D1 and D3 both
fail to complete, the defensible paper is rows 3, 5, 6, 7 and 8: a controlled re-measurement that
overturns specific published decodability claims. That is a smaller paper and still a real one.

**Three rows are backed by data on disk today and were re-verified by an independent auditor on
2026-08-27**: row 5 (`proj_dim` is 512 in every `probe_meta.json`), row 6 (`nuis_view_AP` and `nuis_sex_M`
are present in all 3060 rows), and row 7 (pooled median selectivity is -0.110 for Nodule and -0.021 for
Mass). Rows 4 (in part), 9, 10 and 11 are not. Nothing else in this table depends on unfinished work.

---

## 3. The 2026 decodability-versus-steerability literature, and a plain verdict

**The verdict the plan asked for: our central dissociation claim, stated as a general claim, is already
published. It is published at least four times, by different groups, in four different settings, and one
of those settings is medical and another is vision-language.** It is not published in our setting. Treat
"four" as a floor rather than a count: an audit of this section's own search output on 2026-08-27 found
two further August 2026 papers that had not been triaged (items 8 and 9 below), which moves the honest
figure to six and reinforces rather than weakens the verdict. The
paper must be repositioned accordingly, and the section below is written so the draft cannot overclaim.

Ordered by how close each is to Claim C. Items 1 to 7 were checked against each paper's arXiv abstract by
an independent auditor on 2026-08-27; every statistic quoted below — 3 of 72, 1 of 14, 5 of 10, 4,032
pairs, 71.6%, 29 configurations, n = 1273, r = -0.002 at p = 0.97, -3.6pp against +0.3pp, 0.91 AUROC,
PECS 0.292, 16 models, 17 models, 0.6B to 72B, 1.4M generations, 150 concepts, 0.7 macro-F1 — matches its
source. Items 8 and 9 were added by that audit.

1. **Nadaf, *Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens*, arXiv:2604.02608,
   2026-04-03, revised 2026-05-08.** https://arxiv.org/abs/2604.02608 . Text LLMs, 12 tasks, 6 models
   from 3 families, 4032 directed cross-template pairs. States the decomposition explicitly: the results
   "decompose the linear representation hypothesis into linear decodability and linear steerability and
   show they come apart". Two things matter for us. First, the direction is opposite to our Claim C
   expectation: steering succeeds where the logit lens cannot decode at any layer, while "the converse,
   decodable without steerable, is nearly empty (3 of 72)". Our Claim C predicts the cell they found
   nearly empty. Second, their decoder is the logit lens plus a tuned lens plus a 2-layer MLP probe with
   a Hewitt and Liang control, not a per-locus fitted concept probe, so "decodable" means something
   narrower than it means for us. This paper is the single strongest reason to state Claim C as a
   measurement rather than a prediction, and to report the sign we find whichever way it goes.
2. **Liu, *Decodable but Not Corrected by Fixed Residual-Stream Linear Steering: Evidence from Medical LLM
   Failure Regimes*, arXiv:2605.05715, 2026-05-07.** https://arxiv.org/abs/2605.05715 . Medical, and the
   closest single paper to Claim C. Concept is Overthinking, a behavioural failure regime in medical QA,
   linearly decodable at 71.6% balanced accuracy, while five families of fixed linear steering across 29
   configurations and n = 1273 all give delta about 0, replicated on Qwen2.5-7B and on MMLU-STEM. Reports
   a per-instance probe-steering correlation of r = -0.002, p = 0.97, which is Claim C's statistic. Runs
   real controls: LEACE erasure damages accuracy by 3.6pp (p = 0.01) while 10 random erasures give
   +0.3pp. Differences from us: a text LLM with no vision encoder and no connector, a behavioural regime
   rather than a clinical image finding, and fixed steering rather than a locus sweep.
3. **Liang, Cheng and Wajid, *Encoded but Not Actionable: Auditing the Decode-Generate-Steer Gap in Frozen
   LLMs for Geometric Constraints*, arXiv:2608.17843, 2026-08-18.** https://arxiv.org/abs/2608.17843 .
   Published nine days before this document. Six frozen decoder-only LLMs, CAD constraints. Audits four
   properties in sequence: linear decodability, forced-choice generation, activation-level influence,
   behavioural steerability, and concludes they "can diverge". Runs shuffled-order controls and a
   random-initialisation baseline, finding sketch-level DOF "already highly decodable from randomly
   initialized representations", which is our random-encoder-floor argument arriving independently — with
   the caveat, recorded in delta-table row 4, that **we have not actually run a random-encoder floor**, so
   this is currently a convergence with our plan rather than with our data. The audit structure is close
   to ours; the domain is not.
4. **Pramono, Cai and Kulkarni, *TRAPSBench: Vision-Language Models Encode but Fail to Express Epistemic
   Restraint*, arXiv:2608.13167, 2026-08-13.** https://arxiv.org/abs/2608.13167 . VLMs, 16 models, five
   families. Linear probes decode answerability up to 0.91 AUROC while spontaneous behaviour is poor
   (best PECS 0.292), and steering a single-layer void direction causally induces or suppresses
   abstention. Their phrase for the gap is "The bottleneck is expression, not perception". This is the
   availability-versus-use claim in VLMs, on a procedurally generated physics benchmark rather than
   clinical images, and with steering at a single layer rather than across loci.
5. **Majumdar, Kogel and Bulling, *CARD: Diagnosing Belief to Action Routing Failures in Vision Language
   Models*, arXiv:2608.20763, 2026-08-21.** https://arxiv.org/abs/2608.20763 . Published six days before
   this document. Introduces Cross-Axis Routing Diagnostic, steering along one axis while measuring a
   different axis's prediction, and finds models "fail to incorporate belief representations into their
   next action prediction, effectively leaving valuable information about their partners unused".
   Represented but unused, in VLMs, by construction. Grid-world benchmark, mental-state concepts.
6. **Wu, Zhao and Chen, *When Is a Steerable Concept Representation Real? Measurement Confounds in a
   Cross-Family Audit of Neuroscience Parallels in LLMs*, arXiv:2608.08159, 2026-08-08.**
   https://arxiv.org/abs/2608.08159 . The closest thing published to our control-battery argument. 17
   models, five families, 0.6B to 72B. Shows an apparent emergent scaling of steerability is an artefact
   of "an uncalibrated pipeline": the trend "depends jointly on raw units, the readout metric, and the
   operating point; correcting any one of these removes it". State the corrected result exactly as they
   do, because an earlier revision of this file overstated it as "the trend disappears": under
   residual-norm-comparable interventions with held-out operating-point selection, "concept steering
   remains significant at every scale, but shows no significant trend across the Qwen3 series, although
   the confidence interval does not rule out a moderate positive slope". Steering does not vanish; the
   *scaling* trend fails to survive calibration, and even that is a null, not a refutation. Their closing
   line is
   effectively our thesis: "the main constraint on AI neuroscience is not a lack of phenomena, but a lack
   of comparable measurements and adequate controls." Cite this as convergent support for the battery,
   and do not claim the battery is unprecedented after citing it.
7. **Fan, Cheng, Li, Feizi and Zhou, *When is Your LLM Steerable?*, arXiv:2606.11599, 2026-06-10.**
   https://arxiv.org/abs/2606.11599 . ASTEER testbed, 1.4M steered generations, 150 concepts, each
   labelled success or failure. Predicts steerability from early hidden states at about 0.7 macro-F1.
   Relevant because it establishes that steering success is systematically variable and predictable, so
   a null steering result at a locus needs a matched control before it means anything.
8. **Torop, Masoomi and Dy, *Inverted Detection and Control in Steering Vectors*, arXiv:2608.02957,
   2026-08-03.**
   https://arxiv.org/abs/2608.02957 . Found by the audit inside this document's own query-2 result set and
   missed by the original pass. Identifies "inverted-steering vectors": directions that are *highly
   discriminative* for a concept yet "consistently promote the opposite behavior" when steered along. This
   is a sharper form of the dissociation than a null result, because it breaks the sign rather than the
   magnitude. Directly relevant to D1's `sign_as_expected` column: a direction that beats the random band
   with the wrong sign is a known phenomenon with a name, not an artefact of our pipeline.
9. **Liu, Wang, Wang, Xiao and Lin, *Broken Symmetry in LLM Refusal: Answer Release Is More Local Than
   Refusal Restoration*, arXiv:2608.15772, 2026-08-16.** https://arxiv.org/abs/2608.15772 . Also missed
   by the original pass.
   Bidirectional activation patching under a matched withhold setting: "even when a model generates a
   clean refusal, the correct answer remains linearly recoverable from its hidden states", and releasing
   it needs a single-position patch while re-suppressing it needs interventions across many positions.
   Decodable-but-not-expressed, plus an asymmetry in how much intervention each direction of the effect
   costs. Text LLMs, refusal rather than a clinical concept.

Adjacent work worth citing once each:

- **Nooralahzadeh et al., *Universal Boosts, Specific Suppressors: Sparse Autoencoder Steering of Medical
  Vision-Language Models*, arXiv:2605.24977, 2026-05-24.** https://arxiv.org/abs/2605.24977 . Medical
  VLMs, chest X-ray report generation, decoding-time residual steering on a per-token SAE basis at late
  layers, causal feature screening, on RadVLM, LLaVA-Rad and CheXOne. Relative clinical-composite gains
  of 5.4%, 7.2% and 17.0% on MIMIC-CXR, transferring zero-shot to IU-Xray at +7.7% GREEN. This is
  residual-stream intervention in medical VLMs done well and for a purpose, and it is the nearest
  neighbour to D1's mechanics. Distinctions: SAE features rather than supervised concept probe normals,
  late layers only rather than a locus sweep, and the goal is report quality rather than measuring
  whether a named concept is used.
- **Pedersen, Sydendal, Cheplygina and Sourget, *Look What the Probes Dragged In! Real-World Chest X-ray
  Shortcuts in MedCLIP*, arXiv:2608.12086, 2026-08-12.** https://arxiv.org/abs/2608.12086 . 17 linear
  probes on intermediate layers of MedCLIP's frozen ResNet-50, on NIH-CXR14 pneumothorax and PadChest
  cardiomegaly and pneumothorax. Finds high final AUROC with poor calibration, localised shortcuts such
  as drains appearing at later layers and diffuse shortcuts such as scanner noise earlier, plus data
  quality problems in both NIH-CXR14 and PadChest. Directly relevant twice over: it is layer-wise probing
  on our dataset, and its data-quality finding is a caveat we should acknowledge, since NIH ChestX-ray14
  is our source (`runs/probe_*`, manifest of 26,229 rows).
- **Rajaram, Schwettmann, Andreas and Conmy, *Line of Sight: On Linear Representations in VLLMs*,
  arXiv:2506.04706, 2025-06-05.** https://arxiv.org/abs/2506.04706 . LLaVA-Next, linearly decodable
  ImageNet-class features in the residual stream, shown causal by targeted edits, plus multimodal SAEs.
  The probe-then-edit template for VLMs, one year earlier and non-medical.
- **Takeda and Sakai, *Does medical specialization of VLMs enhance discriminative power?*,
  arXiv:2601.14774, 2026-01-21.** https://arxiv.org/abs/2601.14774 . Feature-distribution comparison of
  medical against non-medical VLMs across lesion-classification datasets. Concludes non-medical models
  with enriched text encoders can produce more refined representations, and that medical specialisation
  is not the decisive factor. Converges with Zhu et al. from a different direction and supports our
  medical-versus-general framing.

**How Claim C should now be written.** Not "we show decodability does not predict use", which is taken.
Instead: the dissociation is established in text LLMs and in non-medical VLMs, and the open question is
where along the vision-to-language pipeline it opens up. Our contribution is the locus resolution and
the medical concepts, and the falsifiable prediction stays in the plan: the divergence is largest after
the connector. Given 2604.02608 found the decodable-not-steerable cell nearly empty in text, our sign
should be reported as a finding either way, and if we find the opposite sign, that is a contrast with
2604.02608 worth its own paragraph.

---

## 4. Methodological ancestors

Verified against the arXiv API on 2026-08-27, and venue pages fetched where a published version exists.
Every ID below returned HTTP 200 with a matching title.

Re-verified independently the same day: all 23 arXiv IDs appearing anywhere in this document were queried
in one `id_list` call against `export.arxiv.org/api/query`; 23 of 23 entries returned, and every title,
first author and submission date matches what this document states. The four ACL Anthology URLs
(`D19-1275`, `2021.tacl-1.10`, `2020.acl-main.647`, `2024.acl-long.828`) all return HTTP 200. The
LessWrong post returned 429 and the Belinkov DOI 403; both are rate-limiting or bot-blocking, not dead
references. The `journal_ref` field of 2603.06054 reads "Transactions on Machine Learning Research, 2026",
which confirms the TMLR attribution in §1.2.

| Work | Reference | Why we cite it |
|---|---|---|
| Alain and Bengio, linear probes | arXiv:1610.01644 (2016-10-05, v4 2018-11-22). https://arxiv.org/abs/1610.01644 | The origin of reading a representation with a linear classifier at each layer. D2's basic instrument |
| Hewitt and Liang, control tasks | arXiv:1909.03368 (2019-09-08). EMNLP-IJCNLP 2019, https://aclanthology.org/D19-1275/ | The selectivity definition we use. Real-label accuracy minus random-label accuracy at matched capacity. Our 39-type construction is the medical-image adaptation. Cite for the definition, and note that language has recurring types and images need a substitute |
| Belinkov, probing survey | arXiv:2102.12452 (2021-02-24, v4 2021-09-22). *Computational Linguistics* 48(1), 2022, DOI 10.1162/coli_a_00422 | The canonical statement of what probing accuracy does and does not license. Cite when framing decodability as availability rather than use |
| Elazar et al., amnesic probing | arXiv:2006.00995 (2020-06-01, v3 2021-02-19). TACL 2021, https://aclanthology.org/2021.tacl-1.10/ | The availability-versus-use distinction, and the removal-based way of testing it. The direct ancestor of Claim B |
| Ravfogel et al., INLP | arXiv:2004.07667 (2020-04-16). ACL 2020, https://aclanthology.org/2020.acl-main.647/ | Iterative nullspace projection, the erasure method amnesic probing runs on |
| Belrose et al., LEACE | arXiv:2306.03819 (2023-06-06, v4 2025-04-03) | Closed-form perfect linear concept erasure. The modern replacement for INLP, and the tool 2605.05715 uses to argue entanglement. Cite in the discussion of what a negative steering result can mean |
| Turner et al., activation addition | arXiv:2308.10248 (2023-08-20, v5 2024-10-10) | ActAdd, the `h + alpha * v` edit D1 performs. Cited by 2603.06054 for the same purpose |
| Panickssery et al., contrastive activation addition | arXiv:2312.06681 (2023-12-09, v4 2024-07-05). ACL 2024, https://aclanthology.org/2024.acl-long.828/ | CAA, difference-of-means directions. `docs/00_PAPER_PLAN.md` §D1 does specify "Fallback: difference of class means", so this is the right citation for that step. **Citation trap, with its provenance corrected: an earlier revision of this file said "the plan calls this Rimsky". It does not — `grep -ri rimsky docs/` returns nothing, and the plan never names CAA or any author for the fallback. The trap is real but it lives in the wider literature, not in our plan: this work was widely cited as "Rimsky et al." and arXiv now lists the first author as Nina Panickssery (confirmed against the arXiv API on 2026-08-27). Cite as Panickssery et al. and do not let a bibliography tool resurrect the old name** |
| Zou et al., representation engineering | arXiv:2310.01405 (2023-10-02, v4 2025-03-03) | RepE, the top-down reading-and-controlling framing that names the probe-then-steer pattern as a method |
| nostalgebraist, logit lens | LessWrong, 2020-08. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens | No arXiv ID and no DOI. Cite the post directly. The origin of reading intermediate layers through the unembedding, and the decoder whose limits 2604.02608 is about |
| Belrose et al., tuned lens | arXiv:2303.08112 (2023-03-14, v6 2025-11-11) | The calibrated successor to the logit lens. Relevant because 2604.02608 shows a diagonal tuned lens closes only 1 of 14 steerable-not-decodable cases, which bounds how much a better decoder can rescue a decodability claim |

Two notes for the citation audit. Zou et al. and the tuned lens both have v4 and v6 revisions dated
2025, so any "2023" year in the bibliography should be the original submission year and stated
consistently. Hewitt and Liang, Ravfogel et al., Elazar et al. and Panickssery et al. all have published
venue versions, and the venue version should be the primary citation with the arXiv ID as a secondary.

---

## 5. Search log

Terms used, so the section can state its recall honestly and a reviewer can repeat it.

**Full-text retrieval.** `arxiv.org/html/2604.08333v1` and `arxiv.org/html/2603.06054v2` downloaded with
curl, stripped to plain text, and searched locally with grep for: `steer`, `random`, `control`,
`baseline`, `sham`, `shuffl`, `permut`, `alpha`, `pool`, `probing head`, `MLP`, `last token`,
`post-layernorm`, `split`, `patient`, `leakage`, `stratif`, `Hewitt`, `selectivity`. Local grep was used
in preference to summarisation because two load-bearing claims in the delta table are claims of
**absence**, and an absence cannot be established from a summary.

**Web search.** `concept decodability probing across layers medical vision-language model 2026`;
`decodability versus steerability probing steering same layer dissociation LLM`; `medical VLM linear
probe steering vector chest X-ray concept intervention residual stream 2026`; `"control task"
selectivity probing medical imaging model interpretability validity`; `nostalgebraist logit lens
interpreting GPT 2020 LessWrong original post`.

**arXiv API, date-restricted to the requested window.** Three structured queries against
`export.arxiv.org/api/query` with `submittedDate:[202608010000 TO 202609010000]`, sorted by submission
date descending. Result counts were re-run by an independent auditor on 2026-08-27; where the re-run
disagrees, both numbers are given, because the field prefix (`abs:` against `all:`) changes the count and
the original queries were recorded in prose rather than as literal query strings. **Record the literal
query string next time.** Use `curl --globoff`, or the square brackets in the date range break the URL.

1. `(abs:probing OR abs:"linear probe" OR abs:probes) AND (abs:medical OR abs:clinical OR abs:radiology)
   AND (abs:"vision-language" OR abs:multimodal OR abs:VLM)`. **9 results** on re-run with `abs:`, 10 with
   `all:`; the original count of 9 reproduces. Only 2608.12086 is relevant, and the auditor confirmed this
   by reading all ten titles: the rest are ultrasound vision-language-action models, an OCT nerve probe,
   CT phenotyping, foundation-model selection, and a modality-failure framework. None probes concepts
   across VLM depth.
2. `(steering OR "activation addition" OR steerability) AND (probe OR probing OR decodability)`.
   **The original count of 28 does not reproduce: the re-run returns 47 under both `abs:` and `all:`.**
   All seven IDs the original named — 2608.17843, 2608.08159, 2608.20763, 2608.13167, 2608.21766,
   2608.11475, 2608.06638 — are present in the 47, so nothing named here is spurious. But the original
   query as described was narrower than the one written down, and 19 results were never triaged. The
   auditor read all 47 titles; two are relevant and were missed. Both are now in §3 as items 8 and 9.
3. `(connector OR projector OR "vision encoder") AND ("layer-wise" OR "across layers" OR "information
   flow" OR depth) AND (probe OR probing)`, widened to 202607010000 onward. 2 results, both already seen.
   Not re-run.

**Recall statement for the section.** Within the requested window of August 2026, no paper was found
that measures clinical-concept decodability across the full depth of a medical VLM. This survived the
audit: reading all 47 hits of the widened query 2 and all 10 of query 1 turned up nothing that does. The nearest is
2608.12086, which probes 17 layers of MedCLIP's ResNet-50 vision encoder on NIH-CXR14 and PadChest, and
stops at the encoder: no connector, no LLM. The decodability-versus-steerability comparison, by
contrast, is actively contested ground: three of the seven papers in §3 appeared in August 2026 alone,
two of them within nine days of this document. The section should be re-run against the same queries
immediately before submission, because the arrival rate in this specific area is currently about one
relevant paper per week.

---

## 6. Consequences for the paper

1. **Do not claim the dissociation as a discovery.** It is published. Claim the locus resolution, the
   medical concepts, the matched controls, and the pipeline axis. Cite 2604.02608 and 2605.05715 in the
   same paragraph that states Claim C.
2. **Do not claim the control battery is unprecedented.** Claim it is unprecedented in medical VLM
   probing, which the survey licenses at 0 of 135, and cite 2608.08159 as convergent evidence that the
   field is arriving at the same conclusion from another direction. This is a stronger position than
   priority, because it makes the paper part of a visible correction rather than an outlier.
3. **Correct the plan's claim about the matched twin.** Zhu et al. already compare Lingshu-7B against
   Qwen2.5-VL-7B. Our version is the controlled probe-curve comparison on base models, not the pairing.
4. **Credit Theodoridis with the observation.** They saw decodable-but-not-steerable in DriveFusionQA
   Count-1 and declined to claim it. Saying so costs nothing and makes the rest of the positioning
   credible.
5. **Rows 7 and 11 are where the paper is most original.** Row 7, that Nodule and Mass do not survive
   their own control, is fully supported by data on disk today. Row 11, D3c, is not implemented. If time
   is short, D3c is worth more than a fifth model.
6. **The section must be re-run before submission.** See §5, and record the literal arXiv API query
   strings this time, because the counts in the original §5 could not be reproduced from the prose
   description.
7. **The headline probe number is settled, so stop calling it a discrepancy.** §0 now shows the brief's
   0.75 / 0.69 / +0.084 is the peak-per-(model, concept) statistic and reproduces. Quote the
   visual-position track, median peak AUROC 0.751 for selectivity +0.076, and name the locus set in the
   same sentence. `docs/10_RESULTS.md` §2.2 reaches the same place independently. Agree with that file on
   one definition of the peak locus set before either number is published, because the two currently
   differ by 0.002 on the control median for that reason alone.

---

## 7. Audit trail

This document was adversarially re-checked on 2026-08-27 by an auditor who re-ran the computations from
the raw CSVs, re-downloaded both focal papers, and re-queried every arXiv ID and both reproducible search
queries. What follows is what changed, so a reader can tell verified content from repaired content.

**Survived unchanged, verified to the digit.** The five probe row counts and their sum of 3060. The pooled
medians 0.7168 / 0.6707 / +0.0623. All nine per-concept selectivity medians. The `nuis_view_AP` range, the
`n_types` value of 39, and the count of 74 loci. The LLM-loci-only and mean-then-median aggregations. Every
quotation from and protocol number in 2604.08333 and 2603.06054, including both load-bearing claims of
absence, which were re-established by grep against a fresh download rather than carried over. All 23 arXiv
IDs, with titles, first authors and dates. Every statistic attributed to the seven §3 papers. The two
corrections to `docs/00_PAPER_PLAN.md` §0: Zhu et al. Figure 4 does read "HuatuoGPT-Vision-7B/Lingshu-7B/
ShizhenGPT-7B-VL vs. Qwen2.5-VL-7B", so the plan's matched-twin claim is indeed wrong.

**Newly found, not present in the earlier revision.** Delta-table row 4 claimed a random-encoder floor
"at every locus". No such control is implemented anywhere in `src/`, and no `probe_results.csv` carries a
column for it. Row 4 is now split: the permutation null is real, the floor is not built. This was the same
class of defect as rows 9 to 11 but was not flagged as one. Two relevant August 2026 papers, 2608.02957 and
2608.15772, sat untriaged in this document's own search output and are now §3 items 8 and 9.

**Corrected.** The §0 headline "discrepancy", which was not a discrepancy and whose peak figure of
0.726 / 0.681 / 0.069 is unreproducible and now marked UNKNOWN. The §0 D1 snapshot, which was stale and
whose row counts included the CSV header and which has now been replaced by a regeneration command rather
than a second table that would decay the same way. The superseded-run selectivity, which is -0.3029 for
lingshu and -0.3062 for qwen, not -0.303 for both. The claim that `docs/00_PAPER_PLAN.md` says "Rimsky",
which it does not. The reading of 2608.08159, which was overstated. The Theodoridis dataset row, which
omitted the two town exceptions. One truncated quotation. The §5 query-2 result count.

**Still open, and not this document's to close.** D1 has no finished job, so rows 9 and 10 of the delta
table remain unwritable. Row 11 is unimplemented, and so is the random-encoder half of row 4. Whether the
results section quotes the peak statistic over `*.vis` loci or over all loci is a decision for whoever
writes it, and this file records both rather than choosing.
