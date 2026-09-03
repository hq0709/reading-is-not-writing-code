# Paper Plan

**Working title:** Decodable Is Not Direction-Specific: Causal Tests at the Final Visual Block of Medical Vision-Language Models
**Venue:** ICLR
**Type:** Empirical/diagnostic
**Date:** 2026-09-03
**Page budget:** 9 pages through Conclusion, including title/abstract and excluding references and appendix
**Section count:** 6

Run: organize the accepted LLaVA Effusion, LLaVA Edema, Qwen Effusion, and independent-row Qwen direction-specificity gates into one paper whose claims remain bound to the registered protocol.

Observation: every final-block cell has positive controlled linear decodability, while none reaches the direction-specific evidentiary endpoint. LLaVA Effusion and Edema stop at the random/sham rung; Qwen Effusion clears that rung on 400 independent patients but stops at the fixed clinical-direction rung because Nodule remains reliably larger.

Gate decision: `PASS`; all 18 adjudicated values map to primary artifacts, the section and page accounting is internally consistent, and the configured pinned read-only Claude review returned `PASS`, disposition `READY`, with no required actions.

Next step: can the accepted evidence-locked outline be drafted into a manuscript whose prose, figures, and tables remain traceable to the four accepted gates? Draft section by section from this plan.

## One-sentence contribution

Under capacity-controlled, patient-split NIH ChestX-ray14 evaluation, robust linear decodability at the consumed final visual block did not establish direction-specific causal influence under registered probe-normal interventions in three model–concept cells.

## Claims–evidence matrix

| Claim | Direct evidence | Status | Planned section |
|---|---|---|---|
| A controlled protocol can distinguish linear availability from direction-specific behavioral influence at the exact visual representation consumed downstream. | Forward-path hook receipts; fixed 512-dimensional logistic readout; patient split; 20 type-to-label controls; patient bootstrap; relative-token dose grid; random, sham, and fixed clinical directions. | Supported protocol contribution | §3 |
| Robust controlled decodability coexists with failure at the registered causal-control rung in all three final-block cells. | AUROC range `0.7742–0.8009`; selectivity CIs exclude zero. LLaVA Effusion and Edema fail the original random/sham conjunction; Qwen Effusion clears random/sham but fails the supplement's clinical-direction criterion. | Supported empirical claim | §4.1–§4.3 |
| Random/sham selectivity is insufficient for clinical direction specificity in the Qwen Effusion cell. | On 400 independent patients, Effusion changes mean `P(yes)` by `0.1945`, above random p95 `0.0878` and absolute sham `0.0157`, while Nodule changes it by `0.2519`; familywise margin `-0.0573`, 95% CI `[-0.0694, -0.0450]`. | Supported empirical claim | §4.3 |
| The paired controlled-decoding contrasts do not resolve an architecture or concept ordering. | Qwen-minus-LLaVA Effusion selectivity `0.0026`, paired 95% CI `[-0.0155, 0.0215]`; Edema-minus-Effusion in LLaVA `0.0221`, paired 95% CI `[-0.0175, 0.0590]`. | Supported boundary | §4.2, §5 |

## Story and paper structure

### §0 Abstract (180–220 words)

- Problem: linear probes show what a representation makes decodable, but a clinical interpretation also needs evidence that the decoded direction selectively affects behavior.
- Approach: compare a fixed-capacity patient-split readout with forward-path, equal-norm probe-normal interventions at the exact final visual block consumed by LLaVA-1.5-7B and Qwen2.5-VL-7B.
- Evidence chain: three registered cells, 20 type-to-label controls, 2,000 patient bootstraps, random/sham/clinical intervention controls, then a prospectively registered 400-patient Qwen replication with a 5,000-draw familywise bootstrap.
- Key result: all cells decode at AUROC `0.77–0.80`; neither LLaVA cell passes random/sham selectivity, and Qwen Effusion replicates above random/sham but has a familywise Effusion-minus-maximum-unrelated margin of `-0.0573` (95% CI `[-0.0694, -0.0450]`), with Nodule realizing the point-estimate maximum.
- Implication: availability, perturbation sensitivity, and clinical direction specificity are distinct evidence levels for medical VLM interpretation.

### §1 Introduction (1.25 pages)

- Opening: a probe can name a clinical signal in a representation; the consequential question is whether changing that named direction selectively changes the model's answer.
- Gap: medical VLM probing and VLM steering are usually reported separately or without capacity, patient, and matched-direction controls at an exact consumed locus.
- Question: when a clinical concept is robustly linearly decodable at the final visual block, does its probe normal have direction-specific causal influence on the answer?
- Approach preview: lock the readout, locus, intervention scale, control families, and uncertainty before using final-test evidence.
- Contributions:
  1. A forward-path, capacity-controlled protocol that separates controlled decodability, random/sham selectivity, and clinical direction specificity.
  2. Three final-block observations in which controlled decodability is positive but the registered direction-specific evidentiary endpoint is not reached.
  3. An independent-patient Qwen replication showing a behavioral response above random/sham yet reliably below a fixed unrelated clinical direction.
- Results preview: emphasize `0.77–0.80` AUROC across three cells and the Qwen familywise Effusion-minus-maximum-unrelated margin `-0.0573` (95% CI `[-0.0694, -0.0450]`), with Nodule realizing the point-estimate maximum.
- Hero figure: three panels. Panel A shows the final visual block, the fixed linear probe, and the downstream intervention path. Panel B aligns the three cells in rows, placing controlled selectivity with bootstrap intervals beside the registered intervention control comparison without sharing an axis. Panel C plots Qwen's independent-row direction effects for Effusion, the random distribution, sham, and five fixed clinical controls, with Nodule highlighted and the familywise margin interval shown.
- Key citations: linear probing (`alain2017understanding`, `hewitt2019designing`, `ravichander2021probing`); medical VLM depth probing (Zhu et al., arXiv:2604.08333, verified record in `docs/11_RELATED_WORK.md`, BibTeX to verify/add); VLM probe-vector steering (Theodoridis et al., TMLR 2026, verified record in `docs/11_RELATED_WORK.md`, BibTeX to verify/add).

### §2 Related Work (0.75 pages)

- Controlled probing: probe capacity, control tasks, and the distinction between decodability and task relevance. Position the present work as a patient-split multimodal application with repeated type-to-label controls.
- Causal interventions on representations: activation steering and probe-vector interventions. Position the present work around equal-norm random directions, coordinate-permutation sham, fixed clinical directions, and a registered direction-specific endpoint.
- Medical VLM interpretability: depth-wise probing and medical-image shortcut analysis. Position the paper at one verified consumed visual locus with causal tests, not as a depth survey.
- Synthesis target: related work should explain why linear availability, generic perturbation sensitivity, and clinical direction specificity answer different questions.
- Citation scaffold: `hewitt2019designing`, `ravichander2021probing`, `turner2024steering`, `pedersen2026medclip`; Zhu et al. and Theodoridis et al. from `docs/11_RELATED_WORK.md`. All entries receive bibliographic verification before prose drafting.

### §3 Registered Measurement Protocol (1.75 pages)

- Data and split: NIH ChestX-ray14; deterministic patient-disjoint train/test split; weak report-mined Effusion and Edema labels; exact test cohorts.
- Models and loci: LLaVA-1.5-7B `encoder.layers.22` and Qwen2.5-VL-7B `model.visual.blocks.31`, each proven to be on the downstream path by hook tests.
- Controlled decodability: mean-pooled features, fixed 512-dimensional Gaussian projection, logistic regression with `C=1`, seed 0, 20 recurring-type random-label maps, AUROC and selectivity, 2,000 patient-cluster bootstraps.
- Probe-normal intervention: per-token relative-norm scaling; registered bidirectional alpha grid; mean `P(yes)` endpoint; 20 isotropic random directions, coordinate-permutation sham, and fixed unrelated clinical normals.
- Original-cell decision rule: maximum concept-consistent nonzero-alpha effect must leave the same-alpha random interval in the registered direction and exceed the maximum absolute sham effect; monotonicity is descriptive.
- Direction-specificity supplement: independent 400-patient cohort; alpha `+0.25`; Effusion, 20 random directions, sham, and five clinical controls; 5,000 patient bootstraps; recompute the maximum unrelated effect inside each replicate.
- Primary familywise statistic: `M = Delta_Effusion - max_d Delta_d`; direction specificity requires a positive Effusion effect above random and sham and a one-sided 95% lower bound for `M` above zero.
- Provenance: immutable commit snapshots, terminal receipts, deterministic summary replay, and pinned read-only cross-family review.
- Key citations: `wang2017chestxray`, `liu2024improved`, `bai2025qwen25vl`, `hewitt2019designing`.

### §4 Results: Availability, Selectivity, Specificity (3.0 pages)

#### §4.1 Controlled decodability is robust in every cell

- Table 1 reports model, concept, locus, AUROC with 95% CI, control mean/spread, and selectivity with 95% CI.
- LLaVA Effusion: AUROC `0.7788`, selectivity `0.1139` (`0.0875–0.1404`).
- LLaVA Edema: AUROC `0.8009`, selectivity `0.1360` (`0.0994–0.1693`).
- Qwen Effusion: AUROC `0.7742`, selectivity `0.1166` (`0.0897–0.1428`).

#### §4.2 Similar availability produces different random/sham intervention outcomes

- Figure 2 shows full registered dose responses with same-alpha random bands and sham effects.
- LLaVA Effusion reaches `0.1470` at alpha `-1`, above random p95 `0.1202` but below absolute sham `0.1546`.
- LLaVA Edema reaches `0.0659` at alpha `-1`, inside the random interval and below absolute sham `0.0913`.
- Qwen Effusion reaches `0.2343` at alpha `+0.25`, above random p95 `0.0963` and absolute sham `0.0852` under the original registered rule.
- Paired selectivity intervals support treating the cells as comparable in availability while leaving architecture and concept ordering unresolved.

#### §4.3 Independent Qwen rows separate generic selectivity from clinical specificity

- Figure 3 shows 27 intervention-direction effects plus the common unsteered baseline, for all 28 registered configurations on the 400-patient cohort.
- Effusion replicates above random and sham: `0.1945` versus random p95 `0.0878` and absolute sham `0.0157`.
- Nodule is larger at `0.2519`; the familywise Effusion-minus-maximum-unrelated margin is `-0.0573`, 95% CI `[-0.0694, -0.0450]`, with one-sided lower bound `-0.0678`.
- The result establishes a reliable, non-direction-specific response under the registered clinical-direction criterion.

#### §4.4 Finding chain

- Finding 1: all three final-block representations make the target label robustly available to the fixed linear readout.
- Finding 2: LLaVA Effusion and Edema stop at the original random/sham rung, so availability does not determine whether the probe normal clears generic intervention controls.
- Finding 3: Qwen Effusion clears the original random/sham rung but stops at the supplement's clinical-direction rung, so generic selectivity does not determine clinical specificity.

### §5 Discussion and Limitations (1.25 pages)

- Interpretation: treat availability, generic intervention sensitivity, and clinical direction specificity as a hierarchy of evidence rather than interchangeable indicators.
- Practical rule: a medical VLM direction should be interpreted clinically only after matched clinical-direction controls, not only random vectors and sham.
- Scope: NIH ChestX-ray14, two 7B architectures, the consumed final visual block, Effusion in both models, Edema in LLaVA, fixed prompts, a linear readout, and relative-token probe-normal interventions.
- Statistical boundary: the paired decoding intervals do not resolve architecture or concept ordering; the design supports a three-cell dissociation, not a population architecture effect.
- Data boundary: NIH labels are report-mined and the evaluation does not localize findings spatially.
- Intervention boundary: the Qwen result is a reliable behavioral shift whose clinical direction is non-specific under the fixed control family; intermediate loci and nonlinear directions remain future tests.
- Specificity boundary: the independent supplement tests the discovery-locked alpha `+0.25` against a fixed family of five unrelated clinical directions.
- Reproducibility: report all row-selection rules, seeds, bootstrap units, direction identities, immutable run IDs, GPU time, and exact model revisions.
- Ethics: frame the work as an interpretability validation study, not a diagnostic-performance claim; discuss the risk of treating decodable features as clinically operative.

### §6 Conclusion (0.5 pages)

- Restate the empirical chain: controlled decodability in every cell, two LLaVA failures against random/sham, and a Qwen effect that replicates above generic controls but remains below a fixed unrelated clinical direction.
- Close with the transferable principle: causal interpretation of decoded clinical concepts requires controls that test direction specificity at the same locus and dose.

## Figure and table plan

| ID | Type | Description | Data source | Priority |
|---|---|---|---|---|
| Figure 1 | Hero/evidence ladder | Panel A: exact consumed-locus probe and intervention. Panel B: three-row decodability/control outcome matrix. Panel C: independent Qwen direction effects and familywise margin. Use colorblind-safe blue/orange plus shape and hatching; remain legible in grayscale. | Four accepted run summaries and hook receipts | High |
| Figure 2 | Dose-response small multiples | One panel per original cell; concept effect across alphas, same-alpha random 5th–95th band, sham markers, zero baseline. Do not normalize decodability and behavioral effect onto a shared scale. | Original intervention summaries | High |
| Figure 3 | Direction comparison | Qwen 400-patient effects for Effusion, 20 random directions, sham, and five fixed clinical normals; annotate Nodule and the bootstrap margin interval. | `direction-specificity-summary.json` and bootstrap artifact | High |
| Table 1 | Controlled decoding | AUROC, bootstrap CI, control mean/spread, selectivity, selectivity CI, model, concept, locus. | Three probe JSON artifacts | High |
| Table 2 | Gate outcomes | One row per cohort with model, concept, cohort identity, alpha rule/value, N, concept effect, random interval/p95, absolute sham, clinical-direction statistic, failed criterion rung, and disposition. Original-cell rows mark the prospectively familywise clinical-direction endpoint `N/A`; the supplement row reports it. | Four accepted summaries | High |
| Appendix A | Protocol detail | Prompts, row selection, alpha grids, random seeds, direction construction, hook tests, and exact model revisions. | `docs/RESEARCH_PLAN.md` and immutable receipts | High |
| Appendix B | Complete controls | Full unrelated-direction and random-direction values, full dose curves, and paired-bootstrap details. | Per-image outputs and bootstrap artifacts | Medium |

**Figure 1 caption draft:** **Decodability, generic perturbation selectivity, and clinical direction specificity are distinct tests.** (A) A fixed-capacity probe reads a clinical label from the exact final visual block consumed downstream; equal-norm interventions then edit that same representation. (B) All three registered cells have positive controlled decoding intervals; LLaVA Effusion and Edema stop at the original random/sham rung, while Qwen Effusion clears it. (C) On 400 independent NIH patients evaluated with Qwen, Qwen Effusion stops at the clinical-direction rung: Effusion steering exceeds random and sham, but Nodule realizes the maximum fixed unrelated effect and the familywise Effusion-minus-maximum-unrelated interval lies below zero. The evidence therefore supports availability in every cell and identifies the exact control rung that bounds probe-normal causal interpretation.

## Page allocation

| Component | Pages |
|---|---:|
| Title and abstract | 0.50 |
| §1 Introduction | 1.25 |
| §2 Related Work | 0.75 |
| §3 Registered Measurement Protocol | 1.75 |
| §4 Results | 3.00 |
| §5 Discussion and Limitations | 1.25 |
| §6 Conclusion | 0.50 |
| **Total** | **9.00** |

## Citation plan

- §1 Introduction: `alain2017understanding`, `ravichander2021probing`, `hewitt2019designing`; Zhu et al. (arXiv:2604.08333) and Theodoridis et al. (TMLR 2026), using the verified records in `docs/11_RELATED_WORK.md` and adding publisher/arXiv-exported BibTeX before drafting.
- §2 Related Work: `hewitt2019designing`, `ravichander2021probing`, `turner2024steering`, `pedersen2026medclip`; organize by controlled probing, representation intervention, and medical VLM interpretation.
- §3 Protocol: `wang2017chestxray`, `liu2024improved`, `bai2025qwen25vl`, `hewitt2019designing`.
- §5 Discussion: cite the closest decodability/task-relevance and medical/VLM steering studies already documented in `docs/11_RELATED_WORK.md`; run citation audit before any submission-ready draft.
- Bibliography policy: reuse existing verified entries where available; retrieve missing BibTeX from primary venue or arXiv records; mark no citation complete from memory.

## Reviewer feedback

The configured `claude-review-concept-flow` transport, pinned to `claude-fable-5-1` at medium effort with read-only evidence, returned `REVIEW_VERDICT: PASS`, `GATE_DISPOSITION: READY`, and `REQUIRED_ACTIONS: NONE`. It scored logical flow, claim–evidence alignment, and the figure/table plan at 9/10; page feasibility and citation positioning at 8/10. The accepted receipt is `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260903T112856162325Z.json`.

## Next actions after this gate

- Draft the manuscript section by section from this evidence-locked plan.
- Generate Figures 1–3 and Tables 1–2 directly from accepted artifacts.
- Compile, then run paper-claim and citation audits before submission status is considered.
