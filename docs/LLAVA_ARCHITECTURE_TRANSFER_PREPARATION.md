# LLaVA architecture-transfer preparation

Status: `PLANNED`. This note identifies the next registration question and the reusable evidence.

## Run

The accepted paired-opportunity run `20260905T064036Z-1af90218a1d8-llava-opportunity` completed the two-configuration comparison on the fixed 100 Consolidation patient pairs. Its result and direct cross-model contrast are recorded in `docs/PAIRED_OPPORTUNITY_ARCHITECTURE_RESULTS.md`.

The accepted LLaVA Effusion run `20260902T191411Z-ffd523c464c8-f99e2f39` supplies consumed-locus activations and six fixed clinical probe normals in `artifacts/probe/directions.npz`: Effusion, Atelectasis, Pneumothorax, Cardiomegaly, Mass and Nodule. The consumed module is `model.vision_tower.encoder.layers.22`. Edema run `20260902T221534Z-110b84618d1b-edema` reuses those activations and adds an Edema normal. The accepted Qwen ownership run `20260904T125710Z-a3bd883540eb-causal-ownership` supplies an ordered, 400-patient cohort receipt for a potential shared-cohort comparison.

## Observation

The paired Consolidation gate leaves positive natural-input behavioral opportunity unresolved in both configurations. Natural-input discrimination and response to an artificial direction write answer different questions.

The first LLaVA probe report certifies Effusion controlled selectivity of 0.11393 with interval [0.08754, 0.14040]. The Edema report certifies 0.13602 with interval [0.09936, 0.16931]. Those reports contain additional clinical normals as controls; eligibility of the other five potential diagonal readers remains to be established. The accepted Effusion and Edema intervention gates both have `selective_cell=false` under their registered random/sham criteria.

The paper task identifies the decisive architecture-transfer question as whether a matching clinical direction outcompetes mismatched directions where LLaVA has both calibrated answer ability and a measurable intervention response beyond generic controls. A full matrix without those prerequisites chiefly extends the boundary of nonconfirmation.

## Gate decision

Prepare a validation-only capability and intervention-opportunity pilot before committing to the full matrix. This is a new prospective branch within the authorized architecture-transfer work. The accepted two-cell results and paired-opportunity adjudications remain the evidence for their original protocols.

Reuse accepted model, processor, consumed-hook, activation and direction receipts. Separate three qualifications in the pilot design: capacity-matched reader eligibility, unsteered answer calibration, and a signed write response relative to the same-dose random and sham controls. A downstream logit change from a hook test establishes implementation sensitivity; the controlled pilot establishes scientific opportunity.

## Next step

Lock the target set, question/encoding set, dose set, patient-disjoint validation/calibration allocation, selection criterion, multiplicity treatment, complete reporting rule, timing pilot and compute cap before collecting new outcomes. Fix a single routing rule from the pilot to a patient-disjoint confirmatory matched-versus-mismatched comparison. Any settings selected by the pilot enter that comparison as validation-selected settings. Reuse the Qwen cohort receipt only if its role and separation from selection/calibration are established by the new registration.

Estimate the remaining reader-qualification work from the accepted activation archive and controlled-probe implementation. Complete independent internal and pinned registration review before dispatch. The pilot's numerical thresholds, sample sizes and full-matrix go/no-go rule are the next concrete design deliverable.
