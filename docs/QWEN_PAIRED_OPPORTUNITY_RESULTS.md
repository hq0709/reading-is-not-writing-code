# Qwen paired Consolidation response

Run: `20260905T053520Z-f0317eaf3f4e-paired-opportunity`, source `f0317eaf3f4e3d97a18c266eb3d84b0f1c09baec`, evaluated the fixed Consolidation anchor on 100 fresh test-patient pairs. The 200 image outcomes completed on one A100 in 31 seconds, ending at 2026-09-05 05:36:11 UTC. Command, dispatcher and cleanup statuses were zero.

## Observation

The patient-weighted finding-present logit difference between positive and negative images is -0.1175, with one-sided 95% lower bound -0.3987 and descriptive two-sided 95% interval [-0.4500, 0.2088]. The registered opportunity criterion, a strictly positive one-sided lower bound, is not met.

Within-patient concordance is 0.545, counting tied margins as one half. The mean paired sigmoid-score difference is -0.01743. Among 52 pairs whose negative image is labelled `No Finding`, the mean logit gap is -0.2115; among the other 48 pairs it is -0.01563. Both strata are descriptive and retain every selected patient.

The cohort matches positive and negative studies within patient on view, sex, integer age and all 13 other disease labels. Patients are absent from the entire original probe manifest and previous intervention cohorts. The fixed image and patient order and full source metadata are retained. The clean-response measurement uses `Is there consolidation in this chest radiograph? Answer yes or no.` and the accepted yes-minus-no maximum-token logit margin.

Generic semantic mapping, exact clean validation repeat and the 128-equivalent throughput pilot passed. The predicted total was 89.9793 seconds under the 600-second allowance. The source-bound validation protocol is documented in `docs/reviews/qwen7b-paired-opportunity-validation.md`.

## Gate decision

The terminal summary gives `opportunity_available=false`. The scientific result is `OBSERVED`; internal terminal verification and pinned result review passed. The opportunity remains unresolved for this model, prompt and metadata-defined paired population. The registered route advances to architecture-transfer registration.

## Next step

Does the corresponding paired behavior appear under another accepted architecture? Register a matched-cohort architecture-transfer measurement before further component geometry.

## Limitations

This is a conditional comparison using source disease labels and acquisition metadata. Within-patient matching does not isolate a pathology intervention or remove unrecorded longitudinal differences. The interval includes both positive and negative mean responses. Concordance and subgroup means are descriptive; the registered mean-gap criterion governs the decision.

## Evidence

The immutable run retains `artifacts/per-image.csv`, `registered-pairs.json`, `meta.json`, `preflight.json`, `paired-opportunity-summary.json` and `paired-opportunity-summary.npz`. The bootstrap uses 5,000 patient resamples with seed 20260912. The cohort reference is `20260905T052025Z-9bc918b414ff-paired-opportunity/artifacts/registered-pairs.json`.

Internal receipt: `/home/qingchan/data/concept-flow/state/paired-opportunity-internal-20260905T054119Z/smallreceipt.json`. Pinned acceptance: `docs/reviews/qwen7b-paired-opportunity-results.md`, receipt `review-20260905T054422273733Z.json`.
