# Research findings

Run: the six-gate result-to-claim pass compared the accepted final-block decoding, intervention-specificity, causal-ownership, and paired input-closure evidence against the registered claim.

Observation: the fixed linear readouts recover clinical labels, but each causal interpretation stops at a registered control boundary. LLaVA stops at random/sham; Qwen Effusion passes generic controls but not clinical-direction specificity; the Qwen matrix identifies neither an owned concept nor a shared alias beyond random directions; and the paired Consolidation gate does not separate closure from random or clinical controls. The resulting evidence establishes an availability-versus-concept-specific-use dissociation for the registered NIH cells and protocols.

Gate decision: `PASS`; the pinned cross-family reviewer judged the scoped claim supported with high confidence and integrity `pass`, and routed the project to `confirm`. No further experiment is authorized by this result.

Next step: can the six-gate evidence be integrated into the evidence-locked manuscript while preserving exact scope, provenance, and the nine-page main-body budget? Update, cross-review, and compile the manuscript.

Evidence: [CLAIMS_FROM_RESULTS.md](CLAIMS_FROM_RESULTS.md); `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T164410087766Z.json`; `.aris/traces/result-to-claim/2026-09-04_run01/`.

## LLaVA validation opportunity

Run: the registered validation-only pilot evaluated six frozen clinical readers and the corresponding clean singleton yes/no margins before admitting any question to a protected write cohort.

Observation: Effusion and Cardiomegaly retain controlled reader selectivity, but no question has a clean-margin AUROC fifth-percentile bound above 0.5. This supports a measurement-stage direction/readout separation. Effusion is the informative follow-up anchor; Cardiomegaly's reader lower bound is only `0.000164`.

Gate decision: `PASS`; `claim_supported=yes`, confidence high, integrity pass and routing action `confirm`. The write screen remains unexecuted by the registered `K=0` rule, and pinned synthesis review accepts the scoped interpretation and next gap. Evidence is [docs/LLAVA_VALIDATION_OPPORTUNITY_RESULTS.md](docs/LLAVA_VALIDATION_OPPORTUNITY_RESULTS.md), `.aris/evidence_precheck.json`, `.aris/traces/result-to-claim/2026-09-05_run01/` and `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T094932091464Z.json`.

Next step: can a prospectively registered Effusion diagnostic distinguish prompt/language-prior sensitivity from answer-token verbalizer routing while independently replicating frozen-direction availability?
