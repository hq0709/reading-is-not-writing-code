# Paired behavioral opportunity registration

Run: protocol and implementation review of the fresh-patient Qwen clean-response gate, after accepted Mass confirmation routes to Consolidation.

Observation: the implementation selects 100 patients from the accepted metadata pool before model loading, retains both source records and all paired responses, and applies the fixed prompt, token scoring, one-A100 preflight, 128-equivalent pilot, 10-minute predicted total and 15-minute cap. It produces the complete 200-outcome Consolidation grid, patient-weighted margin gap, 5,000-draw joint bootstrap, fifth-percentile lower bound, descriptive interval, probability gap, within-patient concordance and `No Finding` strata. The clinical opportunity criterion uses the one-sided bound.

Independent internal review accepted the selection, statistical design, route binding, GPU/immutable-source integration and CPU replay. Directly scoped tests and synthetic NPZ replay passed for both supported concepts. The full local suite completed 219 tests in 84.587 seconds, with no failures and six environment-specific skips. Python parsing and Bash syntax checks passed.

Gate decision: `PASS`, ready for Consolidation dispatch. Pinned review returned `PAIRED_OPPORTUNITY_REGISTRATION PASS` without required changes. It accepted the conditional paired-label interpretation and the separate registration of subsequent baseline-informed geometry. Actual paired outcomes have not been observed at registration.

Next step: does this matched Consolidation cohort have positive paired behavioral opportunity? Execute `scripts/server/run_qwen_paired_opportunity_gate.sh full Consolidation` from a clean pushed immutable source; accept the result after terminal internal and pinned review.

Evidence:

- Protocol: `docs/RESEARCH_PLAN.md#qwen-paired-behavioral-opportunity`.
- Source: `src/qwen_paired_opportunity.py`, `src/run_qwen_paired_opportunity.py` and the server launcher.
- Tests: `tests/test_qwen_paired_opportunity.py`, `tests/test_run_qwen_paired_opportunity.py` and the full local suite.
- Metadata acceptance: `docs/reviews/paired-opportunity-preparation.md`.
- Mass routing acceptance: `docs/reviews/qwen7b-mass-prompt-specificity-results.md`.
- Pinned receipt: `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T051705042400Z.json`; canonical `claude-fable-5-1`, effort `medium`, valid, read-only, tools disabled, exit 0. Server checkout stayed clean at `8fbae038a9a3933e44583fc43cfa7fc64d379647`.
