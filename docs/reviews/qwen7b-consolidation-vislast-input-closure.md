# Qwen Consolidation input-closure hard-gate review

Run: immutable run `20260904T162317Z-8a55a4c2f6b6-consolidation-closure` evaluated the registered 50-patient paired closure protocol and produced 6,450 image-condition outcomes in 445 seconds on one A100. Terminal statuses were zero with no abort. The validator-only direction-order correction at pushed commit `0046d1e506c1b7a13fff02777cc16c8a358accf5` reused the original artifact; all 257 replay arrays match exactly and 131 repository tests ran with one platform skip. The configured `claude-review-concept-flow` transport reviewed the complete decision packet at pinned model `claude-fable-5-1`, medium effort, and read-only permissions.

Observation: the Consolidation probe is eligible with selectivity `0.0642` (95% CI `[0.0302, 0.0976]`). Mean within-patient displacement is `0.1091`, but its one-sided lower bound is `-0.0364`. The concept closure gain is `0.0022`, below the random maximum `0.0025`; the clinical-familywise one-sided lower bound is `-0.0015`. The registered decision is `input_closure=false`.

Gate decision: `PASS`; scientific state `OBSERVED`, gate disposition `READY`, and route `evidence_locked_claim_and_manuscript`, with no required actions. The accepted review receipt is `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T163729487366Z.json`; internal validation is `/home/qingchan/data/concept-flow/state/consolidation-input-closure-validator-replay-a-20260904T1634XXZ-F1ruMl/receipt.json`.

Next step: can the completed six-gate sequence support a precisely scoped concept-specific causal-use claim without another experiment? Run the evidence-locked result-to-claim and manuscript reconciliation gate.
