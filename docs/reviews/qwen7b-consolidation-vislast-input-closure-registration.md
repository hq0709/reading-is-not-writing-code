# Qwen Consolidation input-closure registration review

Run: prospectively register `qwen7b-consolidation-vislast-input-closure` at clean pushed commit `8a55a4c2f6b6feeb7c8ccfaf818a342dffce1a56`. Internal validation re-derived the 50 within-patient pair receipt after excluding all 1,000 prior Qwen intervention rows and passed 130 repository tests with one platform skip. The configured `claude-review-concept-flow` transport reviewed the protocol, runner, cohort selection, endpoint, control family, uncertainty, and budget at pinned model `claude-fable-5-1`, medium effort, and read-only permissions.

Observation: the fixed pair list has SHA-256 `20dcc8cc3aaa8c65a6f7f759a0ff6bf3b0cf4f1b87f4b65cdc6a15a46c6d85e2`. The matching direction and all 126 controls receive the same signed displacement-derived `reltoken` dose capped at `|alpha|=0.25`; the primary clinical maximum is recomputed inside each of 5,000 patient-pair bootstraps.

Gate decision: registration `PASS`; gate disposition `READY` with no required actions. The accepted review receipt is `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T162255731007Z.json`; internal validation is `/home/qingchan/data/concept-flow/state/consolidation-registration-validation-20260904T161712Z-PRV2oo/`.

Next step: can a within-patient Consolidation representation displacement be closed by the matching probe-normal write beyond random, sham, and fixed clinical controls? Evaluate immutable run `20260904T162317Z-8a55a4c2f6b6-consolidation-closure`, then perform terminal replay and pinned hard-gate review.
