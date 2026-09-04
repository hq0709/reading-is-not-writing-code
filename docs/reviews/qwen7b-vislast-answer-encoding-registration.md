# Qwen answer-encoding registration

Run: register and implement `qwen7b-vislast-answer-encoding` at pushed source `21dad2f`. The experiment compares 200 fixed intervention patients under three answer encodings and two signed doses, with 200 disjoint patients supplying clean capability calibration. The full design contains 200,400 scientific image-condition outcomes.

Observation: the fixed direction/control grid, patient split, score orientation, weighted AUROC, unbiased response-energy estimator, completeness checks, and validation-only preflight agree with the protocol. Nineteen targeted tests passed; the two tests affected by the final tolerance and throughput changes passed. Independent runtime review checked packed BF16 hooks at batches 1, 8, and 16. Both launchers passed shell syntax validation.

Gate decision: registration `PASS`, with gate disposition `READY`. The pinned `claude-fable-5-1` reviewer used medium effort and read-only transport. Final receipt: `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T230054355352Z.json`. The accepted descriptive ownership diagnostic is recorded separately in `docs/OWNERSHIP_DIAGNOSTICS.md`.

Next step: does the response follow finding meaning or answer-code identity? Immutable run `20260904T230152Z-21dad2f72726-answer-encoding` performs its validation preflight and then the registered full experiment on GPU 0.
