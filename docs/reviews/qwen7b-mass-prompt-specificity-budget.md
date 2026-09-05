# Mass prompt-specificity execution budget

Run: validation preflight `20260905T014736Z-c5d7debb59f0-mass-prompts` measured 1,024 image-condition equivalents in 51.387 seconds, predicting 8,992.727 seconds for the 179,200-outcome grid. Token mapping, exact zero-dose behavior and the nonzero-logit response passed. The failed preflight is recorded in `docs/EXPERIMENT_REGISTRY.md`; its artifacts contain no calibration or confirmation outcomes.

Observation: a three-hour execution cap with a 2.75-hour predicted-scientific-time gate accommodates the measured validation throughput and reserves 15 minutes for loading and analysis. The original scientific registration remains fixed: the same seed-selected patients, seven prompts, dose, clinical/random/sham directions, calibration eligibility and joint statistical decisions. The slow-throughput test uses a 60-second pilot, which predicts 10,500 seconds and remains above the 9,900-second gate.

Gate decision: prospective budget review `PASS`, with no required corrections. Independent internal review accepted the complete change set, and 11 CPU tests plus syntax checks passed. Pinned receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T015535962388Z.json` verifies `claude-fable-5-1`, medium, valid read-only transport and unchanged clean checkout `c5d7debb59f01e4a7086404090f3ebea4d91a706`. The project-wide 336 GPU-hour and 168 wall-hour limits remain unchanged. Scientific outcomes are `PLANNED`.

Next step: does validation throughput fit the registered allowance on a new immutable execution? Following independent and pinned budget review, dispatch the unchanged full grid with a 180-minute timeout and retain all preflight gates.
