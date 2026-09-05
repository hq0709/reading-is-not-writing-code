# LLaVA paired opportunity result review

Run: `20260905T064036Z-1af90218a1d8-llava-opportunity`, immutable source `1af90218a1d8022bce66452826fa408ef146e432`, completed 200 image outcomes on the fixed 100 pairs in 38 seconds. Terminal command, dispatcher and cleanup statuses are zero.

Observation: strict mapping margins were 2.25, -1.375, 2.375 and -1.25; the repeated validation margin was exactly 0.375. The 128-equivalent pilot took 9.4469 seconds and the total prediction was 95.6220 seconds, within the 600-second bound. Singleton yes IDs were `[3869, 4874, 22483]`, and no IDs were `[694, 1939, 11698]`. The BF16 constructor was fixed in the immutable source; loaded parameter dtype was not separately recorded. Model loading completed with clean stderr.

Independent terminal verification checked eleven decision-bearing new-run files against the dispatcher manifest once. It accepted the complete original source records and patient order, reference copies, 200-row grid, per-patient JSON/NPZ values and exact shared bootstrap indices. The maximum independent numerical discrepancy was `5.55e-17`. Independent internal review accepted the receipt, preflight, endpoints and registered routing.

The LLaVA mean paired gap is -0.07125, with fifth-percentile bound -0.13375 and descriptive interval [-0.14625, 0.00125]. Concordance is 0.455, including 19 ties at half credit. The direct paired contrast against Qwen is -0.090, with interval [-0.210, 0.025].

Gate decision: `PASS`; scientific state `OBSERVED`, `opportunity_available=false`. Pinned review accepted valid nonconfirmation, the uncertainty in the direct model contrast and the evidence report.

Next step: how does direction-by-question semantic competition transfer to LLaVA? Register the controlled architecture-transfer comparison and deliver the accepted paired-response boundary to the paper task.

Evidence:

- Result report: `docs/PAIRED_OPPORTUNITY_ARCHITECTURE_RESULTS.md`.
- Registration: `docs/reviews/llava-paired-opportunity-registration.md`.
- Internal receipt: `/home/qingchan/data/concept-flow/state/llava-opportunity-internal-20260905T064202Z/receipt.json`.
- Pinned receipt: `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T064601848008Z.json`; canonical `claude-fable-5-1`, medium effort, valid and read-only, tools disabled, exit 0. The server checkout stayed clean at the immutable source.
