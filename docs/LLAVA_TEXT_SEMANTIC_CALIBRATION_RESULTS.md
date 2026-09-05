# LLaVA text semantic calibration

## Run

Immutable run `20260905T221642Z-f2b4bafc8a80-llava-text` evaluated the registered sixteen
descriptions under Y, incumbent-A, incumbent-B, candidate-A and candidate-B at pushed source
`f2b4bafc8a802e3f046e82318fd8b5b9aa762c52`. It completed all eighty text-only outcomes in
13.30 model-run seconds and 15 dispatcher seconds on one A100. Command, dispatcher and cleanup
statuses are zero.

## Observation

| Condition | Primary sign errors | Minimum oriented margin | Y LSE sign errors | Positive-minus-negative margins by pair for red / blue |
|---|---:|---:|---:|---|
| Y | 0 | 0.125 | 0 | 5.875, 5.625, 5.250, 1.125 / 5.750, 5.500, 6.000, 1.125 |
| incumbent-A | 7 | -3.500 | — | 3.500, 3.375, 3.375, 0.125 / 3.375, 3.125, 2.875, 0.250 |
| incumbent-B | 8 | -1.000 | — | 0.750, 0.250, 0.875, 0.250 / 1.125, 0.500, 1.125, 0.250 |
| candidate-A | 2 | -2.000 | — | 5.500, 5.875, 5.750, 0.250 / 5.250, 6.250, 5.250, 0.375 |
| candidate-B | 2 | -1.500 | — | 4.250, 4.500, 5.375, 0.375 / 4.125, 5.125, 5.750, 0.375 |

Y passes all sixteen primary signs and all sixteen log-sum-exp signs; its minimum oriented
log-sum-exp margin is `0.12500056`. Candidate-A and candidate-B each miss the pair-4 negative
description for both objects. The complete prompts, retained logits, partitions, token identities,
scores, pair summaries and error identities remain in the immutable artifact directory.

## Gate decision

`PASS`; scientific state `OBSERVED`, gate disposition `READY`, registered route `yes_no_route`.
Independent terminal replay verified the six decision-bearing manifest entries and reproduced all
eighty scores and the route. The pinned result review returned `PASS / OBSERVED / yes_no_route`
with no required actions. This constructed finite suite has no bootstrap or population inference.

## Next step

Can a registered Y-only wording and image-donor comparison separate wording sensitivity from
image-linked discrimination? Prepare that prospective protocol without evaluating its retained
index or donor patients.

Evidence: `/home/qingchan/data/concept-flow/runs/20260905T221642Z-f2b4bafc8a80-llava-text/`;
internal receipt
`/home/qingchan/data/concept-flow/state/llava-text-calibration-internal-20260905T2218Z/receipt.json`;
pinned result receipt
`/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T221855379998Z.json`.
