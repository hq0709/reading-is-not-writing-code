# Research state

Updated: 2026-09-02. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

## Bootstrap gate

- Run: infrastructure bootstrap and immutable smoke run `20260902T071539Z-7a4e7552d9e1-75d3510c`.
- Observation: Git handoff, fixed environment, reviewer transport, ARIS pin, tmux persistence, immutable dispatch, fetch, and safety acceptance have terminal receipts.
- Gate decision: `PASS`.
- Gate disposition: `READY`.
- Next step: can the registered `vis.last` synthetic GPU preflight prove exact forward-path capture?
- Evidence: `/home/qingchan/data/concept-flow/state/aris-audit.env`, `/home/qingchan/data/concept-flow/state/health.env`, `/home/qingchan/data/concept-flow/state/bootstrap-safety-acceptance.env`, `/home/qingchan/data/concept-flow/state/setup-receipt.tsv`, and `/home/qingchan/data/concept-flow/runs/20260902T071539Z-7a4e7552d9e1-75d3510c/`.

## First hard gate: `llava-effusion-vislast-reltoken`

- Run: exact hook preflight `20260902T191229Z-ffd523c464c8-db7dce6e` and immutable full run `20260902T191411Z-ffd523c464c8-f99e2f39` completed the registered NIH protocol from commit `ffd523c464c84417a93c5a6d0a34e5b74e55e76e` on one A100 in 7,022 seconds.
- Observation: exact `encoder.layers.22` capture fired once, alpha zero was a bitwise no-op, and relative-token steering changed the downstream connector by `1.1875` and final logits by `0.1875`. Effusion AUROC was `0.7788` (95% patient-bootstrap CI `0.7538–0.8031`); the 20-seed control mean was `0.6649`, giving selectivity `0.1139` (95% CI `0.0875–0.1404`). The maximum concept-consistent behavioural change was `0.1470` at alpha `-1`, above the same-alpha random 95th percentile `0.1202` and below the maximum absolute sham effect `0.1546`; `selective_cell=false`. The response over `[-0.5,0.5]` was monotone with Spearman rho `1.0`.
- Scientific state: `OBSERVED`; this registered cell contains linearly decodable Effusion information and no selective causal effect under the registered threshold.
- Gate decision: `PASS`; deterministic replay was byte-identical, the decision-bearing dispatcher checksums passed, and the pinned read-only Claude review returned `PASS` with no required actions.
- Gate disposition: `READY`.
- Evidence: run receipt `/home/qingchan/data/concept-flow/runs/20260902T191411Z-ffd523c464c8-f99e2f39/`; internal replay `/home/qingchan/data/concept-flow/state/first-gate-internal-validation-20260902T211900Z/intervention-summary.json`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260902T212456605398Z.json`; review record `docs/reviews/llava-effusion-vislast-reltoken.md`.
- Evidence reuse: the terminal trust, asset, hook, run, internal-validation, and reviewer receipts are inputs to later gates. Validators read them without rehashing the dataset, model, activation shard, or complete run unless a new terminal gate or concrete contamination evidence requires it.
- Next step: does the higher-controlled-decoding Edema cell at the same consumed LLaVA visual locus show selective causal influence under the unchanged threshold? Run the registered second hard gate.
- Writer ownership: server Codex/ARIS while the persistent server session is active; handoff follows the single-writer Git protocol.

## Second hard gate: `llava-edema-vislast-reltoken`

- Run: immutable run `20260902T221534Z-110b84618d1b-edema` completed the fixed Edema protocol from commit `110b84618d1bcfe23d5e3687cf0ff9e2a5f789c2` on one A100 in 1,823 seconds, reusing the accepted prompt-independent `vis.last` activation artifact and the first gate's 200 held-out intervention rows.
- Observation: Edema AUROC was `0.8009` (95% patient-bootstrap CI `0.7648–0.8329`) and selectivity was `0.1360` (95% CI `0.0994–0.1693`). Its point selectivity exceeded Effusion by `0.0221`, with a paired CI of `-0.0175–0.0590`. The maximum concept-consistent behavioural change was `0.0659` at alpha `-1`, within the same-alpha random interval `-0.1304–0.1470` and below the maximum absolute sham effect `0.0913`; `selective_cell=false`. Spearman rho over `[-0.5,0.5]` was `-0.4286`. This is the registered descriptive rank-discordant pair, with uncertainty spanning the selectivity ordering.
- Scientific state: `OBSERVED`.
- Gate decision: `PASS`; terminal checksums passed, deterministic replay was byte-identical, and the pinned read-only Claude review returned `PASS` with no required actions.
- Gate disposition: `READY`.
- Evidence: run receipt `/home/qingchan/data/concept-flow/runs/20260902T221534Z-110b84618d1b-edema/`; internal replay `/home/qingchan/data/concept-flow/state/edema-gate-internal-validation-20260902T224700Z/intervention-summary.json`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260902T224851552731Z.json`; review record `docs/reviews/llava-edema-vislast-reltoken.md`.
- Next step: does the Effusion probe-normal mismatch generalise from LLaVA to Qwen at the consumed final visual block? Run the registered third hard gate.
- Writer ownership: local Codex owns registration and runner implementation; no experiment process is active.

## Third hard gate: `qwen7b-effusion-vislast-reltoken`

- Run: immutable asset stage `20260903T000153Z-caaae3ef346d-qwen-stage`, hook preflight `20260903T000239Z-caaae3ef346d-qwen-hook`, and full run `20260903T000321Z-caaae3ef346d-qwen-full` completed the fixed Qwen Effusion protocol from commit `caaae3ef346d3ac01c76c53ba99b3b7237066639` on one A100 in 2,077 seconds.
- Observation: exact block-31 capture and 26,229-row extraction passed with no failed rows. Effusion AUROC was `0.7742` (95% patient-bootstrap CI `0.7488–0.7993`) and selectivity was `0.1166` (95% CI `0.0897–0.1428`); Qwen-minus-LLaVA selectivity was `0.0026` (paired 95% CI `-0.0155–0.0215`). The maximum concept-consistent behavioural change was `0.2343` at alpha `+0.25`, above the same-alpha random 95th percentile `0.0963` and maximum absolute sham effect `0.0852`; `selective_cell=true`. Spearman rho over `[-0.5,0.5]` was `0.9643`. The reported Nodule-direction effect at alpha `+0.25` was `0.2863`.
- Scientific state: `OBSERVED`; this registered cell contains linearly decodable Effusion information and meets the registered selective causal threshold.
- Gate decision: `PASS`; terminal checksums passed, deterministic replay was byte-identical, and the pinned read-only Claude review returned `PASS` with no required actions.
- Gate disposition: `READY`.
- Evidence: full-run receipt `/home/qingchan/data/concept-flow/runs/20260903T000321Z-caaae3ef346d-qwen-full/`; internal replay `/home/qingchan/data/concept-flow/state/qwen-gate-internal-validation-20260903T003856Z/intervention-summary.json`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260903T004040206620Z.json`; review record `docs/reviews/qwen7b-effusion-vislast-reltoken.md`.
- Next step: what claim is supported by the completed three-cell sequence, including its architecture and direction-specificity boundaries? Evaluate the result-to-claim boundary before registering another experiment.
- Writer ownership: server Codex/ARIS; no experiment process is active.

## Result-to-claim gate

- Run: deterministic evidence pre-check plus the configured pinned Claude read-only review evaluated the accepted three-cell sequence against the primary claim.
- Observation: all 12 cited values exist in their primary artifacts. All three final-block cells have positive controlled selectivity. Both LLaVA cells remain non-selective against the registered conjunction; Qwen Effusion is registered-selective against random/sham, while its fixed Nodule-direction effect (`0.2863`) exceeds its Effusion-direction effect (`0.2343`).
- Scientific state: `OBSERVED`; the supported interpretation is a final-locus dissociation between robust linear decodability and direction-specific causal influence in the registered NIH cells.
- Gate decision: `PASS`; the pinned cross-family review returned `claim_supported=partial`, confidence `high`, integrity status `pass`, and routing action `supplement`.
- Gate disposition: `READY` for prospective supplement registration; no further experiment is currently registered for dispatch.
- Evidence: `CLAIMS_FROM_RESULTS.md`; `findings.md`; `.aris/evidence_precheck.json`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260903T013453088626Z.json`; trace `.aris/traces/result-to-claim/2026-09-02_run01/`.
- Next step: can the registered independent-row Qwen supplement show that the Effusion probe normal exceeds the fixed unrelated clinical directions with familywise bootstrap uncertainty? Dispatch when one A100 satisfies the registered free-memory threshold.
- Writer ownership: server Codex/ARIS; no experiment process is active.

## Direction-specificity supplement: `qwen7b-effusion-vislast-direction-specificity`

- Run: prospectively registered implementation for a 400-patient independent confirmation at the discovery-locked alpha `+0.25`, reusing the accepted Qwen activation and direction artifacts.
- Observation: no new scientific outcome has been consumed. The registered row-selection hash is `76be1da9659973d44a0898fa69c579afc4649f329ee227b19d1216efaa1c8bda`; the primary statistic places the maximum of the five fixed clinical controls inside each of 5,000 patient-bootstrap replicates.
- Scientific state: `PLANNED`.
- Gate decision: registration `PASS`; internal verification passed at clean pushed commit `ff3c1e021912bfa00801c29c920611c99ede0f58`, and the pinned read-only Claude review returned `PASS` with no required actions.
- Gate disposition: `BLOCKED` on GPU availability; both A100s exceed the registered `<500 MiB` used-memory launch threshold, and no experiment process is active.
- Evidence: `docs/RESEARCH_PLAN.md#direction-specificity-supplement`; `/home/qingchan/data/concept-flow/state/direction-specificity-registration-validation-20260903T022000Z/receipt.json`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260903T021501392963Z.json`; `docs/reviews/qwen7b-effusion-vislast-direction-specificity-registration.md`.
- Next step: can the registered independent-row Qwen supplement show that the Effusion probe normal exceeds the fixed unrelated clinical directions with familywise bootstrap uncertainty? Dispatch the immutable payload when one A100 satisfies the free-memory threshold.
- Writer ownership: server Codex/ARIS.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt. `RUNNING -> FAILED` records implementation, measurement, or infrastructure invalidity in the run/failure ledger. `RUNNING -> OBSERVED` requires valid positive or negative scientific evidence. A passed gate advances immediately to the next authorised experiment; validator-only changes reuse the original artifact.
