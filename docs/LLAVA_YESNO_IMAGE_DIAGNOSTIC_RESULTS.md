# LLaVA yes/no image diagnostic results

## Run

Immutable run `20260905T232313Z-2a95dfbed888-llava-yesno` executed the registered 700 index / 700 donor pair cohort under source commit `2a95dfbed888fce12b7481812065e5d68b712e8a`. It completed all 2,800 image outcomes and two text-only priors on one A100 in 260 dispatcher seconds. Validator-only recovery at `06277a1fb8a487a54b9220373957b1f1d988e2f5` reused the unchanged native artifact and wrote no model outcome; terminal validation at `1da1ca3c6da95218a99304f0edb65e3ab407e0de` replayed the registered allocation, scores, bootstrap family, routing and frozen-reader control.

## Observation

| wording | real AUROC (95% percentile CI) | donor AUROC against index labels (95% percentile CI) | donor-own AUROC | image advantage `G` (95% percentile CI) |
|---|---:|---:|---:|---:|
| accepted `0Y` | `0.5259` (`0.4135`, `0.6367`) | `0.4072` (`0.3085`, `0.5107`) | `0.4542` | `0.1187` (`-0.0013`, `0.2383`) |
| alternate `1Y` | `0.5089` (`0.3916`, `0.6255`) | `0.3975` (`0.2868`, `0.5113`) | `0.5117` | `0.1113` (`-0.0357`, `0.2583`) |

The registered 10,000-draw simultaneous family had radius `0.1304`:

| primary contrast | estimate | simultaneous 95% interval |
|---|---:|---:|
| `A0 - 0.5` | `0.0259` | (`-0.1045`, `0.1563`) |
| `G0` | `0.1187` | (`-0.0117`, `0.2491`) |
| `G1 - G0` | `-0.0073` | (`-0.1377`, `0.1231`) |

The opportunity conjunction is false and the wording effect is unresolved. The alternate wording shifts the mean real margin from `0.0488` to `0.7998` while worsening real-image Brier score from `0.2728` to `0.4604`; its descriptive `A1-A0` interval is (`-0.0650`, `0.0300`). Text-only yes margins are `0.8750` and `1.2500`, with answer-token mass `0.9932` and `0.9926`, respectively. Secondary log-sum-exp image advantages are `0.1273` (95% CI `0.0019`, `0.2504`) for `0Y` and `0.1131` (95% CI `-0.0377`, `0.2637`) for `1Y`; these are descriptive and do not replace the primary family.

The frozen clinical reader has AUROC `0.6664`, versus a mean `0.6775` across 20 frozen controls. Its selectivity is `-0.0111` (95% CI `-0.1541`, `0.1185`), so replication is false. Both score implementations retain mean answer-token mass above `0.9964`. The terminal replay's maximum cross-library float32 discrepancy is `3.814697265625e-6`, within the reviewed bounded numeric tolerance while the recorded native values remain authoritative.

## Gate decision

`PASS`; scientific state `OBSERVED`, gate disposition `READY`. The configured pinned read-only Claude result review accepted `opportunity=false`, `wording=unresolved`, and `reader_replicated=false`. Result-to-claim reports `claim_supported=no`, confidence `high`, integrity status `warn` from the inherited stale single-cell tracker, and routing action `pivot` to evidence synthesis. The registered diagnostic is complete and does not authorize a rerun.

## Next step

Can the accepted negative/unresolved diagnostic be integrated into the evidence-locked manuscript without widening its registered scope? Hand the accepted report and receipts to the paper task; no next experiment is registered from this gate.

Evidence: immutable run `/home/qingchan/data/concept-flow/runs/20260905T232313Z-2a95dfbed888-llava-yesno/`; recovered summary `/home/qingchan/data/concept-flow/state/llava-yesno-image-recovery-20260905T234235Z/llava-yesno-image-diagnostic-summary.json`; terminal receipt `/home/qingchan/data/concept-flow/state/llava-yesno-image-internal-20260905T235215Z/receipt.json`; pinned result receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T235348704249Z.json`; result-to-claim trace `.aris/traces/result-to-claim/2026-09-05_run02/`.
