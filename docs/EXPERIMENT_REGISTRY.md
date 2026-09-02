# Experiment registry

This registry indexes runs; the immutable receipt under `/home/qingchan/data/concept-flow/runs/<run-id>/` owns command, commit, environment, logs, statuses, timestamps, GPU inventory, and checksums. Failures are recorded once here and referenced elsewhere only while they affect a decision.

| experiment | question | protocol | state | run ID / evidence | decision effect |
|---|---|---|---|---|---|
| bootstrap-smoke | Can a pushed commit survive SSH disconnect, run in the fixed environment, and produce a verifiable receipt? | `scripts/server/dispatch_run.sh` harmless CUDA command | OBSERVED | `20260902T071539Z-7a4e7552d9e1-75d3510c`; `/home/qingchan/data/concept-flow/runs/20260902T071539Z-7a4e7552d9e1-75d3510c/` | Releases the infrastructure gate |
| corrected-llava-effusion-vislast | Does the true final LLaVA vision block selectively influence Effusion behaviour at a registered relative-token dose? | `docs/RESEARCH_PLAN.md#first-hard-gate`; llava15_7b / NIH / Effusion / vis.last / seed 0 | RUNNING | passing hook `20260902T174410Z-e2d891947414-hook`; active measurement `20260902T174520Z-e2d891947414-firstgate`; commit `e2d89194741475c37c7ace272a93d6092de91773`; earlier failed implementation validations `20260902T173900Z-358463289f6e-hook` and `20260902T174110Z-f893a3140f76-hook`; immutable receipts under `/home/qingchan/data/concept-flow/runs/` | `BLOCKED`: the sole corrected-gate test run is active; release requires complete bootstrap/control/intervention artifacts, receipt verification, and pinned read-only review |

New IDs use positive technical variables, never editing history or a prior failure as identity. A failed run entry records `run_id`, immutable commit, `FAILED`, failure class, receipt path, and current decision effect. A valid negative scientific result is `OBSERVED` and states the mechanism or regime boundary it establishes.
