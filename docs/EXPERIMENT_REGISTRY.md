# Experiment registry

This registry indexes runs; the immutable receipt under `/home/qingchan/data/concept-flow/runs/<run-id>/` owns command, commit, environment, logs, statuses, timestamps, GPU inventory, and checksums. Failures are recorded once here and referenced elsewhere only while they affect a decision.

| experiment | question | protocol | state | run ID / evidence | decision effect |
|---|---|---|---|---|---|
| bootstrap-smoke | Can a pushed commit survive SSH disconnect, run in the fixed environment, and produce a verifiable receipt? | `scripts/server/dispatch_run.sh` harmless CUDA command | OBSERVED | `20260902T071539Z-7a4e7552d9e1-75d3510c`; `/home/qingchan/data/concept-flow/runs/20260902T071539Z-7a4e7552d9e1-75d3510c/` | Releases the infrastructure gate |
| corrected-llava-effusion-vislast | Does the true final LLaVA vision block selectively influence Effusion behaviour at a registered relative-token dose? | `docs/RESEARCH_PLAN.md#first-hard-gate`; llava15_7b / NIH / Effusion / vis.last / seed 0 | PLANNED | failed implementation validations `20260902T173900Z-358463289f6e-hook` at `358463289f6e3fbf5c7ddd0232a63126cbca58dd` and `20260902T174110Z-f893a3140f76-hook` at `f893a3140f76fad7a57b499198bb77b43fdfa736`; immutable receipts under `/home/qingchan/data/concept-flow/runs/` | `BLOCKED`: no NIH image entered the model and no outcome was measured; the fixed namespace and Transformers 5.x hidden-state recorder ordering both require correction and a passing synthetic proof before the one-shot test run |

New IDs use positive technical variables, never editing history or a prior failure as identity. A failed run entry records `run_id`, immutable commit, `FAILED`, failure class, receipt path, and current decision effect. A valid negative scientific result is `OBSERVED` and states the mechanism or regime boundary it establishes.
