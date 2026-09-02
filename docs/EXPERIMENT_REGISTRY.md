# Experiment registry

This registry indexes runs; the immutable receipt under `/home/qingchan/data/concept-flow/runs/<run-id>/` owns command, commit, environment, logs, statuses, timestamps, GPU inventory, and checksums. Failures are recorded once here and referenced elsewhere only while they affect a decision. Terminal asset and run receipts are reused; validators and heartbeats do not rehash complete runs, shards, models, or datasets without a new terminal hard gate or concrete contamination evidence.

| experiment | run | observation | gate decision | next step | evidence |
|---|---|---|---|---|---|
| bootstrap-smoke | `scripts/server/dispatch_run.sh` harmless CUDA command | The pushed commit completed after SSH disconnect and produced a verifiable immutable receipt. | `PASS` | Can the registered `vis.last` synthetic GPU preflight prove exact forward-path capture? | `20260902T071539Z-7a4e7552d9e1-75d3510c`; `/home/qingchan/data/concept-flow/runs/20260902T071539Z-7a4e7552d9e1-75d3510c/` |
| nih-chestxray14-trust-bootstrap | Official NIH asset staging plus independent source/content trust chain | 112,120 images and a 26,229-row research manifest are staged; the trust and asset receipts are terminal evidence. | `PASS` | Can the registered `vis.last` synthetic GPU preflight prove exact forward-path capture? | `/home/qingchan/data/concept-flow/state/nih-chestxray14-independent-verification-5342b4219127/receipt.json`; `/home/qingchan/data/concept-flow/datasets/nih-chestxray14/asset-receipt.json` |
| llava-effusion-vislast-reltoken | `llava15_7b` / NIH / Effusion / `vis.last` / relative-token intervention / seed 0 | Corrected hook preflight `20260902T191425Z-ffd523c464c8-hook` proves exact layer-22 capture, an alpha-zero bitwise no-op, and downstream effect. Full run `20260902T194105Z-caed32c3879a-firstgate` is executing the registered protocol from the staged snapshot. | `BLOCKED` pending terminal full-run evidence and pinned read-only review. | Does the registered intervention produce a selective concept-consistent change relative to same-alpha controls? | `docs/RESEARCH_PLAN.md#first-hard-gate`; `/home/qingchan/data/concept-flow/runs/20260902T191425Z-ffd523c464c8-hook/`; `/home/qingchan/data/concept-flow/runs/20260902T194105Z-caed32c3879a-firstgate/` |

## Implementation failure ledger

| run id | immutable commit | state | failure class | decision effect | evidence |
|---|---|---|---|---|---|
| `20260902T173900Z-358463289f6e-hook` | `3584632` | `FAILED` | registered target module absent | corrected the LLaVA module namespace; no scientific observation | `/home/qingchan/data/concept-flow/runs/20260902T173900Z-358463289f6e-hook/` |
| `20260902T174110Z-f893a3140f76-hook` | `f893a31` | `FAILED` | steering ran after the Transformers hidden-state recorder | prepended the steering hook; no scientific observation | `/home/qingchan/data/concept-flow/runs/20260902T174110Z-f893a3140f76-hook/` |
| `20260902T174520Z-e2d891947414-firstgate` | `e2d8919` | `FAILED` | offline extractor used repository identity instead of the staged snapshot | requires snapshot-bound extraction; no scientific observation | `/home/qingchan/data/concept-flow/runs/20260902T174520Z-e2d891947414-firstgate/` |
| `20260902T191522Z-ffd523c464c8-firstgate` | `ffd523c` | `FAILED` | allocated GPU memory was exhausted by concurrent resident processes during intervention startup | registered unchanged-protocol rerun on the alternate allocated GPU; no scientific observation | `/home/qingchan/data/concept-flow/runs/20260902T191522Z-ffd523c464c8-firstgate/` |

New IDs use positive technical variables. A failed run entry records `run_id`, immutable commit, `FAILED`, failure class, receipt path, and decision effect. A valid negative scientific result is `OBSERVED` and states the mechanism or regime boundary it establishes.
