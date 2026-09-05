# LLaVA validation opportunity

## Run

Immutable run `20260905T093108Z-42a43207c848-llava-validation` used clean pushed source `42a43207c848acfcadecf2f3e0bf866ea71d9dc7`, one A100 and the registered validation-only protocol. It completed 4,200 calibration outcomes in 366 seconds; command, dispatcher and cleanup statuses are zero.

## Observation

The preflight reproduced exact zero-dose behavior, exact repeated clean logits and exact forwarded non-CLS hidden states. Its 2,418.317-second worst-case projection stayed below the 4,500-second gate.

| Question | Positive / negative | Reader selectivity (95% CI) | Reader pass | Clean-margin AUROC (95% CI) | Capability lower bound | Capability pass |
|---|---:|---:|:---:|---:|---:|:---:|
| Effusion | 34 / 666 | 0.1438 [0.0658, 0.2163] | yes | 0.4548 [0.3546, 0.5555] | 0.3722 | no |
| Atelectasis | 43 / 657 | 0.0472 [-0.0351, 0.1198] | no | 0.5275 [0.4473, 0.6111] | 0.4609 | no |
| Pneumothorax | 12 / 688 | 0.0362 [-0.1271, 0.1731] | no | 0.5122 [0.3785, 0.6400] | 0.4028 | no |
| Cardiomegaly | 25 / 675 | 0.0889 [0.0002, 0.1723] | yes | 0.5232 [0.4034, 0.6375] | 0.4240 | no |
| Mass | 39 / 661 | 0.0735 [-0.0023, 0.1473] | no | 0.5518 [0.4584, 0.6514] | 0.4723 | no |
| Nodule | 48 / 652 | 0.0037 [-0.0833, 0.0865] | no | 0.5536 [0.4601, 0.6482] | 0.4731 | no |

Every reader and capability bootstrap produced all 2,000 registered valid draws. Effusion and Cardiomegaly pass the frozen-direction reader screen; none of the six questions passes the clean singleton yes/no capability screen. The joint qualification count is therefore `K=0`. The protected write cohort remained untouched and `per-image.csv` contains only its registered header.

This is a measurement-stage separation: controlled direction-level information is readable for two questions but is not reliably expressed by the fixed answer margin. Cardiomegaly's reader lower bound is only 0.000164, so Effusion is the stronger anchor for follow-up. The pilot does not localize the separation to answer-token routing, verbalizer choice, prompt or language-prior effects; causal write efficacy was not evaluated because no question qualified.

## Gate decision

`PASS`; scientific state `OBSERVED`, gate disposition `READY`, and registered route `evidence_synthesis`. Independent terminal replay verified twelve decision-bearing files from the dispatcher manifest, reproduced allocation, controls, bootstrap seeds and decisions, and found maximum numerical discrepancy `0.0`. The pinned read-only Claude review returned `PASS / READY` with no required actions. The result-to-claim judgment is `claim_supported=yes`, confidence high, integrity pass and routing action `confirm` for the scoped measurement-stage claim. Evidence is the immutable artifact directory `/home/qingchan/data/concept-flow/runs/20260905T093108Z-42a43207c848-llava-validation/artifacts/`, internal receipt `/home/qingchan/data/concept-flow/state/llava-validation-opportunity-internal-20260905T093743Z/receipt.json`, pinned result receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T094007326743Z.json` and deterministic claim pre-check `.aris/evidence_precheck.json`.

## Next step

Can a prospectively registered Effusion diagnostic distinguish prompt/language-prior sensitivity from answer-token verbalizer routing while independently replicating frozen-direction availability? Register a fresh non-protected validation diagnostic with prompt-paraphrase, verbalizer/tokenization, no-image or shuffled-image controls and simultaneous reader replication; do not execute new GPU work until its protocol, implementation and pinned review gates pass.
