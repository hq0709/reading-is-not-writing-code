# Experiment audit report

Run: a fresh `gpt-5.6-sol` agent at ultra reasoning audited immutable run
`20260902T191411Z-ffd523c464c8-f99e2f39` directly against its source, dataset receipts, protocol, tracker,
and result files. This is a same-family provisional integrity review.

Observation:

| Check | Status | Finding |
|---|---|---|
| Ground-truth provenance | `PASS` | Effusion labels come from the registered official NIH label table and not model outputs. |
| Score normalization | `PASS` | Probe AUROC and intervention `P(yes)` are reported directly; feature standardization is train-fitted. |
| Result existence and fidelity | `WARN` | All 228 receipt entries and reconstructed values pass; tracked records required synchronization with the terminal run. |
| Dead code | `WARN` | Every gate-bearing metric is live; one legacy scalar `yes_prob` helper is unused. |
| Scope | `WARN` | Evidence covers one registered model, dataset, concept, and locus cell. |
| Evaluation type | `PASS` | Probe uses real dataset weak labels; intervention is a causal-response measurement on real held-out images; hook preflight is synthetic. |
| Patient split and protocol | `PASS` | Patient separation, exact dose/control grid, source binding, and selective-cell decision reproduce. |

Gate decision: overall audit `WARN`, reason `VALID_SINGLE_CELL_TRACKER_STALE_PINNED_REVIEW_PENDING`.
The warning does not identify a gate-invalidating defect. The pinned cross-family review subsequently
returned `PASS`, and the tracker/result synchronization is included with this report.

Next: which fully specified cross-cell experiment should next test whether decodability rank predicts
selective causal influence? Preserve per-image intervention outputs and realized control-vector identities
in that future registered protocol if they are required for its terminal evidence.
