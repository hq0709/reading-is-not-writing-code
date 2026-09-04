# Ownership response diagnostics

Run: `20260904T224225Z-1ab7576193a7-ownership-diagnostics` analyzes the accepted 400-patient Qwen direction-by-question records at alpha `+0.25`. It preserves the original outcome grid and adds label-aligned discrimination and score-coordinate summaries. Source: `1ab7576`; primary artifact: `/home/qingchan/data/concept-flow/runs/20260904T224225Z-1ab7576193a7-ownership-diagnostics/artifacts/ownership-diagnostics.json`.

Observation: all six named ownership margins remain negative in both probability and logit-margin coordinates. The baseline-to-matched-direction AUROC point estimates are:

| Question | Baseline AUROC | Matched-direction AUROC | Logit ownership margin |
|---|---:|---:|---:|
| Effusion | 0.6178 | 0.5921 | -0.4853 |
| Atelectasis | 0.6235 | 0.5479 | -1.0703 |
| Pneumothorax | 0.6666 | 0.6324 | -0.8944 |
| Cardiomegaly | 0.6217 | 0.5152 | -0.8559 |
| Mass | 0.6636 | 0.6354 | -0.7134 |
| Nodule | 0.6805 | 0.6355 | -0.4134 |

Effusion's mean logit shift is 2.1336 on 29 positive patients and 2.2712 on 371 negative patients. The positive-minus-negative shift difference is -0.1376, with a pointwise patient-bootstrap 95% interval of [-0.4443, 0.1659]. Its Brier score changes from 0.0681 to 0.1035.

The first singular component accounts for 90.51% of the raw logit-effect matrix's squared Frobenius norm. After removing row means and column means and restoring the grand mean, the remaining interaction accounts for 7.91% of that same raw energy. These summarize the matrix geometry; they do not identify a causal shared channel.

Gate decision: the CPU run is terminal-successful. Ten targeted implementation tests and an independent internal code review passed. The descriptive scientific interpretation awaits the pinned read-only result review before paper handoff. AUROC differences above are point estimates; the existing registered gate decisions retain their original inferential status.

Next step: does the observed response structure follow clinical meaning or the answer-code mapping? Run the registered answer-encoding crossover with independent capability calibration and matched perturbation controls.
