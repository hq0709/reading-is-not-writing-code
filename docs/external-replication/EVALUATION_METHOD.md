# Read/Write concept evaluation for vision-language models (cf-transfer-v1 as a benchmark)

This document states the evaluation method that the `cf-transfer-v1` protocol implements, in the form in
which it can be applied to any new checkpoint or dataset. Every rule below is the one the package executes
(`src/cftransfer/`); numbers in the campaign tables were produced by exactly these steps. The method grades a
model on three capabilities per concept and reports them as a leaderboard.

## 1. What is measured

For a checkpoint `M`, a dataset `D` with concept set `C` (six concepts) and a binary label per image, the
method asks three questions per concept `c`:

1. **Readable.** Is `c` linearly decodable from the visual representation the language model consumes?
2. **Answer-capable.** Does `M` answer the yes/no question about `c` better than chance on the same images?
3. **Owned (writable).** Does a write along the readout direction of `c` change `M`'s answer to the question
   about `c` more than it changes the answers about the other concepts, and more than a random or sham write
   of the same size?

"Reading is not writing" is the pattern in which (1) holds, (2) may hold, and (3) does not.

## 2. Locus: where the write happens

- **Primary locus `vis.last`**: the output tensor of the final visual block whose output the consumer reads,
  restricted to the token positions the consumer actually uses (CLS or padding rows are excluded when the
  consumer drops them). Each adapter documents the module path, the consumed mask and any stream that bypasses
  the block (`adapters/*.py`, recorded in `run.json`).
- **Second locus `connector`**: the output of the projector/connector that produces the language-model inputs.
- The steering hook writes `h' = h + alpha * ||h_t|| * v` per consumed token (relative dose, `alpha = +0.25`
  in the main modules), and the same hook pools the consumed tokens for feature extraction, so reading and
  writing use identical positions.

## 3. Procedure per (checkpoint, dataset)

1. **Cohort.** Fixed image cohort per dataset (`data/<ds>/manifests/cohort.csv`): 20,000 train, 16 preflight,
   400 calibration, 600 test rows, patient-disjoint, chosen by seeded hashes recorded in `manifests.py`.
2. **Features.** Pooled consumed-token features at both loci for every cohort row.
3. **Fit.** 512-d PCG64(seed 0) random projection, train-only standardisation, logistic regression
   (C=1, lbfgs, 2000 iterations) per concept, three seeds; 20 type->random-label control probes (label shuffles
   within image-type strata: view x sex x age decade on chest sets, aspect x area buckets on COCO). The write
   direction is `v_c = normalize(P @ (w / max(s, 1e-8)))`; a coordinate-permutation sham of `v_c` and 119
   PCG64(seed 0) random unit directions form the reference family.
4. **Preflight (acceptance gate).** On the 16 preflight rows and both loci: A determinism (0 difference on
   repeat), B alpha-0 identity, C reach (the write changes the consumer input and the answer logits and nothing
   outside the consumed tokens), D batch-vs-single consistency (tolerance 0.25 logits; larger values are
   *declared deviations* and neutralised by the fixed batch composition below), E image-free semantic mapping
   (the model maps "The finding is present/absent" statements to the right answer token under each of the six
   question templates; templates that fail are INELIGIBLE and the cells built on them are recorded as such),
   F throughput, G fp32-vs-model logits. The gate decisions are logged with the numbers.
5. **Scoring.** For every test row and concept question, the runner scores the clean answer and the answers
   under the concept write, the five competitor-concept writes, the sham and the 119 random directions, all in
   one fixed batch composition per question (bf16 kernels are shape-dependent, so every contrast is
   shape-matched). Answer logits are recomputed in fp32 from the final hidden state; the answer margin is
   `max(logit "yes" variants) - max(logit "no" variants)` and `p = sigmoid(margin)`.
6. **Modules.** CORE (the write matrix on the IY template), CALIBRATION (known-label probes, controls and the
   clean answers), PROMPT (the other five templates), DOSE (alpha sweep), REFIT (seeds 1 and 2), LOCUS (the
   connector locus) and LOCUS_CALIBRATION.

## 4. Decision rules

With 2,000 bootstrap draws over test rows (seeds 2026090601-03):

- **Readable**: selectivity `S = AUROC_real - mean(AUROC_control)` has a one-sided 95% lower bound > 0, with at
  least 10 positives and 10 negatives and at least 1,900 valid draws (draws with fewer than 10 usable controls
  are excluded). Otherwise the status is `not_readable`, `insufficient_support` or `insufficient_draws`.
- **Answer-capable**: the clean-answer AUROC on the IY template has a one-sided 95% lower bound > 0.5 (same
  support rules).
- **Write matrix**: `W_qd` = mean change in `p` for question `q` under the write of concept `d`.
  **Ownership** `O_q = W_qq - max_{d != q} W_qd`.
- **Steering reference**: `W_qq > 0`, `W_qq >` the 95th percentile of the 119 random-direction writes and
  `W_qq > |sham|`.
- **Owned**: the steering reference holds and every max-T simultaneous 95% lower bound of `W_qq - W_qd` over
  the five competitors is > 0 (verdict `fixed_family_advantage`). `stronger_competitor` when any upper bound is
  < 0; `unresolved` otherwise.
- **Controls reported with every block (T3)**: signed dose range (the write responds to alpha), refit SD
  (stability over seeds), connector-locus median O (the write at the connector is inert), label-shift gap
  (the answer change from relabelling exceeds the concept write), and PROMPT contrasts (wording IY-WY and
  mapping IA-IB per concept).

## 5. Outputs

- Per block: `runs/<model>/<dataset>/{coverage.csv, run.json, summary.json, probe_scores.parquet,
  outcomes/<MODULE>/*.parquet, preflight.*.json, template_eligibility.json}`.
- Main tables: `runs/main-tables.filled.csv` (`python -m cftransfer.tables`).
- Leaderboard: `runs/leaderboard.{csv,md}` (`python -m cftransfer.leaderboard`), one row per block with
  readable / answer-capable / owned counts and the owned concepts; blocks whose IY template is INELIGIBLE show
  the disposition instead of an ownership count.
- Log: `docs/external-replication/EXECUTION_LOG_mayo.md` (run, observation, gate decision for every step).

## 6. Adding a checkpoint

1. Register the pinned revision and adapter family (`adapters/__init__.py`); write an adapter only if the
   family's consumed boundary is new (`adapters/base.py` documents the contract: `loci()`, `layouts()`,
   `vision_features()`, `encode()`).
2. `python -m cftransfer.smoke --model-key <key>` on one GPU: the smoke must show determinism 0, alpha-0
   identity, reach at both loci and no change outside consumed tokens.
3. `python -m cftransfer.enqueue --models <key> --datasets nih,coco,chexpert`; workers
   (`python -m cftransfer.worker --lane ...`) run prep, preflight and every module; `package`, `analysis`,
   `tables` and `leaderboard` produce the outputs above. Per-model batch overrides live in `enqueue.BATCH_MODEL`.

## 7. Coverage of the campaign (2026-09-14)

22 checkpoints registered; 18 complete on all three datasets, llama32-11 complete with the yes/no modules
INELIGIBLE, iv35-38 and gemma3-27 (COCO/CheXpert) in progress, q25-72 (COCO/CheXpert) and llama32-90 deferred
(tasks held, resumable). The leaderboard is regenerated whenever a block is packaged.
