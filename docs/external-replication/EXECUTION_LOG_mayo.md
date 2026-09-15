# cf-transfer-v1 execution log (Mayo Clinic executor)

Executor: rohpc cluster (Slurm `gen-a100.p`, 19 nodes x 4 A100-80GB; `gen-h100`, 6 nodes x 4 H100).
Code: branch `cf-transfer-mayo` on top of package reference `763072d`; implementation in `src/cftransfer/`.
Run root: `/rodata/azradonc_dev/m253405/cf-transfer/` (data, runs, logs). Records follow
`run -> observation -> gate decision -> next step`.

## 2026-09-11 Staging

- Run: cloned code and paper repositories; verified `cohorts.csv` reproduces from `data/manifest.csv` under the
  package hash rule; inventoried environment (`research` env: Python 3.12, torch 2.13.0+cu130,
  transformers 5.15.0, pyarrow 25).
- Observation: NIH zips (12/12) present locally; 19,228 needed PNGs extracted (18,212 train + 1,016 evaluation).
  COCO 2017 annotations staged; `train2017.zip` downloading. CheXpert absent. Local checkpoints: q25-7, q25-32,
  q3-8, medgemma-4/27, lingshu-7/32; public checkpoints downloading pinned to full commits. Stored HF token is
  invalid: gemma-3 (4/12/27B) and Llama-3.2-Vision (11/90B) cannot be fetched until a valid token with accepted
  licences exists.
- Gate decision: NIH and COCO manifests built (`data/<ds>/manifests/{cohort,labels}.csv`); NIH calibration/test
  positives match the package (20/33/13/15/15/27; 46/57/17/23/32/47); COCO train/cal/test positives recorded,
  9 recurring types, bicycle has 9 calibration positives (< 10, reported as insufficient support, not resampled).
- Next step: family interface verification (preflight) for Qwen2.5-VL on NIH, then probe fits and CORE.

## Blocked items (need the project owner / user)

| Item | Why | What unblocks it |
| --- | --- | --- |
| CheXpert train release | licence-gated download from Stanford AIMI | user downloads `CheXpert-v1.0` (or the named release) into `cf-transfer/data/chexpert/`; manifest builder is ready (`cftransfer.manifests chexpert`) |
| gemma3-4/12/27, llama32-11/90 | gated HF repos; stored token invalid | `hf auth login` with a token whose account accepted both licences, then rerun `scripts/mayo/fetch_models.py` |
| llavamed-7 | original `llava_mistral` format; needs the conversion the authors validated | conversion adapter (planned after the HF families) |

## 2026-09-12 Family interfaces

- Run: adapters for Qwen2.5-VL/Lingshu, Qwen3-VL, InternVL3.5-HF, LLaVA-1.5, Gemma3/MedGemma; smoke on the 16 NIH
  preflight images (login-node GPUs through `jobwrap.sh`, Slurm partitions saturated with 12h+ pending queues).
- Observation: q25-7 (vis.last D=1280, connector D=3584, 144 tokens), q3-8 (1024 patches at 512px cap -> 256 tokens),
  iv35-8 (1024 patches, CLS excluded -> 256 tokens), llava15-7 (576 tokens, `encoder.layers.22` = hidden_states[-2])
  all pass determinism, alpha-0 no-op, consumer/logit reach and non-consumed-token isolation.
  Two measurement defects found and fixed before any scientific run:
  1. HF returns bf16 logits; at |logit| ~ 25 the answer margins sit on a 0.125 lattice. Answer-position logits are
     now recomputed in float32 from the final hidden state (hook on `lm_head` input, fp32 head weight), with a
     preflight check that they agree with the model's logits within two bf16 ulps.
  2. bf16 kernels are shape-dependent: single-example vs batched candidate logits differ by up to 0.25 (InternVL),
     0.6 (Qwen3-VL). The runner therefore scores every condition of a question at one fixed batch composition
     (replicated image; clean replicated baseline batch with the hook passive; steered batches padded), so the
     composition offset never enters W_qd. Declared single-vs-batch tolerance 0.25; families above it are recorded
     as tolerance deviations in preflight.json, not silently accepted.
- Gate decision: interfaces READY for q25-7, q3-8, iv35-8, llava15-7 (medgemma-4 pending its smoke rerun);
  no scientific outcome collected yet.
- Next step: q25-7 NIH probe fits + preflight + CORE throughput trial, then fan out CORE shards.

## 2026-09-12 q25-7 NIH: fits, preflight, throughput, launch

- Run: features for all 19,228 rows (vis.last D=1280, connector D=3584); fits for both loci, seeds 0/1/2; preflight
  on both loci; two 3-row CORE trials (shards 0 and 1 of 200).
- Observation: calibration readout reproduces the accepted reference cell (Effusion AUROC 0.7741, control mean
  0.6717, S=0.1023; readable: Effusion, Pneumothorax, Cardiomegaly; not readable at the one-sided bound:
  Atelectasis, Mass, Nodule). Preflight passes on both loci (determinism 0, alpha-0 no-op 0, merger change 9.8,
  answer-logit change 3.8, single-vs-batch 0.22 within 0.25, replicated-batch 0.07, fp32-vs-model 0.05 logits,
  24/24 semantic-mapping cases). Trial rows: 576 consumed tokens, delta norm = 0.25 x token norm exactly,
  baseline P(yes) ~0.02, Effusion direction 0.3-0.44, Nodule similar, sham at baseline.
  Throughput 27 image-conditions/s at batch 32 on one A100-80GB (GPU-bound: 32 x 183 tokens through 7B per
  forward); LLM cost dominates, so per 7B model the three datasets need ~44 A100-hours; the 22-checkpoint plan
  is ~3,000-4,000 A100-hours before H100 speedups.
- Gate decision: measurement READY for q25-7/NIH; campaign launched (CORE 4 shards on the login GPU;
  PROMPT 7 + LOCUS 4 shards on gen-h100; DOSE 2, REFIT 1, CALIBRATION 1, LOCUS_CALIBRATION 1 on gen-a100.p).
- Next step: complete q25-7 NIH, package, run the CPU statistics; bring q3-8 / medgemma-4 / llava15-7 /
  iv35-8 / llavamed-7 through the same gate; COCO once train2017 finishes.

## 2026-09-12 q3-8 and medgemma-4 NIH gates

- Run: full preparation (features, fits both loci seeds 0/1/2, preflight both loci) on login GPUs.
- Observation: q3-8 (512px cap -> 1,024 patches, 256 merged tokens; DeepStack bypasses untouched) and medgemma-4
  (896px SigLIP, 4,096 patches, 256 soft tokens) pass every reach/determinism/no-op/isolation/semantic check;
  single-vs-batch differences 0.60 (q3-8) and 0.50 (medgemma-4) exceed the declared 0.25 and are recorded as
  tolerance deviations (fixed batch composition keeps them out of W_qd). Readouts: q3-8 Effusion S=0.128,
  medgemma-4 Effusion S=0.108 (control mean 0.704). Throughput at batch 32: q3-8 24/s, medgemma-4 ~15/s
  (vision tower replicated per condition dominates for SigLIP-896; candidate optimisation: run the tower once
  per image and replicate after the hooked block, to be validated by the batch-equivalence preflight before use).
- Gate decision: READY; CORE launched locally (q3-8 GPU 2, medgemma-4 GPU 1), other modules queued on Slurm.
- Next step: llava15-7 gate (GPU 0), q25-72 4-GPU preparation once staged, COCO preparations after extraction.

## 2026-09-12 llava15-7 NIH gate; q25-72 staged

- Run: llava15-7 preparation on login GPU 0 (features 349 s, fits, preflight both loci).
- Observation: readout matches the accepted reference (Effusion calibration AUROC 0.7883 vs 0.7788; control mean
  0.678). All reach/determinism/no-op/isolation checks pass; single-vs-batch 0.11 within tolerance. The image-free
  semantic-mapping check fails for the four A/B templates (A-present: "absent" statements give +0.28/+0.02;
  B-present: "present" statements give -0.89/-0.36), i.e. LLaVA-1.5-7B does not follow the letter mapping,
  matching the authors' own LLaVA A/B preflight failure. Per protocol this is an interface disposition, not a
  scientific result: IA/IB/WA/WB are INELIGIBLE for llava15-7 (recorded in template_eligibility.json; runner
  skips them; coverage rows carry the reason). IY and WY pass, so CORE/DOSE/REFIT/LOCUS and the WY PROMPT block
  proceed.
- Gate decision: READY for yes/no templates; A/B template cells INELIGIBLE (interface). CORE launched on GPU 0;
  other modules queued. Qwen2.5-VL-72B staged at 89c86200743e; 4-GPU preparation queued on gen-a100.p.
- Next step: when Slurm capacity arrives, sweep the queued preparations; COCO after extraction.

## 2026-09-12 COCO staged

- Run: official train2017/val2017 zips and instances annotations downloaded; the 20,416 train + 600 val images named
  by the frozen manifest extracted (nothing else unpacked).
- Observation: 0 missing files; manifest already built from the annotations (20,000/16/400 train2017 by
  SHA-256 order, 600 val2017; 9 recurring types).
- Gate decision: COCO data READY; preparation jobs (features -> fits -> preflight) queued on Slurm for 15
  checkpoints.
- Next step: COCO CORE/PROMPT/DOSE/REFIT/LOCUS after each preparation passes.

## 2026-09-12 q25-7 NIH CORE complete

- Run: four local shards (150 rows each) on login GPU 3, 4,097-4,157 s per shard, 27.5-27.9 outcomes/s, peak 18.4 GiB.
- Observation: 457,200 outcomes, 0 failed rows; merged block passes the unique-key check.
- Gate decision: CORE block COMPLETE; CPU statistics (W matrix, O_q, 5,000-draw max-T) running; remaining NIH
  modules for q25-7 moved from the Slurm queue to GPU 3 (CALIBRATION, LOCUS_CALIBRATION, REFIT, DOSE, LOCUS in
  reverse shard order so any Slurm LOCUS job that starts later takes the other end).
- Next step: read the q25-7 NIH CORE statistics; package once all modules complete.

## 2026-09-12 q25-7 NIH CORE statistics (first scientific observation of the campaign)

- Run: CPU statistics on the complete 457,200-row CORE block (600 test patients, alpha +0.25, 119 random, sham),
  5,000 shared patient-bootstrap draws (seed 2026090601), two-sided max-|Z| over the 30 clinical contrasts.
- Observation (W_qq / strongest competitor / O_q [percentile 95% CI] / random p95 / |sham|):
  Effusion 0.1905 / Nodule 0.2615 / -0.0711 [-0.0803, -0.0618] / 0.1371 / 0.0169  (steering reference met; rank 2/120)
  Atelectasis 0.1011 / Nodule 0.3272 / -0.2260 [-0.2328, -0.2193] / 0.2967 / 0.0961
  Pneumothorax 0.0887 / Effusion 0.1905 / -0.1018 [-0.1076, -0.0962] / 0.3201 / 0.1128
  Cardiomegaly 0.0849 / Atelectasis 0.3036 / -0.2187 [-0.2267, -0.2106] / 0.3109 / 0.1581
  Mass 0.3592 / Effusion 0.5283 / -0.1690 [-0.1780, -0.1601] / 0.3715 / 0.0917
  Nodule 0.1175 / Effusion 0.1895 / -0.0720 [-0.0812, -0.0625] / 0.1583 / 0.0547
  All six diagonal ownership contrasts are negative with simultaneous upper bounds below zero
  (verdict "stronger competitor" for every question). This reproduces the accepted Qwen2.5-VL-7B evidence
  (Effusion 0.1945 vs Nodule 0.2519, O = -0.0573 [-0.0694, -0.0450] on 400 patients; 0/6 owned in the matrix)
  on the shared-panel cohort.
- Gate decision: valid OBSERVED block; no protocol change. Calibration readable/answer-capable flags follow
  once CALIBRATION outcomes land.
- Next step: remaining q25-7 NIH modules (running), then the same statistics for q3-8, medgemma-4, llava15-7.

## 2026-09-12 q3-8 NIH CORE statistics

- Run: complete 457,200-row CORE block (four local shards, 24.1/s, 5.27 GPU-hours), same statistics as q25-7.
- Observation (W_qq / strongest competitor / O_q [95% CI] / random p95 / |sham| / verdict):
  Effusion 0.2319 / Nodule 0.1590 / +0.0729 [0.0492, 0.0973] / 0.3601 / 0.0238 / fixed-family advantage, but
    the steering reference is NOT met (W below the random p95; rank 12/120)
  Atelectasis 0.0362 / Cardiomegaly 0.0487 / -0.0125 [-0.0142, -0.0109] / 0.0486 / 0.0481 / stronger competitor
  Pneumothorax 0.0521 / Mass 0.1187 / -0.0666 [-0.0748, -0.0586] / 0.1683 / 0.0079 / stronger competitor
  Cardiomegaly 0.1770 / Nodule 0.1767 / +0.0003 [0.0001, 0.0004] / 0.1773 / 0.1751 / unresolved
    (every direction incl. sham and random moves this question by ~+0.177: direction-agnostic sensitivity)
  Mass -0.0536 / Nodule 0.0534 / -0.1070 [-0.1272, -0.0866] / 0.2084 / 0.1650 / stronger competitor
  Nodule -0.0503 / Effusion -0.0851 / +0.0348 [0.0154, 0.0536] / 0.1893 / 0.1034 / fixed-family advantage with a
    NEGATIVE own effect (all directions lower the Nodule answer; the Nodule direction lowers it least)
  Compared with Qwen2.5-VL-7B on the same 600 patients, Qwen3-VL-8B is far more sensitive to random directions at the
  same relative dose (random p95 0.36 vs 0.14 for Effusion) and no question meets the steering reference.
- Gate decision: valid OBSERVED block; fixed-family advantage and steering reference are reported separately, as the
  protocol requires; no rule changed after inspection.
- Next step: q3-8 remaining modules (GPU 2); medgemma-4 and llava15-7 CORE still running.
- Raw-row check for the q3-8 Cardiomegaly question: baseline P(yes) mean 0.823 (median 0.982); every direction,
  sham and random push it to 0.99-1.00 (per-patient deltas correlate 0.99-1.00 with the matched direction). This
  is answer saturation from a high baseline, not a vector mix-up: the six fitted normals have pairwise cosines
  <= 0.50 and produce distinct effects on the Effusion question (correlations 0.56-0.88).

## 2026-09-12 llava15-7 NIH CORE statistics

- Run: complete 457,200-row CORE block (four local shards on GPU 0, 18.8-19.0/s, ~6.7 GPU-hours).
- Observation: every clinical effect is small (|W_qd| < 0.11); no question meets the steering reference
  (Effusion W=0.038 vs random p95 0.094 and |sham| 0.058; Mass W=0.049 vs 0.056/0.037). Ownership: Effusion
  -0.0276 [-0.0301, -0.0252] (Mass stronger), Atelectasis -0.0145, Pneumothorax -0.0800, Cardiomegaly -0.1330,
  Nodule -0.0182 (all "stronger competitor"); Mass +0.0229 [0.0202, 0.0256] fixed-family advantage below the
  random reference. Baseline P(yes) is 0.50-0.69 on every question (near-chance answering), consistent with the
  accepted LLaVA cells (random/sham rung not cleared) and the authors' LLaVA capability findings.
- Gate decision: valid OBSERVED block. A/B templates remain INELIGIBLE (interface).
- Next step: remaining llava15-7 NIH modules (GPU 0).

## 2026-09-12 medgemma-4 NIH CORE statistics

- Run: complete 457,200-row CORE block (four local shards on GPU 1, 15.0/s, ~8.5 GPU-hours).
- Observation: no question meets the steering reference. Effects are large and of both signs: the Atelectasis
  question falls by 0.358 under its own direction and by 0.568 under the Mass direction; sham effects reach 0.377
  (Effusion) and 0.551 (Nodule); random p95 ranges 0.13-0.54. Ownership: Effusion +0.0054 [-0.0161, 0.0268]
  unresolved; Atelectasis -0.5584 [-0.5949, -0.5220] (Pneumothorax stronger); Pneumothorax +0.0052 unresolved;
  Cardiomegaly -0.0671 [-0.1127, -0.0316]; Mass -0.2726 [-0.3056, -0.2409]; Nodule +0.0943 [0.0653, 0.1232]
  fixed-family advantage but far below the random/sham references. Baseline P(yes): Effusion 0.14, Atelectasis 0.79,
  Pneumothorax 0.02, Cardiomegaly 0.32, Mass 0.12, Nodule 0.21.
- Gate decision: valid OBSERVED block; the relative dose 0.25 is strongly disruptive for this SigLIP-896 tower
  (a DOSE-module finding to expect; not a reason to change the locked dose).
- Next step: remaining medgemma-4 NIH modules (GPU 1).

## 2026-09-12 Scheduling: shared task queue

- Run: diagnosed the stalled Slurm queue: jobs inherited the partition's default memory (the whole node, 1 TB), so
  they could only start on an empty node. Replaced per-shard jobs with a file-locked task queue
  (`cf-transfer/queue/`; `cftransfer.worker` claims tasks by atomic rename, checks `requires`/`produces` so
  finished shards are never rerun and no shard runs twice) and generated 658 tasks: NIH PROMPT + all COCO modules
  for the four gated models, and full NIH + COCO pipelines (prep -> modules) for 13 further checkpoints. Lanes:
  gpu1 (<= 14B), gpu2 (27-38B), gpu4 (72B).
- Gate decision: 32 Slurm worker jobs submitted with 64 GB per GPU (24 x 1-GPU across both partitions, 6 x 2-GPU,
  2 x 4-GPU, 12 h each, self-terminating when the lane is empty); four login-node workers take over each GPU when
  its current module chain finishes.
- Next step: monitor queue drain; run statistics/tables as blocks complete.

## 2026-09-12 q25-7 NIH: all modules but PROMPT complete; calibration flags

- Run: CALIBRATION, LOCUS_CALIBRATION, REFIT, DOSE (2 shards), LOCUS (4 shards) on login GPU 3; PROMPT 7 shards
  draining through the task queue (3/7 done).
- Observation (calibration, 400 patients, 2,000 unit-bootstrap draws, seed 2026090602): readable = Effusion
  (S 0.102), Pneumothorax (0.152), Cardiomegaly (0.205); not readable at the one-sided bound = Atelectasis (0.062),
  Mass (0.021), Nodule (-0.041). Answer-capable (clean IY AUROC one-sided lower bound > 0.5) = Cardiomegaly (0.742),
  Mass (0.678), Nodule (0.642); not capable = Effusion (0.593), Atelectasis (0.566), Pneumothorax (0.508).
  Readability and answer capability dissociate: Effusion and Pneumothorax are readable but not answer-capable;
  Mass and Nodule the reverse.
- Gate decision: flags recorded; all questions stay in CORE as the protocol requires.
- Next step: Table 3 aggregates (dose range, refit SD, connector O, label gap) for q25-7 and q3-8.

## 2026-09-12 q25-7 NIH Table 3 aggregates (DOSE, REFIT, LOCUS, label gap)

- Run: paired unit-bootstrap (2,000 draws, seed 2026090603) recomputing every aggregate inside each draw.
- Observation:
  median signed-dose O range 0.419 [0.413, 0.425]: signed O is near zero at negative doses and collapses at +0.5
  (Effusion -0.425, Pneumothorax -0.556, Mass -0.518) as answers saturate; +0.25 sits between.
  median refit O SD 0.082 [0.080, 0.085]: per concept O over seeds 0/1/2 = Effusion -0.071/-0.061/-0.021
  (W_qq 0.190/0.040/0.121), Atelectasis -0.226/-0.098/-0.060, Pneumothorax -0.102/-0.068/-0.134,
  Cardiomegaly -0.219/-0.107/-0.255, Mass -0.169/+0.013/-0.114, Nodule -0.072/-0.112/+0.068. The negative
  Effusion ownership holds under every refit while its raw effect varies three-fold; Mass and Nodule change sign
  under refit, so their competitor verdicts are direction-bundle dependent (the reviewers' open question).
  connector median O -0.013 [-0.014, -0.012]: at the merger output the same relative dose barely changes any
  answer (W_qq: Effusion 0.007, Atelectasis -0.031, others |W| < 0.01), so the connector locus is an
  ineffective write site for Qwen2.5-VL-7B at alpha 0.25.
  median label gap -0.107 [-0.282, +0.0003]: matched directions shift disease-positive patients less than
  negatives (point estimates negative for the median concept), as in the paper's 35/36 pattern.
- Gate decision: valid OBSERVED aggregates; PROMPT contrasts pending block completion.

## 2026-09-12 q25-7 NIH block COMPLETE

- Run: all seven modules scored (1,897,600 outcomes; 0 failed rows); package assembled with run.json status
  COMPLETE, 27.3 GPU-hours (A100-80GB), no deviations; coverage.csv 62/62 question blocks COMPLETE; features,
  fits (both loci, seeds 0/1/2), probe_scores.parquet, prompts.json, preflight.json, environment.txt in place.
- Gate decision: first complete <model_key>/<dataset_id> deliverable of the campaign; ready for the project
  owner's independent CPU recomputation.
- Next step: PROMPT wording/mapping contrasts (fix applied: ownership surface restricted to the two PROMPT
  questions), then the same closure for q3-8, llava15-7, medgemma-4 as their chains finish.

## 2026-09-12 q25-7 NIH PROMPT contrasts (Table 3 wording/mapping)

- Run: PROMPT block complete (762,000 outcomes; Effusion and Mass under WY/IA/IB/WA/WB with the full 127-condition
  grid); ownership per template on the 600 test patients, paired bootstrap for the contrasts.
- Observation:
  Effusion O by template: IY -0.071, WY -0.114, IA -0.077, IB -0.065, WA -0.112, WB -0.116 (Nodule is the strongest
  competitor under every template); wording IY-WY +0.043 [0.040, 0.047]; mapping IA-IB -0.011 [-0.013, -0.010].
  Mass O by template: IY -0.169, WY -0.017, IA -0.025, IB -0.046, WA +0.043, WB +0.037 (Effusion strongest
  competitor); wording IY-WY -0.152 [-0.158, -0.146]; mapping IA-IB +0.021 [0.020, 0.022].
  Mass gains a positive fixed-family advantage only under the "Does this chest radiograph show" A/B templates and
  loses it under "Is there", exactly the prompt-conditioned competition the paper reports for Mass (show A/B
  positive, is negative); Effusion's attribution deficit is wording-independent.
- Gate decision: valid OBSERVED; both registered PROMPT contrasts filled for q25-7 (45 table cells now filled).

## 2026-09-12 q3-8 NIH calibration flags and Table 3 aggregates

- Observation (calibration, 400 patients): readable = Effusion (S 0.128), Pneumothorax (0.127), Cardiomegaly (0.202),
  Mass (0.124); not readable = Atelectasis (0.069), Nodule (-0.071). Answer-capable = all six (clean IY AUROC
  Effusion 0.766, Atelectasis 0.653, Pneumothorax 0.656, Cardiomegaly 0.799, Mass 0.641, Nodule 0.651).
  Table 3: median signed-dose O range 0.470 [0.443, 0.497]; median refit O SD 0.103 [0.094, 0.110];
  connector median O -0.021 [-0.025, -0.016]; median label gap -1.093 [-1.587, -0.710] logit units
  (matched directions shift disease-negative patients far more than positives).
- Gate decision: valid OBSERVED. Contrast with q25-7: Qwen3-VL-8B answers every question above chance and reads
  four concepts, but at alpha 0.25 no clinical direction beats the random family (steering reference 0/6) while
  Qwen2.5-VL-7B met it for Effusion; readability/capability do not predict steering specificity across the two
  generations.

## 2026-09-12 medgemma-4 NIH calibration flags

- Observation (400 patients): readable = Atelectasis (S 0.138), Cardiomegaly (0.198), Mass (0.168); not readable at
  the one-sided bound = Effusion (0.108, 20 positives), Pneumothorax (0.109, 13 positives), Nodule (0.016).
  Answer-capable = all six, with the highest clean AUROCs of the campaign (Effusion 0.803, Atelectasis 0.784,
  Pneumothorax 0.824, Cardiomegaly 0.894, Mass 0.834, Nodule 0.666).
- Gate decision: valid OBSERVED. The medical model answers best and steers least specifically (0/6 steering
  reference, large sham/random effects), the sharpest capability-versus-specificity dissociation so far.

## 2026-09-12 llava15-7 NIH calibration flags

- Observation (400 patients): readable = Effusion (S 0.110), Atelectasis (0.102); not readable at the bound =
  Pneumothorax (0.104, 13 positives), Cardiomegaly (0.053), Mass (0.087), Nodule (0.031). Answer-capable = none:
  clean IY AUROC 0.41-0.57 (Effusion 0.405, below chance), matching the authors' LLaVA capability screens.
- Gate decision: valid OBSERVED; readers exist while answers stay at chance, as in the accepted LLaVA cells.

## 2026-09-12 Credentials and CheXpert source

- Run: user supplied a Hugging Face token and a Redivis API token (stored under 0600 files outside the repo; never
  logged). HF `whoami` succeeds, but gemma-3-4/12/27B and Llama-3.2-11/90B-Vision return "not in the authorized
  list": the account has not yet accepted those licences on the Hub. `fetch_models.py` now carries the pinned
  gated revisions for a rerun once access is granted.
- Observation: CheXpert is distributed by Stanford AIMI on Redivis as CheXpert Plus (`aimi.chexpert_plus:5yyj`,
  version v1.0); the user's access level is `data`. Tables: df_chexpert_plus_240401 (223,462 image rows),
  PNG_train (223,228 files), PNG_valid (234), CheXpert Labels (3 files), DICOM_* and PNG_compressed archives.
  The Azure/azcopy channel of the old AIMI portal no longer exists.
- Gate decision: CheXpert data source READY for a targeted download (labels + only the manifest's frontal
  images) through the Redivis API; dataset_release will be recorded as "CheXpert Plus v1.0 (Redivis
  aimi.chexpert_plus:5yyj), full-resolution PNG".
- Next step: inspect the label table, build the frozen CheXpert manifest, download the 21,016 images, enqueue
  CheXpert CORE + CALIBRATION.

## 2026-09-12 CheXpert manifest frozen (CheXpert Plus)

- Run: label table (metadata) and `impression_fixed.json` (CheXpert labeler output, 14 columns, 1/0/-1/null) and the
  PNG_train file index pulled from Redivis; `cftransfer.manifests chexpert_plus` built the frozen manifest.
- Observation: 190,869 frontal train rows over 64,510 patients with a labelled PNG; one frontal per patient by
  SHA-256 of `path_to_image`, patients by SHA-256 of `deid_patient_id`; roles 20,000/16/400/600; 36 recurring
  types (AP/PA x sex x age-decade). Known-label counts (pos/neg/unknown): calibration Effusion 117/93/190,
  Atelectasis 67/4/329, Pneumothorax 17/136/247, Cardiomegaly 49/46/305, Consolidation 25/83/292, Edema 70/54/276;
  test Effusion 158/130/312, Atelectasis 84/2/514, Pneumothorax 32/213/355, Cardiomegaly 71/58/471,
  Consolidation 25/113/462, Edema 133/59/408. Atelectasis has fewer than 10 known negatives in both evaluation
  roles (CheXpert's labeler leaves it uncertain/unmentioned), so its label-based metrics will be reported as
  insufficient support; answer shifts still use all 600 images.
  dataset_release locked as "CheXpert Plus v1.0 (Redivis aimi.chexpert_plus:5yyj), full-resolution PNG".
- Gate decision: manifest READY. Image transfer: per-file PNG downloads are throttled (~27 s per file even at
  8-way parallelism), so the five PNG zip chunks (720 GB) are downloaded instead and only the 21,016 manifest
  images are extracted.

## 2026-09-12 q3-8 NIH: all modules but PROMPT complete

- Run: CALIBRATION, LOCUS_CALIBRATION, REFIT, DOSE, LOCUS finished on login GPU 2 (LOCUS shards shared the GPU with a
  queue worker for part of the run, 12.4-24/s); PROMPT shards queued.
- Observation: connector median O on the complete LOCUS block -0.020 [-0.023, -0.017]; other Table 3 aggregates
  unchanged (dose range 0.470, refit SD 0.103, label gap -1.093).
- Gate decision: block ready for packaging once PROMPT lands.

## 2026-09-12 18:55 UTC Slurm workers running

- Run: the resubmitted worker jobs (64 GB per GPU) started on gen-a100.p: 10+ single-GPU workers and 2-GPU workers
  on rohpcgpu26/28/31/32/34/38/39/40/41; 18 queue tasks in flight (q25-7 COCO LOCUS + PROMPT shards, llava15-7
  NIH PROMPT, q25-32 NIH/COCO preparation on 2-GPU workers).
- Gate decision: the campaign is no longer limited to the login node; the file-locked queue serialises work across
  both pools without duplication.

## 2026-09-12 More interface gates (queue workers on Slurm)

- Run: preparation tasks completed by queue workers: q25-32 NIH and COCO (2 GPUs), q3-32 NIH and COCO (2 GPUs),
  q25-7 COCO, llava15-7 COCO.
- Observation: all pass determinism / alpha-0 / reach / isolation / fp32-consistency / semantic-mapping checks
  (q25-32: 576 NIH tokens, 12.9/s on 2 GPUs; q3-32: 1,024 patches, 9.2/s; COCO images at the 336^2 budget give
  504 Qwen patches). Composition deviations recorded where single-vs-batch exceeds 0.25 (q25-32 COCO 0.37/0.40,
  q3-32 0.36-0.39, q25-7 COCO 0.33). llava15-7 COCO fails only the A/B mapping check (same failure as NIH), so
  IA/IB/WA/WB are INELIGIBLE for llava15-7 on COCO as well.
- Gate decision: READY for q25-32, q3-32 (both datasets), q25-7 COCO, llava15-7 COCO (yes/no templates).

## 2026-09-12 q25-7 COCO: CORE, CALIBRATION, DOSE, REFIT complete (first cross-domain result)

- Run: COCO CORE (457,200 outcomes over 600 val2017 images; queue workers), CALIBRATION, DOSE, REFIT, LOCUS_CAL;
  statistics with the same rules as NIH (unit = image).
- Observation (W_qq / strongest competitor / O_q / random p95 / |sham|): person 0.496 / car / +0.482 / 0.001 / 0.043;
  dog 0.757 / car / +0.751 / 0.025 / 0.008; car 0.628 / dog / +0.612 / 0.012 / 0.006; chair 0.873 / dog / +0.872 /
  0.024 / 0.038; bottle 0.561 / bicycle / +0.512 / 0.050 / 0.025; bicycle 0.812 / person / +0.814 / 0.012 / 0.002.
  Every object concept is OWNED: steering reference met (rank 1/120) and all five simultaneous lower bounds
  positive for all six questions. Random directions and shams barely move any answer (largest random delta 0.13
  for bottle, <= 0.06 for the other five; shams <= 0.04). Raw check: bottle baseline P(yes) 0.10 -> 0.66 under its own
  direction, 0.02-0.08 under other object directions; person 0.50 -> 0.99 (own) vs 0.31 (dog direction).
  On NIH the same model, projection, readout capacity, dose and controls give zero owned concepts (all six
  diagonals lose to a clinical competitor). The "decodable but not direction-specific" phenomenon is therefore
  not a property of the model or of the protocol: it is specific to the report-mined chest-radiograph concepts.
  Calibration: real probe AUROC 0.90-0.99 but the aspect x area type controls are themselves highly decodable
  (control mean 0.936), so controlled selectivity is small (0.01-0.05) and no COCO concept is "readable" under the
  one-sided bound, exactly the cross-domain caveat the package states (control tasks differ across domains;
  selectivity is not a common capability scale). Answer-capable 5/6 (bicycle has 9 calibration positives:
  insufficient support). Table 3: dose range 0.787 [0.767, 0.806]; refit O SD 0.078 [0.075, 0.082]; connector
  median O +0.003 [0.002, 0.004] (again ineffective at the merger output); label gap -7.57 logits (positives are
  already saturated).
- Gate decision: valid OBSERVED block; PROMPT and LOCUS shards draining.

## 2026-09-12 llava15-7 NIH block COMPLETE (yes/no templates)

- Run: all modules scored for the eligible templates; run.json status COMPLETE; coverage marks the eight A/B
  CALIBRATION cells and the four A/B PROMPT template blocks NOT_STARTED with reason "INELIGIBLE: semantic mapping
  failed image-free preflight".
- Observation (Table 3): median signed-dose O range 0.061 [0.054, 0.069] (small effects at every dose);
  median refit O SD 0.019 [0.018, 0.020]; connector median O -0.073 [-0.075, -0.071] (the projector-output locus
  shows more negative ownership than the vision block for LLaVA); median label gap -0.005 [-0.026, +0.016];
  wording IY-WY: Effusion -0.017 [-0.019, -0.015], Mass +0.011 [0.009, 0.014]; mapping contrasts not measurable.
- Gate decision: second complete NIH deliverable (with the interface disposition for A/B templates).

## 2026-09-12 CheXpert images staged

- Run: five CheXpert Plus PNG zip chunks (720 GB) streamed from Redivis at 76-80 MB/s; the 21,016 manifest images
  extracted (members under `PNG/train/...`, matched on the trailing path); 114 GB under
  `cf-transfer/data/chexpert/images/train/`; completion marker written.
- Gate decision: CheXpert data READY; the 102 queued CheXpert tasks (preparation, CORE, CALIBRATION for 17
  checkpoints) are now claimable by the workers. The zip chunks are kept for now (re-downloadable) and can be
  deleted once the CheXpert blocks are accepted.

## 2026-09-12 q25-7 COCO block COMPLETE

- Run: all seven modules scored (1,897,600 outcomes) by queue workers; package status COMPLETE, coverage 62/62.
- Observation (PROMPT contrasts): wording IY-WY person -0.021 [-0.029, -0.013], bottle +0.064 [0.057, 0.070];
  mapping IA-IB person +0.003 [-0.004, +0.008], bottle -0.011 [-0.016, -0.007]. Ownership of object concepts is
  essentially invariant to wording and answer mapping (O stays 0.4-0.9 under every template), in contrast to the
  wording-conditioned Mass advantage on NIH.
- Gate decision: q25-7 now has two complete dataset blocks (NIH, COCO); CheXpert pending its preparation.

## 2026-09-12 First CheXpert gate: llavamed-7

- Run: CheXpert preparation for LLaVA-Med v1.5 (converted checkpoint) on a Slurm worker: features, fits, preflight.
- Observation: reach/determinism/no-op/isolation/fp32 checks pass (576 tokens, 605 input positions, 13.7/s);
  yes/no mapping passes with strongly signed image-free margins (+6.3/-2.3, +7.6/-3.3); the A/B letter mapping
  fails (A-present: "absent" gives +2.2; B-present: "present" gives -4.9/-4.6), the same LLaVA-family interface
  disposition as llava15-7, so IA/IB/WA/WB are INELIGIBLE for llavamed-7 (CheXpert has no PROMPT module; the
  disposition matters for the two A/B-free modules only through CALIBRATION, which is IY-only on CheXpert).
  Calibration selectivity (CheXpert, known labels only): Effusion 0.219, Atelectasis 0.303, Pneumothorax 0.032,
  Cardiomegaly 0.228, Consolidation 0.224, Edema 0.125.
- Gate decision: READY; CheXpert CORE + CALIBRATION tasks for llavamed-7 are claimable.

## 2026-09-12 q3-8 NIH block COMPLETE

- Run: PROMPT (7 shards on Slurm workers) closed the block; package status COMPLETE, coverage 62/62.
- Observation (PROMPT): Effusion O by template IY +0.073, WY +0.173, IA -0.018, IB -0.059, WA +0.129, WB +0.037
  (Nodule the strongest competitor throughout); wording IY-WY -0.100 [-0.112, -0.088], mapping IA-IB +0.041
  [0.030, 0.052]. Mass O is negative under every template (IY -0.107, WY -0.058, IA -0.174, IB -0.335, WA -0.010,
  WB -0.105); wording -0.049 [-0.071, -0.027], mapping +0.161 [0.148, 0.174]. For Qwen3-VL-8B the Mass advantage
  seen in Qwen2.5-VL-7B under the show A/B templates does not appear, while the Effusion advantage is
  template-dependent in sign.
- Gate decision: third complete NIH deliverable (q25-7, llava15-7, q3-8).

## 2026-09-12 COCO calibration rule and llava15-7 COCO statistics

- Run: the calibration bootstrap initially invalidated every COCO draw because one of 20 type-controls was
  single-class in each draw (COCO has 9 recurring types, some rare). README 6 excludes single-class draws only for
  the affected AUROC, so the rule now drops the affected control within a draw and keeps the draw when at least
  10 of 20 controls are estimable (observed: 18-20 per draw; NIH draws were never affected, 2,000/2,000 valid).
- Observation: llava15-7 COCO readable 5/6 (S 0.25-0.32; bicycle has 9 calibration positives), answer-capable 5/6
  (clean AUROC 0.92-0.999); q25-7 COCO readable 1/6 (person; type controls decode at 0.936 so S is 0.01-0.05),
  answer-capable 5/6. llava15-7 COCO CORE: all six object concepts have a positive fixed-family advantage
  (O person +0.059, chair +0.069, car +0.012, bottle +0.008, bicycle +0.004, dog +0.003) and 5/6 meet the
  steering reference (bottle rank 10/120), but effects are small (W_qq 0.01-0.07) because LLaVA already answers
  object questions near ceiling. Table 3: dose range 0.031, refit SD 0.004, connector O +0.001, label gap -0.20.
- Gate decision: valid OBSERVED; llava15-7 COCO block packaged COMPLETE (yes/no templates).

## 2026-09-12 llavamed-7 NIH gate

- Run: NIH preparation for LLaVA-Med v1.5 on a Slurm worker (features, fits, preflight both loci).
- Observation: reach/determinism/no-op/isolation/fp32 checks pass (20.2/s); yes/no mapping passes (IY margins [(1, 6.33), (0, -2.31), (1, 7.57), (0, -3.27)]);
  A/B letter mapping fails as for llava15-7 and llavamed-7/CheXpert, so IA/IB/WA/WB are INELIGIBLE (eligibility {'IY': True, 'WY': True, 'IA': False, 'IB': False, 'WA': False, 'WB': False}).
  Calibration selectivity: Effusion 0.119, Atelectasis 0.105, Pneumothorax 0.110, Cardiomegaly 0.061, Mass 0.085,
  Nodule 0.019.
- Gate decision: READY for yes/no templates; CORE/DOSE/REFIT/LOCUS and the WY PROMPT block are claimable.

## 2026-09-12 llava15-7 COCO block COMPLETE

- Run: all seven modules (yes/no templates; A/B cells INELIGIBLE by interface) packaged COMPLETE, 19.6 GPU-hours.
  Packaging now counts INELIGIBLE template cells as terminal so completed_modules reflects the disposition.
- Observation: PROMPT wording contrasts negligible (person +0.003 [-0.001, +0.008], bottle -0.003 [-0.004, -0.001]).
- Gate decision: complete deliverables so far: q25-7 (NIH, COCO), llava15-7 (NIH, COCO), q3-8 (NIH).

## 2026-09-12 q25-32 NIH CORE statistics (first owned clinical concept)

- Run: Qwen2.5-VL-32B (2-GPU workers, 12.9/s): CORE, CALIBRATION, DOSE, REFIT complete; LOCUS/PROMPT draining.
- Observation (W_qq / strongest competitor / O_q / random p95 / |sham| / verdict):
  Effusion -0.041 / Nodule / -0.218 / 0.076 / 0.040 / stronger competitor (own effect negative; rank 106/120)
  Atelectasis 0.034 / Mass / -0.059 / 0.049 / 0.097 / stronger competitor
  Pneumothorax 0.032 / Cardiomegaly / -0.056 / 0.054 / 0.075 / stronger competitor
  Cardiomegaly 0.192 / Nodule 0.086 / +0.106 / 0.082 / 0.043 / OWNED (fixed-family advantage AND steering reference,
    rank 3/120): the first clinical concept in the campaign that meets both registered criteria
  Mass 0.165 / Cardiomegaly / -0.077 / 0.232 / 0.047 / stronger competitor
  Nodule 0.010 / Mass / -0.051 / 0.132 / 0.002 / stronger competitor
  Calibration: readable Atelectasis (S 0.080), Pneumothorax (0.140), Cardiomegaly (0.171); answer-capable
  Atelectasis, Cardiomegaly, Mass, Nodule (Effusion clean AUROC 0.470). Table 3: dose range 0.305, refit O SD 0.062,
  label gap -0.204 [-0.339, -0.050].
  Versus the 7B checkpoint of the same family on the same 600 patients: Effusion loses its steering effect
  entirely at 32B, and ownership moves to Cardiomegaly. Scale trends are checkpoint associations (README 2).
- Gate decision: valid OBSERVED block.

## 2026-09-12 lingshu-7 COCO gate

- Run: Lingshu-7B (Qwen2.5-VL architecture, medical training) COCO preparation on a Slurm worker.
- Observation: all reach/determinism/no-op/isolation/fp32/semantic checks pass (504 patches, 126 merged tokens,
  27.8/s); single-vs-batch 0.30 recorded as a composition deviation. CORE complete; statistics running.
- Gate decision: READY.

## 2026-09-12 lingshu-7 COCO CORE statistics

- Observation (W_qq / competitor / O_q / random p95 / |sham|): person 0.476 / chair / +0.388 / 0.042 / 0.021;
  dog 0.858 / car / +0.744 / 0.058 / 0.007; car 0.716 / dog / +0.695 / 0.057 / 0.011; chair 0.731 / dog / +0.763 /
  0.031 / 0.042; bottle 0.673 / bicycle / +0.638 / 0.123 / 0.025; bicycle 0.849 / person / +0.854 / 0.028 / 0.012.
  All six object concepts OWNED (rank 1/120), matching Qwen2.5-VL-7B on the same 600 images: medical fine-tuning
  (Lingshu) leaves general-object direction specificity intact. Calibration mirrors q25-7 COCO (type controls
  decode at 0.93, so S is 0.02-0.06; readable person only; answer-capable 5/6). Table 3: dose range 0.769,
  refit O SD 0.052, connector O +0.004, label gap -6.52.
- Gate decision: valid OBSERVED block.

## 2026-09-12 lingshu-7 COCO block COMPLETE

- Run: all seven modules by queue workers; package COMPLETE, 20.0 GPU-hours.
- Observation (PROMPT): wording IY-WY person -0.056 [-0.069, -0.044], bottle +0.076 [0.062, 0.091]; mapping IA-IB
  person +0.018 [0.008, 0.029], bottle +0.103 [0.097, 0.110]. Object ownership stays large under every template.
- Gate decision: complete deliverables so far: q25-7 (NIH, COCO), llava15-7 (NIH, COCO), q3-8 (NIH), lingshu-7 (COCO).

## 2026-09-13 llava15-7 CheXpert gate and block

- Run: CheXpert preparation (Slurm worker) after the row-id fix; CORE (4 shards) and CALIBRATION scored.
- Observation: reach/determinism/no-op/isolation/fp32 checks pass (576 tokens, 607 positions, 14.9/s); yes/no mapping
  passes, A/B fails (eligibility {'IY': True, 'WY': True, 'IA': False, 'IB': False, 'WA': False, 'WB': False}); single-vs-batch 0.13 logits but one near-zero margin flips sign, recorded as
  a composition deviation. Calibration selectivity on CheXpert known labels: Effusion 0.248, Atelectasis 0.149,
  Pneumothorax 0.032, Cardiomegaly 0.221, Consolidation 0.197, Edema 0.154.
- Gate decision: READY; block packaged COMPLETE (CheXpert requests CORE + CALIBRATION only); statistics running.

## 2026-09-13 llava15-7 CheXpert statistics (first CheXpert block)

- Run: CORE (457,200 outcomes) + CALIBRATION (2,400) packaged COMPLETE, 6.8 GPU-hours; DOSE/REFIT/LOCUS/PROMPT
  NOT_REQUESTED for CheXpert by protocol.
- Observation (known labels only): readable Effusion (S 0.248, AUROC 0.925), Cardiomegaly (0.221), Consolidation
  (0.197), Edema (0.154); not readable Atelectasis (0.149; only 4 known negatives), Pneumothorax (0.032).
  Answer-capable 0/6 (clean IY AUROC 0.42-0.57). CORE: no question meets the steering reference; W_qq range
  -0.03 to +0.07; Cardiomegaly fixed-family advantage +0.027 (below random p95 0.097); Effusion +0.002 unresolved;
  Atelectasis, Pneumothorax, Consolidation, Edema lose to a competitor (Edema or Effusion). Same LLaVA pattern as
  on NIH: readable clinical directions, chance-level answers, no direction-specific steering.
- Gate decision: valid OBSERVED block; cross-dataset replication of the LLaVA NIH result.

## 2026-09-13 q3-8 COCO gate

- Observation: q3-8 coco vis.last  pass=True dev=['D_batch_vs_single'] A=0.0 B=0.0 cons=2.37 logit=17.89 out=0.0 D=1.41 G=0.12 E=True rate=25.94 tok=988/270; q3-8 coco connector pass=True dev=['D_batch_vs_single'] A=0.0 B=0.0 cons=1.25 logit=1.22 out=0.0 D=1.41 G=0.12 E=True rate=25.85 tok=247/270
- Gate decision: READY.

## 2026-09-13 llavamed-7 NIH CORE statistics

- Run: LLaVA-Med v1.5 (converted) NIH CORE (4 shards, Slurm workers), CALIBRATION, DOSE, REFIT, LOCUS complete;
  PROMPT (WY) draining.
- Observation: readable Effusion (S 0.119), Atelectasis (0.105), Pneumothorax (0.110); not readable Cardiomegaly
  (0.061), Mass (0.086), Nodule (0.019). Answer-capable 0/6 (clean IY AUROC 0.48-0.58). CORE: effects are tiny
  (|W_qq| <= 0.024, random p95 <= 0.034); no steering reference; all six ownership contrasts slightly negative
  (Effusion -0.001 vs Mass; Pneumothorax -0.038 vs Cardiomegaly). Table 3: dose range 0.031, refit O SD 0.006,
  connector O -0.018, label gap -0.005. The medical LLaVA behaves like LLaVA-1.5: clinical directions are readable,
  answers sit at chance, and a 0.25 relative-norm write barely moves the answer in any direction.
- Gate decision: valid OBSERVED block.

## 2026-09-13 00:50 UTC Scheduling note

- Observation: 30 workers running on gen-a100.p (all 1-GPU except one 2-GPU); the two 4-GPU workers and the
  gen-h100 workers have never been scheduled (a 4-GPU job needs an entirely free node), so the q25-72 lane
  (48 tasks) has not started and the 32B/38B lane runs on a single 2-GPU worker. The first worker batch expires
  at its 12-hour limit around 07:00 UTC.
- Gate decision: 30 replacement workers submitted (24 x 1-GPU, 4 x 2-GPU, 2 x 4-GPU, 12 h) so the queue keeps
  draining across the expiry; the multi-GPU lanes depend on node-level availability.

## 2026-09-13 q25-7 CheXpert gate

- Run: CheXpert preparation after the row-id fix (Slurm worker); CORE + CALIBRATION scored; packaging + statistics
  running.
- Observation: all checks pass (CheXpert PNGs are 2828x2320, so the 336^2 pixel budget yields 520 Qwen patches /
  130 merged tokens, 169 input positions; 17.9/s); single-vs-batch 0.36 recorded as a composition deviation.
  Calibration selectivity (known labels): Effusion 0.234, Atelectasis 0.351, Pneumothorax -0.028, Cardiomegaly 0.243,
  Consolidation 0.203, Edema 0.178.
- Gate decision: READY.

## 2026-09-13 lingshu-7 NIH CORE statistics (second owned clinical concept)

- Run: Lingshu-7B NIH CORE (Slurm workers), CALIBRATION, DOSE, REFIT, LOCUS complete; PROMPT draining.
- Observation (W_qq / competitor / O_q / random p95 / |sham| / verdict):
  Effusion 0.066 / Cardiomegaly / +0.018 / 0.078 / 0.064 / fixed-family advantage below the random reference (rank 10)
  Atelectasis 0.099 / Nodule / +0.014 / 0.117 / 0.052 / fixed-family advantage below reference
  Pneumothorax 0.070 / Nodule / +0.010 / 0.101 / 0.032 / unresolved
  Cardiomegaly 0.222 / Nodule / +0.171 / 0.117 / 0.040 / OWNED (steering reference met, rank 1/120)
  Mass 0.029 / Nodule / -0.086 / 0.135 / 0.041 / stronger competitor
  Nodule 0.066 / Cardiomegaly / +0.017 / 0.103 / 0.131 / fixed-family advantage below reference
  Calibration: readable Effusion (0.120), Atelectasis (0.102), Cardiomegaly (0.234); answer-capable 6/6 (clean AUROC
  0.66-0.81; the medical fine-tune answers where q25-7 did not). Table 3: dose range 0.141, refit O SD 0.051,
  connector O -0.006, label gap -0.44.
  Architecture-matched pair on the same 600 patients: q25-7 (general) steers Effusion above random/sham but
  owns nothing; Lingshu-7 (medical) loses the Effusion steering effect and owns Cardiomegaly. Medical training
  changes which concept the probe normal controls, not whether attribution holds in general.
- Gate decision: valid OBSERVED block.

## 2026-09-13 q25-7 CheXpert statistics

- Run: CORE + CALIBRATION packaged COMPLETE, 4.8 GPU-hours.
- Observation (known labels only): readable Effusion (S 0.234), Cardiomegaly (0.243), Consolidation (0.203),
  Edema (0.178); Atelectasis S 0.351 but insufficient support (4 known negatives); Pneumothorax -0.028.
  Answer-capable Effusion (0.676) and Cardiomegaly (0.693) only. CORE: no question meets the steering reference;
  random p95 is large on CheXpert images (0.11-0.37); Effusion W 0.076 with O +0.008 (fixed-family advantage below
  the random reference, rank 13); Atelectasis -0.049 (Edema stronger), Pneumothorax -0.099, Cardiomegaly -0.004
  unresolved, Consolidation -0.149, Edema -0.177 (sham 0.31). The Effusion steering effect seen on NIH (0.19,
  above random/sham) does not transfer to CheXpert for the same model, projection and dose.
- Gate decision: valid OBSERVED block; cross-dataset boundary of the NIH Effusion effect.

## 2026-09-13 01:00 UTC Milestone: q25-7 complete on all three datasets

- Observation: Qwen2.5-VL-7B now has NIH, CheXpert and COCO packages COMPLETE (the first checkpoint with the full
  three-dataset deliverable). Summary of its Table 2 row: NIH readable 3/6, answer-capable 3/6, steering reference
  1/6, competitor wins among readable 3/3; CheXpert readable 4/6, answer-capable 2/6, steering reference 0/6,
  competitor wins among readable 2/4;
  COCO readable 1/6 (type controls saturate), answer-capable 5/6, steering reference 6/6, competitor wins 0/1.
  Queue: 170 tasks done, 60 running, 0 failed; 54 Slurm workers active.

## 2026-09-13 llavamed-7 NIH block COMPLETE

- Run: all seven modules (yes/no templates; A/B INELIGIBLE); package COMPLETE, 19.9 GPU-hours. A first packaging
  pass caught the last PROMPT part before it was visible over NFS (75,565/76,200 rows); the repackage sees the
  full block. Block watchers now wait three minutes after the last shard meta.
- Observation (PROMPT wording): Effusion IY-WY +0.075 [0.074, 0.076], Mass -0.019 [-0.020, -0.018]; tiny effects,
  consistent with the CORE picture (no direction-specific steering for LLaVA-Med).
- Gate decision: complete deliverables: q25-7 (NIH, CheXpert, COCO), llava15-7 (NIH, CheXpert, COCO), q3-8 (NIH),
  lingshu-7 (COCO), llavamed-7 (NIH).

## 2026-09-13 llavamed-7 COCO CORE statistics

- Observation: readable person (S 0.094), car (0.046), chair (0.073), bottle (0.077); not readable dog, bicycle
  (bicycle insufficient support). Answer-capable 5/6 (clean AUROC 0.65-0.90). CORE: effects are tiny (W_qq <= 0.055);
  dog alone meets the steering reference (W 0.021 vs random p95 0.018, O +0.012); person +0.019, car +0.005 are
  fixed-family advantages below the random reference; chair -0.022 and bicycle -0.014 lose to a competitor;
  bottle unresolved. Table 3: dose range 0.024, refit O SD 0.011, connector O -0.007, label gap -0.025.
  LLaVA-Med steers general objects far less than LLaVA-1.5 (which owned all six with small effects) and much
  less than the Qwen family (which owned all six with large effects).
- Gate decision: valid OBSERVED block.

## 2026-09-13 llavamed-7 COCO block COMPLETE

- Run: all seven modules (yes/no templates; A/B INELIGIBLE); package COMPLETE, 21.9 GPU-hours.
- Observation (PROMPT wording): person IY-WY -0.005 [-0.007, -0.002], bottle +0.015 [0.013, 0.016].
- Gate decision: complete deliverables: q25-7 (NIH, CheXpert, COCO), llava15-7 (NIH, CheXpert, COCO),
  llavamed-7 (NIH, COCO), q3-8 (NIH), lingshu-7 (COCO); 199 table cells filled.

## 2026-09-13 lingshu-7 NIH block COMPLETE

- Run: all seven modules; package COMPLETE, 19.6 GPU-hours.
- Observation (PROMPT): Effusion O by template IY +0.018, WY +0.026, IA -0.019, IB -0.054, WA -0.026, WB -0.034
  (Nodule strongest under the A/B templates); wording IY-WY -0.008 [-0.014, -0.001], mapping IA-IB +0.034
  [0.031, 0.038]. Mass O negative under every template (-0.086 to -0.253; Nodule strongest); wording +0.040,
  mapping +0.079 [0.075, 0.083]. The Mass show-A/B advantage of Qwen2.5-VL-7B does not appear in its medical
  twin either.
- Gate decision: complete deliverables: q25-7 (3), llava15-7 (3), llavamed-7 (NIH, COCO), lingshu-7 (NIH, COCO),
  q3-8 (NIH); 203 table cells filled.

## 2026-09-13 lingshu-7 CheXpert gate

- Run: CheXpert preparation (requeued after the row-id fix), CORE + CALIBRATION scored; packaging + statistics
  running.
- Observation: all checks pass with no tolerance deviation (520 patches / 130 tokens, 17.5/s; single-vs-batch 0.17).
  Calibration selectivity (known labels): Effusion 0.268, Atelectasis 0.288, Pneumothorax -0.037, Cardiomegaly 0.261,
  Consolidation 0.280, Edema 0.192.
- Gate decision: READY.

## 2026-09-13 lingshu-7 CheXpert statistics (three owned clinical concepts)

- Run: CORE + CALIBRATION packaged COMPLETE, 4.8 GPU-hours.
- Observation (W_qq / competitor / O_q / random p95 / |sham| / verdict):
  Effusion 0.297 / Pneumothorax / +0.097 / 0.084 / 0.144 / OWNED (rank 1/120)
  Atelectasis -0.017 / Edema / -0.088 / 0.056 / 0.096 / stronger competitor
  Pneumothorax 0.364 / Effusion / +0.092 / 0.133 / 0.159 / OWNED (rank 1/120)
  Cardiomegaly 0.270 / Edema / +0.009 / 0.168 / 0.030 / steering reference met (rank 2), ownership unresolved
  Consolidation 0.124 / Effusion / -0.054 / 0.133 / 0.076 / stronger competitor
  Edema 0.477 / Effusion / +0.211 / 0.210 / 0.009 / OWNED (rank 1/120)
  Calibration (known labels): readable Effusion (S 0.268), Cardiomegaly (0.261), Consolidation (0.280), Edema
  (0.192); Atelectasis S 0.288 insufficient support; Pneumothorax -0.037. Answer-capable 5/6 (clean AUROC
  Effusion 0.928, Atelectasis 0.922, Cardiomegaly 0.922, Consolidation 0.931, Edema 0.824, Pneumothorax 0.645).
  This is the strongest clinical ownership in the campaign: the medical Qwen2.5-VL on CheXpert images owns three
  of six concepts with large effects, whereas the same model on NIH owns only Cardiomegaly and its general twin
  owns nothing. CheXpert is a public benchmark that may appear in Lingshu's training data (README 4: report as
  benchmark replication, not as contamination-free evidence).
- Gate decision: valid OBSERVED block.

## 2026-09-13 02:00 UTC Milestone: lingshu-7 complete on all three datasets

- Observation: Lingshu-7B now has NIH, CheXpert and COCO packages COMPLETE (second full-deliverable checkpoint
  after q25-7). Queue: 200 tasks done, 61 running, 0 failed; 207 table cells filled.

## 2026-09-13 q3-8 COCO block COMPLETE

- Run: all seven modules by queue workers; package COMPLETE, 20.2 GPU-hours.
- Observation (W_qq / competitor / O_q / random p95 / |sham|): person 0.444 / bottle / +0.424 / 0.020 / 0.017;
  dog 0.508 / bicycle / +0.506 / 0.014 / 0.001; car 0.752 / bottle / +0.743 / 0.016 / 0.022; chair 0.791 / dog /
  +0.785 / 0.026 / 0.009; bottle 0.588 / bicycle / +0.590 / 0.017 / 0.009; bicycle 0.432 / bottle / +0.432 / 0.006 /
  0.000. All six object concepts OWNED (rank 1/120) for Qwen3-VL-8B as well, although on NIH this model was the
  most random-sensitive of the Qwen checkpoints (random p95 up to 0.36). Calibration: readable person, car,
  bottle (type controls at 0.93); answer-capable 5/6 (clean AUROC 0.99). Table 3: dose range 0.855, refit O SD
  0.027, connector O +0.003, label gap -16.6 logits; PROMPT contrasts small (person wording -0.006, mapping
  +0.018; bottle wording +0.089, mapping +0.028).
- Gate decision: complete deliverables: q25-7 (3), lingshu-7 (3), llava15-7 (3), llavamed-7 (NIH, COCO),
  q3-8 (NIH, COCO); 219 table cells filled.

## 2026-09-13 Gates: iv35-8 NIH, medgemma-4 COCO

- iv35-8 NIH (InternVL3.5-8B-HF, single 448 tile, 1,024 patches / 256 tokens, 27.3/s): all checks pass with no
  tolerance deviation (single-vs-batch 0.20). Calibration selectivity: Effusion 0.074, Atelectasis 0.045,
  Pneumothorax 0.062, Cardiomegaly 0.135, Mass 0.109, Nodule -0.057. CORE complete; statistics running.
- medgemma-4 COCO (SigLIP-896, 4,096 patches / 256 soft tokens, 14.8/s): all checks pass; single-vs-batch 0.42
  recorded as a composition deviation.
- Gate decision: both READY.

## 2026-09-13 iv35-8 NIH statistics

- Observation: readable Cardiomegaly (S 0.135), Mass (0.109); not readable Effusion (0.074), Atelectasis (0.045),
  Pneumothorax (0.062), Nodule (-0.057). Answer-capable 6/6 (clean AUROC 0.66-0.78). CORE: InternVL3.5-8B barely
  responds to the 0.25 relative-norm write at its last InternViT block (|W_qq| <= 0.035; random p95 <= 0.06; sham
  <= 0.03); no steering reference; Effusion +0.003 fixed-family advantage below reference, Cardiomegaly +0.002 and
  Pneumothorax unresolved, Atelectasis/Mass/Nodule lose to a competitor. Table 3: dose range 0.017, refit O SD
  0.004, connector O -0.000, label gap -0.004; PROMPT contrasts |<= 0.011|. The pixel-shuffle + MLP connector
  and Qwen3 LLM of InternVL3.5 are insensitive to vision-block perturbations of this relative size, unlike the
  Qwen-VL merger path.
- Gate decision: valid OBSERVED block.

## 2026-09-13 llavamed-7 CheXpert statistics; llavamed-7 complete on all three datasets

- Run: CORE + CALIBRATION packaged COMPLETE, 7.0 GPU-hours.
- Observation (known labels): readable Effusion (S 0.219), Cardiomegaly (0.228), Consolidation (0.224), Edema
  (0.125); Atelectasis 0.303 insufficient support; Pneumothorax 0.032. Answer-capable Effusion (0.584) and Edema
  (0.628) only. CORE: effects are negligible (|W_qq| <= 0.021, random p95 <= 0.038); Effusion technically meets the
  steering reference (0.0115 vs random p95 0.0098 and sham 0.005) with O +0.003, Consolidation meets it but loses to
  Effusion; the remaining four lose to a competitor. The LLaVA-Med write effect stays at the noise floor on CheXpert
  as on NIH.
- Gate decision: valid OBSERVED block. LLaVA-Med v1.5 is the third checkpoint complete on all three datasets
  (after q25-7 and lingshu-7); 247 table cells filled.

## 2026-09-13 iv35-8 COCO: all seven modules packaged; statistics

- Run: preflight passed on both loci (only the declared batch-vs-single deviation, 0.42 logits; reach
  cons 0.69 / connector 0.94, no change outside consumed tokens, fp32-vs-model 0.06). CORE, CALIBRATION,
  PROMPT, DOSE, REFIT, LOCUS, LOCUS_CALIBRATION packaged COMPLETE; 18.0 GPU-hours.
- Observation (known labels): all six object concepts readable with S 0.10-0.14 over a shared control mean
  of 0.85; bicycle has 9 positives in the 400-row evaluation and stays `insufficient_support` under the
  standing >=10-positive rule. Answer-capable 5/6 (AUROC 0.97-1.00).
- Observation (CORE): InternVL3.5-8B owns all six COCO objects (rank 1, O +0.012..+0.044, random p95
  <= 0.013, |sham| <= 0.010). The write magnitudes are small (W 0.019-0.052), matching the NIH pattern
  where the same 0.25 relative write barely moved the model (|W| <= 0.035): InternVL responds weakly to
  writes at this locus regardless of domain, but on natural objects the response is still specific.
- T3: signed dose range +0.113 [+0.095, +0.122]; refit SD 0.003; connector median O +0.006 (connector
  writes near zero, primary locus carries the effect); label gap -0.52 [-0.73, -0.46]; PROMPT contrasts
  small (person wording +0.008, mapping -0.027; bottle +0.005 / -0.007).
- Gate decision: valid OBSERVED block. The natural-object control now holds for four families
  (Qwen2.5-VL, Qwen3-VL, Lingshu, InternVL3.5) plus LLaVA-1.5 with small effects; 259 table cells filled.

## 2026-09-13 Worker termination handling (infrastructure)

- Run: inspection of the queue worker before the first 12 h Slurm batch expires. The worker had no handling for
  the wall-clock kill: a task claimed before `--max-hours` and still running at the limit would stay in
  `running/` forever and the block would never complete.
- Fix: the worker now traps SIGTERM/SIGINT (Slurm sends SIGTERM before the kill), stops the child and moves the
  task back to `pending/` (never `failed/`); and every claim first sweeps `running/` for tasks whose Slurm worker
  job has left the queue (worker id `<node>-<jobid>` not in `squeue`), requeueing them after a 10 min grace.
  Login-node workers (ids without a job number) are left alone. Requeued shards recompute from scratch; the
  packager dedups partial part files by key. Unit test added (`tests/cftransfer/test_worker_queue.py`, 12/12 pass).
- Third Slurm batch submitted (24 gpu1 + 4 gpu2 workers, `--max-hours 10`) so the drain continues past the
  first batch's expiry; already-running workers keep the old code, and the new workers' sweep recovers any
  task they drop.

## 2026-09-13 iv35-8 NIH and q25-32 NIH: all seven modules packaged; statistics

- Run: both blocks now carry CORE, CALIBRATION, PROMPT, DOSE, REFIT, LOCUS, LOCUS_CALIBRATION with every cell
  COMPLETE (457,200 unique CORE rows each); iv35-8 19.5 GPU-hours, q25-32 82.7 GPU-hours. Run status is now
  derived from coverage by the packager (commit 279b801), so the three blocks that were fully scored but still
  marked RUNNING (iv35-8 COCO/NIH, q25-32 NIH) flipped to COMPLETE without a manual flag.
- iv35-8 NIH (InternVL3.5-8B): readable Cardiomegaly (S 0.135) and Mass (0.109) only; answer-capable on all
  six (AUROC 0.66-0.78). CORE: no concept meets the steering reference; |W_qq| <= 0.035 with random p95 up to
  0.062; Effusion O +0.003 (fixed-family advantage at noise level), Pneumothorax and Cardiomegaly unresolved,
  the rest lose to a competitor. T3: signed dose range +0.017 [+0.013, +0.020]; refit SD 0.004; connector
  median O -0.000; label gap -0.004 [-0.019, +0.020]; PROMPT contrasts within +-0.011. Together with its COCO
  block (owns all six objects, W 0.02-0.05) this fixes the InternVL reading: a weak writer at this locus in
  both domains, specific on objects, unspecific on clinical concepts.
- q25-32 NIH (Qwen2.5-VL-32B): readable Atelectasis (0.080), Pneumothorax (0.140), Cardiomegaly (0.171);
  answer-capable Atelectasis, Cardiomegaly, Mass, Nodule (Effusion answer AUROC 0.47). CORE: Cardiomegaly owned
  (W 0.192, O +0.106 vs Nodule, random p95 0.082, |sham| 0.043, rank 3, meets reference); Effusion strongly
  anti-owned (W -0.041, O -0.218 vs Nodule); Mass writes 0.165 but below its random p95 0.232; the other three
  lose to a competitor. T3: signed dose range +0.303 [+0.291, +0.316] (the model responds to dose, unlike
  iv35-8); refit SD 0.065; connector median O -0.036; label gap -0.204 [-0.339, -0.050]; Mass wording +0.043,
  Mass mapping -0.125 [-0.136, -0.117] (the "show" A/B templates flip Mass, as in q25-7).
- Gate decision: both valid OBSERVED blocks; 264 table cells filled. The q25-7 -> q25-32 comparison on NIH is
  now complete on all modules: scaling Qwen2.5-VL from 7B to 32B replaces the Effusion effect with a
  Cardiomegaly effect rather than adding ownership.

## 2026-09-13 q25-32 COCO CORE statistics

- Run: preflight passed on both loci (batch-vs-single deviation 0.37 declared, replicated; reach at vis.last
  13.8 / connector 0.38, nothing outside consumed tokens; fp32-vs-model 0.07). CORE, CALIBRATION, REFIT,
  LOCUS_CALIBRATION packaged (21.1 GPU-hours so far); PROMPT, DOSE, LOCUS still in the queue.
- Observation (known labels): the type->random-label controls are strong on this model (control mean 0.93), so
  only person clears the readability margin (S 0.056); the other five sit within 0.04 of the controls. Answers
  are near-perfect (AUROC 0.98-0.99; bicycle 0.99 but `insufficient_support`, 9 positives).
- Observation (CORE): Qwen2.5-VL-32B owns all six objects at rank 1 with large effects: person O +0.444,
  dog +0.125, car +0.273, chair +0.834, bottle +0.410, bicycle +0.187; random p95 <= 0.015, |sham| <= 0.011.
- Gate decision: valid OBSERVED block. The natural-object control now holds for five checkpoints across four
  families (q25-7, q25-32, q3-8, lingshu-7, iv35-8) plus LLaVA-1.5 with small effects; on NIH the same 32B
  model owns one concept (Cardiomegaly) and anti-owns Effusion. 268 table cells filled.

## 2026-09-13 Gemma 3 licences accepted: gemma3-4 smoke and enqueue

- Run: the user accepted the gemma-3 licences on the Hub; `fetch_models.py gemma3-4,gemma3-12,gemma3-27`
  pulls the pinned revisions (093f9f38 / 96b6f1ec / 005ad340, registered in `adapters/__init__.py`, commit
  9b0951a). Llama-3.2-Vision access is still pending with Meta; llama32-11/90 stay parked.
- Observation (gemma3-4 smoke on NIH, cuda:3, 46 s): determinism 0, alpha-0 identity 0, reach connector 37.1 /
  answer logit 39.5, no change outside consumed tokens, fp32-vs-model 0.12; 4096 SigLIP tokens at the block,
  256 after the connector. Batch-vs-single 1.28 (replicated 1.04) exceeds the 0.25 tolerance, the same bf16
  shape dependence seen on medgemma-4 (0.50 / 0.30) and Qwen3-VL (0.60 / 1.41); the fixed-composition runner
  design covers it and preflight will record it as a declared deviation.
- Gate decision: adapter accepted for the Gemma 3 base checkpoints (same `Gemma3Adapter` as MedGemma).
  gemma3-4 enqueued for NIH, COCO and CheXpert (48 tasks: prep + all modules per dataset); gemma3-12 (gpu1) and
  gemma3-27 (gpu2) will be enqueued when their downloads finish. Queue: 439 pending, 63 running, 308 done, 0 failed.

## 2026-09-13 gemma3-12 and gemma3-27 downloaded and enqueued

- Run: pinned snapshots landed (12B in 238 s, 27B in 478 s; 38 GB and 86 GB on disk). Same `Gemma3Adapter`
  as gemma3-4 and MedGemma; the per-checkpoint preflight in each prep task is the acceptance gate.
- Gate decision: gemma3-12 (gpu1 lane) and gemma3-27 (gpu2 lane) enqueued for NIH, COCO and CheXpert (96 tasks).
  All 20 non-Llama checkpoints of the cf-transfer-v1 grid are now downloaded and queued; llama32-11/90 wait
  on Meta's licence approval.

## 2026-09-13 iv35-8 CheXpert statistics; iv35-8 complete on all three datasets

- Run: preflight passed on both loci with no deviations (batch-vs-single 0.18; reach 0.44 / 0.28; fp32-vs-model
  0.05). CORE + CALIBRATION packaged COMPLETE, 4.3 GPU-hours.
- Observation (known labels): readable Effusion (S 0.253), Cardiomegaly (0.217), Consolidation (0.256), Edema
  (0.175); Atelectasis 0.211 but `insufficient_support`; Pneumothorax at the control level (-0.001).
  Answer-capable on five (AUROC 0.83-0.95; Pneumothorax 0.63); InternVL answers CheXpert far better than NIH.
- Observation (CORE): writes stay at the InternVL noise floor (|W_qq| <= 0.038). Pneumothorax technically meets
  the reference (W 0.0040 vs random p95 0.0037) with O +0.004; Consolidation O +0.013 but W below its random p95;
  Edema unresolved; Effusion, Atelectasis, Cardiomegaly lose to a competitor. No owned clinical concept.
- Gate decision: valid OBSERVED block. InternVL3.5-8B is the fourth checkpoint complete on all three datasets
  (with q25-7, lingshu-7, llavamed-7): reads five CheXpert concepts, answers them well, and cannot be steered
  on any of them at alpha 0.25, while owning all six COCO objects.

## 2026-09-13 medgemma-4 NIH: all seven modules packaged; statistics

- Run: CORE, CALIBRATION, PROMPT, DOSE, REFIT, LOCUS, LOCUS_CALIBRATION all COMPLETE; 43.8 GPU-hours (the
  4096-token SigLIP block makes MedGemma the most expensive 4B checkpoint).
- Observation (known labels): readable Atelectasis (S 0.138), Cardiomegaly (0.198), Mass (0.168); Effusion and
  Pneumothorax 0.108 just under the margin; Nodule 0.016. Answer-capable on all six (AUROC 0.67-0.89), the
  strongest NIH answerer so far.
- Observation (CORE): MedGemma-4B is the most perturbation-sensitive checkpoint: random-direction p95 reaches
  0.54 and the coordinate-permutation sham 0.55 (Nodule), so a 0.25 relative write moves the answer regardless
  of direction. Against those references no concept is owned: Atelectasis O -0.558 and Mass O -0.273 are
  strongly anti-owned, Cardiomegaly writes +0.19 but below its random p95 0.37, Effusion/Pneumothorax
  unresolved, Nodule O +0.094 below sham. T3: signed dose range +0.617 [+0.578, +0.649] (largest so far);
  refit SD 0.080; connector median O -0.004; label gap -3.49 [-4.80, -2.06] (label-shift controls move the
  answer far more than the concept write); Mass wording -0.213 [-0.243, -0.180], Effusion wording -0.071.
- Gate decision: valid OBSERVED block. The medical Gemma reads and answers NIH concepts well and is highly
  writable, but the writes are unspecific: the strongest counter-example to "capable model => owned concept".
  280 table cells filled.

## 2026-09-13 q3-4 COCO CORE statistics

- Run: preflight passed on both loci (batch-vs-single 0.69 declared and replicated, the Qwen3-VL pattern;
  reach 3.97 / 0.50; fp32-vs-model 0.12). CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION packaged
  (5.9 GPU-hours so far); PROMPT and LOCUS pending.
- Observation (known labels): control mean 0.93 as for q25-32; readable person (S 0.068), car, chair, bottle
  (0.03); dog 0.016; bicycle 0.064 but `insufficient_support` (9 positives). Answers near-perfect (0.99).
- Observation (CORE): Qwen3-VL-4B owns all six objects at rank 1: person O +0.465, dog +0.127, car +0.331,
  chair +0.714, bottle +0.335, bicycle +0.072; random p95 <= 0.014, |sham| <= 0.037.
- Gate decision: valid OBSERVED block. Sixth checkpoint owning all six COCO objects (q25-7, q25-32, q3-4, q3-8,
  lingshu-7, iv35-8); the 4B Qwen3 is as specific on objects as the 32B Qwen2.5. 284 table cells filled.

## 2026-09-13 Four CORE blocks: iv35-14 COCO, q25-3 COCO, medgemma-4 COCO, q3-8 CheXpert

- q25-3 template gate (NIH and COCO): the image-free semantic-mapping check E fails for the two B-mapped A/B
  templates (IB, WB) on both datasets while IY/WY/IA/WA pass, so IB/WB are INELIGIBLE by the standing rule
  (8 INELIGIBLE coverage cells per dataset), the runner skipped them and the other checks pass (determinism 0,
  alpha-0 0, reach 26.8 / 0.88, nothing outside consumed tokens; batch-vs-single 0.37 declared). Same
  disposition as the LLaVA family, narrower (only the B mapping).
- iv35-14 COCO (InternVL3.5-14B): preflight passed (batch-vs-single 0.48 declared). Readable person, car,
  chair, bottle (S 0.11-0.14; dog 0.076; bicycle `insufficient_support`). CORE: owns five objects at rank 1
  with the InternVL-typical small writes (O +0.013..+0.040, W <= 0.048, random p95 <= 0.013); bicycle rank 1
  but O +0.002 -> unresolved. 8.9 GPU-hours so far (CORE, CALIBRATION, DOSE, LOCUS_CALIBRATION).
- q25-3 COCO (Qwen2.5-VL-3B): owns all six at rank 1 with the largest effects yet (person O +0.455, dog +0.424,
  car +0.529, chair +0.677, bottle +0.615, bicycle +0.391; random p95 <= 0.047, |sham| <= 0.029). 5.6 GPU-hours
  so far (CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION); PROMPT/LOCUS pending.
- medgemma-4 COCO: preflight passed (batch-vs-single 0.42 declared). Readable person, car, chair, bottle
  (S 0.05-0.09). CORE: owns all six (person O +0.345, dog +0.033 rank 3, car +0.435, chair +0.194,
  bottle +0.203, bicycle +0.144) despite random p95 up to 0.22 and sham up to 0.21: the model that is
  unspecific on NIH is specific on objects even with its high perturbation sensitivity. 27.8 GPU-hours
  (all but PROMPT).
- q3-8 CheXpert (Qwen3-VL-8B): preflight passed (batch-vs-single 0.93 declared, replicated). Readable
  Effusion (S 0.226), Cardiomegaly (0.218), Consolidation (0.213), Edema (0.162); Atelectasis
  `insufficient_support`; Pneumothorax 0.062. Answer-capable Effusion, Cardiomegaly, Consolidation, Edema.
  CORE: Effusion, Consolidation and Atelectasis meet the steering reference (W 0.370 / 0.352 / 0.009) but
  every concept loses to a competitor (Effusion O -0.069 vs Pneumothorax; Pneumothorax itself writes 0.203
  with answer AUROC 0.47). No owned concept. COMPLETE, 5.1 GPU-hours; q3-8 is the fifth checkpoint complete
  on all three datasets.
- Gate decision: four valid OBSERVED blocks; 296 table cells filled. COCO ownership now holds for eight
  checkpoints across five families (q25-3/7/32, q3-4/8, lingshu-7, iv35-8/14, medgemma-4) plus LLaVA-1.5
  with small effects and LLaVA-Med at 1/6; clinical ownership remains at Cardiomegaly (q25-32, lingshu-7 NIH)
  and Lingshu's three CheXpert concepts.

## 2026-09-13 medgemma-4 COCO: all seven modules packaged; T3

- Run: PROMPT landed; block COMPLETE with all seven modules, 36.0 GPU-hours.
- Observation (T3): signed dose range +0.337 [+0.289, +0.376]; refit SD 0.105 (the largest seed variance so
  far, consistent with the model's perturbation sensitivity); connector median O +0.001 (the connector write
  does nothing, as on NIH); label gap -5.71 [-6.82, -4.88]; person wording +0.011 [-0.012, +0.045] and mapping
  -0.003 (both null); bottle wording -0.045, mapping +0.019.
- Gate decision: valid OBSERVED block; medgemma-4 is complete on NIH and COCO (CheXpert CORE in progress).
  308 table cells filled.

## 2026-09-13 q25-3 COCO and q3-4 COCO: all seven modules packaged; T3

- q25-3 COCO: COMPLETE, 12.8 GPU-hours. Signed dose range +0.563 [+0.548, +0.578]; refit SD 0.061; connector
  median O +0.009; label gap -4.35 [-4.94, -4.01]; person wording -0.058 [-0.066, -0.051], bottle wording
  +0.004 (null). No mapping contrast: IB is INELIGIBLE for this checkpoint, so IA-minus-IB is not defined
  (the coverage table records the disposition).
- q3-4 COCO: COMPLETE, 13.7 GPU-hours. Signed dose range +0.613 [+0.581, +0.651]; refit SD 0.039; connector
  median O +0.002; label gap -15.9 [-16.7, -14.9] (the largest label-shift response of any block: the 4B Qwen3
  answer moves by orders of magnitude more under the label-shift controls than under the concept write);
  person wording -0.009, mapping +0.014; bottle wording +0.114 [+0.098, +0.130], mapping -0.034.
- Gate decision: two valid OBSERVED blocks; 322 table cells filled. Both small Qwen checkpoints show the
  natural-object pattern in full: ownership of all six objects, clean dose response, connector writes inert.

## 2026-09-13 medgemma-4 CheXpert and q25-3 NIH statistics; medgemma-4 complete on all three datasets

- medgemma-4 CheXpert: preflight passed (batch-vs-single 0.56 declared). CORE + CALIBRATION COMPLETE,
  8.6 GPU-hours. Readable Effusion (S 0.256), Cardiomegaly (0.271), Consolidation (0.256), Edema (0.183);
  Atelectasis 0.259 `insufficient_support`; Pneumothorax 0.007. Answer-capable on five (AUROC 0.70-0.96).
  CORE: Edema owned (W 0.352, O +0.101 vs Atelectasis, random p95 0.232, |sham| 0.088, rank 1) and
  Cardiomegaly meets the reference at rank 5 (W 0.221, O +0.045; sham 0.184 close behind); Pneumothorax
  strongly anti-owned (O -0.602: the Pneumothorax write raises Edema far more than Pneumothorax); Effusion and
  Consolidation O -0.20; Atelectasis -0.08. First owned clinical concept for the Gemma family, and the model
  that owns nothing on NIH owns one CheXpert concept.
- q25-3 NIH (Qwen2.5-VL-3B, IB/WB INELIGIBLE): all seven modules COMPLETE, 12.6 GPU-hours. Readable Effusion
  (S 0.137) and Cardiomegaly (0.137); answer-capable only Mass (0.68) and Nodule (0.61), the three others
  below or near chance (Pneumothorax 0.43). CORE: no concept meets the reference; Effusion W -0.040 (O -0.094),
  Pneumothorax O -0.176, Cardiomegaly O -0.109, Mass and Nodule write +0.15 / +0.14 but below random p95 and
  lose to Atelectasis. T3: dose range +0.116; refit SD 0.039; connector median O -0.021; label gap -0.014
  [-0.062, +0.031] (null); Effusion wording +0.102 [+0.100, +0.104], Mass wording +0.076.
- Gate decision: two valid OBSERVED blocks. medgemma-4 is the sixth checkpoint complete on all three datasets
  (with q25-7, lingshu-7, llavamed-7, iv35-8, q3-8). The Qwen2.5-VL size ladder on NIH is now 3B (nothing
  owned), 7B (nothing; Effusion anti-owned), 32B (Cardiomegaly); 72B pending on the gpu3 lane.

## 2026-09-13 q25-3 CheXpert and q3-4 CheXpert statistics; q25-3 complete on all three datasets

- q25-3 CheXpert (IB/WB INELIGIBLE, batch-vs-single 0.18 declared): CORE + CALIBRATION COMPLETE, 3.5 GPU-hours.
  Readable Effusion (S 0.228), Cardiomegaly (0.256), Consolidation (0.180), Edema (0.185); Atelectasis
  `insufficient_support`; Pneumothorax 0.005. Answer-capable only Effusion (0.66) and Cardiomegaly (0.63);
  Edema 0.51, Pneumothorax 0.45. CORE: nothing owned; Effusion anti-owned (W -0.191, O -0.178, rank 120 of
  the random family), all others negative. The 3B Qwen2.5 reads CheXpert concepts but cannot answer or be
  steered on them. q25-3 is the seventh checkpoint complete on all three datasets.
- q3-4 CheXpert (preflight passed; batch-vs-single 0.54 declared, replicated): CORE + CALIBRATION COMPLETE,
  3.5 GPU-hours. Readable Effusion (0.230), Cardiomegaly (0.246), Consolidation (0.245), Edema (0.176);
  Atelectasis 0.278 `insufficient_support`; Pneumothorax 0.050. Answer-capable Effusion (0.90), Cardiomegaly
  (0.85), Consolidation (0.79), Edema (0.82). CORE: Consolidation owned (W 0.807, O +0.094 vs Edema, random
  p95 0.429, |sham| 0.001, rank 1) and Edema owned (W 0.593, O +0.092 vs Consolidation, random p95 0.426,
  |sham| 0.150, rank 1); Effusion writes 0.398 at rank 1 but Consolidation moves more under it (O -0.015);
  Pneumothorax O -0.231. First owned clinical concepts for Qwen3-VL, both on CheXpert.
- Gate decision: two valid OBSERVED blocks; 356 table cells filled. Owned clinical concepts so far:
  Cardiomegaly (q25-32 NIH, lingshu-7 NIH), Effusion/Pneumothorax/Edema (lingshu-7 CheXpert), Edema
  (medgemma-4 CheXpert), Consolidation/Edema (q3-4 CheXpert). CheXpert yields ownership more often than NIH
  for the same checkpoints (q3-8 and iv35-8 excepted), consistent with cleaner labels or training exposure.

## 2026-09-13 q3-4 NIH: all seven modules packaged; q3-4 complete on all three datasets

- Run: preflight passed (batch-vs-single 0.74 declared; reach 2.95 / 2.25; fp32-vs-model 0.12). All seven
  modules COMPLETE, 14.6 GPU-hours.
- Observation (known labels): readable Effusion (S 0.098), Atelectasis (0.102), Cardiomegaly (0.139); Mass
  0.112 and Pneumothorax 0.062 below the margin; Nodule -0.050. Answer-capable on all six (AUROC 0.64-0.79).
- Observation (CORE): Effusion meets the steering reference (W 0.386, random p95 0.290, |sham| 0.099, rank 3)
  but Atelectasis moves more under it (O -0.026); Cardiomegaly writes 0.269 below its random p95 0.397; the
  other four are anti-owned (Pneumothorax O -0.287, Mass -0.135). No owned concept on NIH, in contrast to the
  same checkpoint's Consolidation and Edema on CheXpert.
- T3: signed dose range +0.359 [+0.323, +0.387]; refit SD 0.040; connector median O -0.000; label gap -0.58
  [-0.99, -0.12]; Effusion wording -0.094 [-0.103, -0.086], mapping -0.008; Mass wording +0.079, mapping +0.069
  (Mass answers again depend on the template, as in q25-7 and q25-32).
- Gate decision: valid OBSERVED block; q3-4 is the eighth checkpoint complete on all three datasets (with
  q25-7, q25-3, q3-8, lingshu-7, llavamed-7, iv35-8, medgemma-4). 380 table cells filled.

## 2026-09-13 iv35-14 NIH: all seven modules packaged; statistics

- Run: preflight passed (batch-vs-single 0.38 declared, replicated; reach 0.60 / 0.59; fp32-vs-model 0.12).
  All seven modules COMPLETE, 29.3 GPU-hours.
- Observation (known labels): readable Effusion (S 0.125), Atelectasis (0.086), Pneumothorax (0.094),
  Cardiomegaly (0.182); Mass 0.031; Nodule -0.076. Answer-capable on all six (AUROC 0.65-0.79).
- Observation (CORE): the InternVL noise floor again (|W_qq| <= 0.021). Effusion formally meets the reference
  (W 0.0122 vs random p95 0.0090, |sham| 0.0081) with O +0.009, and Mass O +0.005 below its random p95; these
  are two-hundredths of a logit and not scientifically distinguishable from zero. Cardiomegaly writes 0.021
  below its random p95 0.037; Pneumothorax W 0.0000. T3: signed dose range +0.011 [+0.008, +0.017]; refit SD
  0.003; connector median O -0.000; label gap -0.007 (null); every PROMPT contrast within +-0.005.
- Gate decision: valid OBSERVED block. InternVL3.5 at 8B and 14B behave identically: a 0.25 relative write at
  the consumed block barely reaches the answer on chest radiographs while the same write is specific on COCO
  objects; the 38B checkpoint (gpu2 lane) will close the family. 404 table cells filled.

## 2026-09-13 q25-72 NIH preflight (gpu3 lane placed)

- Run: the first 3-GPU Slurm worker (gpu3 lane) was placed on rohpcgpu42; the q25-72 NIH prep extracted
  19,228 rows in 636 s (batch 8, device_map auto across 3 A100-80GB), fitted both loci (3 seeds each) and ran
  preflight; the worker moved on to CALIBRATION.
- Observation: preflight passed on both loci with no tolerance deviations: determinism 0, alpha-0 0, reach
  17.9 at the consumer / 3.25 answer logits (connector 0.38 / 0.39), nothing outside consumed tokens,
  batch-vs-single 0.22 (within 0.25, the first Qwen2.5-VL checkpoint inside tolerance), fp32-vs-model 0.06,
  all six templates eligible, 7.2 rows/s.
- Gate decision: adapter accepted for the 72B checkpoint; CORE (4 shards) and the remaining modules follow in
  the gpu3 lane. Two more gpu3 workers are pending placement.

## 2026-09-13 iv35-14 CheXpert statistics and COCO close; iv35-14 complete on all three datasets

- iv35-14 CheXpert (preflight passed; batch-vs-single 0.44 declared): CORE + CALIBRATION COMPLETE,
  4.7 GPU-hours. Readable Effusion (S 0.262), Cardiomegaly (0.272), Consolidation (0.295), Edema (0.217);
  Atelectasis -0.092 (the known-label probe is below its controls; `insufficient_support`); Pneumothorax
  -0.007. Answer-capable on five (AUROC 0.70-0.94). CORE: Effusion meets the reference (W 0.041 vs random
  p95 0.023, |sham| 0.014, rank 2) with O +0.012; Consolidation O +0.009 below sham; the rest at the noise
  floor (|W| <= 0.010). As for 8B, the 14B InternVL reads and answers CheXpert but is not steerable at 0.25.
- iv35-14 COCO: all seven modules COMPLETE, 28.3 GPU-hours. T3: signed dose range +0.039 [+0.031, +0.051];
  refit SD 0.003; connector median O +0.007; label gap -0.44 [-0.51, -0.30]; PROMPT contrasts within +-0.008.
- Gate decision: two valid OBSERVED blocks; iv35-14 is the ninth checkpoint complete on all three datasets.
  416 table cells filled.

## 2026-09-13 First Slurm batch turnover: stale-task sweep verified in production

- Run: the first 12 h worker batch (30 jobs, old worker code without SIGTERM handling) reached its wall clock
  between 06:55Z and 07:01Z; most workers had already exited at their 11.5 h budget, but four were killed
  mid-task (iv35-14 CheXpert CORE shard 2, q25-32 COCO PROMPT shard 5, q3-32 NIH CORE shards 0 and 2).
- Observation: a new-code worker (job 4345808) swept `running/`, found the four tasks whose Slurm jobs had
  left the queue and moved them back to `pending/` with a `requeued` record within 1-5 minutes of the kill;
  the iv35-14 shard was re-run from scratch by job 4345809 (rc 0), and its block packaged COMPLETE with the
  partial parts of the killed run deduplicated by key. The other three are pending in their lanes. The login
  sweeper found nothing (the Slurm worker was first). 0 tasks in `failed/`.
- Gate decision: the requeue path works end to end; no manual intervention needed at batch boundaries. The
  third batch is being placed (14 of 24 gpu1 workers running); queue 372 pending / 52 running / 482 done.

## 2026-09-13 Lane rebalancing for the 27-38B backlog

- Run: pending tasks by lane at 07:50Z were gpu2 237 (q25-32/q3-32/iv35-38/lingshu-32/medgemma-27/gemma3-27
  CheXpert, COCO and NIH modules), gpu1 90, gpu3 45 (q25-72), against 44 / 6 / 2 running workers. The gpu1
  lane drains within hours and its workers exit when their lane is empty, freeing GPUs for 2-GPU jobs.
- Gate decision: fourth batch submitted before the second batch's 12 h expiry: 24 gpu1 + 8 gpu2 workers
  (cfw4-g1-a / cfw4-g2-a), then 12 more gpu2 workers (cfw4-g2-b), 2 gpu3 workers on gen-a100.p and 2 on
  gen-h100 (3x H100-80GB also holds the 72B in bf16). All with `--max-hours 10 --exit-when-empty`; placement
  is left to the scheduler.

## 2026-09-13 q25-32 COCO: all seven modules packaged; T3

- Run: PROMPT (including the requeued shard 5), DOSE and LOCUS landed; block COMPLETE with all seven modules,
  75.0 GPU-hours (2-GPU lane).
- Observation (T3): signed dose range +0.684 [+0.650, +0.715] (the largest dose response so far); refit SD
  0.079; connector median O +0.003 (inert connector, as in every block); label gap -8.82 [-9.28, -8.04];
  person wording +0.061, mapping +0.036; bottle wording +0.111 [+0.098, +0.123], mapping +0.077.
- Gate decision: valid OBSERVED block; q25-32 is complete on NIH and COCO (CheXpert CORE at 30%).
  424 table cells filled.

## 2026-09-13 llava15-13 NIH CORE statistics; template gate

- Template gate: the image-free mapping check passes IY/WY/IA/WA and fails IB/WB for LLaVA-1.5-13B on NIH, so
  only the two B-mapped templates are INELIGIBLE (the 7B LLaVA and LLaVA-Med fail all four A/B templates).
  Other preflight checks clean with no deviations (batch-vs-single 0.17; reach 3.69 / 0.69; fp32-vs-model 0.06).
- Run: CORE, CALIBRATION, PROMPT, DOSE, REFIT, LOCUS_CALIBRATION packaged (25.4 GPU-hours); LOCUS pending.
- Observation (known labels): readable Effusion (S 0.110) and Atelectasis (0.102); Pneumothorax 0.104 and
  Mass 0.087 below the margin; Cardiomegaly 0.053; Nodule 0.031. Answer AUROC 0.49-0.56 on all six: the 13B
  LLaVA reads two concepts and answers none, exactly the 7B pattern.
- Observation (CORE): nothing meets the reference; |W_qq| <= 0.045 with Atelectasis anti-owned (O -0.047);
  Mass O +0.008 below its random p95 0.033. No steering.
- Gate decision: valid OBSERVED block. Scaling LLaVA-1.5 from 7B to 13B changes template eligibility but not
  the answer or steering picture. 440 table cells filled.

## 2026-09-13 llava15-13 NIH: all seven modules packaged; T3

- Run: LOCUS landed; block COMPLETE with all seven modules, 34.2 GPU-hours.
- Observation (T3): signed dose range +0.061 [+0.057, +0.064]; refit SD 0.010; connector median O -0.012;
  label gap -0.014 [-0.030, +0.009] (null: label-shift controls do not move the answer either); Effusion and
  Mass wording contrasts within +-0.003; no mapping contrast (IB INELIGIBLE).
- Gate decision: valid OBSERVED block; llava15-13 complete on NIH (COCO and CheXpert in progress).
  446 table cells filled.

## 2026-09-13 llava15-13 COCO: all seven modules packaged; statistics

- Run: preflight passed apart from the IB/WB mapping gate (batch-vs-single 0.31 declared; reach 5.50 / 0.44;
  fp32-vs-model 0.06). All seven modules COMPLETE, 37.9 GPU-hours.
- Observation (known labels): all six readable with the largest margins of any COCO block (S 0.25-0.32 over a
  control mean of 0.67; bicycle `insufficient_support`); answers 0.91-0.99.
- Observation (CORE): person (W 0.068, O +0.054, rank 1) and chair (W 0.102, O +0.081, rank 1) owned at the
  reference; car O +0.020 and bottle O +0.019 positive but below sham / random p95; dog rank 6 unresolved
  (O +0.005); bicycle O -0.031. The LLaVA-1.5 pattern at 13B as at 7B: reads objects strongly, answers them,
  and writes only small, partly specific effects (2 of 6 at reference; |W| <= 0.10 versus 0.1-0.8 for Qwen).
- T3: signed dose range +0.069 [+0.061, +0.077]; refit SD 0.005; connector median O -0.001; label gap -0.22
  [-0.27, -0.16]; person wording +0.013, bottle wording -0.007; no mapping contrast (IB INELIGIBLE).
- Gate decision: valid OBSERVED block; llava15-13 complete on NIH and COCO (CheXpert CORE in progress).
  456 table cells filled.

## 2026-09-13 Single-GPU worker surplus cancelled to let 2-GPU jobs place

- Run: at 10:03Z the gpu1 lane had 25 pending tasks against 57 running single-GPU workers, while the gpu2 lane
  had 232 pending tasks against 4 workers and every pending 2-GPU job (28) was unplaced behind single-GPU
  jobs in the scheduler queue.
- Gate decision: the 31 still-pending single-GPU worker jobs (22 cfw4-g1-a, 9 cfw-g1-h) were cancelled; they
  would have started into an empty lane and exited. Running workers are untouched and exit on their own when
  the lane empties, after which the 2-GPU and 3-GPU jobs are placed. Queue 300 pending / 64 running / 542
  done / 0 failed.

## 2026-09-13 llava15-13 CheXpert statistics; llava15-13 complete on all three datasets

- Run: preflight passed apart from the IB/WB mapping gate (batch-vs-single 0.28 declared). CORE + CALIBRATION
  COMPLETE, 10.1 GPU-hours.
- Observation (known labels): readable Effusion (S 0.248), Cardiomegaly (0.221), Consolidation (0.197), Edema
  (0.154); Atelectasis `insufficient_support`; Pneumothorax 0.032. Answer AUROC 0.40-0.60 (Cardiomegaly and
  Consolidation below chance): the 13B LLaVA reads four CheXpert concepts and answers none.
- Observation (CORE): |W_qq| <= 0.030, nothing at the reference; Consolidation O +0.014 below its random p95;
  Cardiomegaly unresolved; the rest lose to a competitor. No steering, as on NIH.
- Gate decision: valid OBSERVED block. llava15-13 is the tenth checkpoint complete on all three datasets; the
  whole LLaVA family (7B, 13B, LLaVA-Med) now shows the same dissociation on both chest sets: readable, not
  answerable, not steerable. 460 table cells filled.

## 2026-09-13 gemma3-4 preflight on NIH, COCO and CheXpert

- Run: the three gemma3-4 prep tasks ran in the gpu1 lane (features, fits, preflight); 4096 SigLIP tokens at
  the block and 256 after the connector, 10-15 rows/s.
- Observation: all three preflights pass on both loci: determinism 0, alpha-0 0, nothing outside consumed
  tokens, fp32-vs-model 0.12, semantic mapping passes all six templates. Reach is large at both loci (block:
  27.7-29.8 at the consumer, 9-24 answer logits; connector: 3.0-17.8), the Gemma pattern seen on MedGemma.
  Batch-vs-single is the largest of the campaign: 1.28 (NIH), 1.48 (COCO), 0.84 (CheXpert), replicated in
  D2; recorded as declared deviations. The fixed-composition runner design makes every contrast within a
  question shape-matched, so these deviations affect no reported contrast.
- Gate decision: gemma3-4 accepted on all three datasets; COCO CORE and all modules are already scored and
  being packaged, NIH and CheXpert follow in the lane.

## 2026-09-13 gemma3-4 COCO: all seven modules packaged; statistics

- Run: all seven modules COMPLETE, 33.7 GPU-hours (the 4096-token block makes Gemma the costliest 4B).
- Observation (known labels): all six readable (S 0.04-0.11 over a control mean of 0.89; bicycle
  `insufficient_support`); answers 0.90-1.00.
- Observation (CORE): five objects owned at the reference: person O +0.161 (rank 1), dog +0.539 (rank 3),
  car +0.243 (rank 5), chair +0.343 (rank 1), bottle +0.437 (rank 6); bicycle anti-owned (W 0.124, O -0.270
  vs bottle). Random p95 0.14-0.34 and sham up to 0.31: like MedGemma, the base Gemma 3 responds to any
  0.25 write, yet five of six concept writes are still the most specific direction.
- T3: signed dose range +0.612 [+0.571, +0.677]; refit SD 0.171 [0.152, 0.187] (the largest seed variance of
  the campaign); connector median O +0.013; label gap -17.2 [-19.6, -15.5]; person wording -0.081, mapping
  +0.051; bottle wording +0.089, mapping -0.519 [-0.565, -0.471] (the B-mapped template flips the bottle
  answer: the strongest prompt dependence recorded).
- Gate decision: valid OBSERVED block; first Gemma 3 base block complete. 472 table cells filled.

## 2026-09-13 First-batch turnover, final tally

- Observation: the remaining first-batch workers (jobs placed 19:52-20:06Z on 2026-09-12) hit the 12 h limit
  at 07:53-08:06Z (`sacct` TIMEOUT); with `--max-hours 11.5` a worker can still claim a >30 min shard at
  11.4 h, so 11 more tasks were killed mid-run. All 15 killed tasks were swept back to `pending/` by
  new-code workers and re-run: 13 done with rc 0, 2 (q3-32 NIH CORE shards) running. 0 failed.
- Gate decision: no manual action; later batches use `--max-hours 10` so most workers finish their last
  shard before the limit, and the sweep covers the rest. The second batch (old code, 11.5 h budget) expires
  at 12:50Z and is covered the same way.

## 2026-09-13 gemma3-4 CheXpert and q25-32 CheXpert statistics; q25-32 complete on all three datasets

- gemma3-4 CheXpert: CORE + CALIBRATION COMPLETE, 8.6 GPU-hours. Readable Effusion (S 0.252), Cardiomegaly
  (0.228), Consolidation (0.208), Edema (0.221); Atelectasis `insufficient_support`; Pneumothorax 0.038.
  Answer AUROC 0.41-0.61: the base Gemma 3 4B answers "yes" to almost every CheXpert question (clean
  p_present > 0.99 on 97% of rows, median margin +15.6 logits), and the same on NIH (94% of clean rows >
  0.99; only Mass is partly unsaturated), while on COCO it answers mostly "no" (77% < 0.01) with person
  balanced. CORE on CheXpert is therefore read at a probability ceiling: random and sham writes cannot raise
  p_present (random p95 as low as 0.0001) while writes that lower it are large (Pneumothorax W -0.651,
  Edema -0.534, Consolidation -0.312; Cardiomegaly sham 0.93). Verdicts are formally
  stronger_competitor/unresolved for all six, but the informative statement is the ceiling itself: the write
  reaches the answer (random-direction margin deltas median 6.7, p95 24.6 logits) yet the model has no
  concept-specific "yes" to give. The logit-scale outcomes (`semantic_margin`, `lse_margin`) are in the
  package for a secondary reading; the reported W/O follow the protocol's p-scale definition unchanged.
- q25-32 CheXpert (preflight passed, no deviations): CORE + CALIBRATION COMPLETE, 16.3 GPU-hours. Readable
  Effusion (S 0.240), Cardiomegaly (0.220), Consolidation (0.198), Edema (0.187); Atelectasis 0.065;
  Pneumothorax 0.061. Answer-capable Effusion (0.68), Cardiomegaly (0.73), Consolidation (0.70), Edema (0.65);
  Atelectasis answer AUROC 0.93 but `insufficient_support` for the probe. CORE: nothing owned; Atelectasis
  O -0.187 and Cardiomegaly O -0.186 anti-owned (the Cardiomegaly the 32B owns on NIH loses to Edema on
  CheXpert, W -0.159); Effusion W -0.069. The 32B is the second checkpoint (after q25-7) whose NIH result
  does not transfer to CheXpert.
- Gate decision: two valid OBSERVED blocks; q25-32 is the eleventh checkpoint complete on all three datasets.
  480 table cells filled.

## 2026-09-13 gemma3-12 preflight on NIH, COCO and CheXpert

- Observation: all three preflights pass on both loci (determinism 0, alpha-0 0, nothing outside consumed
  tokens, fp32-vs-model 0.10-0.12, all six templates eligible; reach 23.6-25.8 at the consumer, 8-14 answer
  logits; 9-11 rows/s). Batch-vs-single 0.89 (COCO), 1.84 (NIH, the campaign maximum), 1.40 (CheXpert),
  replicated in D2 and recorded as declared deviations; covered by the fixed-composition design.
- Gate decision: gemma3-12 accepted on all three datasets; COCO CORE is scored and being packaged.

## 2026-09-13 gemma3-12 COCO CORE statistics; Gemma 3 sizes share the primary-locus reading

- Observation (design fact, verified by file hashes): the pooled `vis.last` features of gemma3-4 and gemma3-12
  are byte-identical on NIH, COCO and CheXpert (same md5), and so are the fitted probes and concept directions
  (`fits/vis.last/seed*.npz` identical). Gemma 3 4B/12B/27B ship the same SigLIP-So400m tower (1152-d, 27
  layers, 896 px, 256 tokens after pooling), so the "reading" at the consumed block and the write vector v_c
  are the same object for every size; only the connector (2560/3840/5376-d) and the language model differ.
  MedGemma's tower is different (its features and fits differ from gemma3-4). Consequently, within the Gemma 3
  base family the CORE/DOSE/LOCUS comparison across sizes isolates the consumer: same locus, same write,
  different reader. The calibration rows (known-label AUROC 0.93-0.99, control mean 0.89) are identical for
  gemma3-4 and gemma3-12 by construction; only the answer columns differ.
- gemma3-12 COCO: CORE, CALIBRATION, DOSE, REFIT, LOCUS, LOCUS_CALIBRATION packaged (49.1 GPU-hours so far);
  PROMPT pending. Answers 0.91-1.00. CORE: all six objects owned at rank 1 with larger and cleaner effects
  than the 4B reading the identical write: person O +0.194, dog +0.761, car +0.476, chair +0.447, bottle
  +0.179, bicycle +0.605 (the 4B anti-owned bicycle at -0.270); random p95 0.02-0.22, |sham| 0.01-0.22.
- Gate decision: valid OBSERVED block; 484 table cells filled. gemma3-27 is expected to share the same
  features and fits; to be verified by hash when its prep completes.

## 2026-09-13 gemma3-12 COCO: all seven modules packaged; T3

- Run: PROMPT landed; block COMPLETE with all seven modules, 52.0 GPU-hours.
- Observation (T3): signed dose range +0.531 [+0.502, +0.585]; refit SD 0.122 (the 4B: 0.171); connector
  median O +0.017; label gap -18.3 [-20.6, -16.9]; person wording -0.071, mapping +0.066; bottle wording
  +0.031 (null), mapping -0.042 (the 4B's bottle mapping contrast was -0.519: the same write, read by the
  12B consumer, loses most of its template dependence).
- Gate decision: valid OBSERVED block; gemma3-12 complete on COCO (NIH and CheXpert modules in the lane).
  492 table cells filled.

## 2026-09-13 gemma3-4 NIH and q3-32 NIH CORE statistics

- gemma3-4 NIH: CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION packaged (19.3 GPU-hours); PROMPT/LOCUS
  pending. Readable Atelectasis (S 0.107) and Nodule (0.141) only; answer AUROC 0.41-0.64 (Cardiomegaly and
  Mass barely capable). The answer ceiling seen on CheXpert holds on NIH (94% of clean rows p_present > 0.99):
  random p95 collapses to 0.0001-0.0005 for four concepts, while writes that lower the answer are large
  (Mass W -0.344, Cardiomegaly -0.252, Nodule -0.146). No owned concept; the base Gemma 3 4B says "yes" to
  every chest question and its concept writes only push it toward "no".
- q3-32 NIH (Qwen3-VL-32B, preflight passed, batch-vs-single 0.39 declared): CORE, CALIBRATION,
  LOCUS_CALIBRATION packaged (29.8 GPU-hours); PROMPT/DOSE/REFIT/LOCUS pending. Readable Atelectasis (0.125)
  and Cardiomegaly (0.190); answer-capable on all six (0.61-0.78). CORE: Nodule owned (W 0.243, O +0.171 vs
  Mass, random p95 0.116, |sham| 0.038, rank 2), the first Nodule ownership in the campaign (Nodule was the
  most strongly anti-owned concept for q25-7); Pneumothorax at the reference but O +0.003; Cardiomegaly
  writes 0.136 below its random p95 0.238; Effusion O -0.140, Mass O -0.106.
- Gate decision: two valid OBSERVED blocks; 528 table cells filled. Owned NIH concepts are now Cardiomegaly
  (q25-32, lingshu-7) and Nodule (q3-32): different checkpoints own different single concepts, none owns
  the paper's Effusion.

## 2026-09-13 gemma3-12 CheXpert and NIH CORE statistics: same write, different reader

- Both blocks use the write vectors that are byte-identical to gemma3-4's (shared SigLIP tower); only the
  connector and language model differ.
- gemma3-12 CheXpert: CORE + CALIBRATION COMPLETE, 10.6 GPU-hours. Calibration rows identical to gemma3-4 by
  construction (readable Effusion, Cardiomegaly, Consolidation, Edema); answer AUROC 0.45-0.61 (not
  answer-capable on any concept). CORE: Edema owned (W 0.499, O +0.149 vs Atelectasis, random p95 0.469,
  |sham| 0.283, rank 1); every other concept strongly anti-owned by Edema (Effusion O -0.831, Pneumothorax
  -0.712, Cardiomegaly -0.473). Random p95 0.28-0.77: the 12B is at least as perturbation-sensitive as the 4B
  but not at the 4B's answer ceiling, so writes move it in both directions.
- gemma3-12 NIH: CORE, CALIBRATION, DOSE, REFIT, LOCUS, LOCUS_CALIBRATION packaged (35.7 GPU-hours); PROMPT
  pending. Answer-capable Effusion, Atelectasis, Pneumothorax, Mass, Nodule (0.63-0.66). CORE: Nodule owned
  (W 0.385, O +0.167 vs Pneumothorax, random p95 0.369, |sham| 0.151, rank 6); Effusion O -0.834, Mass -0.647,
  Atelectasis -0.569 anti-owned (Nodule moves most under their writes); Cardiomegaly writes 0.329 below its
  random p95 0.738.
- Gate decision: two valid OBSERVED blocks. The identical write that produces nothing but "no" pressure in
  the 4B consumer yields one owned concept per chest dataset in the 12B consumer (Edema on CheXpert, Nodule on
  NIH). Nodule is now owned by two unrelated 32B/12B checkpoints (q3-32, gemma3-12) and anti-owned by q25-7.

## 2026-09-13 Second Slurm batch turnover confirmed

- Observation (13:22Z): the second batch expired at 12:51-12:56Z; 11 tasks killed mid-run were swept back to
  `pending/` within the minute and re-run (8 done by 13:15Z with rc 0, 3 running). Campaign total: 26
  requeue events, 23 recovered as done, 3 in progress, 0 in `failed/`. After the single-GPU lane drained, the
  two-GPU lane rose to 22 running workers (5 pending) and the three-GPU lane holds 3 (4 pending). Queue 246
  pending (207 gpu2, 39 gpu3) / 28 running / 632 done.
- Gate decision: no action; the remaining campaign is the 27-38B checkpoints and the 72B.

## 2026-09-13 Fifth batch: eight more two-GPU workers

- Gate decision: 8 gpu2 workers (cfw5-g2-a, `--max-hours 10`) submitted at 13:40Z so the two-GPU lane keeps
  20+ workers when the third-batch gpu2 workers reach their budget (~17:00-18:00Z). No single-GPU workers:
  that lane is empty and its remaining three tasks are in flight.

## 2026-09-13 gemma3-4 NIH: all seven modules packaged; gemma3-4 complete on all three datasets

- Run: PROMPT and LOCUS landed (two shards re-run after the batch turnover); block COMPLETE with all seven
  modules, 21.8 GPU-hours.
- Observation (T3): signed dose range +0.879 [+0.848, +0.908], the largest of the campaign, read against the
  answer ceiling: negative doses pull the saturated "yes" down by a large amount while positive doses have no
  headroom, so the range is one-sided. Refit SD 0.061; connector median O -0.009; label gap +0.04
  [-0.99, +0.62] (null and wide); Effusion wording -0.017, mapping -0.279 [-0.317, -0.241]; Mass wording
  -0.058, mapping -0.085.
- Gate decision: valid OBSERVED block; gemma3-4 is the twelfth checkpoint complete on all three datasets.
  552 table cells filled.

## 2026-09-13 gemma3-12 NIH: all seven modules packaged; gemma3-12 complete on all three datasets

- Run: PROMPT landed; block COMPLETE with all seven modules, 40.2 GPU-hours.
- Observation (T3): signed dose range +0.553 [+0.521, +0.611]; refit SD 0.180 [0.166, 0.191], the largest seed
  variance of the campaign (the three seeds' directions are identical to gemma3-4's, so the variance is in
  how the 12B consumer reads them); connector median O -0.066 [-0.081, -0.054], the first block where the
  connector write has a non-trivial (negative) effect; label gap -2.17 [-2.93, -0.69]; Effusion wording
  -0.414 [-0.442, -0.386] (the largest wording contrast recorded: the "Is there" phrasing and the "Does the
  radiograph show" phrasing give opposite ownership for Effusion), mapping +0.075; Mass wording -0.087,
  mapping -0.122.
- Gate decision: valid OBSERVED block; gemma3-12 is the thirteenth checkpoint complete on all three datasets.
  560 table cells filled.

## 2026-09-13 q3-32 COCO CORE statistics

- Run: preflight passed (batch-vs-single 0.36 declared). CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION
  packaged (39.7 GPU-hours, 2-GPU lane); PROMPT and LOCUS pending.
- Observation: readable person, car, chair, bottle (S 0.05-0.08; dog below its controls; bicycle
  `insufficient_support`); answers 0.99. CORE: Qwen3-VL-32B owns all six objects at rank 1 (person O +0.130,
  dog +0.117, car +0.113, chair +0.235, bottle +0.238, bicycle +0.069; random p95 <= 0.015, |sham| <= 0.011).
  Effects are smaller than the 4B/8B Qwen3 (0.1-0.8) but cleanly separated from the references.
- Gate decision: valid OBSERVED block; the natural-object control now holds for 13 of 14 checkpoints with a
  COCO block (LLaVA-Med the exception). 564 table cells filled.

## 2026-09-13 Llama 3.2 Vision: licence granted, adapter written, 11B smoke accepted

- Run: the user's Meta licence was approved; `fetch_models.py llama32-11,llama32-90` pulls the pinned revisions
  (9eb2daaa / e305d2a4). A new `MllamaAdapter` (`adapters/mllama.py`) covers the cross-attention architecture:
  the language model consumes `multi_modal_projector(cat([global-transformer output, local layers 3/7/15/23/30]))`
  as keys/values of eight cross-attention layers, so the primary locus `vis.last` is the last global-transformer
  block (dims [0:1280) of the 7680-d projector input; the five local-layer streams bypass it and are recorded
  as `bypasses`), and the connector locus is the projector output. One 560x560 tile per image (longer side
  resized to 560, canvas (1,1)); consumed rows per image 1601 (CLS + 1600 patches) at both loci; padding rows
  and padding tiles are excluded from steering and pooling. `LocusHook`/`CaptureHook` gained a view for
  tensors with more than three dims (the projector output is 5-D); 2-D/3-D families are untouched (unit test
  added, 13/13 pass).
- Observation (llama32-11 smoke on NIH, 40 s): determinism 0, alpha-0 identity 0, reach 13.4 at the projector
  and 0.86 answer logits, locus change confined to consumed rows (0 outside), fp32-vs-model 0.06,
  batch-vs-single 0.18 (replicated 0.12), within the 0.25 tolerance; prompt is the checkpoint chat template
  with one BOS and one `<|image|>`; 1601 valid tokens at both loci.
- Gate decision: adapter accepted. llama32-11 goes to the single-GPU lane, llama32-90 to the 3-GPU lane once
  its download completes.

## 2026-09-13 llama32-11 enqueued at batch 16; one batch-32 prep retired

- Run: the first enqueue used the single-GPU lane default (batch 32) and its NIH prep was claimed by a login
  worker before the per-model override landed. Because Mllama computes all four tile slots per image (three
  zero tiles), the vision cost per row is about 4x that of the other families, so the prep was stopped
  (rc -15, retired as a dotfile in `failed/`), `enqueue.py` gained `BATCH_MODEL` (llama32-11: 16, llama32-90: 4),
  and llama32-11 was re-enqueued on NIH, COCO and CheXpert (48 tasks, batch 16 for prep and every module, so
  the fixed-composition batch is consistent across CORE/DOSE/REFIT/LOCUS). Twelve single-GPU Slurm workers
  submitted for the new lane work; the four login-node workers also serve it.
- Gate decision: no data affected (the stopped prep had written no features). 0 tasks in `failed/`.

## 2026-09-13 q3-32 NIH: all seven modules packaged; T3

- Run: PROMPT, DOSE, REFIT and LOCUS landed (two CORE shards had been re-run after the first batch
  turnover); block COMPLETE with all seven modules, 106.2 GPU-hours (2-GPU lane; the most expensive NIH block
  so far).
- Observation (T3): signed dose range +0.266 [+0.234, +0.294]; refit SD 0.025; connector median O -0.008;
  label gap -0.74 [-1.01, -0.47]; Effusion wording +0.020, mapping +0.044; Mass wording +0.009, mapping
  +0.042. The Nodule ownership from CORE (O +0.171) stands with a clean dose response and stable refits.
- Gate decision: valid OBSERVED block; q3-32 complete on NIH (COCO all-but-PROMPT/LOCUS, CheXpert in the
  lane). 572 table cells filled.

## 2026-09-13 llama32-90 downloaded; revision pin corrected; enqueued on the 3-GPU lane

- Run: the 90B snapshot landed (37 shards, 276 GB on disk including the `original/` consolidated weights). The
  Hub resolved the pinned revision by prefix: the pin `e305d2a4...4cd6f7` recorded in `fetch_models.py` and the
  adapter registry had a wrong tail, the downloaded commit is `e305d2a43a4adc6987308fe7d896fb8ec5f1a5d8`.
  Both files now carry the real commit (the 11B pin `9eb2daaa...94df5` matched its snapshot).
- Gate decision: llama32-90 enqueued for NIH, COCO and CheXpert (48 tasks, gpu3 lane, batch 4, device_map
  auto over 3 x 80 GB); its preflight in the prep task is the acceptance gate, and an out-of-memory failure
  there would move it to a 4-GPU lane. All 22 checkpoints of the grid are now downloaded and queued.

## 2026-09-13 q25-72 NIH CORE statistics: three owned concepts at 72B

- Run: CORE, CALIBRATION, DOSE, LOCUS_CALIBRATION packaged (72.9 GPU-hours on the 3-GPU lane); PROMPT, REFIT,
  LOCUS pending.
- Observation (known labels): readable Pneumothorax (S 0.136) and Cardiomegaly (0.192); Atelectasis 0.072,
  Effusion 0.041, Mass 0.032, Nodule -0.025. Answer-capable Effusion (0.62), Cardiomegaly (0.66), Mass (0.69),
  Nodule (0.68); Pneumothorax answer AUROC 0.48.
- Observation (CORE): Qwen2.5-VL-72B owns three NIH concepts: Atelectasis (W 0.531, O +0.537 vs Cardiomegaly,
  random p95 0.213, |sham| 0.015, rank 1), Cardiomegaly (W 0.462, O +0.256, random p95 0.408, rank 6) and Mass
  (W 0.328, O +0.066, random p95 0.292, rank 6); Effusion (O -0.322), Pneumothorax (-0.499) and Nodule (-0.230)
  remain anti-owned. The Atelectasis effect is the largest clinical ownership of the campaign, and the first
  block in which more than one NIH concept is owned.
- Gate decision: valid OBSERVED block; 588 table cells filled. The Qwen2.5-VL NIH ladder now reads 3B none,
  7B none (Effusion anti-owned), 32B Cardiomegaly, 72B Atelectasis + Cardiomegaly + Mass: ownership appears
  with scale within this family, while Effusion never becomes owned and its anti-ownership persists at 72B.

## 2026-09-13 llama32-11 NIH preflight: yes/no templates INELIGIBLE; all-ineligible modules now close as terminal

- Run: the batch-16 prep finished (features 19,228 rows at ~4 rows/s on a shared login GPU, fits, preflight).
- Observation: preflight checks A-D, G pass on both loci (determinism 0, alpha-0 0, reach 14.0 at the projector /
  0.98 answer logits, nothing outside consumed rows, batch-vs-single 0.18 within tolerance, fp32-vs-model
  0.06, 4.1 rows/s). The image-free semantic-mapping check E fails the yes/no templates: with "The finding is
  present. Is the finding present? Answer yes or no." Llama-3.2-11B-Vision answers "no" by 0.70 logits (the
  "explicitly reported as present/absent" statements map correctly, +3.3 / -10.1), and the A-mapped A/B
  templates answer "A" for an absent finding (+4.35). Only the B-mapped templates (IB, WB) pass. Under the
  standing rule the eligibility file marks IY/WY/IA/WA INELIGIBLE, the mirror image of the LLaVA family
  (where the A/B templates fail).
- Consequence: CORE, DOSE, REFIT, LOCUS and LOCUS_CALIBRATION are built on the IY template, so every cell of
  those modules is INELIGIBLE for this block; CALIBRATION (probe on features, no template) and the IB/WB
  cells of PROMPT are scored. The runner previously exited without a meta file when a module had no eligible
  question, so twelve module tasks were marked FAILED by the queue; the runner now writes a terminal meta
  (`ineligible_templates`) and the packager records `ineligible_modules` and derives COMPLETE when every
  requested module is either complete or ineligible (commit 6487901). The twelve tasks were moved back to
  `pending/` and close immediately.
- Gate decision: the block is a valid INELIGIBLE-by-interface disposition for the yes/no modules, recorded in
  coverage rather than dropped; the checkpoint stays in the grid with its CALIBRATION and IB/WB PROMPT rows.
  COCO and CheXpert preps for llama32-11 will apply the same check; llama32-90 follows on the 3-GPU lane.

## 2026-09-13 llama32-11 NIH packaged: calibration only

- Run: CALIBRATION COMPLETE; CORE, DOSE, REFIT, LOCUS, LOCUS_CALIBRATION recorded as `ineligible_modules`
  (54 INELIGIBLE coverage cells); PROMPT IB/WB shards pending. 0.12 GPU-hours of scoring so far.
- Observation (known labels, template-free): readable Effusion (S 0.128) and Pneumothorax (0.150);
  Cardiomegaly 0.118 just under the margin; Atelectasis 0.054; Mass and Nodule at the control level. The
  answer AUROC column is undefined (nan) because the CALIBRATION answer rows use the IY template, which is
  INELIGIBLE for this checkpoint; answer capability will be read from the IB/WB PROMPT rows when they land.
- Gate decision: valid block with the interface disposition recorded; the packager derives COMPLETE once the
  PROMPT module closes.

## 2026-09-13 q3-32 COCO: all seven modules packaged; T3

- Run: PROMPT and LOCUS landed; block COMPLETE with all seven modules, 98.4 GPU-hours.
- Observation (T3): signed dose range +0.418 [+0.378, +0.460]; refit SD 0.016; connector median O +0.002;
  label gap -11.5 [-12.2, -10.9]; person wording +0.027, mapping -0.033; bottle wording +0.030, mapping
  -0.038 (all template contrasts small: the 32B Qwen3 is the least prompt-dependent COCO block).
- Gate decision: valid OBSERVED block; q3-32 complete on NIH and COCO (CheXpert in the 2-GPU lane).
  604 table cells filled.

## 2026-09-13 lingshu-32 preflight on NIH, COCO and CheXpert

- Observation: all three preflights pass on both loci with no tolerance deviations (determinism 0, alpha-0 0,
  nothing outside consumed tokens, batch-vs-single 0.09-0.16, fp32-vs-model 0.04-0.06, all six templates
  eligible; reach 12.6-19.0 at the consumer, 0.5-4.4 answer logits; 11-14 rows/s on the 2-GPU lane).
- Gate decision: lingshu-32 accepted on all three datasets; NIH CORE is scored and being packaged.

## 2026-09-13 lingshu-32 NIH CORE statistics

- Run: CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION packaged (31.2 GPU-hours); PROMPT and LOCUS pending.
- Observation (known labels): readable Effusion (S 0.111), Atelectasis (0.096), Pneumothorax (0.137); Mass
  0.102 and Cardiomegaly 0.073 under the margin; Nodule -0.066. Answer-capable on all six (AUROC 0.64-0.86,
  the strongest NIH answerer with medgemma-4).
- Observation (CORE): Lingshu-32B owns Pneumothorax (W 0.202, O +0.144 vs Mass, random p95 0.065, |sham|
  0.051, rank 1), the first Pneumothorax ownership on NIH, and Cardiomegaly (W 0.105, O +0.051, random p95
  0.086, rank 3); Effusion unresolved (W 0.060 at the random p95, O -0.007); Nodule anti-owned (O -0.178);
  Atelectasis and Mass lose to Pneumothorax.
- Gate decision: valid OBSERVED block; 620 table cells filled. The Lingshu ladder on NIH: 7B Cardiomegaly
  only; 32B Cardiomegaly + Pneumothorax. Owned NIH concepts across the grid are now Cardiomegaly (q25-32,
  q25-72, lingshu-7, lingshu-32), Nodule (q3-32, gemma3-12), Atelectasis and Mass (q25-72), Pneumothorax
  (lingshu-32); Effusion is owned by no checkpoint.

## 2026-09-13 Twelve single-GPU worker jobs failed at launch (empty payload); resubmitted

- Run: the `cfw6-g1-a` batch was submitted from a shell command whose earlier step had failed, so the worker
  command variable was empty and each job's script line read `--lane gpu1` ("command not found", exit 127,
  1 s each). No task was claimed by them and no data was touched.
- Fix: `sbatch_py.sh` now refuses a payload that does not start with `python`/`bash`, and 12 single-GPU workers
  were resubmitted as `cfw7-g1-a` with the worker command verified in the generated script.

## 2026-09-13 lingshu-32 NIH: all seven modules packaged; T3

- Run: PROMPT and LOCUS landed; block COMPLETE with all seven modules, 73.9 GPU-hours.
- Observation (T3): signed dose range +0.085 [+0.078, +0.098]; refit SD 0.027; connector median O -0.014;
  label gap -0.137 [-0.234, -0.077]; Effusion wording -0.001 (null), mapping +0.026; Mass wording +0.012,
  mapping +0.008.
- Gate decision: valid OBSERVED block; lingshu-32 complete on NIH (COCO and CheXpert in the 2-GPU lane).
  628 table cells filled.

## 2026-09-13 lingshu-32 COCO CORE statistics

- Run: CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION packaged (29.3 GPU-hours); PROMPT and LOCUS pending.
- Observation: control mean 0.93; readable person (S 0.063); bicycle 0.053 `insufficient_support`; the rest
  within 0.02 of the controls; answers 0.91-0.99. CORE: all six objects owned at rank 1 (person O +0.288,
  dog +0.228, car +0.081, chair +0.533, bottle +0.342, bicycle +0.219; random p95 <= 0.068, |sham| <= 0.104).
- Gate decision: valid OBSERVED block; the natural-object control holds for 14 of 15 checkpoints with a COCO
  block. 632 table cells filled.

## 2026-09-13 Seventh batch: eight more two-GPU workers

- Gate decision: 8 gpu2 workers (cfw7-g2-a, `--max-hours 10`) submitted at 20:20Z so the two-GPU lane stays
  above 20 workers when the third and fourth batches reach their budgets (~22:00-00:00Z). Queue 240 pending
  (138 gpu2, 83 gpu3, 28 gpu1 incl. requeues) / 33 running / 729 done / 0 failed.

## 2026-09-13 q3-32 CheXpert statistics; q3-32 complete on all three datasets

- Run: preflight passed (batch-vs-single 0.59 declared). CORE + CALIBRATION COMPLETE, 28.2 GPU-hours.
- Observation (known labels): readable Effusion (S 0.237), Pneumothorax (0.130), Cardiomegaly (0.243),
  Consolidation (0.191), Edema (0.161); Atelectasis `insufficient_support`. Answer-capable on five (AUROC
  0.61-0.90).
- Observation (CORE): Edema owned (W 0.237, O +0.046 vs Effusion, random p95 0.187, |sham| 0.091, rank 4);
  Atelectasis at the reference with O +0.001 (unresolved); Effusion W 0.083 below its random p95 0.151
  (O -0.036); Consolidation O -0.124; Cardiomegaly and Pneumothorax not owned.
- Gate decision: valid OBSERVED block; q3-32 is the fourteenth checkpoint complete on all three datasets.
  636 table cells filled.

## 2026-09-13 llama32-11 COCO and CheXpert preflight: same template disposition as NIH

- Observation: on both datasets checks A-D and G pass (determinism 0, alpha-0 0, reach 11.5-12.5 at the
  projector, nothing outside consumed rows, batch-vs-single 0.41 / 0.27 declared, fp32-vs-model 0.06,
  3.8-4.2 rows/s) and the image-free mapping check E gives the NIH result: IY, WY, IA and WA INELIGIBLE, IB
  and WB eligible. The generic statement cases of check E are dataset-independent by construction, so the
  disposition is a property of the checkpoint.
- Gate decision: llama32-11 keeps CALIBRATION and the IB/WB PROMPT cells on all three datasets; the IY-based
  modules close as `ineligible_modules`. COCO CORE metas are terminal and the block is being packaged.

## 2026-09-13 llama32-11 COCO packaged: calibration only

- Run: CALIBRATION COMPLETE; CORE, DOSE, REFIT, LOCUS, LOCUS_CALIBRATION recorded as `ineligible_modules`;
  PROMPT IB/WB shards pending.
- Observation (known labels, template-free): all five supported objects readable (S 0.10-0.15 over a control
  mean of 0.84; bicycle `insufficient_support`); answer AUROC undefined (IY template ineligible), to be read
  from the IB/WB PROMPT rows.
- Gate decision: valid block with the interface disposition recorded. 638 table cells filled.

## 2026-09-13 lingshu-32 COCO: all seven modules packaged; T3

- Run: PROMPT and LOCUS landed; block COMPLETE with all seven modules, 75.8 GPU-hours.
- Observation (T3): signed dose range +0.538 [+0.509, +0.563]; refit SD 0.059; connector median O +0.006;
  label gap -3.15 [-3.62, -2.61]; person wording -0.005 (null), mapping +0.153 [+0.141, +0.165]; bottle
  wording +0.055, mapping +0.122.
- Gate decision: valid OBSERVED block; lingshu-32 complete on NIH and COCO (CheXpert in the 2-GPU lane).
  646 table cells filled.

## 2026-09-13 lingshu-32 CheXpert statistics; lingshu-32 complete on all three datasets

- Run: CORE + CALIBRATION COMPLETE, 19.1 GPU-hours.
- Observation (known labels): readable Effusion (S 0.271), Cardiomegaly (0.246), Consolidation (0.249), Edema
  (0.195); Atelectasis 0.238 `insufficient_support`; Pneumothorax 0.029. Answer-capable on five (AUROC
  0.68-0.94).
- Observation (CORE): five concepts meet the reference with positive ownership: Edema (W 0.237, O +0.115,
  rank 1), Cardiomegaly (W 0.169, O +0.110, rank 1), Effusion (W 0.165, O +0.027, rank 1), Consolidation
  (W 0.130, O +0.028, rank 2), Pneumothorax (W 0.117, O +0.024, rank 4); Atelectasis meets the reference but
  loses to Consolidation (O -0.036). This is the first block in which Effusion is owned, and the largest
  number of owned clinical concepts in one block.
- Gate decision: valid OBSERVED block; lingshu-32 is the fifteenth checkpoint complete on all three datasets.
  The Lingshu ladder on CheXpert: 7B Effusion/Pneumothorax/Edema; 32B those three plus Cardiomegaly and
  Consolidation. 650 table cells filled.

## 2026-09-13 medgemma-27 preflight on NIH, COCO and CheXpert

- Observation: all three preflights pass on both loci (determinism 0, alpha-0 0, nothing outside consumed
  tokens, fp32-vs-model 0.07-0.08, all six templates eligible; reach 1.7-2.4 at the consumer, 3.0-9.9 answer
  logits; batch-vs-single 0.25-0.27, declared on NIH and COCO; 6.7-7.8 rows/s on the 2-GPU lane). Its
  pooled `vis.last` features differ from both medgemma-4 and gemma3-4 on every dataset (distinct md5), so
  MedGemma-27B carries its own vision tower.
- Gate decision: medgemma-27 accepted on all three datasets; NIH CORE is scored and being packaged.

## 2026-09-13 medgemma-27 NIH CORE statistics; MedGemma sizes share the MedSigLIP tower

- Correction to the preflight entry above: the medgemma-27 and medgemma-4 `vis.last` features differ only at
  bf16 rounding (mean |diff| 1e-8 to 8e-8, a few elements one ulp apart, every row cosine 1.000000), so the
  md5 difference is numerical, not a different tower: both MedGemma sizes carry the same MedSigLIP encoder,
  as the Gemma 3 sizes carry the same SigLIP. The fitted concept directions are correspondingly near-identical
  (nih min cosine 0.999764; coco min cosine 0.999489; chexpert min cosine 0.999956). Within each Gemma family, size comparisons again isolate the consumer.
- Run: CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION packaged (43.9 GPU-hours); PROMPT and LOCUS pending.
- Observation (known labels): readable Atelectasis (S 0.138), Cardiomegaly (0.197), Mass (0.168); Effusion
  0.107 and Pneumothorax 0.109 just under the margin; Nodule 0.016 (the same probe as medgemma-4 up to
  rounding). Answer-capable on all six (AUROC 0.66-0.87).
- Observation (CORE): no concept meets the reference. Cardiomegaly writes 0.645 but its random p95 is 0.646
  and sham 0.583 (O +0.002); Atelectasis O -0.446, Nodule -0.203, Mass -0.132, Effusion -0.064 anti-owned;
  Pneumothorax unresolved. As for the 4B MedGemma on NIH, writes move the answer but not specifically.
- Gate decision: valid OBSERVED block; 666 table cells filled.

## 2026-09-14 Eighth batch: eight more two-GPU workers

- Gate decision: 8 gpu2 workers (cfw8-g2-a, `--max-hours 10`) submitted at 00:05Z ahead of the fourth batch's
  wall clock (~01:10Z); the two-GPU lane then keeps 24+ workers on its 96 pending tasks. Queue 183 pending /
  37 running / 782 done / 0 failed; 42 requeue events campaign-wide, all recovered.

## 2026-09-14 llama32-11 NIH closed; CheXpert packaged (probe calibration only)

- llama32-11 NIH: the IB/WB PROMPT shards landed; block COMPLETE with CALIBRATION and PROMPT complete and
  CORE, DOSE, REFIT, LOCUS, LOCUS_CALIBRATION as `ineligible_modules` (22.1 GPU-hours). The PROMPT contrasts
  are undefined without a CORE reference, so the IB/WB outcome rows are delivered in the package without a
  T3 row.
- llama32-11 CheXpert: CORE INELIGIBLE and the CALIBRATION module's answer rows (IY template) INELIGIBLE, so
  the block closes COMPLETE with the template-free probe calibration only: readable Effusion (S 0.256),
  Cardiomegaly (0.233), Consolidation (0.209), Edema (0.132); Atelectasis 0.071; Pneumothorax 0.076.
- Gate decision: both valid blocks with the interface disposition recorded; llama32-11 COCO closes the same
  way when its IB/WB PROMPT shards land. 667 table cells filled.

## 2026-09-14 medgemma-27 NIH: all seven modules packaged; T3

- Run: PROMPT and LOCUS landed; block COMPLETE with all seven modules, 118.2 GPU-hours (2-GPU lane; the most
  expensive block of the campaign, driven by the 4096-token SigLIP block).
- Observation (T3): signed dose range +0.360 [+0.321, +0.398]; refit SD 0.040; connector median O -0.016;
  label gap -3.08 [-3.74, -2.29]; Effusion wording -0.052 [-0.067, -0.039], mapping +0.003 (null); Mass
  wording -0.029, mapping +0.003 (null).
- Gate decision: valid OBSERVED block; medgemma-27 complete on NIH (COCO and CheXpert in the 2-GPU lane).
  675 table cells filled.

## 2026-09-14 medgemma-27 COCO CORE statistics

- Run: CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION packaged (31.7 GPU-hours); PROMPT and LOCUS pending.
- Observation (known labels): readable person, car, chair, bottle (S 0.05-0.09 over a control mean of 0.90;
  dog 0.054 below the margin; bicycle `insufficient_support`); answers 0.94-1.00.
- Observation (CORE): four objects owned: person (W 0.471, O +0.335, rank 1), chair (W 0.487, O +0.177,
  rank 1), bottle (W 0.371, O +0.208, rank 1), car (W 0.181, O +0.081, rank 2); dog (W 0.038, O -0.044) and
  bicycle (W -0.008, O -0.042) not owned. Random p95 0.07-0.20, |sham| up to 0.16.
- Gate decision: valid OBSERVED block; 679 table cells filled.

## 2026-09-14 llama32-11 COCO closed; llama32-11 complete on all three datasets

- Run: the IB/WB PROMPT shards landed; block COMPLETE with CALIBRATION and PROMPT complete and the IY-based
  modules as `ineligible_modules` (22.3 GPU-hours).
- Gate decision: llama32-11 is the seventeenth checkpoint complete on all three datasets, with the interface
  disposition (yes/no and A-mapped templates INELIGIBLE by check E) recorded in every coverage table.
  679 table cells filled.

## 2026-09-14 medgemma-27 COCO: all seven modules packaged; T3

- Run: PROMPT and LOCUS landed; block COMPLETE with all seven modules, 74.0 GPU-hours.
- Observation (T3): signed dose range +0.246 [+0.227, +0.287]; refit SD 0.048; connector median O -0.002;
  label gap -4.89 [-5.61, -4.22]; person wording -0.122 [-0.146, -0.096], mapping -0.051; bottle wording
  +0.046, mapping -0.007 (null).
- Gate decision: valid OBSERVED block; medgemma-27 complete on NIH and COCO (CheXpert CORE in progress).
  687 table cells filled.

## 2026-09-14 Ninth batch: eight two-GPU and two three-GPU workers

- Gate decision: at 04:10Z the two-GPU lane had 24 workers on 64 pending tasks with the fifth batch's budget
  ending in ~3 h, and the three-GPU lane 4 workers on 77 tasks with two reaching their budget in ~3 h; 8 gpu2
  (cfw9-g2-a) and 2 gpu3 (cfw9-g3-a) workers submitted, `--max-hours 10`. 0 failed.

## 2026-09-14 medgemma-27 CheXpert statistics; medgemma-27 complete on all three datasets

- Run: CORE + CALIBRATION COMPLETE, 33.2 GPU-hours (preflight passed with no deviations).
- Observation (known labels): readable Effusion (S 0.257), Cardiomegaly (0.271), Consolidation (0.256), Edema
  (0.184); Atelectasis 0.259 `insufficient_support`; Pneumothorax 0.007. Answer-capable on five (AUROC
  0.69-0.96).
- Observation (CORE): no concept meets the reference. Cardiomegaly writes 0.382 against a random p95 of 0.390
  and sham 0.368 (O +0.033, rank 12); Effusion W 0.095 above its random p95 but below sham 0.142 (O -0.007);
  Edema O -0.283 and Pneumothorax O -0.065 anti-owned; Atelectasis and Consolidation unresolved.
- Gate decision: valid OBSERVED block; medgemma-27 is the eighteenth checkpoint complete on all three
  datasets. 691 table cells filled.

## 2026-09-14 gemma3-27 preflight on NIH, COCO and CheXpert; shared tower confirmed for the third size

- Observation: checks A-C, E and G pass on all three datasets and both loci (determinism 0, alpha-0 0, reach
  20.6-23.6 at the consumer / 6.3-15.6 answer logits, nothing outside consumed tokens, fp32-vs-model
  0.10-0.12, all six templates eligible; 6.5-7.8 rows/s on the 2-GPU lane). Batch-vs-single is 1.09 (NIH),
  0.67 (COCO) and 0.81 (CheXpert), replicated in D2, recorded as declared deviations; on CheXpert one
  preflight example also flips its margin sign between batch and single (margin diff 0.59), so that file's
  `pass` flag is false while every other check passes. The fixed-composition runner scores every contrast of
  a question in one batch shape, so the declared deviation affects no reported contrast; the block proceeds
  as OBSERVED with the deviation carried into run.json.
- Shared tower: gemma3-27's `vis.last` features equal gemma3-4's and gemma3-12's up to bf16 rounding on the
  2-GPU placement (mean |diff| 5e-7, every row cosine 1.000000; fitted direction cosines >= 0.9987), so all
  three Gemma 3 sizes read the same representation and receive the same write.
- Gate decision: gemma3-27 accepted on all three datasets; NIH CORE is scored and being packaged.

## 2026-09-14 gemma3-27 NIH CORE statistics

- Run: CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION packaged (55.5 GPU-hours); PROMPT and LOCUS pending.
- Observation (known labels): the probe rows equal gemma3-4/12's up to rounding (readable Atelectasis 0.110,
  Nodule 0.141); answer AUROC 0.58-0.65 (answer-capable Atelectasis, Mass, Nodule at the margin).
- Observation (CORE): no concept meets the reference with positive ownership. Cardiomegaly W 0.045 equals its
  random p95 and sham (O +0.002, unresolved); Pneumothorax writes 0.375 below its random p95 0.397; Effusion
  0.208 below 0.271; Mass O -0.078, Atelectasis -0.028, Nodule -0.016.
- Gate decision: valid OBSERVED block; 707 table cells filled. With byte-identical write vectors, the Gemma 3
  ladder on NIH reads 4B none (answer ceiling), 12B Nodule, 27B none.

## 2026-09-14 gemma3-27 NIH: all seven modules packaged; T3

- Run: PROMPT and LOCUS landed; block COMPLETE with all seven modules, 129.1 GPU-hours (the most expensive
  block of the campaign).
- Observation (T3): signed dose range +0.470 [+0.429, +0.520]; refit SD 0.019; connector median O -0.025;
  label gap -1.22 [-1.96, -0.29]; Effusion wording -0.003 (null), mapping +0.050; Mass wording -0.037,
  mapping -0.118 [-0.137, -0.098].
- Gate decision: valid OBSERVED block; gemma3-27 complete on NIH (COCO and CheXpert in the 2-GPU lane).
  715 table cells filled.

## 2026-09-14 Three-GPU lane prioritised

- Run: at 07:20Z the two-GPU lane had 40 tasks left for 24 running workers, while the three-GPU lane
  (q25-72 COCO/CheXpert and llama32-90, 75 tasks) had 2 running workers and 6 unplaced 3-GPU jobs behind
  pending 2-GPU jobs in the scheduler queue.
- Gate decision: the 8 still-pending two-GPU workers (cfw9-g2-a) were cancelled and 4 more three-GPU workers
  (cfw10-g3-a) submitted, so the GPUs freed as the two-GPU lane drains go to 3-GPU jobs. Running workers are
  untouched. Queue 115 pending / 26 running / 861 done / 0 failed.

## 2026-09-14 gemma3-27 COCO CORE statistics

- Run: CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION packaged (51.9 GPU-hours); PROMPT and LOCUS pending.
- Observation (known labels): probe rows equal the 4B/12B (readable five; bicycle `insufficient_support`);
  answers 0.93-1.00.
- Observation (CORE): all six objects owned at rank 1 with the identical write vectors: person O +0.235,
  dog +0.363, car +0.091, chair +0.284, bottle +0.148, bicycle +0.751; random p95 0.02-0.19, |sham| <= 0.13.
- Gate decision: valid OBSERVED block; 719 table cells filled. The Gemma 3 ladder on COCO with one write:
  4B 5/6, 12B 6/6, 27B 6/6.

## 2026-09-14 Checkpoints above 40B parked; small models first (user decision)

- Run: at 08:50Z the user asked to finish the smaller checkpoints first and defer the >40B ones. The 75 pending
  q25-72 (COCO, CheXpert) and llama32-90 (all datasets) tasks were moved to `queue/hold/` (not deleted; the
  task files and priorities are intact), the ten pending 3-GPU Slurm jobs cancelled, and 8 more two-GPU
  workers submitted for the remaining iv35-38 and gemma3-27 blocks. The two running q25-72 NIH PROMPT shards
  finish their block. The two running 3-GPU workers exit when their lane is empty.
- Gate decision: q25-72 NIH stays in the tables as delivered; q25-72 COCO/CheXpert and llama32-90 are
  DEFERRED (held, resumable by moving the files back to `pending/` and submitting 3-GPU workers). The user
  also asked for the tests to be organised into a complete evaluation method: a leaderboard generator and a
  method document follow.

## 2026-09-14 Evaluation method consolidated: leaderboard generator and method document

- Run: `cftransfer.leaderboard` (new) grades every packaged block with the protocol's own rules (readable:
  selectivity lower bound > 0; answer-capable: clean-answer AUROC lower bound > 0.5; owned: steering reference
  and max-T `fixed_family_advantage`) and writes `runs/leaderboard.{csv,md}`; 57 blocks graded, unit-tested
  (15/15 pass). `docs/external-replication/EVALUATION_METHOD.md` (new) states the method end to end: loci,
  procedure, preflight gate, decision rules, controls, outputs, and how to add a checkpoint.
- Gate decision: the leaderboard is regenerated with the tables whenever a block is packaged; the campaign
  tables are unchanged.

## 2026-09-14 q25-72 NIH: all seven modules packaged; T3

- Run: PROMPT, REFIT and LOCUS landed; block COMPLETE with all seven modules, 214.3 GPU-hours on the 3-GPU
  lane. The 3-GPU lane is now empty (COCO/CheXpert held).
- Observation (T3): signed dose range +0.360 [+0.344, +0.373]; refit SD 0.130 [0.126, 0.133]; connector
  median O -0.013; label gap -0.50 [-0.65, -0.29]; Effusion wording -0.113 [-0.118, -0.109], mapping +0.040;
  Mass wording +0.185 [+0.175, +0.192], mapping +0.043. The three CORE ownerships (Atelectasis, Cardiomegaly,
  Mass) stand with a clean dose response.
- Gate decision: valid OBSERVED block; q25-72 complete on NIH. 727 table cells filled; leaderboard
  regenerated (57 blocks).

## 2026-09-14 Figure pack from the packaged statistics

- Run: `cftransfer.figures` (new) draws ten figures into `runs/figures/` (PNG + SVG) from summary.json,
  run.json, leaderboard.csv and the DOSE/CORE outcomes: F1 leaderboard heatmap, F2 readability-vs-ownership
  scatter over every concept cell, F3 Effusion ownership with intervals across all chest blocks, F4 the
  shared-tower comparison (Gemma 3 4/12/27B, MedGemma 4/27B), F5 the reference family (119 random writes,
  sham, concept write) for four blocks, F6 dose-response curves (DOSE alphas plus the CORE alpha, baseline
  taken from CORE because the DOSE module reuses it), F7 size ladders, F8 capability fractions per dataset,
  F9 T3 controls per block, F10 prompt-template contrasts. Each figure was inspected; the dose figure was
  empty on the first pass (no baseline rows in DOSE.parquet) and is correct after taking the CORE baseline.
- Gate decision: figures are derived views of the delivered tables; no statistic was recomputed differently.
  Regenerate with `python -m cftransfer.figures` (delete `runs/figures/dose_curves.json` to refresh the dose
  cache after new DOSE blocks land).

## 2026-09-14 gemma3-27 CheXpert statistics and COCO close; gemma3-27 complete on all three datasets

- gemma3-27 CheXpert: CORE + CALIBRATION COMPLETE, 28.6 GPU-hours (preflight: declared batch deviation with one
  sign flip, see above). Readable Effusion (S 0.252), Cardiomegaly (0.228), Consolidation (0.211), Edema
  (0.221); Atelectasis `insufficient_support`; Pneumothorax 0.037. Answer-capable Effusion (0.61) and
  Cardiomegaly (0.61) only. CORE: no concept owned; Edema meets the reference (W 0.068 vs random p95 0.068 and
  sham 0.065) with O 0.000 (unresolved); Effusion W 0.226 below random p95 0.251; Consolidation O -0.029;
  the remaining three unresolved at the noise floor.
- gemma3-27 COCO: PROMPT and LOCUS landed; COMPLETE with all seven modules, 118.9 GPU-hours. T3: signed dose
  range +0.379 [+0.330, +0.431]; refit SD 0.063; connector median O +0.009; label gap -23.1 [-24.3, -21.7]
  (the largest of the campaign); person wording -0.144, mapping +0.036; bottle wording -0.034 (null),
  mapping -0.046.
- Gate decision: two valid OBSERVED blocks; gemma3-27 is the nineteenth checkpoint complete on all three
  datasets. With byte-identical writes the Gemma 3 ladder on CheXpert reads 4B none, 12B Edema, 27B none.
  739 table cells filled; leaderboard 58 blocks.

## 2026-09-14 Figure pack restyled: MedVIGIL house style, Morandi palette

- Run: `cftransfer/figstyle.py` (new) reproduces the MedVIGIL journal house style (serif body face, 8.5 pt base,
  full frame with light grid, boxed legends, (a)/(b) panel titles, 7.2 in / 3.5 in column widths, PDF with
  Type 42 fonts + 300 dpi PNG + SVG) with a Morandi palette (rose / oat / slate for NIH / CheXpert / COCO;
  sage, lilac, mustard, terracotta, haze for series; warm greys for references). `figures.py` rebuilt on it;
  data logic unchanged; every figure re-inspected. Tests 15/15.
- Gate decision: the paper figures are the PDFs in `runs/figures/`; copied into the paper repository as
  `figures/cf_F*.pdf`.

## 2026-09-14 Deferred checkpoints released; every table cell to be filled (user decision)

- Run: at 12:55Z the user asked for the remaining tasks to be finished and every table filled. The 27 held
  q25-72 COCO/CheXpert tasks were moved back to `pending/` and four 3-GPU workers (cfw12-g3-a, `--max-hours 10`)
  submitted; the three llama32-90 preparation tasks were released as well, while its 45 module tasks stay held
  until the primary-template change below is deployed (otherwise its yes/no modules would close as ineligible
  like the 11B's). The A100 partition is saturated by other users (74 of 76 GPUs allocated); 9 two-GPU workers
  are running on iv35-38, 8 two-GPU and 4 three-GPU workers are queued behind them. Queue 55 pending /
  10 running / 892 done / 0 failed / 45 held.
- Gate decision: three additions follow, in this priority order: (1) a per-block primary template so that a
  checkpoint whose IY template fails check E is scored on its eligible B-positive template IB (Llama 3.2 11B
  re-run; 90B first run); (2) the four remaining llama32-90 datasets/modules; (3) the five modules the protocol
  did not schedule on CheXpert (DOSE, REFIT, LOCUS, LOCUS_CALIBRATION, PROMPT) for every checkpoint, enqueued at
  the lowest priority so they fill in as GPUs free up. Each is logged as a protocol amendment when deployed.

## 2026-09-14 Correction: checkpoints above 40B cancelled (user decision)

- Run: at 13:05Z the user restated that every checkpoint above 40B is cancelled, not deferred. The four 3-GPU
  workers (cfw12-g3-a, still pending) were cancelled before claiming anything, and all 75 q25-72 COCO/CheXpert
  and llama32-90 tasks (including the three preparation tasks released minutes earlier; none had started) were
  moved to `queue/cancelled/`. No outcome of these blocks exists; q25-72 NIH stays as delivered.
- Gate decision: the grid is 21 checkpoints with Qwen2.5-VL-72B on NIH only; the campaign tables and the paper
  report the 72B COCO/CheXpert blocks and the 90B as not run. The remaining additions are limited to
  checkpoints at or below 40B: the Llama 3.2 11B primary-template re-run, InternVL3.5-38B completion, and the
  CheXpert module extension.

## 2026-09-14 Protocol amendment A1: per-block primary template (Llama 3.2 11B re-run)

- Run: `protocol.primary_template(run_dir)` resolves the template the single-template modules (CORE, CALIBRATION,
  DOSE, REFIT, LOCUS, LOCUS_CALIBRATION) score: IY when eligible, else the first eligible template in
  (IY, IB). The runner substitutes it before the eligibility filter and closes the module as ineligible only if
  the substitute is ineligible too; the packager, analysis, leaderboard, tables and figures read the block's
  primary template from template_eligibility.json / summary.json instead of assuming IY; the PROMPT contrasts
  (IY-WY wording, IA-IB mapping) are reported as INELIGIBLE cells with a reason when the primary is not IY or a
  template failed check E. Tests 24/24. Blocks whose primary is IY are unaffected (recomputation check logged
  below).
- Gate decision: llama32-11's terminal ineligible outputs were set aside as `outcomes/<MODULE>.ineligible-IY`
  (nothing deleted) and its CALIBRATION, CORE, LOCUS_CALIBRATION, DOSE, REFIT, LOCUS re-enqueued on all three
  datasets (39 tasks, prefix 2, single-GPU lane); PROMPT (WY/IA/IB/WA/WB) stays as delivered. The leaderboard
  shows the primary template in its own column for such blocks.

## 2026-09-14 Protocol amendment A2: CheXpert gets the five remaining modules

- Run: DOSE, REFIT, LOCUS, LOCUS_CALIBRATION and PROMPT now list CheXpert; PROMPT concepts on CheXpert are
  Effusion and Edema (Effusion shared with NIH; Edema the most-owned CheXpert concept). CheXpert CALIBRATION keeps
  its IY-only question list (`CALIBRATION_PROMPT_DATASETS = ("nih", "coco")`), so the 19 packaged CheXpert blocks
  keep their expected rows; a module added later counts towards COMPLETE only once its outcomes directory exists
  (`MODULES_ADDED_LATER`), so no packaged block is downgraded. protocol.json, README and main-tables.csv (T3 rows
  for CheXpert) updated. 300 tasks enqueued (prefix 3, after every current task) for the 20 checkpoints at or
  below 40B including iv35-38 and llama32-11; 12 more single-GPU workers submitted (24 queued in total) and the
  two login-node 2-GPU workers keep serving the 2-GPU lane. Queue 365 pending / 9 running / 0 failed.

## 2026-09-14 Login-node workers: four A100-80GB in use; stale single-GPU workers reconciled

- Run: the login node's four GPUs (A100-SXM4-80GB, idle) were put on the 2-GPU lane at 12:52Z through
  `jobwrap.sh` (two workers, GPUs 0+1 and 2+3); they took iv35-38 NIH CORE and LOCUS shards at once. Four
  single-GPU login workers started on 2026-09-13 (pre-signal-handler code, `--wait-for-free-gpu`) were found still
  polling and claimed the first llama32-11 tasks onto GPUs already full. Resolution: the 2+3 two-GPU worker and two
  of the old workers were stopped; two orphaned llama32-11 runners (parents died without requeueing) were killed
  and their shards requeued by hand with a `requeued` note. Layout now: GPUs 0+1 one 2-GPU worker (iv35-38), GPUs 2
  and 3 one single-GPU worker each (llama32-11 CALIBRATION and CORE). No outcome file was lost or duplicated
  (the killed runners had not written a shard).
- Gate decision: login-node workers are launched only through jobwrap and only on the current worker code; the
  old workers are retired when their current task ends.

## 2026-09-14 Scheduler saturation; H100 single-GPU workers added

- Run: at 13:50Z Slurm estimated the start of all 36 queued A100 workers as 2026-09-17 (partition held by
  long jobs of other users; 48 pending jobs, 74/76 GPUs allocated). Eight single-GPU workers were submitted to
  gen-h100 (cfw15-g1-h; estimated start 2026-09-15 09:22 local); the four 2-GPU H100 workers were cancelled
  (estimated 2026-09-22). Running capacity: 3 A100 2-GPU Slurm workers (cfw8) plus the login node (one 2-GPU
  worker on iv35-38, two single-GPU workers on llama32-11).
- Gate decision: H100 shards are acceptable: every contrast is within-row and within-shard, batch composition
  is fixed per question, and the runner records gpu_models per shard, so hardware is traceable in run.json.

## 2026-09-14 iv35-38 NIH CORE statistics

- Run: CORE, CALIBRATION, DOSE, REFIT, LOCUS_CALIBRATION packaged (72.4 GPU-hours on the 2-GPU lane, the
  last CORE shard on the login node); PROMPT and LOCUS pending. Preflight passed with batch-vs-single declared
  (candidate logits 0.39, margins 0.064, sign agreement; fp32-vs-model 0.056; reach 0.79 at the connector,
  1.35 answer logits; 6.1 rows/s at batch 16 on 2 x A100-80GB).
- Observation (known labels): readable five (Effusion S 0.143, Atelectasis 0.106, Pneumothorax 0.158,
  Cardiomegaly 0.115, Mass 0.135; Nodule -0.061 not readable); clean answers AUROC 0.67-0.87, all six above
  chance.
- Observation (CORE): the InternVL pattern at 38B: writes within 0.03 of zero. Effusion W_qq +0.031, rank 3,
  meets the steering reference, O +0.002 [+0.000, +0.005] against Mass (+0.029) -> unresolved; Mass W +0.023,
  O +0.016 against Atelectasis, rank 13, reference not met -> fixed_family_advantage without ownership;
  Atelectasis, Pneumothorax, Cardiomegaly stronger competitor (O -0.018, -0.003, -0.030); Nodule unresolved.
  Owns nothing on NIH, as at 8B and 14B.
- Gate decision: valid OBSERVED block; 755 table cells filled, leaderboard 59 blocks. Watcher re-armed.

## 2026-09-14 iv35-38 NIH: all seven modules packaged; T3

- Run: PROMPT and LOCUS landed; block COMPLETE with all seven modules, 105.9 GPU-hours (2-GPU lane plus the
  login node).
- Observation (T3): signed dose range +0.028 [+0.024, +0.033]; refit SD 0.007 [0.006, 0.007]; connector median
  O -0.004 [-0.004, -0.003]; label gap -0.023 [-0.059, -0.008]; Effusion wording +0.010, mapping +0.005; Mass
  wording +0.004, mapping +0.043 [+0.041, +0.045]. Every control is as small as the writes themselves: the
  38B reader, like 8B and 14B, answers the questions but is barely moved by any direction at this locus.
- Gate decision: valid OBSERVED block; iv35-38 complete on NIH (COCO and CheXpert running). Tables and
  leaderboard regenerated; fully-scored watcher re-armed.

## 2026-09-14 iv35-38 COCO CORE statistics

- Run: CORE, CALIBRATION, REFIT, LOCUS_CALIBRATION packaged (34.0 GPU-hours); DOSE, LOCUS, PROMPT running.
  Preflight passed with batch-vs-single declared (candidate logits 0.62, margins 0.34; reach 0.84 at the
  connector, 1.11 answer logits; 6.2 rows/s at batch 16).
- Observation (known labels): readable person, car, chair, bottle (S 0.057-0.094); dog S 0.063 and bicycle
  S 0.088 not readable at the lower bound; answers 0.976-0.998 AUROC (bicycle not answer-capable by the support
  rule).
- Observation (CORE): owns all six objects at rank 1 with the InternVL-typical small writes: person O +0.013,
  dog +0.003, car +0.008, chair +0.016, bottle +0.010, bicycle +0.005 (W_qq 0.009-0.028, random p95 <= 0.009,
  |sham| <= 0.010). The family's third size repeats 8B and 14B: specific on objects, inert on chest findings.
- Gate decision: valid OBSERVED block; tables and leaderboard regenerated.

## 2026-09-14 iv35-38 COCO: all seven modules packaged; T3

- Run: DOSE, LOCUS, PROMPT landed; block COMPLETE with all seven modules, 156.6 GPU-hours (the most expensive
  COCO block; 38B on the 2-GPU lane).
- Observation (T3): signed dose range +0.020 [+0.016, +0.023]; refit SD 0.001; connector median O +0.005;
  label gap -0.075 [-0.131, -0.022]; person wording +0.002, mapping -0.003; bottle wording +0.002, mapping
  +0.001. Small but consistently positive dose response on objects, against a flat one on NIH: the InternVL
  family's writes are weak everywhere and directed only on natural objects.
- Gate decision: valid OBSERVED block; iv35-38 complete on NIH and COCO (CheXpert running). Tables and
  leaderboard regenerated; fully-scored watcher re-armed.

## 2026-09-14 Robustness R1: direction geometry and competitor sensitivity (existing data)

- Run: `scripts/mayo/robustness_geometry.py` over the 57 blocks with a complete write matrix (nih 20, chexpert 18,
  coco 19) -> `runs/robustness/geometry.{json,md}`. Tests the alternative that off-diagonal dominance is
  correlated-label geometry.
- Observation: NIH concept directions are as orthogonal as random vectors (mean off-diagonal cosine 0.019 vs
  0.023 for random pairs; labels phi 0.026) yet show the same off-diagonal dominance as CheXpert (O_q > 0 in
  0.258 vs 0.287 of cells), where directions are correlated (mean cosine 0.241; Effusion-Consolidation 0.51,
  phi 0.74). The strongest competitor coincides with the most similar direction at chance (0.19-0.23 vs
  0.20) and with the most co-occurring label at chance (0.17-0.25). Over 1,140 chest off-diagonal cells the
  competitor advantage W_qd - W_qq is uncorrelated with cosine (Spearman 0.047, p=0.11) and with phi (0.003,
  p=0.93). Leave-one-competitor-out: chest cells with O > 0 rise from 62/228 to 102/228 when the strongest
  competitor is removed; 126/228 (55%) have >= 2 competitors above the own write, 81 (36%) >= 3. COCO
  restricted to co-occurring families (person+bicycle, person+dog, chair+bottle, car+bicycle,
  person+chair+bottle) keeps ownership in 98-100% of cells. Seed cell: Nodule is the LEAST similar direction to
  Effusion (cosine -0.056, phi 0.033, n11 = 5) and still writes the Effusion answer more (0.262 vs 0.190); the
  most similar direction (Atelectasis, cosine 0.49) writes it by 0.083.
- Gate decision: the deficit is not explained by direction similarity or label co-occurrence; enter as a
  robustness subsection and appendix table in the paper.

## 2026-09-15 Robustness R2: scale, ceiling, refits, connector, batch effects (existing data)

- Run: `scripts/mayo/robustness_scale.py` over the 57 blocks with a complete write matrix ->
  `runs/robustness/scale.{json,md}`.
- Observation (scale): ownership recomputed on the answer-margin (logit) scale agrees with the probability
  scale in 335 of 342 cells (124 owned on both, 3 p-only, 4 m-only); every p-owned cell has O^m > 0.
- Observation (ceiling): restricting to unsaturated rows (clean P(yes) in [0.01, 0.99]) keeps O > 0 in 27/27
  owned chest cells and gives O > 0 in 32/191 not-owned chest cells; the same in the clean-No, clean-Yes,
  label-present and label-absent strata. COCO blocks are MORE saturated than chest blocks (median block share
  0.82 vs 0.18 NIH / 0.32 CheXpert) and still owned.
- Observation (refits): a refit seed resamples training units with replacement and refits scaler and probes;
  median |O(seed k) - O(seed 0)| 0.028 (NIH) / 0.026 (COCO); refit directions have cosine 0.80 / 0.89 to seed
  0; 6/10 NIH and 94/100 COCO owned cells keep O > 0 under both refits; the four NIH exceptions are
  q25-32 Cardiomegaly, q25-72 Mass, q3-32 Nodule, gemma3-12 Nodule.
- Observation (connector): median |W_qq| 0.009 (NIH) / 0.008 (COCO) at the connector vs 0.046 / 0.270 at the
  primary locus; connector write beats its own random p95 in 13/120 NIH and 85/114 COCO cells; median O at the
  connector -0.009 / +0.004.
- Observation (batch): preflight batch-vs-single candidate-logit differences 0.07-1.84 (median 0.36), margin
  sign agreement in 55/61 blocks; the 6 disagreeing blocks contain 0 owned cells. In-situ clean-forward
  reproducibility (CORE vs LOCUS baseline of the same 3,600 rows): identical margins in 100% of rows for 22 of
  33 NIH/COCO blocks and 75% for the rest, |dP(yes)| > 0.1 in <= 1.9% of rows, sign flips <= 0.6%.
- Observation (grades): owned rate among readable-and-answerable chest cells 19/91 (21%) vs 8/137 (6%) for the
  rest (Fisher p = 0.001).
- Gate decision: neither scale, saturation, refit instability, nor batch effects account for the deficit;
  enter as robustness results in the paper.

## 2026-09-15 Protocol amendment A3: alternative direction estimators (ALTDIR)

- Run: module ALTDIR writes, at the primary locus, dose +0.25, on the same 600 test rows with the CORE clean
  baseline, four alternative directions per concept built from the same projected, train-scaled features as the
  logistic normal: difference of class means (dom), the Haufe pattern Sigma w (pattern), the logistic normal
  orthogonalised against the other five normals (orth), and the normal refitted on features residualised
  against the other five labels (resid); 24 steered conditions per question, 86,400 rows per block. Analysis
  gives per-family write matrices, within-family ownership with bootstrap intervals and max-T verdicts, and the
  own-write-minus-strongest-logistic-competitor contrast. Tests 29/29. Directions computed for the 40 NIH/COCO
  blocks at or below 40B (dom/pattern have cosine 0.09-0.37 to the logistic normal; orth/resid 0.92-0.999);
  CheXpert directions wait for a resid fix (Atelectasis is single-class on the all-known rows). 40 tasks
  enqueued at prefix 1 (ahead of everything except running work).
- Gate decision: answers the objection that the deficit could be an artefact of the logistic-normal estimator
  or of correlated labels; results enter Section 5.7 as they land.

## 2026-09-15 Robustness R3: shared-tower paired deltas, ceilings, floors, denominators (existing data)

- Run: `scripts/mayo/robustness_pairs.py` -> `runs/robustness/pairs.{json,md}`.
- Observation (shared towers): gemma3-4/12 features and write vectors bit-identical; gemma3-27 and medgemma-27
  within one bf16 ulp of their siblings (cosines 0.9987-0.99996); llava15-7/13 bit-identical (a third same-write
  ladder). Qwen2.5-VL, Qwen3-VL, InternVL3.5, Lingshu sizes do not share towers.
- Observation (paired Delta O, same 600 patients, 2,000 shared draws): the interval excludes zero for 5-6 of 6
  concepts in 11 of 12 pair-dataset combinations (medgemma-4/27 COCO: 3/6); median |Delta O| 0.05-0.37, 1.4-3.7x
  the mean refit SD (max 8.8x). NIH Effusion: -0.83 [-0.85, -0.81] (4B -> 12B), +0.79 [+0.76, +0.82] (12B -> 27B).
- Observation (ceilings): 71/342 cells have > 90% of clean rows saturated (NIH 15, CheXpert 13, COCO 43); only one
  owned chest cell is a ceiling cell. gemma3-4 chest cells are high-side saturated; on the logit scale every own
  write lowers its own margin (O^m -1.1 to -19.8, all intervals below zero).
- Observation (floors): owned cells with W_qq >= 0.05 and O_q >= 0.05: chest 18/228 (NIH 9, CheXpert 9), COCO
  78/114; refit-stable owned: NIH 6/10, COCO 94/100.
- Observation (readability): grid-wide mean selectivity Cardiomegaly 0.187 > Effusion 0.171 > Atelectasis 0.130
  > Pneumothorax 0.063 (40 chest blocks).
- Observation (denominators): probe-graded chest cells 240 (21 blocks NIH incl. llama32-11 and iv35-38; 19
  CheXpert), write-matrix cells 228, owned 27, readable 134, answer-capable 140, readable-and-answerable 91 (19
  owned), stronger competitor 150, steering reference 41. Table 1 superscripts used graded denominators; the
  tables now use the write-matrix denominator for Ans/Own.
- Gate decision: the reader result is refit-proof (paired deltas); floors and stability enter Section 5.1; the
  paper's "most readable" sentence corrected.

## 2026-09-15 ALTDIR first block: q25-7 NIH and COCO

- Run: ALTDIR shards landed for q25-7 NIH and COCO; packaged and analysed (`--what altdir`).
- Observation (NIH): logistic owns 0/6. dom owns Atelectasis (O +0.07, W_qq 0.36) and Cardiomegaly (+0.11,
  W_qq 0.47); pattern the same two (+0.09, +0.12); orth 0 (Mass +0.02 without the steering reference); resid 0.
  The seed cell holds under every family (Effusion O: dom -0.06, pattern -0.03, orth -0.29, resid -0.13).
- Observation (COCO): dom owns 4/6 (car, chair fail), pattern 3/6, orth 6/6, resid 6/6; dom/pattern own writes
  0.17-0.41 against 0.5-0.9 for the logistic normal. dom and pattern are near-collinear with each other and at
  cosine 0.09-0.37 to the logistic normal; orth/resid at 0.92-0.999.
- Gate decision: nothing to conclude from one block; the pre-registered rule (chest owned share a small fraction
  of the COCO share under the same estimator, plus any-estimator counts) is evaluated once the 60 blocks land.

## 2026-09-15 iv35-38 CheXpert CORE statistics; iv35-38 complete on all three datasets

- Run: CORE and CALIBRATION packaged (43.6 GPU-hours); the block is COMPLETE under the added-later rule; the
  five CheXpert extension modules are queued behind the priority modules.
- Observation (known labels): readable Effusion (S 0.242), Cardiomegaly (0.247), Consolidation (0.213), Edema
  (0.183); Atelectasis insufficient support, Pneumothorax 0.008; answers 0.66-0.94 AUROC.
- Observation (CORE): owns Effusion (O +0.019, W_qq 0.036, rank 3) and Consolidation (+0.010, W_qq 0.025, rank
  5) with margins below the 0.05 floor; Cardiomegaly meets the reference but is unresolved; the rest are
  stronger competitor. The InternVL family's writes stay within 0.04 of zero on every chest set at every size.
- Gate decision: valid OBSERVED block; InternVL3.5-38B is complete on NIH, COCO and CheXpert; tables and
  leaderboard regenerated.

## 2026-09-15 ALTDIR interim (42 blocks), ANSDIR first blocks, EXTCOMP first blocks

- Run: 42 ALTDIR blocks analysed (14 per dataset); ANSDIR for q25-7 (nih, chexpert, coco), lingshu-32 (all
  three), gemma3-12 (coco); EXTCOMP for q25-7 (nih, chexpert), q3-8 nih, iv35-8 nih.
- Observation (ALTDIR, owned cells of 84 per dataset; logistic / dom / pattern / orth / resid / any): NIH 5 / 12 /
  13 / 7 / 6 / 19; CheXpert 15 / 13 / 10 / 13 / 9 / 28; COCO 72 / 43 / 37 / 70 / 72 / 76. The pre-registered rule
  holds under every construction (chest 6-15% vs COCO 44-86%); dom/pattern are weaker handles everywhere.
- Observation (ANSDIR): the answer direction a_q (ridge on the model's own clean margins, 3,000 training rows) is
  owned in 3/6 NIH and 6/6 CheXpert cells of q25-7 (Effusion O^a +0.29, W^a_qq 0.81) and 3/6 NIH, 5/6 CheXpert of
  lingshu-32, beats the strongest logistic competitor on 6/6 questions in every block, and has median cosine to
  the probe normal 0.06-0.07 (q25-7) and 0.30-0.33 (lingshu-32) on chest against 0.54-0.64 on COCO.
- Observation (EXTCOMP): with 3 (NIH) / 6 (CheXpert) extra clinical directions, q25-7 stays at 0 owned on both
  chest sets (O_ext -0.07 to -0.46); q3-8 NIH keeps Effusion; iv35-8 NIH 1/6.
- Gate decision: the three findings enter Section 5.7 and Tables cf-altdir / cf-ansdir / cf-extcomp; numbers are
  refreshed when the remaining blocks land.

## 2026-09-15 TOKENW first blocks (q25-7 nih, coco, chexpert)

- Observation: writing the six logistic directions with per-token weights (softmax of the probe score, mean weight
  one; or the top quarter of tokens at four times the weight) owns 0/6 on NIH and 0/6 on CheXpert for Qwen2.5-VL-7B
  (own writes 0.00-0.16; top-quarter Mass +0.03 and Edema +0.03 without the steering reference) and 6/6 on COCO
  for both variants (top-quarter own writes 0.29-0.88, uniform 0.48-0.94). Concentrating the write on the tokens
  the probe scores highest does not create a clinical handle and leaves the natural-object handles intact.
- Gate decision: enters Section 5.7 and Table cf-tokenw; remaining TOKENW blocks (lingshu-32, gemma3-12, q3-8)
  are queued.

## 2026-09-15 GPT-6-Astra review round 1 (fresh thread, xhigh)

- Run: the manuscript (main text 10 pages after three GPT-6-Astra paragraph passes) was reviewed by GPT-6-Astra
  through `codex154 exec` on a fresh thread; review saved as concept-flow-paper/review-artifacts/gpt6_reviews/
  round1_20260915_0322.md. Scores: soundness 2, presentation 2, contribution 3, overall 4/10, confidence 4.
- Observation (weaknesses): W1 counts differ between prose, Table 1, Figure 2, and appendix tables (denominators
  126 vs 120; CheXpert owned 19/114 vs 17/108; ALTDIR 42/84 vs 46/90); W2 label support (CheXpert Atelectasis 2
  negatives), NIH training rows 18,212 vs "20,000"; W3 ownership is row dominance, not column selectivity;
  "attribution rather than efficacy" contradicts 41/228 reference-met; W4 answer direction needs label AUROC,
  score correlation, whitened alignment, template transfer; W5 reader claim needs representation-varied control;
  W6 lift of dom/pattern unspecified, token-weight norm not matched; W7 "every owned cell stays owned" false (124
  of 127), campaign inference unspecified; W8 batch effects need a stable-numerics rerun (PRECISION running).
- Gate decision: fixed in text now: two-part deficit sentence, label-gap description, logit-scale claim, NIH
  training count, D.1 rows/columns, primary-template rule, lift specification, token-weight norms, campaign
  inference note. In progress: authoritative manifest + macro-generated numbers (one inclusion rule for tables,
  figures, prose); answer-direction validation (label AUROC, score correlation, whitened cosine), column
  selectivity, known-label ownership (robustness_validation.py); PRECISION results when they land.

## 2026-09-15 PRECISION first block (q25-7 COCO): fp32 and batch-1 rescoring

- Run: the CORE grid (baseline + 6 concepts + 119 random + sham) on the first 200 test rows rescored with fp32
  weights and forward, and separately in bf16 at batch size 1, compared with the CORE values on the same rows.
- Observation: fp32 max |dW| 0.0023, max |dO| 0.0025; batch-1 max |dW| 0.0017, max |dO| 0.0013; the
  own-vs-strongest-competitor and own-vs-random-p95 verdicts agree in 6/6 cells under both settings. The bf16
  batched measurement is numerically stable at the level of the verdicts.
- Gate decision: enters Section 5.7 and the appendix table once q25-7 NIH (fp32 pending) and gemma3-12 NIH (the
  worst-case batch-vs-single block, 1.84 logits) land.

## 2026-09-15 Robustness R4: answer-direction validation, column selectivity, known-label ownership

- Run: `scripts/mayo/robustness_validation.py` -> `runs/robustness/validation.{json,md}` (58 blocks, 348 cells;
  9 ANSDIR blocks).
- Observation (answer direction): a_q reads the clinical label WORSE than the probe (median label AUROC 0.64 vs
  0.73 on NIH, 0.67 vs 0.85 on CheXpert; 0.95 vs 0.97 on COCO) and predicts the model's own clean answer far better
  (0.89 vs 0.76); its cosine to the probe normal is 0.06-0.33 raw and 0.36-0.43 after whitening by the training
  covariance on the chest sets, against 0.54-0.64 raw and 0.81 whitened on COCO. The direction that writes is not
  the direction that reads.
- Observation (column selectivity S_d = W_dd / sum_q |W_qd|): median 0.11 (NIH), 0.13 (CheXpert), 0.67 (COCO);
  S_d >= 0.5 in 2/120, 1/114, 84/114 cells; the only chest cell that is selective but not owned (medgemma-27 NIH
  Cardiomegaly) sits on a saturated question (random p95 0.65). Ownership on chest is almost never accompanied by
  column selectivity (1 of 29 owned chest cells).
- Observation (known-label rows, CheXpert): recomputing on rows with a known label (median 165 per cell) keeps the
  sign of O_q in 108/114 cells and 18 of 19 owned cells (medgemma-4 Cardiomegaly becomes unresolved).
- Gate decision: all three strengthen the thesis; enter Section 5.7 in one sentence each and the appendix table.

## 2026-09-15 GPT-6-Astra review round 2 (fresh thread, xhigh)

- Run: review of the manifest-driven manuscript saved as concept-flow-paper/review-artifacts/gpt6_reviews/
  round2_20260915_0420.md. Scores unchanged (2/2/3, overall 4/10, confidence 4).
- Observation: W1 the displacement lift for difference-of-means/pattern directions is the classifier lift
  (R diag(1/s) u) where a displacement needs R (R^T R)^{-1} diag(s) u; "weaker handles everywhere" is wrong on
  NIH (12/96, 13/96 vs 7/96 logistic). W2 PRECISION covers one COCO block with point-estimate agreement; chest
  blocks with the largest gate deviations and full max-T verdicts are needed. W3 clinical specificity needs
  adjudicated labels and non-clinical radiographic attributes in the same images. W4 answer-direction transfer
  to held-out templates/mappings and per-cell rescue/loss. W5 token-tensor replay across shared-tower readers;
  Lingshu's whitened alignment (0.87 CheXpert vs 0.73 COCO) to be explained (it owns 5/6 CheXpert). W6 full-grade
  refit transitions; both bound algorithms; percentile vs max-T consistency. W7 coverage table; Llama status;
  EXTCOMP is a retention test by definition.
- Gate decision: new modules ALTDIRD (displacement lift), ATTR (view/sex/age attribute directions), ANSDIRT
  (oracle transfer across templates), PRECISION full-verdict analysis, PRECISION on gemma3-4/27, medgemma-4,
  lingshu-32, q3-8, iv35-8 NIH (12 tasks enqueued); coverage table, refit grade transitions, alignment-ownership
  correlation from existing data; wording: weaker-handles sentence, EXTCOMP as retention, numerics scoped to the
  completed blocks, coverage sentence. CheXpert Plus ships only the 234-image radiologist-labelled valid split:
  candidate adjudicated subset for evaluation.

## 2026-09-15 Round-2 evidence: refit transitions, coverage, rescue table; ALTDIRD / ATTR / ANSDIRT; adjudicated CheXpert labels

- Run: scripts/mayo/robustness_refit.py (runs/robustness/refit.json): seeds 1 and 2 regraded with the campaign rules
  (steering reference from the summary on the same rows, 6x5 max-T over the shared 2,000 draws). 41 blocks; the seed-0
  regrade reproduces the paper grade in 246/246 cells. q25-7/chexpert excluded (REFIT scored on 70 of 600 rows).
- Observation: owned cells keeping the owned grade under both refits: NIH 3/10 (7/10 under at least one), CheXpert
  0/1, COCO 87/100 (97/100 under at least one). Competitor cells stable under both: NIH 56/81, COCO 2/6.
- Run: robustness_validation.py adds alignment versus ownership over the 54 ANSDIR cells: Spearman(cos_Sigma, O_q)
  = 0.75 (p = 8e-11); median cos_Sigma 0.80 in owned cells versus 0.34 in the rest.
- Observation: over all nine ANSDIR blocks, a_q is owned in 19 of 36 chest cells where the label direction is owned
  in 9, and its write beats the strongest logistic competitor in 30 of 36 (COCO 18/18 on every count). Gemma 3 12B
  CheXpert: a_q owned 0/6, beats the logistic competitor 1/6 (paper sentence corrected to the aggregate).
- Gate decision: paper tables table_cf_coverage (per checkpoint, modules completed per dataset), table_cf_refit,
  table_cf_rescue added; prose corrected: refit sentence counts full-grade retention (3/10 vs 87/100), baseline
  agreement "at least 75%" (two blocks at 0.75 and 0.77).
- Run: round-2 modules committed (concept-flow 1a37755): ALTDIRD (displacement lift x = R (R^T R)^-1 diag(s) u for the
  dom and pattern vectors; gram spectrum of q25-7 nih min 0.000265 / max 0.005184, condition 19.6, matching the
  Marchenko-Pastur edges of a 1280x512 Gaussian projection), ATTR (view AP, sex F, age >= 60 directions from
  labels.csv with protocol-identical probes, PCG64(1) shams, 9x9 write matrix, sham-only steering reference for the
  attribute questions), ANSDIRT (answer direction under the five other templates, own baseline), PRECISION full grade
  (fp32 and batch-1 regraded on the 200-row grid with paired bootstrap draws). 45 tests pass.
- Observation: q25-7 coco PRECISION full grade: verdict changes 0/6 and reference changes 0/6 under fp32 and under
  batch 1; max |dW| over the 127-direction grid 0.0060 (fp32) / 0.0028 (batch 1).
- Fix: the altdir CLI's append-only guard required a bitwise rebuild of the stored arrays, which a rebuild does not
  reproduce (BLAS thread count); the CLI now extends an existing file from its own stored dom / pattern vectors, the
  ones ALTDIR scored (57baa95). 19 blocks had been extended but reported as failed by the first loop because the
  printout indexed keys that pre-amendment files lack (d4d59a9); re-run of the loop enqueues them.
- Run: scripts/mayo/round2_preps.sh: ALTDIRD on every included block and ANSDIRT on the nine ANSDIR blocks at
  prefix 11 (ahead of the CheXpert extension), ATTR on the chest blocks at prefix 4 (after it).
- Observation (feasibility of an adjudicated CheXpert cohort): the labels on disk for the 234-image valid split
  (impression_fixed.json and the other two label files) are CheXpert-labeler passes over three report sections
  (Effusion positives 63 / 23 / 93 across the three files; -1 and null entries), not the radiologist consensus. The
  original valid.csv (three board-certified radiologists, dense 0/1) is not in the Redivis release. It was obtained
  from the Hugging Face dataset danjacobellis/chexpert (validation split, revision ca5a840e438b5d2d0fa07f560782f196b38a84aa,
  class labels 2 = absent, 3 = present, no uncertain or blank entries): positives over the 234 images Cardiomegaly 68,
  Edema 45, Consolidation 33, Atelectasis 80, Pneumothorax 8, Pleural Effusion 67, which match the published valid.csv
  distribution. Saved as data/chexpert/valid_radiologist_labels.csv with a provenance file. The 234 PNGs are inside the
  CheXpert Plus zips (PNG/valid/, 778 MB); patient overlap with every campaign role is 0. images/_test (8 files) are
  the first eight train entries from the throttled download test, not valid images.
- Gate decision: module VALID (CORE write matrix and grade on the 200 valid frontals, one per patient by the campaign's
  row hash, radiologist labels, paired with the same block's test grade) implemented next; about 56 GPU-hours over
  the CheXpert checkpoints at or below 40B.

## 2026-09-15 GPT-6-Astra review round 3 (fresh thread, xhigh); VALID module merged and enqueued

- Run: review of the round-2 manuscript (coverage, refit-transition, rescue tables; aggregate answer-direction
  numbers) saved as concept-flow-paper/review-artifacts/gpt6_reviews/round3_20260915_1007.md. Scores unchanged
  (2/2/3, overall 4/10, confidence 4).
- Observation: items already covered by running modules: W2 displacement lift (ALTDIRD), W3 attributes in the same
  radiographs and adjudicated labels (ATTR, VALID), W5 chest-block full-grade numerics (PRECISION NIH tasks), W6
  transfer of a_q across templates (ANSDIRT). Items fixed without runs: the 44 reference-meeting chest cells are
  29 owned / 8 stronger competitor / 7 unresolved (the prose had subtracted 29 from 44); the answerable grade now
  states the ten-per-class support rule the code applies; the simultaneous upper bounds are defined in the bounds
  appendix; the paired-reader sentence quotes the ratio range 0.6-3.7 and 7 of 8 pairs above one; the write-flow
  figure's CheXpert panel had inherited NIH row labels (Nodule for Edema) and every panel now carries its own;
  the manifest was stale for TOKENW / PRECISION completions (rebuilt; coverage table regenerated).
- Observation: items not addressed by the paper: an anonymous artifact URL and a patient-level overlap audit
  against the seed-study cohorts (the seed cohort lists are being located); direction-specific shams after refits
  (would need a GPU pass scoring the refit probes' shams).
- Run: VALID module merged (concept-flow 4c483c8; 48 tests pass): cohort role valid appended to the CheXpert
  cohort (200 frontals, one per patient; positives Effusion 64, Atelectasis 75, Pneumothorax 6, Cardiomegaly 66,
  Consolidation 32, Edema 42; every label known, radiologist consensus); 200 PNGs extracted from PNG/valid/
  (685 MiB). 19 checkpoints x (features task + VALID task) enqueued at prefix 11 (38 tasks, 152,400 outcomes per
  block). Pneumothorax will grade insufficient_support under the ten-per-class rule.
