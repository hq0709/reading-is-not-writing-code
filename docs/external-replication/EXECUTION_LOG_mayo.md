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
