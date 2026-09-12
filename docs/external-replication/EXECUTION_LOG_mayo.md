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
