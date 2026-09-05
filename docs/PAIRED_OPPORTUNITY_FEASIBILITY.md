# Paired-input opportunity: metadata feasibility

Run: read-only NIH audits on `asimov1` at 2026-09-05 02:38 and 03:17 UTC counted eligible patients and image pairs after excluding all five Qwen intervention cohorts. The first audit examined the sampled manifest, with a second grouping-based calculation reproducing its pair counts. The second used original source labels and the available PNG inventory under the same patient-split rule.

## Observation

The full available NIH pool contains **4,480 test patients and 13,874 images absent from the entire original manifest**. Holding view, sex, exact age and all 13 non-target disease labels fixed permits **124 Mass patients / 593 pairs** and **120 Consolidation patients / 1,205 pairs**. All corresponding files are available. This provides a fresh-patient basis for the next opportunity registration.

### Sampled manifest

The sampled manifest contains 1,659 test patients and 5,343 images. The five mutually disjoint exclusion cohorts contain 167, 400, 400, 50 and 200 patients, respectively: 1,217 in total. The remaining sampled pool contains **442 patients and 890 unique images**.

Each table entry is **patients / positive-negative image pairs**. Pair counts enumerate combinations; choosing one pair per patient limits the sample size to the patient count.

| Nested matching condition | Mass | Consolidation |
|---|---:|---:|
| Same patient | 25 / 169 | 1 / 8 |
| + same view | 19 / 93 | 1 / 5 |
| + same sex | 19 / 93 | 1 / 5 |
| + exact age | 12 / 80 | 1 / 5 |
| + eight other disease labels | 8 / 27 | 1 / 1 |
| + `no_finding` | 2 / 6 | 0 / 0 |

The nine available disease labels are Effusion, Atelectasis, Pneumothorax, Consolidation, Cardiomegaly, Edema, Infiltration, Mass and Nodule. The strict final row matches all nine other clinical fields, including `no_finding`, while permitting the target to differ. In `src/build_manifest.py`, `no_finding` comes directly from the source `No Finding` tag. It is distinct from the eight non-target disease labels and is not reconstructed as their complement. Its handling therefore requires an explicit clinical interpretation when defining pairs.

Exact within-patient agreement on view, sex, age and all ten clinical fields supplies 56 patients with 210 unordered unchanged-label image pairs, or 94 image-disjoint pairs. Of these patients, 55 allow Mass-negative pairs and two allow Mass-positive pairs; one patient contributes both statuses. All 56 allow Consolidation-negative pairs. Five of the twelve age-matched Mass transition patients also allow an image-disjoint unchanged-label pair; both strict all-field Mass patients do.

Consolidation's sampled-pool scarcity follows the patient exclusions: 51 mixed-label patients remained after the first three intervention cohorts, and the accepted closure cohort used 50 of them.

### Original NIH image pool

The original metadata contains 112,120 images from 30,805 patients. All 112,120 PNG filenames are available and match metadata. The authoritative `patient_split` rule in `src/build_manifest.py` assigns 6,139 patients / 22,229 images to test. Applying the five-cohort exclusion union leaves 4,922 patients / 15,311 images. Of these, 442 patients already occur in the sampled manifest and have 1,437 available images; the other 4,480 patients are absent from that manifest altogether, including its training rows.

The table below uses only the 4,480 entirely manifest-absent patients. Entries are **patients / positive-negative pairs**; stages are cumulative.

| Matching condition | Mass | Consolidation |
|---|---:|---:|
| Same patient | 252 / 7,448 | 242 / 14,407 |
| + same view and sex | 237 / 4,497 | 235 / 10,239 |
| + exact age | 213 / 3,188 | 221 / 7,027 |
| + all 13 other disease labels | 124 / 593 | 120 / 1,205 |
| + `No Finding` status | 60 / 210 | 73 / 408 |

Full source labels add Emphysema, Fibrosis, Hernia, Pleural_Thickening and Pneumonia to the nine retained diseases. The source has zero rows combining `No Finding` with a disease. Therefore the last tier restricts both endpoints to patients' images carrying another disease; the preceding tier also admits a target-only positive paired with a `No Finding` negative. Among its pairs, 383 Mass and 797 Consolidation pairs have a `No Finding` negative. Matching this aggregate status changes the clinical population rather than simply adding an independent nuisance control.

Raw metadata supplies `Patient Sex`, `View Position`, `Patient Age`, `Follow-up #`, original image width/height and x/y pixel spacing, all populated. Follow-up indices range from 0 to 183, and ages from 0 to 95. These fields make a richer prospective pair receipt possible than the sampled manifest alone.

## Gate decision

Metadata availability is `OBSERVED`. The existing original NIH image pool supports a fresh-patient opportunity registration for either route, with all 13 non-target disease labels available for matching. Use the entirely manifest-absent test-patient population as the preparation basis. These are availability ceilings, not power estimates or selected cohorts.

The existing Mass confirmation remains `RUNNING` under its registered protocol. Geometry preparation is in `docs/RESEARCH_PLAN.md#paired-opportunity-preparation`; concept and prompt routing await the accepted terminal confirmation result.

## Next step

Does the matched, fresh-patient cohort provide a positive paired behavioral opportunity under the selected prompt? After the Mass confirmation is accepted, register its selected concept/prompt, one-pair-per-patient selection, full source-label matching, `No Finding` handling, opportunity statistic and budget before model evaluation.

## Limitations

The sampled manifest omits five source disease labels, follow-up indices, dimensions and pixel spacing; the extension can retain them from the raw source. The source has no acquisition dates, elapsed-time column or scanner/protocol identity. Follow-up indices alone do not establish elapsed clinical time. Pairing concerns distinct images with source-label agreement, and metadata agreement does not isolate a lesion-specific image change. File availability and label counts do not measure paired behavioral opportunity or statistical power.

## Evidence

Both committed utilities completed with exit status 0 at 03:33 UTC from `b8145c601dee787f3d67d42ef49209191e6296a4`. Their output is retained in `docs/evidence/paired-opportunity-metadata.json`. Every table above reproduced; all 200 active calibration patients were covered by the existing exclusion union. Cross-source agreement passed for all 26,229 manifest rows and 393,435 values across 15 fields.

Manifest: `/home/qingchan/data/concept-flow/datasets/nih-chestxray14/manifest.csv`.

Original labels: the same dataset directory's `Data_Entry_2017_v2020.csv`; image inventory: `png/images`. The second audit verified stored manifest patient IDs and splits against the raw labels and the existing `concept-flow-v1` split function.

The accepted asset receipt binds the staged 26,229-row manifest to `data/manifest.csv` in source commit `5ccdd6fd3903f899e76cc06728f8ff4ee14a9544`. `scripts/server/stage_first_gate_assets.py` copies those committed rows and inserts the resolved `image_path`; it preserves their demographics and labels. The original-source audit checks the retained fields against the v2020 labels before reporting extension availability. Active Mass calibration uses the last 200 ownership rows, already contained in the ownership patient exclusion; the audit makes this coverage explicit.

Exclusion receipts, relative to `/home/qingchan/data/concept-flow/runs/`:

- `20260903T000321Z-caaae3ef346d-qwen-full/artifacts/intervention-summary.json`: `eval_row_ids`, mapped to patient IDs through the manifest.
- `20260903T103508Z-456c81bad460-7a3edc4b/artifacts/registered-rows.json`: `row_ids`.
- `20260904T125710Z-a3bd883540eb-causal-ownership/artifacts/registered-rows.json`: `row_ids`.
- `20260904T162317Z-8a55a4c2f6b6-consolidation-closure/artifacts/registered-pairs.json`: `pairs`, including `patient_id`, `positive_row_id` and `negative_row_id`.
- `20260905T015725Z-1c9820d7b576-mass-prompts/artifacts/registered-rows.json`: `row_ids`.

From the repository root, the sampled-pool audit is reproducible with the fixed environment and `python -B -m scripts.audit_paired_opportunity --manifest /home/qingchan/data/concept-flow/datasets/nih-chestxray14/manifest.csv --runs-root /home/qingchan/data/concept-flow/runs`. Its output contains counts and provenance, with patient exclusions applied to every image.

The original-pool audit uses `python -B -m scripts.audit_nih_paired_pool --dataset-root /home/qingchan/data/concept-flow/datasets/nih-chestxray14 --runs-root /home/qingchan/data/concept-flow/runs`. Both utilities read metadata and print JSON.
