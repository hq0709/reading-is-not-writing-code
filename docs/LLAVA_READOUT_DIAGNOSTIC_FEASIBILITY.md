# LLaVA readout diagnostic: validation-patient feasibility

## Run

Read-only metadata audits on `asimov1` on 5 September 2026 used the original NIH labels, the accepted manifest and the accepted validation pilot's twenty control assignments. No model was loaded and no new image outcome or activation was read.

## Observation

The original manifest contains 8,297 patients. Applying `src/build_manifest.py:patient_split` with its existing `concept-flow-v1` rule to patients absent from the entire manifest yields 2,223 validation patients. For each patient, the lexicographically smallest original `Image Index` defines the representative image before attaching its disease labels. These representatives include 77 Effusion-positive and 2,146 Effusion-negative images. All 2,223 files exist in the accepted PNG inventory.

The representatives cover 36 `view_AP × sex_M × age-decade` types. Every type is covered by all twenty accepted assignment maps, so a frozen-control replication has complete label support without extending the maps or fitting new controls. A selected cohort's class counts and numerical validity still require its own prospective receipt.

After fixing the natural 700-index / 700-donor design and seed `20260916`, a metadata-only allocation check gives 25 positive / 675 negative index patients and 22 positive / 678 negative donors. The smallest class among the twenty control assignments has 153 index patients and 133 donors. The other 823 patients remain outside the diagnostic. Index endpoint identities are `19570` / `00019570_000.png` and `28163` / `00028163_000.png`; donor endpoints are `15575` / `00015575_000.png` and `12195` / `00012195_000.png`.

This population is patient-disjoint from every original training, validation and test row, including the pilot's 700 calibration patients, sixteen preflight patients and protected 100 write patients. The separate paired-opportunity extension used the fixed test split; it remains disjoint from this validation population.

## Gate decision

Metadata availability is `OBSERVED`. The source pool supports a fresh validation-only readout diagnostic and simultaneous frozen-reader replication. Availability does not establish diagnostic power or model capability.

## Next step

Can the real-image advantage change under fixed wording and answer-encoding contrasts? Lock the sample allocation, prompts, independent donor controls, scoring and inference before evaluating any of these images with the VLM.

## Evidence

- Server checkout: clean `bafd1d0` at audit time.
- Original labels: `/home/qingchan/data/concept-flow/datasets/nih-chestxray14/Data_Entry_2017_v2020.csv`.
- Accepted manifest: the same directory's `manifest.csv`.
- Image inventory: the same directory's `png/images`.
- Frozen control maps: `prepared.json` from immutable run `20260905T093108Z-42a43207c848-llava-validation`.
- Read-only audit outputs in project task `01a06ec5-fec9-7241-b066-72c1bc5632b8`: chunks `5c798f`, `ba9f3e` and fixed-allocation check `f86f22`.
