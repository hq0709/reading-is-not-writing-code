# Research plan

## Question and registered claim

Do medical vision-language models use the clinical information that can be decoded from their internal representations?

Primary falsifiable claim: under a capacity-controlled, patient-split evaluation, behavioural utilisation is substantially lower than linear decodability across architectures, and the locus of peak decodability does not reliably predict the locus of selective causal influence.

Falsification: the claim is rejected if completed, valid intervention sweeps with corrected locus hooks and preregistered activation-scale doses show that decodability rank reliably predicts selective behavioural effects across the registered model/concept cells, with uncertainty excluding the null relationship. A true negative is retained as `OBSERVED`; metrics, doses, loci, or control identities are not changed after seeing results to rescue the headline.

## Method identity and forbidden rescues

The method identity is fixed: patient-level splits; the same capacity-matched linear readout across loci; repeated type-to-random-label controls; trained-versus-random initialisation floors; explicit nuisance probes; bidirectional, equal-norm intervention controls; preregistered dose scale; bootstrap uncertainty; and immutable run evidence. Forbidden rescues include replacing AUROC/selectivity after inspection, removing failed cells, changing prompts only for unfavourable concepts without a registered sensitivity analysis, adding a stronger decoder, changing the intervention vector family, or relabelling invalid runs as scientific negatives.

## Current evidence boundary

The repository already contains released observations for controlled decodability, utilisation, planted-ground-truth tests, direction mismatch, and an ISIC second modality. These remain governed by `docs/02_RESULTS.md` and `docs/10_RESULTS.md`. They are not reclassified here. `docs/10_RESULTS.md` also records invalid or incomplete intervention evidence, including an off-forward-path LLaVA `vis.last` hook, incorrect dose scaling, missing repeated control draws, and absent bootstrap intervals.

## First hard gate

The first hard gate is `llava-effusion-vislast-reltoken`. It uses `llava15_7b`, NIH ChestX-ray14 with the existing patient-level train/test split, concept `Effusion`, and the actual final CLIP vision block (`encoder.layers.22`) exposed as `vis.last`. Re-extract that locus after a hook test proves it is on the forward path. Fit the fixed 512-dimensional logistic readout (`C=1`, seed 0) and report AUROC/selectivity with 2,000 patient-level test bootstrap resamples. Repeat the type-to-label control over seeds 0–19 and report `real - mean(control)` plus the control spread.

The decisive intervention uses `src/intervene.py` with `alpha-mode=reltoken`, alphas `-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1`, control alphas `-1,-0.5,-0.25,0.25,0.5,1`, 20 random directions, seed 0, and 200 held-out test images. The primary causal metric is the maximum signed change in mean `P(yes)` at a registered nonzero alpha, compared at the same alpha with the 5th–95th percentile of random directions and the sham/unrelated directions. A selective cell must leave that random interval in the concept-consistent direction and exceed the maximum absolute sham effect; dose monotonicity over `[-0.5,0.5]` is reported, not used to move the threshold. Cost is GPU wall time from the immutable receipt.

The gate is `READY` only when hook, bootstrap/control, intervention, receipt, and read-only Claude review evidence all pass. A hook/measurement defect is `FAILED`. A valid cell that does not meet the selective threshold is `OBSERVED`; it narrows the causal-use claim and does not trigger metric, dose, model, or locus replacement. Final test evidence is used once for this gate; implementation tuning uses only train/validation or synthetic fixtures before that point.

Dispatcher checksums, metadata, asset receipts, trust receipts, and terminal verification are reused across this gate. Hashing is limited to the necessary registered manifest at an explicit terminal gate or after concrete contamination evidence. Validator-only changes reuse the original artifact; gate `PASS` advances immediately to the next authorised experiment.

## Second hard gate

The next registered cell is `llava-edema-vislast-reltoken`. It holds the model, patient split, consumed visual locus, readout capacity, intervention family, dose scale, evaluation rows, and causal threshold fixed while changing the clinical concept to `Edema`. The prompt is `Is there pulmonary edema in this chest radiograph? Answer yes or no.` This cell is the highest-information same-locus contrast because the released exploratory evidence places Edema at the strongest controlled-decoding separation while the model's answer ranking remains near chance.

The probe uses the accepted `vis.last` activation artifact from immutable run `20260902T191411Z-ffd523c464c8-f99e2f39`: the prompt-independent output of `encoder.layers.22` for all 26,229 manifest rows. It applies the same seed-0 Gaussian projection to 512 dimensions, train-only standardisation, logistic regression with `C=1`, and 2,000 patient-level test bootstrap resamples with seed `20260827`. Type-to-label controls use the same 39 `view_AP × sex_M × age decade` strata and seeds 0–19. The Edema and Effusion selectivity estimates use the same patient bootstrap resamples and report their paired difference interval.

The intervention uses the same 200 held-out test row IDs as the first gate, `alpha-mode=reltoken`, concept alphas `-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1`, control alphas `-1,-0.5,-0.25,0.25,0.5,1`, 20 isotropic random directions, seed 0, and coordinate-permutation sham. The fixed unrelated directions are `Effusion`, `Atelectasis`, `Pneumothorax`, `Cardiomegaly`, `Mass`, and `Nodule`; all directions are fitted through the same capacity-matched projection.

The selective-cell criterion is unchanged: the maximum concept-consistent change in mean `P(yes)` at a registered nonzero alpha must exceed the same-alpha random 95th percentile and the maximum absolute sham effect. Unrelated-direction effects and Spearman monotonicity over `[-0.5,0.5]` are reported. If the paired selectivity interval places Edema above Effusion and Edema is selective, the two registered cells form one descriptive rank-concordant pair; if Edema has higher selectivity and remains nonselective, they form one descriptive rank-discordant pair. The pair does not estimate a population correlation. This is one prospectively registered new cell with one paired ordering contrast; the existing maximum control statistic remains the within-cell familywise test, so no additional multiplicity correction is introduced at this gate.

The immutable payload is `["bash","scripts/server/run_cross_cell_gate.sh"]`, dispatched on one registered GPU from a clean pushed `main`. The runner validates and references the accepted source activation receipt, records that source separately from the current probe and intervention commit, and writes all new outputs under its own immutable run directory.

## Third hard gate

The next registered cell is `qwen7b-effusion-vislast-reltoken`. It tests whether the two non-selective LLaVA observations generalise across architecture by changing the model family while holding the general-domain setting, NIH patient split, Effusion concept, prompt, consumed final visual-block semantics, readout capacity, intervention family, evaluation rows, and causal threshold fixed. The model is `Qwen/Qwen2.5-VL-7B-Instruct` at full revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, with processor resolution pinned by `min_pixels=max_pixels=336²`. The prompt is `Is there a pleural effusion in this chest radiograph? Answer yes or no.`

Before scientific extraction, a synthetic preflight must verify the registered 32-block architecture, exact `model.visual.blocks.31` `vis.last` hook, a bitwise alpha-zero no-op, and downstream changes at `model.visual.merger` and final logits under an alpha `+0.1` relative-token intervention. The model is staged atomically below `/home/qingchan/data/concept-flow/models/` from the pinned public Hugging Face revision, with the five published LFS weight-shard hashes recorded in its asset receipt. The full run extracts only `vis.last` for all 26,229 registered manifest rows at batch size 8 and requires exact unique row coverage with no failures.

The probe uses the seed-0 Gaussian projection to 512 dimensions divided by `sqrt(512)`, train-only standardisation, and logistic regression with `C=1`, `max_iter=2000`, and seed 0. It fits Effusion plus the fixed unrelated directions `Atelectasis`, `Pneumothorax`, `Cardiomegaly`, `Mass`, and `Nodule` through the same projection and scaler. Type-to-label controls use the same 39 `view_AP × sex_M × age decade` strata and seeds 0–19. AUROC and `real - mean(control)` use 2,000 patient-cluster bootstrap draws with seed `20260827`. The Qwen-minus-accepted-LLaVA Effusion selectivity difference uses identical ordered test rows and patient bootstrap counts. The intervention is eligible for decode-to-use interpretation only when the Qwen selectivity interval excludes zero in the positive direction.

The intervention reuses the first gate's exact 200 Effusion test row IDs, uses `alpha-mode=reltoken`, concept alphas `-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1`, control alphas `-1,-0.5,-0.25,0.25,0.5,1`, 20 isotropic random directions, seed 0, coordinate-permutation sham, and the five fixed unrelated probe normals. The expected grid contains 165 rows. The unchanged selective-cell criterion is the maximum concept-consistent change in mean `P(yes)` over the six registered nonzero control alphas, which must exceed the same-alpha random 95th percentile and the maximum absolute sham effect. Unrelated effects and Spearman monotonicity over `[-0.5,0.5]` are reported.

This is one prospectively registered new cell and one paired probe contrast. The maximum-over-dose quantity remains the single within-cell statistic, and no population correlation or architecture-family hypothesis is estimated from three cells, so no additional cross-cell multiplicity correction is introduced. An eligible selective result establishes a LLaVA/Qwen architecture boundary; an eligible non-selective result replicates the probe-normal mismatch across two general architectures. A valid probe that does not pass the availability criterion remains an `OBSERVED` cell but does not enter the decode-to-use comparison.

The model stage uses immutable payload `["python","scripts/server/stage_qwen7b_asset.py"]` with no GPU. After its asset receipt is accepted, the synthetic preflight uses `["bash","scripts/server/run_qwen_effusion_vislast_gate.sh","hook"]` on one registered GPU. The full-run payload is `["bash","scripts/server/run_qwen_effusion_vislast_gate.sh","full","<immutable-hook-run>/artifacts/hook-verification.json"]`, with the placeholder resolved to the accepted hook receipt before dispatch. Terminal internal replay and the pinned Claude read-only review precede `PASS`.

## Evaluation and fairness

- Train/test separation is patient-level; validation may select implementation parameters but never report headline evidence.
- Model comparisons are architecture-matched where claimed; unmatched models are descriptive replication.
- Report confidence intervals, repeated random-control distributions, all registered cells, and multiplicity handling before aggregate claims.
- Record opportunity/oracle bounds: probe performance bounds decodable opportunity; calibrated image-space planting and maximum validated intervention response bound attainable behavioural movement.
- Every hard gate receives internal verification followed by the pinned Claude read-only review. Review cannot create evidence.

## Data and model prerequisites

Fresh infrastructure smoke tests require no private data. Full table regeneration requires a server-local ChestX-ray14 image tree and manifest with image paths; ISIC requires its terms and a contact email; LLaVA-Med requires the converted local model directory expected by `src/registry.py`. Missing external assets are `BLOCKED`, never `FAILED` experiments.
