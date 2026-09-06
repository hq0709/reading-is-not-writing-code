# Concept Flow — measuring what a medical VLM *uses*, not just what it encodes

Research code, registered protocols, accepted evidence, and manuscript dependencies
for Concept Flow: clinical readout, steering efficacy, and direction-specific competition.
The local workspace keeps this repository in `concept-flow-code/` and the independent,
Overleaf-facing manuscript repository in the sibling `concept-flow-paper/` directory. The accepted
paper snapshot remains in [`paper/`](paper/) for result-to-claim verification; current publication
work lives in the private [`wy-coliney/concept-flow-paper`](https://github.com/wy-coliney/concept-flow-paper)
repository. Keep the two repositories as siblings.

## Research progress and records

Start with [Research state](docs/RESEARCH_STATE.md) for the current stage, completed
research, accepted findings, paper delivery, remaining decisions and task ownership.
It is the authoritative progress page and is updated at research and delivery checkpoints.

| Record | Purpose |
| --- | --- |
| [Research state](docs/RESEARCH_STATE.md) | Current progress, evidence map, paper/report links and next decision |
| [Research plan](docs/RESEARCH_PLAN.md) | Prospective questions, locked protocols and gate criteria |
| [Experiment registry](docs/EXPERIMENT_REGISTRY.md) | Run identities, execution status, artifacts and failure ledger |
| [Research reviews](docs/reviews/) | Protocol, implementation and result acceptance records |
| [Manuscript and reviews](https://github.com/wy-coliney/concept-flow-paper) | Current publication sources, PDF, manuscript reviews and review-package archive |
| [Project contract](AGENTS.md) | Harness entry point and links to writing, evidence and remote-operation constraints |

The following sections document the original utilisation analysis and its released
tables. Its scripts and reproduction claims apply to that analysis; the current
registered research and manuscript dependencies are indexed in the records above.

## Original utilisation analysis

A linear probe decodes a clinical finding from a layer of a medical vision-language model, and the finding
is said to be *represented* there. We argue that such a score is the sum of three things, and that the
field reports the sum:

```
T  =  probe on the trained model            what is usually reported
F  =  the SAME probe on the SAME architecture with randomly initialised weights
B  =  AUROC of the model's own P(yes) to "is there a <finding>?"

utilisation  U = (B - F) / (T - F)
```

`U` is 0 when the model's answer separates the labels no better than a probe on an untrained network,
1 when it matches any linear readout of that layer, and negative below the floor. Across five 7B–8B
models and nine chest-radiograph findings the median is **9%**, and in **22 of 45** cells the answer falls
**below** the untrained floor.

## What is here

| path | contents |
|---|---|
| `src/` | extraction, probing, behavioural scoring, interventions, direction fitting, planted lesions, and every figure script |
| `runs/*.csv` | the released result tables — every number in the paper is computed from these |
| `docs/` | the evidence ledger, the figure standard, and the literature survey |
| `fonts/` | the vendored fonts the figures need (Lato, Tinos) |
| `paper/` | the accepted, evidence-locked LaTeX snapshot used by repository checks |

Not here: ~105 GB of cached activations, model weights, and the source images. **The tables in `runs/`
are the released artefact** — every number in the paper is computed from them, and they are what to read
if you want the results without rerunning anything.

The scripts split into two kinds, and it matters which you have the inputs for:

| kind | scripts | needs |
|---|---|---|
| **regenerate the tables** | `extract.py`, `probe.py`, `behaviour.py`, `intervene.py`, `utilisation.py` | the images, the model weights, and a GPU |
| **read the tables** | every `fig_*.py`, `paperfigs.py`, `make_tables.py`, `verify_draft.py` | nothing but this repository |

To make that true rather than aspirational, the repository ships everything the second row needs and
nothing more:

- `runs/*.csv` — the summary tables
- `runs/{probe,rndprobe}_*/probe_results.csv` — the per-locus sweeps, every locus of every model, trained
  and untrained
- `runs/behav_*/per_image.npz` — the model's own score for each of the 1000 held-out images
- `data/manifest.csv` — labels, patient ids and the **exact patient-level split**, derived from the public
  ChestX-ray14 label file. The `image_path` column is dropped: it was machine-specific, and the images are
  not ours to redistribute.

Twelve megabytes in total, and it reproduces every figure and every number in the paper.

So a fresh clone reproduces **every figure and every quoted number** with no download and no GPU. Only
regenerating the tables themselves needs the data.

## Two checks that run on every build

Research code drifts from the paper it supports. Two scripts make that a build failure rather than a
reviewer's discovery:

```bash
python src/verify_draft.py     # recomputes all 56 numbers quoted in the paper from runs/*.csv; exits 1 on a mismatch
python src/lint_tex.py <tex>   # loose prose inside a float, a \ref with no \label, a figure never referenced, a missing image
python src/figcheck.py         # imported by each figure script: overlapping labels, text over data, clipped text, crowded ticks
```

`verify_draft.py` exists because two numbers in an early draft disagreed with each other. `figcheck.py`
exists because label collisions kept surviving visual review — on its first run it found nine rotated
labels overlapping by 4–5 pt in a figure that had already been checked by eye.

## Reproducing

```bash
conda create -n conceptflow python=3.11 && conda activate conceptflow
pip install -r requirements.txt

export HF_HOME=/path/to/hf_cache
export CXR14_ROOT=/path/to/chestxray14/images

python src/extract.py --manifest data/manifest.csv --arch lingshu7b --gpus 0,1,2,3 --out runs/act_lingshu7b
python src/probe.py   --acts runs/act_lingshu7b --out runs/probe_lingshu7b
python src/extract_random_init.py --arch lingshu7b --out runs/act_rnd_lingshu7b   # the floor
python src/behaviour.py --arch lingshu7b --out runs/behav_lingshu7b               # the answer
python src/utilisation.py                                                        # -> runs/utilisation_final.csv
python src/paperfigs.py && python src/fig_heat.py && python src/fig_floor.py      # figures
```

A 7B model in bf16 is about 16 GB, so one replica fits on one 80 GB card with room for activations;
extraction is data-parallel across GPUs with no cross-device communication.

## Data

- **Chest radiographs**: NIH ChestX-ray14 ([Wang et al., 2017](https://arxiv.org/abs/1705.02315)).
  26,229 frontal images, nine findings. Split **by patient**, not by image: the dataset has several
  studies per patient and an image-level split leaks anatomy across the boundary.
- **Dermoscopy**: the [ISIC archive](https://www.isic-archive.com/). 12,012 dermoscopic images over
  4,109 patients with six diagnoses at 300+ examples, filtered from the full index of 553,019 records.

Neither dataset is redistributed here. Both are public; follow their own terms.

## Models

Chosen so that two *architecture-matched* medical/general pairs exist, which is what makes the floor
shared and the comparison clean:

| medical | general | share an untrained floor |
|---|---|---|
| Lingshu-7B | Qwen2.5-VL-7B | yes, exactly |
| LLaVA-Med-7B | LLaVA-1.5-7B | yes, exactly |
| — | InternVL3-8B | — |

## Licence

Code: MIT (`LICENSE`). Vendored fonts keep their own: **Lato** under the SIL Open Font Licence 1.1,
**Tinos** under the Apache Licence 2.0.
