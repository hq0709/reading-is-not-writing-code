| checkpoint | NIH read/answer/own | CheXpert read/answer/own | COCO read/answer/own | owned clinical concepts | primary template |
|---|---|---|---|---|---|
| q25-3 | 2/2/0 | 4/2/0 | 4/5/6 |  |  |
| q25-7 | 3/3/0 | 4/2/0 | 1/5/6 |  |  |
| q25-32 | 3/4/1 | 4/4/0 | 1/5/6 | nih: Cardiomegaly |  |
| q25-72 | 2/4/3 | - | - | nih: Atelectasis, Cardiomegaly, Mass |  |
| q3-4 | 3/6/0 | 4/4/2 | 4/5/6 | chexpert: Consolidation, Edema |  |
| q3-8 | 4/6/0 | 4/4/0 | 3/5/6 |  |  |
| q3-32 | 2/6/1 | 5/5/1 | 4/5/6 | nih: Nodule; chexpert: Edema |  |
| iv35-8 | 2/6/0 | 4/5/1 | 5/5/6 | chexpert: Pneumothorax |  |
| iv35-14 | 4/6/1 | 4/5/1 | 4/5/5 | nih: Effusion; chexpert: Effusion |  |
| iv35-38 | 5/6/0 | 4/5/2 | 4/5/6 | chexpert: Consolidation, Effusion |  |
| gemma3-4 | 2/2/0 | 4/1/0 | 5/5/5 |  |  |
| gemma3-12 | 2/5/1 | 4/0/1 | 5/5/6 | nih: Nodule; chexpert: Edema |  |
| gemma3-27 | 2/3/0 | 4/2/0 | 5/5/6 |  |  |
| medgemma-4 | 3/6/0 | 4/5/2 | 4/5/6 | chexpert: Cardiomegaly, Edema |  |
| medgemma-27 | 3/6/0 | 4/5/0 | 4/5/4 |  |  |
| llama32-11 | 2/4/0 | 4/-/INELIGIBLE | 5/-/INELIGIBLE |  | nih: IB |
| llava15-7 | 2/0/0 | 4/0/0 | 5/5/5 |  |  |
| llava15-13 | 2/0/0 | 4/1/0 | 5/5/2 |  |  |
| lingshu-7 | 3/6/1 | 4/5/3 | 1/5/6 | nih: Cardiomegaly; chexpert: Edema, Effusion, Pneumothorax |  |
| llavamed-7 | 3/0/0 | 4/2/1 | 4/5/1 | chexpert: Effusion |  |
| lingshu-32 | 3/6/2 | 4/5/5 | 1/5/6 | nih: Cardiomegaly, Pneumothorax; chexpert: Cardiomegaly, Consolidation, Edema, Effusion, Pneumothorax |  |

Cells are counts over the 6 concepts of a dataset: readable at the consumed visual block / answer-capable on the clean yes/no question / owned by the concept write (protocol rules in `cftransfer.leaderboard`). INELIGIBLE marks blocks whose yes/no template fails the image-free semantic-mapping preflight, where ownership is not defined. The primary-template column lists blocks whose IY template failed that preflight and which were scored on the eligible B-positive A/B template instead (IB); answer capability and ownership there refer to that template.
