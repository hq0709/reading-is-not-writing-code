"""Fixed protocol constants, loaded from the machine-readable package so nothing is retyped.

Everything an executor might be tempted to "adjust" lives here and is read from
docs/external-replication/protocol.json (templates, concepts, finding phrases, image phrases,
seeds, dose grids, direction counts). Module grids are derived from README section 5.4.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "docs" / "external-replication"
PROTOCOL = json.loads((PKG / "protocol.json").read_text(encoding="utf-8"))
PROTOCOL_ID = PROTOCOL["protocol_id"]

TEMPLATES: dict[str, str] = PROTOCOL["templates"]                      # IY WY IA IB WA WB
TEMPLATE_ORDER = ["IY", "WY", "IA", "IB", "WA", "WB"]
YESNO_TEMPLATES = {"IY", "WY"}
AB_TEMPLATES = {"IA", "IB", "WA", "WB"}
B_POSITIVE_TEMPLATES = {"IB", "WB"}          # B means present, A means absent
# Primary template: the one yes/no-style template that CORE, CALIBRATION, DOSE, REFIT, LOCUS and LOCUS_CALIBRATION
# score. It is IY unless the block's image-free preflight (check E) marked IY INELIGIBLE, in which case the first
# eligible template in this order takes its place (IB: A/B mapping with B = present), so the block is scored on an
# eligible interface instead of being closed as ineligible. Resolved per block by primary_template(run_dir).
PRIMARY_TEMPLATE_FALLBACK = ("IY", "IB")

DATASETS = ["nih", "chexpert", "coco"]
CONCEPTS = {
    "nih": PROTOCOL["dataset"]["nih_concepts"],
    "chexpert": PROTOCOL["dataset"]["chexpert_concepts"],
    "coco": PROTOCOL["dataset"]["coco_concepts"],
}
FINDING_PHRASES = {
    "nih": dict(zip(PROTOCOL["concepts"], PROTOCOL["finding_phrases"])),
    "chexpert": dict(zip(PROTOCOL["dataset"]["chexpert_concepts"], PROTOCOL["dataset"]["chexpert_finding_phrases"])),
    "coco": dict(zip(PROTOCOL["dataset"]["coco_concepts"], PROTOCOL["dataset"]["coco_finding_phrases"])),
}
IMAGE_PHRASE = PROTOCOL["dataset"]["templates_image_phrase"]
for _ds in ("nih", "chexpert"):                                  # attribute questions render with the protocol templates
    FINDING_PHRASES[_ds].update({"view_AP": "an anteroposterior (portable) projection", "sex_F": "a female patient",
                                 "age_60": "a patient aged sixty or older"})
PROMPT_CONCEPTS = PROTOCOL["modules"]["PROMPT"]["concepts_by_dataset"]   # nih: Effusion, Mass; coco: person, bottle; chexpert: Effusion, Edema
# CALIBRATION additionally scores the PROMPT concepts under the five other templates on these datasets only (README 5.4).
# CheXpert is deliberately excluded: every CheXpert CALIBRATION block ran IY-only before PROMPT was extended to it.
CALIBRATION_PROMPT_DATASETS = ("nih", "coco")
# Modules added to CheXpert after the first campaign wave (CORE + CALIBRATION). A CheXpert block packaged before one of
# them is started is judged on the modules it has: package.build counts such a module as requested only once its
# outcomes directory exists (the runner creates it when the module starts).
# ALTDIR (alternative direction estimators, reviewer objection on estimator dependence) was added to every dataset
# after the campaign's blocks were packaged; it is enqueued explicitly (enqueue --modules ALTDIR) and its coverage is
# requested only once its outcomes directory exists, so packaged blocks keep COMPLETE until it starts.
MODULES_ADDED_LATER = {"nih": ("ALTDIR", "EXTCOMP", "TOKENW", "PRECISION", "ANSDIR", "ALTDIRD", "ATTR", "ANSDIRT"),
                       "coco": ("ALTDIR", "TOKENW", "PRECISION", "ANSDIR", "ALTDIRD", "ANSDIRT"),
                       "chexpert": ("PROMPT", "DOSE", "REFIT", "LOCUS", "LOCUS_CALIBRATION", "ALTDIR", "EXTCOMP", "TOKENW", "ANSDIR",
                                    "ALTDIRD", "ATTR", "ANSDIRT", "VALID")}
# ALTDIR direction families, in condition order: difference of means, Haufe pattern, orthogonalised logistic normal,
# logistic normal refitted on label-residualised features (altdir.py); every family is lifted with the logistic rule.
ALTDIR_FAMILIES = ("dom", "pattern", "orth", "resid")
# ALTDIRD: the two displacement families lifted as displacements, x = R (R^T R)^{-1} diag(s) u (minimum-norm preimage
# of the standardised displacement u), instead of the coefficient lift R diag(1/s) beta (altdir.displacement_lift).
ALTDIRD_FAMILIES = ("dom_disp", "pattern_disp")
# ATTR: non-clinical radiographic attributes of the same images, fitted like the protocol normals (attr.py):
# view_AP (1 = AP/portable, 0 = PA), sex_F (1 = female, 0 = male), age_60 (1 = age >= 60); >= ATTR_MIN_CLASS_ROWS known
# training rows per class (NIH and CheXpert both qualify on all three). Questions use the protocol templates with the
# attribute phrase in the {finding} slot.
ATTR_CONCEPTS = ["view_AP", "sex_F", "age_60"]
ATTR_PHRASES = {"view_AP": "an anteroposterior (portable) projection", "sex_F": "a female patient",
                "age_60": "a patient aged sixty or older"}
ATTR_MIN_CLASS_ROWS = 100
ATTR_SHAM_SEED = 1                 # PCG64(1): one coordinate permutation per attribute, in ATTR_CONCEPTS order
# ANSDIRT: the IY-fitted answer directions written under the five other templates (own clean baseline per template)
ANSDIRT_TEMPLATES = ("WY", "IA", "IB", "WA", "WB")
# VALID: the CORE grid (primary template, PRIMARY_ALPHA, the 127-direction core family of the seed-0 fit) on the `valid`
# role of CheXpert: one frontal per patient of the official CheXpert validation split (200 rows, manifests.build_chexpert_valid),
# whose labels are the radiologist consensus rather than the labeler output of every other role. Graded with the CORE rules
# on those rows and compared with the block's CORE grade on the labeler-labelled test rows (analysis.valid).
VALID_ROWS = 200
# EXTCOMP: extra dataset labels fitted like the protocol normals (extcomp.py) and added to the competitor family. The
# frozen lists are the manifest labels with at least EXTCOMP_MIN_SUPPORT known positives AND negatives in the training
# rows (counts checked again by the prep); the excluded labels and the reason are recorded next to them.
EXTCOMP_MIN_SUPPORT = 100
EXTCOMP_LABELS = {"nih": ["Consolidation", "Edema", "Infiltration"],
                  "chexpert": ["Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "Pneumonia", "Support Devices"]}
EXTCOMP_EXCLUDED = {"nih": {"no_finding": "no-finding label, excluded by design"},
                    "chexpert": {"No Finding": "no-finding label, excluded by design",
                                 "Pleural Other": "12 known negatives in the training rows (< 100)"}}
# TOKENW: the six logistic directions written with per-token weights (hooks.token_weights): "tokenw" = softmax over the
# consumed tokens of the token's probe score (temperature 1) rescaled to mean 1; "topq" = the top TOKENW_TOPQ_FRACTION of
# tokens by probe score with weight 1/fraction (mean 1). Same total dose as CORE's uniform write.
TOKENW_VARIANTS = ("tokenw", "topq")
TOKENW_TOPQ_FRACTION = 0.25
# PRECISION: the CORE grid on the first PRECISION_ROWS test rows under two numerics settings selected by the runner's
# --numerics flag: "fp32" (weights and forward in float32, CORE batch composition) and "batch1" (bf16, batch size 1 for
# every condition). Every other module runs NUMERICS_DEFAULT; outcomes carry it in the `numerics` column.
PRECISION_SETTINGS = ("fp32", "batch1")
PRECISION_ROWS = 200
NUMERICS_DEFAULT = "bf16-batched"
MODULE_SETTINGS = {"PRECISION": PRECISION_SETTINGS}       # modules scored once per setting (rows x settings)
# The seven modules of the planned campaign (protocol.json total_planned_outcomes) and the modules added afterwards as
# addenda (each enqueued explicitly, each counted in total_planned_outcomes.addenda, each in MODULES_ADDED_LATER).
PLANNED_MODULES = ("CORE", "CALIBRATION", "PROMPT", "DOSE", "REFIT", "LOCUS", "LOCUS_CALIBRATION")
ADDENDUM_MODULES = ("ALTDIR", "EXTCOMP", "TOKENW", "PRECISION", "ANSDIR", "ALTDIRD", "ATTR", "ANSDIRT", "VALID")
# ANSDIR (answer-direction oracle, ansdir.py): per question q a ridge regression of the clean answer margin on the
# projected, train-scaled features of the first ANSDIR_N_TRAIN training rows (alpha by 5-fold CV over ANSDIR_ALPHAS),
# lifted like the logistic normals; written with the six a_d plus the coordinate-permutation sham of a_q.
ANSDIR_N_TRAIN = 3000
ANSDIR_ALPHAS = (0.1, 1.0, 10.0, 100.0)
ANSDIR_CV_FOLDS = 5
COCO_CATEGORY_IDS = {"person": 1, "dog": 18, "car": 3, "chair": 62, "bottle": 44, "bicycle": 2}

# probe / direction constants
PROJECTION_DIM = PROTOCOL["probe"]["projection_dim"]          # 512
PROJECTION_SEED = PROTOCOL["probe"]["projection_seed"]        # 0
CONTROL_SEEDS = list(PROTOCOL["probe"]["control_seeds"])      # 0..19
REFIT_SEEDS = list(PROTOCOL["probe"]["refit_seeds"])          # 1, 2
LOGREG = dict(C=1.0, max_iter=2000, solver="lbfgs", class_weight=None, random_state=0)
N_RANDOM = PROTOCOL["primary"]["random_directions"]           # 119
RANDOM_SEED = PROTOCOL["intervention"]["random_seed"]         # 0
PRIMARY_ALPHA = float(PROTOCOL["primary"]["alpha"])           # 0.25
DOSE_ALPHAS = [float(a) for a in PROTOCOL["modules"]["DOSE"]["alpha"]]   # -0.5 -0.25 -0.1 0.1 0.5
DOSE_N_RANDOM = 20
DOSE_ROWS = 200

# bootstrap seeds (README section 6, computation details)
BOOT_CORE_SEED = PROTOCOL["primary"]["bootstrap_seed"]        # 2026090601
BOOT_CALIBRATION_SEED = 2026090602
BOOT_DIAGNOSTIC_SEED = 2026090603
BOOT_CORE_DRAWS = PROTOCOL["primary"]["bootstrap_draws"]      # 5000
BOOT_CALIBRATION_DRAWS = 2000

LOCI = {"primary": "vis.last", "connector": "connector"}     # logical locus ids used in every table

MODELS = {m["model_key"]: m for m in PROTOCOL["planned_models"]}
MODEL_ORDER = [m["model_key"] for m in PROTOCOL["planned_models"]]


def render_question(dataset_id: str, concept: str, template_id: str) -> str:
    """Exact question text for one (dataset, concept, template); syntax is not the executor's to edit."""
    return TEMPLATES[template_id].format(
        finding=FINDING_PHRASES[dataset_id][concept], image_phrase=IMAGE_PHRASE[dataset_id]
    )


def primary_template(run_dir: Path | str) -> str:
    """The block's primary template: IY when it is eligible (or no template_eligibility.json exists yet), else the
    first eligible template in PRIMARY_TEMPLATE_FALLBACK order, else IY (the runner's eligibility filter then closes
    the module as ineligible)."""
    path = Path(run_dir) / "template_eligibility.json"
    if not path.exists():
        return "IY"
    elig = json.loads(path.read_text(encoding="utf-8"))
    for t in PRIMARY_TEMPLATE_FALLBACK:
        if elig.get(t, {}).get("eligible", True):
            return t
    return "IY"


def direction_ids(n_random: int = N_RANDOM, sham_question: str | None = None, with_baseline: bool = True,
                  concepts: list[str] | None = None, dataset_id: str = "nih") -> list[str]:
    """Canonical direction identifiers in protocol order: baseline, concept:<name>, random:###, sham:<question>."""
    out = ["baseline"] if with_baseline else []
    out += [f"concept:{c}" for c in (concepts or CONCEPTS[dataset_id])]
    out += [f"random:{i:03d}" for i in range(n_random)]
    if sham_question is not None:
        out.append(f"sham:{sham_question}")
    return out


def direction_kind(direction_id: str) -> str:
    return direction_id.split(":", 1)[0]


@dataclass(frozen=True)
class ModuleSpec:
    module: str
    role: str                       # cohort role scored
    row_limit: int | None           # first N rows of the role in cohorts order (DOSE)
    templates: tuple[str, ...]
    concepts_rule: str              # "all" | "prompt" | "attr" (three attribute questions + the six clinical ones)
    alphas: tuple[float, ...]
    directions: str                 # "core" | "clean" | "dose" | "refit" | "altdir" | "extcomp" | "tokenw" | "ansdir" | "altdird" | "attr" | "ansdirt"
    fit_seeds: tuple[int, ...]
    locus: str                      # "primary" | "connector"
    baseline_module: str | None     # module whose clean baseline is reused (DOSE/REFIT)
    datasets: tuple[str, ...]


MODULES: dict[str, ModuleSpec] = {
    "CORE": ModuleSpec("CORE", "test", None, ("IY",), "all", (PRIMARY_ALPHA,), "core", (0,), "primary", None,
                       ("nih", "chexpert", "coco")),
    "CALIBRATION": ModuleSpec("CALIBRATION", "calibration", None, ("IY",), "all", (0.0,), "clean", (0,), "primary", None,
                              ("nih", "chexpert", "coco")),
    "PROMPT": ModuleSpec("PROMPT", "test", None, ("WY", "IA", "IB", "WA", "WB"), "prompt", (PRIMARY_ALPHA,), "core", (0,),
                         "primary", None, ("nih", "chexpert", "coco")),
    "DOSE": ModuleSpec("DOSE", "test", DOSE_ROWS, ("IY",), "all", tuple(DOSE_ALPHAS), "dose", (0,), "primary", "CORE",
                       ("nih", "chexpert", "coco")),
    "REFIT": ModuleSpec("REFIT", "test", None, ("IY",), "all", (PRIMARY_ALPHA,), "refit", tuple(REFIT_SEEDS), "primary",
                        "CORE", ("nih", "chexpert", "coco")),
    "LOCUS": ModuleSpec("LOCUS", "test", None, ("IY",), "all", (PRIMARY_ALPHA,), "core", (0,), "connector", None,
                        ("nih", "chexpert", "coco")),
    "LOCUS_CALIBRATION": ModuleSpec("LOCUS_CALIBRATION", "calibration", None, ("IY",), "all", (0.0,), "clean", (0,),
                                    "connector", None, ("nih", "chexpert", "coco")),
    # alternative direction estimators at the primary locus, seed 0, primary dose; clean baseline reused from CORE
    "ALTDIR": ModuleSpec("ALTDIR", "test", None, ("IY",), "all", (PRIMARY_ALPHA,), "altdir", (0,), "primary", "CORE",
                         ("nih", "chexpert", "coco")),
    # extended competitor set: every extra dataset label's logistic direction, CORE baseline reused
    "EXTCOMP": ModuleSpec("EXTCOMP", "test", None, ("IY",), "all", (PRIMARY_ALPHA,), "extcomp", (0,), "primary", "CORE",
                          ("nih", "chexpert")),
    # token-weighted writes of the six logistic directions (softmax / top-quarter weights), CORE baseline reused
    "TOKENW": ModuleSpec("TOKENW", "test", None, ("IY",), "all", (PRIMARY_ALPHA,), "tokenw", (0,), "primary", "CORE",
                         ("nih", "chexpert", "coco")),
    # the CORE grid (own baseline) on the first 200 test rows, once per numerics setting (fp32, batch1)
    "PRECISION": ModuleSpec("PRECISION", "test", PRECISION_ROWS, ("IY",), "all", (PRIMARY_ALPHA,), "core", (0,), "primary", None,
                            ("nih", "coco")),
    # answer-direction oracle: the six ridge answer directions a_d plus the sham of a_q, CORE baseline and random family reused
    "ANSDIR": ModuleSpec("ANSDIR", "test", None, ("IY",), "all", (PRIMARY_ALPHA,), "ansdir", (0,), "primary", "CORE",
                         ("nih", "chexpert", "coco")),
    # displacement-lifted dom / pattern families, CORE baseline reused
    "ALTDIRD": ModuleSpec("ALTDIRD", "test", None, ("IY",), "all", (PRIMARY_ALPHA,), "altdird", (0,), "primary", "CORE",
                          ("nih", "chexpert", "coco")),
    # attribute questions (own baseline) and clinical questions (CORE baseline) under the 3 attribute + 6 clinical directions
    "ATTR": ModuleSpec("ATTR", "test", None, ("IY",), "attr", (PRIMARY_ALPHA,), "attr", (0,), "primary", "CORE",
                       ("nih", "chexpert")),
    # answer directions under the five other templates, own baseline per (question, template)
    "ANSDIRT": ModuleSpec("ANSDIRT", "test", None, ANSDIRT_TEMPLATES, "all", (PRIMARY_ALPHA,), "ansdirt", (0,), "primary", None,
                          ("nih", "chexpert", "coco")),
    # the CORE grid (own baseline) on the radiologist-labelled valid rows of CheXpert
    "VALID": ModuleSpec("VALID", "valid", None, ("IY",), "all", (PRIMARY_ALPHA,), "core", (0,), "primary", None, ("chexpert",)),
}


def question_list(dataset_id: str, module: str, primary: str = "IY") -> list[tuple[str, str]]:
    """(concept, template_id) pairs a module scores for one dataset, in protocol order. `primary` is the block's
    primary template (primary_template(run_dir)); it replaces the "IY" placeholder of the single-template modules.
    PROMPT's five templates and CALIBRATION's extra prompt-concept templates are never substituted, so a block whose
    primary is IB scores IB twice for the prompt concepts in CALIBRATION (deduplicated on merge)."""
    spec = MODULES[module]
    concepts = CONCEPTS[dataset_id]
    if spec.concepts_rule == "prompt":
        concepts = PROMPT_CONCEPTS[dataset_id]
    elif spec.concepts_rule == "attr":
        concepts = ATTR_CONCEPTS + list(concepts)
    out = [(c, primary if t == "IY" else t) for t in spec.templates for c in concepts]
    if module == "CALIBRATION" and dataset_id in CALIBRATION_PROMPT_DATASETS:
        # NIH/COCO calibration also scores the two PROMPT concepts under the five other templates (README 5.4)
        out += [(c, t) for t in ("WY", "IA", "IB", "WA", "WB") for c in PROMPT_CONCEPTS[dataset_id]]
    return out


def conditions_for(module: str, dataset_id: str, concept: str) -> list[tuple[str, float]]:
    """(direction_id, alpha) list for one question in a module; baseline rows carry alpha 0."""
    spec = MODULES[module]
    sham = f"sham:{concept}"
    if spec.directions == "clean":
        return [("baseline", 0.0)]
    if spec.directions == "core":
        ids = ["baseline"] + [f"concept:{c}" for c in CONCEPTS[dataset_id]] + \
              [f"random:{i:03d}" for i in range(N_RANDOM)] + [sham]
        return [(d, 0.0 if d == "baseline" else spec.alphas[0]) for d in ids]
    if spec.directions == "dose":
        ids = [f"concept:{c}" for c in CONCEPTS[dataset_id]] + [f"random:{i:03d}" for i in range(DOSE_N_RANDOM)] + [sham]
        return [(d, a) for a in spec.alphas for d in ids]
    if spec.directions == "refit":
        ids = [f"concept:{c}" for c in CONCEPTS[dataset_id]] + [sham]
        return [(d, spec.alphas[0]) for d in ids]
    if spec.directions == "altdir":
        # every family's six concept directions, family by family: 24 steered conditions, no baseline rows
        ids = [f"{fam}:{c}" for fam in ALTDIR_FAMILIES for c in CONCEPTS[dataset_id]]
        return [(d, spec.alphas[0]) for d in ids]
    if spec.directions == "extcomp":
        return [(f"extra:{k}", spec.alphas[0]) for k in EXTCOMP_LABELS[dataset_id]]
    if spec.directions == "tokenw":
        return [(f"{v}:{c}", spec.alphas[0]) for v in TOKENW_VARIANTS for c in CONCEPTS[dataset_id]]
    if spec.directions == "ansdir":
        return [(f"ans:{c}", spec.alphas[0]) for c in CONCEPTS[dataset_id]] + [(f"anssham:{concept}", spec.alphas[0])]
    if spec.directions == "altdird":
        return [(f"{fam}:{c}", spec.alphas[0]) for fam in ALTDIRD_FAMILIES for c in CONCEPTS[dataset_id]]
    if spec.directions == "attr":
        # attribute questions score their own clean baseline (no other module has it); clinical questions reuse CORE's
        dirs = [f"attr:{a}" for a in ATTR_CONCEPTS] + [f"concept:{c}" for c in CONCEPTS[dataset_id]]
        if concept in ATTR_CONCEPTS:
            return [("baseline", 0.0)] + [(d, spec.alphas[0]) for d in dirs] + [(f"attrsham:{concept}", spec.alphas[0])]
        return [(d, spec.alphas[0]) for d in dirs] + [(sham, spec.alphas[0])]
    if spec.directions == "ansdirt":
        return [("baseline", 0.0)] + [(f"ans:{c}", spec.alphas[0]) for c in CONCEPTS[dataset_id]] + [(f"anssham:{concept}", spec.alphas[0])]
    raise ValueError(spec.directions)


def expected_rows(module: str, dataset_id: str, n_rows: int | None = None) -> int:
    spec = MODULES[module]
    if dataset_id not in spec.datasets:
        return 0
    n = n_rows if n_rows is not None else {"test": 600, "calibration": 400, "valid": VALID_ROWS}[spec.role]
    if spec.row_limit:
        n = min(n, spec.row_limit)
    total = 0
    for concept, _t in question_list(dataset_id, module):
        total += len(conditions_for(module, dataset_id, concept)) * len(spec.fit_seeds)
    return total * n * len(MODULE_SETTINGS.get(module, (None,)))


if __name__ == "__main__":
    for ds in DATASETS:
        for m in MODULES:
            print(f"{ds:9s} {m:18s} {expected_rows(m, ds):>9,d}")
