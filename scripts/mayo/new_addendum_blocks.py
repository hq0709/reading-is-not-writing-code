"""Print `model dataset module` for addendum modules whose shard(s) landed but whose summary.json lacks the analysis key."""
import json, glob, os
R = "/rodata/azradonc_dev/m253405/cf-transfer/runs"


SHARDS = {"ANSDIRT": 2, "ATTRRAND": 2, "PROJSEED": 2, "SEMEND": 4}   # PROJSEED: one task (and one meta) per projection seed
Q = "/rodata/azradonc_dev/m253405/cf-transfer/queue"


def busy(rd, mod):
    """A task of this module for this block is still pending or running. Per-setting / per-seed tasks carry the setting
    between the module name and the model (PRECISION-fp32-..., PROJSEED-s1-...), so both shapes are matched."""
    m, d = rd.rstrip("/").split("/")[-2:]
    return any(glob.glob(f"{Q}/{st}/{pat}") for st in ("pending", "running")
               for pat in (f"*-{mod}-{m}-{d}-*", f"*-{mod}-*-{m}-{d}-*"))


def partial(a, n=None):
    """True when any n_scored_rows_min inside the module summary is below the summary's n_rows."""
    if isinstance(a, dict):
        n = a.get("n_rows", n)
        if n and isinstance(a.get("n_scored_rows_min"), int) and a["n_scored_rows_min"] < n:
            return True
        return any(partial(v, n) for v in a.values())
    return False
MODS = {"ANSDIR": "ansdir", "EXTCOMP": "extcomp", "TOKENW": "tokenw", "PRECISION": "precision", "ALTDIRD": "altdird",
        "ATTRQ": "attrq", "ATTR": "attr", "ANSDIRT": "ansdirt",
        "VALID": "valid", "ATTRRAND": "attr", "VALIDFIT": "validfit", "PROJSEED": "projseed",
        "TOWERSWAP": "towerswap", "REPLAY": "replay", "SEMEND": "semend"}
# ATTRRAND has no analysis key of its own: it is folded into `attr`, so a block whose stored `attr` was computed before
# ATTRRAND landed (no attribute random family in it) is reported as needing the analysis again. ATTRQ is listed BEFORE
# ATTR so that in one pass the attribute answerability is computed first and `attr` can read it; an `attr` computed
# before ATTRQ landed (its answerability still measured on the test rows) is likewise reported as stale.
# FGOBJ_CALIBRATION is folded into `fgobj` the same way: it carries no analysis key of its own, and an `fgobj` computed
# before it landed has no clean-answer AUROC for the fine-grained cells, so those cells cannot be pooled with the
# campaign's cells and the analysis is reported as stale.
MODS["FGOBJ"] = "fgobj"
MODS["FGOBJ_CALIBRATION"] = "fgobj"
SHARDS["FGOBJ"] = 4
for mod, key in MODS.items():
    for d in sorted(glob.glob(f"{R}/*/*/outcomes/{mod}")):
        metas = glob.glob(f"{d}/meta-*.json")
        if not metas:
            continue
        if mod == "PRECISION" and not (any("fp32" in os.path.basename(m) for m in metas) and any("batch1" in os.path.basename(m) for m in metas)):
            continue
        if any(json.load(open(m)).get("outcomes", 0) == 0 and json.load(open(m)).get("todo", 1) != 0 for m in metas):
            continue
        rd = os.path.dirname(os.path.dirname(d)); sp = os.path.join(rd, "summary.json")
        s = json.load(open(sp)) if os.path.exists(sp) else {}
        # analyse when the key is missing, or when the stored analysis saw fewer scored rows than the module's row count
        # (it ran while shards were still landing)
        stale = False
        if mod == "ATTRRAND":                     # attr is stored but was computed without the attribute random family
            stale = ((s.get("attr") or {}).get("attrrand") or {}).get("available") is not True
        if mod in ("ATTR", "ATTRRAND") and s.get("attrq"):   # attr predates this block's ATTRQ answerability
            stale = stale or ((s.get("attr") or {}).get("answerability") or {}).get("attrq_available") is not True
        if mod == "FGOBJ_CALIBRATION":            # fgobj is stored but was computed without the clean-answer margins
            stale = ((s.get("fgobj") or {}).get("probe_grade_source") or {}).get("answer_available") is not True
        if key not in s or stale or (partial(s[key]) and len(metas) >= SHARDS.get(mod, 1) and not busy(rd, mod)):
            print(os.path.basename(os.path.dirname(rd)), os.path.basename(rd), mod)
