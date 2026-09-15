"""Print `model dataset module` for addendum modules whose shard(s) landed but whose summary.json lacks the analysis key."""
import json, glob, os
R = "/rodata/azradonc_dev/m253405/cf-transfer/runs"
MODS = {"ANSDIR": "ansdir", "EXTCOMP": "extcomp", "TOKENW": "tokenw", "PRECISION": "precision"}
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
        if key not in s:
            print(os.path.basename(os.path.dirname(rd)), os.path.basename(rd), mod)
