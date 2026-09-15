"""Print blocks whose ALTDIR shard has landed (meta file present) but whose summary.json has no 'altdir' key."""
import json, glob, os
R = "/rodata/azradonc_dev/m253405/cf-transfer/runs"
for meta in sorted(glob.glob(f"{R}/*/*/outcomes/ALTDIR/meta-*.json")):
    rd = os.path.dirname(os.path.dirname(os.path.dirname(meta)))
    m = json.load(open(meta))
    if m.get("outcomes", 0) == 0 and m.get("todo", 1) != 0:
        continue
    sp = os.path.join(rd, "summary.json")
    s = json.load(open(sp)) if os.path.exists(sp) else {}
    if "altdir" not in s:
        print(os.path.basename(os.path.dirname(rd)), os.path.basename(rd))
