"""Print (model, dataset) pairs whose CORE block is complete (4 shard meta files) but not yet analysed (no 'core' in summary.json)."""
import glob, json, os, sys
R = "/rodata/azradonc_dev/m253405/cf-transfer/runs"
out = []
for d in sorted(glob.glob(f"{R}/*/*/outcomes/CORE")):
    if len(glob.glob(f"{d}/meta-*.json")) < 4:
        continue
    run = os.path.dirname(os.path.dirname(d)); m, ds = run.split("/")[-2:]
    s = f"{run}/summary.json"
    if os.path.exists(s) and "core" in json.load(open(s)):
        continue
    out.append(f"{m} {ds}")
print("\n".join(out))
