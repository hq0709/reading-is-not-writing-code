"""Print (model, dataset) pairs whose CORE block is complete (4 shard meta files) but not yet analysed (no 'core' in summary.json)."""
import glob, json, os, sys, time
R = "/rodata/azradonc_dev/m253405/cf-transfer/runs"
out = []
for d in sorted(glob.glob(f"{R}/*/*/outcomes/CORE")):
    metas = glob.glob(f"{d}/meta-*.json")
    if len(metas) < 4 or max(os.path.getmtime(m) for m in metas) > time.time() - 180:
        continue
    run = os.path.dirname(os.path.dirname(d)); m, ds = run.split("/")[-2:]
    s = f"{run}/summary.json"
    if os.path.exists(s) and "core" in json.load(open(s)):
        continue
    out.append(f"{m} {ds}")
print("\n".join(out))
