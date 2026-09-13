"""Print (model, dataset) pairs whose every requested module has all shard meta files but run.json is not COMPLETE."""
import glob, json, os, sys, time
sys.path.insert(0, "/rodata/azradonc_dev/m253405/concept-flow/src")
from cftransfer.protocol import MODULES, expected_rows
from cftransfer.enqueue import SHARDS
R = "/rodata/azradonc_dev/m253405/cf-transfer/runs"
for run in sorted(glob.glob(f"{R}/*/*")):
    m, ds = run.split("/")[-2:]
    if ds not in ("nih", "coco", "chexpert") or not os.path.isdir(f"{run}/outcomes"):
        continue
    rj = f"{run}/run.json"
    if os.path.exists(rj) and json.load(open(rj)).get("status") == "COMPLETE":
        continue
    ok = True
    for mod, spec in MODULES.items():
        if ds not in spec.datasets or expected_rows(mod, ds) == 0:
            continue
        metas = glob.glob(f"{run}/outcomes/{mod}/meta-*.json")
        # NFS visibility can lag between compute and login nodes: require the newest meta to be >= 3 minutes old
        if len(metas) < SHARDS[mod] or max(os.path.getmtime(m) for m in metas) > time.time() - 180:
            ok = False; break
    if ok:
        print(f"{m} {ds}")
