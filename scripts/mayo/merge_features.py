"""Concatenate the train-only and evaluation-only feature files into the canonical features/<locus>.npz."""
import json, sys
from pathlib import Path
import numpy as np
run = Path(sys.argv[1])          # .../runs/<model>/<dataset>
for lid in ("vis.last", "connector"):
    parts = [np.load(run / "features_train" / f"{lid}.npz"), np.load(run / "features" / f"{lid}.npz")]
    ids = np.concatenate([p["row_id"] for p in parts])
    assert len(set(ids.tolist())) == len(ids), "duplicate row ids across feature parts"
    np.savez(run / "features" / f"{lid}.npz", row_id=ids, x=np.concatenate([p["x"] for p in parts]),
             valid_token_count=np.concatenate([p["valid_token_count"] for p in parts]),
             fit_role=np.concatenate([p["fit_role"] for p in parts]))
    print(lid, ids.shape)
m = json.loads((run / "features_train" / "features_meta.json").read_text())
e = json.loads((run / "features" / "features_meta.json").read_text())
m["rows"] = m["rows"] + e["rows"]; m["seconds"] = m["seconds"] + e["seconds"]; m["roles"] = ["train", "preflight", "calibration", "test"]
(run / "features" / "features_meta.json").write_text(json.dumps(m, indent=1))
print("merged", m["rows"], "rows")
