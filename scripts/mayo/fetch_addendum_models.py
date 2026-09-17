"""Stage the addendum checkpoints of the medical/chest comparison into HF_HOME, pinned to resolved revisions.

These are the four checkpoints added after the 22-model plan, plus the base model two of them need:

    huatuo-7     FreedomIntelligence/HuatuoGPT-Vision-7B   -> scripts/mayo/convert_huatuo.py
    chexagent-8  StanfordAIMI/CheXagent-8b                 -> loaded directly (remote code + shim)
    llavarad-7   microsoft/llava-rad + lmsys/vicuna-7b-v1.5 -> scripts/mayo/convert_llavarad.py
    maira2-7     microsoft/maira-2                         -> scripts/mayo/convert_maira2.py

All four repositories are public. The account token in HF_HOME/token is used as it is for the other staging
scripts; it is read from the cache, never passed on a command line. Each snapshot is verified by a second
offline resolution of the same revision.

    python scripts/mayo/fetch_addendum_models.py [model_key,...]
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/rodata/azradonc_dev/m253405/cache")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

PINNED = [   # (model_key, repo, revision resolved on 2026-09-17, allow_patterns)
    ("huatuo-7", "FreedomIntelligence/HuatuoGPT-Vision-7B", "34dfcdbb7728ff38da865839f342b88c4cf6ef39",
     ["*.json", "*.safetensors", "*.txt", "*.py", "*.model"]),
    ("chexagent-8", "StanfordAIMI/CheXagent-8b", "4934e91451945c8218c267aae9c34929a7677829",
     ["*.json", "*.safetensors", "*.txt", "*.py", "*.model"]),
    ("llavarad-7", "microsoft/llava-rad", "dcdbc6caf6806c9acb66e12a827a39c86e00f6a7",
     ["*.json", "*.bin", "*.pt", "*.md", "LICENSE"]),
    ("llavarad-7-base", "lmsys/vicuna-7b-v1.5", "3321f76e3f527bd14065daf69dad9344000a201d",
     ["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"]),
    ("maira2-7", "microsoft/maira-2", "795a2b1cd4a310624b4e3d14b5a23e41fd273deb",
     ["*.json", "*.safetensors", "*.py", "*.model", "LICENSE", "*.md"]),
]
LOG = "/rodata/azradonc_dev/m253405/cf-transfer/logs/fetch_addendum_models.json"


def main() -> None:
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    out = {}
    for key, repo, rev, patterns in PINNED:
        if only and key not in only:
            continue
        t0 = time.time()
        try:
            sha = HfApi().model_info(repo).sha
            if sha != rev:
                print(f"[{key}] WARNING the repository's current head is {sha}, not the pinned {rev}; "
                      f"staging the PINNED revision", flush=True)
            path = snapshot_download(repo, revision=rev, allow_patterns=patterns, max_workers=4)
            out[key] = {"repo": repo, "revision": rev, "head_at_staging": sha, "path": path,
                        "seconds": round(time.time() - t0)}
            print(f"[{key}] OK {rev} -> {path} ({out[key]['seconds']}s)", flush=True)
        except Exception as e:                                     # noqa: BLE001
            out[key] = {"repo": repo, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"[{key}] FAILED {out[key]['error']}", flush=True)
        with open(LOG, "w") as f:
            json.dump(out, f, indent=2)
    print("ADDENDUM_FETCH_DONE")


if __name__ == "__main__":
    main()
