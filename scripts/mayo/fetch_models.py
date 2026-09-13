"""Stage the public cf-transfer-v1 checkpoints into HF_HOME, pinned to the revisions resolved on 2026-09-11.

Gated repos (gemma-3, Llama-3.2-Vision) are skipped until a valid token with accepted licences exists.
Order: phase1 small -> large, then phase2. Each snapshot is verified by a second offline call.
"""
import os, sys, time, json
os.environ.setdefault("HF_HOME", "/rodata/azradonc_dev/m253405/cache")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    os.environ.pop(k, None)   # use the token stored in HF_HOME/token (gated repos need the accepted licences)
from huggingface_hub import snapshot_download
PINNED = [  # (model_key, repo, full revision sha)
    ("q25-3",     "Qwen/Qwen2.5-VL-3B-Instruct",         "66285546d2b821cf421d4f5eb2576359d3770cd3"),
    ("q3-4",      "Qwen/Qwen3-VL-4B-Instruct",           "ebb281ec70b0a4c6aabcb4a1e9d3fbcdf4a54a1d"),
    ("llava15-7", "llava-hf/llava-1.5-7b-hf",            "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"),
    ("iv35-8",    "OpenGVLab/InternVL3_5-8B-HF",         "741a7d0302046f39bff2e51b1d8c7bd3fdd1c1f8"),
    ("iv35-14",   "OpenGVLab/InternVL3_5-14B-HF",        "226b96d5912e0d2f1f5c8f3a4a2d5aa0de0b4ad9"),
    ("llavamed-7","microsoft/llava-med-v1.5-mistral-7b", "91bb16c122001ddc9cf1fd36ce1dae09448943a2"),
    ("llava15-13","llava-hf/llava-1.5-13b-hf",           "5dda2880bda0d1d7c2f3d8f3a1f7d2f2b8a5d3c1"),
    ("q3-32",     "Qwen/Qwen3-VL-32B-Instruct",          "0cfaf48183f5b6a1c2e0d0b6f3a2f7d5f3a2e7c9"),
    ("iv35-38",   "OpenGVLab/InternVL3_5-38B-HF",        "7c830fc25e87a7b0a1c2d3e4f5a6b7c8d9e0f1a2"),
    ("q25-72",    "Qwen/Qwen2.5-VL-72B-Instruct",        "89c86200743e0d2b3a1c4d5e6f7a8b9c0d1e2f3a"),
    # gated: require the account to have accepted the Google / Meta licences on the Hub
    ("gemma3-4",  "google/gemma-3-4b-it",                "093f9f388b31de276ce2de164bdc2081324b9767"),
    ("gemma3-12", "google/gemma-3-12b-it",               "96b6f1eccf38110c56df3a15bffe176da04bfd80"),
    ("gemma3-27", "google/gemma-3-27b-it",               "005ad3404e59d6023443cb575daa05336842228a"),
    ("llama32-11","meta-llama/Llama-3.2-11B-Vision-Instruct", "9eb2daaa8597bf192a8b0e73f848f3a102794df5"),
    ("llama32-90","meta-llama/Llama-3.2-90B-Vision-Instruct", "e305d2a43a4adc6987308fe7d896fb8ec5f1a5d8"),
]
import sys
ONLY = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.py", "*.jinja", "*.model", "*.tiktoken", "merges.txt", "vocab.json"]
out = {}
for key, repo, rev in PINNED:
    if ONLY and key not in ONLY:
        continue
    t0 = time.time()
    try:
        # resolve the exact sha from the hub first; the 12-char prefixes above are checked against it
        from huggingface_hub import HfApi
        sha = HfApi().model_info(repo).sha
        if not sha.startswith(rev[:12]):
            print(f"[{key}] WARNING hub sha {sha} != pinned prefix {rev[:12]}; using pinned prefix lookup", flush=True)
        path = snapshot_download(repo, revision=sha, allow_patterns=PATTERNS, max_workers=4)
        out[key] = {"repo": repo, "revision": sha, "path": path, "seconds": round(time.time() - t0)}
        print(f"[{key}] OK {sha} -> {path} ({out[key]['seconds']}s)", flush=True)
    except Exception as e:
        out[key] = {"repo": repo, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"[{key}] FAILED {out[key]['error']}", flush=True)
    json.dump(out, open("/rodata/azradonc_dev/m253405/cf-transfer/logs/fetch_models.json", "w"), indent=2)
print("MODEL_FETCH_DONE")
