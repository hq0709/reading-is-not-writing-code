"""model_key -> adapter factory. Revisions are the full commits resolved on 2026-09-11 (locked config)."""
from __future__ import annotations

from ..protocol import MODELS
from .gemma import Gemma3Adapter
from .internvl import InternVLAdapter
from .llava import LlavaAdapter
from .llavamed import LlavaMedAdapter
from .qwen import Qwen3VLAdapter, QwenVLAdapter

REVISIONS = {
    "q25-3": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "q25-7": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "q25-32": "7cfb30d71a1f4f49a57592323337a4a4727301da",
    "q25-72": "89c86200743eec961a297729e7990e8f2ddbc4c5",
    "q3-4": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "q3-8": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "q3-32": "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
    "iv35-8": "741a7d03020411e666c6109218ab71e08151ef86", "iv35-14": "226b96d5912e69159abc0384cefcbd51487fdce0", "iv35-38": "7c830fc25e874f9a861087e3c70587f2b9b555f3",
    "gemma3-4": "093f9f388b31de276ce2de164bdc2081324b9767", "gemma3-12": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
    "gemma3-27": "005ad3404e59d6023443cb575daa05336842228a",
    "medgemma-4": "290cda5eeccbee130f987c4ad74a59ae6f196408",
    "medgemma-27": "2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f",
    "llama32-11": None, "llama32-90": None,
    "llava15-7": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "llava15-13": "5dda2880bda009266dda7c4baff660b95ca64540",
    "lingshu-7": "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9",
    "lingshu-32": "36b98277cacb60db86f34b75ce0540b1ea35183c",
    "llavamed-7": "91bb16c122001ddc9cf1fd36ce1dae09448943a2",
}

FAMILY_ADAPTER = {
    "q25-3": QwenVLAdapter, "q25-7": QwenVLAdapter, "q25-32": QwenVLAdapter, "q25-72": QwenVLAdapter,
    "lingshu-7": QwenVLAdapter, "lingshu-32": QwenVLAdapter,
    "q3-4": Qwen3VLAdapter, "q3-8": Qwen3VLAdapter, "q3-32": Qwen3VLAdapter,
    "gemma3-4": Gemma3Adapter, "gemma3-12": Gemma3Adapter, "gemma3-27": Gemma3Adapter,
    "medgemma-4": Gemma3Adapter, "medgemma-27": Gemma3Adapter,
    "iv35-8": InternVLAdapter, "iv35-14": InternVLAdapter, "iv35-38": InternVLAdapter,
    "llava15-7": LlavaAdapter, "llava15-13": LlavaAdapter,
    "llavamed-7": LlavaMedAdapter,
}


def get_adapter(model_key: str, revision: str | None = None):
    if model_key not in FAMILY_ADAPTER:
        raise KeyError(f"no adapter registered for {model_key}")
    rev = revision or REVISIONS.get(model_key)
    if not rev:
        raise RuntimeError(f"{model_key}: revision not locked yet (checkpoint not staged)")
    return FAMILY_ADAPTER[model_key](model_key, MODELS[model_key]["model_id"], rev)
