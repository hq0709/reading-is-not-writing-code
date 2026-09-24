"""Interface smoke test for echo-7 before committing a full prep run.

Checks the three things the campaign depends on and nothing else: the checkpoint loads through the
adapter, the masked answer position yields a yes/no margin, and two identical forwards agree bitwise.
"""
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/rodata/azradonc_dev/m253405/concept-flow/src")
from cftransfer.adapters import get_adapter                                   # noqa: E402

ad = get_adapter("echo-7").load(device_map="cuda:0")
print("loaded:", type(ad.model).__name__, "| visual blocks:", ad.n_blocks, "| merge unit:", ad.merge_unit)
q = "Is there a pleural effusion in this chest radiograph? Answer yes or no."
print("prompt tail:", repr(ad.prompt_text(q)[-90:]))

img = Image.fromarray(np.uint8(np.random.RandomState(0).rand(336, 336, 3) * 255))
enc = ad.encode([img], [q])
print("input id tail:", enc["input_ids"][0, -6:].tolist(), "| config mask id:", ad.model.config.mask_token_id)

with torch.no_grad():
    lg = ad.forward_last_logits(enc)
c = ad.candidates["IY"]
pos = torch.logsumexp(lg[0, list(c.positive_ids)], 0)
neg = torch.logsumexp(lg[0, list(c.negative_ids)], 0)
print(f"margin(yes-no) = {(pos - neg).item():+.4f}   P(yes) = {torch.sigmoid(pos - neg).item():.4f}")
top = torch.topk(lg[0], 8)
print("top-8 at the masked position:",
      [(ad.tokenizer.decode([i]), round(v.item(), 2)) for v, i in zip(top.values, top.indices)])

with torch.no_grad():
    lg2 = ad.forward_last_logits(enc)
print("determinism: max |delta logit| =", (lg - lg2).abs().max().item())
