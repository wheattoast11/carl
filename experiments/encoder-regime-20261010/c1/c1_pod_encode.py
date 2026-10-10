"""C1: encode the man1-256 anchor texts with the base encoder and the adapter checkpoint on the pod."""
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch
from carl_encoders import worker

corpus, model_path, checkpoint, out = (Path(a) for a in sys.argv[1:5])
device = sys.argv[5] if len(sys.argv) > 5 else "cuda:0"
texts = [json.loads(line)["text"] for line in corpus.read_text().splitlines()]
out.mkdir(parents=True, exist_ok=True)


def encode(model):
    model.eval()
    rows = []
    with torch.inference_mode():
        for text in texts:
            request = {"parts": [{"modality": "text", "text": text}], "recipe": "document", "max_tokens": 2048}
            rows.append(worker.raw_forward(model, worker.features_for(model, request))[0].float().cpu().tolist())
    assert len(rows) == 256 and all(len(r) == 768 for r in rows)
    return b"".join(struct.pack("<768f", *r) for r in rows)


receipt = {"corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(), "checkpoint": str(checkpoint)}
for name, restore in (("base768.f32", False), ("cand768.f32", True)):
    model = worker.load_model(str(model_path), device, {"text"}, "float32")
    if restore:
        model, _ = worker.restore_candidate(model, str(checkpoint))
        receipt["trainer_state"] = json.loads((checkpoint / "trainer_state.json").read_text()).get("status")
    raw = encode(model)
    (out / name).write_bytes(raw)
    receipt[name] = hashlib.sha256(raw).hexdigest()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
(out / "encode-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt))
