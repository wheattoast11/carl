"""C3: NAME-style training parents from Fedora package summaries (dnf repoquery name: summary), embedded through :8083 like c2_train_corpus.py; anchors held out by text and by cosine."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "corpus-rpm"
ANCHOR_DIR = Path.home() / ".carl/experiments/encoder-regime-20261010/anchor-sets/man1-256"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
sys.path.insert(0, "/var/home/zero/codebox-os/apps/dashboard-api")
os.environ.setdefault("TERMINALS_EMBED_MODEL_DIR", "/var/home/zero/source/universes/engine/.maxwell/models/embeddinggemma-2/914f7f89142e33e77833254d9c9b90c3cef7303b")
from lib import embeddings  # noqa: E402

held = {json.loads(line)["text"] for line in (ANCHOR_DIR / "corpus.jsonl").read_text().splitlines()}
anchors = np.frombuffer((ANCHOR_DIR / "parents.f32").read_bytes(), dtype="<f4").reshape(-1, 768)
listing = subprocess.run(["dnf", "repoquery", "--available", "--qf", "%{name}: %{summary}\n"], capture_output=True, text=True, check=True).stdout
texts, seen = [], set()
for line in listing.splitlines():
    name, sep, summary = line.partition(": ")
    summary = " ".join(summary.split())
    text = f"{name}: {summary}"
    if not sep or len(summary) < 20 or summary.lower() in seen or text in held:
        continue
    seen.add(summary.lower())
    texts.append(text)
texts.sort()
if LIMIT:
    texts = texts[:: max(1, len(texts) // LIMIT)][:LIMIT]
OUT.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen("http://127.0.0.1:8083/health", timeout=10) as r:
    health = json.loads(r.read())
endpoint = {"space_ref": health["space_ref"], "execution_ref": health["execution_ref"], "execution": health["execution"],
            "binding": health["binding"], "task_roles": ["document", "query"]}
corpus_sha = hashlib.sha256("".join(json.dumps({"index": i, "text": t}) + "\n" for i, t in enumerate(texts)).encode()).hexdigest()
rows, kept, dropped, t0 = [], [], 0, time.time()
for start in range(0, len(texts), 256):
    batch = texts[start:start + 256]
    vectors = embeddings.embed(batch, task="document", index_generation_ref="sha256:" + corpus_sha, admitted_endpoint=endpoint)
    assert len(vectors) == len(batch) and all(len(v) == 768 for v in vectors)
    v = np.asarray(vectors, np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    near = (v @ anchors.T).max(1) > 0.99
    dropped += int(near.sum())
    for text, row, bad in zip(batch, v, near):
        if not bad:
            kept.append(text)
            rows.append(row)
    print(json.dumps({"done": start + len(batch), "total": len(texts), "dropped": dropped, "s": round(time.time() - t0, 1)}), flush=True)
raw = b"".join(struct.pack("<768f", *row) for row in rows)
corpus = "".join(json.dumps({"index": i, "text": t}) + "\n" for i, t in enumerate(kept))
(OUT / "train-corpus.jsonl").write_text(corpus)
(OUT / "train-parents.f32").write_bytes(raw)
receipt = {"schema": "carl.c3-corpus/v1", "source": "dnf repoquery --available name: summary, unique summaries >= 20 chars, sorted",
           "n": len(kept), "dropped_cosine_gt_0.99": dropped, "held_out_exact": len(held), "endpoint": endpoint,
           "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(), "parents_sha256": hashlib.sha256(raw).hexdigest(),
           "anchors_parents_sha256": hashlib.sha256((ANCHOR_DIR / "parents.f32").read_bytes()).hexdigest(), "seconds": round(time.time() - t0, 1)}
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt))
