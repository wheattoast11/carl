"""C2: encode every man-page description outside the A1 anchor set as 768-d training parents."""
import hashlib
import json
import os
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

OUT = Path(sys.argv[1])
ANCHORS = Path(sys.argv[2])
sys.path.insert(0, "/var/home/zero/codebox-os/apps/dashboard-api")
os.environ.setdefault("TERMINALS_EMBED_MODEL_DIR", "/var/home/zero/source/universes/engine/.maxwell/models/embeddinggemma-2/914f7f89142e33e77833254d9c9b90c3cef7303b")
from lib import embeddings  # noqa: E402

held = {json.loads(line)["text"] for line in ANCHORS.read_text().splitlines()}
listing = subprocess.run(["apropos", "."], capture_output=True, text=True, check=True).stdout
texts, seen = [], set()
for line in listing.splitlines():
    head, sep, desc = line.partition(" - ")
    desc = " ".join(desc.split())
    name = head.split(" (")[0].strip()
    text = f"{name}: {desc}"
    if not sep or len(desc) < 20 or desc in seen or text in held:
        continue
    seen.add(desc)
    texts.append(text)
texts.sort()
OUT.mkdir(parents=True, exist_ok=True)
corpus = "".join(json.dumps({"index": i, "text": t}) + "\n" for i, t in enumerate(texts))
(OUT / "train-corpus.jsonl").write_text(corpus)
corpus_sha = hashlib.sha256(corpus.encode()).hexdigest()
with urllib.request.urlopen("http://127.0.0.1:8083/health", timeout=10) as r:
    health = json.loads(r.read())
endpoint = {"space_ref": health["space_ref"], "execution_ref": health["execution_ref"], "execution": health["execution"],
            "binding": health["binding"], "task_roles": ["document", "query"]}
raw = bytearray()
for start in range(0, len(texts), 256):
    batch = texts[start:start + 256]
    vectors = embeddings.embed(batch, task="document", index_generation_ref="sha256:" + corpus_sha, admitted_endpoint=endpoint)
    assert len(vectors) == len(batch) and all(len(v) == 768 for v in vectors)
    raw += b"".join(struct.pack("<768f", *v) for v in vectors)
    print(start + len(batch), len(texts), flush=True)
(OUT / "train-parents.f32").write_bytes(bytes(raw))
print(json.dumps({"n": len(texts), "held_out": len(held), "corpus_sha256": corpus_sha,
                  "parents_sha256": hashlib.sha256(raw).hexdigest()}))
