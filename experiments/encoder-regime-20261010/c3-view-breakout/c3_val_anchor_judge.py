"""C3: second held-out anchor set (256 c2 validation rows never trained on) judged with the shipped 256 int8 head and strix-mind rank_arm 9a3ac79."""
import json, runpy, struct, hashlib
from pathlib import Path
import numpy as np
from c3_lib import OUT, split, sha

arm = runpy.run_path("/var/home/zero/strix-mind/lib/rank_arm.py")
_, val = split()
rng = np.random.default_rng(20261010 + 256)
rows = []
W = np.frombuffer((OUT / "view-head-256.f32").read_bytes(), dtype="<f4").reshape(256, 768)
for draw in range(3):
    idx = np.sort(rng.choice(len(val), 256, replace=False))
    parents = val[idx].astype("<f4")
    views = parents @ W.T
    views = (views / np.linalg.norm(views, axis=1, keepdims=True)).astype("<f4")
    j = arm["judge"]([list(struct.unpack("<256f", r.tobytes())) for r in views],
                     [list(struct.unpack("<768f", r.tobytes())) for r in parents], element_bytes=1)
    rows.append({"draw": draw, "parents_sha256": sha(parents.tobytes()), "candidate": j["rows"]["candidate"],
                 "margin": j["rows"]["margin_over_random_projection"], "shuffled": j["rows"]["shuffled_across_anchors"], "passed": j["passed"]})
    print(json.dumps(rows[-1]), flush=True)
report = {"schema": "carl.c3-second-anchor-judge/v1", "anchor_set": "c2 validation rows (held out from head training), 3 seeded draws of 256",
          "head_sha256": sha((OUT / "view-head-256.f32").read_bytes()), "predicted_candidate": 0.917, "rows": rows}
(OUT / "val-anchor-judge.json").write_text(json.dumps(report, indent=2) + "\n")
