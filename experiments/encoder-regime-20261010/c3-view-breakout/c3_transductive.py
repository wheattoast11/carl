"""C3 transductive arm: fit the 128 view on the indexed set itself (anchors, or anchors plus train), unsupervised; judged separately from the inductive heads."""
from __future__ import annotations

import json
import runpy
import struct
from pathlib import Path

import numpy as np
import torch

from c3_lib import ANCHORS, K, OUT, SEED, agreement, load, pca_basis, rank_loss, sha, split

RANK_ARM = Path("/var/home/zero/strix-mind/lib/rank_arm.py")
arm = runpy.run_path(str(RANK_ARM))
parents = arm["read_f32"](ANCHORS, arm["PARENT"])
anchors = load(ANCHORS)
train, _ = split()


def judge(views: np.ndarray) -> dict:
    return arm["judge"]([list(struct.unpack("<128f", r.astype("<f4").tobytes())) for r in views], parents)


def norm(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def fit_rank(rows: np.ndarray, w0: np.ndarray, steps: int, pop: int, seed: int = SEED) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lin = torch.nn.Linear(768, 128, bias=False)
    lin.weight.data = torch.tensor(w0.T.copy())
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    R = torch.tensor(rows)
    for _ in range(steps):
        idx = torch.tensor(rng.choice(len(R), min(pop, len(R)), replace=False))
        x = R[idx]
        loss = rank_loss(x, torch.nn.functional.normalize(lin(x), dim=1), K, 60, 0.02, 0.02)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lin.weight.data.numpy().T.copy()


rows = []
basis = pca_basis(anchors, 128)
rows.append(("pca-anchors", "transductive: PCA on the 256 indexed anchors", norm(anchors @ basis)))
rows.append(("rank-anchors", "transductive: rank loss on the 256 indexed anchors, PCA init", norm(anchors @ fit_rank(anchors, basis, 600, 256))))
joint = np.concatenate([train, anchors])
jb = pca_basis(joint, 128)
rows.append(("pca-joint", "transductive: PCA on train plus anchors", norm(anchors @ jb)))
rows.append(("rank-joint", "transductive: rank loss on train plus anchors (anchors 1 of 23 rows)", norm(anchors @ fit_rank(joint, jb, 1500, 256))))
weighted = np.concatenate([train, np.repeat(anchors, 8, axis=0)])
rows.append(("rank-joint-w8", "transductive: rank loss on train plus anchors repeated 8x", norm(anchors @ fit_rank(weighted, jb, 1500, 256))))
report = {"schema": "carl.c3-transductive/v1", "evidence_class": "executed; views fit on the judged anchors, not an inductive encoder", "rows": []}
for tag, what, views in rows:
    (OUT / f"views-transductive-{tag}.f32").write_bytes(views.astype("<f4").tobytes())
    j = judge(views)
    report["rows"].append({"tag": tag, "what": what, "judge_passed": j["passed"], "rows": j["rows"], "views_sha256": sha(views.astype("<f4").tobytes())})
    print(json.dumps(report["rows"][-1]), flush=True)
report["rank_arm_sha256"] = sha(RANK_ARM.read_bytes())
(OUT / "transductive.json").write_text(json.dumps(report, indent=2) + "\n")
