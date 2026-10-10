"""C3 ceiling: how much top-10 structure of man1-256 any n-wide view can carry; tie gaps, PCA width table, anchor-fitted oracles."""
from __future__ import annotations

import json

import numpy as np
import torch

from c3_lib import ANCHORS, K, OUT, SEED, agreement, load, pca_basis, rank_loss, split

anchors = load(ANCHORS)
train, val = split()
report = {"schema": "carl.c3-ceiling/v1", "anchors": len(anchors), "k": K}

s = anchors @ anchors.T
np.fill_diagonal(s, -np.inf)
sorted_s = -np.sort(-s, axis=1)
gap = sorted_s[:, K - 1] - sorted_s[:, K]
report["tie_gap_rank10_vs_11"] = {"mean": float(gap.mean()), "median": float(np.median(gap)),
                                  "p10": float(np.percentile(gap, 10)), "below_0.005": int((gap < 0.005).sum()),
                                  "below_0.01": int((gap < 0.01).sum())}
report["sim_rank1_mean"] = float(sorted_s[:, 0].mean())
report["sim_rank10_mean"] = float(sorted_s[:, K - 1].mean())
report["sim_rank11_mean"] = float(sorted_s[:, K].mean())

widths = [64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 768]
pca_rows = []
for w in widths:
    basis = pca_basis(train, w)
    row = {"width": w, "anchors_pca_train": agreement(anchors @ basis, anchors), "val_pca_train": agreement(val @ basis, val)}
    if w < 768:
        own = pca_basis(anchors, w)
        row["anchors_pca_on_anchors_leaky"] = agreement(anchors @ own, anchors)
    for bits, scale in (("fp16", None), ("int8", 127.0)):
        v = anchors @ basis
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        if scale is None:
            q = v.astype(np.float16).astype(np.float32)
        else:
            q = np.round(v / np.abs(v).max(1, keepdims=True) * scale) / scale
        row[f"anchors_pca_train_{bits}"] = agreement(q, anchors)
        row[f"bytes_{bits}"] = w * (2 if scale is None else 1)
    pca_rows.append(row)
report["pca_width_table"] = pca_rows

# Oracles fitted on the anchors themselves (leak by construction): the best a linear map or a free embedding can do at this width.
A = torch.tensor(anchors)
oracle = []
for w in (64, 96, 128):
    torch.manual_seed(SEED)
    lin = torch.nn.Linear(768, w, bias=False)
    lin.weight.data = torch.tensor(pca_basis(anchors, w).T.copy())
    free = torch.nn.Parameter(torch.tensor(anchors @ pca_basis(anchors, w)))
    for name, params, fn in (("linear", lin.parameters(), lambda: lin(A)), ("free", [free], lambda: free)):
        opt = torch.optim.Adam(params, lr=3e-3)
        best = 0.0
        for step in range(600):
            v = torch.nn.functional.normalize(fn(), dim=1)
            loss = rank_loss(A, v, K, 60, 0.02, 0.02)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 99:
                with torch.no_grad():
                    best = max(best, agreement(torch.nn.functional.normalize(fn(), dim=1).numpy(), anchors))
        oracle.append({"width": w, "map": name, "anchors_agreement_leaky": best})
report["anchor_fitted_oracles"] = oracle
(OUT / "ceiling.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=1))
