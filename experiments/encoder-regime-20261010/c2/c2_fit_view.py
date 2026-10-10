"""C2: fit a 768->128 view head (linear, or residual MLP with --mlp) so 128-d neighbours match 768-d neighbours; anchors never seen in training."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1])
ANCHORS = Path(sys.argv[2])
MLP = "--mlp" in sys.argv
SEED, VIEW, K, TAU, STEPS, BATCH = 20261010, 128, 10, 0.05, (6000 if MLP else 3000), 1024
TAG = "mlp-" if MLP else ""


def load(path):
    x = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(-1, 768).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def agreement(views, parents, k=K):
    def nn(m):
        m = m / np.linalg.norm(m, axis=1, keepdims=True)
        s = m @ m.T
        np.fill_diagonal(s, -np.inf)
        return np.argsort(-s, axis=1)[:, :k]
    a, b = nn(views), nn(parents)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


rng = np.random.default_rng(SEED)
X = load(OUT / "train-parents.f32")
order = rng.permutation(len(X))
val, train = X[order[:1024]], X[order[1024:]]
_, _, vt = np.linalg.svd(train - train.mean(0), full_matrices=False)
W0 = vt[:VIEW].T.copy()
report = {"n_train": len(train), "n_val": len(val), "val_pca": agreement(val @ W0, val),
          "val_prefix": agreement(val[:, :VIEW], val)}

torch.manual_seed(SEED)
linear = torch.nn.Linear(768, VIEW, bias=False)
linear.weight.data = torch.tensor(W0.T.copy())
hidden = torch.nn.Sequential(torch.nn.Linear(768, 1536), torch.nn.GELU(), torch.nn.Linear(1536, VIEW))
torch.nn.init.zeros_(hidden[2].weight)
torch.nn.init.zeros_(hidden[2].bias)
modules = torch.nn.ModuleList([linear, hidden] if MLP else [linear])


def head(x):
    return linear(x) + hidden(x) if MLP else linear(x)


def apply(rows):
    with torch.no_grad():
        return head(torch.tensor(rows)).numpy()


T = torch.tensor(train)
opt = torch.optim.AdamW(modules.parameters(), lr=1e-3, weight_decay=1e-4 if MLP else 0.0)
mask = None
best = (report["val_pca"], 0, {k: v.clone() for k, v in modules.state_dict().items()})
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(T), (BATCH,))
    x = T[idx]
    if mask is None:
        mask = torch.eye(BATCH, dtype=torch.bool)
    teacher = (x @ x.T).masked_fill(mask, -1e9) / TAU
    v = torch.nn.functional.normalize(head(x), dim=1)
    student = (v @ v.T).masked_fill(mask, -1e9) / TAU
    loss = torch.nn.functional.kl_div(student.log_softmax(1), teacher.log_softmax(1), log_target=True, reduction="batchmean")
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 250 == 0:
        score = agreement(apply(val), val)
        print(json.dumps({"step": step, "loss": round(loss.item(), 5), "val": score}), flush=True)
        if score > best[0]:
            best = (score, step, {k: v.clone() for k, v in modules.state_dict().items()})

report.update(val_selected=best[0], selected_step=best[1])
modules.load_state_dict(best[2])
torch.save(best[2], OUT / (TAG + "view-head.pt"))
blob = b"".join(v.numpy().astype("<f4").tobytes() for _, v in sorted(best[2].items()))
A = load(ANCHORS)
views = apply(A)
views = (views / np.linalg.norm(views, axis=1, keepdims=True)).astype("<f4")
(OUT / (TAG + "views.f32")).write_bytes(views.tobytes())
report.update(head_sha256=hashlib.sha256(blob).hexdigest(),
              views_sha256=hashlib.sha256(views.tobytes()).hexdigest())
(OUT / (TAG + "fit-report.json")).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report))
