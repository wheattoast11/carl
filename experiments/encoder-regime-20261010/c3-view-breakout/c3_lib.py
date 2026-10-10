"""C3: shared data, agreement metric and view-head training for the 768->n view study on man1-256."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path.home() / ".carl/experiments/encoder-regime-20261010"
TRAIN = ROOT / "c2/train-parents.f32"
LARGE = ROOT / "c2/corpus-large"
RPM = Path(__file__).resolve().parent / "corpus-rpm"
ANCHORS = ROOT / "anchor-sets/man1-256/parents.f32"
OUT = Path(__file__).resolve().parent
SEED, K = 20261010, 10
torch.set_num_threads(6)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> np.ndarray:
    x = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(-1, 768).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def split(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Same permutation as c2_fit_view.py: first 1024 rows are validation."""
    x = load(TRAIN)
    order = np.random.default_rng(seed).permutation(len(x))
    return x[order[1024:]], x[order[:1024]]


def large(sec1: float = 0.3, ossl: float = 0.05, plus_c2: bool = True, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """corpus-large rows (1024 held out for val) with sampling weights: section 1 mass `sec1`, 3ossl mass `ossl`, rest shared; c2 train appended at the rest weight."""
    x = load(LARGE / "train-parents.f32")
    sections = np.array([json.loads(line)["section"] for line in (LARGE / "train-corpus.jsonl").read_text().splitlines()])
    order = np.random.default_rng(seed).permutation(len(x))
    val, keep = x[order[:1024]], order[1024:]
    x, sections = x[keep], sections[keep]
    if plus_c2:
        c2 = split(seed)[0]
        x = np.concatenate([x, c2])
        sections = np.concatenate([sections, np.array(["c2"] * len(c2))])
    is1, isossl = sections == "1", sections == "3ossl"
    rest = ~(is1 | isossl)
    weights = np.where(is1, sec1 / is1.sum(), np.where(isossl, ossl / isossl.sum(), (1 - sec1 - ossl) / rest.sum()))
    return x, val, weights


def rpm(plus_c2: bool = True, c2_mass: float = 0.3, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """corpus-rpm rows (1024 held out for val), c2 train appended with sampling mass `c2_mass`."""
    x = load(RPM / "train-parents.f32")
    order = np.random.default_rng(seed).permutation(len(x))
    val, x = x[order[:1024]], x[order[1024:]]
    weights = np.full(len(x), 1.0 / len(x))
    if plus_c2:
        c2 = split(seed)[0]
        weights = np.concatenate([np.full(len(x), (1 - c2_mass) / len(x)), np.full(len(c2), c2_mass / len(c2))])
        x = np.concatenate([x, c2])
    return x, val, weights


def topk(m: np.ndarray, k: int = K) -> np.ndarray:
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    s = m @ m.T
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1)[:, :k]


def agreement(views: np.ndarray, parents: np.ndarray, k: int = K) -> float:
    a, b = topk(views, k), topk(parents, k)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def subset_agreement(views: np.ndarray, parents: np.ndarray, size: int = 256, draws: int = 32, seed: int = SEED) -> dict:
    """Judge regime: top-k agreement inside random populations of `size`, like the 256 anchors."""
    rng = np.random.default_rng(seed)
    scores = [agreement(views[idx], parents[idx]) for idx in (rng.choice(len(views), size, replace=False) for _ in range(draws))]
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "min": float(np.min(scores)), "draws": draws, "size": size}


def pca_basis(train: np.ndarray, width: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(train - train.mean(0), full_matrices=False)
    return vt[:width].T.copy()


class Head(torch.nn.Module):
    def __init__(self, w0: np.ndarray, hidden: int = 0):
        super().__init__()
        self.linear = torch.nn.Linear(768, w0.shape[1], bias=False)
        self.linear.weight.data = torch.tensor(w0.T.copy())
        self.mlp = None
        if hidden:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(768, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, w0.shape[1]))
            torch.nn.init.zeros_(self.mlp[2].weight)
            torch.nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        y = self.linear(x)
        return y + self.mlp(x) if self.mlp is not None else y

    def apply(self, rows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            v = self.forward(torch.tensor(rows)).numpy()
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def kd_loss(x, v, tau):
    mask = torch.eye(len(x), dtype=torch.bool)
    teacher = (x @ x.T).masked_fill(mask, -1e9) / tau
    student = (v @ v.T).masked_fill(mask, -1e9) / tau
    return torch.nn.functional.kl_div(student.log_softmax(1), teacher.log_softmax(1), log_target=True, reduction="batchmean")


def rank_loss(x, v, k, hard, margin, temp):
    """Pairwise logistic loss: each of the k teacher neighbours must outrank the next `hard` teacher ranks in the student."""
    n = len(x)
    mask = torch.eye(n, dtype=torch.bool)
    teacher = (x @ x.T).masked_fill(mask, -1e9)
    order = torch.argsort(teacher, dim=1, descending=True)
    pos, neg = order[:, :k], order[:, k:k + hard]
    student = v @ v.T
    s_pos = torch.gather(student, 1, pos)
    s_neg = torch.gather(student, 1, neg)
    diff = s_neg[:, None, :] - s_pos[:, :, None] + margin
    return torch.nn.functional.softplus(diff / temp).mean() * temp


def train_head(train: np.ndarray, val: np.ndarray, width: int, loss: str, steps: int, pop: int, groups: int, lr: float,
               hidden: int = 0, tau: float = 0.05, hard: int = 50, margin: float = 0.02, temp: float = 0.02,
               mix: float = 0.5, eval_every: int = 250, seed: int = SEED, log=None, weights: np.ndarray | None = None) -> tuple[Head, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if weights is not None:
        weights = weights / weights.sum()
    head = Head(pca_basis(train, width), hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4 if hidden else 0.0)
    T = torch.tensor(train)
    best = (agreement(head.apply(val), val), 0, {k: v.clone() for k, v in head.state_dict().items()})
    history = [{"step": 0, "val": best[0]}]
    t0 = time.time()
    for step in range(1, steps + 1):
        total = 0.0
        opt.zero_grad()
        for _ in range(groups):
            idx = torch.tensor(rng.choice(len(T), pop, replace=False, p=weights))
            x = T[idx]
            v = torch.nn.functional.normalize(head(x), dim=1)
            if loss == "kd":
                part = kd_loss(x, v, tau)
            elif loss == "rank":
                part = rank_loss(x, v, K, hard, margin, temp)
            else:
                part = mix * kd_loss(x, v, tau) + (1 - mix) * rank_loss(x, v, K, hard, margin, temp)
            (part / groups).backward()
            total += part.item() / groups
        opt.step()
        if step % eval_every == 0:
            score = agreement(head.apply(val), val)
            row = {"step": step, "loss": round(total, 5), "val": score, "s": round(time.time() - t0, 1)}
            history.append(row)
            if log:
                log(json.dumps(row))
            if score > best[0]:
                best = (score, step, {k: v.clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[2])
    return head, {"val_selected": best[0], "selected_step": best[1], "history": history}
