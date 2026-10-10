"""C3 train: fit one 768->width view head under a named loss, score it on val and on the man1-256 anchors, write a receipt."""
from __future__ import annotations

import argparse
import json
import runpy
import struct
import sys
from pathlib import Path

import numpy as np

from c3_lib import ANCHORS, OUT, agreement, large, load, rpm, sha, split, subset_agreement, train_head

RANK_ARM = Path("/var/home/zero/strix-mind/lib/rank_arm.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--loss", choices=("kd", "rank", "mix"), default="rank")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--hard", type=int, default=50)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--predicted", type=float, help="predicted anchor agreement, written before the run")
    ap.add_argument("--man1-frac", type=float, help="sampling mass given to man section 1 rows (man1-mask.npy)")
    ap.add_argument("--corpus", choices=("c2", "large", "rpm"), default="c2")
    ap.add_argument("--sec1", type=float, default=0.3)
    ap.add_argument("--ossl", type=float, default=0.05)
    ap.add_argument("--no-c2", action="store_true")
    ap.add_argument("--subset", type=int, help="train on this many rows drawn once by seed")
    ap.add_argument("--judge", action="store_true", help="run strix-mind rank_arm.judge (128 wide only)")
    a = ap.parse_args()
    if a.corpus == "large":
        train, val, large_weights = large(a.sec1, a.ossl, not a.no_c2)
    elif a.corpus == "rpm":
        train, val, large_weights = rpm(not a.no_c2, a.sec1)
    else:
        train, val = split()
        large_weights = None
    if a.subset:
        train = train[np.random.default_rng(20261010 + a.subset).choice(len(train), a.subset, replace=False)]
    anchors = load(ANCHORS)
    weights = large_weights
    if a.man1_frac is not None and a.corpus == "c2":
        full = np.load(OUT / "man1-mask.npy")
        mask = full[np.random.default_rng(20261010).permutation(len(full))[1024:]]
        weights = np.where(mask, a.man1_frac / mask.sum(), (1 - a.man1_frac) / (~mask).sum())
    head, fit = train_head(train, val, a.width, a.loss, a.steps, a.pop, a.groups, a.lr, a.hidden, a.tau, a.hard, a.margin,
                           a.temp, a.mix, log=lambda s: print(s, flush=True), weights=weights)
    views = head.apply(anchors)
    receipt = {"schema": "carl.c3-view-fit/v1", "tag": a.tag, "args": vars(a), "n_train": len(train), "n_val": len(val),
               "val_full": fit["val_selected"], "val_pop256": subset_agreement(head.apply(val), val),
               "anchors_agreement": agreement(views, anchors), "selected_step": fit["selected_step"], "history": fit["history"],
               "views_sha256": sha(views.astype("<f4").tobytes()), "predicted_anchor_agreement": a.predicted}
    (OUT / f"views-{a.tag}.f32").write_bytes(views.astype("<f4").tobytes())
    if a.judge and a.width == 128:
        arm = runpy.run_path(str(RANK_ARM))
        parents = arm["read_f32"](ANCHORS, arm["PARENT"])
        cands = [list(struct.unpack("<128f", r.astype("<f4").tobytes())) for r in views]
        receipt["judge"] = arm["judge"](cands, parents)
        receipt["rank_arm_sha256"] = sha(RANK_ARM.read_bytes())
    (OUT / f"receipt-{a.tag}.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in ("tag", "val_full", "val_pop256", "anchors_agreement", "selected_step", "predicted_anchor_agreement")}
                     | ({"judge_passed": receipt["judge"]["passed"], "judge_rows": receipt["judge"]["rows"]} if "judge" in receipt else {})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
