#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=6
PY=/var/home/zero/carl/.venv/bin/python
$PY - <<'PYEOF' > pca-man1.json
import json, numpy as np
from c3_lib import split, load, ANCHORS, OUT, pca_basis, agreement
train, val = split(); anchors = load(ANCHORS)
mask = np.load(OUT / "man1-mask.npy")[np.random.default_rng(20261010).permutation(6795)[1024:]]
out = {"man1_train_rows": int(mask.sum())}
for w in (128, 192, 256):
    out[f"anchors_pca_man1only_{w}"] = agreement(anchors @ pca_basis(train[mask], w), anchors)
    out[f"anchors_pca_all_{w}"] = agreement(anchors @ pca_basis(train, w), anchors)
print(json.dumps(out, indent=1))
PYEOF
$PY c3_train.py --tag mix128 --loss mix --pop 256 --groups 4 --steps 1500 --predicted 0.835 --judge > fit-mix128.log 2>&1
$PY c3_train.py --tag rank128-hard100-t01 --loss rank --hard 100 --temp 0.01 --pop 256 --groups 4 --steps 1500 --predicted 0.83 --judge > fit-rank128-hard100-t01.log 2>&1
$PY c3_train.py --tag rank128-mlp --loss rank --hidden 1536 --pop 256 --groups 4 --steps 1500 --predicted 0.835 --judge > fit-rank128-mlp.log 2>&1
$PY c3_train.py --tag rank128-man1-0.7 --loss rank --man1-frac 0.7 --pop 256 --groups 4 --steps 1500 --predicted 0.84 --judge > fit-rank128-man1-0.7.log 2>&1
echo BATCH1-DONE > batch1.done
