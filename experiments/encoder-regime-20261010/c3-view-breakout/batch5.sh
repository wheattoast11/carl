#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f corpus-rpm/receipt.json ]; do sleep 10; done
export OMP_NUM_THREADS=6
PY=/var/home/zero/carl/.venv/bin/python
$PY - <<'PYEOF' > pca-rpm.json
import json, numpy as np
from c3_lib import rpm, load, ANCHORS, pca_basis, agreement, split
anchors = load(ANCHORS)
x, val, _ = rpm(plus_c2=False)
c2 = split()[0]
out = {"rows_rpm_train": len(x)}
for w in (128, 192, 256):
    out[f"pca{w}_rpm"] = agreement(anchors @ pca_basis(x, w), anchors)
    out[f"pca{w}_rpm_plus_c2"] = agreement(anchors @ pca_basis(np.concatenate([x, c2]), w), anchors)
rng = np.random.default_rng(1)
for n in (6000, 12000, 24000):
    out[f"pca128_rpm_n{n}"] = agreement(anchors @ pca_basis(x[rng.choice(len(x), n, replace=False)], 128), anchors)
print(json.dumps(out, indent=1))
PYEOF
cat pca-rpm.json
$PY c3_train.py --tag rpm-rank128-mlp --corpus rpm --loss rank --hidden 1536 --hard 100 --temp 0.01 --pop 256 --groups 4 --steps 2500 --predicted 0.86 --judge > fit-rpm-rank128-mlp.log 2>&1
$PY c3_train.py --tag rpm-rank128-linear --corpus rpm --loss rank --hard 100 --temp 0.01 --pop 256 --groups 4 --steps 2000 --predicted 0.845 --judge > fit-rpm-rank128-linear.log 2>&1
echo BATCH5-DONE > batch5.done
