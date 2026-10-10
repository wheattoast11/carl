"""C3: corpus-large shas, section counts and PCA 128/256 agreement on the anchors."""
import collections, hashlib, json, sys
import numpy as np
from pathlib import Path
from c3_lib import LARGE, ANCHORS, load, pca_basis, agreement, split
out = {"schema": "carl.c3-pca-large/v1"}
for f in ("train-parents.f32", "train-corpus.jsonl"):
    out[f"{f}_sha256"] = hashlib.sha256((LARGE / f).read_bytes()).hexdigest()
X = load(LARGE / "train-parents.f32"); anchors = load(ANCHORS)
sec = np.array([json.loads(l)["section"] for l in (LARGE / "train-corpus.jsonl").read_text().splitlines()])
out["rows"] = len(X); out["sections_top"] = collections.Counter(sec.tolist()).most_common(6)
m = sec != "3ossl"; m1 = sec == "1"; c2 = split()[0]
out["pca128_large_all"] = agreement(anchors @ pca_basis(X, 128), anchors)
out["pca128_large_no3ossl"] = agreement(anchors @ pca_basis(X[m], 128), anchors)
out["pca128_large_sec1_only"] = agreement(anchors @ pca_basis(X[m1], 128), anchors)
out["pca128_large_no3ossl_plus_c2"] = agreement(anchors @ pca_basis(np.concatenate([X[m], c2]), 128), anchors)
out["pca256_large_no3ossl"] = agreement(anchors @ pca_basis(X[m], 256), anchors)
out["max_cos_large_to_anchor"] = float((X @ anchors.T).max())
Path("pca-large.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
