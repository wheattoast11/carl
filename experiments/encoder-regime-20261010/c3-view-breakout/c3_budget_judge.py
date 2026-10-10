"""C3 ruling 60: judge byte-budget views (PCA int8 at 320 and 384, best inductive 128 fp32) with the amended rank_arm from strix-scratch."""
import json, runpy, struct
from pathlib import Path
import numpy as np
from c3_lib import ANCHORS, OUT, load, pca_basis, sha, split

ARM_PATH = OUT / "strix-scratch/strix-mind/lib/rank_arm.py"
arm = runpy.run_path(str(ARM_PATH))
parents = arm["read_f32"](ANCHORS, arm["PARENT"])
anchors = load(ANCHORS)
train, _ = split()
rows = []
for width in (320, 384):
    v = anchors @ pca_basis(train, width)
    v = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype("<f4")
    (OUT / f"views-pca{width}.f32").write_bytes(v.tobytes())
    cands = [list(struct.unpack(f"<{width}f", r.tobytes())) for r in v]
    j = arm["judge"](cands, parents, element_bytes=1)
    rows.append({"tag": f"pca{width}-int8", "basis": "PCA on c2 train (5771 rows)", "views_sha256": sha(v.tobytes()), "judge": j})
    print(json.dumps({"tag": rows[-1]["tag"], "passed": j["passed"], "rows": j["rows"]}), flush=True)
best = np.frombuffer((OUT / "views-rank128-mlp-man1-0.7.f32").read_bytes(), dtype="<f4").reshape(-1, 128)
j = arm["judge"]([list(struct.unpack("<128f", r.tobytes())) for r in best], parents, element_bytes=4)
rows.append({"tag": "rank128-mlp-man1-0.7-fp32", "basis": "best inductive 128 head", "views_sha256": sha(best.tobytes()), "judge": j})
print(json.dumps({"tag": rows[-1]["tag"], "passed": j["passed"], "rows": j["rows"]}), flush=True)
report = {"schema": "carl.c3-budget-judge/v1", "ruling": 60, "budget_bytes": 512, "rank_arm": str(ARM_PATH), "rank_arm_sha256": sha(ARM_PATH.read_bytes()),
          "patch_sha256": sha((OUT / "rank-arm-byte-budget.patch").read_bytes()), "rows": rows}
(OUT / "budget-judge.json").write_text(json.dumps(report, indent=2) + "\n")
