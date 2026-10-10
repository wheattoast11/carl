"""C3 table: one json from every receipt in this directory; inductive 128 heads, transductive views, PCA width and byte rows, data scaling."""
from __future__ import annotations

import json
from pathlib import Path

from c3_lib import OUT, sha

inductive = []
for path in sorted(OUT.glob("receipt-*.json")):
    r = json.loads(path.read_text())
    row = {"tag": r["tag"], "loss": r["args"]["loss"], "width": r["args"]["width"], "hidden": r["args"]["hidden"], "hard": r["args"]["hard"],
           "temp": r["args"]["temp"], "steps": r["args"]["steps"], "n_train": r["n_train"], "man1_frac": r["args"].get("man1_frac"),
           "predicted": r["predicted_anchor_agreement"], "anchors": r["anchors_agreement"], "val_full": r["val_full"],
           "val_pop256_mean": r["val_pop256"]["mean"], "views_sha256": r["views_sha256"][:12]}
    if "judge" in r:
        row["judge_passed"] = r["judge"]["passed"]
        row["judge_rows"] = r["judge"]["rows"]
    inductive.append(row)
ceiling = json.loads((OUT / "ceiling.json").read_text())
table = {"schema": "carl.c3-view-breakout/v1", "owner": "carl-14", "anchor_set": "man1-256 parents.f32 605da1ea", "floor": 0.9, "k": 10,
         "c2_reference": {"kd_linear_judge": 0.818359375, "kd_mlp_judge": 0.81875, "source": "c2/c2-judge-receipt.json, c2/c2-mlp-judge-receipt.json"},
         "inductive_128_heads": inductive,
         "transductive_128_views": json.loads((OUT / "transductive.json").read_text()),
         "tie_gap_rank10_vs_11": ceiling["tie_gap_rank10_vs_11"],
         "anchor_fitted_oracles": ceiling["anchor_fitted_oracles"],
         "pca_width_bytes": [{k: row[k] for k in ("width", "anchors_pca_train", "anchors_pca_train_int8", "bytes_int8", "anchors_pca_train_fp16", "bytes_fp16")}
                             for row in ceiling["pca_width_table"]],
         "pca128_data_scaling": json.loads((OUT / "pca-scaling.json").read_text())["rows"],
         "pca_man1_population": json.loads((OUT / "pca-man1.json").read_text())}
body = json.dumps(table, indent=1, sort_keys=True) + "\n"
(OUT / "c3-table.json").write_text(body)
print(json.dumps({"rows": len(inductive), "sha256": sha(body.encode())[:12]}))
for row in inductive:
    print(row["tag"], "pred", row["predicted"], "anchors", round(row["anchors"], 4), "judge", row.get("judge_passed"))
