"""E1: embed man1-256 through one Bonsai 2 group variant, judge it with rank_arm, and read tok/s with llama-bench."""
from __future__ import annotations

import hashlib
import json
import runpy
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

RUNTIME = Path("/var/home/zero/llm-workspace/runtimes/llama-bonsai2-vk-ss-b1e2013/bin")
ANCHORS = Path.home() / ".carl/experiments/encoder-regime-20261010/anchor-sets/man1-256"
RANK_ARM = Path("/var/home/zero/strix-mind/lib/rank_arm.py")
OUT = Path(__file__).resolve().parent
PROJECTION_SEED = 20261010 ^ 0xE1


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def embed(model: Path, label: str, texts: list[str]) -> np.ndarray:
    port = free_port()
    log = (OUT / f"server-{label}.log").open("w")
    proc = subprocess.Popen([str(RUNTIME / "llama-server"), "-m", str(model), "--alias", f"e1-{label}", "--host", "127.0.0.1",
                             "--port", str(port), "--embeddings", "--pooling", "mean", "-ngl", "99", "-c", "4096",
                             "-b", "4096", "-ub", "4096", "--parallel", "1", "--no-webui"], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            if proc.poll() is not None:
                raise RuntimeError(f"server {label} exited {proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except OSError:
                pass
            time.sleep(1)
        rows = []
        for i in range(0, len(texts), 16):
            data = post(f"http://127.0.0.1:{port}/v1/embeddings", {"model": f"e1-{label}", "input": texts[i:i + 16]})["data"]
            rows += [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        return np.asarray(rows, np.float32)
    finally:
        proc.terminate()
        proc.wait(60)
        log.close()


def bench(model: Path) -> dict:
    out = subprocess.run([str(RUNTIME / "llama-bench"), "-m", str(model), "-ngl", "99", "-p", "512", "-n", "128", "-r", "3", "-o", "json"],
                         capture_output=True, text=True, check=True).stdout
    return {f"{'pp' if r['n_prompt'] else 'tg'}": {"avg_ts": r["avg_ts"], "stddev_ts": r["stddev_ts"]} for r in json.loads(out)}


def main() -> int:
    label, model = sys.argv[1], Path(sys.argv[2])
    if sys.argv[3:] == ["bench"]:
        receipt_path = OUT / f"receipt-{label}.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["bench"] = bench(model)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"label": label, "bench": receipt["bench"]}))
        return 0
    arm = runpy.run_path(str(RANK_ARM))
    texts = [json.loads(line)["text"] for line in (ANCHORS / "corpus.jsonl").read_text().splitlines()]
    parents = arm["read_f32"](ANCHORS / "parents.f32", arm["PARENT"])
    hidden = embed(model, label, texts)
    (OUT / f"hidden-{label}.f32").write_bytes(hidden.astype("<f4").tobytes())
    projection = np.random.default_rng(PROJECTION_SEED).standard_normal((hidden.shape[1], arm["VIEW"])).astype(np.float32)
    candidates = (hidden @ projection).astype(np.float32)
    (OUT / f"candidates-{label}.f32").write_bytes(candidates.astype("<f4").tobytes())
    result = arm["judge"]([list(struct.unpack("<128f", r.astype("<f4").tobytes())) for r in candidates], parents)
    receipt = {"schema": "carl.e1-group-arm/v1", "node": "E1", "label": label, "model": str(model), "model_bytes": model.stat().st_size,
               "model_sha256": sha(model), "runtime": str(RUNTIME), "hidden_width": int(hidden.shape[1]), "pooling": "mean",
               "projection": {"kind": "seeded-gaussian", "seed": PROJECTION_SEED, "shape": [int(hidden.shape[1]), arm["VIEW"]]},
               "rank_arm_sha256": sha(RANK_ARM), "parents_sha256": sha(ANCHORS / "parents.f32"), "judge": result}
    (OUT / f"receipt-{label}.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in ("label", "model_bytes")} | {"rows": result["rows"], "arms": result["arms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
