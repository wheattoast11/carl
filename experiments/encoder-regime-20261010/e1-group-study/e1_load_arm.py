"""E1 g64 load arm: serve the g64 file on mainline bf942164 and on prism b1e2013, greedy-complete the same prompts, compare."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
MODEL = OUT / "bonsai2-27b-g64.gguf"
RUNTIMES = {"mainline-bf942164": OUT / "mainline-bf942164/src/build/bin/llama-server",
            "prism-b1e2013": Path("/var/home/zero/llm-workspace/runtimes/llama-bonsai2-vk-ss-b1e2013/bin/llama-server")}
PROMPTS = ["The capital of France is", "def fibonacci(n):", "Water boils at a temperature of"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(name: str, binary: Path) -> dict:
    port = free_port()
    log_path = OUT / f"load-{name}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen([str(binary), "-m", str(MODEL), "--alias", f"e1-load-{name}", "--host", "127.0.0.1", "--port", str(port),
                                 "-ngl", "99", "-c", "2048", "--parallel", "1", "--no-webui"], stdout=log, stderr=subprocess.STDOUT)
        try:
            loaded = False
            for _ in range(600):
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if json.loads(r.read()).get("status") == "ok":
                            loaded = True
                            break
                except OSError:
                    pass
                time.sleep(1)
            outputs = []
            if loaded:
                for p in PROMPTS:
                    body = json.dumps({"prompt": p, "n_predict": 24, "temperature": 0, "top_k": 1, "cache_prompt": False}).encode()
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(req, timeout=300) as r:
                            outputs.append(json.loads(r.read())["content"])
                    except urllib.error.HTTPError as exc:
                        outputs.append({"http_error": exc.code, "body": exc.read().decode(errors="replace")[:300]})
            log.flush()
            unparsed = [line.split("unparsed Content-only output: ", 1)[1].strip() for line in log_path.read_text(errors="replace").splitlines()
                        if "unparsed Content-only output: " in line]
            return {"loaded": loaded, "exit_code": proc.poll(), "outputs": outputs, "unparsed_text_from_log": unparsed, "log": str(log_path)}
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(60)


def main() -> int:
    result = {name: run(name, binary) for name, binary in RUNTIMES.items()}
    result["prompts"] = PROMPTS
    result["outputs_equal"] = result["mainline-bf942164"]["outputs"] == result["prism-b1e2013"]["outputs"]
    (OUT / "load-arm-g64.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
