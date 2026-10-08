"""Witness actual installed hosts against synthetic localhost providers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("CARL_PROBE_DIR", str(ROOT / "build/native-probes")))
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(SCRATCH)
os.environ["CARL_HOME"] = str(SCRATCH / "carl-state")
for k in list(os.environ):
    if k.startswith(("ANTHROPIC_", "OPENAI_")):
        del os.environ[k]
os.environ.update(
    OPENAI_API_KEY="carl-offline-synthetic",
    ANTHROPIC_API_KEY="carl-offline-synthetic",
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
)
import carl_studio.harness.runtime as runtime_mod
from carl_studio.harness.runtime import HarnessRuntime
from carl_studio.harness.types import DelegationContext, DelegationRequest, Host
from carl_studio.mcp.tasks import MCPTaskStore
from carl_studio.session import Session

EVENTS = []
orig_receive = runtime_mod.JSONProcess.receive


def receive(self):
    e = orig_receive(self)
    method = e.get("method", e.get("type"))
    params = e.get("params", {})
    if method and ("mcp" in method.lower() or "error" in method.lower()):
        EVENTS.append(
            {
                "method": method,
                "keys": list(params),
                "status": params.get("status"),
                "error": params.get("error"),
                "requestedSchema": params.get("requestedSchema"),
                "mode": params.get("mode"),
                "serverName": params.get("serverName"),
            }
        )
    return e


runtime_mod.JSONProcess.receive = receive
SOURCE_HASHES = {
    n: hashlib.sha256((ROOT / "src/carl_studio/harness" / n).read_bytes()).hexdigest()
    for n in ("adapters.py", "runtime.py", "types.py")
}
SOURCE_HASHES.update(
    {
        path: hashlib.sha256((ROOT / "src/carl_studio" / path).read_bytes()).hexdigest()
        for path in (
            "mcp/server.py",
            "mcp/output_schemas.py",
            "mcp/protocol.py",
            "training/preparation.py",
            "types/preparation.py",
        )
    }
)
META = []
STARTED = threading.Event()
MODE = sys.argv[2] if len(sys.argv) > 2 else "complete"
HOST = sys.argv[1]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        out = json.dumps(
            {
                "object": "list",
                "data": [{"id": "carl-test", "object": "model", "created": 0, "owned_by": "carl"}],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        tools = body.get("tools", [])
        names = [x.get("name") or x.get("function", {}).get("name") for x in tools]
        META.append(
            {
                "path": self.path,
                "stream": body.get("stream"),
                "tool_count": len(tools),
                "carl_tools": [n for n in names if n and "carl" in n],
                "tool_names": names,
                "tool_types": [t.get("type") for t in tools],
                "nested_tool_names": [
                    {
                        "name": t.get("name"),
                        "type": t.get("type"),
                        "tools": [x.get("name") for x in t.get("tools", [])],
                    }
                    for t in tools
                    if "tools" in t
                ],
            }
        )
        STARTED.set()
        serialized = json.dumps(body)
        META[-1]["prepared_result_present"] = "Eprep_" in serialized
        desired_tool = "prepare_training" if MODE == "prepare" else "get_coherence_metrics"
        tool_arguments = (
            {
                "config_yaml": json.dumps(
                    {
                        "run_name": "native-preparation",
                        "base_model": "fixture/source",
                        "dataset_repo": "fixture/train",
                        "eval_dataset_repo": "fixture/eval",
                        "output_repo": "fixture/candidate",
                        "method": "sft",
                        "compute_target": "local",
                        "max_steps": 4,
                        "push_to_hub": False,
                    }
                )
            }
            if MODE == "prepare"
            else {"logits_summary": '{"embedding_dim":3072}'}
        )
        META[-1]["metrics_result_present"] = all(
            k in serialized for k in ("kappa", "sigma", "t_star", "embedding_dim")
        )
        chosen = next((n for n in names if n and desired_tool in n), None)
        chosen_scope = None
        if chosen is None:
            for t in tools:
                for x in t.get("tools", []):
                    if x.get("name") == desired_tool:
                        chosen = x["name"]
                        chosen_scope = t["name"]
        call_tool = MODE in {"metrics", "prepare"} and len(META) == 1 and chosen is not None
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        def event(kind, obj):
            if kind:
                self.wfile.write(("event: " + kind + "\n").encode())
            self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
            self.wfile.flush()

        try:
            if MODE == "cancel":
                for _ in range(200):
                    self.wfile.write(b": waiting\n\n")
                    self.wfile.flush()
                    time.sleep(0.1)
                return
            if "/messages" in self.path:
                message = {
                    "id": "msg_carl",
                    "type": "message",
                    "role": "assistant",
                    "model": "carl-test",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
                if call_tool:
                    event("message_start", {"type": "message_start", "message": message})
                    event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "tool_use",
                                "id": "tool_carl",
                                "name": chosen,
                                "input": {},
                            },
                        },
                    )
                    event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(tool_arguments),
                            },
                        },
                    )
                    event("content_block_stop", {"type": "content_block_stop", "index": 0})
                    event(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                            "usage": {"output_tokens": 1},
                        },
                    )
                    event("message_stop", {"type": "message_stop"})
                    return
                event("message_start", {"type": "message_start", "message": message})
                event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "CARL_NATIVE_OK"},
                    },
                )
                event("content_block_stop", {"type": "content_block_stop", "index": 0})
                event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 1},
                    },
                )
                event("message_stop", {"type": "message_stop"})
            elif "/chat/completions" in self.path:
                common = {
                    "id": "chat_carl",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "carl-test",
                }
                event(
                    None,
                    {
                        **common,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "CARL_NATIVE_OK"},
                                "finish_reason": None,
                            }
                        ],
                    },
                )
                event(
                    None,
                    {
                        **common,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                resp = {
                    "id": "resp_carl",
                    "object": "response",
                    "created_at": 0,
                    "status": "in_progress",
                    "model": "carl-test",
                    "output": [],
                }
                item = {
                    "id": "msg_carl",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }
                ev = lambda k, **kw: event(k, {"type": k, **kw})
                if call_tool:
                    item = {
                        "type": "function_call",
                        "id": "fc_carl",
                        "call_id": "tool_carl",
                        "name": chosen,
                        "arguments": json.dumps(tool_arguments),
                        "status": "completed",
                    }
                    if chosen_scope:
                        item["namespace"] = chosen_scope
                    ev("response.created", response=resp)
                    ev(
                        "response.output_item.added",
                        output_index=0,
                        item={**item, "arguments": "", "status": "in_progress"},
                    )
                    ev(
                        "response.function_call_arguments.delta",
                        item_id="fc_carl",
                        output_index=0,
                        delta=item["arguments"],
                    )
                    ev(
                        "response.function_call_arguments.done",
                        item_id="fc_carl",
                        output_index=0,
                        arguments=item["arguments"],
                    )
                    ev("response.output_item.done", output_index=0, item=item)
                    resp.update(
                        status="completed",
                        output=[item],
                        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    )
                    ev("response.completed", response=resp)
                    return
                ev("response.created", response=resp)
                ev("response.output_item.added", output_index=0, item=item)
                ev(
                    "response.content_part.added",
                    item_id="msg_carl",
                    output_index=0,
                    content_index=0,
                    part={"type": "output_text", "text": "", "annotations": []},
                )
                ev(
                    "response.output_text.delta",
                    item_id="msg_carl",
                    output_index=0,
                    content_index=0,
                    delta="CARL_NATIVE_OK",
                )
                ev(
                    "response.output_text.done",
                    item_id="msg_carl",
                    output_index=0,
                    content_index=0,
                    text="CARL_NATIVE_OK",
                )
                item.update(
                    status="completed",
                    content=[{"type": "output_text", "text": "CARL_NATIVE_OK", "annotations": []}],
                )
                ev("response.output_item.done", output_index=0, item=item)
                resp.update(
                    status="completed",
                    output=[item],
                    usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                ev("response.completed", response=resp)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}"
os.environ["ANTHROPIC_BASE_URL"] = base
os.environ["CARL_CODEX_CONFIG"] = json.dumps(
    {
        "model": "carl-test",
        "model_provider": "carl-offline",
        "model_providers": {
            "carl-offline": {
                "name": "CARL offline",
                "base_url": base + "/v1",
                "wire_api": "responses",
                "requires_openai_auth": False,
                "env_key": "OPENAI_API_KEY",
            }
        },
    }
)
os.environ["CARL_OPENCODE_CONFIG"] = json.dumps(
    {
        "model": "carl-offline/carl-test",
        "provider": {
            "carl-offline": {
                "npm": "@ai-sdk/openai",
                "name": "CARL offline",
                "options": {"baseURL": base + "/v1", "apiKey": "carl-offline-synthetic"},
                "models": {
                    "carl-test": {"name": "CARL test", "limit": {"context": 8192, "output": 256}}
                },
            }
        },
    }
)


async def main():
    root = SCRATCH / f"{HOST}-{MODE}-{uuid.uuid4().hex[:6]}"
    root.mkdir()
    session = Session()
    store = MCPTaskStore(root / "tasks.db")
    rt = HarnessRuntime(session, DelegationContext.local(owner="offline-probe", root=root), store)
    started = time.monotonic()
    states = []
    answers = 0
    cancel_errors = []
    try:
        task = await rt.submit(
            DelegationRequest(
                host=Host(HOST), instruction="Return CARL_NATIVE_OK.", workdir=root, timeout_s=40
            )
        )
        tid = task["task_id"]
        cancelled = False
        while True:
            value = rt.get(tid)
            state = (value["status"], value.get("progress"), bool(value.get("pending_input")))
            if not states or states[-1] != state:
                states.append(state)
            if value.get("pending_input"):
                rt.reply(tid, value["pending_input"]["request_id"], MODE in {"metrics", "prepare"})
                answers += 1
            if MODE == "cancel" and STARTED.is_set() and not cancelled:
                await asyncio.sleep(0.2)
                try:
                    await rt.cancel(tid)
                except Exception as exc:  # noqa: BLE001
                    cancel_errors.append(type(exc).__name__)
                cancelled = True
            if value["status"] in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.1)
        final = rt.get(tid)
        receipt = {
            "host": HOST,
            "mode": MODE,
            "status": final["status"],
            "error": final.get("error"),
            "states": states,
            "permission_replies": answers,
            "cancel_errors": cancel_errors,
            "provider_requests": META,
            "result_match": rt.read_result(tid) == "CARL_NATIVE_OK"
            if final["status"] == "completed"
            else None,
            "elapsed_s": round(time.monotonic() - started, 2),
            "native_session_bound": bool(final.get("metadata", {}).get("native_session")),
            "protocol_events": EVENTS,
            "source_hashes": SOURCE_HASHES,
            "test_overrides": {"startup_delay": False, "child_carl_home_injection": False},
        }
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps({"receipt": str(root / "receipt.json"), **receipt}))
    finally:
        await rt.close()
        server.shutdown()
        server.server_close()
    if MODE == "metrics":
        assert (
            receipt["status"] == "completed"
            and receipt["result_match"]
            and receipt["permission_replies"] >= 1
            and not META[0]["metrics_result_present"]
            and any(m["metrics_result_present"] for m in META[1:])
        ), "Native CARL tool acceptance failed"
    elif MODE == "prepare":
        assert (
            receipt["status"] == "completed"
            and receipt["result_match"]
            and not META[0]["prepared_result_present"]
            and any(item["prepared_result_present"] for item in META[1:])
        ), "Native preparation tool acceptance failed"
    elif MODE == "cancel":
        assert receipt["status"] == "cancelled" and not cancel_errors, (
            "Native cancellation acceptance failed"
        )


asyncio.run(main())
