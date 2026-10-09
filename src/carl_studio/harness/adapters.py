"""Native host protocols using CARL-owned subprocess handles."""

from __future__ import annotations

import json
import os
import queue
import secrets
import selectors
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from carl_core.errors import CARLError

from carl_studio.handles.subprocess import SubprocessToolkit
from carl_studio.harness.types import DelegationRequest, Host

FRAME_LIMIT = 1024 * 1024
OUTPUT_LIMIT = 8 * FRAME_LIMIT
Approval = Callable[[str, dict[str, Any]], bool]
Checkpoint = Callable[[], None]


def list_harnesses() -> list[dict[str, Any]]:
    """Discover native executables without launching agents or reading credentials."""
    return [
        {
            "host": host.value,
            "executable": shutil.which(host.value),
            "available": shutil.which(host.value) is not None,
        }
        for host in Host
    ]


class JSONProcess:
    """Drain bounded protocol frames while permission replies are pending."""

    def __init__(
        self,
        toolkit: SubprocessToolkit,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        checkpoint: Checkpoint,
        timeout_s: float,
    ) -> None:
        self.toolkit, self.checkpoint = toolkit, checkpoint
        self.handle = toolkit.spawn(
            argv, cwd=cwd, env=env, interactive=True, process_group=True, ttl_s=int(timeout_s) + 60
        )
        self.process = toolkit.protocol_process(self.handle["ref_id"])
        self.selector = selectors.DefaultSelector()
        for stream in (self.process.stdout, self.process.stderr):
            assert stream is not None
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ)
        self.frames: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=128)
        self.invalidated: set[str] = set()
        self.responses: dict[str, dict[str, Any]] = {}
        self.error: CARLError | None = None
        self.stop = threading.Event()
        self.write_lock = threading.Lock()
        self.reader = threading.Thread(target=self._pump, daemon=True)
        self.reader.start()

    def _pump(self) -> None:
        buffer = bytearray()
        total = 0
        try:
            while not self.stop.is_set():
                for key, _ in self.selector.select(0.05):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        self.selector.unregister(key.fileobj)
                        continue
                    total += len(chunk)
                    if total > OUTPUT_LIMIT:
                        raise CARLError(
                            "Host output limit exceeded", code="carl.agent.output_limit"
                        )
                    if key.fileobj is not self.process.stdout:
                        continue
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        line, _, rest = buffer.partition(b"\n")
                        buffer = bytearray(rest)
                        if len(line) > FRAME_LIMIT:
                            raise CARLError(
                                "Host frame limit exceeded", code="carl.agent.frame_limit"
                            )
                        raw: Any = json.loads(line)
                        if not isinstance(raw, dict):
                            raise CARLError("Malformed host frame", code="carl.agent.protocol")
                        value = cast(dict[str, Any], raw)
                        if "id" in value and "method" not in value:
                            if len(self.responses) >= 128:
                                raise CARLError(
                                    "Native reply limit", code="carl.agent.output_limit"
                                )
                            self.responses[str(value["id"])] = value
                        if value.get("type") == "control_cancel_request":
                            self.invalidated.add(str(value.get("request_id")))
                        elif value.get("method") == "serverRequest/resolved":
                            self.invalidated.add(str(value.get("params", {}).get("requestId")))
                        try:
                            self.frames.put_nowait(value)
                        except queue.Full as exc:
                            raise CARLError(
                                "Host event queue limit", code="carl.agent.output_limit"
                            ) from exc
                    if len(buffer) > FRAME_LIMIT:
                        raise CARLError("Host frame limit exceeded", code="carl.agent.frame_limit")
                if not self.selector.get_map():
                    self.error = CARLError("Host stream ended", code="carl.agent.eof")
                    return
        except CARLError as exc:
            self.error = exc
        except (OSError, ValueError, UnicodeDecodeError):
            self.error = CARLError("Malformed host stream", code="carl.agent.protocol")

    def send(self, value: dict[str, Any], *, check: bool = True) -> None:
        if check:
            self.checkpoint()
        data = (json.dumps(value, ensure_ascii=False) + "\n").encode()
        if not self.write_lock.acquire(timeout=1):
            raise CARLError("Native writer busy", code="carl.agent.stdin_timeout")
        try:
            self.toolkit.write_stdin(
                self.handle["ref_id"],
                data,
                timeout_s=1,
                checkpoint=self.checkpoint if check else None,
            )
        finally:
            self.write_lock.release()

    def receive(self) -> dict[str, Any]:
        while True:
            self.checkpoint()
            try:
                return self.frames.get(timeout=0.05)
            except queue.Empty:
                if self.error is not None:
                    raise self.error

    def request_live(self, native_id: str) -> bool:
        return native_id not in self.invalidated and self.error is None

    def response(self, request_id: int) -> dict[str, Any]:
        while True:
            self.checkpoint()
            value = self.responses.pop(str(request_id), None)
            if value is not None:
                if "error" in value:
                    raise CARLError("Native request rejected", code="carl.agent.native_rejected")
                return value["result"]
            if self.error is not None:
                raise self.error
            time.sleep(0.02)

    def close(self) -> None:
        self.stop.set()
        self.toolkit.terminate(self.handle["ref_id"], grace_s=1, process_group=True)
        self.reader.join(timeout=2)
        self.selector.close()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()


def host_environment(host: Host, root: Path, profile: Path) -> dict[str, str]:
    """Pass only the selected host's environment and owned child context."""
    common = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR", "CARL_ENCODER_MODEL", "CARL_ENCODER_PYTHON"}
    prefixes = {
        Host.CODEX: ("OPENAI_",),
        Host.CLAUDE: ("ANTHROPIC_",),
        Host.OPENCODE: ("OPENAI_", "ANTHROPIC_"),
    }[host]
    env = {
        key: value for key, value in os.environ.items() if key in common or key.startswith(prefixes)
    }
    env.update(
        {
            "CARL_DELEGATION_DEPTH": "1",
            "CARL_WORKSPACE_ROOT": str(root),
            "CARL_CHILD_SCOPE": "1",
            "CARL_LOG_LEVEL": "error",
        }
    )
    from carl_studio.settings import carl_home

    env["CARL_HOME"] = str(carl_home())
    if host == Host.CODEX:
        env["CODEX_HOME"] = str(profile)
        supplied: dict[str, Any] = json.loads(os.environ.get("CARL_CODEX_CONFIG", "{}"))
        config = {
            key: supplied[key]
            for key in ("model", "model_provider", "model_providers")
            if key in supplied
        }
        config["mcp_servers"] = {
            "carl": {
                "command": sys.executable,
                "args": ["-m", "carl_studio.mcp"],
                "env": {
                    "CARL_CHILD_SCOPE": "1",
                    "CARL_DELEGATION_DEPTH": "1",
                    "CARL_WORKSPACE_ROOT": str(root),
                    **{key: os.environ[key] for key in ("CARL_ENCODER_MODEL", "CARL_ENCODER_PYTHON") if key in os.environ},
                },
            }
        }
        config["analytics"] = {"enabled": False}
        (profile / "config.toml").write_text(
            "\n".join(json.dumps(key) + " = " + _toml_value(value) for key, value in config.items())
        )
    elif host == Host.CLAUDE:
        env["CLAUDE_CONFIG_DIR"] = str(profile)
    else:
        for name in ("CONFIG", "DATA", "CACHE", "STATE"):
            env[f"XDG_{name}_HOME"] = str(profile / name.lower())
        env.update({"OPENCODE_DISABLE_PROJECT_CONFIG": "1", "OPENCODE_DISABLE_MODELS_FETCH": "1"})
    return env


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ",".join(_toml_value(item) for item in cast(list[Any], value)) + "]"
    if isinstance(value, dict):
        items: dict[str, Any] = cast(dict[str, Any], value)
        return (
            "{"
            + ",".join(json.dumps(key) + "=" + _toml_value(item) for key, item in items.items())
            + "}"
        )
    if isinstance(value, (int, float)):
        return str(value)
    raise CARLError("Unsupported native configuration value", code="carl.agent.config")


class CodexAdapter:
    """Codex app-server's native turn, approval, and interruption protocol."""

    def run(self, link: JSONProcess, request: DelegationRequest, approve: Approval) -> str:
        from carl_studio import __version__

        self.link = link
        self.thread_id: str | None = None
        self.turn_id: str | None = None
        self.mcp_status: str | None = None
        self.continuation_id = 100
        link.send(
            {
                "id": 0,
                "method": "initialize",
                "params": {"clientInfo": {"name": "carl", "version": __version__}},
            }
        )
        self._response(0)
        link.send({"method": "initialized", "params": {}})
        params: dict[str, Any] = {
            "cwd": str(request.workdir),
            "ephemeral": True,
            "approvalPolicy": "on-request",
            "sandbox": "workspace-write" if request.write else "read-only",
        }
        if request.model:
            params["model"] = request.model
        link.send({"id": 1, "method": "thread/start", "params": params})
        self.thread_id = self._response(1)["thread"]["id"]
        while self._startup_status() != "ready":
            message = link.receive()
            self._observe_startup(message)
            if self._startup_status() in {"failed", "cancelled"}:
                raise CARLError("CARL MCP startup failed", code="carl.agent.mcp_startup")
        policy = (
            {
                "type": "workspaceWrite",
                "writableRoots": [str(request.workdir)],
                "networkAccess": False,
                "excludeSlashTmp": True,
                "excludeTmpdirEnvVar": True,
            }
            if request.write
            else {"type": "readOnly", "networkAccess": False}
        )
        link.send(
            {
                "id": 2,
                "method": "turn/start",
                "params": {
                    "threadId": self.thread_id,
                    "sandboxPolicy": policy,
                    "input": [{"type": "text", "text": request.instruction}],
                },
            }
        )
        self.turn_id = self._response(2)["turn"]["id"]
        text: list[str] = []
        while True:
            event = link.receive()
            params = event.get("params", {})
            if params.get("threadId") != self.thread_id:
                continue
            if params.get("turnId") not in (None, self.turn_id):
                continue
            method = event.get("method", "")
            if method in {
                "item/commandExecution/requestApproval",
                "item/fileChange/requestApproval",
            }:
                allowed = approve(str(event["id"]), {"kind": method, **params})
                if not link.request_live(str(event["id"])):
                    continue
                link.send(
                    {"id": event["id"], "result": {"decision": "accept" if allowed else "decline"}}
                )
            elif method == "mcpServer/elicitation/request":
                schema = params.get("requestedSchema", {})
                permitted = (
                    params.get("serverName") == "carl"
                    and params.get("mode") == "form"
                    and not schema.get("properties")
                    and not schema.get("required")
                )
                allowed = permitted and approve(str(event["id"]), {"kind": method, **params})
                if link.request_live(str(event["id"])):
                    link.send(
                        {
                            "id": event["id"],
                            "result": {
                                "action": "accept" if allowed else "decline",
                                "content": {} if allowed else None,
                            },
                        }
                    )
            elif "id" in event and method:
                link.send(
                    {
                        "id": event["id"],
                        "error": {"code": -32601, "message": "Unsupported CARL host request"},
                    }
                )
            elif method == "item/agentMessage/delta":
                text.append(str(params.get("delta", "")))
            elif (
                method == "item/completed" and params.get("item", {}).get("type") == "agentMessage"
            ):
                if not text:
                    text.append(str(params["item"].get("text", "")))
            elif method == "turn/completed" and params.get("turn", {}).get("id") == self.turn_id:
                if params["turn"]["status"] != "completed":
                    raise CARLError("Native turn did not complete", code="carl.agent.native_failed")
                return "".join(text)

    def _response(self, request_id: int) -> dict[str, Any]:
        while True:
            message = self.link.receive()
            self._observe_startup(message)
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise CARLError("Host rejected request", code="carl.agent.protocol")
                return message["result"]

    def _startup_status(self) -> str | None:
        return self.mcp_status

    def _observe_startup(self, message: dict[str, Any]) -> None:
        if message.get("method") == "mcpServer/startupStatus/updated":
            params = message.get("params", {})
            if params.get("name") == "carl" or params.get("serverName") == "carl":
                status = params.get("status")
                self.mcp_status = (
                    cast(dict[str, Any], status).get("type") if isinstance(status, dict) else status
                )

    def interrupt(self) -> None:
        if self.thread_id and self.turn_id:
            self.link.send(
                {
                    "id": 99,
                    "method": "turn/interrupt",
                    "params": {"threadId": self.thread_id, "turnId": self.turn_id},
                },
                check=False,
            )

    def send_input(self, instruction: str) -> None:
        request_id = self.continuation_id
        self.continuation_id += 1
        self.link.send(
            {
                "id": request_id,
                "method": "turn/steer",
                "params": {
                    "threadId": self.thread_id,
                    "expectedTurnId": self.turn_id,
                    "input": [{"type": "text", "text": instruction}],
                },
            }
        )
        result = self.link.response(request_id)
        if result.get("turnId") != self.turn_id:
            raise CARLError("Native turn changed", code="carl.agent.session")


class ClaudeAdapter:
    """Claude Code's duplex user and control frames."""

    def run(self, link: JSONProcess, request: DelegationRequest, approve: Approval) -> str:
        self.link = link
        self.session_id: str | None = None
        link.send(
            {
                "type": "control_request",
                "request_id": "carl-init",
                "request": {"subtype": "initialize", "hooks": None},
            }
        )
        while True:
            event = link.receive()
            if event.get("type") == "control_response":
                response = event.get("response", {})
                if response.get("request_id") == "carl-init":
                    if response.get("subtype") != "success":
                        raise CARLError("Host initialization failed", code="carl.agent.protocol")
                    break
        link.send(
            {
                "type": "user",
                "message": {"role": "user", "content": request.instruction},
                "parent_tool_use_id": None,
                "session_id": "default",
            }
        )
        text: list[str] = []
        while True:
            event = link.receive()
            kind = event.get("type")
            if kind == "system" and event.get("subtype") == "init":
                self.session_id = event.get("session_id")
            elif kind == "control_request":
                native = event["request"]
                allowed = native.get("subtype") == "can_use_tool" and approve(
                    event["request_id"], {"kind": "can_use_tool", **native}
                )
                response = (
                    {"behavior": "allow", "updatedInput": native.get("input", {})}
                    if allowed
                    else {"behavior": "deny", "message": "Caller declined"}
                )
                if not link.request_live(str(event["request_id"])):
                    continue
                link.send(
                    {
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": event["request_id"],
                            "response": response,
                        },
                    }
                )
            elif kind == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        text.append(str(block.get("text", "")))
            elif kind == "result":
                if event.get("is_error"):
                    raise CARLError("Native turn failed", code="carl.agent.native_failed")
                return str(event.get("result", "")) or "".join(text)

    def interrupt(self) -> None:
        self.link.send(
            {
                "type": "control_request",
                "request_id": "carl-interrupt",
                "request": {"subtype": "interrupt"},
            },
            check=False,
        )

    def send_input(self, instruction: str) -> None:
        self.link.send(
            {
                "type": "user",
                "message": {"role": "user", "content": instruction},
                "parent_tool_use_id": None,
                "session_id": self.session_id or "default",
            }
        )


class OpenCodeAdapter:
    """Owned, authenticated loopback HTTP/SSE sessions."""

    def run(
        self,
        toolkit: SubprocessToolkit,
        executable: str,
        request: DelegationRequest,
        env: dict[str, str],
        approve: Approval,
        checkpoint: Checkpoint,
    ) -> str:
        import base64
        import re

        import httpx

        env["OPENCODE_SERVER_PASSWORD"] = secrets.token_urlsafe(32)
        supplied = json.loads(os.environ.get("CARL_OPENCODE_CONFIG", "{}"))
        config = {key: supplied[key] for key in ("provider", "model") if key in supplied}
        config.update(
            {
                "autoupdate": False,
                "share": "disabled",
                "mcp": {
                    "carl": {
                        "type": "local",
                        "enabled": True,
                        "command": [sys.executable, "-m", "carl_studio.mcp"],
                        "environment": {
                            "CARL_CHILD_SCOPE": "1",
                            "CARL_DELEGATION_DEPTH": "1",
                            "CARL_WORKSPACE_ROOT": str(request.workdir),
                            **{key: os.environ[key] for key in ("CARL_ENCODER_MODEL", "CARL_ENCODER_PYTHON") if key in os.environ},
                        },
                    }
                },
                "permission": {
                    "*": "ask",
                    "bash": "deny",
                    "webfetch": "deny",
                    "websearch": "deny",
                    "external_directory": "deny",
                },
            }
        )
        if request.model:
            config["model"] = request.model
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
        handle = toolkit.spawn(
            launch_arguments(Host.OPENCODE, executable, request),
            cwd=request.workdir,
            env=env,
            process_group=True,
            ttl_s=int(request.timeout_s) + 60,
        )
        process = toolkit.protocol_process(handle["ref_id"])
        self.toolkit, self.handle = toolkit, handle
        self.client: Any = None
        self.session_id: str | None = None
        selector = selectors.DefaultSelector()
        assert process.stdout is not None and process.stderr is not None
        for pipe in (process.stdout, process.stderr):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ)
        startup = bytearray()
        self.events: queue.Queue[dict[str, Any] | Exception] = queue.Queue(maxsize=128)
        opened = threading.Event()
        stopped = threading.Event()
        reader: threading.Thread | None = None
        try:
            url: str | None = None
            while url is None:
                checkpoint()
                for key, _ in selector.select(0.05):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    startup.extend(chunk)
                    if len(startup) > FRAME_LIMIT:
                        raise CARLError("Host startup output limit", code="carl.agent.output_limit")
                    match = re.search(rb"http://127\.0\.0\.1:(\d+)", startup)
                    if match:
                        url = "http://127.0.0.1:" + match.group(1).decode()
                if not selector.get_map():
                    raise CARLError("Host exited during startup", code="carl.agent.eof")
            auth = base64.b64encode(f"opencode:{env['OPENCODE_SERVER_PASSWORD']}".encode()).decode()
            self.client = httpx.Client(
                base_url=url, headers={"Authorization": f"Basic {auth}"}, trust_env=False, timeout=5
            )
            response = self.client.post("/session", json={"title": "CARL delegation"})
            response.raise_for_status()
            self.session_id = response.json()["id"]

            def receive_events() -> None:
                try:
                    with self.client.stream(
                        "GET",
                        "/event",
                        params={"directory": str(request.workdir)},
                        timeout=request.timeout_s + 5,
                    ) as stream:
                        stream.raise_for_status()
                        opened.set()
                        for line in stream.iter_lines():
                            if stopped.is_set():
                                return
                            if not line.startswith("data:"):
                                continue
                            if len(line.encode()) > FRAME_LIMIT:
                                raise ValueError("SSE frame limit")
                            event = json.loads(line[5:])
                            while not stopped.is_set():
                                try:
                                    self.events.put(event, timeout=0.1)
                                    break
                                except queue.Full:
                                    continue
                except (OSError, ValueError, httpx.HTTPError):
                    opened.set()
                    try:
                        self.events.put_nowait(CARLError("SSE stream ended", code="carl.agent.eof"))
                    except queue.Full:
                        pass

            reader = threading.Thread(target=receive_events, daemon=True)
            reader.start()
            while not opened.wait(0.05):
                checkpoint()
            response = self.client.post(
                f"/session/{self.session_id}/prompt_async",
                json={"parts": [{"type": "text", "text": request.instruction}]},
            )
            response.raise_for_status()
            while True:
                checkpoint()
                try:
                    event = self.events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if isinstance(event, Exception):
                    raise event
                properties = event.get("properties", {})
                if properties.get("sessionID") != self.session_id:
                    continue
                kind = event.get("type")
                if kind == "permission.asked":
                    native_id = properties["id"]
                    allowed = approve(native_id, {"kind": "permission.asked", **properties})
                    self.client.post(
                        f"/permission/{native_id}/reply",
                        json={"reply": "once" if allowed else "reject"},
                    ).raise_for_status()
                elif kind == "session.error":
                    raise CARLError("Native session failed", code="carl.agent.native_failed")
                elif kind == "session.idle" or (
                    kind == "session.status" and properties.get("status", {}).get("type") == "idle"
                ):
                    messages = self.client.get(f"/session/{self.session_id}/message")
                    messages.raise_for_status()
                    text: list[str] = []
                    history: list[dict[str, Any]] = messages.json()
                    for message in history:
                        info = message.get("info", {})
                        if info.get("role") != "assistant":
                            continue
                        if info.get("error"):
                            raise CARLError(
                                "Native message failed", code="carl.agent.native_failed"
                            )
                        text.extend(
                            part.get("text", "")
                            for part in message.get("parts", [])
                            if part.get("type") == "text"
                        )
                    return "".join(text)
        finally:
            stopped.set()
            selector.close()
            toolkit.terminate(handle["ref_id"], grace_s=1, process_group=True)
            if self.client is not None:
                self.client.close()
            if reader is not None:
                reader.join(timeout=2)

    def interrupt(self) -> None:
        if self.client is not None and self.session_id:
            import httpx

            try:
                self.client.post(f"/session/{self.session_id}/abort", json={}).raise_for_status()
            except httpx.HTTPError as exc:
                raise CARLError("Native abort raced teardown", code="carl.agent.interrupt") from exc

    def send_input(self, instruction: str) -> None:
        self.client.post(
            f"/session/{self.session_id}/prompt_async",
            json={"parts": [{"type": "text", "text": instruction}]},
        ).raise_for_status()


def launch_arguments(host: Host, executable: str, request: DelegationRequest) -> list[str]:
    """Select fixed native launch modes without disabling permission checks."""
    if host == Host.CODEX:
        return [executable, "app-server", "--stdio"]
    if host == Host.CLAUDE:
        tools = "Read,Glob,Grep,Edit,Write" if request.write else "Read,Glob,Grep"
        child_mcp = {
            "mcpServers": {
                "carl": {
                    "command": sys.executable,
                    "args": ["-m", "carl_studio.mcp"],
                    "env": {
                        "CARL_CHILD_SCOPE": "1",
                        "CARL_DELEGATION_DEPTH": "1",
                        "CARL_WORKSPACE_ROOT": str(request.workdir),
                        **{key: os.environ[key] for key in ("CARL_ENCODER_MODEL", "CARL_ENCODER_PYTHON") if key in os.environ},
                    },
                }
            }
        }
        argv = [
            executable,
            "--print",
            "--bare",
            "--restricted",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            json.dumps(child_mcp),
            "--permission-mode",
            "manual",
            "--permission-prompt-tool",
            "stdio",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--tools",
            tools,
        ]
        if request.model:
            argv.extend(["--model", request.model])
        from carl_studio.plugin import source_root

        argv.extend(["--plugin-dir", str(source_root())])
        return argv
    return [executable, "serve", "--pure", "--hostname", "127.0.0.1", "--port", "0"]
