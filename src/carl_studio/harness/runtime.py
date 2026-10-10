"""Caller-bound delegation through CARL's existing task and resource owners."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from carl_core.errors import CARLError
from carl_core.hashing import content_hash_bytes
from carl_core.interaction import ActionType

from carl_studio.harness.adapters import (
    ClaudeAdapter,
    CodexAdapter,
    JSONProcess,
    OpenCodeAdapter,
    host_environment,
    launch_arguments,
    list_harnesses,
)
from carl_studio.harness.types import DelegationContext, DelegationRequest, Host
from carl_studio.mcp.tasks import MCPTask, MCPTaskStore
from carl_studio.session import Session


class _Cancelled(Exception):
    pass


@dataclass
class _LiveTask:
    request: DelegationRequest
    context: DelegationContext
    generation: str
    cancel: threading.Event = field(default_factory=threading.Event)
    answer_ready: threading.Event = field(default_factory=threading.Event)
    pending: dict[str, Any] | None = None
    answer: bool = False
    adapter: Any = None
    link: JSONProcess | None = None
    worker: asyncio.Task[None] | None = None
    deadline: float = 0
    lock: Any = field(default_factory=threading.RLock)
    continuation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    native_terminal: bool = False


class HarnessRuntime:
    """One Session's live executors backed by the shared MCP task store."""

    def __init__(self, session: Session, context: DelegationContext, store: MCPTaskStore) -> None:
        self.session, self.context, self.store = session, context, store
        self.live: dict[str, _LiveTask] = {}

    async def submit(
        self,
        request: DelegationRequest,
        *,
        executable: str | None = None,
        approved_context: DelegationContext | None = None,
    ) -> dict[str, Any]:
        """Admit and start a task without retaining its instruction in SQLite."""
        context = approved_context or self.context
        if context.owner != self.context.owner:
            raise CARLError("Caller context changed", code="carl.agent.task_owner")
        directory = context.admit(request.model_copy(update={"native_session": None}))
        if request.native_session:
            for task_id, live in self.live.items():
                task = self._owned(task_id)
                if task.metadata.get("native_session") != request.native_session:
                    continue
                async with live.continuation_lock:
                    task = self._owned(task_id)
                    self._checkpoint(task_id, live)
                    if (
                        live.native_terminal
                        or task.is_terminal
                        or directory != live.request.workdir
                        or request.host != live.request.host
                        or (request.write and not live.request.write)
                    ):
                        raise CARLError("Resume scope changed", code="carl.agent.session")
                    identities = dict(task.metadata.get("continuations", {}))
                    digest = content_hash_bytes(request.instruction.encode())
                    if request.request_id in identities:
                        if identities[request.request_id] != digest:
                            raise CARLError(
                                "Resume request reused", code="carl.agent.request_conflict"
                            )
                        return self.get(task_id)
                    if len(identities) >= 32:
                        raise CARLError("Continuation limit", code="carl.agent.capacity")
                    await asyncio.to_thread(live.adapter.send_input, request.instruction)
                    identities[request.request_id] = digest
                    self.store.update_metadata(task_id, {"continuations": identities})
                return self.get(task_id)
            raise CARLError("Native session is not live and owned", code="carl.agent.session")
        binary = executable or shutil.which(request.host.value)
        if binary is None:
            raise CARLError("Native harness is not installed", code="carl.agent.unavailable")
        request = request.model_copy(update={"workdir": directory})
        metadata = {
            "schema_version": 1,
            "owner": self.context.owner,
            "request_id": request.request_id,
            "host": request.host.value,
            "workdir": str(directory),
            "write": request.write,
            "generation": uuid.uuid4().hex,
            "owner_pid": os.getpid(),
            "deadline": time.time() + request.timeout_s,
        }
        params = request.model_dump(mode="json")
        task = self.store.create_delegation(params, metadata)
        if task.task_id in self.live or task.is_terminal:
            return self.get(task.task_id)
        live = _LiveTask(
            request=request,
            context=context,
            generation=task.metadata["generation"],
            deadline=time.monotonic() + request.timeout_s,
        )
        self.live[task.task_id] = live
        live.worker = asyncio.create_task(self._execute(task, live, binary))
        self.session.chain.record(
            ActionType.EXTERNAL,
            "agent.delegate",
            input={"task_id": task.task_id, "host": request.host.value},
            output={"generation": live.generation},
            success=True,
        )
        return self.get(task.task_id)

    def _owned(self, task_id: str) -> MCPTask:
        task = self.store.get(task_id)
        if task is None or task.metadata.get("owner") != self.context.owner:
            raise CARLError("Task is not owned by this caller", code="carl.agent.task_owner")
        return task

    def get(self, task_id: str) -> dict[str, Any]:
        """Read an owned task and its bounded, transient permission question."""
        task = self._owned(task_id)
        value = task.to_dict()
        live = self.live.get(task_id)
        if live is not None:
            with live.lock:
                if live.pending:
                    value["pending_input"] = {
                        key: val for key, val in live.pending.items() if key != "native_id"
                    }
        return value

    def reply(self, task_id: str, request_id: str, approve: bool) -> dict[str, Any]:
        """Answer exactly one live permission request within the caller's grant."""
        task = self._owned(task_id)
        live = self.live.get(task_id)
        if task.is_terminal or live is None:
            raise CARLError("Task is no longer live", code="carl.agent.stale_reply")
        with live.lock:
            pending = live.pending
            if (
                pending is None
                or pending["request_id"] != request_id
                or pending["generation"] != live.generation
                or live.answer_ready.is_set()
                or live.cancel.is_set()
                or time.monotonic() >= live.deadline
                or (live.link is not None and not live.link.request_live(pending["native_id"]))
            ):
                raise CARLError("Permission reply is stale", code="carl.agent.stale_reply")
            context = live.context
            if (
                approve
                and pending["requires_write"]
                and not (context.allow_write and live.request.write)
            ):
                raise CARLError("Approval exceeds write grant", code="carl.agent.permission")
            live.answer = approve
            live.answer_ready.set()
        return {"task_id": task_id, "accepted": True}

    async def cancel(self, task_id: str) -> dict[str, Any]:
        """Interrupt and reap execution before acknowledging cancellation."""
        task = self._owned(task_id)
        if task.is_terminal:
            return {"task_id": task_id, "cancelled": False, "reason": "already_terminal"}
        live = self.live.get(task_id)
        if live is None:
            raise CARLError("Execution owner unavailable", code="carl.agent.owner_unavailable")
        live.cancel.set()
        self.store.update_metadata(task_id, {"cancel_requested": True})
        try:
            if live.adapter is not None:
                await asyncio.to_thread(live.adapter.interrupt)
        except (CARLError, OSError, ValueError, RuntimeError, AttributeError):
            self.store.update_metadata(task_id, {"native_interrupt_failed": True})
        if live.worker is not None:
            await asyncio.shield(live.worker)
        current = self._owned(task_id)
        return {
            "task_id": task_id,
            "cancelled": current.status == "cancelled",
            "status": current.status,
        }

    def _checkpoint(self, task_id: str, live: _LiveTask) -> None:
        if live.cancel.is_set():
            raise _Cancelled()
        if time.monotonic() >= live.deadline:
            raise CARLError("Delegation deadline expired", code="carl.agent.timeout")
        task = self.store.get(task_id)
        if task is None or task.metadata.get("cancel_requested"):
            raise _Cancelled()
        native = getattr(live.adapter, "thread_id", None) or getattr(
            live.adapter, "session_id", None
        )
        if native and task.metadata.get("native_session") != native:
            self.store.update_metadata(task_id, {"native_session": native})

    def _approve(
        self, task_id: str, live: _LiveTask, native_id: str, details: dict[str, Any]
    ) -> bool:
        from carl_core.safepath import safe_resolve

        context = live.context
        if details.get("kind") == "item/commandExecution/requestApproval":
            return False
        raw_values: Any = details.get("input", {})
        values: dict[str, Any] = (
            cast(dict[str, Any], raw_values) if isinstance(raw_values, dict) else {}
        )
        for key in ("file_path", "path", "cwd", "grantRoot", "blocked_path"):
            value = details.get(key) or values.get(key)
            if isinstance(value, str):
                try:
                    safe_resolve(value, context.root)
                except CARLError:
                    return False
        if details.get("networkApprovalContext") or details.get("permission") in {
            "external_directory",
            "webfetch",
            "websearch",
            "bash",
        }:
            return False
        requires_write = (
            "fileChange" in details.get("kind", "")
            or details.get("tool_name") in {"Write", "Edit", "MultiEdit"}
            or details.get("permission") == "edit"
        )
        question_ref = self.session.data_toolkit.open_bytes(json.dumps(details).encode())
        with live.lock:
            live.answer_ready.clear()
            live.pending = {
                "request_id": uuid.uuid4().hex,
                "native_id": native_id,
                "generation": live.generation,
                "kind": details.get("kind", "permission"),
                "requires_write": requires_write,
                "details_ref": question_ref,
            }
        self.store.update_metadata(
            task_id,
            {
                "pending_input": {
                    key: value
                    for key, value in live.pending.items()
                    if key not in {"native_id", "details_ref"}
                }
            },
        )
        try:
            while not live.answer_ready.wait(0.05):
                self._checkpoint(task_id, live)
                if live.link is not None and not live.link.request_live(native_id):
                    return False
                current = self.store.get(task_id)
                queued = current.metadata.get("pending_reply") if current is not None else None
                if queued and queued.get("request_id") == live.pending["request_id"]:
                    self.reply(task_id, queued["request_id"], bool(queued.get("approve")))
            self._checkpoint(task_id, live)
            return live.answer
        finally:
            with live.lock:
                live.pending = None
            self.store.update_metadata(task_id, {"pending_input": None, "pending_reply": None})

    async def _execute(self, task: MCPTask, live: _LiveTask, binary: str) -> None:
        self.store.mark_running(task.task_id)
        self.store.mark_progress(task.task_id, 0.05)
        try:
            output = await asyncio.to_thread(self._drive, task.task_id, live, binary)
            self._checkpoint(task.task_id, live)
            if not output.strip():
                raise CARLError("Host returned no answer", code="carl.agent.empty_result")
            artifact_dir = self.store.path.parent / "agent_artifacts" / task.task_id
            artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = artifact_dir / "result.txt"
            payload = output.encode()
            with path.open("xb") as stream:
                os.chmod(path, 0o600)
                stream.write(payload)
            result = {
                "artifact": {
                    "path": str(path),
                    "sha256": content_hash_bytes(payload),
                    "bytes": len(payload),
                },
                "host": live.request.host.value,
            }
            self.store.mark_completed(task.task_id, result)
        except _Cancelled:
            self.store.cancel(task.task_id)
        except Exception as exc:  # noqa: BLE001
            current = self.store.get(task.task_id)
            if live.cancel.is_set() or (
                current is not None and current.metadata.get("cancel_requested")
            ):
                self.store.cancel(task.task_id)
            else:
                code = exc.code if isinstance(exc, CARLError) else "carl.agent.failed"
                self.store.mark_failed(task.task_id, CARLError("Delegation failed", code=code))
        finally:
            with live.lock:
                live.pending = None
            current = self.store.get(task.task_id)
            self.session.chain.record(
                ActionType.EXTERNAL,
                "agent.terminal",
                input={"task_id": task.task_id},
                output={"status": current.status if current else "unknown"},
                success=current is not None and current.status == "completed",
            )
            self.live.pop(task.task_id, None)

    def _drive(self, task_id: str, live: _LiveTask, binary: str) -> str:
        checkpoint = lambda: self._checkpoint(task_id, live)

        def approve(native_id: str, details: dict[str, Any]) -> bool:
            return self._approve(task_id, live, native_id, details)

        with tempfile.TemporaryDirectory(prefix="carl-host-") as profile:
            context = live.context
            env = host_environment(live.request.host, context.root, Path(profile))
            self.store.update_metadata(task_id, {"spawn_started": True})
            if live.request.host == Host.OPENCODE:
                adapter = OpenCodeAdapter()
                live.adapter = adapter
                return adapter.run(
                    self.session.subprocess_toolkit, binary, live.request, env, approve, checkpoint
                )
            adapter = CodexAdapter() if live.request.host == Host.CODEX else ClaudeAdapter()
            live.adapter = adapter
            link = JSONProcess(
                self.session.subprocess_toolkit,
                launch_arguments(live.request.host, binary, live.request),
                cwd=live.request.workdir,
                env=env,
                checkpoint=checkpoint,
                timeout_s=live.request.timeout_s,
            )
            live.link = link
            try:
                self.store.update_metadata(task_id, {"native_pid": link.process.pid})
                return adapter.run(link, live.request, approve)
            finally:
                live.native_terminal = True
                link.close()

    def read_result(self, task_id: str) -> str:
        """Read the requested final artifact after checking its content identity."""
        task = self._owned(task_id)
        result: Any = task.result
        if task.status != "completed" or not isinstance(result, dict):
            raise CARLError("Task has no completed result", code="carl.agent.result")
        artifact: dict[str, Any] = cast(dict[str, Any], result)["artifact"]
        from carl_core.safepath import safe_resolve

        path = safe_resolve(
            artifact["path"], self.store.path.parent / "agent_artifacts", must_exist=True
        )
        payload = path.read_bytes()
        if content_hash_bytes(payload) != artifact["sha256"]:
            raise CARLError("Result artifact changed", code="carl.agent.artifact_changed")
        return payload.decode()

    async def close(self) -> None:
        """Stop every owned live task before releasing Session resources."""
        for task_id, live in list(self.live.items()):
            if live.worker is not None and not live.worker.done():
                await self.cancel(task_id)
        self.session.teardown()


_runtimes: dict[str, HarnessRuntime] = {}
_peer_refs: dict[str, Any] = {}


def get_runtime(ctx: Any = None) -> HarnessRuntime:
    """Resolve the local MCP execution owner lazily."""
    from carl_studio.mcp.protocol import ACTIVE_CONTEXT
    from carl_studio.mcp.server import get_bound_connection

    current = ctx if ctx is not None else ACTIVE_CONTEXT.get()
    bound = get_bound_connection()
    if bound is not None and bound.transport_choice == "stdio":
        key = "stdio:" + bound.connection_id
    elif current is not None:
        headers = getattr(current, "headers", None)
        token = headers.get("mcp-session-id") if headers else None
        if token and current.protocol_version < "2026-07-28":
            key = "http:" + token
        elif current.protocol_version and current.protocol_version < "2026-07-28":
            params = current.session.client_params
            key = "legacy:" + str(id(params))
            _peer_refs[key] = params
        else:
            raise CARLError(
                "Delegation requires a bound caller connection", code="carl.agent.caller_binding"
            )
    else:
        key = "local"
    if key not in _runtimes:
        from carl_studio.mcp.tasks import get_default_store

        root = Path(os.environ.get("CARL_WORKSPACE_ROOT", str(Path.cwd())))
        session = Session(workspace=str(root))
        context = DelegationContext.local(
            owner=uuid.uuid4().hex, root=root, allow_write=os.environ.get("CARL_ALLOW_WRITE") == "1"
        )
        _runtimes[key] = HarnessRuntime(session, context, get_default_store())
    return _runtimes[key]


async def close_runtime() -> None:
    """Close only an initialized execution owner."""
    for runtime in list(_runtimes.values()):
        await runtime.close()
    _runtimes.clear()
    _peer_refs.clear()


__all__ = ["HarnessRuntime", "close_runtime", "get_runtime", "list_harnesses"]
