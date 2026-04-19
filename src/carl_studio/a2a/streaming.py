"""Server-Sent Events (SSE) streaming for the A2A v1.0 spec.

Implements the two streaming endpoints in the Linux Foundation A2A spec:

* ``message/stream`` — mirrors ``message/send`` but yields an SSE stream of
  ``task.status_update`` / ``task.artifact_update`` events as the task
  progresses on our side, instead of returning a single JSON-RPC result.
* ``tasks/subscribe`` — the same stream for an already-existing task. The
  client supplies the task id and we replay current status then push live
  deltas until the task reaches a terminal state (done / failed / cancelled).

The helpers here yield **raw SSE byte frames** — each frame is terminated with
a blank line per the SSE wire format
(https://html.spec.whatwg.org/multipage/server-sent-events.html). Callers
glue these frames into an HTTP response body; we do not own the transport
layer.

Lazy imports:

* ``anyio`` is imported inside the helpers so ``import carl_studio.a2a.streaming``
  stays cheap and the module is importable on environments where the ``a2a``
  extra is not installed. ``anyio`` is also a direct dep (see
  ``pyproject.toml`` ``dependencies``) so the import path is almost never
  actually cold — but we still guard it for defence-in-depth.

Cancellation semantics:

* Each generator listens for consumer cancellation (``anyio`` task cancel,
  async-for break, client disconnect propagated as
  ``anyio.get_cancelled_exc_class``) and exits cleanly. We do not rely on
  finally-time cleanup for the generator — each iteration commits before we
  sleep so a kill at any point leaves the bus state consistent.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from carl_core.connection import ConnectionUnavailableError

from carl_studio.a2a.message import A2AMessage
from carl_studio.a2a.spec import task_to_jsonrpc_result
from carl_studio.a2a.task import A2ATaskStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from carl_studio.a2a.connection import A2APeerConnection, A2AServerConnection


# ---------------------------------------------------------------------------
# SSE event names. Spec-mandated — do not rename.
# ---------------------------------------------------------------------------

EVENT_STATUS = "task.status_update"
EVENT_ARTIFACT = "task.artifact_update"

# Terminal A2A task states. When we observe one we close the stream.
_TERMINAL_STATES: frozenset[A2ATaskStatus] = frozenset(
    {A2ATaskStatus.DONE, A2ATaskStatus.FAILED, A2ATaskStatus.CANCELLED}
)

# Default poll cadence in seconds when watching the bus for deltas.
_DEFAULT_POLL_INTERVAL = 0.1


def _sse_frame(event: str, payload: dict[str, Any]) -> bytes:
    """Encode a single SSE frame ``event: <name>\\ndata: <json>\\n\\n``.

    ``payload`` is serialized as a single-line JSON string — SSE mandates
    one ``data:`` field per line, so we keep it compact to avoid multi-line
    framing and strip any accidental newlines from inputs.
    """
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace(
        "\n", " "
    )
    return f"event: {event}\ndata: {body}\n\n".encode()


def _require_anyio() -> Any:
    """Lazy-import anyio, raising a typed error with an install hint."""
    try:
        import anyio  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ConnectionUnavailableError(
            "anyio is required for A2A streaming",
            context={"hint": "pip install anyio"},
        ) from exc
    return anyio


async def stream_task_updates(
    server: A2AServerConnection,
    task_id: str,
    *,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
    max_iterations: int | None = None,
) -> AsyncIterator[bytes]:
    """Yield SSE frames for a task until it reaches a terminal state.

    Parameters
    ----------
    server
        The :class:`A2AServerConnection` whose :class:`LocalBus` holds the
        task being watched.
    task_id
        The task to stream.
    poll_interval
        Seconds between bus polls. Small values hammer sqlite; typical
        production values are 0.1 - 0.5s.
    max_iterations
        Optional safety stop (tests use this to bound a misbehaving bus).
        ``None`` means "stream until terminal".

    Yields
    ------
    bytes
        Valid SSE frames. The first frame is always the current status; any
        artifact messages attached to the task flush as additional frames;
        status transitions emit a trailing ``task.status_update`` frame each
        time the task status changes.

    Raises
    ------
    ConnectionUnavailableError
        If the server has no bus attached or the requested task cannot be
        found at the first poll.
    """
    anyio = _require_anyio()
    bus = server.bus  # raises ConnectionUnavailableError if unopened

    # Seed: first poll MUST find the task. If it doesn't, fail loudly — the
    # spec treats subscribe-before-create as an error, not an empty stream.
    task = bus.get(task_id)
    if task is None:
        raise ConnectionUnavailableError(
            f"tasks/subscribe: task not found: {task_id}",
            context={"spec_name": server.spec.name, "task_id": task_id},
        )

    last_status: A2ATaskStatus | None = None
    seen_messages: set[str] = set()

    # Emit initial status snapshot.
    yield _sse_frame(EVENT_STATUS, task_to_jsonrpc_result(task))
    last_status = task.status

    iteration = 0
    while True:
        iteration += 1
        if max_iterations is not None and iteration > max_iterations:
            return

        if task.status in _TERMINAL_STATES:
            # Flush any unseen artifacts before we close.
            for msg in bus.get_messages(task_id):
                if msg.id in seen_messages:
                    continue
                seen_messages.add(msg.id)
                if msg.type == "artifact":
                    yield _sse_frame(EVENT_ARTIFACT, _artifact_payload(task_id, msg))
            return

        try:
            await anyio.sleep(poll_interval)
        except anyio.get_cancelled_exc_class():
            # Consumer went away — exit cleanly.
            return

        refreshed = bus.get(task_id)
        if refreshed is None:
            # Task vanished mid-stream (e.g. external DB reset). Treat as
            # terminal to avoid spinning forever.
            return
        task = refreshed

        # Flush any new artifact messages.
        for msg in bus.get_messages(task_id):
            if msg.id in seen_messages:
                continue
            seen_messages.add(msg.id)
            if msg.type == "artifact":
                yield _sse_frame(EVENT_ARTIFACT, _artifact_payload(task_id, msg))

        if task.status != last_status:
            yield _sse_frame(EVENT_STATUS, task_to_jsonrpc_result(task))
            last_status = task.status


async def stream_message(
    peer_or_server: A2APeerConnection | A2AServerConnection,
    message: A2AMessage,
    *,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
    max_iterations: int | None = None,
) -> AsyncIterator[bytes]:
    """Yield SSE frames for ``message/stream``.

    * When called on an :class:`A2AServerConnection`, we post the message to
      our local bus (creating a new task for it) and stream that task's
      progress.
    * When called on an :class:`A2APeerConnection`, we forward the message
      via ``message/send`` on the remote, pull the returned task id, and
      stream updates by polling ``tasks/get`` on the peer. (The peer may
      also run its own native SSE stream; we model that as a future
      enhancement — polling is always correct.)
    """
    # Local import to avoid top-level cycle with connection.py, which
    # imports this module symbolically under TYPE_CHECKING.
    from carl_studio.a2a.connection import A2APeerConnection, A2AServerConnection
    from carl_studio.a2a.task import A2ATask

    anyio = _require_anyio()

    if isinstance(peer_or_server, A2AServerConnection):
        # Server-side: materialize a task + post a starter message.
        payload_any: Any = message.payload
        skill_hint: str = "stream"
        if isinstance(payload_any, dict):
            payload_map: dict[str, Any] = dict(payload_any)  # type: ignore[arg-type]
            skill_val: Any = payload_map.get("skill", "stream")
            skill_hint = str(skill_val) if skill_val is not None else "stream"
        task = A2ATask(
            id=message.task_id or _new_task_id(),
            skill=skill_hint,
            inputs={"text": str(message.payload)},
            sender=message.sender,
        )
        server = peer_or_server
        server.bus.post(task)
        async for frame in stream_task_updates(
            server,
            task.id,
            poll_interval=poll_interval,
            max_iterations=max_iterations,
        ):
            yield frame
        return

    if not isinstance(peer_or_server, A2APeerConnection):  # pyright: ignore[reportUnnecessaryIsInstance]
        # Defensive: callers may pass a random object; the union above is
        # what pyright sees, but at runtime we want a real TypeError rather
        # than an AttributeError buried deep in send_message.
        raise TypeError(
            "stream_message expects A2APeerConnection or A2AServerConnection, "
            f"got {type(peer_or_server).__name__}"
        )

    peer = peer_or_server
    # Dispatch ``message/send`` to the peer to create the task remotely.
    body = await peer.send_message(message)
    body_dict: dict[str, Any] = dict(body) if isinstance(body, dict) else {}  # type: ignore[arg-type,reportUnnecessaryIsInstance]
    result_any: Any = body_dict.get("result")
    if not isinstance(result_any, dict):
        raise ConnectionUnavailableError(
            "peer returned no result for message/stream bootstrap",
            context={"spec_name": peer.spec.name},
        )
    result: dict[str, Any] = dict(result_any)  # type: ignore[arg-type]
    raw_id: Any = result.get("id")
    task_id = str(raw_id) if raw_id is not None else ""
    if not task_id:
        raise ConnectionUnavailableError(
            "peer returned no task id for message/stream",
            context={"spec_name": peer.spec.name},
        )

    # Emit initial snapshot, then poll ``tasks/get`` for updates.
    last_state: str | None = None
    iteration = 0
    while True:
        iteration += 1
        if max_iterations is not None and iteration > max_iterations:
            return
        try:
            task_model = await peer.get_task(task_id)
        except ConnectionUnavailableError:
            return
        state = task_model.status.value
        if state != last_state:
            yield _sse_frame(EVENT_STATUS, task_to_jsonrpc_result(task_model))
            last_state = state
        if task_model.status in _TERMINAL_STATES:
            return
        try:
            await anyio.sleep(poll_interval)
        except anyio.get_cancelled_exc_class():
            return


# ---------------------------------------------------------------------------
# Parsing helper — consumes the generator output for client-side callers.
# ---------------------------------------------------------------------------


async def parse_sse_events(
    stream: AsyncIterator[bytes],
) -> AsyncIterator[dict[str, Any]]:
    """Parse a raw SSE byte stream back into ``{"event": ..., "data": ...}`` dicts.

    Lenient: ignores comment lines (``: foo``), accepts either LF or CRLF
    line endings, drops blank frames. Designed to mirror what a spec-compliant
    A2A client would do when consuming our ``stream_task_updates`` output.
    """
    buf = ""
    async for chunk in stream:
        buf += chunk.decode("utf-8", errors="replace")
        # Normalize CRLF -> LF so the frame splitter works uniformly.
        buf = buf.replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in buf:
            frame, buf = buf.split("\n\n", 1)
            event = ""
            data_lines: list[str] = []
            for raw_line in frame.split("\n"):
                line = raw_line.rstrip("\r")
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())
            if not event and not data_lines:
                continue
            data_str = "\n".join(data_lines)
            try:
                data_obj: Any = json.loads(data_str) if data_str else {}
            except json.JSONDecodeError:
                data_obj = {"raw": data_str}
            yield {"event": event, "data": data_obj}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _artifact_payload(task_id: str, msg: A2AMessage) -> dict[str, Any]:
    """Shape an A2A ``task.artifact_update`` event payload from an A2AMessage."""
    payload_any: Any = msg.payload or {}
    payload: dict[str, Any] = (
        dict(payload_any) if isinstance(payload_any, dict) else {}  # type: ignore[arg-type]
    )
    return {
        "taskId": task_id,
        "artifact": {
            "name": payload.get("name", ""),
            "data": payload.get("data", {}),
            "messageId": msg.id,
            "timestamp": msg.timestamp,
        },
    }


def _new_task_id() -> str:
    from uuid import uuid4

    return str(uuid4())


__all__ = [
    "EVENT_STATUS",
    "EVENT_ARTIFACT",
    "stream_task_updates",
    "stream_message",
    "parse_sse_events",
]
