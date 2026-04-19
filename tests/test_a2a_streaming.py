"""Tests for :mod:`carl_studio.a2a.streaming`.

Covers SSE frame encoding, ``stream_task_updates`` event flow (status +
artifact), ``stream_message`` server-side dispatch, and the client-side
``parse_sse_events`` round-trip.

All tests run in-process against a fresh :class:`LocalBus` — no network,
no ``httpx`` required. Matches the existing ``test_a2a_connection.py``
pattern: async logic wrapped in ``asyncio.run`` so pytest-asyncio is
not required.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest

from carl_core.connection import ConnectionUnavailableError, reset_registry

from carl_studio.a2a import (
    EVENT_ARTIFACT,
    EVENT_STATUS,
    A2AMessage,
    A2AServerConnection,
    A2ATask,
    LocalBus,
    parse_sse_events,
    stream_message,
    stream_task_updates,
)


@pytest.fixture(autouse=True)
def _reset_registry():  # pyright: ignore[reportUnusedFunction]
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def tmp_bus(tmp_path: Path):
    bus = LocalBus(db_path=tmp_path / "a2a.db")
    yield bus
    bus.close()


async def _iter_bytes(frames: list[bytes]) -> AsyncIterator[bytes]:
    for f in frames:
        yield f


def _collect_events(frames: list[bytes]) -> list[dict[str, Any]]:
    async def _run() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async for e in parse_sse_events(_iter_bytes(frames)):
            events.append(e)
        return events

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# SSE framing
# ---------------------------------------------------------------------------


class TestSSEFraming:
    def test_status_update_yields_valid_sse_frame(self, tmp_bus: LocalBus) -> None:
        async def run() -> list[bytes]:
            server = A2AServerConnection(bus=tmp_bus)
            await server.open()
            try:
                task = A2ATask(id="t1", skill="train")
                tmp_bus.post(task)
                frames: list[bytes] = []
                gen = stream_task_updates(
                    server, "t1", poll_interval=0.01, max_iterations=3
                )
                async for frame in gen:
                    frames.append(frame)
                    if len(frames) >= 1:
                        break
                # AsyncIterator protocol does not advertise aclose, but the
                # generator object does. Casting here keeps pyright quiet
                # without disabling strict checks for the rest of the file.
                await cast(AsyncGenerator[bytes, None], gen).aclose()
                return frames
            finally:
                await server.close()

        frames = asyncio.run(run())
        assert frames, "expected at least one frame"
        text = frames[0].decode("utf-8")
        assert text.startswith("event: ")
        assert "\ndata: " in text
        assert text.endswith("\n\n")
        first_line = text.split("\n", 1)[0]
        assert first_line == f"event: {EVENT_STATUS}"

    def test_missing_task_raises_connection_unavailable(self, tmp_bus: LocalBus) -> None:
        async def run() -> None:
            server = A2AServerConnection(bus=tmp_bus)
            await server.open()
            try:
                gen = stream_task_updates(server, "nope", poll_interval=0.01)
                async for _ in gen:
                    pass
            finally:
                await server.close()

        with pytest.raises(ConnectionUnavailableError):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# Status transitions + artifacts.
# ---------------------------------------------------------------------------


class TestTaskStreamEventSequence:
    def test_status_transitions_emit_frames(self, tmp_bus: LocalBus) -> None:
        async def run() -> list[bytes]:
            server = A2AServerConnection(bus=tmp_bus)
            await server.open()
            try:
                task = A2ATask(id="tstream", skill="train")
                tmp_bus.post(task)
                frames: list[bytes] = []

                async def _consume() -> None:
                    gen = stream_task_updates(
                        server, "tstream", poll_interval=0.01, max_iterations=50
                    )
                    async for frame in gen:
                        frames.append(frame)

                async def _driver() -> None:
                    await asyncio.sleep(0.03)
                    tmp_bus.update(task.mark_running())
                    await asyncio.sleep(0.05)
                    tmp_bus.update(
                        task.mark_running().mark_done({"accuracy": 0.9})
                    )

                await asyncio.gather(_consume(), _driver())
                return frames
            finally:
                await server.close()

        frames = asyncio.run(run())
        events = _collect_events(frames)
        names = [e["event"] for e in events]
        assert names.count(EVENT_STATUS) >= 2
        states = [e["data"]["status"]["state"] for e in events]
        assert states[0] == "submitted"
        assert states[-1] == "completed"

    def test_artifact_messages_emit_artifact_frames(self, tmp_bus: LocalBus) -> None:
        async def run() -> list[bytes]:
            server = A2AServerConnection(bus=tmp_bus)
            await server.open()
            try:
                task = A2ATask(id="tart", skill="train")
                tmp_bus.post(task)
                frames: list[bytes] = []

                async def _consume() -> None:
                    gen = stream_task_updates(
                        server, "tart", poll_interval=0.01, max_iterations=50
                    )
                    async for frame in gen:
                        frames.append(frame)

                async def _driver() -> None:
                    await asyncio.sleep(0.03)
                    tmp_bus.publish_message(
                        A2AMessage.artifact(
                            "tart", "model-ckpt", {"uri": "hf://foo/bar"}
                        )
                    )
                    await asyncio.sleep(0.05)
                    tmp_bus.update(task.mark_done({"accuracy": 1.0}))

                await asyncio.gather(_consume(), _driver())
                return frames
            finally:
                await server.close()

        frames = asyncio.run(run())
        events = _collect_events(frames)
        artifact_events = [e for e in events if e["event"] == EVENT_ARTIFACT]
        assert artifact_events, "expected at least one task.artifact_update event"
        art_data = artifact_events[0]["data"]
        assert art_data["taskId"] == "tart"
        assert art_data["artifact"]["name"] == "model-ckpt"
        assert art_data["artifact"]["data"]["uri"] == "hf://foo/bar"

    def test_terminal_status_closes_stream_immediately(self, tmp_bus: LocalBus) -> None:
        async def run() -> list[bytes]:
            server = A2AServerConnection(bus=tmp_bus)
            await server.open()
            try:
                task = A2ATask(id="tterm", skill="train")
                tmp_bus.post(task)
                tmp_bus.update(task.mark_done({"ok": True}))
                frames: list[bytes] = []
                async for frame in stream_task_updates(
                    server, "tterm", poll_interval=0.01, max_iterations=3
                ):
                    frames.append(frame)
                return frames
            finally:
                await server.close()

        frames = asyncio.run(run())
        assert len(frames) == 1
        events = _collect_events(frames)
        assert events[0]["data"]["status"]["state"] == "completed"


# ---------------------------------------------------------------------------
# stream_message server-side dispatch.
# ---------------------------------------------------------------------------


class TestStreamMessageServer:
    def test_server_side_creates_task_and_streams(self, tmp_bus: LocalBus) -> None:
        async def run() -> list[bytes]:
            server = A2AServerConnection(bus=tmp_bus)
            await server.open()
            try:
                msg = A2AMessage(
                    id="m1",
                    task_id="smtask",
                    type="progress",
                    payload={"skill": "train", "text": "hello"},
                    sender="user",
                )
                frames: list[bytes] = []

                async def _consume() -> None:
                    gen = stream_message(
                        server, msg, poll_interval=0.01, max_iterations=50
                    )
                    async for frame in gen:
                        frames.append(frame)

                async def _driver() -> None:
                    # Wait for server to materialize the task, then terminate.
                    for _ in range(30):
                        task = tmp_bus.get("smtask")
                        if task is not None:
                            tmp_bus.update(task.mark_done({"ok": True}))
                            return
                        await asyncio.sleep(0.01)
                    raise AssertionError("server never posted the task")

                await asyncio.gather(_consume(), _driver())
                return frames
            finally:
                await server.close()

        frames = asyncio.run(run())
        assert frames
        events = _collect_events(frames)
        assert events[-1]["data"]["status"]["state"] == "completed"

    def test_stream_message_rejects_non_connection(self) -> None:
        async def run() -> None:
            gen = stream_message(
                "not-a-connection",  # type: ignore[arg-type]
                A2AMessage(id="x", task_id="y", type="log"),
            )
            async for _ in gen:
                pass

        with pytest.raises(TypeError):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# parse_sse_events lenient input.
# ---------------------------------------------------------------------------


class TestParseSSEEvents:
    def test_parses_single_frame(self) -> None:
        events = _collect_events(
            [b"event: task.status_update\ndata: {\"ok\":1}\n\n"]
        )
        assert events == [{"event": "task.status_update", "data": {"ok": 1}}]

    def test_ignores_comments_and_blank_lines(self) -> None:
        raw = b": heartbeat\n\nevent: task.status_update\ndata: {\"k\":\"v\"}\n\n"
        events = _collect_events([raw])
        assert len(events) == 1
        assert events[0]["data"] == {"k": "v"}

    def test_handles_crlf_line_endings(self) -> None:
        raw = b"event: task.artifact_update\r\ndata: {\"x\":1}\r\n\r\n"
        events = _collect_events([raw])
        assert events == [{"event": "task.artifact_update", "data": {"x": 1}}]

    def test_malformed_json_returned_as_raw(self) -> None:
        raw = b"event: task.status_update\ndata: not-json\n\n"
        events = _collect_events([raw])
        assert events[0]["data"] == {"raw": "not-json"}


# ---------------------------------------------------------------------------
# Round-trip.
# ---------------------------------------------------------------------------


def test_sse_round_trip_preserves_event_payload(tmp_bus: LocalBus) -> None:
    async def run() -> list[bytes]:
        server = A2AServerConnection(bus=tmp_bus)
        await server.open()
        try:
            task = A2ATask(id="rt", skill="train", sender="client")
            tmp_bus.post(task)
            tmp_bus.update(task.mark_done({"score": 0.777}))
            frames: list[bytes] = []
            async for frame in stream_task_updates(
                server, "rt", poll_interval=0.01, max_iterations=3
            ):
                frames.append(frame)
            return frames
        finally:
            await server.close()

    frames = asyncio.run(run())
    events = _collect_events(frames)
    data = events[-1]["data"]
    assert data["id"] == "rt"
    assert data["status"]["state"] == "completed"
    assert data["metadata"]["skill"] == "train"
