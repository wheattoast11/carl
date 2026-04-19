"""Tests for carl_studio.heartbeat (ARC-003 + ARC-006).

Covers the three seams:

* :class:`HeartbeatLoop` — daemon-thread worker that drains the queue.
* :class:`HeartbeatConnection` — :class:`AsyncBaseConnection` adapter.
* :func:`poll_and_print` — the UI surface helper.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from carl_core.connection.lifecycle import ConnectionState
from carl_core.interaction import ActionType, InteractionChain

from carl_studio.db import LocalDB
from carl_studio.heartbeat import (
    HeartbeatConnection,
    HeartbeatLoop,
    HeartbeatPhase,
    ORDERED_PHASES,
)
from carl_studio.heartbeat.phases import HeartbeatPhase as PhaseReexport
from carl_studio.sticky import StickyQueue

_WAIT_TIMEOUT_S = 2.0
_POLL_INTERVAL_S = 0.01


@pytest.fixture()
def queue(tmp_path: Path) -> StickyQueue:
    """Isolated :class:`StickyQueue` rooted in ``tmp_path``."""
    db = LocalDB(tmp_path / "carl.db")
    return StickyQueue(db)


@pytest.fixture()
def chain() -> InteractionChain:
    """Fresh :class:`InteractionChain` for every test."""
    return InteractionChain()


def _wait_for(
    predicate: Callable[[], bool],
    timeout_s: float = _WAIT_TIMEOUT_S,
) -> bool:
    """Busy-wait until ``predicate()`` is truthy or ``timeout_s`` elapses.

    Returns the final value of ``predicate()`` so callers can assert.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


# ---------------------------------------------------------------------------
# HeartbeatPhase ordering
# ---------------------------------------------------------------------------


def test_phase_ordering_is_fixed_and_complete() -> None:
    assert ORDERED_PHASES[0] is HeartbeatPhase.INTERVIEW
    assert ORDERED_PHASES[-1] is HeartbeatPhase.AWAIT
    assert len(ORDERED_PHASES) == len(HeartbeatPhase)
    # Re-export check — surface module must expose the same symbol.
    assert PhaseReexport is HeartbeatPhase


# ---------------------------------------------------------------------------
# HeartbeatLoop — core behaviour
# ---------------------------------------------------------------------------


def test_heartbeat_processes_queued_note(queue: StickyQueue, chain: InteractionChain) -> None:
    """A queued note transitions to ``done`` with a phases-bearing result."""
    note = queue.append("train a small classifier", priority=5)
    loop = HeartbeatLoop(queue, chain, poll_interval_s=_POLL_INTERVAL_S)
    loop.start()
    try:
        ok = _wait_for(lambda: (queue.get(note.id) or note).status == "done")
        assert ok, "heartbeat did not flip note to done within timeout"
    finally:
        loop.stop()

    final = queue.get(note.id)
    assert final is not None
    assert final.status == "done"
    assert final.result is not None
    assert final.result.get("phases") == [p.value for p in ORDERED_PHASES]
    assert final.result.get("note_id") == note.id


def test_heartbeat_records_cycle_start_and_end(
    queue: StickyQueue, chain: InteractionChain
) -> None:
    """Each cycle emits start + end telemetry on :class:`InteractionChain`."""
    note = queue.append("reward shaping check", priority=5)
    loop = HeartbeatLoop(queue, chain, poll_interval_s=_POLL_INTERVAL_S)
    loop.start()
    try:
        _wait_for(lambda: (queue.get(note.id) or note).status == "done")
    finally:
        loop.stop()

    cycle_steps = chain.by_action(ActionType.HEARTBEAT_CYCLE)
    # Expect exactly one start + one end for a single queued note.
    starts = [s for s in cycle_steps if s.name == "cycle:start"]
    ends = [s for s in cycle_steps if s.name == "cycle:end"]
    assert len(starts) == 1, f"expected 1 cycle:start, got {len(starts)}"
    assert len(ends) == 1, f"expected 1 cycle:end, got {len(ends)}"

    start_step, end_step = starts[0], ends[0]
    assert start_step.input is not None
    assert start_step.input.get("note_id") == note.id
    assert start_step.input.get("content", "").startswith("reward shaping")

    assert end_step.output is not None
    assert end_step.output.get("phases") == [p.value for p in ORDERED_PHASES]
    assert end_step.success is True
    assert end_step.duration_ms is not None
    assert end_step.duration_ms >= 0.0


def test_heartbeat_idempotent_start(queue: StickyQueue, chain: InteractionChain) -> None:
    """Calling :meth:`start` twice returns the same thread — no stacked daemons."""
    loop = HeartbeatLoop(queue, chain, poll_interval_s=_POLL_INTERVAL_S)
    try:
        t1 = loop.start()
        t2 = loop.start()
        assert t1 is t2
        assert t1.is_alive()
        # Only one thread should have our name.
        named = [t for t in threading.enumerate() if t.name == "carl-heartbeat"]
        assert len(named) == 1
    finally:
        loop.stop()


def test_heartbeat_stop_joins_cleanly(queue: StickyQueue, chain: InteractionChain) -> None:
    """:meth:`stop` returns within the timeout and leaves ``is_running`` False."""
    loop = HeartbeatLoop(queue, chain, poll_interval_s=_POLL_INTERVAL_S)
    loop.start()
    assert loop.is_running
    t0 = time.monotonic()
    loop.stop(timeout=2.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"stop() took {elapsed:.3f}s"
    assert not loop.is_running

    # Re-stopping an already-stopped loop is a no-op and must not raise.
    loop.stop(timeout=0.1)


def test_heartbeat_loop_rejects_bad_poll_interval(
    queue: StickyQueue, chain: InteractionChain
) -> None:
    with pytest.raises(ValueError, match="poll_interval_s"):
        HeartbeatLoop(queue, chain, poll_interval_s=0.0)
    with pytest.raises(ValueError, match="poll_interval_s"):
        HeartbeatLoop(queue, chain, poll_interval_s=-1.0)


def test_heartbeat_processes_multiple_notes_in_priority_order(
    queue: StickyQueue, chain: InteractionChain
) -> None:
    """High-priority notes are drained first."""
    low = queue.append("low task", priority=1)
    time.sleep(0.01)
    high = queue.append("high task", priority=9)

    loop = HeartbeatLoop(queue, chain, poll_interval_s=_POLL_INTERVAL_S)
    loop.start()
    try:
        ok = _wait_for(
            lambda: (
                (queue.get(low.id) or low).status == "done"
                and (queue.get(high.id) or high).status == "done"
            )
        )
        assert ok, "both notes should have completed"
    finally:
        loop.stop()

    # The high-priority cycle:start must come before the low one.
    starts = [
        s for s in chain.by_action(ActionType.HEARTBEAT_CYCLE) if s.name == "cycle:start"
    ]
    assert len(starts) == 2
    note_ids_in_order = [s.input.get("note_id") for s in starts]  # type: ignore[union-attr]
    assert note_ids_in_order == [high.id, low.id]


# ---------------------------------------------------------------------------
# HeartbeatConnection — AsyncBaseConnection contract
# ---------------------------------------------------------------------------


def test_heartbeat_connection_lifecycle(queue: StickyQueue) -> None:
    """``open()`` → READY, ``close()`` → CLOSED, with registry integration."""
    conn = HeartbeatConnection(queue, poll_interval_s=_POLL_INTERVAL_S)
    assert conn.state == ConnectionState.INIT

    async def drive() -> None:
        await conn.open()
        try:
            assert conn.state == ConnectionState.READY
            assert conn.loop.is_running
        finally:
            await conn.close()

    asyncio.run(drive())

    assert conn.state == ConnectionState.CLOSED
    assert not conn.loop.is_running
    assert conn.stats.opens == 1
    assert conn.stats.closes == 1


def test_heartbeat_connection_drains_queue_once_open(queue: StickyQueue) -> None:
    """The connection's loop processes notes once in READY state."""
    note = queue.append("first work item", priority=5)

    conn = HeartbeatConnection(queue, poll_interval_s=_POLL_INTERVAL_S)

    async def drive() -> None:
        await conn.open()
        try:
            # Give the worker up to 2s (ample for a 10ms poll) to drain.
            async def wait_done() -> None:
                while True:
                    current = queue.get(note.id)
                    if current is not None and current.status == "done":
                        return
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(wait_done(), timeout=_WAIT_TIMEOUT_S)
        finally:
            await conn.close()

    asyncio.run(drive())

    final = queue.get(note.id)
    assert final is not None and final.status == "done"


def test_heartbeat_connection_drain_status_returns_messages(queue: StickyQueue) -> None:
    """:meth:`drain_status` surfaces buffered phase strings and then clears."""
    note = queue.append("drain-status probe", priority=5)
    conn = HeartbeatConnection(queue, poll_interval_s=_POLL_INTERVAL_S)

    async def drive() -> None:
        await conn.open()
        try:
            async def wait_done() -> None:
                while True:
                    current = queue.get(note.id)
                    if current is not None and current.status == "done":
                        return
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(wait_done(), timeout=_WAIT_TIMEOUT_S)
        finally:
            await conn.close()

    asyncio.run(drive())

    messages = conn.drain_status()
    assert messages, "drain_status should return at least cycle:start/end"
    joined = " ".join(m if isinstance(m, str) else str(m) for m in messages)
    assert "cycle:start" in joined
    assert "cycle:end" in joined
    # Second drain is empty — the first call cleared the buffer.
    assert conn.drain_status() == []


def test_heartbeat_connection_rejects_bad_status_buffer(queue: StickyQueue) -> None:
    with pytest.raises(ValueError, match="status_buffer"):
        HeartbeatConnection(queue, status_buffer=0)


def test_heartbeat_connection_async_context_manager(queue: StickyQueue) -> None:
    """``async with`` drives open/close and leaves the connection CLOSED."""

    async def drive() -> HeartbeatConnection:
        c = HeartbeatConnection(queue, poll_interval_s=_POLL_INTERVAL_S)
        async with c as entered:
            assert entered is c
            assert c.state == ConnectionState.READY
        return c

    conn = asyncio.run(drive())
    assert conn.state == ConnectionState.CLOSED


def test_heartbeat_connection_uses_provided_chain(queue: StickyQueue) -> None:
    """A caller-supplied :class:`InteractionChain` is the recipient of telemetry."""
    chain = InteractionChain()
    note = queue.append("custom-chain probe", priority=5)
    conn = HeartbeatConnection(queue, chain=chain, poll_interval_s=_POLL_INTERVAL_S)

    async def drive() -> None:
        await conn.open()
        try:
            async def wait_done() -> None:
                while True:
                    current = queue.get(note.id)
                    if current is not None and current.status == "done":
                        return
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(wait_done(), timeout=_WAIT_TIMEOUT_S)
        finally:
            await conn.close()

    asyncio.run(drive())

    cycle_steps = chain.by_action(ActionType.HEARTBEAT_CYCLE)
    assert len(cycle_steps) >= 2
    assert any(s.name == "cycle:start" for s in cycle_steps)
    assert any(s.name == "cycle:end" for s in cycle_steps)


# ---------------------------------------------------------------------------
# surface.poll_and_print
# ---------------------------------------------------------------------------


def test_poll_and_print_forwards_messages(queue: StickyQueue) -> None:
    """:func:`poll_and_print` drains the connection onto a console-like sink."""
    # Import late so the module is typed against the real CampConsole.
    from carl_studio.console import CampConsole
    from carl_studio.heartbeat.surface import poll_and_print

    note = queue.append("surface-probe note", priority=5)
    conn = HeartbeatConnection(queue, poll_interval_s=_POLL_INTERVAL_S)

    async def drive() -> None:
        await conn.open()
        try:
            async def wait_done() -> None:
                while True:
                    current = queue.get(note.id)
                    if current is not None and current.status == "done":
                        return
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(wait_done(), timeout=_WAIT_TIMEOUT_S)
        finally:
            await conn.close()

    asyncio.run(drive())

    console = CampConsole()
    printed = poll_and_print(conn, console)
    # At minimum the start/end cycle lines.
    assert printed >= 2
    # Buffer cleared — second poll returns zero.
    assert poll_and_print(conn, console) == 0
