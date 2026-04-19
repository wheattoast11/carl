"""Tests for the E1 graceful-shutdown discipline.

Three layers are exercised:

* :class:`HeartbeatLoop._run_cycle` — on exception the note flips back to
  ``queued`` (not ``processing`` forever, and not prematurely ``done``).
* :class:`HeartbeatLoop._aloop` — stop-event is only honoured between
  cycles, not mid-cycle, so a running cycle always finishes.
* :mod:`carl_studio.heartbeat.daemon` — the ``_install_signal_handlers``
  helper flips the async stop-event; ``_run`` shuts down cleanly when
  that event is set.

The daemon-level test drives the async helper directly instead of
spawning a subprocess — subprocess signal tests are flaky on CI and
offer no additional coverage of our own code (only of the OS's signal
plumbing).
"""
# pyright: reportPrivateUsage=false, reportPrivateImportUsage=false

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from carl_core.interaction import InteractionChain

from carl_studio.db import LocalDB
from carl_studio.heartbeat import daemon as daemon_mod
from carl_studio.heartbeat.connection import HeartbeatConnection
from carl_studio.heartbeat.loop import HeartbeatLoop
from carl_studio.heartbeat.phases import HeartbeatPhase
from carl_studio.sticky import StickyNote, StickyQueue

_WAIT_TIMEOUT_S = 3.0
_POLL_INTERVAL_S = 0.01


def _wait_for(
    predicate: Callable[[], bool],
    timeout_s: float = _WAIT_TIMEOUT_S,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


@pytest.fixture()
def queue(tmp_path: Path) -> StickyQueue:
    db = LocalDB(tmp_path / "carl.db")
    return StickyQueue(db)


@pytest.fixture()
def chain() -> InteractionChain:
    return InteractionChain()


# ---------------------------------------------------------------------------
# _run_cycle — exception path flips the note back to queued
# ---------------------------------------------------------------------------


class _RaisingLoop(HeartbeatLoop):
    """Test double that raises from a chosen phase."""

    def __init__(
        self,
        queue: StickyQueue,
        chain: InteractionChain,
        *,
        raise_on: HeartbeatPhase,
    ) -> None:
        super().__init__(queue, chain, poll_interval_s=_POLL_INTERVAL_S)
        self._raise_on = raise_on

    async def _run_phase(
        self,
        phase: HeartbeatPhase,
        note: StickyNote,
        acc: dict[str, Any],
    ) -> None:
        if phase is self._raise_on:
            raise RuntimeError(f"forced failure at {phase.value}")
        await super()._run_phase(phase, note, acc)


def test_cycle_exception_requeues_note(queue: StickyQueue, chain: InteractionChain) -> None:
    """An exception mid-cycle must flip the note back to ``queued``."""
    note = queue.append("boom", priority=5)
    loop = _RaisingLoop(queue, chain, raise_on=HeartbeatPhase.EXECUTE)

    async def drive() -> None:
        # We don't start the thread — we drive ``_run_cycle`` directly so
        # the test is deterministic.
        claimed = queue.dequeue()
        assert claimed is not None and claimed.id == note.id
        await loop._run_cycle(queue, claimed)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(drive())

    after = queue.get(note.id)
    assert after is not None
    assert after.status == "queued", (
        f"expected queued after exception, got {after.status}"
    )
    assert after.started_at is None


def test_cycle_success_still_marks_done(queue: StickyQueue, chain: InteractionChain) -> None:
    """Happy path — no exception — still transitions to ``done``."""
    note = queue.append("ok", priority=5)
    loop = HeartbeatLoop(queue, chain, poll_interval_s=_POLL_INTERVAL_S)

    async def drive() -> None:
        claimed = queue.dequeue()
        assert claimed is not None
        await loop._run_cycle(queue, claimed)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(drive())

    after = queue.get(note.id)
    assert after is not None
    assert after.status == "done"


# ---------------------------------------------------------------------------
# _aloop — stop only honoured between cycles
# ---------------------------------------------------------------------------


class _SlowPhaseLoop(HeartbeatLoop):
    """Test double — pauses inside one phase so we can race ``stop()``."""

    def __init__(self, queue: StickyQueue, chain: InteractionChain) -> None:
        super().__init__(queue, chain, poll_interval_s=_POLL_INTERVAL_S)
        self._entered = threading.Event()
        self._may_finish = threading.Event()

    async def _run_phase(
        self,
        phase: HeartbeatPhase,
        note: StickyNote,
        acc: dict[str, Any],
    ) -> None:
        if phase is HeartbeatPhase.EXECUTE:
            self._entered.set()
            # Block until the test allows the phase to complete. We use
            # a thread-safe event because the test drives it from a
            # different thread than the worker.
            while not self._may_finish.is_set():
                await asyncio.sleep(0.01)
        await super()._run_phase(phase, note, acc)


def test_stop_event_between_cycles_not_mid_cycle(
    queue: StickyQueue,
    chain: InteractionChain,
) -> None:
    """``stop()`` during a cycle must NOT abort it — the note still
    transitions to ``done`` when the phase loop returns."""
    note = queue.append("survive", priority=5)
    loop = _SlowPhaseLoop(queue, chain)
    loop.start()
    try:
        # Wait for the worker to enter EXECUTE.
        ok = loop._entered.wait(timeout=_WAIT_TIMEOUT_S)  # pyright: ignore[reportPrivateUsage]
        assert ok, "worker did not enter EXECUTE within timeout"

        # Signal stop while the cycle is mid-phase. The worker must still
        # finish this cycle — that's the graceful-shutdown contract.
        loop._stop_event.set()  # pyright: ignore[reportPrivateUsage]
        # Release the phase so the cycle can complete.
        loop._may_finish.set()  # pyright: ignore[reportPrivateUsage]

        ok = _wait_for(lambda: not loop.is_running)
        assert ok, "worker did not exit after stop"
    finally:
        loop._may_finish.set()  # pyright: ignore[reportPrivateUsage]
        loop.stop(timeout=_WAIT_TIMEOUT_S)

    final = queue.get(note.id)
    assert final is not None
    assert final.status == "done", (
        f"cycle should have completed despite stop; got status={final.status}"
    )


# ---------------------------------------------------------------------------
# HeartbeatConnection shutdown timeout honours env var
# ---------------------------------------------------------------------------


def test_connection_shutdown_timeout_from_env(
    monkeypatch: pytest.MonkeyPatch,
    queue: StickyQueue,
) -> None:
    monkeypatch.setenv("CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S", "45")
    conn = HeartbeatConnection(queue, poll_interval_s=_POLL_INTERVAL_S)
    assert conn._shutdown_timeout_s == 45.0  # pyright: ignore[reportPrivateUsage]


def test_connection_shutdown_timeout_default(
    monkeypatch: pytest.MonkeyPatch,
    queue: StickyQueue,
) -> None:
    monkeypatch.delenv("CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S", raising=False)
    conn = HeartbeatConnection(queue, poll_interval_s=_POLL_INTERVAL_S)
    assert conn._shutdown_timeout_s == 30.0  # pyright: ignore[reportPrivateUsage]


def test_connection_shutdown_timeout_explicit_wins(
    monkeypatch: pytest.MonkeyPatch,
    queue: StickyQueue,
) -> None:
    monkeypatch.setenv("CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S", "9999")
    conn = HeartbeatConnection(
        queue,
        poll_interval_s=_POLL_INTERVAL_S,
        shutdown_timeout_s=7.5,
    )
    assert conn._shutdown_timeout_s == 7.5  # pyright: ignore[reportPrivateUsage]


def test_connection_shutdown_timeout_invalid_env_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    queue: StickyQueue,
) -> None:
    monkeypatch.setenv("CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S", "not-a-number")
    conn = HeartbeatConnection(queue, poll_interval_s=_POLL_INTERVAL_S)
    assert conn._shutdown_timeout_s == 30.0  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# daemon._install_signal_handlers — event flips on signal
# ---------------------------------------------------------------------------


def test_install_signal_handlers_posix_flips_event() -> None:
    """On POSIX, SIGTERM flips the asyncio stop-event."""

    async def drive() -> bool:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        daemon_mod._install_signal_handlers(stop_event, loop)
        # Send ourselves the signal — the loop's add_signal_handler
        # converts it to a loop-thread callback that flips the event.
        os.kill(os.getpid(), signal.SIGTERM)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            return False
        return stop_event.is_set()

    ok = asyncio.run(drive())
    assert ok is True


# ---------------------------------------------------------------------------
# daemon._run — full lifecycle with a pre-set stop event
# ---------------------------------------------------------------------------


def test_daemon_run_returns_zero_on_graceful_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_run`` returns 0 when the stop-event is set, after draining any
    queued work and closing the connection cleanly."""
    db_path = tmp_path / "carl.db"
    # Seed one note so the daemon actually does work before stopping.
    seed_db = LocalDB(db_path)
    StickyQueue(seed_db).append("wake up and do work", priority=5)
    seed_db.close()

    # Pre-flip the stop event shortly after startup by monkeypatching the
    # signal-handler installer to fire immediately.
    original_install = daemon_mod._install_signal_handlers

    def _fast_install(
        event: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        original_install(event, loop)
        # Schedule a near-immediate set() so the daemon exits quickly.
        loop.call_later(0.3, event.set)

    monkeypatch.setattr(daemon_mod, "_install_signal_handlers", _fast_install)
    # Fast poll so the loop actually drains the seed note before stop.
    monkeypatch.setenv("CARL_HEARTBEAT_POLL_INTERVAL_S", "0.01")
    monkeypatch.setenv("CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S", "5")
    # Disable in-loop maintenance so this test is deterministic.
    monkeypatch.setenv("CARL_MAINTENANCE_INTERVAL_CYCLES", "0")

    rc = asyncio.run(daemon_mod._run(str(db_path)))
    assert rc == 0


def test_daemon_reclaims_stale_notes_on_boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A note wedged in ``processing`` before boot gets flipped to ``queued``."""
    db_path = tmp_path / "carl.db"
    seed_db = LocalDB(db_path)
    queue = StickyQueue(seed_db)
    wedged = queue.append("pre-crash work", priority=5)
    claimed = queue.dequeue()
    assert claimed is not None and claimed.id == wedged.id
    # Force an old ``started_at`` so the boot-time reclaim flips it.
    from datetime import datetime, timedelta, timezone

    past = datetime.now(timezone.utc) - timedelta(hours=1)
    past_str = past.strftime("%Y-%m-%dT%H:%M:%SZ")
    with seed_db._connect() as conn:  # pyright: ignore[reportPrivateUsage]
        conn.execute(
            "UPDATE sticky_notes SET started_at = ? WHERE id = ?",
            (past_str, wedged.id),
        )
        conn.commit()
    seed_db.close()

    original_install = daemon_mod._install_signal_handlers

    def _fast_install(
        event: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        original_install(event, loop)
        loop.call_later(0.6, event.set)

    monkeypatch.setattr(daemon_mod, "_install_signal_handlers", _fast_install)
    monkeypatch.setenv("CARL_HEARTBEAT_POLL_INTERVAL_S", "0.01")
    monkeypatch.setenv("CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S", "5")
    monkeypatch.setenv("CARL_MAINTENANCE_INTERVAL_CYCLES", "0")

    rc = asyncio.run(daemon_mod._run(str(db_path)))
    assert rc == 0

    # After boot-time reclaim + subsequent cycle, the note should be ``done``.
    verify_db = LocalDB(db_path)
    verify_queue = StickyQueue(verify_db)
    final = verify_queue.get(wedged.id)
    verify_db.close()
    assert final is not None
    assert final.status == "done", (
        f"expected boot-time reclaim + drain to complete the note; got "
        f"status={final.status}"
    )


def test_daemon_main_help_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``carl-heartbeat --help`` exits 0 and prints usage."""
    rc = daemon_mod.main(["--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "carl-heartbeat" in out
    assert "--db" in out


def test_daemon_resolve_poll_interval_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CARL_HEARTBEAT_POLL_INTERVAL_S", raising=False)
    assert daemon_mod._resolve_poll_interval() == 5.0


def test_daemon_resolve_poll_interval_invalid_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARL_HEARTBEAT_POLL_INTERVAL_S", "nonsense")
    assert daemon_mod._resolve_poll_interval() == 5.0


def test_daemon_resolve_poll_interval_respects_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARL_HEARTBEAT_POLL_INTERVAL_S", "0.5")
    assert daemon_mod._resolve_poll_interval() == 0.5


def test_daemon_resolve_poll_interval_rejects_nonpositive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARL_HEARTBEAT_POLL_INTERVAL_S", "0")
    assert daemon_mod._resolve_poll_interval() == 5.0
    monkeypatch.setenv("CARL_HEARTBEAT_POLL_INTERVAL_S", "-1")
    assert daemon_mod._resolve_poll_interval() == 5.0
