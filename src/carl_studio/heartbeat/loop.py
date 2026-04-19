"""Async heartbeat loop driven by the sticky-note queue.

The loop is owned by :class:`HeartbeatConnection` but packaged as a
standalone class so tests (and alternate surfaces — e.g. a CLI
``carl heartbeat run`` command) can drive it directly without the connection
lifecycle.

Design notes
------------
* **Daemon thread**: the loop runs on its own thread so it never blocks the
  foreground chat UI or CLI process. The thread is a ``daemon`` so process
  exit tears it down cleanly even if ``stop()`` is missed (e.g. on Ctrl-C).
* **Single asyncio loop per thread**: we own the thread, so we own the
  event loop. We never schedule coroutines onto a caller's loop.
* **Cooperative stop**: :meth:`stop` flips a :class:`threading.Event` and
  ``join``s. The worker checks the event at every phase boundary and between
  queue polls so shutdown is prompt.
* **Telemetry envelope**: every cycle emits two
  :attr:`~carl_core.interaction.ActionType.HEARTBEAT_CYCLE` steps on the
  shared :class:`InteractionChain` — one with ``name="cycle:start"`` and one
  with ``name="cycle:end"``. The start step carries the note id and a short
  preview of its content; the end step carries the final result dict.
* **Status callback**: an optional ``on_status`` callable receives
  per-event strings (or dicts) so callers can surface progress in a UI
  without having to reach into the chain.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from carl_core.interaction import ActionType, InteractionChain

from carl_studio.db import LocalDB
from carl_studio.heartbeat.phases import ORDERED_PHASES, HeartbeatPhase
from carl_studio.sticky import StickyNote, StickyQueue

StatusCallback = Callable[["str | dict[str, Any]"], None]

_LOG = logging.getLogger("carl.heartbeat.loop")

# Env override for periodic maintenance cadence. A value of 0 disables the
# in-loop maintenance tick entirely — useful for tests and for deployments
# that run ``carl db maintenance`` out of band. Values < 0 are clamped to 0.
_MAINTENANCE_INTERVAL_ENV = "CARL_MAINTENANCE_INTERVAL_CYCLES"
_MAINTENANCE_SECONDS_ENV = "CARL_MAINTENANCE_INTERVAL_SECONDS"
_DEFAULT_MAINTENANCE_INTERVAL_CYCLES = 100
# Idle-daemon safety net: even when the queue has been empty for hours, run
# maintenance at least once per hour so WAL truncation and retention sweep
# still happen. Cycle-trigger alone would starve a quiet deployment.
_DEFAULT_MAINTENANCE_INTERVAL_SECONDS = 3600.0


def _noop_status(_msg: str | dict[str, Any]) -> None:
    """Default :class:`StatusCallback` — swallows every message."""


class HeartbeatLoop:
    """Daemon-thread async worker that drains a :class:`StickyQueue`.

    Parameters
    ----------
    queue
        The sticky-note queue to drain.
    chain
        Shared interaction chain for telemetry. The loop appends
        :attr:`ActionType.HEARTBEAT_CYCLE` steps on start and end of every
        cycle.
    on_status
        Optional callable invoked with a short status string on every
        phase transition and cycle boundary. Runs on the worker thread,
        so must be non-blocking; the default is a no-op.
    poll_interval_s
        Sleep between queue polls when the queue is empty. Keep small
        enough to stay responsive but large enough to avoid a tight loop.
        Tests pass ``0.01``.

    Raises
    ------
    ValueError
        If ``poll_interval_s`` is not a positive finite number.
    """

    def __init__(
        self,
        queue: StickyQueue,
        chain: InteractionChain,
        *,
        on_status: StatusCallback | None = None,
        poll_interval_s: float = 5.0,
        maintenance_interval_cycles: int | None = None,
    ) -> None:
        if poll_interval_s != poll_interval_s or poll_interval_s <= 0.0:
            # NaN check uses self-inequality; float('inf') technically
            # passes the positivity test but would hang the loop, so we
            # also reject infinities below.
            raise ValueError(
                "HeartbeatLoop: poll_interval_s must be a positive finite number",
            )
        if poll_interval_s == float("inf"):
            raise ValueError(
                "HeartbeatLoop: poll_interval_s must be a positive finite number",
            )
        # SQLite connections cached on :class:`LocalDB` are main-thread
        # only (``check_same_thread=True``). The worker runs on a daemon
        # thread, so we remember the DB path and rebuild the queue handle
        # inside the worker. The public constructor still accepts the
        # caller's :class:`StickyQueue` because it's the ergonomic surface —
        # we only dip into ``_db.path`` for rebinding.
        self._queue = queue
        self._db_path = queue._db.path  # pyright: ignore[reportPrivateUsage]
        self._chain = chain
        self._on_status: StatusCallback = on_status or _noop_status
        self._poll_interval_s = float(poll_interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._maintenance_interval_cycles = self._resolve_maintenance_interval(
            maintenance_interval_cycles,
        )
        self._maintenance_interval_seconds = self._resolve_maintenance_seconds()

    @staticmethod
    def _resolve_maintenance_interval(explicit: int | None) -> int:
        """Resolve the maintenance cadence from arg → env → default.

        Negative/zero values disable the in-loop maintenance tick — callers
        that want to run maintenance out of band (``carl db maintenance``)
        should set ``CARL_MAINTENANCE_INTERVAL_CYCLES=0``.
        """
        if explicit is not None:
            return max(0, int(explicit))
        raw = os.environ.get(_MAINTENANCE_INTERVAL_ENV, "").strip()
        if raw:
            try:
                return max(0, int(raw))
            except ValueError:
                _LOG.warning(
                    "invalid %s=%r; falling back to default %d",
                    _MAINTENANCE_INTERVAL_ENV,
                    raw,
                    _DEFAULT_MAINTENANCE_INTERVAL_CYCLES,
                )
        return _DEFAULT_MAINTENANCE_INTERVAL_CYCLES

    @staticmethod
    def _resolve_maintenance_seconds() -> float:
        """Resolve the time-based maintenance cadence from env → default."""
        raw = os.environ.get(_MAINTENANCE_SECONDS_ENV, "").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                _LOG.warning(
                    "invalid %s=%r; falling back to default %.0fs",
                    _MAINTENANCE_SECONDS_ENV,
                    raw,
                    _DEFAULT_MAINTENANCE_INTERVAL_SECONDS,
                )
        return _DEFAULT_MAINTENANCE_INTERVAL_SECONDS

    # -- public lifecycle -------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while the worker thread is alive."""
        t = self._thread
        return t is not None and t.is_alive()

    def start(self) -> threading.Thread:
        """Spawn the worker thread. Idempotent — returns the existing thread
        if one is already running, so repeated ``start()`` calls do not
        create stacked daemons.
        """
        with self._start_lock:
            existing = self._thread
            if existing is not None and existing.is_alive():
                return existing
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._runner,
                name="carl-heartbeat",
                daemon=True,
            )
            thread.start()
            self._thread = thread
            return thread

    def stop(self, *, timeout: float = 3.0) -> None:
        """Signal the worker to exit and ``join`` it.

        ``timeout`` is passed to :meth:`threading.Thread.join` — callers
        on a tight shutdown budget can pass ``0.0`` to return immediately
        without waiting. Never raises.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        if thread.is_alive():
            thread.join(timeout=timeout)

    # -- internals --------------------------------------------------------

    def _runner(self) -> None:
        """Thread entry point. Owns its own asyncio event loop.

        Rebuilds a :class:`StickyQueue` on a fresh :class:`LocalDB` so the
        worker thread owns its own SQLite connection — the caller's handle
        stays bound to its creating thread (SQLite's
        ``check_same_thread=True`` default).
        """
        local_db: LocalDB | None = None
        try:
            local_db = LocalDB(self._db_path)
            thread_queue = StickyQueue(local_db)
            asyncio.run(self._aloop(thread_queue))
        except BaseException as exc:  # pragma: no cover - defensive
            # Funnel any unhandled error through the status callback so the
            # daemon does not die silently. We deliberately do not re-raise:
            # the worker thread has no one to propagate to.
            try:
                self._on_status({"event": "heartbeat.error", "error": repr(exc)})
            except BaseException:
                pass
        finally:
            if local_db is not None:
                try:
                    local_db.close()
                except BaseException:  # pragma: no cover - defensive
                    pass

    async def _aloop(self, queue: StickyQueue) -> None:
        """Main async loop: poll the queue, run a cycle, repeat.

        The stop-event is only checked **between** cycles so a cycle that
        has begun always runs to completion on graceful shutdown. The
        phase loop inside :meth:`_run_cycle` no longer short-circuits on
        ``stop`` — that used to leave the note half-processed, which
        defeats the point of graceful shutdown.

        Maintenance fires on whichever trigger hits first: cycle count OR
        wall-clock time. The time-based trigger guarantees idle daemons
        still truncate WAL and run retention — the cycle trigger alone
        would starve a quiet deployment indefinitely.
        """
        cycles_since_maintenance = 0
        last_maintenance_monotonic = time.monotonic()
        while not self._stop_event.is_set():
            note = queue.dequeue()
            if note is None:
                await self._sleep_interruptible(self._poll_interval_s)
            else:
                await self._run_cycle(queue, note)
                cycles_since_maintenance += 1

            elapsed = time.monotonic() - last_maintenance_monotonic
            cycle_trigger = (
                self._maintenance_interval_cycles > 0
                and cycles_since_maintenance >= self._maintenance_interval_cycles
            )
            time_trigger = elapsed >= self._maintenance_interval_seconds
            if cycle_trigger or time_trigger:
                cycles_since_maintenance = 0
                last_maintenance_monotonic = time.monotonic()
                self._run_maintenance(queue)

    def _run_maintenance(self, queue: StickyQueue) -> None:
        """Run a maintenance tick — retention sweep + WAL checkpoint + reclaim.

        Called every ``maintenance_interval_cycles`` completed cycles inside
        the worker thread. Failures are logged and surfaced via
        :attr:`_on_status` but never propagate — maintenance is best-effort,
        the loop must keep running.
        """
        # Retention days come from the env so operators can lengthen or
        # shorten retention without editing code. Invalid/missing values
        # fall back to 30.
        raw = os.environ.get("CARL_STICKY_RETENTION_DAYS", "").strip()
        try:
            retention_days = int(raw) if raw else 30
        except ValueError:
            retention_days = 30
        if retention_days < 0:
            retention_days = 0
        try:
            reclaimed = queue.reclaim_stale()
            stats = queue.maintenance(retention_days=retention_days)
            payload: dict[str, Any] = {
                "event": "heartbeat.maintenance",
                "reclaimed": reclaimed,
                "notes_deleted": stats["notes_deleted"],
                "wal_checkpoint": stats["wal_checkpoint"],
            }
            self._on_status(payload)
            _LOG.info(
                "heartbeat maintenance: reclaimed=%s deleted=%s",
                reclaimed,
                stats["notes_deleted"],
            )
        except BaseException as exc:  # pragma: no cover - defensive
            _LOG.exception("heartbeat maintenance failed")
            try:
                self._on_status(
                    {"event": "heartbeat.maintenance_failed", "error": repr(exc)},
                )
            except BaseException:
                pass

    async def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep in small increments so a :meth:`stop` signal is honoured
        within at most ``0.05s`` regardless of the poll interval.
        """
        # If callers pass a tiny interval (tests), one await is enough.
        if seconds <= 0.05:
            await asyncio.sleep(seconds)
            return
        remaining = seconds
        step = 0.05
        while remaining > 0.0 and not self._stop_event.is_set():
            await asyncio.sleep(min(step, remaining))
            remaining -= step

    async def _run_cycle(self, queue: StickyQueue, note: StickyNote) -> None:
        """Run the full ordered-phases pipeline for ``note``.

        Emits ``cycle:start``/``cycle:end`` telemetry, calls :meth:`_run_phase`
        per phase, and marks the note ``done`` via
        :meth:`StickyQueue.complete`. ``queue`` is the per-thread handle
        constructed in :meth:`_runner`.

        On a successful phase loop (no exception), the note transitions to
        ``done``. On an exception mid-cycle the note is flipped back to
        ``queued`` via :meth:`StickyQueue.requeue` so a retry picks it up
        rather than leaving it wedged in ``processing`` forever. This
        discipline is the foundation of the graceful-shutdown contract —
        :meth:`_aloop` no longer breaks mid-cycle on ``stop_event`` so a
        running cycle always finishes, one way or the other.
        """
        t_start = time.monotonic()
        preview = note.content[:120] if note.content else ""

        self._chain.record(
            ActionType.HEARTBEAT_CYCLE,
            "cycle:start",
            input={"note_id": note.id, "content": preview},
            output={},
            success=True,
            duration_ms=0.0,
        )
        self._on_status(f"[heartbeat] cycle:start {note.id}")

        result: dict[str, Any] = {"phases": [], "note_id": note.id}
        success = True
        error_repr: str | None = None
        # ``cancelled`` signals the note should flip back to ``queued`` on
        # exit rather than ``done``. Set by the exception branch below.
        requeue_on_exit = False
        try:
            for phase in ORDERED_PHASES:
                # NOTE: intentionally no ``stop_event.is_set()`` check here.
                # Graceful shutdown is handled **between** cycles in
                # :meth:`_aloop`; once a cycle has begun we run it to
                # completion. Interrupting mid-phase left the note wedged
                # under the previous design.
                await self._run_phase(phase, note, result)
        except BaseException as exc:
            success = False
            error_repr = repr(exc)
            result["error"] = error_repr
            requeue_on_exit = True
        finally:
            elapsed_ms = (time.monotonic() - t_start) * 1000.0
            self._chain.record(
                ActionType.HEARTBEAT_CYCLE,
                "cycle:end",
                input={"note_id": note.id},
                output=result,
                success=success,
                duration_ms=elapsed_ms,
            )
            if requeue_on_exit:
                # Exception path — flip back to ``queued`` for retry. We
                # deliberately do **not** call :meth:`complete` here; the
                # note is not done. :meth:`StickyQueue.requeue` is a no-op
                # if the row has already moved (e.g. an operator archived
                # it out of band), which is the safe default.
                try:
                    queue.requeue(note.id)
                except BaseException as exc:  # pragma: no cover - defensive
                    self._on_status(
                        {
                            "event": "heartbeat.requeue_failed",
                            "note_id": note.id,
                            "error": repr(exc),
                        },
                    )
            else:
                # Happy path — record the result and transition to ``done``.
                try:
                    queue.complete(note.id, result)
                except BaseException as exc:  # pragma: no cover - defensive
                    self._on_status(
                        {
                            "event": "heartbeat.complete_failed",
                            "note_id": note.id,
                            "error": repr(exc),
                        },
                    )
            self._on_status(f"[heartbeat] cycle:end {note.id}")
            # The ``error_repr`` local is preserved for log correlation even
            # when the caller doesn't subscribe to status — no-op assignment
            # keeps the symbol referenced so static analysis does not flag
            # it as unused on the happy path.
            _ = error_repr

    async def _run_phase(
        self,
        phase: HeartbeatPhase,
        note: StickyNote,
        acc: dict[str, Any],
    ) -> None:
        """Per-phase hook. Default is a logged no-op sleep.

        Subclasses or composition layers replace this with real behaviour
        (LLM calls, sandboxed tools, etc.). We keep a small ``asyncio.sleep``
        here so the loop yields control under test and real workloads.
        """
        self._on_status(f"[heartbeat] {phase.value} {note.id}")
        await asyncio.sleep(0.001)
        # The accumulator is created in :meth:`_run_cycle` with a
        # ``list[str]`` under ``phases``; ``setdefault`` just re-hydrates
        # the same list if a subclass forgot to pre-seed it.
        phases_list: list[str] = acc.setdefault("phases", [])
        phases_list.append(phase.value)


__all__ = ["HeartbeatLoop", "StatusCallback"]
