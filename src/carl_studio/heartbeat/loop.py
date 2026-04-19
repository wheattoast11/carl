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
import threading
import time
from collections.abc import Callable
from typing import Any

from carl_core.interaction import ActionType, InteractionChain

from carl_studio.db import LocalDB
from carl_studio.heartbeat.phases import ORDERED_PHASES, HeartbeatPhase
from carl_studio.sticky import StickyNote, StickyQueue

StatusCallback = Callable[["str | dict[str, Any]"], None]


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
        """Main async loop: poll the queue, run a cycle, repeat."""
        while not self._stop_event.is_set():
            note = queue.dequeue()
            if note is None:
                await self._sleep_interruptible(self._poll_interval_s)
                continue
            await self._run_cycle(queue, note)

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
        try:
            for phase in ORDERED_PHASES:
                if self._stop_event.is_set():
                    result["stopped"] = True
                    break
                await self._run_phase(phase, note, result)
        except BaseException as exc:  # pragma: no cover - defensive
            success = False
            error_repr = repr(exc)
            result["error"] = error_repr
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
            # Always complete the note so the queue doesn't wedge on a
            # permanently ``processing`` row. The ``result`` dict captures
            # any error envelope for downstream inspection.
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
