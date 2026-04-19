"""Heartbeat daemon entrypoint — 24/7 worker with graceful SIGTERM handling.

``python -m carl_studio.heartbeat.daemon`` (or the console script
``carl-heartbeat`` registered via ``pyproject.toml``) boots a long-lived
worker that:

1. Reclaims any ``sticky_notes`` rows left in ``processing`` by a prior
   crash — this is ticket **E2** on the hardening spec. Without the
   reclaim, a SIGKILL during :class:`~carl_studio.heartbeat.loop.HeartbeatLoop._run_cycle`
   wedges the note forever because the row is already in ``processing``
   but nobody owns it.
2. Opens the :class:`HeartbeatConnection` (which starts the loop in its
   own daemon thread — see :mod:`carl_studio.heartbeat.loop`).
3. Waits on an ``asyncio.Event`` that SIGINT / SIGTERM trip. We use
   :meth:`asyncio.AbstractEventLoop.add_signal_handler` where supported
   (POSIX) and fall back to :func:`signal.signal` on platforms that do
   not (Windows dev boxes). Both paths flip the same event, so shutdown
   is uniform.
4. Closes the connection — which joins the worker thread using the
   configured shutdown budget (``CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S``,
   default 30s). The in-flight cycle runs to completion; no half-
   processed note is left behind.

Environment variables honoured:

``CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S``
    Seconds the close step waits for the in-flight cycle. Default 30.
``CARL_MAINTENANCE_INTERVAL_CYCLES``
    How many completed cycles between in-loop maintenance ticks. Default
    100. Zero disables the in-loop tick (run ``carl db maintenance`` out
    of band).
``CARL_STICKY_RETENTION_DAYS``
    Days to keep ``archived`` sticky notes before the retention sweep
    deletes them. Default 30.
``CARL_HEARTBEAT_POLL_INTERVAL_S``
    Seconds between queue polls when the queue is empty. Default 5.
``CARL_LOG_LEVEL`` / ``CARL_LOG_JSON``
    Forwarded to :func:`carl_studio.logging_config.configure_logging`.

Exit codes
----------
* ``0`` — graceful shutdown (SIGTERM / SIGINT received).
* ``1`` — startup failure (DB unusable, signal handler install failed,
  etc.). The exception is logged before the process exits.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from carl_studio.db import LocalDB
from carl_studio.heartbeat.connection import HeartbeatConnection
from carl_studio.logging_config import configure_logging
from carl_studio.sticky import StickyQueue

_LOG = logging.getLogger("carl.heartbeat.daemon")

_DEFAULT_POLL_INTERVAL_S = 5.0
_POLL_INTERVAL_ENV = "CARL_HEARTBEAT_POLL_INTERVAL_S"


def _resolve_poll_interval() -> float:
    """Resolve the loop poll interval from env with a safe default.

    Invalid values fall back to the default — the daemon must not crash
    on a mis-typed env var. Non-positive values are also rejected because
    :class:`HeartbeatLoop` requires ``poll_interval_s > 0``.
    """
    raw = os.environ.get(_POLL_INTERVAL_ENV, "").strip()
    if not raw:
        return _DEFAULT_POLL_INTERVAL_S
    try:
        val = float(raw)
    except ValueError:
        _LOG.warning(
            "invalid %s=%r; falling back to default %.1fs",
            _POLL_INTERVAL_ENV,
            raw,
            _DEFAULT_POLL_INTERVAL_S,
        )
        return _DEFAULT_POLL_INTERVAL_S
    if val <= 0.0 or val != val:  # NaN check via self-inequality
        _LOG.warning(
            "invalid %s=%r (must be > 0); falling back to default %.1fs",
            _POLL_INTERVAL_ENV,
            raw,
            _DEFAULT_POLL_INTERVAL_S,
        )
        return _DEFAULT_POLL_INTERVAL_S
    return val


def _install_signal_handlers(
    stop_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Install SIGINT / SIGTERM handlers that flip ``stop_event``.

    POSIX path uses :meth:`loop.add_signal_handler` — integrates cleanly
    with the running event loop. Windows/Jupyter (where that method
    raises :class:`NotImplementedError`) falls back to the process-level
    :func:`signal.signal`. In the fallback path we schedule the
    ``set()`` onto the loop from the signal handler thread so the
    ``asyncio.Event`` is always flipped on the loop's own thread.
    """

    def _posix_set() -> None:
        _LOG.info("heartbeat daemon received signal, initiating shutdown")
        stop_event.set()

    def _fallback_set(_signum: int, _frame: Any) -> None:
        # Signal handlers run on the main thread but do not run on the
        # event loop's thread on Windows; ``call_soon_threadsafe`` guards
        # us either way.
        _LOG.info("heartbeat daemon received signal, initiating shutdown")
        loop.call_soon_threadsafe(stop_event.set)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _posix_set)
        except (NotImplementedError, RuntimeError):
            # ``NotImplementedError`` on Windows; ``RuntimeError`` if the
            # loop is already running with a different policy. Fall back
            # to the process-level handler in both cases.
            signal.signal(sig, _fallback_set)


async def _run(db_path: str | Path | None) -> int:
    """Main async driver — open the connection, wait for a signal, close."""
    configure_logging()
    try:
        db = LocalDB(db_path) if db_path else LocalDB()
    except Exception:
        _LOG.exception("heartbeat daemon failed to open LocalDB")
        return 1

    # E2 — reclaim rows wedged by a prior crash **before** the loop
    # starts dequeuing. If we didn't, the crash-survivors would sit in
    # ``processing`` forever while the loop drained fresh ``queued`` work
    # past them.
    boot_queue = StickyQueue(db)
    try:
        reclaimed = boot_queue.reclaim_stale()
    except Exception:
        _LOG.exception("heartbeat daemon boot-time reclaim failed")
        reclaimed = 0
    if reclaimed:
        _LOG.info(
            "heartbeat daemon reclaimed %d stale note(s) on boot",
            reclaimed,
        )
    else:
        _LOG.info("heartbeat daemon boot: no stale notes to reclaim")

    queue = StickyQueue(db)
    poll_interval_s = _resolve_poll_interval()
    conn = HeartbeatConnection(queue, poll_interval_s=poll_interval_s)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(stop_event, loop)

    try:
        await conn.open()
    except Exception:
        _LOG.exception("heartbeat daemon failed to open HeartbeatConnection")
        try:
            db.close()
        except Exception:
            pass
        return 1

    _LOG.info("heartbeat daemon ready; waiting for SIGTERM/SIGINT")
    try:
        await stop_event.wait()
    finally:
        _LOG.info("heartbeat daemon closing connection")
        try:
            await conn.close()
        except Exception:
            _LOG.exception("heartbeat daemon error during close()")
        try:
            db.close()
        except Exception:
            _LOG.exception("heartbeat daemon error closing LocalDB")
    _LOG.info("heartbeat daemon exited cleanly")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. ``argv`` is accepted for symmetry; currently unused.

    The one CLI input we honour is the ``--db`` flag — optional, defaults
    to ``~/.carl/carl.db`` through :class:`LocalDB`. We parse it manually
    to avoid a dependency on Typer from this process-level entrypoint (the
    goal is the smallest possible import graph at daemon start).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    db_path: str | None = None
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--db" and i + 1 < len(args):
            db_path = args[i + 1]
            i += 2
            continue
        if tok.startswith("--db="):
            db_path = tok.split("=", 1)[1]
            i += 1
            continue
        if tok in ("-h", "--help"):
            _print_help()
            return 0
        # Unknown args — refuse silently rather than crash; this keeps
        # systemd unit files forgiving.
        i += 1
    try:
        return asyncio.run(_run(db_path))
    except KeyboardInterrupt:
        # Already handled via signal handler in practice, but if the
        # handler hasn't landed yet (e.g. Ctrl-C during startup) the
        # asyncio.run() raises KeyboardInterrupt — treat that as a
        # clean exit, not a crash.
        return 0


def _print_help() -> None:
    """Emit a short help screen. Kept stdlib-only; no Typer."""
    sys.stdout.write(
        "carl-heartbeat — background worker daemon\n"
        "\n"
        "Usage: carl-heartbeat [--db PATH]\n"
        "\n"
        "Options:\n"
        "  --db PATH    Override the LocalDB path (default: ~/.carl/carl.db)\n"
        "  -h, --help   Show this help and exit\n"
        "\n"
        "Environment:\n"
        "  CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S  (default 30)\n"
        "  CARL_HEARTBEAT_POLL_INTERVAL_S     (default 5)\n"
        "  CARL_MAINTENANCE_INTERVAL_CYCLES   (default 100, 0 disables)\n"
        "  CARL_STICKY_RETENTION_DAYS         (default 30)\n"
        "  CARL_LOG_LEVEL / CARL_LOG_JSON     (see carl_studio.logging_config)\n",
    )


__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
