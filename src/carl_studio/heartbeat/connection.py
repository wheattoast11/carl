"""HeartbeatConnection — :class:`AsyncBaseConnection` adapter for the loop.

The adapter exists so the heartbeat participates in the global
:class:`~carl_core.connection.ConnectionRegistry` and inherits the standard
lifecycle FSM (``INIT → CONNECTING → READY → CLOSING → CLOSED``). That
means ``carl connections list`` surfaces the heartbeat alongside every
other CARL channel, and an error during startup is recorded in
:attr:`ConnectionStats.last_error_at` just like any other connection.

The loop itself — the thing that does real work — lives in
:mod:`carl_studio.heartbeat.loop`. This adapter is intentionally thin: it
constructs the loop, starts it on ``_connect``, stops it on ``_close``, and
exposes a small :meth:`drain_status` helper for UIs that want to surface
phase-transition strings between chat turns.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from carl_core.connection.base import AsyncBaseConnection
from carl_core.connection.spec import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
)
from carl_core.interaction import InteractionChain

from carl_studio.envutil import env_float
from carl_studio.heartbeat.loop import HeartbeatLoop
from carl_studio.sticky import StickyQueue

# The parse-warn-fallback chain for this env var lives in
# :func:`carl_studio.envutil.env_float`; no module-level logger is needed here.
_SHUTDOWN_TIMEOUT_ENV = "CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S"
_DEFAULT_SHUTDOWN_TIMEOUT_S = 30.0

_SPEC = ConnectionSpec(
    name="carl.heartbeat",
    scope=ConnectionScope.ONE_P,
    kind=ConnectionKind.UTILITY,
    direction=ConnectionDirection.BIDIRECTIONAL,
    transport=ConnectionTransport.IN_PROCESS,
    trust=ConnectionTrust.PUBLIC,
    metadata={"subsystem": "heartbeat"},
)


class HeartbeatConnection(AsyncBaseConnection):
    """Lifecycle-managed adapter over :class:`HeartbeatLoop`.

    Parameters
    ----------
    queue
        The :class:`StickyQueue` to drain.
    chain
        Optional :class:`InteractionChain` for telemetry. If omitted, a
        fresh chain is created so the loop always has somewhere to record
        ``HEARTBEAT_CYCLE`` steps — callers that want to share telemetry
        with the outer agent should always pass their own chain.
    poll_interval_s
        Forwarded to :class:`HeartbeatLoop`. Default ``5.0`` (production);
        tests pass a small value like ``0.01``.
    status_buffer
        Maximum number of recent status messages to retain in memory for
        :meth:`drain_status`. Default ``50``.
    connection_id
        Optional override for :attr:`connection_id`. If omitted, a
        registry-unique id is generated.
    """

    spec = _SPEC

    def __init__(
        self,
        queue: StickyQueue,
        chain: InteractionChain | None = None,
        *,
        poll_interval_s: float = 5.0,
        status_buffer: int = 50,
        connection_id: str | None = None,
        shutdown_timeout_s: float | None = None,
    ) -> None:
        if status_buffer <= 0:
            raise ValueError(
                "HeartbeatConnection: status_buffer must be a positive int",
            )
        effective_chain = chain if chain is not None else InteractionChain()
        super().__init__(chain=effective_chain, connection_id=connection_id)
        self._queue = queue
        self._status: deque[str | dict[str, Any]] = deque(maxlen=status_buffer)
        # We own the chain we pass to the loop — the lifecycle recorder on
        # ConnectionBase may also append events onto the same chain via
        # _record_event. That is the intended design: the chain is the
        # single source of truth for everything this connection did.
        assert self._chain is not None  # guaranteed by the `effective_chain` above
        self._loop = HeartbeatLoop(
            queue,
            self._chain,
            on_status=self._status.append,
            poll_interval_s=poll_interval_s,
        )
        # Shutdown budget: arg → env → default. ``env_float`` with
        # ``minimum=0.0`` enforces "clamp negatives" while keeping the
        # caller site a single line. The default 30s matches what most
        # container orchestrators grant between SIGTERM and SIGKILL so
        # the in-flight cycle has a realistic window to finish.
        self._shutdown_timeout_s = env_float(
            _SHUTDOWN_TIMEOUT_ENV,
            default=_DEFAULT_SHUTDOWN_TIMEOUT_S,
            explicit=shutdown_timeout_s,
            minimum=0.0,
        )

    # -- lifecycle hooks (AsyncBaseConnection contract) -----------------

    async def _connect(self) -> None:
        """Start the worker thread. Idempotent via :meth:`HeartbeatLoop.start`."""
        self._loop.start()

    async def _close(self) -> None:
        """Stop the worker thread using the configured shutdown budget.

        Honours ``CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S`` (default 30s). The
        in-flight cycle has up to the budget to finish before the join
        returns. The worker thread will still exit once the event loop
        notices ``_stop_event`` — this just bounds how long ``close()``
        itself blocks.
        """
        self._loop.stop(timeout=self._shutdown_timeout_s)

    # -- adapter surface --------------------------------------------------

    @property
    def queue(self) -> StickyQueue:
        """The sticky queue being drained. Exposed for tests and CLI surfaces."""
        return self._queue

    @property
    def loop(self) -> HeartbeatLoop:
        """The underlying :class:`HeartbeatLoop`. Primarily for tests."""
        return self._loop

    def drain_status(self) -> list[str | dict[str, Any]]:
        """Return and clear buffered status messages.

        Non-blocking. Intended to be called between chat turns so the UI
        can render phase transitions without having to subscribe to a
        streaming channel.
        """
        out: list[str | dict[str, Any]] = list(self._status)
        self._status.clear()
        return out


__all__ = ["HeartbeatConnection"]
