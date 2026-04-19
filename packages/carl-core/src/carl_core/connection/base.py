"""BaseConnection and AsyncBaseConnection — the two abstract anchors that
every CARL connection inherits from.

The split mirrors :mod:`carl_core.retry` (sync :func:`retry` vs async
:func:`async_retry`) — both bases share the FSM / stats / telemetry
plumbing; what differs is whether the lifecycle hooks are ``def`` or
``async def``. Downstream code picks whichever matches its transport:
stdio / subprocess adapters use :class:`BaseConnection`; HTTP / SSE /
WebSocket surfaces use :class:`AsyncBaseConnection`.

Design invariants
-----------------
* The state machine is driven entirely by ``_transition`` under
  ``self._lock``. Concrete subclasses never write to ``self._state``
  directly.
* ``open`` and ``close`` are idempotent-on-terminal: calling ``close()``
  on an already-closed connection is a no-op; ``open()`` on anything
  past INIT raises.
* ``close()`` never raises — errors from ``_close`` are recorded but
  swallowed so shutdown can complete.
* Telemetry emission is best-effort: if the optional
  :class:`InteractionChain` is not supplied, ``_record_event`` is a
  no-op.
* Registration with the global :class:`ConnectionRegistry` happens on
  a successful ``open`` and is undone on ``close``; the registry holds
  weak references, so early-GC is safe.
"""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator, TypeVar

from carl_core.interaction import ActionType, InteractionChain

from .coherence import ChannelCoherence
from .errors import ConnectionClosedError, ConnectionTransitionError
from .lifecycle import VALID_TRANSITIONS, ConnectionState
from .spec import ConnectionSpec, ConnectionTrust

_BC = TypeVar("_BC", bound="BaseConnection")
_ABC = TypeVar("_ABC", bound="AsyncBaseConnection")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_conn_id() -> str:
    return f"conn-{uuid.uuid4().hex[:12]}"


@dataclass
class ConnectionStats:
    """Lightweight runtime counters for a single connection."""

    opens: int = 0
    closes: int = 0
    transactions_ok: int = 0
    transactions_err: int = 0
    last_error_at: datetime | None = None


class ConnectionBase:
    """Shared FSM / stats / telemetry core.

    Not instantiable on its own — concrete users should subclass
    :class:`BaseConnection` or :class:`AsyncBaseConnection`. The split
    between sync and async is purely about which lifecycle hooks are
    coroutines; everything in this mixin is agnostic.

    Exposed as a public symbol so :class:`ConnectionRegistry` can type-hint
    against it; downstream code should not subclass this directly.
    """

    spec: ConnectionSpec  # declared as a class attribute on every subclass

    def __init__(
        self,
        *,
        chain: InteractionChain | None = None,
        connection_id: str | None = None,
    ) -> None:
        if not hasattr(type(self), "spec"):
            raise TypeError(
                f"{type(self).__name__} must declare a class-level "
                f"`spec: ConnectionSpec` attribute",
            )
        self._state: ConnectionState = ConnectionState.INIT
        self._lock = threading.RLock()
        self._connection_id: str = connection_id or _new_conn_id()
        self._opened_at: datetime | None = None
        self._closed_at: datetime | None = None
        self._last_error: BaseException | None = None
        self._stats = ConnectionStats()
        self._chain = chain
        self._channel_coherence: ChannelCoherence = ChannelCoherence.empty()

    # -- identity ---------------------------------------------------------

    @property
    def connection_id(self) -> str:
        return self._connection_id

    @property
    def state(self) -> ConnectionState:
        with self._lock:
            return self._state

    @property
    def stats(self) -> ConnectionStats:
        """Snapshot of runtime counters (returned by value — caller-safe)."""
        with self._lock:
            return ConnectionStats(
                opens=self._stats.opens,
                closes=self._stats.closes,
                transactions_ok=self._stats.transactions_ok,
                transactions_err=self._stats.transactions_err,
                last_error_at=self._stats.last_error_at,
            )

    @property
    def last_error(self) -> BaseException | None:
        with self._lock:
            return self._last_error

    # -- FSM --------------------------------------------------------------

    def _transition(self, target: ConnectionState) -> None:
        """Atomic FSM transition. Raises on disallowed target.

        Holds ``self._lock`` for the duration so concurrent callers observe
        either the pre- or post-transition state, never a torn read.
        """
        with self._lock:
            allowed = VALID_TRANSITIONS[self._state]
            if target not in allowed:
                raise ConnectionTransitionError(
                    f"invalid transition "
                    f"{self._state.value!r} -> {target.value!r}",
                    context={
                        "from": self._state.value,
                        "to": target.value,
                        "allowed": sorted(s.value for s in allowed),
                        "connection_id": self._connection_id,
                        "spec_name": self.spec.name,
                    },
                )
            self._state = target
            if target == ConnectionState.READY and self._opened_at is None:
                self._opened_at = _utcnow()
                self._stats.opens += 1
            elif target == ConnectionState.CLOSED:
                self._closed_at = _utcnow()
                self._stats.closes += 1

    def _safe_transition(self, target: ConnectionState) -> bool:
        """Transition if allowed; silently absorb invalid-transition errors.

        Used on error / cleanup paths where the FSM may already be terminal.
        Returns True if the transition succeeded.
        """
        try:
            self._transition(target)
        except ConnectionTransitionError:
            return False
        return True

    # -- telemetry / error tracking --------------------------------------

    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            self._last_error = exc
            self._stats.last_error_at = _utcnow()
            self._stats.transactions_err += 1

    def _record_event(
        self,
        name: str,
        *,
        success: bool,
        duration_ms: float | None = None,
        **ctx: Any,
    ) -> None:
        if self._chain is None:
            return
        self._chain.record(
            action=ActionType.EXTERNAL,
            name=name,
            input={
                "connection_id": self._connection_id,
                "spec": self.spec.to_dict(),
                **ctx,
            },
            success=success,
            duration_ms=duration_ms,
        )

    # -- registry integration --------------------------------------------

    def _register(self) -> None:
        # Absolute import to keep pyright happy about relative-path
        # resolution inside a function body. registry.py uses a
        # TYPE_CHECKING-only import of us, so there is no runtime cycle.
        from carl_core.connection.registry import get_registry

        get_registry().register(self)

    def _unregister(self) -> None:
        from carl_core.connection.registry import get_registry

        get_registry().unregister(self._connection_id)

    # -- guards -----------------------------------------------------------

    def require_ready(self) -> None:
        """Raise :class:`ConnectionClosedError` if the connection is not READY.

        Accepted states: only :attr:`ConnectionState.READY`. Any other
        state, including TRANSACTING, is rejected — concurrent transacts
        must use ``transact()`` which handles its own gating.
        """
        with self._lock:
            current = self._state
        if current == ConnectionState.READY:
            return
        if current in {ConnectionState.CLOSED, ConnectionState.ERROR, ConnectionState.CLOSING}:
            raise ConnectionClosedError(
                f"connection is {current.value}, not ready",
                context={
                    "connection_id": self._connection_id,
                    "spec_name": self.spec.name,
                    "state": current.value,
                    "last_error": repr(self._last_error)
                    if self._last_error
                    else None,
                },
            )
        raise ConnectionClosedError(
            f"connection is {current.value}, not ready",
            context={
                "connection_id": self._connection_id,
                "spec_name": self.spec.name,
                "state": current.value,
            },
        )

    # -- cross-channel coherence -----------------------------------------

    def channel_coherence(self) -> ChannelCoherence:
        """Return the last published :class:`ChannelCoherence` for this
        connection. Empty (all-zero) until a subclass / caller publishes.
        """
        with self._lock:
            return self._channel_coherence

    def publish_channel_coherence(self, cc: ChannelCoherence) -> None:
        """Replace the last-published coherence snapshot.

        Any subclass or external supervisor may call this to make its
        per-channel coherence observable to monitors / tests / cross-
        channel isomorphism checks.
        """
        with self._lock:
            self._channel_coherence = cc

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary of current state + stats for logs / CLI."""
        with self._lock:
            return {
                "connection_id": self._connection_id,
                "state": self._state.value,
                "spec": self.spec.to_dict(),
                "opened_at": self._opened_at.isoformat() if self._opened_at else None,
                "closed_at": self._closed_at.isoformat() if self._closed_at else None,
                "stats": {
                    "opens": self._stats.opens,
                    "closes": self._stats.closes,
                    "transactions_ok": self._stats.transactions_ok,
                    "transactions_err": self._stats.transactions_err,
                    "last_error_at": self._stats.last_error_at.isoformat()
                    if self._stats.last_error_at
                    else None,
                },
                "channel_coherence": self._channel_coherence.as_dict(),
            }


# ---------------------------------------------------------------------------
# Synchronous base
# ---------------------------------------------------------------------------


class BaseConnection(ConnectionBase, ABC):
    """Abstract base for synchronous CARL connections.

    Concrete subclasses implement :meth:`_connect` and :meth:`_close`; if
    the transport requires handshake or authentication phases, override
    :meth:`_handshake` and :meth:`_authenticate` (defaults are no-ops).

    Usage::

        class MyTrainingBackend(BaseConnection):
            spec = ConnectionSpec(...)

            def _connect(self): ...
            def _close(self): ...

        with MyTrainingBackend() as conn:
            with conn.transact("submit"):
                ...
    """

    @abstractmethod
    def _connect(self) -> None:
        """Open the underlying transport. May raise
        :class:`~carl_core.connection.errors.ConnectionUnavailableError`.
        """

    def _handshake(self) -> None:
        """Perform post-connect protocol handshake. Default: no-op."""

    def _authenticate(self) -> None:
        """Present / verify credentials. Default: no-op. Override for
        :attr:`ConnectionTrust.AUTHENTICATED` (or higher) specs.
        """

    @abstractmethod
    def _close(self) -> None:
        """Tear down the underlying transport. Must be idempotent and
        safe to call from any non-CLOSED state. Errors are recorded by
        :meth:`close` but not re-raised.
        """

    def open(self) -> None:
        """INIT -> CONNECTING [-> AUTHENTICATING] -> READY.

        Raises:
            ConnectionTransitionError: if called from any state but INIT.
            ConnectionUnavailableError / ConnectionAuthError: from hooks.
        """
        self._transition(ConnectionState.CONNECTING)
        try:
            self._connect()
            self._handshake()
            if self.spec.trust != ConnectionTrust.PUBLIC:
                self._transition(ConnectionState.AUTHENTICATING)
                self._authenticate()
            self._transition(ConnectionState.READY)
        except BaseException as exc:
            self._record_error(exc)
            self._safe_transition(ConnectionState.ERROR)
            self._record_event("connection.open", success=False, error=repr(exc))
            raise
        self._register()
        self._record_event("connection.open", success=True)

    def close(self) -> None:
        """non-terminal -> CLOSING -> CLOSED. Idempotent; does not raise."""
        with self._lock:
            current = self._state
        if current == ConnectionState.CLOSED:
            return

        if current != ConnectionState.ERROR:
            # Best-effort transition to CLOSING. A concurrent actor may have
            # already pushed us terminal, which is fine — we still run the
            # close hook and ensure we land on CLOSED.
            self._safe_transition(ConnectionState.CLOSING)

        try:
            self._close()
        except BaseException as exc:
            self._record_error(exc)
            # Do not propagate — shutdown must complete.

        self._safe_transition(ConnectionState.CLOSED)
        self._unregister()
        self._record_event("connection.close", success=True)

    # -- context managers ------------------------------------------------

    def __enter__(self: _BC) -> _BC:
        self.open()
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()

    @contextmanager
    def transact(self: _BC, op_name: str = "transact") -> Iterator[_BC]:
        """Bracket a unit of work in a READY -> TRANSACTING -> READY cycle.

        On exception the connection transitions to DEGRADED (not ERROR) so
        callers may retry or decide to :meth:`close`. Telemetry records
        success / failure and wall-clock duration.
        """
        self.require_ready()
        self._transition(ConnectionState.TRANSACTING)
        start = time.monotonic()
        success = True
        err: BaseException | None = None
        try:
            yield self
        except BaseException as exc:
            success = False
            err = exc
            self._record_error(exc)
            self._safe_transition(ConnectionState.DEGRADED)
            raise
        else:
            self._transition(ConnectionState.READY)
            with self._lock:
                self._stats.transactions_ok += 1
        finally:
            duration_ms = (time.monotonic() - start) * 1000.0
            self._record_event(
                f"connection.{op_name}",
                success=success,
                duration_ms=duration_ms,
                error=repr(err) if err else None,
            )


# ---------------------------------------------------------------------------
# Asynchronous base
# ---------------------------------------------------------------------------


class AsyncBaseConnection(ConnectionBase, ABC):
    """Abstract base for asynchronous CARL connections.

    Identical FSM / stats / telemetry semantics as :class:`BaseConnection`;
    the difference is that lifecycle hooks are coroutines and the public
    lifecycle methods are ``async``.
    """

    @abstractmethod
    async def _connect(self) -> None:  # pragma: no cover - abstract
        """Open the underlying transport. Async variant."""

    async def _handshake(self) -> None:
        """Post-connect protocol handshake. Default: no-op."""

    async def _authenticate(self) -> None:
        """Credential negotiation. Default: no-op."""

    @abstractmethod
    async def _close(self) -> None:  # pragma: no cover - abstract
        """Tear down the underlying transport. Async variant."""

    async def open(self) -> None:
        self._transition(ConnectionState.CONNECTING)
        try:
            await self._connect()
            await self._handshake()
            if self.spec.trust != ConnectionTrust.PUBLIC:
                self._transition(ConnectionState.AUTHENTICATING)
                await self._authenticate()
            self._transition(ConnectionState.READY)
        except BaseException as exc:
            self._record_error(exc)
            self._safe_transition(ConnectionState.ERROR)
            self._record_event("connection.open", success=False, error=repr(exc))
            raise
        self._register()
        self._record_event("connection.open", success=True)

    async def close(self) -> None:
        with self._lock:
            current = self._state
        if current == ConnectionState.CLOSED:
            return
        if current != ConnectionState.ERROR:
            self._safe_transition(ConnectionState.CLOSING)
        try:
            await self._close()
        except BaseException as exc:
            self._record_error(exc)
        self._safe_transition(ConnectionState.CLOSED)
        self._unregister()
        self._record_event("connection.close", success=True)

    async def __aenter__(self: _ABC) -> _ABC:
        await self.open()
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        await self.close()

    @asynccontextmanager
    async def transact(
        self: _ABC, op_name: str = "transact"
    ) -> AsyncIterator[_ABC]:
        """Async variant of :meth:`BaseConnection.transact`."""
        self.require_ready()
        self._transition(ConnectionState.TRANSACTING)
        start = time.monotonic()
        success = True
        err: BaseException | None = None
        try:
            yield self
        except BaseException as exc:
            success = False
            err = exc
            self._record_error(exc)
            self._safe_transition(ConnectionState.DEGRADED)
            raise
        else:
            self._transition(ConnectionState.READY)
            with self._lock:
                self._stats.transactions_ok += 1
        finally:
            duration_ms = (time.monotonic() - start) * 1000.0
            self._record_event(
                f"connection.{op_name}",
                success=success,
                duration_ms=duration_ms,
                error=repr(err) if err else None,
            )


__all__ = [
    "BaseConnection",
    "AsyncBaseConnection",
    "ConnectionBase",
    "ConnectionStats",
]
