"""EnvironmentConnection — bridges carl-core's connection primitive to the
TRL-facing BaseEnvironment contract.

Design decision: composition, not multiple inheritance.
-------------------------------------------------------
``BaseEnvironment`` declares ``spec: EnvironmentSpec`` as a class attribute —
the registry (``register_environment``) and validator (``validate_environment``)
both enforce that ``cls.spec`` **is** an :class:`EnvironmentSpec` via
``isinstance`` / ``hasattr`` checks. TRL's environment_factory also reads
``cls.spec`` to discover tools, max_turns, dataset columns.

``BaseConnection`` also declares ``spec: ConnectionSpec``, and its runtime
machinery (``_record_event``, ``_transition`` error context, ``to_dict``) calls
``self.spec.to_dict()`` and ``self.spec.name`` assuming the value is a
``ConnectionSpec``. ``EnvironmentSpec`` has no ``.to_dict()`` method.

Multiple inheritance with a single ``spec`` attribute can only satisfy one
contract. A property-based workaround that swaps types based on access path is
fragile, breaks ``isinstance`` checks in the registry, and would require
invasive monkey-patching of BaseConnection's internal calls.

Composition keeps each primitive honest:
  * ``spec: EnvironmentSpec`` — the public class attribute. TRL, the registry,
    and the validator see what they expect.
  * ``connection_spec: ConnectionSpec`` — declarative ConnectionSpec owned by
    the EnvironmentConnection subclass.
  * An internal ``_EnvConnectionAdapter(BaseConnection)`` owns the FSM, stats,
    telemetry emission, and the lifecycle lock. Lifecycle hooks
    (``_connect`` / ``_close``) delegate back to the environment's own hooks.

Concrete env classes (CodingSandboxEnv, SQLSandboxEnv, ...) subclass
EnvironmentConnection. They still declare ``spec = EnvironmentSpec(...)`` for
TRL. They may override ``connection_spec`` to refine the name / scope / trust.

The zero-arg ``__init__`` contract is preserved — TRL's auto-discovery path
calls ``cls()`` with no args; we build the internal adapter lazily inside
``__init__``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from carl_core.connection import (
    BaseConnection,
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionState,
    ConnectionStats,
    ConnectionTransport,
    ConnectionTrust,
)
from carl_core.interaction import InteractionChain

from carl_studio.environments.protocol import BaseEnvironment

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


# ---------------------------------------------------------------------------
# Default ConnectionSpec for in-process builtin environments.
# ---------------------------------------------------------------------------

DEFAULT_ENVIRONMENT_CONNECTION_SPEC: ConnectionSpec = ConnectionSpec(
    name="carl.env.default",
    scope=ConnectionScope.ONE_P,
    kind=ConnectionKind.ENVIRONMENT,
    direction=ConnectionDirection.BIDIRECTIONAL,
    transport=ConnectionTransport.IN_PROCESS,
    trust=ConnectionTrust.PUBLIC,
)


class _EnvConnectionAdapter(BaseConnection):
    """Private BaseConnection adapter owned by an EnvironmentConnection.

    Delegates its two abstract lifecycle hooks back to the owning environment
    instance so concrete envs override ``_env_connect`` / ``_env_close`` on
    the public class without having to know about the internal adapter.
    """

    # Class-level placeholder — each instance overrides with the owner's
    # concrete connection_spec in __init__. BaseConnection's __init__ only
    # checks ``hasattr(type(self), 'spec')``, so this satisfies the gate
    # and the per-instance override via ``object.__setattr__`` is what the
    # FSM actually reads.
    spec: ConnectionSpec = DEFAULT_ENVIRONMENT_CONNECTION_SPEC

    def __init__(
        self,
        owner: EnvironmentConnection,
        *,
        connection_spec: ConnectionSpec,
        chain: InteractionChain | None = None,
        connection_id: str | None = None,
    ) -> None:
        # Instance-level override of the class-level sentinel. BaseConnection
        # reads self.spec (attribute lookup), which prefers the instance dict.
        self.spec = connection_spec
        super().__init__(chain=chain, connection_id=connection_id)
        self._owner = owner

    def _connect(self) -> None:
        # Friend-class access: the adapter exists to drive lifecycle hooks
        # on its owning environment. The owner's ``_env_connect`` is an
        # override point for subclasses, private-prefixed to keep it off
        # TRL's public tool discovery surface.
        self._owner._env_connect()  # pyright: ignore[reportPrivateUsage]

    def _close(self) -> None:
        self._owner._env_close()  # pyright: ignore[reportPrivateUsage]


class EnvironmentConnection(BaseEnvironment):
    """A CARL environment that also participates in the connection lifecycle.

    Inherits ``BaseEnvironment`` so the registry, validator, and TRL's
    environment_factory continue to work unchanged. Composes an internal
    :class:`_EnvConnectionAdapter` (a ``BaseConnection`` subclass) for the
    FSM / stats / telemetry plumbing defined by ``carl_core.connection``.

    Class attributes
    ----------------
    spec
        Declared by concrete subclasses as an :class:`EnvironmentSpec`.
        Inherited unchanged from :class:`BaseEnvironment`.
    connection_spec
        Declarative :class:`ConnectionSpec` for this env's connection. Default
        is an in-process, public-trust, bidirectional env connection; override
        per-env for more specific names / endpoints.

    Notes
    -----
    * ``__init__`` still takes **no arguments** — TRL's environment_factory
      auto-discovers envs and calls ``cls()`` with zero args. Chain injection
      for telemetry happens post-construction via :meth:`attach_chain`.
    * ``_env_connect`` and ``_env_close`` are no-ops by default; override for
      transport that needs setup/teardown.
    * Tool methods should stay synchronous and exception-free at the boundary
      (return error strings, as the existing builtins do) — the connection
      FSM only moves to DEGRADED if ``transact`` catches an exception.
    """

    connection_spec: ConnectionSpec = DEFAULT_ENVIRONMENT_CONNECTION_SPEC

    def __init__(self) -> None:
        # Zero-arg init — mandatory for TRL auto-discovery.
        super().__init__()
        # One lock guards both the environment and the adapter; the adapter
        # has its own RLock internally, so this one is only for composition-
        # level invariants (e.g. initialization ordering).
        self._conn_lock = threading.RLock()
        # Attribute name chosen to avoid collisions with subclasses that
        # already use ``self._conn`` for their own transport (e.g. the SQL
        # sandbox's sqlite3.Connection).
        self._connection: _EnvConnectionAdapter = _EnvConnectionAdapter(
            self,
            connection_spec=type(self).connection_spec,
            chain=None,
            connection_id=None,
        )

    # -- chain / telemetry wiring ----------------------------------------

    def attach_chain(self, chain: InteractionChain | None) -> None:
        """Attach or replace the :class:`InteractionChain` used for telemetry.

        Must be called before :meth:`open` for ``connection.open`` to land on
        the chain. Passing ``None`` disables telemetry.
        """
        with self._conn_lock:
            # BaseConnection stores the chain as a protected instance field;
            # there is no public setter. The adapter is owned by this class
            # so direct rebinding is the intended composition pattern.
            self._connection._chain = chain  # pyright: ignore[reportPrivateUsage]

    @property
    def chain(self) -> InteractionChain | None:
        """The active InteractionChain, if any."""
        return self._connection._chain  # pyright: ignore[reportPrivateUsage]

    # -- connection surface (delegation to the internal adapter) ---------

    def _env_connect(self) -> None:
        """Override to set up any transport-level state. Default: no-op."""

    def _env_close(self) -> None:
        """Override to tear down transport-level state. Default: no-op.

        Must be idempotent and safe to call from any state.
        """

    def open(self) -> None:
        """Open the connection. Delegates to the internal adapter."""
        self._connection.open()

    def close(self) -> None:
        """Close the connection. Idempotent; never raises."""
        self._connection.close()

    @property
    def state(self) -> ConnectionState:
        return self._connection.state

    @property
    def stats(self) -> ConnectionStats:
        return self._connection.stats

    @property
    def connection_id(self) -> str:
        return self._connection.connection_id

    def to_dict(self) -> dict[str, Any]:
        """Merged summary: connection FSM state + environment spec."""
        payload = self._connection.to_dict()
        payload["environment_spec"] = {
            "lane": self.spec.lane.value,
            "name": self.spec.name,
            "tools": list(self.spec.tools),
            "max_turns": self.spec.max_turns,
            "reward_type": self.spec.reward_type,
            "multimodal": self.spec.multimodal,
            "dataset_columns": list(self.spec.dataset_columns),
        }
        return payload

    def require_ready(self) -> None:
        """Raise if the connection is not in the READY state."""
        self._connection.require_ready()

    def transact(self, op_name: str = "transact") -> AbstractContextManager[BaseConnection]:
        """Context manager wrapping a READY -> TRANSACTING -> READY cycle.

        Returns the underlying ``BaseConnection`` so callers can chain
        telemetry or guards off the adapter if needed. Errors inside the
        ``with`` block move the FSM to DEGRADED and re-raise.
        """
        return self._connection.transact(op_name)

    def __enter__(self) -> EnvironmentConnection:
        self.open()
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()

    # -- BaseEnvironment overrides --------------------------------------

    def reset(self, **kwargs: Any) -> str | None:
        """Reset the env and record a ``connection.reset`` telemetry event.

        Subclasses still override this; call ``super().reset(**kwargs)`` as
        with the plain :class:`BaseEnvironment`. This wrapper does NOT drive
        the FSM — it only emits telemetry so ``reset`` can be called at any
        lifecycle point (TRL may reset mid-training without opening).
        """
        result = super().reset(**kwargs)
        # Best-effort telemetry; safe even if the connection was never opened.
        # The adapter is owned by this class; _record_event is the intended
        # emission path and has no public alias.
        self._connection._record_event(  # pyright: ignore[reportPrivateUsage]
            "connection.reset",
            success=True,
            env_name=self.spec.name,
            lane=self.spec.lane.value,
            kwargs_keys=sorted(kwargs.keys()),
        )
        return result

    def _record_turn(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
    ) -> None:
        """Record a turn in env history AND emit a ``connection.turn`` event.

        Matches the parent's signature exactly — concrete envs call this the
        same way they always have. The connection-side emission is a no-op
        when no chain is attached.
        """
        super()._record_turn(tool_name, args, result)
        self._connection._record_event(  # pyright: ignore[reportPrivateUsage]
            "connection.turn",
            success=True,
            tool=tool_name,
            args_keys=sorted(args.keys()),
            result_length=len(result),
            turn=self._turn_count,
            reward=self.reward,
        )

    # -- safe shutdown --------------------------------------------------

    def __del__(self) -> None:
        # Best-effort close so interpreter shutdown doesn't leak connection
        # registry entries. BaseConnection.close() is non-raising by contract.
        try:
            conn = getattr(self, "_connection", None)
            if conn is not None and conn.state != ConnectionState.CLOSED:
                conn.close()
        except (TypeError, AttributeError):
            pass
        # Invoke the parent's __del__ for any env-specific cleanup.
        try:
            super().__del__()
        except (TypeError, AttributeError):
            pass


__all__ = [
    "EnvironmentConnection",
    "DEFAULT_ENVIRONMENT_CONNECTION_SPEC",
]
