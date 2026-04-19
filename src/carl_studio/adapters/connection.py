"""TrainingConnection — Connection-aware base for every training backend.

Every concrete adapter (TRL, Unsloth, Axolotl, Atropos, Tinker) gains the
``carl_core.connection`` lifecycle by inheriting :class:`TrainingConnection`
instead of being a standalone class. The existing ``UnifiedBackend``
protocol continues to describe the duck-typed contract the rest of the
stack speaks (``.name``, ``.available()``, ``.submit()``, ``.status()``,
``.logs()``, ``.cancel()``), so callers do not change.

The port is opt-in-by-subclassing: adapters become ``TrainingConnection``
subclasses and gain ``open()`` / ``close()`` / ``transact()`` and typed
``InteractionChain`` telemetry. Callers that just want to ``submit()``
without running a lifecycle keep working because the connection defaults
to a no-op ``_connect`` / ``_close`` — the state machine is optional for
adapters that have no persistent transport.

A helper :meth:`submit_with_telemetry` is provided for callers that want
the state transitions to trace every submit through the chain.
"""

from __future__ import annotations

from typing import Any

from carl_core.connection import (
    BaseConnection,
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
    ConnectionUnavailableError,
)

from .protocol import BackendJob

# Default spec for a generic training backend. Concrete adapters override
# `spec` on their class to carry a backend-specific name / endpoint. We
# describe the *shape* of the 3P training interaction here — egress over
# subprocess with API-key-style trust.
DEFAULT_TRAINING_SPEC = ConnectionSpec(
    name="carl.training.generic",
    scope=ConnectionScope.THREE_P,
    kind=ConnectionKind.TRAINING,
    direction=ConnectionDirection.EGRESS,
    transport=ConnectionTransport.SUBPROCESS,
    trust=ConnectionTrust.AUTHENTICATED,
)


class TrainingConnection(BaseConnection):
    """Connection-aware base for training adapters.

    Subclasses declare their own class-level :attr:`spec` (an instance of
    :class:`~carl_core.connection.ConnectionSpec`) and implement the
    :class:`~carl_studio.adapters.protocol.UnifiedBackend` surface
    (``name``, ``available``, ``submit``, ``status``, ``logs``,
    ``cancel``). The default ``_connect`` checks :meth:`available` and
    raises :class:`ConnectionUnavailableError` when the backend is not
    reachable; ``_close`` is a no-op because most adapters spawn
    subprocesses per-run rather than holding a persistent transport.

    The ``name`` attribute (string, required by ``UnifiedBackend``) is
    kept as a plain class attribute so duck-typed consumers continue to
    read it without touching the connection FSM.
    """

    spec: ConnectionSpec = DEFAULT_TRAINING_SPEC

    name: str = "generic"

    # -- Connection lifecycle hooks --------------------------------------

    def _connect(self) -> None:
        """Verify the backend is reachable.

        Backends that need more than an availability probe (e.g., log in
        to an API, allocate a session) should override. The default is
        enough for adapter classes that are always available in-process
        (like TRL) or that check for a CLI binary at submit time.
        """
        try:
            ok = self.available()
        except Exception as exc:  # pragma: no cover - defensive
            raise ConnectionUnavailableError(
                f"backend {self.name!r} availability probe raised",
                context={"backend": self.name},
                cause=exc,
            ) from exc
        if not ok:
            raise ConnectionUnavailableError(
                f"backend {self.name!r} is not available",
                context={"backend": self.name},
            )

    def _close(self) -> None:
        """Default: no persistent transport to tear down."""

    # -- Duck-typed UnifiedBackend surface (abstract here) ---------------

    def available(self) -> bool:  # pragma: no cover - subclass hook
        raise NotImplementedError

    def submit(self, carl_config: dict[str, Any]) -> BackendJob:  # pragma: no cover
        raise NotImplementedError

    def status(self, run_id: str) -> BackendJob:  # pragma: no cover
        raise NotImplementedError

    def logs(self, run_id: str, *, tail: int = 100) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def cancel(self, run_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    # -- Optional telemetry-wrapped submit -------------------------------

    def submit_with_telemetry(self, carl_config: dict[str, Any]) -> BackendJob:
        """Run :meth:`submit` inside a ``transact("submit")`` bracket.

        Useful when the adapter instance has already been opened and you
        want per-submit state transitions + chain events. Callers that
        prefer the direct path (the historical one) can still invoke
        :meth:`submit` without opening the connection.
        """
        with self.transact("submit"):
            job = self.submit(carl_config)
        self._record_event(
            "connection.submit.job",
            success=True,
            run_id=job.run_id,
            backend=job.backend,
        )
        return job


__all__ = ["TrainingConnection", "DEFAULT_TRAINING_SPEC"]
