"""Protocol for unified training backends.

Every adapter implements :class:`UnifiedBackend` and returns
:class:`BackendJob` records from ``submit()`` and ``status()``. The protocol is
intentionally thin: we translate ``carl.yaml`` into the target system's
native config and shell out (or call its Python API) — we do not reimplement
trainers here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from carl_core.errors import CARLError


class AdapterError(CARLError):
    """Raised by :class:`UnifiedBackend` implementations for any adapter-level
    failure (missing CLI, missing import, submission rejection, etc).

    Callers can branch on the stable ``code`` field. Built-in codes:
      * ``carl.adapter`` — generic/base
      * ``carl.adapter.unavailable`` — backend not installed/reachable
      * ``carl.adapter.submit`` — submission failed at the backend boundary
      * ``carl.adapter.status`` — status probe failed
      * ``carl.adapter.translation`` — carl.yaml could not be translated
    """

    code = "carl.adapter"


class BackendStatus:
    """Canonical status strings used across adapters.

    Keep these as plain string constants (not an Enum) so backend-specific
    values can be stored in ``BackendJob.raw`` without coupling the consumer
    to a fixed vocabulary.
    """

    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    _ALL: frozenset[str] = frozenset(
        {PENDING, PROVISIONING, RUNNING, COMPLETED, FAILED, CANCELED}
    )

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        return status in {cls.COMPLETED, cls.FAILED, cls.CANCELED}

    @classmethod
    def is_known(cls, status: str) -> bool:
        return status in cls._ALL


@dataclass
class BackendJob:
    """A handle to a training run submitted through a :class:`UnifiedBackend`.

    ``run_id`` is opaque and backend-specific. Consumers must not parse it —
    they should round-trip it through ``status()``/``logs()``/``cancel()``.
    """

    run_id: str
    backend: str
    status: str = BackendStatus.PENDING
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    logs_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "run_id": self.run_id,
            "backend": self.backend,
            "status": self.status,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": dict(self.metrics),
            "logs_url": self.logs_url,
            "raw": dict(self.raw),
        }


@runtime_checkable
class UnifiedBackend(Protocol):
    """Minimum contract every training adapter must satisfy.

    Implementations may provide additional methods, but the five below are
    what the registry and CLI speak.
    """

    name: str

    def available(self) -> bool:
        """Return True if the backend is installed/reachable locally.

        Implementations must NOT raise from ``available()`` — it is used to
        decide whether to even attempt a submission.
        """

    def submit(self, carl_config: dict[str, Any]) -> BackendJob:
        """Translate ``carl_config`` → backend config, submit, return a
        :class:`BackendJob`.

        Raises:
            AdapterError: if the backend is unavailable or submission fails.
        """

    def status(self, run_id: str) -> BackendJob:
        """Return the current status for ``run_id``.

        Raises:
            AdapterError: on auth/availability issues or unknown run.
        """

    def logs(self, run_id: str, *, tail: int = 100) -> list[str]:
        """Return the last ``tail`` log lines for ``run_id``.

        Raises:
            AdapterError: on availability issues or unknown run.
        """

    def cancel(self, run_id: str) -> bool:
        """Cancel a running job. Return True if the cancel was accepted.

        Raises:
            AdapterError: on availability issues or unknown run.
        """
