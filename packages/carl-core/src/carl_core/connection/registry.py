"""Global, weakref-backed registry of live connections.

Serves two jobs:

1. **Inspection** — a process can ask "what's currently open?" without
   each caller maintaining its own bookkeeping. Useful for CLI surfaces
   like ``carl connections list`` and for sync hooks that need to know
   whether there are outstanding operations before shutdown.
2. **Graceful shutdown** — :meth:`ConnectionRegistry.close_all` iterates
   every live connection and calls its ``close`` path, even if the owning
   code has forgotten about it.

Weak references mean the registry never keeps a connection alive past
normal refcount-based collection. The only bookkeeping cost is two
dictionary operations per open/close.
"""

from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .base import ConnectionBase


class ConnectionRegistry:
    """Thread-safe registry of live connection instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registry: weakref.WeakValueDictionary[str, ConnectionBase] = (
            weakref.WeakValueDictionary()
        )

    def register(self, connection: ConnectionBase) -> None:
        """Add ``connection`` to the registry. Overwrites any prior entry
        with the same ``connection_id`` (which should be impossible for
        auto-generated IDs but is safe for explicit ones).
        """
        with self._lock:
            self._registry[connection.connection_id] = connection

    def unregister(self, connection_id: str) -> None:
        """Remove ``connection_id`` if present. No-op if absent."""
        with self._lock:
            self._registry.pop(connection_id, None)

    def get(self, connection_id: str) -> ConnectionBase | None:
        """Look up a live connection by id, or ``None`` if unknown/collected."""
        with self._lock:
            return self._registry.get(connection_id)

    def live(self) -> list[ConnectionBase]:
        """Snapshot list of currently live connections (order undefined)."""
        with self._lock:
            return list(self._registry.values())

    def count(self) -> int:
        with self._lock:
            return len(self._registry)

    def close_all(self) -> int:
        """Close every live connection. Returns the number actually closed.

        Async connections are **skipped** by this sync helper — their
        owners must orchestrate their own shutdown because invoking
        ``asyncio.run`` here would be unsafe from within an existing
        event loop.
        """
        from .base import BaseConnection  # local import avoids cycle at load

        closed = 0
        for conn in self.live():
            if not isinstance(conn, BaseConnection):
                continue
            try:
                conn.close()
            except BaseException:
                # close() is documented as non-raising; if a subclass breaks
                # the contract we still want to sweep the rest.
                continue
            closed += 1
        return closed

    def to_dict(self) -> list[dict[str, Any]]:
        """List-of-dicts summary for logs / CLI output."""
        return [c.to_dict() for c in self.live()]


# Process-wide singleton, constructed lazily so the import cost is a
# single object allocation when the registry is first touched. Named in
# snake_case because it's mutable — pyright flags uppercase module globals
# as immutable constants.
_global_registry: ConnectionRegistry | None = None
_global_lock = threading.Lock()


def get_registry() -> ConnectionRegistry:
    """Return the process-wide :class:`ConnectionRegistry`."""
    global _global_registry
    if _global_registry is None:
        with _global_lock:
            if _global_registry is None:
                _global_registry = ConnectionRegistry()
    return _global_registry


def reset_registry() -> None:
    """Replace the global registry with a fresh, empty one.

    Intended for tests only — production code should not call this.
    """
    global _global_registry
    with _global_lock:
        _global_registry = ConnectionRegistry()


__all__ = [
    "ConnectionRegistry",
    "get_registry",
    "reset_registry",
]
