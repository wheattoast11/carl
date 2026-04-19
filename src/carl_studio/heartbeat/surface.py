"""Small UI helpers that render :class:`HeartbeatConnection` status.

Kept separate from :mod:`carl_studio.heartbeat.connection` so the adapter
stays import-light (no Rich / CampConsole dependency at import time). The
surface module is only touched by CLI entrypoints that already pull the
console in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from carl_studio.heartbeat.connection import HeartbeatConnection

if TYPE_CHECKING:  # pragma: no cover - typing only
    from carl_studio.console import CampConsole


def poll_and_print(conn: HeartbeatConnection, console: "CampConsole") -> int:
    """Drain buffered heartbeat status onto ``console``.

    Non-blocking. Intended to be called between chat turns so the user
    sees phase transitions without the heartbeat having to own the UI.

    Returns the number of messages printed.
    """
    messages = conn.drain_status()
    for msg in messages:
        console.info(msg if isinstance(msg, str) else str(msg))
    return len(messages)


__all__ = ["poll_and_print"]
