"""UTC time helpers. Single source of truth for the ISO-8601 format the
rest of the codebase expects.
"""
from __future__ import annotations

from datetime import datetime, timezone

ISO8601_Z = "%Y-%m-%dT%H:%M:%SZ"


def now_iso() -> str:
    """Return current UTC time as an ISO-8601 string with ``Z`` suffix.

    Second-resolution — matches the format used by ``~/.carl`` SQLite rows
    (``sticky_notes.created_at``, etc.) and every JSONL trace file in the
    project. Microseconds are truncated.
    """
    return datetime.now(timezone.utc).strftime(ISO8601_Z)


def now_iso_ms() -> str:
    """Millisecond-resolution ISO-8601 variant for logging."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = ["ISO8601_Z", "now_iso", "now_iso_ms"]
