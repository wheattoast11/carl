"""Central logging configuration for carl-studio.

Honors ``CARL_LOG_LEVEL`` (fallback: ``CARLSettings.log_level``) and
``CARL_LOG_JSON=1`` for single-line JSON output suitable for container
scraping. Installs a single StreamHandler on the root logger so every
``logging.getLogger(__name__)`` call in the codebase has a baseline
handler.

Design notes:

* Idempotent by default — re-calling ``configure_logging()`` without
  ``force=True`` is a no-op when a handler is already attached. The CLI
  entrypoint calls it once; tests that need to reconfigure pass
  ``force=True``.
* Emits to ``sys.stderr`` so human-readable CLI output (Rich, typer,
  etc.) stays on stdout.
* The JSON formatter preserves any ``extra={...}`` attributes the caller
  adds to ``LogRecord`` — anything JSON-serializable is copied verbatim,
  everything else collapses to ``repr``.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Attributes that the stdlib ``logging`` framework attaches to every
# ``LogRecord``. Anything not on this list is treated as user-supplied
# ``extra={...}`` metadata and included in the JSON payload.
_RESERVED_RECORD_ATTRS: frozenset[str] = frozenset({
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
})

_VALID_LEVELS: frozenset[str] = frozenset({
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
})


class _JSONFormatter(logging.Formatter):
    """Emit one JSON object per log record on a single line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, val in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS:
                continue
            # ``record.extra={...}`` attributes land here. Only include
            # JSON-serializable values verbatim; fall back to repr() for
            # everything else so the line stays valid JSON.
            try:
                json.dumps(val)
                payload[key] = val
            except (TypeError, ValueError):
                payload[key] = repr(val)
        return json.dumps(payload, default=str)


def _resolve_level(level: str | None) -> int:
    """Resolve a string level name to a logging constant.

    Priority: explicit argument, then ``CARL_LOG_LEVEL`` env var, then
    default ``INFO``. Unknown values fall back to ``INFO`` rather than
    raising — logging config should never crash the CLI on a typo.
    """
    raw = level or os.environ.get("CARL_LOG_LEVEL") or "info"
    normalized = raw.strip().upper()
    if normalized not in _VALID_LEVELS:
        normalized = "INFO"
    # ``getattr`` defaults to ``INFO`` if the constant is missing for
    # any reason (e.g. a future Python version rename).
    return int(getattr(logging, normalized, logging.INFO))


def _resolve_json_flag(json_output: bool | None) -> bool:
    """Resolve the JSON output flag from arg, env, or default ``False``."""
    if json_output is not None:
        return bool(json_output)
    env = os.environ.get("CARL_LOG_JSON", "").strip().lower()
    return env in ("1", "true", "yes", "on")


def configure_logging(
    *,
    level: str | None = None,
    json_output: bool | None = None,
    force: bool = False,
) -> None:
    """Install a single root handler at the given level.

    Idempotent by default — if a handler is already attached and
    ``force=False``, this returns without modification. Tests that need
    to reset the root logger pass ``force=True``.

    Arguments
    ---------
    level
        Logging level name (``"debug"``, ``"info"``, ``"warning"``,
        ``"error"``, ``"critical"``). When ``None``, reads
        ``CARL_LOG_LEVEL`` and falls back to ``"info"``.
    json_output
        When ``True``, emit single-line JSON. When ``None``, reads
        ``CARL_LOG_JSON`` and treats ``"1"``, ``"true"``, ``"yes"``,
        ``"on"`` (case-insensitive) as truthy.
    force
        Replace any existing handlers. Used by tests to reset state
        between runs.
    """
    root = logging.getLogger()
    if root.handlers and not force:
        return

    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    level_val = _resolve_level(level)
    root.setLevel(level_val)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level_val)

    if _resolve_json_flag(json_output):
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(handler)


__all__ = ["configure_logging"]
