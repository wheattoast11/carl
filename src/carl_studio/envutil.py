"""Small env-var parsing helpers. Single pattern, single warning style.

Every daemon/heartbeat knob converges here instead of each call site
hand-rolling the same ``arg-or-env-or-default`` scaffolding. Three shapes
cover every current caller:

* :func:`env_int` — integer with optional floor.
* :func:`env_float` — float with optional floor.
* :func:`env_bool` — permissive truth parser ({0,1,true,false,yes,no,on,off}).

The helpers deliberately do **not** raise on malformed input. A daemon must
not crash on a mistyped env var — it must log a warning and fall back to
the default. ``logging.warning`` fires once per call with the offending
value so an operator sees the mistake in the daemon logs on the next
restart.
"""

from __future__ import annotations

import logging
import os
from typing import TypeVar

_LOG = logging.getLogger("carl.envutil")

_T = TypeVar("_T", int, float)


def _parse(name: str, raw: str, ctor: type[_T], default: _T) -> _T:
    """Invoke ``ctor(raw)``; on failure log a warning and return ``default``."""
    try:
        return ctor(raw)
    except (TypeError, ValueError):
        _LOG.warning("invalid %s=%r; falling back to %r", name, raw, default)
        return default


def env_int(
    name: str,
    *,
    default: int,
    explicit: int | None = None,
    minimum: int | None = None,
) -> int:
    """Resolve ``explicit`` → env[name] → ``default``; clamp to ``minimum``.

    Parameters
    ----------
    name
        The environment variable to consult when ``explicit`` is ``None``.
    default
        Fallback value when neither ``explicit`` nor the env var provides a
        usable integer.
    explicit
        Caller-supplied override. When non-``None`` short-circuits env/default
        resolution; the value is still coerced via ``int(...)`` so a
        ``float``-shaped override still lands as an integer.
    minimum
        Optional floor. When set, a resolved value below ``minimum`` is
        clamped upward. Passing ``minimum=0`` is the canonical "reject
        negative" shape.
    """
    if explicit is not None:
        value = int(explicit)
    else:
        raw = os.environ.get(name, "").strip()
        value = _parse(name, raw, int, default) if raw else default
    if minimum is not None and value < minimum:
        return minimum
    return value


def env_float(
    name: str,
    *,
    default: float,
    explicit: float | None = None,
    minimum: float | None = None,
) -> float:
    """Resolve ``explicit`` → env[name] → ``default``; clamp to ``minimum``.

    See :func:`env_int` for the argument contract. Returns a ``float``;
    non-finite values (``inf``, ``nan``) are parsed but **not** rejected —
    callers that need finite inputs (e.g. poll intervals) should validate
    themselves.
    """
    if explicit is not None:
        value = float(explicit)
    else:
        raw = os.environ.get(name, "").strip()
        value = _parse(name, raw, float, default) if raw else default
    if minimum is not None and value < minimum:
        return minimum
    return value


def env_bool(name: str, *, default: bool = False) -> bool:
    """Permissive boolean parser for environment variables.

    Truthy: ``{"1", "true", "yes", "on"}`` (case-insensitive).
    Falsy: ``{"0", "false", "no", "off"}`` (case-insensitive).
    Anything else (including empty) returns ``default``.
    """
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


__all__ = ["env_bool", "env_float", "env_int"]
