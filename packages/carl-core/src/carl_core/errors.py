"""Typed error hierarchy for CARL."""
from __future__ import annotations

import traceback
from typing import Any, cast


class CARLError(Exception):
    """Base class for all CARL errors.

    Carries a stable `code` for programmatic matching, a `context` dict for
    structured debugging info, and a preserved cause chain.

    Secrets policy: keys in `context` whose NAME contains 'key', 'token',
    'secret', 'password', 'authorization' are auto-redacted in to_dict().
    """

    code: str = "carl.error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.context: dict[str, Any] = dict(context or {})
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self, *, include_traceback: bool = False) -> dict[str, Any]:
        """Serialize for logs / telemetry with secrets redacted."""
        out: dict[str, Any] = {
            "code": self.code,
            "message": str(self),
            "type": type(self).__name__,
            "context": _redact(self.context),
        }
        if self.__cause__ is not None:
            out["cause"] = {
                "type": type(self.__cause__).__name__,
                "message": str(self.__cause__),
            }
        if include_traceback:
            out["traceback"] = traceback.format_exception(
                type(self), self, self.__traceback__
            )
        return out


class ConfigError(CARLError):
    code = "carl.config"


class ValidationError(CARLError):
    code = "carl.validation"


class CredentialError(CARLError):
    code = "carl.credential"


class NetworkError(CARLError):
    code = "carl.network"


class BudgetError(CARLError):
    code = "carl.budget"


class PermissionError(CARLError):  # noqa: A001 — intentional CARL override
    code = "carl.permission"


class CARLTimeoutError(CARLError):
    """Not named TimeoutError to avoid shadowing the stdlib in `except TimeoutError:` blocks."""

    code = "carl.timeout"


_SENSITIVE_TOKENS = ("key", "token", "secret", "password", "authorization", "bearer")


def _is_sensitive(name: str) -> bool:
    lower = str(name).lower()
    return any(tok in lower for tok in _SENSITIVE_TOKENS)


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        items = cast(dict[Any, Any], obj)
        return {
            k: ("***REDACTED***" if _is_sensitive(str(k)) else _redact(v))
            for k, v in items.items()
        }
    if isinstance(obj, (list, tuple)):
        seq = cast("list[Any] | tuple[Any, ...]", obj)
        return [_redact(v) for v in seq]
    return obj


__all__ = [
    "CARLError",
    "ConfigError",
    "ValidationError",
    "CredentialError",
    "NetworkError",
    "BudgetError",
    "PermissionError",
    "CARLTimeoutError",
]
