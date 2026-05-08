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


# v0.10 remote entitlements (carl.camp signed-JWT tier verification) ---------
#
# These error codes form the failure taxonomy for the studio-side
# ``EntitlementsClient`` that fetches the Ed25519-signed JWT from
# ``carl.camp``'s ``GET /api/platform/entitlements`` route, verifies the
# signature against ``/.well-known/carl-camp-jwks.json``, and caches the
# result locally with 15-min TTL plus a 24h offline-grace window. See
# ``src/carl_studio/entitlements.py`` and ``docs/v10_remote_entitlements_spec.md``.


class RemoteEntitlementError(CARLError):
    """Local tier check passed PAID but remote ed25519-verified JWT says FREE
    (or a feature is not in the entitlements list). Caller should deny the
    requested action.

    Code: ``carl.gate.tier_remote_mismatch``
    """

    code = "carl.gate.tier_remote_mismatch"


class EntitlementsNetworkError(NetworkError):
    """The carl.camp ``/api/platform/entitlements`` endpoint was unreachable.
    Caller should fall back to offline-grace cache or deny.

    Code: ``carl.entitlements.network_unavailable``
    """

    code = "carl.entitlements.network_unavailable"


class EntitlementsSignatureError(ValidationError):
    """JWT signature verification failed. Either tampered token, wrong kid,
    or our JWKS cache is stale and the signing kid is unknown.

    Code: ``carl.entitlements.signature_invalid``
    """

    code = "carl.entitlements.signature_invalid"


class EntitlementsCacheError(ValidationError):
    """Local cache file at ``~/.carl/entitlements_cache.json`` is corrupted
    or structurally invalid. Treated as a cache miss (caller refetches).

    Code: ``carl.entitlements.cache_corrupt``
    """

    code = "carl.entitlements.cache_corrupt"


class JWKSStaleError(NetworkError):
    """JWKS cache fetch failed AND the locally-cached JWKS does not contain
    the kid referenced by the JWT we're verifying. Distinct from
    :class:`EntitlementsNetworkError` because the JWT itself is fresh; only
    the pubkey lookup is stale.

    Code: ``carl.entitlements.jwks_stale``
    """

    code = "carl.entitlements.jwks_stale"


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
    "RemoteEntitlementError",
    "EntitlementsNetworkError",
    "EntitlementsSignatureError",
    "EntitlementsCacheError",
    "JWKSStaleError",
]
