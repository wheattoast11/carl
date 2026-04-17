"""Shared carl.camp session and profile helpers.

Adds TTL-aware profile caching, JWT auto-refresh on HTTP 401, transparent
retry on transient network failures, and a `stale` freshness signal so
callers can surface "served from cache" states without raising.

Error hierarchy
---------------
``CampError`` is now a :class:`carl_core.errors.CARLError` subclass. The
concrete errors raised by :func:`fetch_camp_profile` use multiple
inheritance so they are catchable as either the typed carl_core parent
(``NetworkError`` / ``CredentialError``) or the legacy ``CampError``:

* :class:`CampNetworkError` ← ``NetworkError`` + ``CampError``
* :class:`CampCredentialError` ← ``CredentialError`` + ``CampError``
* :class:`CampHttpError` ← ``CampError`` (non-retryable 4xx body)

This keeps every existing ``except CampError:`` site working while giving
new code access to the typed taxonomy.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from carl_core.errors import CARLError, CredentialError, NetworkError
from carl_core.retry import RetryPolicy, retry
from pydantic import BaseModel, Field

DEFAULT_CARL_CAMP_BASE = "https://carl.camp"
DEFAULT_CARL_CAMP_SUPABASE_URL = "https://ywtyyszktjfrzogwnjyo.supabase.co"
_CHECK_TIER_FUNCTION = "check-tier"
_REFRESH_FUNCTION = "auth/refresh"
_CAMP_PROFILE_KEY = "camp_profile"
_CAMP_PROFILE_CACHED_AT_KEY = "camp_profile_cached_at"
_CAMP_PROFILE_FETCHED_AT_KEY = "camp_profile_fetched_at"
_CAMP_TIER_TTL_HOURS = 48

# TTL policy ----------------------------------------------------------------
# Fresh window: 24h. Past that, resolve_camp_profile() refreshes by default.
# Stale-serve window: up to 7× TTL (1 week) — returned with `stale=True`
# when the network is unreachable, so callers can render cached data
# without masking the freshness problem.
CAMP_PROFILE_TTL_S: int = 24 * 60 * 60
_CAMP_PROFILE_STALE_MAX_S: int = 7 * CAMP_PROFILE_TTL_S

# Retry policy for the raw HTTP layer. CredentialError + non-retryable
# CARL errors are NOT in `retryable` — only true transient IO is retried.
_FETCH_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    backoff_base=1.0,
    max_delay=5.0,
    retryable=(NetworkError, IOError, ConnectionError, TimeoutError),
)


class CampError(CARLError):
    """Legacy base for camp errors — retained for backward compat.

    New code should catch :class:`carl_core.errors.CARLError` or one of
    its subclasses. Existing ``except CampError:`` call sites still
    capture every failure because the concrete errors raised by
    :func:`fetch_camp_profile` inherit from both their typed carl_core
    parent and :class:`CampError`.
    """

    code = "carl.camp"


class CampNetworkError(NetworkError, CampError):
    """Camp-specific transport failure — a NetworkError and a CampError."""

    code = "carl.camp.network"


class CampCredentialError(CredentialError, CampError):
    """Camp-specific auth failure — a CredentialError and a CampError."""

    code = "carl.camp.auth_expired"


class CampHttpError(CampError):
    """Non-retryable 4xx response with structured detail."""

    code = "carl.camp.http_error"


class CampSession(BaseModel):
    """Local cached session state for carl.camp."""

    jwt: str | None = None
    refresh_token: str | None = None
    supabase_url: str | None = None
    cached_tier: str | None = None
    cached_profile_at: str | None = None

    @property
    def authenticated(self) -> bool:
        return bool(self.jwt)

    @property
    def configured(self) -> bool:
        return bool(self.supabase_url)


class CampProfile(BaseModel):
    """Typed account profile returned by the public check-tier contract."""

    tier: str = "free"
    plan: str | None = None
    status: str = "unknown"
    current_period_end: str | None = None
    cancel_at_period_end: bool = False
    stripe_customer_id: str | None = None
    user_id: str | None = None
    email: str | None = None
    credits_total: int = 0
    credits_remaining: int = 0
    credits_used: int = 0
    credits_monthly_included: int = 0
    payment_methods: list[str] = Field(default_factory=list)
    wallet_auth_enabled: bool = False
    x402_enabled: bool = False
    observability_opt_in: bool = False
    telemetry_opt_in: bool = False
    usage_tracking_enabled: bool = False
    contract_witnessing: bool = False
    contract_terms_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Freshness signal: True when the profile was served from cache after a
    # failed refresh (cache within the stale-serve window). Excluded from
    # persisted JSON so cached copies never come back stale-by-default.
    stale: bool = Field(default=False, exclude=True)

    @property
    def is_active_paid(self) -> bool:
        return self.tier == "paid" and self.status == "active"

    @property
    def days_remaining(self) -> int | None:
        if not self.current_period_end:
            return None
        try:
            end = datetime.fromisoformat(self.current_period_end.replace("Z", "+00:00"))
            delta = end - datetime.now(timezone.utc)
            return max(0, delta.days)
        except Exception:
            return None

    @property
    def payment_summary(self) -> str:
        if self.payment_methods:
            return ", ".join(self.payment_methods)
        if self.x402_enabled and self.wallet_auth_enabled:
            return "wallet + x402"
        if self.x402_enabled:
            return "x402"
        if self.wallet_auth_enabled:
            return "wallet"
        return "card"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "plan": self.plan,
            "status": self.status,
            "current_period_end": self.current_period_end,
            "cancel_at_period_end": self.cancel_at_period_end,
            "stripe_customer_id": self.stripe_customer_id,
            "user_id": self.user_id,
            "email": self.email,
            "credits_total": self.credits_total,
            "credits_remaining": self.credits_remaining,
            "credits_used": self.credits_used,
            "credits_monthly_included": self.credits_monthly_included,
            "payment_methods": self.payment_methods,
            "payment_summary": self.payment_summary,
            "wallet_auth_enabled": self.wallet_auth_enabled,
            "x402_enabled": self.x402_enabled,
            "observability_opt_in": self.observability_opt_in,
            "telemetry_opt_in": self.telemetry_opt_in,
            "usage_tracking_enabled": self.usage_tracking_enabled,
            "contract_witnessing": self.contract_witnessing,
            "contract_terms_url": self.contract_terms_url,
            "days_remaining": self.days_remaining,
            "is_active_paid": self.is_active_paid,
            "stale": self.stale,
            "metadata": self.metadata,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_db(db: Any | None) -> Any:
    if db is not None:
        return db
    from carl_studio.db import LocalDB

    return LocalDB()


def load_camp_session(db: Any | None = None) -> CampSession:
    """Load the locally cached camp session without network access."""
    db = _load_db(db)
    return CampSession(
        jwt=db.get_auth("jwt"),
        refresh_token=db.get_auth("refresh_token"),
        supabase_url=db.get_config("supabase_url"),
        cached_tier=db.get_auth("tier"),
        cached_profile_at=db.get_config(_CAMP_PROFILE_CACHED_AT_KEY),
    )


def load_cached_camp_profile(db: Any | None = None) -> CampProfile | None:
    """Load the last cached camp profile, if present."""
    db = _load_db(db)
    raw = db.get_config(_CAMP_PROFILE_KEY)
    if not raw:
        return None
    try:
        return CampProfile.model_validate_json(raw)
    except Exception:
        return None


def cache_camp_profile(
    profile: CampProfile,
    db: Any | None = None,
    *,
    clock: Callable[[], float] = time.time,
) -> CampProfile:
    """Cache the latest camp profile for offline-safe account reads.

    Persists tier (for quick look-ups), the JSON profile, a wall-clock
    ISO timestamp for display, and a float epoch timestamp used for the
    TTL math in :func:`resolve_camp_profile`.
    """
    db = _load_db(db)
    db.set_auth("tier", profile.tier, ttl_hours=_CAMP_TIER_TTL_HOURS)
    db.set_config(_CAMP_PROFILE_KEY, profile.model_dump_json())
    db.set_config(_CAMP_PROFILE_CACHED_AT_KEY, _now_iso())
    db.set_config(_CAMP_PROFILE_FETCHED_AT_KEY, f"{clock():.3f}")
    return profile


def _cached_profile_age_s(
    db: Any,
    *,
    clock: Callable[[], float] = time.time,
) -> float | None:
    """Return age of cached profile in seconds, or None if no timestamp.

    None is distinct from "very old cache" — callers branch on it.
    """
    ts_raw = db.get_config(_CAMP_PROFILE_FETCHED_AT_KEY)
    if not ts_raw:
        return None
    try:
        ts = float(ts_raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, clock() - ts)


def _store_tokens(db: Any, jwt: str, refresh_token: str | None) -> None:
    """Persist a newly refreshed JWT (and refresh token) to LocalDB."""
    db.set_auth("jwt", jwt, ttl_hours=24)
    if refresh_token:
        # Refresh tokens live past the JWT's 24h TTL.
        db.set_auth("refresh_token", refresh_token, ttl_hours=24 * 30)


def _http_get_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
) -> tuple[int, dict[str, Any]]:
    """GET JSON, returning (status, body). Transport failures raise NetworkError.

    HTTP-level errors (including 401) return normally so the caller can
    branch on them. Only true transport failures raise.
    """
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            raw = resp.read()
            body: dict[str, Any] = json.loads(raw) if raw else {}
            return status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read() if exc.fp else b""
        body_text = raw.decode(errors="replace") if raw else str(exc)
        try:
            parsed = json.loads(body_text) if body_text else {}
            if not isinstance(parsed, dict):
                parsed = {"detail": parsed}
        except (ValueError, TypeError):
            parsed = {"detail": body_text}
        return int(exc.code), parsed
    except urllib.error.URLError as exc:
        raise CampNetworkError(
            f"Network error: {exc.reason}",
            code="carl.camp.network",
            context={"url": url},
            cause=exc,
        ) from exc
    except TimeoutError as exc:
        raise CampNetworkError(
            f"Network error: timeout after {timeout}s",
            code="carl.camp.timeout",
            context={"url": url, "timeout_s": timeout},
            cause=exc,
        ) from exc


def _http_post_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
) -> tuple[int, dict[str, Any]]:
    """POST JSON body, returning (status, parsed_body)."""
    data = json.dumps(payload).encode("utf-8")
    merged = dict(headers)
    merged.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=merged, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            raw = resp.read()
            body: dict[str, Any] = json.loads(raw) if raw else {}
            return status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read() if exc.fp else b""
        body_text = raw.decode(errors="replace") if raw else str(exc)
        try:
            parsed = json.loads(body_text) if body_text else {}
            if not isinstance(parsed, dict):
                parsed = {"detail": parsed}
        except (ValueError, TypeError):
            parsed = {"detail": body_text}
        return int(exc.code), parsed
    except urllib.error.URLError as exc:
        raise CampNetworkError(
            f"Network error: {exc.reason}",
            code="carl.camp.network",
            context={"url": url},
            cause=exc,
        ) from exc
    except TimeoutError as exc:
        raise CampNetworkError(
            f"Network error: timeout after {timeout}s",
            code="carl.camp.timeout",
            context={"url": url, "timeout_s": timeout},
            cause=exc,
        ) from exc


def _refresh_jwt(
    supabase_url: str,
    refresh_token: str,
    *,
    timeout: int,
) -> tuple[str, str | None]:
    """Exchange a refresh token for a fresh JWT.

    Returns ``(jwt, rotated_refresh_or_none)``. Raises
    :class:`CampCredentialError` with ``code="carl.camp.auth_expired"`` on
    refresh failure — caller must re-login.
    """
    url = f"{supabase_url}/functions/v1/{_REFRESH_FUNCTION}"
    status, body = _http_post_json(
        url,
        payload={"refresh_token": refresh_token},
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    if status >= 400:
        raise CampCredentialError(
            "Session refresh failed — please re-authenticate.",
            code="carl.camp.auth_expired",
            context={"status": status, "hint": "run: carl camp login"},
        )
    new_jwt = body.get("access_token") or body.get("jwt")
    if not isinstance(new_jwt, str) or not new_jwt:
        raise CampCredentialError(
            "Refresh endpoint returned no access token.",
            code="carl.camp.auth_expired",
            context={"hint": "run: carl camp login"},
        )
    new_refresh = body.get("refresh_token")
    if new_refresh is not None and not isinstance(new_refresh, str):
        new_refresh = None
    return new_jwt, new_refresh


def fetch_camp_profile(
    jwt: str,
    supabase_url: str,
    timeout: int = 10,
    *,
    db: Any | None = None,
    refresh_token: str | None = None,
) -> CampProfile:
    """Fetch the current camp profile from the shared check-tier contract.

    JWT refresh: on HTTP 401 the function attempts a refresh-token
    exchange (using ``refresh_token`` or, if ``db`` is supplied, the
    stored token) and retries the request exactly once with the new JWT.
    On refresh failure it raises :class:`CampCredentialError` with
    ``code="carl.camp.auth_expired"`` and a login hint.

    HTTP calls are wrapped in :func:`carl_core.retry.retry` so transient
    errors (NetworkError, IOError, ConnectionError, TimeoutError) retry
    up to 3 times with exponential backoff.
    """
    if not jwt or not supabase_url:
        raise CampCredentialError(
            "Not authenticated. Run: carl camp login",
            code="carl.camp.not_authenticated",
            context={"hint": "run: carl camp login"},
        )

    url = f"{supabase_url}/functions/v1/{_CHECK_TIER_FUNCTION}"

    # Closure state so the retry wrapper can observe token rotation.
    state: dict[str, Any] = {
        "jwt": jwt,
        "refresh_token": refresh_token,
        "tried_refresh": False,
    }

    def _attempt() -> CampProfile:
        headers = {
            "Authorization": f"Bearer {state['jwt']}",
            "Content-Type": "application/json",
        }
        status, body = _http_get_json(url, headers=headers, timeout=timeout)

        if status == 401 and not state["tried_refresh"]:
            state["tried_refresh"] = True
            current_refresh = state.get("refresh_token")
            if current_refresh is None and db is not None:
                current_refresh = db.get_auth("refresh_token")
            if not current_refresh:
                raise CampCredentialError(
                    "Camp API error (401): JWT expired and no refresh token "
                    "is stored. Run: carl camp login",
                    code="carl.camp.auth_expired",
                    context={"status": 401, "hint": "run: carl camp login"},
                )
            new_jwt, rotated_refresh = _refresh_jwt(
                supabase_url, current_refresh, timeout=timeout
            )
            state["jwt"] = new_jwt
            if rotated_refresh:
                state["refresh_token"] = rotated_refresh
            if db is not None:
                _store_tokens(db, new_jwt, rotated_refresh or current_refresh)
            # Retry original request with refreshed JWT.
            headers = {
                "Authorization": f"Bearer {state['jwt']}",
                "Content-Type": "application/json",
            }
            status, body = _http_get_json(url, headers=headers, timeout=timeout)

        if status == 401:
            raise CampCredentialError(
                "Authentication failed after token refresh.",
                code="carl.camp.auth_expired",
                context={"status": status, "hint": "run: carl camp login"},
            )
        if status == 403:
            raise CampHttpError(
                f"Camp API error ({status}): forbidden",
                code="carl.camp.forbidden",
                context={"status": status},
            )
        if status == 429:
            # Surface as retryable NetworkError so the retry wrapper backs off.
            raise CampNetworkError(
                f"Network error: carl.camp rate limited ({status}).",
                code="carl.camp.rate_limited",
                context={"status": status},
            )
        if 500 <= status < 600:
            raise CampNetworkError(
                f"Network error: carl.camp transient error ({status}).",
                code="carl.camp.transient",
                context={"status": status},
            )
        if status >= 400:
            detail = body.get("detail") if isinstance(body, dict) else None
            raise CampHttpError(
                f"Camp API error ({status}): {detail or body}",
                code="carl.camp.http_error",
                context={"status": status},
            )

        try:
            return CampProfile.model_validate(body)
        except Exception as exc:
            raise CampError(
                f"Unexpected error: invalid camp profile payload: {exc}",
                code="carl.camp.invalid_payload",
                cause=exc,
            ) from exc

    try:
        return retry(_attempt, policy=_FETCH_RETRY_POLICY)
    except CampError:
        raise
    except CARLError:
        # Any other typed error from carl_core bubbles up unchanged.
        raise
    except Exception as exc:  # pragma: no cover — final defensive guard
        raise CampError(
            f"Unexpected error reaching carl.camp: {exc}",
            code="carl.camp.unreachable",
            cause=exc,
        ) from exc


def _emit_tier_change(
    console: Any | None,
    old_tier: str | None,
    new_tier: str,
) -> None:
    """Emit an info line about a tier transition when a CLI console is passed."""
    if console is None or old_tier == new_tier:
        return
    msg = f"Camp tier updated: {old_tier or 'unknown'} → {new_tier}"
    for method_name in ("info", "print", "log"):
        method = getattr(console, method_name, None)
        if callable(method):
            try:
                method(msg)
                return
            except Exception:
                continue


def resolve_camp_profile(
    refresh: bool = True,
    db: Any | None = None,
    *,
    force_refresh: bool = False,
    console: Any | None = None,
    clock: Callable[[], float] = time.time,
) -> tuple[CampSession, CampProfile | None, str]:
    """Resolve the current account profile with TTL-aware caching.

    Decision ladder:
      1. ``force_refresh`` OR cache older than :data:`CAMP_PROFILE_TTL_S`,
         with a live session → fetch fresh.
      2. Fetch success → cache, return ``(session, profile, "remote")``.
      3. Fetch failure within the stale-serve window (7× TTL) → return
         cached profile with ``stale=True`` and source ``"cache"``.
      4. Fetch failure with no valid cache → raise
         :class:`CampNetworkError` with ``code="carl.camp.unreachable"``.
      5. Legacy ``refresh=False`` path: serve cache only.

    When the newly fetched tier differs from the cached tier, ``LocalDB``
    is updated and (when ``console`` is supplied) an info message is
    emitted so CLI callers can surface the change.
    """
    db = _load_db(db)

    session = load_camp_session(db=db)
    want_refresh = bool(refresh) or bool(force_refresh)

    if want_refresh and session.jwt and session.supabase_url:
        age = _cached_profile_age_s(db, clock=clock)
        cache_is_fresh = age is not None and age < CAMP_PROFILE_TTL_S

        if not force_refresh and cache_is_fresh:
            cached = load_cached_camp_profile(db=db)
            if cached is not None:
                cached.stale = False
                return session, cached, "cache"
            # Missing cached payload despite a fresh timestamp — fall through
            # and fetch fresh.

        try:
            profile = fetch_camp_profile(
                session.jwt,
                session.supabase_url,
                db=db,
                refresh_token=session.refresh_token,
            )
        except CredentialError:
            # Auth expired — surface it, don't silently serve cache.
            raise
        except NetworkError as exc:
            cached = load_cached_camp_profile(db=db)
            if cached is not None and age is not None and age < _CAMP_PROFILE_STALE_MAX_S:
                cached.stale = True
                return session, cached, "cache"
            raise CampNetworkError(
                "carl.camp is unreachable and no valid cached profile is available.",
                code="carl.camp.unreachable",
                context={"cache_age_s": age},
                cause=exc,
            ) from exc
        except CampError as exc:
            # Any other camp-layer error — treat like a network failure for
            # the unified stale-serve path.
            cached = load_cached_camp_profile(db=db)
            if cached is not None and age is not None and age < _CAMP_PROFILE_STALE_MAX_S:
                cached.stale = True
                return session, cached, "cache"
            raise CampNetworkError(
                str(exc),
                code="carl.camp.unreachable",
                cause=exc,
            ) from exc

        prev_tier = session.cached_tier
        cache_camp_profile(profile, db=db, clock=clock)
        if prev_tier and prev_tier != profile.tier:
            _emit_tier_change(console, prev_tier, profile.tier)
        profile.stale = False
        return session, profile, "remote"

    cached = load_cached_camp_profile(db=db)
    if cached is not None:
        cached.stale = False
        return session, cached, "cache"
    return session, None, "none"


__all__ = [
    "DEFAULT_CARL_CAMP_BASE",
    "DEFAULT_CARL_CAMP_SUPABASE_URL",
    "CAMP_PROFILE_TTL_S",
    "CampError",
    "CampNetworkError",
    "CampCredentialError",
    "CampHttpError",
    "CampSession",
    "CampProfile",
    "load_camp_session",
    "load_cached_camp_profile",
    "cache_camp_profile",
    "fetch_camp_profile",
    "resolve_camp_profile",
]
