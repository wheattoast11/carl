"""Shared carl.camp session and profile helpers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_CARL_CAMP_BASE = "https://carl.camp"
DEFAULT_CARL_CAMP_SUPABASE_URL = "https://ywtyyszktjfrzogwnjyo.supabase.co"
_CHECK_TIER_FUNCTION = "check-tier"
_CAMP_PROFILE_KEY = "camp_profile"
_CAMP_PROFILE_CACHED_AT_KEY = "camp_profile_cached_at"
_CAMP_TIER_TTL_HOURS = 48


class CampError(Exception):
    """Raised when carl.camp session or profile operations fail."""


class CampSession(BaseModel):
    """Local cached session state for carl.camp."""

    jwt: str | None = None
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
            "metadata": self.metadata,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_camp_session(db: Any | None = None) -> CampSession:
    """Load the locally cached camp session without network access."""
    if db is None:
        from carl_studio.db import LocalDB

        db = LocalDB()
    return CampSession(
        jwt=db.get_auth("jwt"),
        supabase_url=db.get_config("supabase_url"),
        cached_tier=db.get_auth("tier"),
        cached_profile_at=db.get_config(_CAMP_PROFILE_CACHED_AT_KEY),
    )


def load_cached_camp_profile(db: Any | None = None) -> CampProfile | None:
    """Load the last cached camp profile, if present."""
    if db is None:
        from carl_studio.db import LocalDB

        db = LocalDB()
    raw = db.get_config(_CAMP_PROFILE_KEY)
    if not raw:
        return None
    try:
        return CampProfile.model_validate_json(raw)
    except Exception:
        return None


def cache_camp_profile(profile: CampProfile, db: Any | None = None) -> CampProfile:
    """Cache the latest camp profile for offline-safe account reads."""
    if db is None:
        from carl_studio.db import LocalDB

        db = LocalDB()
    db.set_auth("tier", profile.tier, ttl_hours=_CAMP_TIER_TTL_HOURS)
    db.set_config(_CAMP_PROFILE_KEY, profile.model_dump_json())
    db.set_config(_CAMP_PROFILE_CACHED_AT_KEY, _now_iso())
    return profile


def fetch_camp_profile(jwt: str, supabase_url: str, timeout: int = 10) -> CampProfile:
    """Fetch the current camp profile from the shared check-tier function."""
    if not jwt or not supabase_url:
        raise CampError("Not authenticated. Run: carl camp login")

    url = f"{supabase_url}/functions/v1/{_CHECK_TIER_FUNCTION}"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data: dict[str, Any] = json.loads(resp.read())
            return CampProfile.model_validate(data)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace") if exc.fp else str(exc)
        raise CampError(f"Camp API error ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise CampError(f"Network error: {exc.reason}") from exc
    except CampError:
        raise
    except Exception as exc:
        raise CampError(f"Unexpected error: {exc}") from exc


def resolve_camp_profile(
    refresh: bool = True,
    db: Any | None = None,
) -> tuple[CampSession, CampProfile | None, str]:
    """Resolve the current account profile, preferring fresh data then cached data."""
    if db is None:
        from carl_studio.db import LocalDB

        db = LocalDB()

    session = load_camp_session(db=db)
    if refresh and session.jwt and session.supabase_url:
        try:
            profile = fetch_camp_profile(session.jwt, session.supabase_url)
            cache_camp_profile(profile, db=db)
            return session, profile, "remote"
        except CampError:
            cached = load_cached_camp_profile(db=db)
            if cached is not None:
                return session, cached, "cache"
            raise

    cached = load_cached_camp_profile(db=db)
    if cached is not None:
        return session, cached, "cache"
    return session, None, "none"
