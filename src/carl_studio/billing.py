"""Billing utilities for carl.camp subscription management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from carl_studio.camp import CampError, CampProfile, fetch_camp_profile

CARL_CAMP_BASE = "https://carl.camp"
CHECKOUT_MONTHLY_URL = f"{CARL_CAMP_BASE}/checkout?plan=monthly"
CHECKOUT_ANNUAL_URL = f"{CARL_CAMP_BASE}/checkout?plan=annual"
BILLING_PORTAL_URL = f"{CARL_CAMP_BASE}/billing"
PRICING_URL = f"{CARL_CAMP_BASE}/pricing"


class BillingError(Exception):
    """Raised when billing API calls fail."""


class SubscriptionStatus:
    """Current subscription state for a user."""

    tier: str  # "free" | "paid"
    plan: str | None  # "monthly" | "annual" | None
    status: str  # "active" | "cancelled" | "past_due" | "unknown"
    current_period_end: str | None  # ISO date string
    cancel_at_period_end: bool
    stripe_customer_id: str | None

    def __init__(self, **kwargs: Any) -> None:
        self.tier = str(kwargs.get("tier", "free"))
        raw_plan = kwargs.get("plan")
        self.plan = str(raw_plan) if raw_plan is not None else None
        self.status = str(kwargs.get("status", "unknown"))
        raw_end = kwargs.get("current_period_end")
        self.current_period_end = str(raw_end) if raw_end is not None else None
        self.cancel_at_period_end = bool(kwargs.get("cancel_at_period_end", False))
        raw_cid = kwargs.get("stripe_customer_id")
        self.stripe_customer_id = str(raw_cid) if raw_cid is not None else None

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for --json output."""
        return {
            "tier": self.tier,
            "plan": self.plan,
            "status": self.status,
            "current_period_end": self.current_period_end,
            "cancel_at_period_end": self.cancel_at_period_end,
            "stripe_customer_id": self.stripe_customer_id,
            "days_remaining": self.days_remaining,
            "is_active_paid": self.is_active_paid,
        }


def subscription_status_from_profile(profile: CampProfile) -> SubscriptionStatus:
    """Project a shared camp profile onto the billing subscription view."""
    return SubscriptionStatus(
        tier=profile.tier,
        plan=profile.plan,
        status=profile.status,
        current_period_end=profile.current_period_end,
        cancel_at_period_end=profile.cancel_at_period_end,
        stripe_customer_id=profile.stripe_customer_id,
    )


def get_subscription_status(jwt: str, supabase_url: str) -> SubscriptionStatus:
    """Fetch current subscription from the shared camp profile contract."""
    try:
        profile = fetch_camp_profile(jwt, supabase_url)
        return subscription_status_from_profile(profile)
    except CampError as exc:
        raise BillingError(str(exc)) from exc
