"""Billing utilities for carl.camp subscription management."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

CARL_CAMP_BASE = "https://carl.camp"
CHECKOUT_MONTHLY_URL = f"{CARL_CAMP_BASE}/checkout?plan=monthly"
CHECKOUT_ANNUAL_URL = f"{CARL_CAMP_BASE}/checkout?plan=annual"
BILLING_PORTAL_URL = f"{CARL_CAMP_BASE}/billing"
PRICING_URL = f"{CARL_CAMP_BASE}/pricing"


class BillingError(Exception):
    """Raised when billing API calls fail."""


class SubscriptionStatus:
    """Current subscription state for a user."""

    tier: str                       # "free" | "paid"
    plan: str | None                # "monthly" | "annual" | None
    status: str                     # "active" | "cancelled" | "past_due" | "unknown"
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
            end = datetime.fromisoformat(
                self.current_period_end.replace("Z", "+00:00")
            )
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


def get_subscription_status(jwt: str, supabase_url: str) -> SubscriptionStatus:
    """Fetch current subscription from the check-tier Edge Function.

    Calls the Supabase check-tier Edge Function which reads user_profiles
    joined with the Stripe FDW. Returns a SubscriptionStatus with tier,
    plan, and renewal date.

    Raises BillingError on any network or API failure — callers must handle
    graceful offline fallback.
    """
    url = f"{supabase_url}/functions/v1/check-tier"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data: dict[str, Any] = json.loads(resp.read())
            return SubscriptionStatus(**data)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace") if e.fp else str(e)
        raise BillingError(f"Billing API error ({e.code}): {body}") from e
    except urllib.error.URLError as e:
        raise BillingError(f"Network error: {e.reason}") from e
    except BillingError:
        raise
    except Exception as e:
        raise BillingError(f"Unexpected error: {e}") from e
