"""Credit balance management.

All network calls use stdlib urllib -- zero external deps.
Errors surface as CreditError so callers can handle offline gracefully.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel

from carl_studio.camp import CampError, CampProfile, fetch_camp_profile


class CreditError(Exception):
    """Raised when credit operations fail."""


class CreditBalance(BaseModel):
    """Current credit state for a user."""

    total: int = 0
    """Lifetime credits purchased."""

    remaining: int = 0
    """Credits available now."""

    used: int = 0
    """Credits consumed."""

    included_monthly: int = 0
    """Credits from subscription (200 monthly, 300 annual)."""

    @property
    def sufficient(self) -> bool:
        """Whether the user has any credits at all."""
        return self.remaining > 0

    def can_afford(self, cost: int) -> bool:
        """Whether the user can afford a specific credit cost."""
        return self.remaining >= cost


# ---------------------------------------------------------------------------
# Remote credit operations (Supabase Edge Functions)
# ---------------------------------------------------------------------------


def credit_balance_from_profile(profile: CampProfile) -> CreditBalance:
    """Project a shared camp profile onto the credits balance view."""
    return CreditBalance(
        total=profile.credits_total,
        remaining=profile.credits_remaining,
        used=profile.credits_used,
        included_monthly=profile.credits_monthly_included,
    )


def get_credit_balance(jwt: str, supabase_url: str) -> CreditBalance:
    """Fetch credit balance from the shared camp profile contract."""
    try:
        profile = fetch_camp_profile(jwt, supabase_url)
        return credit_balance_from_profile(profile)
    except CampError as exc:
        raise CreditError(str(exc)) from exc


def deduct_credits(
    jwt: str,
    supabase_url: str,
    amount: int,
    job_id: str,
    reason: str = "",
) -> bool:
    """Pre-deduct credits before job submission.

    Returns True on success. Raises CreditError on failure.
    The server enforces idempotency via job_id -- duplicate deductions
    for the same job_id are rejected.
    """
    if amount <= 0:
        raise CreditError(f"Deduction amount must be positive, got {amount}")

    url = f"{supabase_url}/functions/v1/deduct-credits"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }
    body = json.dumps(
        {
            "amount": amount,
            "job_id": job_id,
            "reason": reason,
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data: dict[str, Any] = json.loads(resp.read())
            return bool(data.get("success", False))
    except urllib.error.HTTPError as e:
        body_str = e.read().decode(errors="replace") if e.fp else str(e)
        raise CreditError(f"Deduction failed ({e.code}): {body_str}") from e
    except urllib.error.URLError as e:
        raise CreditError(f"Deduction failed (network): {e.reason}") from e
    except CreditError:
        raise
    except Exception as e:
        raise CreditError(f"Deduction failed: {e}") from e


def refund_credits(
    jwt: str,
    supabase_url: str,
    amount: int,
    job_id: str,
    reason: str = "",
) -> bool:
    """Refund credits when a job fails before starting.

    Returns True on success. Raises CreditError on failure.
    The server enforces idempotency via job_id -- duplicate refunds
    for the same job_id are no-ops.
    """
    if amount <= 0:
        raise CreditError(f"Refund amount must be positive, got {amount}")

    url = f"{supabase_url}/functions/v1/refund-credits"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }
    body = json.dumps(
        {
            "amount": amount,
            "job_id": job_id,
            "reason": reason,
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data: dict[str, Any] = json.loads(resp.read())
            return bool(data.get("success", False))
    except urllib.error.HTTPError as e:
        body_str = e.read().decode(errors="replace") if e.fp else str(e)
        raise CreditError(f"Refund failed ({e.code}): {body_str}") from e
    except urllib.error.URLError as e:
        raise CreditError(f"Refund failed (network): {e.reason}") from e
    except CreditError:
        raise
    except Exception as e:
        raise CreditError(f"Refund failed: {e}") from e
