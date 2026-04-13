"""Tier gating for PAID-only features."""
from __future__ import annotations

import functools
import os
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class TierError(Exception):
    """Raised when a PAID feature is accessed without subscription."""

    pass


def get_tier() -> str:
    """Get current subscription tier.

    In production, this checks the local subscription cache.
    For now, reads CARL_TIER env var (default: "free").
    """
    return os.environ.get("CARL_TIER", "free").lower()


def requires_paid(func: F) -> F:
    """Decorator that gates a function behind PAID subscription.

    Raises TierError if tier is "free".
    Set CARL_TIER=paid to bypass for testing.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tier = get_tier()
        if tier == "free":
            raise TierError(
                f"{func.__name__} requires a PAID subscription. "
                f"Upgrade at: https://carl.camp/upgrade"
            )
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
