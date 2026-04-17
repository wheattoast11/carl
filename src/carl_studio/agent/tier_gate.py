"""Agent tier gating — delegates to carl_studio.tier for subscription truth."""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from carl_studio.tier import (
    Tier,
    TierGateError,
    detect_effective_tier,
    tier_allows,
)

logger = logging.getLogger(__name__)

# Backwards-compat alias so existing ``from carl_studio.agent import TierError``
# continues to work without changes at call sites.
TierError = TierGateError


def _effective_tier() -> Tier:
    """Resolve the current effective tier via the canonical system."""
    from carl_studio.settings import CARLSettings

    settings = CARLSettings.load()
    return detect_effective_tier(settings.tier)


def get_tier() -> str:
    """Return the effective tier as a lowercase string.

    Delegates to :func:`carl_studio.tier.detect_effective_tier` which reads
    from the canonical SQLite subscription cache — not an env var.
    """
    return _effective_tier().value


def requires_paid(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that gates a function behind PAID tier.

    Raises :class:`TierGateError` (aliased as ``TierError``) when the
    effective tier does not satisfy the ``experiment`` feature gate.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        current = _effective_tier()
        if not tier_allows(current, "experiment"):
            raise TierGateError(
                feature="experiment",
                required=Tier.PAID,
                current=current,
            )
        return func(*args, **kwargs)

    return wrapper
