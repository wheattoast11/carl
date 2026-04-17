"""Unit tests for carl_core.tier primitives.

Covers the pure, dependency-free surface: enum values, aliases, ordering,
feature lookup, and gate error. Settings-backed behaviour (effective tier,
tier_gate decorator) is exercised in carl-studio's test_settings.py and
test_uat.py.
"""

from __future__ import annotations

import pytest

from carl_core.tier import (
    FEATURE_TIERS,
    Tier,
    TierGateError,
    feature_tier,
    tier_allows,
)


# ---------------------------------------------------------------------------
# Enum values and aliases
# ---------------------------------------------------------------------------


class TestTierValues:
    def test_free_value(self) -> None:
        assert Tier.FREE.value == "free"

    def test_paid_value(self) -> None:
        assert Tier.PAID.value == "paid"

    def test_pro_is_paid_alias(self) -> None:
        assert Tier.PRO is Tier.PAID
        assert Tier.PRO.value == "paid"

    def test_enterprise_is_paid_alias(self) -> None:
        assert Tier.ENTERPRISE is Tier.PAID
        assert Tier.ENTERPRISE.value == "paid"

    def test_string_behavior(self) -> None:
        # Tier inherits from str, so equality with the raw string works.
        assert Tier.FREE == "free"
        assert Tier.PAID == "paid"

    def test_from_string(self) -> None:
        assert Tier("free") is Tier.FREE
        assert Tier("paid") is Tier.PAID

    def test_from_string_invalid(self) -> None:
        with pytest.raises(ValueError):
            Tier("platinum")


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestTierOrdering:
    def test_free_less_than_paid(self) -> None:
        assert Tier.FREE < Tier.PAID

    def test_paid_greater_than_free(self) -> None:
        assert Tier.PAID > Tier.FREE

    def test_free_le_free(self) -> None:
        assert Tier.FREE <= Tier.FREE

    def test_paid_ge_paid(self) -> None:
        assert Tier.PAID >= Tier.PAID

    def test_paid_ge_free(self) -> None:
        assert Tier.PAID >= Tier.FREE

    def test_free_not_greater_than_paid(self) -> None:
        assert not (Tier.FREE > Tier.PAID)

    def test_paid_not_less_than_free(self) -> None:
        assert not (Tier.PAID < Tier.FREE)

    def test_pro_equals_paid_for_ordering(self) -> None:
        # Aliases compare equal because they ARE the same enum member.
        assert Tier.PRO == Tier.PAID
        assert Tier.ENTERPRISE == Tier.PAID
        assert Tier.PRO >= Tier.ENTERPRISE
        assert Tier.PRO <= Tier.ENTERPRISE

    def test_comparison_with_non_tier_returns_notimplemented(self) -> None:
        # Python routes NotImplemented through reflected operators; end
        # result is TypeError when both sides decline.
        with pytest.raises(TypeError):
            _ = Tier.FREE < 1  # type: ignore[operator]
        with pytest.raises(TypeError):
            _ = Tier.FREE > object()  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


class TestFeatureRegistry:
    def test_registry_is_dict(self) -> None:
        assert isinstance(FEATURE_TIERS, dict)
        assert len(FEATURE_TIERS) > 0

    def test_free_features_include_core_loop(self) -> None:
        assert FEATURE_TIERS["observe"] is Tier.FREE
        assert FEATURE_TIERS["train"] is Tier.FREE
        assert FEATURE_TIERS["eval"] is Tier.FREE
        assert FEATURE_TIERS["bench"] is Tier.FREE
        assert FEATURE_TIERS["align"] is Tier.FREE
        assert FEATURE_TIERS["learn"] is Tier.FREE
        assert FEATURE_TIERS["push"] is Tier.FREE
        assert FEATURE_TIERS["bundle"] is Tier.FREE

    def test_paid_features_include_autonomy(self) -> None:
        assert FEATURE_TIERS["train.send_it"] is Tier.PAID
        assert FEATURE_TIERS["train.auto_gate"] is Tier.PAID
        assert FEATURE_TIERS["train.scheduled"] is Tier.PAID
        assert FEATURE_TIERS["mcp"] is Tier.PAID
        assert FEATURE_TIERS["mcp.serve"] is Tier.PAID
        assert FEATURE_TIERS["dashboard"] is Tier.PAID
        assert FEATURE_TIERS["sync.cloud"] is Tier.PAID
        assert FEATURE_TIERS["marketplace.publish"] is Tier.PAID

    def test_feature_tier_known_free(self) -> None:
        assert feature_tier("observe") is Tier.FREE

    def test_feature_tier_known_paid(self) -> None:
        assert feature_tier("train.send_it") is Tier.PAID

    def test_feature_tier_unknown_defaults_to_free(self) -> None:
        # Permissive by default.
        assert feature_tier("not-a-real-feature") is Tier.FREE
        assert feature_tier("") is Tier.FREE


# ---------------------------------------------------------------------------
# tier_allows
# ---------------------------------------------------------------------------


class TestTierAllows:
    def test_free_allows_free_feature(self) -> None:
        assert tier_allows(Tier.FREE, "observe") is True

    def test_free_does_not_allow_paid_feature(self) -> None:
        assert tier_allows(Tier.FREE, "train.send_it") is False

    def test_paid_allows_free_feature(self) -> None:
        assert tier_allows(Tier.PAID, "observe") is True

    def test_paid_allows_paid_feature(self) -> None:
        assert tier_allows(Tier.PAID, "train.send_it") is True

    def test_unknown_feature_permits_free(self) -> None:
        assert tier_allows(Tier.FREE, "something.new") is True

    def test_pro_alias_behaves_like_paid(self) -> None:
        assert tier_allows(Tier.PRO, "train.send_it") is True

    def test_enterprise_alias_behaves_like_paid(self) -> None:
        assert tier_allows(Tier.ENTERPRISE, "dashboard") is True


# ---------------------------------------------------------------------------
# TierGateError
# ---------------------------------------------------------------------------


class TestTierGateError:
    def test_error_is_exception(self) -> None:
        err = TierGateError("mcp.serve", Tier.PAID, Tier.FREE)
        assert isinstance(err, Exception)

    def test_error_records_fields(self) -> None:
        err = TierGateError("mcp.serve", Tier.PAID, Tier.FREE)
        assert err.feature == "mcp.serve"
        assert err.required is Tier.PAID
        assert err.current is Tier.FREE

    def test_error_message_mentions_feature_and_tiers(self) -> None:
        err = TierGateError("train.send_it", Tier.PAID, Tier.FREE)
        msg = str(err)
        assert "train.send_it" in msg
        assert "Paid" in msg
        assert "Free" in msg

    def test_error_message_contains_upgrade_url(self) -> None:
        err = TierGateError("mcp", Tier.PAID, Tier.FREE)
        assert "carl.camp/pricing" in str(err)
