"""Tests for carl_studio.billing and carl_studio.billing_cli."""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Minimal Typer app wrapping the billing commands for isolated test invocation.
# We do NOT touch cli.py — build a local app fixture instead.
# ---------------------------------------------------------------------------

import typer

from carl_studio.billing import (
    BILLING_PORTAL_URL,
    CHECKOUT_ANNUAL_URL,
    CHECKOUT_MONTHLY_URL,
    BillingError,
    SubscriptionStatus,
    get_subscription_status,
)
from carl_studio.billing_cli import billing_portal, subscription_status, upgrade


def _make_app() -> typer.Typer:
    a = typer.Typer()
    a.command(name="upgrade")(upgrade)
    a.command(name="billing")(billing_portal)
    a.command(name="subscription")(subscription_status)
    return a


runner = CliRunner()
app = _make_app()


# ---------------------------------------------------------------------------
# SubscriptionStatus unit tests
# ---------------------------------------------------------------------------


class TestSubscriptionStatus:
    def test_defaults(self) -> None:
        s = SubscriptionStatus()
        assert s.tier == "free"
        assert s.plan is None
        assert s.status == "unknown"
        assert s.current_period_end is None
        assert s.cancel_at_period_end is False
        assert s.stripe_customer_id is None

    def test_is_active_paid_true(self) -> None:
        s = SubscriptionStatus(tier="paid", status="active")
        assert s.is_active_paid is True

    def test_is_active_paid_false_wrong_tier(self) -> None:
        s = SubscriptionStatus(tier="free", status="active")
        assert s.is_active_paid is False

    def test_is_active_paid_false_wrong_status(self) -> None:
        s = SubscriptionStatus(tier="paid", status="past_due")
        assert s.is_active_paid is False

    def test_days_remaining_future(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=14)).isoformat()
        s = SubscriptionStatus(current_period_end=future)
        dr = s.days_remaining
        assert dr is not None
        assert 13 <= dr <= 14  # allow rounding

    def test_days_remaining_past(self) -> None:
        past = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        s = SubscriptionStatus(current_period_end=past)
        assert s.days_remaining == 0

    def test_days_remaining_none_when_no_end(self) -> None:
        s = SubscriptionStatus()
        assert s.days_remaining is None

    def test_days_remaining_zulu_suffix(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=7)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        s = SubscriptionStatus(current_period_end=future)
        dr = s.days_remaining
        assert dr is not None
        assert 6 <= dr <= 7

    def test_days_remaining_invalid_string(self) -> None:
        s = SubscriptionStatus(current_period_end="not-a-date")
        assert s.days_remaining is None

    def test_cancel_at_period_end_coercion(self) -> None:
        s = SubscriptionStatus(cancel_at_period_end=1)
        assert s.cancel_at_period_end is True

    def test_stripe_customer_id_none(self) -> None:
        s = SubscriptionStatus(stripe_customer_id=None)
        assert s.stripe_customer_id is None

    def test_stripe_customer_id_set(self) -> None:
        s = SubscriptionStatus(stripe_customer_id="cus_abc123")
        assert s.stripe_customer_id == "cus_abc123"

    def test_monthly_plan(self) -> None:
        s = SubscriptionStatus(tier="paid", plan="monthly", status="active")
        assert s.plan == "monthly"
        assert s.is_active_paid is True

    def test_annual_plan(self) -> None:
        s = SubscriptionStatus(tier="paid", plan="annual", status="active")
        assert s.plan == "annual"

    def test_cancelled_not_active_paid(self) -> None:
        s = SubscriptionStatus(tier="paid", status="cancelled")
        assert s.is_active_paid is False

    def test_to_dict_shape(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        s = SubscriptionStatus(
            tier="paid",
            plan="monthly",
            status="active",
            current_period_end=future,
            cancel_at_period_end=False,
            stripe_customer_id="cus_xyz",
        )
        d = s.to_dict()
        assert d["tier"] == "paid"
        assert d["plan"] == "monthly"
        assert d["status"] == "active"
        assert d["is_active_paid"] is True
        assert isinstance(d["days_remaining"], int)
        assert d["cancel_at_period_end"] is False
        assert d["stripe_customer_id"] == "cus_xyz"


# ---------------------------------------------------------------------------
# get_subscription_status HTTP tests
# ---------------------------------------------------------------------------


def _make_http_response(data: dict[str, Any], status: int = 200) -> MagicMock:
    """Build a mock HTTP response context manager."""
    body = json.dumps(data).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestGetSubscriptionStatus:
    def test_success_returns_status(self) -> None:
        payload = {
            "tier": "paid",
            "plan": "monthly",
            "status": "active",
            "current_period_end": "2026-05-12T00:00:00Z",
            "cancel_at_period_end": False,
            "stripe_customer_id": "cus_abc",
        }
        mock_resp = _make_http_response(payload)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            s = get_subscription_status("jwt-tok", "https://example.supabase.co")

        assert s.tier == "paid"
        assert s.plan == "monthly"
        assert s.status == "active"
        assert s.stripe_customer_id == "cus_abc"
        assert s.is_active_paid is True

    def test_401_raises_billing_error(self) -> None:
        err = urllib.error.HTTPError(
            url="https://example.supabase.co/functions/v1/check-tier",
            code=401,
            msg="Unauthorized",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b"Unauthorized"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(BillingError, match="401"):
                get_subscription_status("bad-jwt", "https://example.supabase.co")

    def test_403_raises_billing_error(self) -> None:
        err = urllib.error.HTTPError(
            url="https://example.supabase.co/functions/v1/check-tier",
            code=403,
            msg="Forbidden",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b"Forbidden"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(BillingError, match="403"):
                get_subscription_status("jwt", "https://example.supabase.co")

    def test_network_error_raises_billing_error(self) -> None:
        err = urllib.error.URLError(reason="Connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(BillingError, match="Network error"):
                get_subscription_status("jwt", "https://example.supabase.co")

    def test_unexpected_error_raises_billing_error(self) -> None:
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            with pytest.raises(BillingError, match="Unexpected error"):
                get_subscription_status("jwt", "https://example.supabase.co")

    def test_free_tier_response(self) -> None:
        payload: dict[str, Any] = {
            "tier": "free",
            "plan": None,
            "status": "unknown",
            "cancel_at_period_end": False,
        }
        mock_resp = _make_http_response(payload)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            s = get_subscription_status("jwt", "https://example.supabase.co")

        assert s.tier == "free"
        assert s.is_active_paid is False


# ---------------------------------------------------------------------------
# Fixtures for CLI tests
# ---------------------------------------------------------------------------


def _patch_effective_tier(tier: str):
    """Context manager: patch billing_cli._effective_tier to return tier."""
    return patch("carl_studio.billing_cli._effective_tier", return_value=tier)


def _patch_db(jwt: str | None, supabase_url: str | None):
    """Patch LocalDB so no ~/.carl/carl.db is accessed."""
    mock_db = MagicMock()
    mock_db.get_auth.return_value = jwt
    mock_db.get_config.return_value = supabase_url
    return patch("carl_studio.billing_cli.LocalDB", return_value=mock_db)


# ---------------------------------------------------------------------------
# carl camp upgrade CLI tests
# ---------------------------------------------------------------------------


class TestUpgradeCommand:
    def test_already_paid_shows_manage_message(self) -> None:
        with _patch_effective_tier("paid"), _patch_db(None, None):
            result = runner.invoke(app, ["upgrade"])
        assert result.exit_code == 0
        assert "Already CARL Paid" in result.output
        assert "carl camp billing" in result.output

    def test_already_paid_no_browser_open(self) -> None:
        with _patch_effective_tier("paid"), _patch_db(None, None), \
                patch("webbrowser.open") as mock_browser:
            runner.invoke(app, ["upgrade"])
        mock_browser.assert_not_called()

    def test_free_shows_feature_list(self) -> None:
        with _patch_effective_tier("free"), \
                patch("webbrowser.open"):
            result = runner.invoke(app, ["upgrade"])
        assert "--send-it" in result.output or "send-it" in result.output or "send_it" in result.output
        assert "carl camp login" in result.output

    def test_free_opens_monthly_by_default(self) -> None:
        with _patch_effective_tier("free"), \
                patch("webbrowser.open") as mock_browser:
            result = runner.invoke(app, ["upgrade"])
        assert result.exit_code == 0
        mock_browser.assert_called_once()
        call_url = mock_browser.call_args[0][0]
        assert "monthly" in call_url or call_url == CHECKOUT_MONTHLY_URL

    def test_free_annual_opens_annual_url(self) -> None:
        with _patch_effective_tier("free"), \
                patch("webbrowser.open") as mock_browser:
            result = runner.invoke(app, ["upgrade", "--annual"])
        assert result.exit_code == 0
        mock_browser.assert_called_once()
        call_url = mock_browser.call_args[0][0]
        assert "annual" in call_url or call_url == CHECKOUT_ANNUAL_URL

    def test_no_open_skips_browser(self) -> None:
        with _patch_effective_tier("free"), \
                patch("webbrowser.open") as mock_browser:
            result = runner.invoke(app, ["upgrade", "--no-open"])
        assert result.exit_code == 0
        mock_browser.assert_not_called()

    def test_already_paid_with_subscription_shows_days(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=12)).isoformat()
        mock_status = SubscriptionStatus(
            tier="paid", status="active", current_period_end=future
        )
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch("carl_studio.billing_cli.get_subscription_status", return_value=mock_status):
            result = runner.invoke(app, ["upgrade"])
        assert "12" in result.output or "days" in result.output

    def test_already_paid_billing_error_graceful(self) -> None:
        """BillingError during status check still shows 'Already Paid'."""
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch(
                    "carl_studio.billing_cli.get_subscription_status",
                    side_effect=BillingError("offline"),
                ):
            result = runner.invoke(app, ["upgrade"])
        assert result.exit_code == 0
        assert "Already CARL Paid" in result.output

    def test_cancel_at_period_end_shows_warning(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        mock_status = SubscriptionStatus(
            tier="paid",
            status="active",
            current_period_end=future,
            cancel_at_period_end=True,
        )
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch("carl_studio.billing_cli.get_subscription_status", return_value=mock_status):
            result = runner.invoke(app, ["upgrade"])
        assert "cancel" in result.output.lower()


# ---------------------------------------------------------------------------
# carl camp billing CLI tests
# ---------------------------------------------------------------------------


class TestBillingPortalCommand:
    def test_not_logged_in_warns(self) -> None:
        with _patch_db(None, None):
            result = runner.invoke(app, ["billing"])
        assert result.exit_code == 1
        assert "Not logged in" in result.output
        assert "carl camp login" in result.output

    def test_logged_in_opens_browser(self) -> None:
        with _patch_db("jwt-tok", None), \
                patch("webbrowser.open") as mock_browser:
            result = runner.invoke(app, ["billing"])
        assert result.exit_code == 0
        mock_browser.assert_called_once_with(BILLING_PORTAL_URL)

    def test_no_open_skips_browser(self) -> None:
        with _patch_db("jwt-tok", None), \
                patch("webbrowser.open") as mock_browser:
            result = runner.invoke(app, ["billing", "--no-open"])
        assert result.exit_code == 0
        mock_browser.assert_not_called()

    def test_shows_url_in_output(self) -> None:
        with _patch_db("jwt-tok", None), \
                patch("webbrowser.open"):
            result = runner.invoke(app, ["billing"])
        assert BILLING_PORTAL_URL in result.output or "billing" in result.output


# ---------------------------------------------------------------------------
# carl subscription CLI tests
# ---------------------------------------------------------------------------


class TestSubscriptionCommand:
    def test_no_jwt_warns_login(self) -> None:
        with _patch_db(None, None):
            result = runner.invoke(app, ["subscription"])
        assert result.exit_code == 0
        assert "Not logged in" in result.output or "carl camp login" in result.output

    def test_no_supabase_url_shows_cached_tier(self) -> None:
        with _patch_db("jwt-tok", None), _patch_effective_tier("free"):
            result = runner.invoke(app, ["subscription"])
        assert result.exit_code == 0
        # Should show something about tier without crashing
        assert "free" in result.output.lower() or "Supabase" in result.output

    def test_billing_error_shows_cached_fallback(self) -> None:
        with _patch_db("jwt-tok", "https://example.supabase.co"), \
                _patch_effective_tier("free"), \
                patch(
                    "carl_studio.billing_cli.get_subscription_status",
                    side_effect=BillingError("network down"),
                ):
            result = runner.invoke(app, ["subscription"])
        assert result.exit_code == 0
        assert "carl.camp" in result.output or "cached" in result.output.lower()
        assert "free" in result.output.lower()

    def test_success_shows_tier_and_plan(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=20)).isoformat()
        mock_status = SubscriptionStatus(
            tier="paid",
            plan="annual",
            status="active",
            current_period_end=future,
        )
        with _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch(
                    "carl_studio.billing_cli.get_subscription_status",
                    return_value=mock_status,
                ):
            result = runner.invoke(app, ["subscription"])
        assert result.exit_code == 0
        out = result.output.lower()
        assert "paid" in out
        assert "annual" in out
        assert "active" in out

    def test_json_output(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        mock_status = SubscriptionStatus(
            tier="paid",
            plan="monthly",
            status="active",
            current_period_end=future,
        )
        with _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch(
                    "carl_studio.billing_cli.get_subscription_status",
                    return_value=mock_status,
                ):
            result = runner.invoke(app, ["subscription", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["tier"] == "paid"
        assert data["plan"] == "monthly"
        assert data["is_active_paid"] is True
        assert isinstance(data["days_remaining"], int)

    def test_json_output_on_billing_error_fallback(self) -> None:
        with _patch_db("jwt-tok", "https://example.supabase.co"), \
                _patch_effective_tier("free"), \
                patch(
                    "carl_studio.billing_cli.get_subscription_status",
                    side_effect=BillingError("offline"),
                ):
            result = runner.invoke(app, ["subscription", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["tier"] == "free"
        assert data["status"] == "cached"

    def test_cancel_at_period_end_shows_warning(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        mock_status = SubscriptionStatus(
            tier="paid",
            status="active",
            current_period_end=future,
            cancel_at_period_end=True,
        )
        with _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch(
                    "carl_studio.billing_cli.get_subscription_status",
                    return_value=mock_status,
                ):
            result = runner.invoke(app, ["subscription"])
        assert "cancel" in result.output.lower()

    def test_free_tier_shows_upgrade_hint(self) -> None:
        mock_status = SubscriptionStatus(tier="free", status="unknown")
        with _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch(
                    "carl_studio.billing_cli.get_subscription_status",
                    return_value=mock_status,
                ):
            result = runner.invoke(app, ["subscription"])
        assert "upgrade" in result.output.lower()

    def test_no_supabase_url_json_output(self) -> None:
        with _patch_db("jwt-tok", None), _patch_effective_tier("free"):
            result = runner.invoke(app, ["subscription", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["tier"] == "free"
