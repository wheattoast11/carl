"""Tests for carl_studio.credits -- balance, estimate, and CLI."""

from __future__ import annotations

import json
import math
import urllib.error
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from carl_studio.credits.balance import (
    CreditBalance,
    CreditError,
    deduct_credits,
    get_credit_balance,
    refund_credits,
)
from carl_studio.credits.estimate import (
    BUNDLES,
    CREDIT_RATES,
    INCLUDED_CREDITS,
    METHOD_STEP_SECONDS,
    CreditEstimate,
    best_bundle,
    estimate_job_cost,
)

# ---------------------------------------------------------------------------
# Minimal Typer app wrapping credits commands for isolated test invocation.
# We do NOT touch cli.py -- build a local app fixture instead.
# ---------------------------------------------------------------------------

from carl_studio.credits._cli import credits_app

runner = CliRunner()


def _make_app() -> typer.Typer:
    """Wrap credits_app in a top-level Typer for testing."""
    a = typer.Typer()
    a.add_typer(credits_app, name="credits")
    return a


app = _make_app()


# ---------------------------------------------------------------------------
# Test helpers
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


def _patch_effective_tier(tier: str):
    return patch("carl_studio.credits._cli._effective_tier", return_value=tier)


def _patch_db(jwt: str | None, supabase_url: str | None):
    """Patch _get_auth to avoid touching ~/.carl/carl.db."""
    return patch(
        "carl_studio.credits._cli._get_auth",
        return_value=(jwt, supabase_url),
    )


# ===========================================================================
# CreditBalance unit tests
# ===========================================================================


class TestCreditBalance:
    def test_defaults(self) -> None:
        b = CreditBalance()
        assert b.total == 0
        assert b.remaining == 0
        assert b.used == 0
        assert b.included_monthly == 0

    def test_sufficient_true(self) -> None:
        b = CreditBalance(remaining=1)
        assert b.sufficient is True

    def test_sufficient_false(self) -> None:
        b = CreditBalance(remaining=0)
        assert b.sufficient is False

    def test_can_afford_exact(self) -> None:
        b = CreditBalance(remaining=50)
        assert b.can_afford(50) is True

    def test_can_afford_more_than_enough(self) -> None:
        b = CreditBalance(remaining=100)
        assert b.can_afford(50) is True

    def test_can_afford_insufficient(self) -> None:
        b = CreditBalance(remaining=10)
        assert b.can_afford(50) is False

    def test_can_afford_zero_cost(self) -> None:
        b = CreditBalance(remaining=0)
        assert b.can_afford(0) is True

    def test_model_dump(self) -> None:
        b = CreditBalance(total=500, remaining=200, used=300, included_monthly=200)
        d = b.model_dump()
        assert d["total"] == 500
        assert d["remaining"] == 200
        assert d["used"] == 300
        assert d["included_monthly"] == 200

    def test_full_lifecycle_values(self) -> None:
        b = CreditBalance(total=2000, remaining=1500, used=500, included_monthly=300)
        assert b.sufficient is True
        assert b.can_afford(1500) is True
        assert b.can_afford(1501) is False


# ===========================================================================
# get_credit_balance HTTP tests
# ===========================================================================


class TestGetCreditBalance:
    def test_success_returns_balance(self) -> None:
        payload = {
            "credits_total": 500,
            "credits_remaining": 350,
            "credits_used": 150,
            "credits_monthly_included": 200,
        }
        mock_resp = _make_http_response(payload)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            b = get_credit_balance("jwt-tok", "https://example.supabase.co")

        assert b.total == 500
        assert b.remaining == 350
        assert b.used == 150
        assert b.included_monthly == 200

    def test_401_raises_credit_error(self) -> None:
        err = urllib.error.HTTPError(
            url="https://example.supabase.co/functions/v1/check-tier",
            code=401,
            msg="Unauthorized",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b"Unauthorized"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(CreditError, match="401"):
                get_credit_balance("bad-jwt", "https://example.supabase.co")

    def test_network_error_raises_credit_error(self) -> None:
        err = urllib.error.URLError(reason="Connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(CreditError, match="Network error"):
                get_credit_balance("jwt", "https://example.supabase.co")

    def test_unexpected_error_raises_credit_error(self) -> None:
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            with pytest.raises(CreditError, match="Unexpected error"):
                get_credit_balance("jwt", "https://example.supabase.co")

    def test_missing_fields_default_to_zero(self) -> None:
        mock_resp = _make_http_response({"tier": "paid"})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            b = get_credit_balance("jwt", "https://example.supabase.co")
        assert b.total == 0
        assert b.remaining == 0
        assert b.used == 0
        assert b.included_monthly == 0


# ===========================================================================
# deduct_credits tests
# ===========================================================================


class TestDeductCredits:
    def test_success_returns_true(self) -> None:
        mock_resp = _make_http_response({"success": True})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = deduct_credits(
                "jwt", "https://example.supabase.co", 50, "job-123", "test"
            )
        assert result is True

    def test_server_returns_false(self) -> None:
        mock_resp = _make_http_response({"success": False})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = deduct_credits(
                "jwt", "https://example.supabase.co", 50, "job-123"
            )
        assert result is False

    def test_http_error_raises(self) -> None:
        err = urllib.error.HTTPError(
            url="",
            code=402,
            msg="Payment Required",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b"Insufficient credits"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(CreditError, match="402"):
                deduct_credits("jwt", "https://example.supabase.co", 50, "job-123")

    def test_network_error_raises(self) -> None:
        err = urllib.error.URLError(reason="Timeout")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(CreditError, match="network"):
                deduct_credits("jwt", "https://example.supabase.co", 50, "job-123")

    def test_zero_amount_raises(self) -> None:
        with pytest.raises(CreditError, match="positive"):
            deduct_credits("jwt", "https://example.supabase.co", 0, "job-123")

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(CreditError, match="positive"):
            deduct_credits("jwt", "https://example.supabase.co", -10, "job-123")


# ===========================================================================
# refund_credits tests
# ===========================================================================


class TestRefundCredits:
    def test_success_returns_true(self) -> None:
        mock_resp = _make_http_response({"success": True})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = refund_credits(
                "jwt", "https://example.supabase.co", 50, "job-123", "early fail"
            )
        assert result is True

    def test_server_returns_false(self) -> None:
        mock_resp = _make_http_response({"success": False})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = refund_credits(
                "jwt", "https://example.supabase.co", 50, "job-123"
            )
        assert result is False

    def test_http_error_raises(self) -> None:
        err = urllib.error.HTTPError(
            url="",
            code=400,
            msg="Bad Request",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b"No deduction found"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(CreditError, match="400"):
                refund_credits("jwt", "https://example.supabase.co", 50, "job-123")

    def test_zero_amount_raises(self) -> None:
        with pytest.raises(CreditError, match="positive"):
            refund_credits("jwt", "https://example.supabase.co", 0, "job-123")


# ===========================================================================
# CreditEstimate and estimate_job_cost tests
# ===========================================================================


class TestCreditEstimate:
    def test_a100_80_steps_grpo_env(self) -> None:
        est = estimate_job_cost("a100-large", 80, 20.0)
        assert est.hardware == "a100-large"
        assert est.rate_per_min == 1.0
        # 80 * 20 = 1600s = 26.67 min -> ceil(26.67) = 27 credits
        assert est.estimated_minutes == pytest.approx(26.7, abs=0.1)
        assert est.estimated_credits == 27
        assert est.total_with_buffer == math.ceil(27 * 1.2)

    def test_l4x1_low_rate(self) -> None:
        est = estimate_job_cost("l4x1", 100, 5.0)
        # 100 * 5 = 500s = 8.33 min. rate=0.3. ceil(0.3*8.33)=ceil(2.5)=3
        assert est.rate_per_min == 0.3
        assert est.estimated_credits == 3
        assert est.total_with_buffer == math.ceil(3 * 1.2)

    def test_h200_high_rate(self) -> None:
        est = estimate_job_cost("h200", 80, 20.0)
        # 80*20=1600s=26.67min. rate=2.5. ceil(2.5*26.67)=ceil(66.67)=67
        assert est.rate_per_min == 2.5
        assert est.estimated_credits == 67
        assert est.total_with_buffer == math.ceil(67 * 1.2)

    def test_local_zero_cost(self) -> None:
        est = estimate_job_cost("local", 200, 10.0)
        assert est.rate_per_min == 0.0
        assert est.estimated_credits == 0
        assert est.total_with_buffer == 0

    def test_unknown_hardware_defaults_to_1(self) -> None:
        est = estimate_job_cost("some-future-gpu", 60, 10.0)
        assert est.rate_per_min == 1.0

    def test_zero_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            estimate_job_cost("a100-large", 0)

    def test_negative_per_step_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            estimate_job_cost("a100-large", 80, -1.0)

    def test_all_hardware_in_credit_rates(self) -> None:
        """Ensure all known hardware flavors are covered."""
        expected = {
            "l4x1", "l4x4", "l40sx1", "l40sx4", "l40sx8",
            "a10g-large", "a10g-largex2", "a10g-largex4",
            "a100-large", "a100-largex2", "a100-largex4", "a100-largex8",
            "h200", "local",
        }
        assert set(CREDIT_RATES.keys()) == expected

    def test_all_bundles_present(self) -> None:
        assert set(BUNDLES.keys()) == {"starter", "explorer", "researcher"}
        for name, info in BUNDLES.items():
            assert "credits" in info
            assert "price_usd" in info
            assert int(info["credits"]) > 0
            assert float(info["price_usd"]) > 0

    def test_bundle_ordering(self) -> None:
        """Bundles should be ordered by credits ascending."""
        credits_list = [int(BUNDLES[n]["credits"]) for n in ("starter", "explorer", "researcher")]
        assert credits_list == sorted(credits_list)

    def test_method_step_seconds_completeness(self) -> None:
        assert set(METHOD_STEP_SECONDS.keys()) == {"sft", "grpo-text", "grpo-env", "grpo-vision"}

    def test_included_credits_plans(self) -> None:
        assert INCLUDED_CREDITS["monthly"] == 200
        assert INCLUDED_CREDITS["annual"] == 300

    def test_estimate_model_dump(self) -> None:
        est = estimate_job_cost("a100-large", 80, 20.0)
        d = est.model_dump()
        assert "hardware" in d
        assert "total_with_buffer" in d
        assert "estimated_usd" in d

    def test_estimated_usd_positive(self) -> None:
        est = estimate_job_cost("a100-large", 80, 20.0)
        assert est.estimated_usd > 0

    def test_buffer_always_gte_raw(self) -> None:
        """total_with_buffer >= estimated_credits for all non-zero rates."""
        for hw in CREDIT_RATES:
            if CREDIT_RATES[hw] == 0.0:
                continue
            est = estimate_job_cost(hw, 80, 20.0)
            assert est.total_with_buffer >= est.estimated_credits


class TestBestBundle:
    def test_small_need_gets_starter(self) -> None:
        assert best_bundle(50) == "starter"

    def test_exact_starter(self) -> None:
        assert best_bundle(100) == "starter"

    def test_medium_need_gets_explorer(self) -> None:
        assert best_bundle(200) == "explorer"

    def test_large_need_gets_researcher(self) -> None:
        assert best_bundle(1000) == "researcher"

    def test_exact_researcher(self) -> None:
        assert best_bundle(2000) == "researcher"

    def test_exceeds_all_returns_none(self) -> None:
        assert best_bundle(5000) is None

    def test_zero_gets_starter(self) -> None:
        assert best_bundle(0) == "starter"


# ===========================================================================
# CLI: carl credits show
# ===========================================================================


class TestCreditsShowCLI:
    def test_no_jwt_shows_login_hint(self) -> None:
        with _patch_effective_tier("paid"), _patch_db(None, None):
            result = runner.invoke(app, ["credits", "show"])
        assert result.exit_code == 0
        assert "Not logged in" in result.output

    def test_free_tier_shows_upgrade(self) -> None:
        with _patch_effective_tier("free"):
            result = runner.invoke(app, ["credits", "show"])
        assert result.exit_code == 0
        assert "carl upgrade" in result.output

    def test_success_renders_balance(self) -> None:
        balance_data = {
            "credits_total": 500,
            "credits_remaining": 350,
            "credits_used": 150,
            "credits_monthly_included": 200,
        }
        mock_resp = _make_http_response(balance_data)
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(app, ["credits", "show"])
        assert result.exit_code == 0
        assert "350" in result.output
        assert "150" in result.output
        assert "500" in result.output

    def test_low_balance_shows_warning(self) -> None:
        balance_data = {
            "credits_total": 100,
            "credits_remaining": 20,
            "credits_used": 80,
            "credits_monthly_included": 0,
        }
        mock_resp = _make_http_response(balance_data)
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(app, ["credits", "show"])
        assert "Low balance" in result.output or "carl credits buy" in result.output

    def test_offline_graceful(self) -> None:
        err = urllib.error.URLError(reason="Connection refused")
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch("urllib.request.urlopen", side_effect=err):
            result = runner.invoke(app, ["credits", "show"])
        assert result.exit_code == 0
        assert "offline" in result.output.lower() or "carl login" in result.output

    def test_json_output(self) -> None:
        balance_data = {
            "credits_total": 500,
            "credits_remaining": 350,
            "credits_used": 150,
            "credits_monthly_included": 200,
        }
        mock_resp = _make_http_response(balance_data)
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(app, ["credits", "show", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["remaining"] == 350


# ===========================================================================
# CLI: carl credits estimate
# ===========================================================================


class TestCreditsEstimateCLI:
    def test_default_renders_table(self) -> None:
        result = runner.invoke(app, ["credits", "estimate"])
        assert result.exit_code == 0
        assert "a100-large" in result.output
        assert "credits/min" in result.output

    def test_custom_hardware_and_steps(self) -> None:
        result = runner.invoke(app, [
            "credits", "estimate",
            "--hardware", "h200",
            "--steps", "100",
            "--method", "grpo-vision",
        ])
        assert result.exit_code == 0
        assert "h200" in result.output

    def test_invalid_method_exits_1(self) -> None:
        result = runner.invoke(app, [
            "credits", "estimate", "--method", "not-real",
        ])
        assert result.exit_code == 1
        assert "Unknown method" in result.output

    def test_json_output(self) -> None:
        result = runner.invoke(app, ["credits", "estimate", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "hardware" in data
        assert "total_with_buffer" in data

    def test_bundle_recommendation_shown(self) -> None:
        result = runner.invoke(app, [
            "credits", "estimate",
            "--hardware", "l4x1",
            "--steps", "10",
            "--method", "sft",
        ])
        assert result.exit_code == 0
        # Small job should recommend starter bundle
        assert "starter" in result.output or "bundle" in result.output.lower()

    def test_sft_method(self) -> None:
        result = runner.invoke(app, [
            "credits", "estimate", "--method", "sft",
        ])
        assert result.exit_code == 0

    def test_grpo_text_method(self) -> None:
        result = runner.invoke(app, [
            "credits", "estimate", "--method", "grpo-text",
        ])
        assert result.exit_code == 0


# ===========================================================================
# CLI: carl credits buy
# ===========================================================================


class TestCreditsBuyCLI:
    def test_valid_bundle_opens_browser(self) -> None:
        with _patch_effective_tier("paid"), \
                patch("webbrowser.open") as mock_browser:
            result = runner.invoke(app, ["credits", "buy", "starter"])
        assert result.exit_code == 0
        mock_browser.assert_called_once()
        call_url = mock_browser.call_args[0][0]
        assert "bundle=starter" in call_url

    def test_default_bundle_is_explorer(self) -> None:
        with _patch_effective_tier("paid"), \
                patch("webbrowser.open") as mock_browser:
            result = runner.invoke(app, ["credits", "buy"])
        assert result.exit_code == 0
        call_url = mock_browser.call_args[0][0]
        assert "bundle=explorer" in call_url

    def test_researcher_bundle(self) -> None:
        with _patch_effective_tier("paid"), \
                patch("webbrowser.open") as mock_browser:
            result = runner.invoke(app, ["credits", "buy", "researcher"])
        assert result.exit_code == 0
        assert "$120" in result.output or "120" in result.output
        assert "2000" in result.output

    def test_invalid_bundle_exits_1(self) -> None:
        with _patch_effective_tier("paid"):
            result = runner.invoke(app, ["credits", "buy", "mega"])
        assert result.exit_code == 1
        assert "Unknown bundle" in result.output

    def test_no_open_skips_browser(self) -> None:
        with _patch_effective_tier("paid"), \
                patch("webbrowser.open") as mock_browser:
            result = runner.invoke(app, ["credits", "buy", "starter", "--no-open"])
        assert result.exit_code == 0
        mock_browser.assert_not_called()

    def test_free_tier_blocked(self) -> None:
        with _patch_effective_tier("free"):
            result = runner.invoke(app, ["credits", "buy", "starter"])
        assert result.exit_code == 0
        assert "carl upgrade" in result.output

    def test_shows_per_credit_price(self) -> None:
        with _patch_effective_tier("paid"), \
                patch("webbrowser.open"):
            result = runner.invoke(app, ["credits", "buy", "starter"])
        assert "Per credit" in result.output or "$0.080" in result.output


# ===========================================================================
# CLI: carl credits history
# ===========================================================================


class TestCreditsHistoryCLI:
    def test_not_logged_in(self) -> None:
        with _patch_effective_tier("paid"), _patch_db(None, None):
            result = runner.invoke(app, ["credits", "history"])
        assert result.exit_code == 0
        assert "carl login" in result.output

    def test_success_renders_table(self) -> None:
        history_data = {
            "transactions": [
                {
                    "type": "deduction",
                    "amount": 50,
                    "job_id": "job-abc123",
                    "balance_after": 450,
                    "created_at": "2026-04-12T10:00:00Z",
                },
                {
                    "type": "purchase",
                    "amount": 500,
                    "job_id": None,
                    "balance_after": 500,
                    "created_at": "2026-04-10T08:00:00Z",
                },
            ]
        }
        mock_resp = _make_http_response(history_data)
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(app, ["credits", "history"])
        assert result.exit_code == 0
        assert "deduction" in result.output or "purchase" in result.output

    def test_empty_history(self) -> None:
        mock_resp = _make_http_response({"transactions": []})
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(app, ["credits", "history"])
        assert result.exit_code == 0
        assert "No transactions" in result.output

    def test_json_output(self) -> None:
        history_data = {
            "transactions": [
                {"type": "purchase", "amount": 100, "created_at": "2026-04-12T10:00:00Z"},
            ]
        }
        mock_resp = _make_http_response(history_data)
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(app, ["credits", "history", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["type"] == "purchase"

    def test_offline_graceful(self) -> None:
        err = urllib.error.URLError(reason="Connection refused")
        with _patch_effective_tier("paid"), \
                _patch_db("jwt-tok", "https://example.supabase.co"), \
                patch("urllib.request.urlopen", side_effect=err):
            result = runner.invoke(app, ["credits", "history"])
        assert result.exit_code == 0
        assert "offline" in result.output.lower() or "cannot" in result.output.lower()

    def test_free_tier_blocked(self) -> None:
        with _patch_effective_tier("free"):
            result = runner.invoke(app, ["credits", "history"])
        assert result.exit_code == 0
        assert "carl upgrade" in result.output
