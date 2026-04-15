"""Tests for the shared carl.camp session/profile layer."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from carl_studio.camp import (
    CampError,
    CampProfile,
    CampSession,
    cache_camp_profile,
    fetch_camp_profile,
    load_cached_camp_profile,
    resolve_camp_profile,
)
from carl_studio.cli.billing import account_status


def _make_app() -> typer.Typer:
    test_app = typer.Typer()
    camp_app = typer.Typer()
    camp_app.command(name="account")(account_status)
    test_app.add_typer(camp_app, name="camp")
    return test_app


runner = CliRunner()
app = _make_app()


class FakeDB:
    def __init__(self) -> None:
        self.auth: dict[str, str] = {}
        self.config: dict[str, str] = {}

    def get_auth(self, key: str) -> str | None:
        return self.auth.get(key)

    def set_auth(self, key: str, value: str, ttl_hours: int = 24) -> None:
        self.auth[key] = value

    def get_config(self, key: str, default: str | None = None) -> str | None:
        return self.config.get(key, default)

    def set_config(self, key: str, value: str) -> None:
        self.config[key] = value


def _make_http_response(data: dict[str, Any]) -> MagicMock:
    body = json.dumps(data).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestCampProfile:
    def test_fetch_camp_profile_parses_capabilities(self) -> None:
        payload = {
            "tier": "paid",
            "plan": "annual",
            "status": "active",
            "credits_total": 1200,
            "credits_remaining": 900,
            "credits_used": 300,
            "credits_monthly_included": 300,
            "payment_methods": ["card", "wallet"],
            "wallet_auth_enabled": True,
            "x402_enabled": True,
            "observability_opt_in": True,
            "telemetry_opt_in": True,
            "usage_tracking_enabled": False,
            "contract_witnessing": True,
            "contract_terms_url": "https://carl.camp/terms/agent",
        }
        with patch("urllib.request.urlopen", return_value=_make_http_response(payload)):
            profile = fetch_camp_profile("jwt-tok", "https://example.supabase.co")

        assert profile.tier == "paid"
        assert profile.plan == "annual"
        assert profile.payment_summary == "card, wallet"
        assert profile.wallet_auth_enabled is True
        assert profile.x402_enabled is True
        assert profile.contract_witnessing is True

    def test_resolve_camp_profile_falls_back_to_cache(self) -> None:
        db = FakeDB()
        db.set_auth("jwt", "jwt-tok")
        db.set_config("supabase_url", "https://example.supabase.co")
        cached = CampProfile(tier="free", status="cached", credits_remaining=0)
        cache_camp_profile(cached, db=db)

        with patch("carl_studio.camp.fetch_camp_profile", side_effect=CampError("offline")):
            session, profile, source = resolve_camp_profile(refresh=True, db=db)

        assert session.authenticated is True
        assert profile is not None
        assert profile.tier == "free"
        assert source == "cache"

    def test_cache_round_trip(self) -> None:
        db = FakeDB()
        profile = CampProfile(tier="paid", status="active", credits_remaining=42)
        cache_camp_profile(profile, db=db)

        loaded = load_cached_camp_profile(db=db)
        assert loaded is not None
        assert loaded.tier == "paid"
        assert loaded.credits_remaining == 42
        assert db.get_auth("tier") == "paid"


class TestCampAccountCLI:
    def test_account_json_uses_resolved_profile(self) -> None:
        profile = CampProfile(
            tier="paid", status="active", credits_remaining=250, x402_enabled=True
        )
        session = CampSession(
            jwt="jwt-tok", supabase_url="https://example.supabase.co", cached_tier="paid"
        )
        with patch(
            "carl_studio.cli.billing.resolve_camp_profile",
            return_value=(session, profile, "remote"),
        ):
            result = runner.invoke(app, ["camp", "account", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["authenticated"] is True
        assert data["source"] == "remote"
        assert data["profile"]["tier"] == "paid"
        assert data["profile"]["x402_enabled"] is True
