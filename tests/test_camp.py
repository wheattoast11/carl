"""Tests for the shared carl.camp session/profile layer."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from carl_core.errors import CARLError, CredentialError, NetworkError
from typer.testing import CliRunner

from carl_studio.camp import (
    CAMP_PROFILE_TTL_S,
    CampCredentialError,
    CampError,
    CampNetworkError,
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
    """In-memory stand-in for LocalDB — mirrors the get_auth/get_config surface."""

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


class _Clock:
    """Deterministic clock for TTL testing."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_http_response(data: dict[str, Any], status: int = 200) -> MagicMock:
    body = json.dumps(data).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _http_error(status: int, payload: dict[str, Any]) -> urllib.error.HTTPError:
    """Build an HTTPError whose .read() returns a JSON body."""
    body = json.dumps(payload).encode()
    return urllib.error.HTTPError(
        url="http://example.supabase.co/functions/v1/check-tier",
        code=status,
        msg="err",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body),
    )


def _noop_retry(fn, *, policy=None, sleep=None):  # noqa: ARG001
    """Test-only retry substitute that invokes fn() up to max_attempts with no sleep."""
    from carl_core.retry import RetryPolicy

    p = policy or RetryPolicy()
    last: BaseException | None = None
    for attempt in range(1, p.max_attempts + 1):
        try:
            return fn()
        except p.retryable as exc:
            last = exc
            if attempt == p.max_attempts:
                break
    assert last is not None
    raise last


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
        clock = _Clock()
        cache_camp_profile(cached, db=db, clock=clock)

        # Age the cache past TTL so fetch is attempted, force fetch to fail,
        # and verify the stale-serve path returns cache with stale=True.
        clock.advance(CAMP_PROFILE_TTL_S * 2)
        with patch(
            "carl_studio.camp.fetch_camp_profile",
            side_effect=CampNetworkError("offline", code="carl.camp.network"),
        ):
            session, profile, source = resolve_camp_profile(
                refresh=True, db=db, clock=clock
            )

        assert session.authenticated is True
        assert profile is not None
        assert profile.tier == "free"
        assert source == "cache"
        assert profile.stale is True

    def test_cache_round_trip(self) -> None:
        db = FakeDB()
        profile = CampProfile(tier="paid", status="active", credits_remaining=42)
        cache_camp_profile(profile, db=db)

        loaded = load_cached_camp_profile(db=db)
        assert loaded is not None
        assert loaded.tier == "paid"
        assert loaded.credits_remaining == 42
        assert db.get_auth("tier") == "paid"
        # stale defaults to False on freshly-loaded payloads.
        assert loaded.stale is False


class TestCampProfileTTL:
    def test_profile_cache_respects_ttl(self) -> None:
        """Fresh cache (< 24h) skips the network; stale cache triggers fetch."""
        db = FakeDB()
        db.set_auth("jwt", "jwt-tok")
        db.set_config("supabase_url", "https://example.supabase.co")

        clock = _Clock()
        cached = CampProfile(tier="free", status="active", credits_remaining=10)
        cache_camp_profile(cached, db=db, clock=clock)

        fetch_mock = MagicMock()
        with patch("carl_studio.camp.fetch_camp_profile", fetch_mock):
            # Within TTL — cache served, no network call.
            clock.advance(CAMP_PROFILE_TTL_S - 1)
            _, profile, source = resolve_camp_profile(refresh=True, db=db, clock=clock)
            assert source == "cache"
            assert profile is not None
            assert profile.stale is False
            assert fetch_mock.call_count == 0

            # Past TTL — fetch is invoked.
            fetch_mock.return_value = CampProfile(
                tier="paid", status="active", credits_remaining=100
            )
            clock.advance(2)  # now well past TTL
            _, profile, source = resolve_camp_profile(refresh=True, db=db, clock=clock)
            assert source == "remote"
            assert profile is not None
            assert profile.tier == "paid"
            assert fetch_mock.call_count == 1

    def test_force_refresh_ignores_fresh_cache(self) -> None:
        db = FakeDB()
        db.set_auth("jwt", "jwt-tok")
        db.set_config("supabase_url", "https://example.supabase.co")
        clock = _Clock()
        cache_camp_profile(
            CampProfile(tier="free", status="active"), db=db, clock=clock
        )
        fetch_mock = MagicMock(
            return_value=CampProfile(tier="paid", status="active", credits_remaining=5)
        )
        with patch("carl_studio.camp.fetch_camp_profile", fetch_mock):
            _, profile, source = resolve_camp_profile(
                refresh=True, db=db, force_refresh=True, clock=clock
            )
        assert source == "remote"
        assert profile is not None
        assert profile.tier == "paid"
        assert fetch_mock.call_count == 1

    def test_profile_stale_flag_on_fetch_failure(self) -> None:
        """Fetch fails; cache within stale-serve window → stale=True."""
        db = FakeDB()
        db.set_auth("jwt", "jwt-tok")
        db.set_config("supabase_url", "https://example.supabase.co")
        clock = _Clock()
        cache_camp_profile(
            CampProfile(tier="paid", status="active", credits_remaining=50),
            db=db,
            clock=clock,
        )
        # Past TTL but well inside the 7×TTL stale window.
        clock.advance(CAMP_PROFILE_TTL_S * 3)

        with patch(
            "carl_studio.camp.fetch_camp_profile",
            side_effect=CampNetworkError("boom", code="carl.camp.network"),
        ):
            session, profile, source = resolve_camp_profile(
                refresh=True, db=db, clock=clock
            )
        assert source == "cache"
        assert profile is not None
        assert profile.tier == "paid"
        assert profile.stale is True
        assert session.authenticated is True

    def test_no_cache_and_fetch_fails_raises(self) -> None:
        """No cache + fetch failure → NetworkError(code=carl.camp.unreachable)."""
        db = FakeDB()
        db.set_auth("jwt", "jwt-tok")
        db.set_config("supabase_url", "https://example.supabase.co")
        clock = _Clock()

        with patch(
            "carl_studio.camp.fetch_camp_profile",
            side_effect=CampNetworkError("boom", code="carl.camp.network"),
        ):
            with pytest.raises(NetworkError) as exc_info:
                resolve_camp_profile(refresh=True, db=db, clock=clock)
        assert exc_info.value.code == "carl.camp.unreachable"
        # Still catchable as CampError for legacy callers.
        assert isinstance(exc_info.value, CampError)

    def test_cache_beyond_stale_window_and_fetch_fails_raises(self) -> None:
        """Cache older than 7× TTL cannot be served — raise NetworkError."""
        db = FakeDB()
        db.set_auth("jwt", "jwt-tok")
        db.set_config("supabase_url", "https://example.supabase.co")
        clock = _Clock()
        cache_camp_profile(
            CampProfile(tier="free", status="active"), db=db, clock=clock
        )
        # Move past the 7-day stale-serve window.
        clock.advance(CAMP_PROFILE_TTL_S * 8)

        with patch(
            "carl_studio.camp.fetch_camp_profile",
            side_effect=CampNetworkError("boom", code="carl.camp.network"),
        ):
            with pytest.raises(NetworkError) as exc_info:
                resolve_camp_profile(refresh=True, db=db, clock=clock)
        assert exc_info.value.code == "carl.camp.unreachable"


class TestJwtRefresh:
    def test_jwt_401_triggers_refresh(self) -> None:
        """401 → refresh-token exchange → retry original request with new JWT."""
        db = FakeDB()
        db.set_auth("jwt", "old-jwt")
        db.set_auth("refresh_token", "refresh-123")
        db.set_config("supabase_url", "https://example.supabase.co")

        profile_payload = {
            "tier": "paid",
            "status": "active",
            "credits_remaining": 777,
        }
        refresh_payload = {
            "access_token": "new-jwt",
            "refresh_token": "refresh-rotated",
        }

        call_count = {"n": 0}

        def fake_urlopen(req, timeout: int = 10):  # noqa: ARG001
            call_count["n"] += 1
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            if "auth/refresh" in url:
                # Verify refresh request carried the stored token.
                body = req.data.decode() if req.data else ""
                assert "refresh-123" in body
                return _make_http_response(refresh_payload, status=200)
            auth = req.headers.get("Authorization", "")
            if "old-jwt" in auth:
                raise _http_error(401, {"message": "expired"})
            assert "new-jwt" in auth
            return _make_http_response(profile_payload, status=200)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            profile = fetch_camp_profile(
                "old-jwt",
                "https://example.supabase.co",
                db=db,
                refresh_token="refresh-123",
            )

        assert profile.tier == "paid"
        assert profile.credits_remaining == 777
        # JWT + refresh token rotated in LocalDB.
        assert db.get_auth("jwt") == "new-jwt"
        assert db.get_auth("refresh_token") == "refresh-rotated"
        # 3 urlopen calls: first check-tier (401), refresh, retried check-tier.
        assert call_count["n"] == 3

    def test_refresh_failure_raises_credential_error(self) -> None:
        """Refresh endpoint returns 401 → CampCredentialError with login hint."""
        db = FakeDB()
        db.set_auth("jwt", "old-jwt")
        db.set_auth("refresh_token", "bad-refresh")
        db.set_config("supabase_url", "https://example.supabase.co")

        def fake_urlopen(req, timeout: int = 10):  # noqa: ARG001
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            if "auth/refresh" in url:
                raise _http_error(401, {"message": "invalid_grant"})
            raise _http_error(401, {"message": "expired"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with pytest.raises(CredentialError) as exc_info:
                fetch_camp_profile(
                    "old-jwt",
                    "https://example.supabase.co",
                    db=db,
                    refresh_token="bad-refresh",
                )
        assert exc_info.value.code == "carl.camp.auth_expired"
        hint = exc_info.value.context.get("hint", "")
        assert "carl camp login" in str(hint)
        # Multi-inheritance: also a CampError for legacy callers.
        assert isinstance(exc_info.value, CampError)
        assert isinstance(exc_info.value, CampCredentialError)

    def test_jwt_401_without_refresh_token_raises(self) -> None:
        """401 with no stored refresh token → immediate CredentialError."""
        db = FakeDB()
        db.set_auth("jwt", "old-jwt")
        db.set_config("supabase_url", "https://example.supabase.co")

        def fake_urlopen(req, timeout: int = 10):  # noqa: ARG001
            raise _http_error(401, {"message": "expired"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with pytest.raises(CredentialError) as exc_info:
                fetch_camp_profile(
                    "old-jwt",
                    "https://example.supabase.co",
                    db=db,
                )
        assert exc_info.value.code == "carl.camp.auth_expired"


class TestTransientRetry:
    def test_transient_503_retries(self) -> None:
        """503 is retryable — fails twice, then succeeds on the 3rd attempt."""
        attempts = {"n": 0}
        payload = {"tier": "paid", "status": "active", "credits_remaining": 1}

        def fake_urlopen(req, timeout: int = 10):  # noqa: ARG001
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise _http_error(503, {"message": "upstream"})
            return _make_http_response(payload, status=200)

        # Patch the retry helper so we don't actually sleep.
        with patch("urllib.request.urlopen", side_effect=fake_urlopen), patch(
            "carl_studio.camp.retry", side_effect=_noop_retry
        ):
            profile = fetch_camp_profile("jwt", "https://example.supabase.co")

        assert profile.tier == "paid"
        assert attempts["n"] == 3

    def test_network_error_after_retries_raises(self) -> None:
        """Persistent URLError raises CampNetworkError after max_attempts."""

        def fake_urlopen(req, timeout: int = 10):  # noqa: ARG001
            raise urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen), patch(
            "carl_studio.camp.retry", side_effect=_noop_retry
        ):
            with pytest.raises(NetworkError) as exc_info:
                fetch_camp_profile("jwt", "https://example.supabase.co")
        # Also catchable as CampError.
        assert isinstance(exc_info.value, CampError)


class TestTierSync:
    def test_tier_change_updates_localdb(self) -> None:
        """Remote tier change writes through to LocalDB + notifies console."""
        db = FakeDB()
        db.set_auth("jwt", "jwt-tok")
        db.set_auth("tier", "free")  # previously observed tier
        db.set_config("supabase_url", "https://example.supabase.co")
        clock = _Clock()

        console = MagicMock()
        new_profile = CampProfile(tier="paid", status="active", credits_remaining=1)

        with patch("carl_studio.camp.fetch_camp_profile", return_value=new_profile):
            _, profile, source = resolve_camp_profile(
                refresh=True,
                db=db,
                console=console,
                clock=clock,
                force_refresh=True,
            )

        assert source == "remote"
        assert profile is not None
        assert profile.tier == "paid"
        assert db.get_auth("tier") == "paid"
        console.info.assert_called_once()
        assert "paid" in console.info.call_args.args[0]

    def test_same_tier_does_not_notify_console(self) -> None:
        db = FakeDB()
        db.set_auth("jwt", "jwt-tok")
        db.set_auth("tier", "paid")
        db.set_config("supabase_url", "https://example.supabase.co")
        clock = _Clock()

        console = MagicMock()
        with patch(
            "carl_studio.camp.fetch_camp_profile",
            return_value=CampProfile(tier="paid", status="active"),
        ):
            resolve_camp_profile(
                refresh=True,
                db=db,
                console=console,
                clock=clock,
                force_refresh=True,
            )
        console.info.assert_not_called()


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


class TestCampErrorCompat:
    def test_camp_error_still_exported(self) -> None:
        """Legacy callers can still import and raise CampError."""
        err = CampError("legacy")
        assert isinstance(err, Exception)
        assert str(err) == "legacy"

    def test_typed_errors_are_camp_errors(self) -> None:
        """CampNetworkError/CampCredentialError satisfy ``except CampError:``."""
        net = CampNetworkError("offline", code="carl.camp.network")
        cred = CampCredentialError("expired", code="carl.camp.auth_expired")
        assert isinstance(net, CampError)
        assert isinstance(net, NetworkError)
        assert isinstance(net, CARLError)
        assert isinstance(cred, CampError)
        assert isinstance(cred, CredentialError)
        assert isinstance(cred, CARLError)
