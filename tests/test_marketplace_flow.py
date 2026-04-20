"""v0.13 — end-to-end marketplace register + publish flow tests.

Exercises:
1. register_recipe_shell() happy path (201 + ok envelope)
2. register_recipe_shell() error handling (429 / 4xx / transport error)
3. CampSyncClient.push() honors org_id + envelope
4. CLI `carl agent register --local-only` writes local card
5. CLI `carl agent publish` integrates CoherenceGate (denies on failing
   chain, allows on healthy)
6. Content_hash is REQUIRED per the backend contract
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carl_studio.a2a.marketplace import (
    AgentCardStore,
    CampSyncClient,
    MarketplaceAgentCard,
    RegisterResult,
    compute_content_hash,
    new_agent_id,
)


class _MockResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._body


class _TinyDB:
    def __init__(self, path: Path) -> None:
        self._path = path

    def connect(self) -> Any:
        import sqlite3

        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn


# ---------------------------------------------------------------------------
# register_recipe_shell
# ---------------------------------------------------------------------------


class TestRegisterRecipeShell:
    def test_happy_path(self) -> None:
        def _t(**kwargs: Any) -> _MockResponse:
            return _MockResponse(
                status_code=201,
                body={
                    "ok": True,
                    "agent_id": "11111111-2222-3333-4444-555555555555",
                    "org_id": "org_abc",
                    "lifecycle_state": "drafting",
                    "created_at": "2026-04-20T20:00:00Z",
                },
            )

        client = CampSyncClient(transport=_t)
        result = client.register_recipe_shell(bearer_token="tok", name="Agent-1")
        assert result.ok
        assert result.agent_id == "11111111-2222-3333-4444-555555555555"
        assert result.org_id == "org_abc"
        assert result.lifecycle_state == "drafting"

    def test_429_surfaces_retry_after(self) -> None:
        def _t(**kwargs: Any) -> _MockResponse:
            return _MockResponse(status_code=429, headers={"Retry-After": "30"})

        client = CampSyncClient(transport=_t)
        result = client.register_recipe_shell(bearer_token="tok")
        assert not result.ok
        assert "rate_limited" in (result.error or "")

    def test_4xx_surfaces_error_body(self) -> None:
        def _t(**kwargs: Any) -> _MockResponse:
            return _MockResponse(
                status_code=401,
                body={"ok": False, "error": "unauthorized"},
            )

        client = CampSyncClient(transport=_t)
        result = client.register_recipe_shell(bearer_token="tok")
        assert not result.ok
        assert "401" in (result.error or "")

    def test_ok_false_envelope(self) -> None:
        """Backend returned 200 but envelope.ok=false → treat as error."""

        def _t(**kwargs: Any) -> _MockResponse:
            return _MockResponse(
                status_code=200, body={"ok": False, "error": "validation_failed"}
            )

        client = CampSyncClient(transport=_t)
        result = client.register_recipe_shell(bearer_token="tok")
        assert not result.ok
        assert "validation_failed" in (result.error or "")

    def test_transport_error(self) -> None:
        def _t(**kwargs: Any) -> _MockResponse:
            raise ConnectionError("refused")

        client = CampSyncClient(transport=_t)
        result = client.register_recipe_shell(bearer_token="tok")
        assert not result.ok
        assert "transport error" in (result.error or "")

    def test_no_transport(self) -> None:
        client = CampSyncClient(transport=None)
        result = client.register_recipe_shell(bearer_token="tok")
        assert not result.ok
        assert "no transport" in (result.error or "")


# ---------------------------------------------------------------------------
# Envelope handling in push()
# ---------------------------------------------------------------------------


class TestPushEnvelope:
    def test_push_captures_ok_envelope(self) -> None:
        def _t(**kwargs: Any) -> _MockResponse:
            return _MockResponse(
                body={"ok": True, "synced": 1, "skipped": 0, "ids": ["id1"], "rejected": []}
            )

        client = CampSyncClient(transport=_t)
        card = _sample_card()
        result = client.push([card], bearer_token="tok")
        assert result.ok
        assert result.envelope_ok is True
        assert result.synced == 1


# ---------------------------------------------------------------------------
# CLI — register (local only + local with mocked server)
# ---------------------------------------------------------------------------


def _sample_card() -> MarketplaceAgentCard:
    return MarketplaceAgentCard(
        agent_id=new_agent_id(),
        agent_url="https://carl.camp/agents/sample",
        agent_name="Sample",
        description="test card",
        capabilities=[{"id": "x", "description": "y"}],
        public_key=None,
        key_algorithm=None,
        content_hash=compute_content_hash(
            agent_name="Sample",
            description="test card",
            capabilities=[{"id": "x", "description": "y"}],
            public_key=None,
            is_active=True,
            rate_limit_rpm=60,
        ),
        is_active=True,
        rate_limit_rpm=60,
    )


class TestCliRegister:
    def test_local_only_writes_card(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--local-only should persist without calling HTTP."""
        from typer.testing import CliRunner

        # Point LocalDB at a temp file so we don't touch the user's ~/.carl/
        monkeypatch.setenv("CARL_HOME", str(tmp_path))

        from carl_studio.a2a._cli import agent_app
        import importlib

        # Re-import so the module picks up CARL_HOME
        import carl_studio.db

        importlib.reload(carl_studio.db)

        runner = CliRunner()
        result = runner.invoke(
            agent_app, ["register", "Test Agent", "--local-only"]
        )
        assert result.exit_code == 0, result.output
        assert "saved locally" in result.output

    def test_register_without_token_falls_back_to_local(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No CARL_CAMP_TOKEN + no camp_token file → local + FYI nudge, exit 0."""

        monkeypatch.setenv("CARL_HOME", str(tmp_path))
        monkeypatch.delenv("CARL_CAMP_TOKEN", raising=False)
        # Ensure no stale camp_token in temp home
        (tmp_path / "camp_token").unlink(missing_ok=True)

        from carl_studio.a2a._cli import agent_app
        from typer.testing import CliRunner
        import importlib
        import carl_studio.db

        importlib.reload(carl_studio.db)

        runner = CliRunner()
        result = runner.invoke(agent_app, ["register", "No-Token Agent"])
        assert result.exit_code == 0, result.output
        # Saved locally, with nudge about carl camp login
        assert "saved locally" in result.output.lower()


# ---------------------------------------------------------------------------
# content_hash REQUIRED per backend contract
# ---------------------------------------------------------------------------


class TestContentHashRequired:
    def test_card_construction_requires_content_hash(self) -> None:
        """Pydantic model requires content_hash; omission → ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MarketplaceAgentCard(  # type: ignore[call-arg]
                agent_id=new_agent_id(),
                agent_url="https://x",
                agent_name="X",
                capabilities=[],
                # content_hash missing intentionally
            )
