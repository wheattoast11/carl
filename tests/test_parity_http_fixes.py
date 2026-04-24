"""Parity-reply v0.18 Track D HTTP fixes — surgical tests.

Covers:
* ``cli/resonant.py::publish_cmd`` — 409 Conflict branch per parity
  reply §Q2 (semantic message on stderr, exit 1, no retry).
* ``a2a/_cli.py::register`` → ``CampSyncClient.register_recipe_shell`` —
  ``Idempotency-Key`` header + single retry on 5xx / timeout per §Q5,
  plus divergence warning when two successful attempts return
  different ``agent_id`` values.

Tests run with Typer's CliRunner and a monkeypatched HTTP transport so
no real network hits happen. ``time.sleep`` inside the retry is patched
to a no-op so the suite stays fast.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner


# ---------------------------------------------------------------------------
# Shared helpers (borrowed shape from test_cli_resonant.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route ~/.carl under ``tmp_path`` so tests don't touch real state."""

    monkeypatch.setenv("HOME", str(tmp_path))
    import importlib

    import carl_studio.resonant_store as mod

    importlib.reload(mod)
    return tmp_path


@pytest.fixture
def runner() -> CliRunner:
    # Newer Click merges stderr into ``result.output`` by default; we
    # assert against the combined stream throughout.
    return CliRunner()


def _make_resonant():  # noqa: ANN202
    from carl_core.eml import EMLNode, EMLOp, EMLTree
    from carl_core.resonant import make_resonant

    tree = EMLTree(
        root=EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
            right=EMLNode(op=EMLOp.CONST, const=1.0),
        ),
        input_dim=1,
    )
    projection = np.array([[1.0]])
    readout = np.array([[1.0]])
    return make_resonant(tree, projection, readout, metadata={"kind": "test"})


def _resonant_app():  # noqa: ANN202
    from carl_studio.cli.resonant import resonant_app

    return resonant_app


# ---------------------------------------------------------------------------
# §Q2 — 409 branch on resonant publish
# ---------------------------------------------------------------------------


class Test409ResonantPublishConflict:
    @pytest.fixture(autouse=True)
    def chdir_and_mkdir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # Project root must NOT equal $HOME (project_context home-guard
        # excludes it). HOME is pinned to tmp_path by isolated_home, so
        # the project lives one level deeper.
        proj = tmp_path / "proj"
        proj.mkdir()
        monkeypatch.chdir(proj)
        (proj / ".carl").mkdir(parents=True, exist_ok=True)
        (proj / "carl.yaml").write_text("name: test\n")

    def test_409_exits_1_with_semantic_message(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """409 → exit 1, semantic message on stderr, no retry attempted.

        Per carl.camp 2026-04-22 (commit fc535e5), the 409 body
        deliberately omits the existing resonant's id — ownership-
        privacy; we don't leak another user's resonant id to a
        hash-collision visitor. The CLI mirrors that intent: no id
        rendered in the error message.
        """
        from carl_studio.resonant_store import save_resonant

        save_resonant("p409", _make_resonant())
        monkeypatch.setenv("CARL_CAMP_TOKEN", "fake-token")

        call_count = {"n": 0}

        def fake_post(
            url: str, body: bytes, headers: dict[str, str], timeout: float = 20.0
        ) -> tuple[int, bytes, dict[str, str]]:
            call_count["n"] += 1
            return (
                409,
                # Body still returned by the server for debugging, but
                # the CLI no longer surfaces it — see privacy note above.
                json.dumps({"error": "content_hash_taken"}).encode(),
                dict[str, str](),
            )

        monkeypatch.setattr("carl_studio.cli.resonant._http_post_bytes", fake_post)

        result = runner.invoke(
            _resonant_app(),
            ["publish", "p409", "--base-url", "https://carl.camp"],
        )
        # Exit 1 = user-error tier (NOT 2 which is reserved for gated/
        # infra-policy refusals like non-HTTPS).
        assert result.exit_code == 1, result.output
        assert "409 Conflict" in result.output
        assert "already published under a different signer" in result.output
        assert "rename" in result.output or "tweak" in result.output
        # No retry — the 409 path is intentionally terminal.
        assert call_count["n"] == 1

    def test_409_with_unparseable_body_still_exits_cleanly(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Body that fails JSON decode must not crash the branch."""
        from carl_studio.resonant_store import save_resonant

        save_resonant("p409b", _make_resonant())
        monkeypatch.setenv("CARL_CAMP_TOKEN", "fake-token")

        def fake_post(
            url: str, body: bytes, headers: dict[str, str], timeout: float = 20.0
        ) -> tuple[int, bytes, dict[str, str]]:
            return 409, b"<html>not json</html>", dict[str, str]()

        monkeypatch.setattr("carl_studio.cli.resonant._http_post_bytes", fake_post)

        result = runner.invoke(
            _resonant_app(),
            ["publish", "p409b", "--base-url", "https://carl.camp"],
        )
        assert result.exit_code == 1
        assert "409 Conflict" in result.output
        assert "already published" in result.output


class Test200ResonantPublishRegression:
    @pytest.fixture(autouse=True)
    def chdir_and_mkdir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # See note on Test409ResonantPublishConflict.chdir_and_mkdir:
        # project root must NOT equal $HOME.
        proj = tmp_path / "proj"
        proj.mkdir()
        monkeypatch.chdir(proj)
        (proj / ".carl").mkdir(parents=True, exist_ok=True)
        (proj / "carl.yaml").write_text("name: test\n")

    def test_200_still_prints_published_status(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Happy path must not regress after adding the 409 branch."""
        from carl_studio.resonant_store import save_resonant

        save_resonant("p200", _make_resonant())
        monkeypatch.setenv("CARL_CAMP_TOKEN", "fake-token")

        def fake_post(
            url: str, body: bytes, headers: dict[str, str], timeout: float = 20.0
        ) -> tuple[int, bytes, dict[str, str]]:
            return (
                200,
                json.dumps({"id": "res_ok", "content_hash": "deadbeef"}).encode(),
                dict[str, str](),
            )

        monkeypatch.setattr("carl_studio.cli.resonant._http_post_bytes", fake_post)

        result = runner.invoke(
            _resonant_app(),
            ["publish", "p200", "--base-url", "https://carl.camp", "--json"],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert data["status"] == "published"
        assert data["resonant_id"] == "res_ok"


# ---------------------------------------------------------------------------
# §Q5 — Idempotency-Key + single retry on 5xx / timeout
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Mimics the shape ``CampSyncClient`` expects from its transport."""

    def __init__(
        self,
        status: int,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._body


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the 1s retry wait so the suite stays fast."""

    import time as _time

    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(_time, "sleep", _noop)


def _expected_idempotency_key(org_id: str | None, name: str, content_hash: str) -> str:
    return hashlib.sha256(f"{org_id or ''}|{name}|{content_hash}".encode("utf-8")).hexdigest()


class TestIdempotencyKeyComputation:
    def test_register_includes_sha256_idempotency_header(
        self, no_sleep: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Header = sha256(org_id|name|content_hash); sent on POST."""
        from carl_studio.a2a.marketplace import CampSyncClient

        captured: dict[str, Any] = {}

        def fake_transport(*, url: str, headers: dict[str, str], json: dict[str, Any]):  # noqa: ANN202, A002
            captured["headers"] = dict(headers)
            captured["url"] = url
            captured["body"] = json
            return _FakeResponse(201, {"ok": True, "agent_id": "srv-uuid-1"})

        client = CampSyncClient(transport=fake_transport)
        key = _expected_idempotency_key("org-a", "agent-x", "ch-abc")
        result = client.register_recipe_shell(
            bearer_token="tok",
            name="agent-x",
            org_id="org-a",
            idempotency_key=key,
        )

        assert result.ok
        assert result.agent_id == "srv-uuid-1"
        assert captured["headers"]["Idempotency-Key"] == key
        assert captured["headers"]["Authorization"] == "Bearer tok"
        assert captured["url"].endswith("/api/agents/register")


class TestRetryOn5xx:
    def test_502_then_200_retries_with_same_key(
        self, no_sleep: None
    ) -> None:
        """First attempt 502 → retry with the SAME key → second 200 succeeds."""
        from carl_studio.a2a.marketplace import CampSyncClient

        seen_keys: list[str | None] = []
        statuses = [502, 200]

        def fake_transport(*, url: str, headers: dict[str, str], json: dict[str, Any]):  # noqa: ANN202, A002
            seen_keys.append(headers.get("Idempotency-Key"))
            status = statuses.pop(0)
            if status == 502:
                return _FakeResponse(502, {"error": "upstream"})
            return _FakeResponse(200, {"ok": True, "agent_id": "srv-uuid-retry"})

        client = CampSyncClient(transport=fake_transport)
        key = _expected_idempotency_key(None, "agent-x", "ch-xyz")
        result = client.register_recipe_shell(
            bearer_token="tok",
            name="agent-x",
            idempotency_key=key,
        )

        assert result.ok
        assert result.agent_id == "srv-uuid-retry"
        # Both attempts carried the SAME key.
        assert seen_keys == [key, key]
        assert result.divergence_warning is None

    def test_502_twice_fails_with_clear_error(
        self, no_sleep: None
    ) -> None:
        """502 on both attempts → terminal, error surfaced via ``error``."""
        from carl_studio.a2a.marketplace import CampSyncClient

        calls = {"n": 0}

        def fake_transport(*, url: str, headers: dict[str, str], json: dict[str, Any]):  # noqa: ANN202, A002
            calls["n"] += 1
            return _FakeResponse(502, {"error": "still upstream"})

        client = CampSyncClient(transport=fake_transport)
        result = client.register_recipe_shell(
            bearer_token="tok",
            name="agent-x",
            idempotency_key="abc",
        )

        assert not result.ok
        assert result.error is not None
        assert "502" in result.error
        # Exactly one retry — total two attempts.
        assert calls["n"] == 2


class TestRetryOnTimeout:
    def test_timeout_retries_once_then_succeeds(
        self, no_sleep: None
    ) -> None:
        """TimeoutError on first attempt → retry → second 200 succeeds."""
        from carl_studio.a2a.marketplace import CampSyncClient

        calls = {"n": 0}

        def fake_transport(*, url: str, headers: dict[str, str], json: dict[str, Any]):  # noqa: ANN202, A002
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("fake socket timeout")
            return _FakeResponse(200, {"ok": True, "agent_id": "srv-timeout-recovered"})

        client = CampSyncClient(transport=fake_transport)
        result = client.register_recipe_shell(
            bearer_token="tok",
            name="agent-x",
            idempotency_key="abc",
        )

        assert result.ok
        assert result.agent_id == "srv-timeout-recovered"
        assert calls["n"] == 2

    def test_oserror_timeout_also_retries(
        self, no_sleep: None
    ) -> None:
        """requests.Timeout is OSError-derived — must also trigger retry."""
        from carl_studio.a2a.marketplace import CampSyncClient

        calls = {"n": 0}

        def fake_transport(*, url: str, headers: dict[str, str], json: dict[str, Any]):  # noqa: ANN202, A002
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate requests.Timeout / socket.timeout shape.
                raise OSError("Connection timed out")
            return _FakeResponse(200, {"ok": True, "agent_id": "recovered-from-oserror"})

        client = CampSyncClient(transport=fake_transport)
        result = client.register_recipe_shell(
            bearer_token="tok",
            name="agent-x",
            idempotency_key="abc",
        )

        assert result.ok
        assert result.agent_id == "recovered-from-oserror"
        assert calls["n"] == 2


class TestDivergenceWarning:
    def test_divergence_surfaced_when_retry_yields_different_agent_id(
        self, no_sleep: None
    ) -> None:
        """Server flakes 502 on attempt 1 while still echoing an id;
        retry succeeds with a DIFFERENT id — warning must surface."""
        from carl_studio.a2a.marketplace import CampSyncClient

        calls = {"n": 0}

        def fake_transport(*, url: str, headers: dict[str, str], json: dict[str, Any]):  # noqa: ANN202, A002
            calls["n"] += 1
            if calls["n"] == 1:
                # 502 status — retry-eligible — but the body leaks an
                # ``agent_id`` (simulates a flaky gateway behind the
                # register service that wrote the row then crashed
                # before the 2xx envelope was sealed).
                return _FakeResponse(502, {"agent_id": "srv-A", "error": "gateway"})
            # Retry returns 200 with a DIFFERENT id — the server didn't
            # honor Idempotency-Key. Surface divergence.
            return _FakeResponse(200, {"ok": True, "agent_id": "srv-B"})

        client = CampSyncClient(transport=fake_transport)
        result = client.register_recipe_shell(
            bearer_token="tok",
            name="agent-x",
            idempotency_key="k",
        )

        assert result.ok
        assert result.agent_id == "srv-B"
        assert result.divergence_warning is not None
        assert "srv-A" in result.divergence_warning
        assert "srv-B" in result.divergence_warning
        assert "using second" in result.divergence_warning
        assert calls["n"] == 2


class TestHappyPathNoRetry:
    def test_first_200_no_retry_no_divergence(
        self, no_sleep: None
    ) -> None:
        """200 on first attempt → no retry, no divergence, clean result."""
        from carl_studio.a2a.marketplace import CampSyncClient

        calls = {"n": 0}

        def fake_transport(*, url: str, headers: dict[str, str], json: dict[str, Any]):  # noqa: ANN202, A002
            calls["n"] += 1
            return _FakeResponse(200, {"ok": True, "agent_id": "srv-fast-path"})

        client = CampSyncClient(transport=fake_transport)
        result = client.register_recipe_shell(
            bearer_token="tok",
            name="agent-x",
            idempotency_key="abc",
        )

        assert result.ok
        assert result.agent_id == "srv-fast-path"
        assert result.divergence_warning is None
        # Exactly one attempt — no retry burned.
        assert calls["n"] == 1


class TestCliRegisterWiring:
    """End-to-end: ``carl agent register`` computes the key and passes it."""

    def test_register_computes_and_forwards_idempotency_key(
        self, no_sleep: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """CLI computes sha256(org|name|content_hash) and forwards it."""
        from carl_studio.a2a._cli import agent_app

        captured: dict[str, Any] = {}

        # Patch the transport factory so register_recipe_shell uses our fake.
        def fake_transport(*, url: str, headers: dict[str, str], json: dict[str, Any]):  # noqa: ANN202, A002
            captured["register_headers"] = dict(headers)
            captured["register_body"] = json
            captured["register_url"] = url
            return _FakeResponse(
                201,
                {"ok": True, "agent_id": "srv-from-cli", "lifecycle_state": "recipe_shell"},
            )

        monkeypatch.setattr(
            "carl_studio.a2a._cli._make_http_transport", lambda: fake_transport
        )
        monkeypatch.setattr(
            "carl_studio.a2a._cli._resolve_bearer_token", lambda: "fake-bearer"
        )

        # Use the real push client only for the sync step — return OK
        # without calling the network.
        from carl_studio.a2a.marketplace import (
            CampSyncClient,
            MarketplaceAgentCard,
            SyncResult,
        )

        def fake_push(
            _self: CampSyncClient,
            cards: list[MarketplaceAgentCard],
            *,
            bearer_token: str,  # noqa: ARG001
            org_id: str | None = None,  # noqa: ARG001
        ) -> SyncResult:
            return SyncResult(synced=len(cards), skipped=0, ids=[c.agent_id for c in cards])

        monkeypatch.setattr(
            "carl_studio.a2a.marketplace.CampSyncClient.push", fake_push
        )

        # Route the LocalDB writes through tmp_path so we don't touch
        # the user's real ~/.carl.
        from carl_studio.a2a.marketplace import compute_content_hash

        agent_name = "test-agent"
        expected_ch = compute_content_hash(
            agent_name=agent_name,
            description=None,
            capabilities=[],
            public_key=None,
            is_active=True,
            rate_limit_rpm=60,
        )
        expected_key = _expected_idempotency_key(None, agent_name, expected_ch)

        runner = CliRunner()
        result = runner.invoke(
            agent_app,
            ["register", agent_name, "--url", "https://carl.camp/agents/test-agent"],
        )

        assert result.exit_code == 0, result.output
        assert "register_headers" in captured, "register route never invoked"
        assert captured["register_headers"].get("Idempotency-Key") == expected_key
