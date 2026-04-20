"""v0.10 W7 — marketplace agent card (MarketplaceAgentCard + AgentCardStore + CampSyncClient).

Contract pinned here matches ``docs/v10_agent_card_supabase_spec.md``
as updated with the backend-side agent's 2026-04-20 clarifications.

Covered:
1. compute_content_hash is deterministic and excludes timestamps/ids.
2. MarketplaceAgentCard.to_sync_payload() drops local-only fields.
3. AgentCardStore round-trip (save, get, list, delete).
4. AgentCardStore upsert (last-write-wins on agent_id).
5. CampSyncClient happy path (200 → synced/skipped/ids/rejected).
6. CampSyncClient batch-limit enforcement (>100 → error).
7. CampSyncClient 429 rate-limit surfaced.
8. CampSyncClient transport error surfaced.
9. CampSyncClient empty batch is a no-op.
10. org_id, when provided, appears in the POST body.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carl_studio.a2a.marketplace import (
    AgentCardStore,
    CampSyncClient,
    MarketplaceAgentCard,
    SyncResult,
    compute_content_hash,
    new_agent_id,
)


# ---------------------------------------------------------------------------
# Minimal LocalDB stand-in — just a sqlite3 wrapper with .connect()
# ---------------------------------------------------------------------------


class _TinyDB:
    def __init__(self, path: Path) -> None:
        self._path = path

    def connect(self) -> Any:
        import sqlite3

        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn


@pytest.fixture
def db(tmp_path: Path) -> _TinyDB:
    return _TinyDB(tmp_path / "marketplace.db")


@pytest.fixture
def sample_card() -> MarketplaceAgentCard:
    caps = [
        {"id": "search", "description": "semantic arxiv search"},
        {"id": "synthesize", "description": "markdown synthesis"},
    ]
    return MarketplaceAgentCard(
        agent_id=new_agent_id(),
        agent_url="https://carl.camp/agents/my-researcher",
        agent_name="My Research Agent",
        description="Reads papers, cites correctly.",
        capabilities=caps,
        public_key="a" * 64,
        key_algorithm="ed25519",
        content_hash=compute_content_hash(
            agent_name="My Research Agent",
            description="Reads papers, cites correctly.",
            capabilities=caps,
            public_key="a" * 64,
            is_active=True,
            rate_limit_rpm=60,
        ),
        is_active=True,
        rate_limit_rpm=60,
    )


# ---------------------------------------------------------------------------
# 1. content_hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self) -> None:
        args = dict(
            agent_name="A",
            description="d",
            capabilities=[{"id": "x"}],
            public_key=None,
            is_active=True,
            rate_limit_rpm=60,
        )
        assert compute_content_hash(**args) == compute_content_hash(**args)

    def test_key_order_invariant(self) -> None:
        # Same semantic card via different key order → same hash
        a = compute_content_hash(
            agent_name="A",
            description="d",
            capabilities=[{"id": "x", "description": "y"}],
            public_key=None,
            is_active=True,
            rate_limit_rpm=60,
        )
        b = compute_content_hash(
            agent_name="A",
            description="d",
            capabilities=[{"description": "y", "id": "x"}],  # flipped
            public_key=None,
            is_active=True,
            rate_limit_rpm=60,
        )
        assert a == b

    def test_field_change_changes_hash(self) -> None:
        baseline = compute_content_hash(
            agent_name="A",
            description="d",
            capabilities=[],
            public_key=None,
            is_active=True,
            rate_limit_rpm=60,
        )
        altered = compute_content_hash(
            agent_name="A-changed",  # <--
            description="d",
            capabilities=[],
            public_key=None,
            is_active=True,
            rate_limit_rpm=60,
        )
        assert baseline != altered


# ---------------------------------------------------------------------------
# 2. to_sync_payload
# ---------------------------------------------------------------------------


class TestSyncPayload:
    def test_drops_local_only_fields(self, sample_card: MarketplaceAgentCard) -> None:
        payload = sample_card.to_sync_payload()
        # Local-only fields must NOT appear
        assert "created_at" not in payload
        assert "updated_at" not in payload
        assert "key_algorithm" not in payload  # stays local until server schema adds it
        # Required fields present
        for key in (
            "agent_id",
            "agent_url",
            "agent_name",
            "capabilities",
            "content_hash",
            "is_active",
            "rate_limit_rpm",
        ):
            assert key in payload


# ---------------------------------------------------------------------------
# 3. AgentCardStore round-trip
# ---------------------------------------------------------------------------


class TestAgentCardStore:
    def test_save_and_get_round_trip(
        self, db: _TinyDB, sample_card: MarketplaceAgentCard
    ) -> None:
        store = AgentCardStore(db)
        store.save(sample_card)
        loaded = store.get(sample_card.agent_id)
        assert loaded is not None
        assert loaded.agent_name == sample_card.agent_name
        assert loaded.capabilities == sample_card.capabilities
        assert loaded.content_hash == sample_card.content_hash

    def test_get_missing_returns_none(self, db: _TinyDB) -> None:
        store = AgentCardStore(db)
        assert store.get("nonexistent") is None

    def test_list_returns_saved_cards(
        self, db: _TinyDB, sample_card: MarketplaceAgentCard
    ) -> None:
        store = AgentCardStore(db)
        store.save(sample_card)
        listed = store.list_all()
        assert len(listed) == 1
        assert listed[0].agent_id == sample_card.agent_id

    def test_delete_removes(
        self, db: _TinyDB, sample_card: MarketplaceAgentCard
    ) -> None:
        store = AgentCardStore(db)
        store.save(sample_card)
        assert store.delete(sample_card.agent_id) is True
        assert store.get(sample_card.agent_id) is None

    def test_upsert_on_same_agent_id(
        self, db: _TinyDB, sample_card: MarketplaceAgentCard
    ) -> None:
        store = AgentCardStore(db)
        store.save(sample_card)

        # Modify and re-save with the same agent_id
        modified = sample_card.model_copy(
            update={"agent_name": "Renamed Agent", "content_hash": "new-hash"}
        )
        store.save(modified)

        # Only one row, updated
        all_cards = store.list_all()
        assert len(all_cards) == 1
        assert all_cards[0].agent_name == "Renamed Agent"
        assert all_cards[0].content_hash == "new-hash"


# ---------------------------------------------------------------------------
# 4. CampSyncClient — transport-level behavior via mocked transports
# ---------------------------------------------------------------------------


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


class TestCampSyncClient:
    def test_empty_batch_is_noop(self, sample_card: MarketplaceAgentCard) -> None:
        client = CampSyncClient(transport=lambda **kwargs: _MockResponse())
        result = client.push([], bearer_token="tok")
        assert result.ok
        assert result.synced == 0
        assert result.ids == []

    def test_happy_path_200(self, sample_card: MarketplaceAgentCard) -> None:
        captured: dict[str, Any] = {}

        def _transport(**kwargs: Any) -> _MockResponse:
            captured.update(kwargs)
            return _MockResponse(
                body={
                    "synced": 1,
                    "skipped": 0,
                    "ids": [sample_card.agent_id],
                    "rejected": [],
                }
            )

        client = CampSyncClient(transport=_transport)
        result = client.push([sample_card], bearer_token="tok")

        assert result.ok
        assert result.synced == 1
        assert result.ids == [sample_card.agent_id]
        assert captured["headers"]["Authorization"] == "Bearer tok"
        assert "cards" in captured["json"]

    def test_org_id_included_when_provided(
        self, sample_card: MarketplaceAgentCard
    ) -> None:
        captured: dict[str, Any] = {}

        def _transport(**kwargs: Any) -> _MockResponse:
            captured.update(kwargs)
            return _MockResponse(body={"synced": 1, "skipped": 0, "ids": ["x"], "rejected": []})

        client = CampSyncClient(transport=_transport)
        client.push([sample_card], bearer_token="tok", org_id="org_abc")
        assert captured["json"]["org_id"] == "org_abc"

    def test_org_id_omitted_when_none(self, sample_card: MarketplaceAgentCard) -> None:
        captured: dict[str, Any] = {}

        def _transport(**kwargs: Any) -> _MockResponse:
            captured.update(kwargs)
            return _MockResponse(body={"synced": 1, "skipped": 0, "ids": ["x"], "rejected": []})

        client = CampSyncClient(transport=_transport)
        client.push([sample_card], bearer_token="tok")
        assert "org_id" not in captured["json"]

    def test_batch_limit_enforced(self, sample_card: MarketplaceAgentCard) -> None:
        client = CampSyncClient(transport=lambda **kwargs: _MockResponse())
        # 101 cards must be rejected client-side (don't send)
        cards = [
            sample_card.model_copy(update={"agent_id": new_agent_id()}) for _ in range(101)
        ]
        result = client.push(cards, bearer_token="tok")
        assert not result.ok
        assert "100" in (result.error or "")

    def test_rate_limit_surfaced(self, sample_card: MarketplaceAgentCard) -> None:
        def _transport(**kwargs: Any) -> _MockResponse:
            return _MockResponse(status_code=429, headers={"Retry-After": "42"})

        client = CampSyncClient(transport=_transport)
        result = client.push([sample_card], bearer_token="tok")
        assert not result.ok
        assert "rate_limited" in (result.error or "")
        assert "42" in (result.error or "")

    def test_http_400_surfaced(self, sample_card: MarketplaceAgentCard) -> None:
        def _transport(**kwargs: Any) -> _MockResponse:
            return _MockResponse(
                status_code=400, body={"error": "missing content_hash"}
            )

        client = CampSyncClient(transport=_transport)
        result = client.push([sample_card], bearer_token="tok")
        assert not result.ok
        assert "400" in (result.error or "")

    def test_transport_exception_surfaced(
        self, sample_card: MarketplaceAgentCard
    ) -> None:
        def _transport(**kwargs: Any) -> _MockResponse:
            raise ConnectionError("refused")

        client = CampSyncClient(transport=_transport)
        result = client.push([sample_card], bearer_token="tok")
        assert not result.ok
        assert "transport error" in (result.error or "")
        assert "refused" in (result.error or "")

    def test_rejected_cards_surfaced(self, sample_card: MarketplaceAgentCard) -> None:
        def _transport(**kwargs: Any) -> _MockResponse:
            return _MockResponse(
                body={
                    "synced": 0,
                    "skipped": 0,
                    "ids": [],
                    "rejected": [
                        {"agent_url": sample_card.agent_url, "reason": "agent_not_found"}
                    ],
                }
            )

        client = CampSyncClient(transport=_transport)
        result = client.push([sample_card], bearer_token="tok")
        assert result.ok
        assert result.rejected is not None
        assert result.rejected[0]["reason"] == "agent_not_found"
