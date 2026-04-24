"""Marketplace agent-cards — user-registered entries synced to carl.camp.

v0.10 W7. Implements `POST /api/sync/agent-cards` per the contract
registered in ``docs/v10_agent_card_supabase_spec.md`` and verified
against the carl.camp backend agent.

Shape alignment with the backend contract (2026-04-20):
- Each card has an ``agent_id`` (uuid → ``recipes.id``) that must exist
  in carl.camp before the sync call. Recipes are minted via the
  separate ``POST /api/agents/register`` endpoint; the returned uuid is
  what carl-studio stores locally.
- Sync is idempotent via ``content_hash`` — sha256 over the card's
  canonical JSON serialization (excluding timestamps and ids).
- FREE tier: register locally, show a one-line FYI notice.
- PAID tier: register locally AND push to carl.camp.

This module is MIT-licensed. It mirrors the MIT ``TerminalAgent`` shape
from ``@terminals-tech/agent`` (verified MIT at package.json + LICENSE
2026-04-20) as inspiration; no BUSL code was copied.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Canonical row shape — mirrors the backend contract column-for-column
# ---------------------------------------------------------------------------


class MarketplaceAgentCard(BaseModel):
    """A user-registered agent card — local LocalDB row + carl.camp sync shape.

    Fields match the ``POST /api/sync/agent-cards`` contract exactly
    (``agent_id``, ``agent_url``, ``agent_name``, not ``id``/``slug``/``name``).
    ``content_hash`` is computed by :meth:`compute_content_hash` and must
    be present on sync.
    """

    model_config = ConfigDict(arbitrary_types_allowed=False)

    agent_id: str = Field(description="UUID from carl.camp POST /api/agents/register")
    agent_url: str = Field(description="Full URL; UNIQUE per (org_id, agent_url)")
    agent_name: str = Field(description="Display name; NOT 'name'")
    description: str | None = None
    capabilities: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=100,  # Fano V1 boundedness: hard cap on array length
        description=(
            "Free-form JSONB array. Marketplace UI renders best when each "
            "element has {id: str, description: str} (A2A-compatible). "
            "Capped at 100 elements to preserve BITC axiom 1 (finite support)."
        ),
    )
    public_key: str | None = Field(
        default=None,
        description="ed25519 hex public key; private key stays in ~/.carl/wallet.enc",
    )
    key_algorithm: str | None = Field(
        default=None,
        description="Forward-compat: 'ed25519' | 'falcon' | 'ml-dsa' | None. Default ed25519 on register.",
    )
    content_hash: str = Field(
        description="sha256-hex over canonical JSON; drives server idempotency"
    )
    is_active: bool = True
    rate_limit_rpm: int = 60

    # Locally-tracked metadata (NOT sent to server)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_sync_payload(self) -> dict[str, Any]:
        """Serialize for POST /api/sync/agent-cards. Excludes local-only fields."""

        return {
            "agent_id": self.agent_id,
            "agent_url": self.agent_url,
            "agent_name": self.agent_name,
            "description": self.description,
            "capabilities": self.capabilities,
            "public_key": self.public_key,
            "content_hash": self.content_hash,
            "is_active": self.is_active,
            "rate_limit_rpm": self.rate_limit_rpm,
        }


# ---------------------------------------------------------------------------
# content_hash canonicalization
# ---------------------------------------------------------------------------


def compute_content_hash(
    *,
    agent_name: str,
    description: str | None,
    capabilities: list[dict[str, Any]],
    public_key: str | None,
    is_active: bool,
    rate_limit_rpm: int,
) -> str:
    """Deterministic sha256-hex over the content-defining fields only.

    Excludes agent_id, agent_url, timestamps, and any other fields that
    change without meaning. Canonicalizes via ``json.dumps`` with
    sorted keys + compact separators so two client-side computations of
    the same semantic card produce the same hash.
    """

    canonical = json.dumps(
        {
            "agent_name": agent_name,
            "description": description,
            "capabilities": capabilities,
            "public_key": public_key,
            "is_active": is_active,
            "rate_limit_rpm": rate_limit_rpm,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Local storage (LocalDB-backed)
# ---------------------------------------------------------------------------


_AGENT_CARD_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS agent_cards (
    agent_id        TEXT PRIMARY KEY,
    agent_url       TEXT NOT NULL,
    agent_name      TEXT NOT NULL,
    description     TEXT,
    capabilities    TEXT NOT NULL DEFAULT '[]',
    public_key      TEXT,
    key_algorithm   TEXT,
    content_hash    TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1,
    rate_limit_rpm  INTEGER NOT NULL DEFAULT 60,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_agent_cards_url ON agent_cards(agent_url);
"""


class AgentCardStore:
    """LocalDB CRUD for agent cards. SQLite-backed; mirrors what carl.camp stores.

    Supports two LocalDB connection shapes:
    1. ``db.connect()`` returns a context manager (carl-studio's LocalDB)
    2. ``db.connect()`` returns a raw sqlite3.Connection (test helpers)
    """

    def __init__(self, db: Any) -> None:
        self._db = db
        self._ensure_schema()

    def _conn_ctx(self) -> Any:
        """Return a context manager yielding a sqlite connection.

        Normalizes both LocalDB shapes so the rest of the store can use
        ``with self._conn_ctx() as conn:`` uniformly.
        """

        from contextlib import contextmanager

        raw = self._db.connect()
        # If it's already a context manager (has __enter__), use as-is.
        if hasattr(raw, "__enter__") and hasattr(raw, "__exit__"):
            return raw

        @contextmanager
        def _ctx() -> Any:
            try:
                yield raw
            finally:
                if hasattr(raw, "close"):
                    raw.close()

        return _ctx()

    def _ensure_schema(self) -> None:
        with self._conn_ctx() as conn:
            conn.executescript(_AGENT_CARD_TABLE_DDL)
            conn.commit()

    def save(self, card: MarketplaceAgentCard) -> None:
        """Upsert by agent_id (last-write-wins)."""

        with self._conn_ctx() as conn:
            conn.execute(
                """INSERT INTO agent_cards (
                    agent_id, agent_url, agent_name, description, capabilities,
                    public_key, key_algorithm, content_hash, is_active, rate_limit_rpm,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    agent_url=excluded.agent_url,
                    agent_name=excluded.agent_name,
                    description=excluded.description,
                    capabilities=excluded.capabilities,
                    public_key=excluded.public_key,
                    key_algorithm=excluded.key_algorithm,
                    content_hash=excluded.content_hash,
                    is_active=excluded.is_active,
                    rate_limit_rpm=excluded.rate_limit_rpm,
                    updated_at=excluded.updated_at
                """,
                (
                    card.agent_id,
                    card.agent_url,
                    card.agent_name,
                    card.description,
                    json.dumps(card.capabilities),
                    card.public_key,
                    card.key_algorithm,
                    card.content_hash,
                    1 if card.is_active else 0,
                    card.rate_limit_rpm,
                    card.created_at.isoformat(),
                    card.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def get(self, agent_id: str) -> MarketplaceAgentCard | None:
        with self._conn_ctx() as conn:
            row = conn.execute(
                "SELECT * FROM agent_cards WHERE agent_id = ?", (agent_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_card(row)

    def list_all(
        self,
        *,
        limit: int = 500,
        offset: int = 0,
    ) -> list[MarketplaceAgentCard]:
        """List stored cards. Fano V1 boundedness fix: pagination required.

        Default ``limit=500`` preserves the original ergonomic ("show me
        everything") for small local stores while bounding memory for
        stores that grow past a few hundred cards.
        """
        if limit < 0 or offset < 0:
            raise ValueError(f"limit/offset must be non-negative (got limit={limit}, offset={offset})")
        with self._conn_ctx() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_cards ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_card(r) for r in rows]

    def count(self) -> int:
        """Total stored cards — cheap counter for pagination planning."""
        with self._conn_ctx() as conn:
            row = conn.execute("SELECT COUNT(*) FROM agent_cards").fetchone()
            return int(row[0]) if row is not None else 0

    def delete(self, agent_id: str) -> bool:
        with self._conn_ctx() as conn:
            cur = conn.execute(
                "DELETE FROM agent_cards WHERE agent_id = ?", (agent_id,)
            )
            conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_card(row: Any) -> MarketplaceAgentCard:
        # Row supports both tuple and sqlite3.Row / mapping access
        def _get(key: str, idx: int) -> Any:
            try:
                return row[key]
            except (TypeError, KeyError, IndexError):
                return row[idx]

        return MarketplaceAgentCard(
            agent_id=_get("agent_id", 0),
            agent_url=_get("agent_url", 1),
            agent_name=_get("agent_name", 2),
            description=_get("description", 3),
            capabilities=json.loads(_get("capabilities", 4) or "[]"),
            public_key=_get("public_key", 5),
            key_algorithm=_get("key_algorithm", 6),
            content_hash=_get("content_hash", 7),
            is_active=bool(_get("is_active", 8)),
            rate_limit_rpm=int(_get("rate_limit_rpm", 9)),
            created_at=_parse_datetime(_get("created_at", 10)),
            updated_at=_parse_datetime(_get("updated_at", 11)),
        )


def _parse_datetime(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Sync — push local cards to carl.camp
# ---------------------------------------------------------------------------


@dataclass
class SyncResult:
    """Outcome of a POST /api/sync/agent-cards call."""

    synced: int = 0
    skipped: int = 0
    ids: list[str] | None = None
    rejected: list[dict[str, Any]] | None = None
    error: str | None = None
    # Server returns {ok: true, ...} envelope — captured so call sites
    # can distinguish transport success from business-logic success.
    envelope_ok: bool = True

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class RegisterResult:
    """Outcome of a POST /api/agents/register call.

    Shape of the real endpoint (2026-04-20):
        201 → {ok: true, agent_id, org_id, lifecycle_state, created_at}

    v0.18 Track D (§Q5): ``divergence_warning`` is populated when a retry
    happened AND the second attempt returned a different ``agent_id``
    than the first. Callers should surface this to the operator.

    ``observed_agent_id`` captures the ``agent_id`` returned in the
    response body regardless of whether the attempt was otherwise
    successful — this lets the retry layer detect divergence even when
    attempt 1 was wrapped in an error envelope.
    """

    agent_id: str | None = None
    org_id: str | None = None
    lifecycle_state: str | None = None
    created_at: str | None = None
    error: str | None = None
    # v0.18 Track D — divergence detection across retry attempts.
    divergence_warning: str | None = None
    observed_agent_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.agent_id is not None


class CampSyncClient:
    """HTTP client for POST /api/sync/agent-cards.

    Pluggable ``transport`` callable for testing; production uses the
    carl.camp HTTP surface via the existing ``CampProfile`` session.
    Batches ≤100 cards per request per the backend contract. On 429,
    honors ``Retry-After`` via v0.8's ``BreakAndRetryStrategy``.
    """

    _BATCH_LIMIT = 100

    def __init__(
        self,
        *,
        transport: "Any | None" = None,
        base_url: str = "https://carl.camp",
    ) -> None:
        self._transport = transport
        self._base_url = base_url

    # ------------------------------------------------------------------
    # Shared HTTP plumbing (v0.16 simplify)
    # ------------------------------------------------------------------

    def _require_transport(self) -> str | None:
        """Return an error message when no transport is configured, else None."""
        if self._transport is None:
            return "no transport configured; provide one for live sync"
        return None

    @staticmethod
    def _read_headers(response: Any) -> dict[str, str]:
        headers = getattr(response, "headers", None)
        if headers is None:
            return {}
        try:
            return dict(headers)
        except Exception:  # pragma: no cover — headers may be a custom mapping
            return {}

    @classmethod
    def _envelope_error(cls, response: Any) -> str | None:
        """Return a human-readable error string when the envelope is non-OK.

        Handles the three carl.camp failure modes uniformly:
        * 429 → rate-limited (surfaces Retry-After)
        * 4xx/5xx → http <status>: <error-body>
        * 200 + ``{"ok": false, ...}`` → <error-body> verbatim

        Returns ``None`` when the response passed all checks and the
        caller should parse the payload for success fields.
        """
        status = getattr(response, "status_code", None)
        if status == 429:
            retry_after = cls._read_headers(response).get("Retry-After")
            return f"rate_limited (Retry-After={retry_after})"
        if status is not None and status >= 400:
            try:
                detail = response.json() if hasattr(response, "json") else {}
            except Exception:
                detail = {}
            return f"http {status}: {detail.get('error', 'unknown')}"
        try:
            payload = response.json() if hasattr(response, "json") else {}
        except Exception:
            payload = {}
        if isinstance(payload, dict) and not payload.get("ok", True):
            return str(payload.get("error", "unknown"))
        return None

    @staticmethod
    def _parse_payload(response: Any) -> dict[str, Any]:
        try:
            payload = response.json() if hasattr(response, "json") else {}
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def push(
        self,
        cards: list[MarketplaceAgentCard],
        *,
        bearer_token: str,
        org_id: str | None = None,
    ) -> SyncResult:
        """Push up to ``_BATCH_LIMIT`` cards in a single request."""

        if not cards:
            return SyncResult(synced=0, skipped=0, ids=[], rejected=[])
        if len(cards) > self._BATCH_LIMIT:
            return SyncResult(
                error=f"batch exceeds {self._BATCH_LIMIT} cards (got {len(cards)})"
            )

        if (err := self._require_transport()) is not None:
            return SyncResult(error=err)

        body: dict[str, Any] = {"cards": [c.to_sync_payload() for c in cards]}
        if org_id is not None:
            body["org_id"] = org_id

        url = f"{self._base_url}/api/sync/agent-cards"
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }

        try:
            response = self._transport(url=url, headers=headers, json=body)  # type: ignore[misc]
        except Exception as exc:
            return SyncResult(error=f"transport error: {exc}")

        if (env_err := self._envelope_error(response)) is not None:
            return SyncResult(error=env_err)

        payload = self._parse_payload(response)
        return SyncResult(
            synced=int(payload.get("synced", 0)),
            skipped=int(payload.get("skipped", 0)),
            ids=list(payload.get("ids", [])),
            rejected=list(payload.get("rejected", [])),
            envelope_ok=bool(payload.get("ok", True)),
        )

    def register_recipe_shell(
        self,
        *,
        bearer_token: str,
        name: str | None = None,
        org_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RegisterResult:
        """Call ``POST /api/agents/register`` to mint a recipe-shell.

        The carl.camp endpoint creates an empty ``recipes`` row owned by
        the caller's org and returns a UUID. That UUID is what subsequent
        ``/api/sync/agent-cards`` calls reference as ``agent_id``.

        v0.18 Track D (§Q5): when ``idempotency_key`` is provided it
        travels as the ``Idempotency-Key`` header. On a transient failure
        (5xx status OR a network timeout surfaced as ``OSError``/
        ``TimeoutError`` from the transport) we retry ONCE with a 1s
        wait and the SAME Idempotency-Key. If both attempts succeed but
        return different ``agent_id`` values, the second result is used
        and ``divergence_warning`` is populated.
        """

        if (err := self._require_transport()) is not None:
            return RegisterResult(error=err)

        url = f"{self._base_url}/api/agents/register"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if org_id is not None:
            body["org_id"] = org_id

        first = self._attempt_register(url=url, headers=headers, body=body)

        # Retry ONLY when eligible: 5xx status OR a transient network
        # exception. A 4xx or a successful 2xx is terminal on the first
        # attempt — no retry, no divergence check needed.
        if not self._should_retry_register(first):
            return self._finalize_register(first)

        # Single-retry wait — 1s per spec. ``time.sleep`` is fine here;
        # this is a CLI-boundary call, not a hot loop.
        import time as _time

        _time.sleep(1.0)
        second = self._attempt_register(url=url, headers=headers, body=body)

        # Second attempt is terminal; surface whatever it yielded.
        # Detect agent_id divergence across the two attempts (carl.camp
        # echoes the same id when honoring the Idempotency-Key; a
        # divergent id would indicate the server didn't honor it). We
        # compare ``observed_agent_id`` — the raw body value — not
        # ``agent_id`` so a flaky 5xx that still echoed an id is caught.
        first_aid = first.observed_agent_id
        second_aid = second.observed_agent_id
        if (
            first_aid is not None
            and second_aid is not None
            and first_aid != second_aid
            and second.error is None
        ):
            warning = (
                f"divergence: attempt 1 returned {first_aid}; "
                f"attempt 2 returned {second_aid}; using second"
            )
            return RegisterResult(
                agent_id=second.agent_id,
                org_id=second.org_id,
                lifecycle_state=second.lifecycle_state,
                created_at=second.created_at,
                observed_agent_id=second.observed_agent_id,
                divergence_warning=warning,
            )
        return self._finalize_register(second)

    @staticmethod
    def _should_retry_register(result: "RegisterResult") -> bool:
        """Eligibility gate for the single retry.

        Per §Q5: retry only on 5xx status OR network timeout.
        ``_attempt_register`` encodes a transient network fault as
        ``error="transport timeout: <...>"`` or ``"http 5NN: ..."``;
        anything else (4xx, unknown envelope, success) is terminal.
        """
        if result.error is None:
            return False
        msg = result.error
        if msg.startswith("transport timeout:"):
            return True
        if msg.startswith("http "):
            # Parse the status token: "http 502: ..." → "502".
            try:
                token = msg[5:].split(":", 1)[0]
                return token.startswith("5") and len(token) == 3
            except (IndexError, ValueError):
                return False
        return False

    @staticmethod
    def _finalize_register(result: "RegisterResult") -> "RegisterResult":
        """Pass-through finalizer kept separate so retry paths share a
        single exit shape."""
        return result

    def _attempt_register(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> "RegisterResult":
        """Single request cycle. Encodes network timeouts as a
        retry-eligible ``transport timeout:`` error so the retry layer
        can branch on the error class without sniffing the exception
        type twice.

        Always records ``observed_agent_id`` from the response body
        when present, regardless of envelope state — this feeds the
        divergence check across retries even when an attempt's envelope
        was wrapped in an error.
        """
        try:
            response = self._transport(url=url, headers=headers, json=body)  # type: ignore[misc]
        except (TimeoutError, OSError) as exc:
            # OSError covers socket.timeout, requests.Timeout (subclass
            # via RequestException→OSError), and urllib.error.URLError
            # when the wrapped reason is socket.timeout. This is the
            # "network timeout" limb of §Q5's retry trigger.
            return RegisterResult(error=f"transport timeout: {exc}")
        except Exception as exc:
            # Non-timeout transport failures (DNS, TLS, etc.) are
            # terminal — no retry, propagate directly. Keeps the policy
            # conservative.
            return RegisterResult(error=f"transport error: {exc}")

        # Parse the payload once — we need ``agent_id`` both for the
        # success path AND for divergence detection across retries.
        payload = self._parse_payload(response)
        observed_aid = payload.get("agent_id")
        observed_aid_str = observed_aid if isinstance(observed_aid, str) else None

        if (env_err := self._envelope_error(response)) is not None:
            return RegisterResult(error=env_err, observed_agent_id=observed_aid_str)

        if observed_aid_str is None:
            return RegisterResult(error="response missing agent_id")

        return RegisterResult(
            agent_id=observed_aid_str,
            org_id=payload.get("org_id"),
            lifecycle_state=payload.get("lifecycle_state"),
            created_at=payload.get("created_at"),
            observed_agent_id=observed_aid_str,
        )


def new_agent_id() -> str:
    """Generate a fresh UUID string. Server may replace on register."""

    return str(uuid.uuid4())


__all__ = [
    "MarketplaceAgentCard",
    "compute_content_hash",
    "AgentCardStore",
    "CampSyncClient",
    "SyncResult",
    "RegisterResult",
    "new_agent_id",
]
