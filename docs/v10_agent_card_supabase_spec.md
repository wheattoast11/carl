---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.10-A
status: implementation-ready
---

# Agent Card + Supabase spec (v0.10-A #1, MIT-clean)

This spec is implementation-ready. It's grounded in `lib/sync/electric-bridge.ts`,
`lib/sync/sync-scopes.ts`, and the PGLite schema v1.18 in terminals-landing-new.
No BUSL code is copied; MIT-compatible shapes are mirrored.

## The flow (concrete, verified)

```
FREE tier
─────────
carl agent register --name MyAgent
  → carl-studio LocalDB: INSERT INTO agent_cards (...)  -- local SQLite
  → CampConsole.notice("ℹ Agent registered locally. Upgrade to publish: carl camp upgrade")

PAID tier
─────────
carl agent register --name MyAgent
  → carl-studio LocalDB: INSERT INTO agent_cards (...)
  → HTTP POST https://carl.camp/api/sync/agent-cards  (JWT-authenticated)
      → Backend calls the same pattern as deployment-sync.ts:
          supabase.from('agent_cards').upsert(row, { onConflict: 'id' })
      → Returns synced count
```

The carl.camp backend is the mediator. carl-studio (Python CLI) does not
call Supabase directly — it sends an HTTP POST, and the backend (which
has the Supabase service key) does the upsert.

## `AgentCardRow` — the canonical shape

Pydantic model for carl-studio, mirrors the proposed PGLite v1.19 schema.
Mirrors column-for-column what `electric-bridge.ts::pushToSupabase` expects.

```python
# src/carl_studio/a2a/agent_card.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class AgentCardRow(BaseModel):
    """Agent card — dual-destination: carl-studio LocalDB + Supabase.

    The shape is aligned with terminals-tech's deployment-sync.ts pattern
    (last-write-wins conflict strategy, UNIQUE(user_id, slug) invariant).
    """

    id: str = Field(description="UUID or prefixed ID, e.g. 'ac_01HXYZ...'")
    user_id: str = Field(description="Authenticated user ID from carl.camp JWT")
    slug: str = Field(description="URL-safe slug, unique per user")
    name: str = Field(description="Display name")
    description: str | None = Field(default=None, description="Markdown")
    manifest: dict[str, Any] = Field(
        description="Full A2A-compatible agent manifest (role, capabilities, version)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Tags, tier, x402_receipt_hash, etc.",
    )
    visibility: str = Field(
        default="private",
        description="'private' | 'public' | 'unlisted'",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        json_schema_extra = {
            "example": {
                "id": "ac_01HXYZABC123",
                "user_id": "user_xyz789",
                "slug": "my-research-agent",
                "name": "My Research Agent",
                "description": "Reads arxiv, summarizes, cites.",
                "manifest": {
                    "role": "researcher",
                    "capabilities": ["search", "synthesis", "citation"],
                    "version": "1.0.0",
                },
                "metadata": {"tier": "paid", "tags": ["research"]},
                "visibility": "private",
            }
        }
```

**Why this shape:** mirrors `TerminalAgent` (from `@terminals-tech/agent`,
MIT) plus the fields `electric-bridge.ts` requires for sync. `manifest`
stays JSONB-compatible so future schema evolution doesn't require a
migration.

## PGLite v1.19 migration (proposal, for terminals-tech side)

```sql
-- lib/db/migrations/v1.19-agent-cards.sql  (terminals-tech side)
CREATE TABLE IF NOT EXISTS agent_cards (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  slug TEXT NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  manifest JSONB NOT NULL,
  metadata JSONB DEFAULT '{}',
  visibility TEXT DEFAULT 'private',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_agent_cards_user
  ON agent_cards(user_id);

CREATE INDEX IF NOT EXISTS idx_agent_cards_visibility
  ON agent_cards(visibility)
  WHERE visibility != 'private';
```

## Sync scope registration (terminals-tech side, mirror of deployment-sync)

Based on `lib/sync/sync-scopes.ts` patterns (pattern verified, not copied):

```typescript
// lib/sync/scopes/agent-cards.ts  (terminals-tech side, new)
export const AGENT_CARDS_SCOPE: SyncScopeDef = {
  id: "agent_cards",
  table: "agent_cards",
  conflictStrategy: "last-write-wins",   // monotonic evolution, last edit wins
  priority: 10,
  autoSync: true,                         // push on every edit
  description: "A2A agent card registry",
};
```

**Rationale:** agent cards evolve monotonically (user edits replace
prior state). `last-write-wins` matches the pattern used for
`user_stacks`, `chat_sessions`, `interactions`. Contrast with
`workspace_state` (client-wins) which is local-only.

## carl-studio changes (v0.10-A #1 scope)

### New files

1. **`src/carl_studio/a2a/agent_card.py`** (~60 LOC)
   - `AgentCardRow` Pydantic model (above)
   - `canonical_slug(name) -> str` helper

2. **`src/carl_studio/a2a/agent_card_store.py`** (~80 LOC)
   - `AgentCardStore(db: LocalDB)` — wraps LocalDB with `agent_cards` table CRUD
   - Uses `ConfigRegistry[AgentCardRow]` pattern from v0.8.0 (typed store over LocalDB KV)
   - Methods: `register(row)`, `get(slug)`, `list_local(user_id)`, `delete(id)`

3. **`src/carl_studio/a2a/sync_agent_cards.py`** (~50 LOC)
   - `push_to_camp(store, camp_client)` — HTTP POST to `carl.camp/api/sync/agent-cards`
   - Uses existing `CampProfile` HTTP contract (no new deps)
   - Gated by `@tier_gate(Tier.PAID, feature="agent_marketplace_publish")`

4. **`tests/test_agent_card.py`** (~150 LOC) — 8 tests
   - Pydantic shape round-trip
   - LocalDB CRUD (create, get, list, delete)
   - FREE tier: register local, FYI notice, no HTTP call
   - PAID tier: register local + HTTP push
   - Tier-gate denial raises TierGateError with `gate_code`
   - Slug uniqueness enforcement
   - Last-write-wins on re-register
   - Visibility default + override

### Modified files

5. **`src/carl_studio/db.py`** — add `agent_cards` table DDL to schema init
6. **`src/carl_studio/a2a/_cli.py`** — extend with:
   - `carl agent register` — wires `AgentCardStore.register()` + optional sync
   - `carl agent list` — reads local store
   - `carl agent sync` — explicit re-push (PAID only)
   - `carl agent card <slug>` — prints the rendered card via CampConsole
7. **`docs/operations.md`** — add `CARL_AGENT_CARD_VISIBILITY_DEFAULT` env var
8. **`README.md`** — add two-line mention under Features

### FYI nudge pattern (per Tej's "no popups, just side notifications" rule)

When a FREE-tier user runs `carl agent register`, the success path ends with:

```python
console.notice(
    "ℹ  Agent registered locally. "
    "Upgrade to publish to the carl.camp marketplace: "
    "[bold cyan]carl camp upgrade[/bold cyan]"
)
```

Single line, no popup, no blocking modal. Same pattern already used by
`credits/_cli.py` and `consent.py`. Matches Tej's explicit rule.

## What does NOT ship in v0.10-A #1

- **384-dim pgvector embeddings for agent discovery.** Deferred to
  v0.10-B (requires MiniLM or similar at the carl-studio side — we
  don't want transformers.js pulled into CLI). Shape is known
  (`normalizeEmbedding()` + `DEFAULT_EMBEDDING_DIM = 384` from
  `mesh-sink.ts`) and will be adopted when embeddings land.
- **Agent-discovery / marketplace search.** Backend-side (carl.camp),
  not carl-studio.
- **Federation via AT Protocol DIDs.** terminals.tech uses Bluesky-style
  DIDs for decentralized discovery; this is a v0.11+ question because
  it requires DID generation + PDS endpoint wiring on carl.camp.
- **x402 receipt hash field population.** `metadata.x402_receipt_hash`
  is present in the shape but populated only when x402 is wired into
  `carl agent register` payment flow. Defer to later.
- **MCP tool registration for agent cards.** "Peer agents call into
  published cards via MCP" — valuable but requires the A2A protocol
  glue on carl.camp side. v0.10-B.

## Testing strategy

- Unit tests in `tests/test_agent_card.py` (above).
- Integration test: mock `CampProfile` HTTP client; assert POST payload
  matches `AgentCardRow.model_dump_json()` exactly.
- No live Supabase hits in CI — the Supabase write is backend-side.
  carl-studio only verifies it sent the correct POST body.

## Verification commands (once shipped)

```bash
carl agent register --name "My Research Agent"
# → FREE: "Agent registered locally. Upgrade to publish: carl camp upgrade"
# → PAID: "Agent registered locally and published to carl.camp."

carl agent list
# → table of local agent cards

carl agent card my-research-agent
# → rendered card detail

carl agent sync
# → PAID: re-pushes all local cards to carl.camp
# → FREE: TierGateError → "Upgrade to sync to carl.camp"
```

## Cost estimate

**Total LOC: ~340** (60 model + 80 store + 50 sync + 150 tests + small CLI wires).
**Ship sequencing:** v0.10.0-alpha, alongside `carl-update` + `carl-env`
implementations. Backend (carl.camp) work is parallel and not blocking
for carl-studio's side of this spec — the Python CLI just sends well-shaped
POSTs; the backend handles Supabase when ready.

## License

MIT-clean throughout. `TerminalAgent` shape inspiration from MIT
`@terminals-tech/agent`. No BUSL code copied. The sync pattern
(last-write-wins via `upsert({onConflict: 'id'})`) is standard Supabase
usage, not a proprietary algorithm.
