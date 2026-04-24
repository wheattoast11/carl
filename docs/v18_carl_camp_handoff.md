---
last_updated: 2026-04-24
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.18.1
---

# v0.18 handoff to the carl.camp agent

This note is the structured artifact for the carl.camp web/agent team
so they can ingest v0.18 surface changes **without editing carl-studio**.
carl-studio is MIT and ships the client; carl.camp ships the server +
web app that consumes it.

## What carl.camp must know (concrete, no ambiguity)

### 1. `carl init --json` contract is now stable

**Client emits** (at `typer.echo` from `src/carl_studio/cli/init.py:85`):

```json
{
  "status": "probed",
  "steps_done": [],
  "global_config": "/path/to/~/.carl/config.yaml",
  "first_run_marker": "/path/to/~/.carl/.initialized",
  "chain": {
    "steps": [
      {
        "action": "cli_cmd",
        "name": "probe_state",
        "input": {"mode": "json"},
        "output": {
          "first_run_complete": false,
          "camp_session": false,
          "llm_provider_detected": null,
          "training_extras_healthy": false,
          "project_config_present": false,
          "consent_set": false,
          "context_present": false
        },
        "success": true,
        ...
      }
    ]
  }
}
```

**carl.camp should**: consume the 7 keys in `chain.steps[].output` (find
the step where `name == "probe_state"`). If an agent-card bootstrap
flow needs to decide whether a fresh `carl init` is needed, this is the
single source of truth.

### 2. Trust registry is LOCAL ONLY

`~/.carl/trust.yaml` is client-side. It gates the bare-entry UX and
**does not** sync to carl.camp. If carl.camp wants per-org project trust,
that's a new server-side concept with no client surface today.

### 3. Session CLI is project-aware

`carl session list/show/delete` walks up from cwd looking for `.carl/`.
Sessions are stored at `<project_root>/.carl/sessions/*.json` with
`schema_version=1`. carl.camp **does not** have a session sync
endpoint today; if it adds one, the contract should key on
`(project_root_hash, session_id)` — not on cwd, which changes per
terminal.

### 4. Router subcommand registry

`src/carl_studio/cli/entry.py:REGISTERED_SUBCOMMANDS` is a frozenset
of 53 subcommands. If carl.camp adds a new client-side verb, it must
be added to this set AND wired via `cli/wiring.py`. Anything else gets
interpreted as a bare prompt → dropped into chat.

### 5. Resonant publish contract (unchanged since v0.9.1)

Still `POST {CARL_CAMP_BASE}/api/resonants` per
`docs/eml_signing_protocol.md` §5.1. Headers: `X-Carl-User-Secret`
(base64), `X-Carl-Projection`, `X-Carl-Readout`, `Authorization`.
**Client refuses non-HTTPS bases unless `--dry-run`.**

### 6. Agent-card registration contract (unchanged since v0.13)

Still `POST /api/agents/register` + `POST /api/sync/agent-cards`.
Response envelope: `{ok, synced, skipped, ids, rejected}`.
Idempotency-via-`content_hash`. Last-write-wins.

### 7. CARL_CAMP_HF_TOKEN invariant (unchanged, load-bearing)

When carl.camp dispatches managed-slime jobs, it MUST use
`CARL_CAMP_HF_TOKEN` — **never** the user's encrypted HF token.
Mixing paths leaks credentials. This is enforced at dispatch, not in
carl-studio.

## What carl.camp can build NOW on top of v0.18

All items below use existing carl-studio public surface; no changes to
carl-studio required.

| Feature | carl.camp surface | Uses |
|---|---|---|
| "Is this user fresh-install?" probe | server-side ingestion of `init --json` output | `status` + 7 probe keys |
| Per-org trusted projects | new server endpoint (not wired in client today) | N/A on client |
| Session replay sharing | new `/api/sessions` endpoint | reads `<project>/.carl/sessions/*.json` format |
| Marketplace agent discovery | existing `/api/agents/*` | `MarketplaceAgentCard` + `CampSyncClient` |
| Resonant leaderboard | existing `/api/resonants/*` | EML signing protocol |
| Managed-slime dispatch | new `/api/train/slime/submit` | `SlimeArgs.model_json_schema()` for validation |

## What the web site should show (suggested, not binding)

The user-facing story to surface on carl.camp:

- **"One binary, four entry modes."** Short visual: `carl` / `carl "q"`
  / `carl -p "q"` / `carl <verb>`. Match the table at
  `docs/v18_journey_coverage.md`.
- **"Trust is local."** Surface `carl trust status` output in a "your
  machine" section; don't promise cloud sync we don't deliver.
- **"Session portability."** Show the `schema_version=1` format and a
  future "carl session export/import" CTA (not shipped; would be
  client work).
- **"Agent + Resonant economics live."** Link to the existing
  marketplace + resonant publish flows with the real envelope schemas.

## Items that require BOTH sides to agree before shipping

- **Remote tier verification** (carl-studio v0.10-A item, per
  `CLAUDE.md`). Needs `docs/v10_remote_entitlements_spec.md` stub
  first, then a signed-entitlements endpoint on carl.camp. Deferred
  until spec lands.
- **Managed-slime dispatcher.** Needs `CARL_CAMP_HF_TOKEN` path
  enforced at the dispatch endpoint. carl-studio will refuse to send
  a user HF token when `--managed` is selected.
- **Session sync.** If built, keyed on `(project_root_hash, session_id)`.

## What I (Claude, in carl-studio) will NOT do unilaterally

- Edit the carl.camp repo from here.
- Hit live carl.camp endpoints with real credentials without explicit
  per-batch authorization (see `tests/journeys/BATCHES.md` online
  batches ON-A through ON-F).
