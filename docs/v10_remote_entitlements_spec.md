---
last_updated: 2026-04-21
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.10-spec (implementation pending)
---

# v0.10 remote-tier entitlements — spec

**Status:** spec only. No implementation yet. Owner: next-session agent_2.

## Problem

Today, `detect_effective_tier()` reads local SQLite state (`~/.carl/db.sqlite`).
That's fast, offline, and survives network outages — but it's also trivially
tamperable. Any user can edit the SQLite to claim `PAID` and unlock
`train.slime.managed`, `mcp.serve`, and the rest of the paid surface.

For the v0.10 "managed" features that spend carl.camp's compute budget (or
otherwise bind to real-money billing via Lodge), we need a verifiable
entitlement surface that's:

1. **Fast on the happy path** — local check still dominates latency.
2. **Tamper-resistant** — a local edit cannot unlock paid features; the
   remote signature is required.
3. **Offline-tolerant** — carl.camp being down must not turn into a hard
   denial; 24h grace with degraded behavior is the failure mode.
4. **Minimally invasive** — no new top-level surface; extend the existing
   `@tier_gate` decorator with an optional flag.

## Non-goals

- Remote checks on every CLI invocation. `carl train --dry-run` stays
  offline-first; `@tier_gate` without `verify_remote=True` is unchanged.
- Replacing local SQLite. The local tier read stays the authoritative
  fast path; remote is a **counter-signature**, not a replacement.
- Per-request round-trips. 15-minute cache means a hot user pays the
  network cost ~4× per hour max.

## Architecture

```
                                    ┌─────────────────────┐
                                    │ carl.camp signer    │
                                    │ (ed25519 keypair)   │
                                    │ pubkey: published   │
                                    │ @ /.well-known/...  │
                                    └────────┬────────────┘
                                             │ signs
                                             ▼
┌──────────────┐                   ┌──────────────────────┐
│  carl CLI    │  GET              │ /api/platform/       │
│ (local)      │─────────────────▶ │   entitlements       │
│              │ ◀─────────────────│ → EntitlementJWT     │
│              │    JWT response    │   (RS256 / EdDSA)   │
└──────┬───────┘                   └──────────────────────┘
       │ verify + cache (15 min)
       ▼
┌──────────────────────────┐
│ ~/.carl/                 │
│   entitlements_cache.json│
│                          │
│   tier_gate(verify_remote│
│     =True) → read cache  │
└──────────────────────────┘
```

### JWT shape

Header:
```json
{"alg": "EdDSA", "typ": "JWT", "kid": "carl-camp-ent-202604"}
```

Payload (all required):
```json
{
  "iss": "https://carl.camp",
  "sub": "user:<sig_public_component>",
  "iat": 1735344000,
  "exp": 1735344900,
  "tier": "paid",
  "entitlements": {
    "train.slime.managed": true,
    "train.slime.moe_presets": true,
    "train.slime.async_disaggregated": true,
    "mcp.serve": true,
    "compute.multi_backend": true,
    "marketplace.publish": true
  },
  "quota": {
    "managed_gpu_hours_remaining": 128,
    "resonants_published_remaining": 1000
  },
  "cache_ttl_s": 900
}
```

Signature: Ed25519 over `base64url(header).base64url(payload)`. Verified
client-side with the public key shipped in the CLI (`carl_core.signing`
already has `pynacl` via `[constitutional]`).

**Subject format.** `user:<sig_public_component>` where
`sig_public_component = sha256(user_secret)[:16]` — the v0.9.1 identity
fingerprint from `resonant_store.identity_fingerprint()`. No PII; maps
directly to carl.camp's user record.

**Cache TTL** is advisory: the CLI caps it at 15 minutes regardless of the
server-supplied value. Prevents a malicious remote from setting ttl=∞.

### Decorator extension

Add `verify_remote: bool = False` to `@tier_gate`. When True, after the
local tier check passes, the decorator:

1. Reads `~/.carl/entitlements_cache.json` if present + not expired.
2. If no cache or expired, fires a background `GET /api/platform/entitlements`
   with `Authorization: Bearer <CARL_CAMP_TOKEN>`.
3. Verifies the returned JWT using the embedded Ed25519 public key.
4. Checks `jwt.entitlements[feature] == True`. Deny if False.
5. Checks `jwt.quota` if the feature has a quota key; deny on zero.
6. Persists the JWT to the cache file (mode 0600) with its verified payload.

On any of: missing token, network failure, signature failure, expired JWT:

- **If cache exists and is ≤ 24h stale**: allow the action, log a warning
  `carl.gate.tier_remote_stale`.
- **If cache doesn't exist**: deny with `carl.gate.tier_remote_unverified`
  (require a one-time online check before the grace window starts).
- **If cache is > 24h stale**: deny with `carl.gate.tier_remote_grace_expired`.

### New error codes

All under `carl.gate.*` per the existing convention:

```
carl.gate.tier_insufficient        (existing, local check failed)
carl.gate.tier_remote_mismatch     (remote says NO, local says YES)
carl.gate.tier_remote_unverified   (never verified, no cache)
carl.gate.tier_remote_stale        (cache > 15min but <= 24h — allowed)
carl.gate.tier_remote_grace_expired (cache > 24h — denied)
```

## API contract (carl.camp side)

**Endpoint:** `GET /api/platform/entitlements`

**Auth:** `Authorization: Bearer <CARL_CAMP_TOKEN>` (existing convention).

**Request:** no body. Optional `?include_quota=true` for quota fields.

**Response:** `200 OK` with the JWT as a bare string, `text/plain`. No
JSON wrapper — simpler to verify. On `401`, `403`, `404`: deny path.

**Rate limit:** 60 requests / hour / token. CLI caches 15 min, so a
well-behaved client burns 4 / hour.

## Implementation sketch (not in this session)

1. `packages/carl-core/src/carl_core/entitlements.py` (new, MIT):
   - `EntitlementJWT` Pydantic model.
   - `verify_entitlement_jwt(token: str, pubkey: bytes) -> EntitlementJWT`.
   - `EntitlementCache` — handles the 0600 file I/O + TTL logic.
2. `src/carl_studio/tier.py` — extend `tier_gate` decorator with
   `verify_remote: bool`. Compose the remote verifier AFTER the local
   `tier_allows` check.
3. `src/carl_studio/camp_client.py` (new): the HTTP client that fetches
   the JWT. Existing `BreakAndRetryStrategy` from `carl_core.resilience`
   wraps it.
4. `packages/carl-core/src/carl_core/errors.py` — add the four new
   `carl.gate.tier_remote_*` error subclasses.
5. Tests:
   - Happy path: local OK + remote OK → allow.
   - Remote denies: local OK + remote NO → `tier_remote_mismatch`.
   - Cache hit: second call within 15 min does not re-fetch.
   - Grace: network fails + cache ≤ 24h old → warn + allow.
   - Grace expired: cache > 24h → deny.
   - Tampered JWT: signature verify fails → treat as no-cache.
   - Clock skew: iat > now + 5min → reject.

## Public key distribution

Two options, in rough order of preference:

1. **Ship the pubkey in the CLI binary** (`carl_core/entitlements.py`
   contains `CAMP_ENTITLEMENT_PUBKEY: bytes = b"..."`). Rotations require
   a CLI upgrade. Simple, offline-safe, matches the `CARL_CAMP_TOKEN`
   convention. **Recommended.**
2. **Serve the pubkey at `https://carl.camp/.well-known/carl-camp-ent-pubkey`**
   with a 7-day cache and pin-on-first-use. More flexible but adds a
   bootstrap round-trip. Defer to v0.11 if we ever need rotation.

## Key distribution for signing

carl.camp's signer key lives in carl.camp's secret store (not in this
spec's scope). Rotation cadence: annual, or on compromise. The `kid`
field in the JWT header allows multiple active keys; the CLI ships with
the last two valid kids and their pubkeys.

## Open questions (for user review before implementation)

1. **Public-key rotation cadence.** Is annual rotation acceptable, or do
   we want quarterly? Annual simplifies the CLI; quarterly increases
   blast-radius containment.
2. **Cache location.** `~/.carl/entitlements_cache.json` is proposed.
   Alternative: store inside the existing `db.sqlite` under a
   `entitlements_cache` table so the cache lifecycle matches the rest
   of carl-state. Leaning toward flat file for simpler zero-knowledge
   semantics (the agent can't read SQLite tables it wasn't granted).
3. **Grace window duration.** 24h is proposed. For carl.camp outages,
   is this too lenient (fraud window) or too strict (genuine outage)?
   Tuning parameter.
4. **Does `verify_remote` default to True for `train.slime.managed`
   at landing time, or do we ship it as `False` initially and flip in
   a later patch?** Leaning "False in v0.10 landing, True in v0.10.1"
   so we can observe the error-code surface in the wild before it
   becomes an enforcement boundary.
5. **Signed vs encrypted.** The spec uses a signed JWT (integrity only).
   Do we also want the entitlement claims to be encrypted at rest so
   a leaked cache file doesn't reveal user quota state? Probably not
   worth it — the entitlement itself is not sensitive beyond the
   `sig_public_component` subject.

## Cross-system invariants this preserves

- **`CARL_CAMP_HF_TOKEN` invariant** (from 2026-04-21 carl.camp handoff):
  the managed-slime dispatcher reads its OWN HF token, not the user's.
  This spec does not change that — the entitlement JWT says "yes, this
  user may dispatch to the managed path." The dispatcher then uses
  `CARL_CAMP_HF_TOKEN` for the HF operations.
- **Gate on autonomy, not capability** (tier.py:16-24): `verify_remote`
  is ONLY for autonomy-flavored features (`train.slime.managed`,
  `mcp.serve`, `orchestration`, `marketplace.publish`). Pure
  capability features (`train`, `train.grpo`, `train.slime`) stay
  local-only.
- **`InteractionChain` observability**: every entitlement check emits
  a `GATE_CHECK` Step with the feature name, the local tier, the
  verdict, and a 12-hex fingerprint of the JWT (or "unverified"). No
  token values, no quota values — just the decision shape.

## Rollout

1. Ship the new module + decorator flag + error codes + cache I/O.
2. Flip `verify_remote=True` on the first managed feature
   (`train.slime.managed`) in a follow-up patch.
3. Monitor `carl.gate.tier_remote_*` error rates via the existing
   telemetry surface for 2 weeks.
4. Flip remaining features in priority order: `mcp.serve` →
   `compute.multi_backend` → `marketplace.publish` → the rest.

No backfill required — unverified users hit the "no cache" path once,
then roll into the happy cadence.
