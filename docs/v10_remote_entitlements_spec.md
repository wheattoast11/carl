---
last_updated: 2026-05-07
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.10.0
---

# v0.10 remote-tier entitlements

**Status:** implemented in v0.10.0. See "Implementation map" below for file paths.

## Problem

`detect_effective_tier()` reads local SQLite (`~/.carl/db.sqlite`). Fast, offline, and survives outages — but trivially tamperable. Any user can edit the SQLite to claim `PAID` and unlock `train.slime.managed`, `mcp.serve`, and the rest of the paid surface.

For features that spend carl.camp's compute budget (or bind to real-money billing via Lodge), we need a verifiable entitlement surface that's:

1. **Fast on the happy path** — local check still dominates latency.
2. **Tamper-resistant** — a local edit cannot unlock paid features; the remote signature is required.
3. **Offline-tolerant** — carl.camp being down must not turn into a hard denial; 24h grace with degraded behavior is the failure mode.
4. **Minimally invasive** — no new top-level surface; extend the existing `@tier_gate` decorator with an optional flag.

## Non-goals

- Remote checks on every CLI invocation. `carl train --dry-run` stays offline-first; `@tier_gate` without `verify_remote=True` is unchanged.
- Replacing local SQLite. The local tier read stays the authoritative fast path; remote is a **counter-signature**, not a replacement.
- Per-request round-trips. 15-minute cache means a hot user pays the network cost ~4× per hour max.

## Architecture

```
                                    ┌─────────────────────┐
                                    │ carl.camp signer    │
                                    │ Ed25519 keypair     │
                                    │ (Supabase Vault)    │
                                    └────────┬────────────┘
                                             │ signs
                                             ▼
┌──────────────┐   GET /api/platform/        ┌────────────────────────┐
│ carl-studio  │   entitlements              │ /.well-known/          │
│ Entitlements │──Authorization: Bearer ────▶│   carl-camp-jwks.json  │
│   Client     │◀──── EntitlementJWT ────────│   (public-key list)    │
└──────┬───────┘     (EdDSA / 15-min exp)    └────────────────────────┘
       │ verify (pynacl) + cache (15 min)
       ▼
┌──────────────────────────┐
│ ~/.carl/                 │
│   entitlements_cache.json│  mode 0600 — atomic write via tmp+replace
│   jwks_cache.json        │  pin-on-first-use; additive rotation OK
└──────────────────────────┘
```

## Decision log (locked in v0.10.0)

| Decision | Choice | Rationale |
|---|---|---|
| Algorithm | Ed25519 (`alg: "EdDSA"`) | pynacl already in `[constitutional]` extra; node:crypto Ed25519 is built-in. |
| Key storage | Supabase Vault (private) + `entitlement_keys` table (public material) | Vault audit trail; rotation without Vercel redeploy. |
| Key rotation | Manual every 180 days | Auto-rotation needs studio CLI consent; manual is safer for v0.10. |
| JWT exp | 15 minutes (`ttl_seconds: 900`) | Short enough that revocation propagates; long enough for offline-grace overlap. |
| HTTP cache | `Cache-Control: private, max-age=600` (10 min) | Strictly less than JWT exp so a client at the cache boundary still has 5 min of token TTL. |
| Tier shape | normalized `tier: "FREE" \| "PAID"` + `tier_label` (raw 3-value) + `entitlements: [{key, granted_at}]` | Studio's binary semantic survives; carl.camp telemetry keeps fidelity. |
| Verification | local-fast-path-then-async (per AP-1 / slime-adapter memory) | `carl train --dry-run` never blocks on network. |
| JWKS staleness | pin-on-first-use; additive rotation accepted | Defends against silent kid swap; permits the planned 180-day rotation pattern. |
| Skew tolerance | ±5 minutes on `exp` and `nbf` | Standard JWT skew budget; matches typical NTP drift. |
| Entitlements cap | 100 grants per JWT | Hard cap in `signPlatformJwt`; oversized payload would blow the 600s HTTP cache budget. |
| Offline grace | 24 hours from `cached_at` | Plan-locked. Beyond 24h the cache is rejected as `EntitlementsNetworkError`. |

## API contract

### `GET https://carl.camp/api/platform/entitlements`

**Request headers:**
- `Authorization: Bearer <supabase-jwt>` — required

**Response 200:**
- `Content-Type: application/json`
- `Cache-Control: private, max-age=600`
- `Retry-After`, `X-RateLimit-*` headers from rate-limit middleware

```json
{
  "ok": true,
  "token": "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCIsImtpZCI6ImNjLTIwMjYtMDUtMDcifQ.<payload>.<sig>",
  "expires_in": 900,
  "tier": "PAID",
  "tier_label": "managed_lodge",
  "entitlements": [
    {"key": "lodge.managed_compute", "granted_at": "2026-05-07T12:00:00Z"}
  ],
  "key_id": "cc-2026-05-07"
}
```

**Error envelopes** (all wrapped by carl.camp's `fail()` helper):
- `401 unauthorized` — bearer missing or invalid
- `403 no_org_for_user` — user has no membership
- `422 entitlements_cap_exceeded` — server-side oversized list (bug, not user error)
- `429 rate_limited` — body has `retry_after_s`; response also carries `Retry-After: <seconds>` header
- `500 membership_lookup_failed | tier_lookup_failed | entitlements_lookup_failed | signing_failed`
- `503 no_active_key` — no row in `entitlement_keys WHERE active=true` (operator action needed)

### JWT shape

**Header:**
```json
{"alg":"EdDSA","typ":"JWT","kid":"cc-2026-05-07"}
```
- Field order is byte-exact via hand-formatted JSON literal in `signPlatformJwt`.

**Payload:**
```json
{
  "iss": "https://carl.camp",
  "aud": "carl-studio",
  "sub": "<user-uuid>",
  "org_id": "<org-uuid>",
  "tier": "FREE | PAID",
  "tier_label": "free | managed_payg | managed_lodge",
  "entitlements": [{"key": "...", "granted_at": "<iso8601>"}],
  "iat": 1715000000,
  "nbf": 1715000000,
  "exp": 1715000900,
  "jti": "<uuidv4>"
}
```

- `iat`, `nbf`, `exp` are integer seconds since epoch (`Math.floor`, not floats).
- `nbf == iat` (no skew on issuing side; verifier handles skew).
- `jti` is `crypto.randomUUID()` — fresh per token, supports replay detection.

**Signature:**
- Bytes: `Ed25519(privKey, utf8(`${b64urlHeader}.${b64urlPayload}`))`
- Encoded: `base64url` (no padding)

### `GET https://carl.camp/.well-known/carl-camp-jwks.json`

**Response 200:**
- `Content-Type: application/jwk-set+json`
- `Cache-Control: public, max-age=3600, s-maxage=3600`

```json
{
  "keys": [
    {"kty":"OKP","crv":"Ed25519","kid":"cc-2026-05-07","alg":"EdDSA","use":"sig","x":"<32-byte-pub-base64url>"}
  ]
}
```

- `vault_secret_name` and `active` columns are intentionally NEVER exposed.
- Retired keys (`retired_at IS NOT NULL`) are filtered out so a verifier never accepts a JWT signed by a rolled key.

## Implementation map

### carl.camp side (TypeScript)

| File | Role |
|---|---|
| `supabase/migrations/042_entitlement_keys.sql` | Table + `vault_read_secret(p_name)` SECURITY DEFINER RPC. Service-role-only mutation. |
| `scripts/bootstrap-ed25519-key.ts` | Operator runbook: generate keypair, print SQL for Supabase SQL editor (default dry-run); `--apply` is best-effort. |
| `src/lib/platform-jwt.ts` | `signPlatformJwt(args)` — Ed25519 signer with 5s in-mem key cache. Hand-formatted JSON header for byte-exact field order. |
| `src/app/api/platform/entitlements/route.ts` | `GET` handler: auth → rate limit (30/min) → membership → tier → entitlements → sign → respond. |
| `src/app/.well-known/carl-camp-jwks.json/route.ts` | `GET` handler: JWKS list of non-retired keys. Next.js 16 `"use cache"` + `cacheLife({revalidate: 3600})`. |
| `tests/lib/platform-jwt.test.ts` | 10 vitest stubs incl. round-trip verify, jti uniqueness, cache hit, no-key 503, PEM-non-leak, oversized-array cap. |
| `tests/api/jwks.test.ts` | 6 vitest stubs incl. retired-key filter and `vault_secret_name` non-leak. |
| `tests/api/platform-entitlements.test.ts` | 9 vitest stubs incl. tier projection, full claims, cap-exceeded, real-keypair round-trip. |

### carl-studio side (Python)

| File | Role |
|---|---|
| `packages/carl-core/src/carl_core/errors.py` | 5 new error subclasses (`carl.gate.tier_remote_mismatch`, 4× `carl.entitlements.*`). |
| `src/carl_studio/entitlements.py` | `EntitlementsClient` — `fetch_remote`, `verify_jwt`, `fetch_jwks`, `cache_get/set`, `is_offline_grace_valid`. Module-level `default_client()`, `fetch_remote_async()`. |
| `src/carl_studio/tier.py` | `tier_gate(..., verify_remote=False)` extension. Local-fast-path-then-async ladder. |
| `src/carl_studio/cli/entitlements_cmd.py` | `carl entitlements show [--json] [--refresh]`. |
| `src/carl_studio/cli/wiring.py` | Registers `entitlements_app` under root. |
| `src/carl_studio/cli/entry.py` | `entitlements` added to `REGISTERED_SUBCOMMANDS`. |
| `src/carl_studio/cli/startup.py` | `doctor` payload gains `entitlements: {status, age_s, key_id}` block. |
| `tests/test_entitlements.py` | 11 pytest stubs. |
| `tests/test_tier_resolver.py` | 8 new tests for `verify_remote=True`. |
| `tests/test_cli.py` | 5 new tests for `carl entitlements show` + doctor integration. |

## verify_remote ladder (the local-fast-path doctrine)

```
@tier_gate(Tier.PAID, feature="train.slime.managed", verify_remote=True)
def some_paid_function(...):
    ...

# When called:
1. Existing local BaseGate runs first.
2a. Local ALLOWS:
     ─ schedule fetch_remote_async(jwt) on a background thread (fire-and-forget)
     ─ return fn(*args, **kwargs)  # Caller never waits on the network.
     ─ If the async verify finds a mismatch, it logs to InteractionChain;
       the next call sees the updated cache and surfaces the deny then.
2b. Local DENIES:
     ─ try EntitlementsClient.cache_get()
     ─ if cache says PAID + has matching grant + within 24h grace:
         emit_gate_event(kind="tier_remote_grace", feature, key_id, cached_at)
         return fn(*args, **kwargs)  # Cache override (off-network, fast).
     ─ otherwise raise TierGateError as before.
```

Security guarantee: NO cache OR cache without matching grant OR cache beyond 24h → hard deny. The cache override is the *only* path that softens a local deny, and it's bounded by the offline-grace window.

## Cache layout

### `~/.carl/entitlements_cache.json` (mode 0600, atomic via tmp+rename)

```json
{
  "tier": "PAID",
  "tier_label": "managed_lodge",
  "entitlements": [{"key": "lodge.managed_compute", "granted_at": "2026-05-07T12:00:00Z"}],
  "cached_at": "2026-05-07T12:00:00Z",
  "expires_at": "2026-05-07T12:15:00Z",
  "key_id": "cc-2026-05-07",
  "org_id": "<uuid>",
  "sub": "<uuid>",
  "jwt": "<raw-token>"
}
```

- `cache_get()` parses + validates; bad JSON or missing fields raises `EntitlementsCacheError` and renames the file to `entitlements_cache.corrupt-<ts>.json` for forensics.
- Mode 0600 is set on the *tmp* file before write, so a crash mid-write doesn't leak readable secrets at the umask default.

### `~/.carl/jwks_cache.json` (mode 0600)

```json
{
  "keys": [{"kty":"OKP","crv":"Ed25519","kid":"cc-2026-05-07","alg":"EdDSA","use":"sig","x":"..."}],
  "fetched_at": "2026-05-07T12:00:00Z",
  "fingerprint": "sha256:..."
}
```

Pin-on-first-use:
- First fetch records `fingerprint = sha256(json.dumps(keys, sort_keys=True))`.
- Subsequent fetches with a different fingerprint → check whether the cached kids are a subset of the new kids (additive rotation).
- Additive: accept silently.
- Non-additive (silent swap): raise `JWKSStaleError`.

## Error code reference

| Code | Class | Meaning | Retry? |
|---|---|---|---|
| `carl.gate.tier_remote_mismatch` | `RemoteEntitlementError` | Local PAID, remote signed FREE for this feature | No — caller should deny. |
| `carl.entitlements.network_unavailable` | `EntitlementsNetworkError` | carl.camp unreachable AND offline-grace expired | Yes (with backoff). |
| `carl.entitlements.signature_invalid` | `EntitlementsSignatureError` | JWT didn't verify against JWKS | No — possible tamper. |
| `carl.entitlements.cache_corrupt` | `EntitlementsCacheError` | `~/.carl/entitlements_cache.json` unparseable | Yes — refetch. |
| `carl.entitlements.jwks_stale` | `JWKSStaleError` | JWKS fetch failed AND cached JWKS doesn't contain JWT's kid | Yes (with backoff). |

## CLI surface

```
$ carl entitlements show
entitlements cache: fresh
  tier         : PAID (managed_lodge)
  key_id       : cc-2026-05-07
  cached_at    : 2026-05-07T12:00:00+00:00
  expires_at   : 2026-05-07T12:15:00+00:00
  age          : 42s
  entitlements (1):
    - lodge.managed_compute (granted 2026-05-07T12:00:00+00:00)

$ carl entitlements show --json
{ ... }

$ carl entitlements show --refresh
# Forces a fetch first, then prints. Best-effort: prints warning and falls
# through to cache if refresh fails.

$ carl doctor --json | jq .entitlements
{
  "status": "fresh",
  "age_s": 42,
  "key_id": "cc-2026-05-07"
}
```

Status taxonomy:
- `fresh` — `cached_at` within 15 min (cache is canonical)
- `stale` — older than 15 min, ≤ 1 hr (network may help)
- `offline_grace` — within 24 hr but past 1 hr; network down → cache is truth
- `offline_grace_expired` — beyond 24 hr → cache rejected
- `missing` — no cache file
- `corrupt` — cache file unparseable
- `unavailable` — `carl_studio.entitlements` module not importable (defence in depth in `carl doctor`)

## Operator runbook

### First-time bootstrap

1. Apply `042_entitlement_keys.sql` to prod via Supabase MCP `apply_migration` or the dashboard SQL editor.
2. Run `bun run scripts/bootstrap-ed25519-key.ts` (dry-run by default). Default `kid = cc-YYYY-MM-DD` UTC; override via `--kid <name>`.
3. Paste the printed SQL into the Supabase SQL editor:
   - `SELECT vault.create_secret('<priv-pem>', 'carl_camp_jwt_priv_<kid>')`
   - `INSERT INTO entitlement_keys (...) VALUES (...)`
4. Verify: `curl https://carl.camp/.well-known/carl-camp-jwks.json | jq .` returns the new kid.
5. From a studio install: `carl entitlements show --refresh` should print a fresh entry.

### 180-day rotation

1. Generate a new keypair with a new `kid` (`bun run scripts/bootstrap-ed25519-key.ts --kid cc-rotate-N`).
2. Insert into `entitlement_keys` with `active = false` (additive — both keys in JWKS).
3. Wait for studio JWKS caches to refresh (1 hr).
4. `UPDATE entitlement_keys SET active = false WHERE kid = '<old-kid>'; UPDATE ... SET active = true WHERE kid = '<new-kid>';`
5. Wait `15 min + 5 min skew` for outstanding JWTs to expire.
6. `UPDATE entitlement_keys SET retired_at = now() WHERE kid = '<old-kid>'`.
7. Old key disappears from JWKS on the next 1-hr cycle.

`vault_read_secret('carl_camp_jwt_priv_<old-kid>')` continues to work as long as the secret remains in Vault — clean up at your discretion.

## Verification

- `pytest tests/test_entitlements.py tests/test_tier_resolver.py tests/test_cli.py -k 'entitlements or doctor or verify_remote'` — all green
- `bun run test tests/api/platform-entitlements.test.ts tests/api/jwks.test.ts tests/lib/platform-jwt.test.ts` — all green
- `curl https://carl.camp/.well-known/carl-camp-jwks.json | jq '.keys[].kid'` — shows the active kid
- `carl entitlements show --refresh` — round-trips through the issuer and prints `fresh`

## References

- carl-studio CLAUDE.md "Deferred roadmap → v0.10 remote-tier verification" — the original spec sentence
- `docs/v10_master_plan.md` — historical Fano-consensus record
- `/Users/terminals/.claude/plans/put-together-a-plan-sleepy-crystal.md` — implementation plan (Phases A, B, C)
