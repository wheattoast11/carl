---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.9.0
---

# Handoff prompt for the carl.camp platform agent

This file is the paste-ready prompt to hand to the **carl.camp platform
agent** (a different agent session working on Supabase + x402 + a2a +
MCP + provider integration for the CARL platform). Copy everything
between the `===BEGIN===` and `===END===` markers into that agent's
first message and it will have enough context to act.

The long-form technical spec lives at
`docs/carlcamp_eml_integration_brief.md` (1896 words, authored by
carl-studio T10 on 2026-04-20). This prompt is the activation wrapper
that tells the carl.camp agent what just changed, what to read, what
to ship first, and where to push back on unclear requirements.

===BEGIN===

# carl.camp v0.9.x — EML × Resonants integration brief

**Principal:** Tej Desai  
**Date:** 2026-04-20  
**Upstream release:** carl-studio v0.9.0 (shipped today)  
**Your role:** Deliver the platform-side integration for EML primitives and
Resonants. You own Supabase, x402 payment rails, a2a marketplace,
MCP clients, and provider/environment hooks for training runs.

## 0. What just shipped in carl-studio that you need to know

The CARL studio team just shipped a 10-team swarm adding an algebraic
primitive called **EML** — an Adam-trainable symbolic tree of depth
≤ 4 based on `eml(x, y) = exp(x) - ln(y)` (Odrzywolek 2026,
arXiv:2603.21852). The primitive is now a first-class entity in CARL:

- Public (MIT): `carl_core.eml.EMLTree` + `carl_core.resonant.Resonant`.
  A Resonant = `(EMLTree, projection matrix, readout matrix, identity hash)`.
  Tiny — 2-100 trainable params, evaluable in <10 µs on CPU, signable.
- Private (BUSL): `terminals_runtime.eml` — hardware-attested fitter,
  HMAC-SHA256 signing keyed on `hw_fingerprint XOR user_secret`.
- Paper: `observable-computation/papers/eml-symbolic-witness.md`
  (Zenodo-publishable, 6627 words + 5 figures).

Wire format for shipping a Resonant between services is called an
**`.emlt` blob** — serialized `EMLTree` + `projection` + `readout` +
Ed25519 signature. Typical size: **~300 bytes**. The format is
deterministic; two agents decoding the same bytes produce bit-identical
output via the Rust softfloat in `terminals-core/primitives/softfloat.rs`.

## 1. What you MUST read before writing any code

In order of importance:

1. **`docs/carlcamp_eml_integration_brief.md`** — THE technical spec
   for your work. Full Supabase schema, x402 endpoints, a2a card
   extensions, MCP tool JSONSchemas, provider webhooks. Every section
   has acceptance criteria. Treat this as your source of truth.
2. **`docs/summercamp_eml_ux.md`** — the UX surface the platform is
   serving. Informs what endpoints need to exist for the web client
   to feel drop-dead simple.
3. **`docs/hardware_interface_spec.md`** — attestation flow and
   device-to-platform registration; you will need this for `admin.attest-device`.
4. **`papers/eml-symbolic-witness.md`** (in observable-computation repo)
   — the honest-caveats section (§7) tells you the limits of what
   you're storing and what the platform should NOT promise.
5. **`carl_core/eml.py` and `carl_core/resonant.py` source** — the
   canonical Python implementations. Reading these clarifies the
   object shape you are persisting.

## 2. Priority order — what to ship first

### Sprint 1 (block everything else on this)

1. **Supabase schema migrations.** Create the four new tables from
   the brief §2.a with RLS enabled:
   - `resonants` (id uuid, user_id uuid, tree_bytes bytea, signature
     bytea, input_dim int, output_dim int, trust_tier text check in
     ('software', 'hardware'), metadata jsonb, hash text unique indexed,
     created_at timestamptz)
   - `constitutional_ledger_blocks` (user_id uuid, block_id bigint,
     prev_hash text, policy_id text, verdict float, signature bytea,
     signer_pubkey bytea, timestamp_ns bigint, PRIMARY KEY (user_id,
     block_id), CHECK block_id ≥ 0)
   - `eml_marketplace` (resonant_id references resonants.id, price_cents
     int, seller_id references auth.users, tier text, downloads bigint
     default 0, rating float)
   - `training_runs` add column `eml_head_id uuid nullable` with FK to
     resonants

2. **Upload endpoint with signature verify.** `POST /api/resonants`
   accepts a multipart body (metadata JSON + `.emlt` blob), calls the
   carl-studio `verify_signature()` helper (or reimplements the HMAC
   check server-side), rejects on mismatch with error code
   `carl.eml.attestation_failed`. Enforce depth ≤ 4 on the server
   after decoding — the Python `EMLTree.depth()` method is the
   reference.

3. **Download + purchase endpoint.** `POST /api/resonants/{id}/purchase`
   runs x402 settlement, returns the `.emlt` blob + a platform
   signature header. Charge 10% platform fee; seller gets 90%.
   Typical price band: $0.50–$5.00. Stripe Connect for payout per
   existing carl.camp payment plumbing.

### Sprint 2

4. **Constitutional ledger persistence.** Accept `/api/ledger/append`
   with a signed `LedgerBlock` payload. Verify the hash chain
   (prev_hash matches the prior block for this user). Reject with
   `carl.constitutional.chain_invalid` on break.

5. **a2a AgentCard extensions.** Add four fields
   (`supports_eml: bool`, `resonant_ids: list`, `eml_policy_hash: str`,
   `eml_depth_cap: int`) to the AgentCard schema served at
   `/api/agents/card/{id}`. Backward-compatible — absent fields default.

6. **Discovery filter.** `GET /api/agents?supports_eml=true&domain=X`
   returns agents offering compiled Resonants for the given domain
   tag. Add a Postgres GIN index on `agents.resonant_ids`.

### Sprint 3

7. **MCP tool descriptors** for `eml_evaluate` and `eml_fit_request`
   per brief §2.d. The fit tool routes to the private
   terminals-runtime fitter over a secured endpoint — hardware
   attestation required. The eval tool is public.

8. **Provider webhook** `POST /api/hooks/run_complete` (brief §2.e).
   When a training run completes, the provider posts the fitted
   resonant hash + signature. Platform inserts into marketplace if
   the user opted in (`publish_marketplace` consent flag).

9. **Observability.** Prometheus counters + Grafana board for:
   `carl_eml_fits_total`, `carl_eml_evals_total`,
   `carl_resonant_purchases_total`, `carl_ledger_blocks_total`.

## 3. Non-negotiables

- **Signature verification is required on EVERY insert.** A Resonant
  without a valid HMAC is a pollution event. Reject, do not store.
- **Depth ≤ 4 is enforced server-side.** Client might lie.
- **MIT/BUSL boundary is respected.** Do not embed any code from
  `terminals_runtime` in the public platform service. The fitter runs
  on hardware-attested compute only.
- **Consent gates:** `publish_marketplace`, `bioreaction_logging`,
  `hardware_enabled` must default OFF and require explicit opt-in.
  See `carl-studio/src/carl_studio/consent.py` for the canonical
  pattern.
- **Rate limits:** 10 fit/hr on free tier, 100 fit/hr on paid. Evals
  are unlimited within reason. Tier check before every private-runtime
  round-trip.
- **No secrets in logs.** The user_secret XOR piece of HMAC keys
  must never reach your database or logs in any form.
- **`content_hash` field is deterministic.** If you see two different
  Resonants with the same hash, that's a UUID collision or a seed
  bug in carl-studio — flag and escalate, do not silently dedupe.

## 4. Questions to confirm before Sprint 1

Ask Tej or the carl-studio team if any of these are ambiguous:

1. Which x402 facilitator is active in staging? (Coinbase CDP vs
   alternatives)
2. Stripe Connect or direct payout for the 90% seller cut?
3. Is the admin-tier hardware-attestation flow already wired to the
   platform, or do you need to build it end-to-end?
4. Does carl.camp serve traffic from staging subdomain first, or go
   straight to production?
5. What is the migration window — can you break existing agent cards
   with the new field additions (via default values) or do you need a
   compatibility shim?
6. Who owns the Prometheus + Grafana infrastructure — platform team
   or devops? Where does the board live?

## 5. Out of scope for v0.9.x platform work

Do NOT build the following yet — they are deferred to later sprints:

- On-chain inscription of ledger heads (Ethereum/Solana bridge). The
  ledger is a Supabase hash-chain table for v0.9.x; on-chain anchoring
  is a separate initiative.
- ZK-SNARK proof generation for policy eval. Architecture exists in
  the paper but circuit not written.
- Marketplace UI (drag-drop composition, heartbeat dashboard). Those
  are web-client scope, not platform-API scope. Ship the API first;
  UI consumes it.
- Hardware device enrollment flow for USB/wristband/cap. The software
  API exists (`admin.attest-device`) but no hardware exists yet to
  enroll. Stub the endpoint with a software-only fallback for now.

## 6. How to check in

Write back with:

- Which sprint you are on
- Any blocker on §4 questions
- Migration files landed (with Supabase migration IDs)
- Endpoints deployed (staging vs prod)
- Any divergence from the brief that you want to discuss

If you hit a contradiction between this prompt and
`carlcamp_eml_integration_brief.md`, **the brief wins**. If you hit a
contradiction between the brief and the actual code in
`carl-studio/packages/carl-core/src/carl_core/eml.py`, **the code
wins** — ping the carl-studio team to update the brief.

Ship it clean. No half-wired endpoints. No placeholder schema.
Respect the depth cap. Sign everything.

===END===

## Usage notes for Tej

- Paste the content between the markers into the carl.camp agent's
  first turn.
- Expect Sprint 1 to take ~1-2 sessions depending on how much Supabase
  setup is already done.
- Expect them to ask all 6 questions from §4 — have answers ready.
- If they need anything from carl-studio that isn't in the brief,
  they'll ping back; route through T10 or update the brief.

## Feedback loop

If carl.camp surfaces inconsistencies in the brief, update
`docs/carlcamp_eml_integration_brief.md` in-place and log the diff in
this file's changelog below:

### Changelog
- 2026-04-20 — initial handoff prompt authored, matching brief v1 authored
  by T10 on 2026-04-20.
