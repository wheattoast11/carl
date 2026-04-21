---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.9.0
---

# carl.camp agent — consolidated activation prompt (v0.9.0)

Single paste-ready brief covering platform integration, content, contest,
feature activation, and rollout sequencing. Supersedes the earlier
narrower `docs/carlcamp_agent_handoff_prompt.md` (platform-only).

Copy everything between the `===BEGIN===` and `===END===` markers.

===BEGIN===

# carl.camp v0.9.x — EML × Resonants full activation brief

**To:** carl.camp platform agent
**From:** Tej Desai + carl-studio session (Claude Opus 4.7, 1M context)
**Date:** 2026-04-20
**Upstream release:** carl-studio v0.9.0 — shipped, tagged, pushed to
`github.com/wheattoast11/carl` at commit `89baa83`
**Your mandate:** own everything platform-side for EML × Resonants.
That includes integration plumbing, landing-page content, the first
contest mechanic, feature flags, activation sequencing, and viral
surface. You have authority to ship what you ship; flag back anything
that requires carl-studio or Tej input.

This is one brief for one agent — but it's four jobs bundled
(integration, content, contest, activation). Do them in priority
order. If you hit a contradiction with anything else you've been
told, **this brief wins**. If you hit a contradiction with carl-studio
code, **code wins** — ping back for the brief to be updated.

## 0. The 60-second pitch (own this for content work)

CARL just shipped a new kind of AI object called a **Resonant**.
It's a ~300-byte signed artifact you can train in 30 seconds on a
laptop, compose like lego, and deploy at the edge in microseconds.
Each Resonant is a symbolic expression — you can print it as a
formula. It's audit-grade by construction, marketplace-tradeable,
and survives being unplugged because its memory IS its math.

The math is grounded in a paper (Odrzywolek 2026, arXiv:2603.21852):
one binary operator `eml(x, y) = exp(x) - ln(y)` plus the constant
`1` is enough to express every elementary function when you nest
them up to 4 deep. CARL turns that theorem into a shippable object.

This is not a replacement for transformers. It's the tiny,
signable, marketplaceable sibling they were waiting for.

## 1. What to read first (in order)

1. `docs/carlcamp_eml_integration_brief.md` — full technical spec for
   platform plumbing (Supabase schema, x402 endpoints, a2a cards, MCP
   tools, provider hooks). This is your source of truth for §4.
2. `docs/summercamp_eml_ux.md` — the UX surface the web client will
   consume. Informs what endpoints are needed and how error states
   surface in the app.
3. `docs/hardware_interface_spec.md` — attestation flow and device
   registration; dormant until hardware exists but the software
   surface ships now with a fallback for software-only mode.
4. `docs/eml_wire_format.md` — Python ↔ Rust wire bijection. Platform
   stores Python-flavored `.emlt` blobs (with `b"EML\x01"` magic, u16
   input-dim, inline f64 constants). Do NOT attempt to transcode to
   the Rust wire format server-side.
5. `packages/carl-core/src/carl_core/eml.py` + `resonant.py` — the
   canonical Python object shapes you are persisting.
6. `papers/eml-symbolic-witness.md` in the observable-computation
   repo — the honest-caveats section (§7) tells you what the platform
   MUST NOT promise. Read before writing marketing copy.
7. `docs/v0_9_deferred_items.md` §2 — your queue.

## 2. Success definition (what "done" looks like)

- A logged-in user can train a Resonant end-to-end in <30 seconds
  and see it persisted + marketplace-listable.
- A buyer can purchase and download a Resonant via x402 with
  signature verification before delivery.
- Two agents discover each other's Resonants via a2a and compose.
- A provider run completes, and the fitted Resonant posts back to
  the user's account automatically (opt-in).
- The landing page converts a cold visitor to a waitlist signup at
  >= 5% with zero paid traffic.
- The first contest ("Resonant Fit-Off") has a working leaderboard
  and at least one submission per week post-launch.
- All of this without shipping a signature-verify bypass, a depth-5
  tree, or a "governance primitive" claim in marketing copy.

## 3. Hard non-negotiables (memorize)

- **Signature verify on EVERY insert.** No exceptions. Pollution is
  a data-quality incident, not a bug.
- **Depth ≤ 4 enforced server-side.** Decode the tree, check depth,
  reject if over. Client might lie.
- **MIT/BUSL boundary respected.** Never embed `terminals_runtime`
  code in the public platform service. Private fitter runs on
  hardware-attested compute only.
- **Consent flags default OFF.** `publish_marketplace`,
  `bioreaction_logging`, `hardware_enabled` require explicit opt-in.
- **Rate limits enforced server-side.** 10 fit/hr free, 100 fit/hr
  paid. Check tier before any private-runtime round-trip.
- **No secrets in logs.** Ever. The user's secret component of the
  HMAC key never hits your database or observability stack.
- **No "governance primitive" framing in copy.** EML is a
  log-barrier reward shape with tropical-algebra lineage. It is NOT
  a proven governance primitive. Paper §7 is explicit — read it.
- **Content-hash is deterministic.** Collision = escalation, not
  silent dedupe.

## 4. Platform integration (Sprint 1–3)

### Sprint 1 — block everything else on this

1. **Supabase schema + RLS** per `carlcamp_eml_integration_brief.md`
   §2.a. Four tables: `resonants`, `constitutional_ledger_blocks`,
   `eml_marketplace`, and a `training_runs.eml_head_id` column.
   Index `resonants.hash`. Index `resonants.user_id`. Index
   `eml_marketplace.tier`.
2. **Upload endpoint** `POST /api/resonants` — multipart body with
   metadata JSON + `.emlt` blob. Verify signature server-side using
   the public HMAC-verify helper (NOT the private fitter). Reject
   depth > 4 with `carl.eml.depth_exceeded`. Reject bad signature
   with `carl.eml.attestation_failed`. Insert only on pass.
3. **Purchase endpoint** `POST /api/resonants/{id}/purchase` —
   x402 settlement, returns blob + platform-countersignature header.
   10% platform fee, 90% to seller. Band: $0.50–$5.00.

### Sprint 2

4. **Ledger append** `POST /api/ledger/append` — signed
   `LedgerBlock` payload. Verify prev-hash matches head. Reject
   broken chain with `carl.constitutional.chain_invalid`.
5. **a2a AgentCard extensions** per brief §2.c: four new nullable
   fields (`supports_eml`, `resonant_ids`, `eml_policy_hash`,
   `eml_depth_cap`). Backward-compat via default-null.
6. **Discovery filter** `GET /api/agents?supports_eml=true&domain=X`
   — GIN index on `agents.resonant_ids` for array membership lookup.

### Sprint 3

7. **MCP tool descriptors** per brief §2.d for `eml_evaluate`
   (public) and `eml_fit_request` (routes to private runtime,
   hardware attestation required, tier-gated).
8. **Provider webhook** `POST /api/hooks/run_complete` per brief
   §2.e. When a training run completes, the provider posts the
   fitted Resonant hash + signature. Platform inserts into the
   marketplace iff the user opted in to `publish_marketplace`.
9. **Observability** — Prometheus counters: `carl_eml_fits_total`,
   `carl_eml_evals_total`, `carl_resonant_purchases_total`,
   `carl_ledger_blocks_total`. Grafana board with heartbeat,
   fit-rate, and marketplace volume panels.

## 5. Landing page + content

### 5.1 Tagline shortlist (pick two to A/B)

- "AI in 300 bytes. Auditable. Signable. Yours."
- "Train an agent on your laptop in 30 seconds. Deploy it on a
  thermostat in microseconds."
- "The 300-byte AI. Ship a formula instead of a model."
- "Your model as a signed artifact. Not a black box."
- "Resonants: the tiny symbolic agents that fit in a text message."

Do NOT use: "governance", "constitutional AI" (Anthropic owns that
framing), "matter-antimatter" (dropped — the paper explicitly
disclaims).

### 5.2 Hero section

One sentence: "CARL Resonants are signed 300-byte agents you can
train in 30 seconds and evaluate in microseconds."

Two bullets under it:
- **Auditable by construction** — every Resonant prints as a
  closed-form math expression. Attach it to a PR. Read it at a
  glance.
- **Composable** — `compose(r1, r2)` is itself a Resonant. Build up
  skills the same way you compose functions.

CTA: "Train your first Resonant" (→ 30-second fit-a-sine wizard).

### 5.3 How-it-works section (3 steps, keep literal)

1. **Upload or generate data.** 100 rows minimum. CSV or JSON.
2. **Pick a depth.** 1 (simplest), 2 (useful), 3 (rich), 4 (max).
   Most users want 2 or 3.
3. **Click train.** Adam on the tree. Convergence in under a
   second. Your Resonant is signed, hashable, and shippable.

### 5.4 Honest limits (ship this, don't hide it)

A "What Resonants are NOT" section, grounded in paper §7:

- Not a general LLM. Resonants solve narrow, elementary tasks.
- Not a governance primitive. They're reward shapes and skill
  modules.
- Not a replacement for transformers. They sit on top, beside, or
  beneath a transformer in your stack.
- Not infinite depth. Adam reliably trains depth ≤ 4. Depth 5 is
  lottery; depth 6 has a documented 0/448 random-init failure rate.

Linking this section directly from the hero builds trust. Do it.

### 5.5 Social proof slots

- "Built on Odrzywolek 2026 (arXiv:2603.21852)."
- "Companion paper: Symbolic Coherence Witnesses via Elementary
  Function Trees" (link to Zenodo when published).
- "Open-sourced MIT. The fitter is BUSL-1.1, change-date 2030-04-09."

### 5.6 Docs site

Mirror `docs/summercamp_eml_ux.md` as the user-facing docs. Keep
paper-quality language; no emoji; code examples on every page.
Every CLI example from CLAUDE.md renders as a copy-paste code
block.

## 6. Contest + community activation

### 6.1 Launch mechanic: "Resonant Fit-Off"

A weekly challenge. Format:

1. **Monday**: platform posts a target function (e.g., "fit
   `f(x) = 2·sin(x) + cos(3x)` on `x ∈ [0, 2π]`").
2. **Monday–Friday**: users submit Resonants. Each submission is
   scored on: (a) MSE on hidden test set, (b) tree depth (smaller
   wins tie-breaks), (c) speed to convergence.
3. **Friday**: leaderboard snapshots, top-3 get x402 prize splits
   from a seed pool + optional community-staked pool.
4. **Saturday**: winning Resonants publish to marketplace with
   "Week N winner" badge. Free-tier users can browse; paid-tier
   users can buy.

Implementation surface:
- `POST /api/contests/{id}/submit` — accepts `.emlt` blob +
  metadata. Same signature verify as normal upload.
- `GET /api/contests/{id}/leaderboard` — public endpoint, sorted.
- Deterministic eval harness on a fixed test set (use
  `carl_core.eml.EMLTree.forward_batch`).
- Weekly cron to close + snapshot + announce.

### 6.2 Community layer

- Public profile pages: `carl.camp/@username` showing their
  Resonants, composition tree, and heartbeat uptime.
- Follow graph (who's training, who's composing).
- "Compose with @username's tree" button that pre-fills the
  composition UI with their published Resonant on the left side.
- No comments, no replies, no reaction emojis — keep the surface
  signal-heavy. If you want a social layer, add a /showcase feed
  that ranks by composition count, not likes.

### 6.3 Developer contest (secondary)

A one-time hackathon-style launch:
- Prize: $5K + free paid tier for a year.
- Categories: smallest useful Resonant, most-composed Resonant,
  best domain application, most creative use of the ledger.
- Judges: Tej + 2 external reviewers.
- Submission closes 4 weeks post-launch.

## 7. Feature flags + activation sequencing

Use a flags table in Supabase. Flip flags via admin-only SQL
procedures. No runtime config on user clients — all server-driven.

Flag list:
- `eml_upload_enabled` (bool, per-tier) — gate uploads
- `eml_marketplace_enabled` (bool) — gate buying
- `eml_composition_enabled` (bool) — gate `compose_resonants`
- `eml_ledger_enabled` (bool) — gate constitutional writes
- `eml_contest_enabled` (bool, per-contest-id) — gate submissions
- `eml_hardware_enrollment_enabled` (bool) — gate `attest-device`
  endpoint (default OFF until v0.10)
- `eml_provider_webhook_enabled` (bool) — gate auto-publish from
  provider runs

Rollout phases:

**Phase A — private alpha (week 0, ~10 users).**
All flags OFF by default. Flipped ON per-user-id via admin SQL.
Tej's account + 5–10 handpicked users. Goal: exercise every
endpoint, catch the 80% of bugs. No landing page yet, no contest.

**Phase B — private beta (weeks 1–3, ~100 users).**
Waitlist signup enabled on a minimal landing. Invites via magic
link. `eml_upload_enabled=true` for invitees. Marketplace still
OFF — users can train + save, not buy. First Resonant Fit-Off
runs for invitees only, "warm-up week" copy.

**Phase C — public launch (week 4).**
Landing page fully live with hero + how-it-works + honest-limits.
Free tier open. `eml_upload_enabled` and
`eml_marketplace_enabled` both ON for free tier with 10 fit/hr
rate limit. First public Resonant Fit-Off week. Composition UI
visible but behind the free-tier rate limit so users try + hit
it + upgrade.

**Phase D — paid tier + provider webhooks (week 6).**
`eml_provider_webhook_enabled=true`. Paid tier at $20/mo or $0.10
per fit (metered). Provider hooks auto-publish fitted Resonants
for paid users. Hardware-attested lane dormant.

**Phase E — hardware lane (v0.10.0+).**
`eml_hardware_enrollment_enabled=true`. Attestation flow live.
Hardware-attested trust tier visible on marketplace. Prioritize
once the USB/wristband/cap physically exist.

## 8. Viral + FOMO angles (content work)

### 8.1 The "tweet-sized AI" demo

Record a 15-second loop: user pastes a `.emlt` blob from a tweet
into CARL, it decodes, evaluates, visualizes. Post on launch
day. Title: "This tweet contains an AI."

### 8.2 The composition gallery

A live-updating page showing composed Resonants as a directed
graph. "Alice's `sine_fit` + Bob's `phase_shift` = Carol's
`modulated_sine`." Every node is clickable and plays a 3-second
formula-reveal animation. Ship this by end of Phase B.

### 8.3 The "hard-to-fake" badge

A Resonant's identity hash is verifiable client-side. Ship a
Twitter/Bluesky bot that renders any shared `.emlt` link as a
preview card with the formula, size, and a green "signature
verified" badge (or red "tampered"). Viral by construction —
everyone wants the verified card.

### 8.4 The "your model, on a wristband" angle

Ship a fake-if-you-must product page for the USB+wristband+cap
device tree at `carl.camp/hardware` with a big "WAITLIST" button.
Do NOT take payment. Do NOT ship a fake ship-date. Build
anticipation via concept renders + spec + the software-only mode
that's actually working today. This converts hardware hype into
software adoption — the bridge is that every software user today
is a waitlisted hardware user tomorrow.

### 8.5 Scientific credibility

Quote the paper in the footer. Link to the carl-core source on
GitHub. Offer an "open docs" page that literally renders the
paper in-browser with live-computable formulas. This is what
separates CARL from hype-AI — you can read the math.

## 9. Questions to confirm with Tej before Sprint 1

1. Which x402 facilitator is in staging? (Coinbase CDP / other)
2. Stripe Connect for the 90% seller cut, or direct payouts?
3. Admin-tier hardware-attestation lane — build end-to-end, or
   wait for v0.10 hardware?
4. Staging subdomain or go straight to production for Phase A?
5. Who owns Prometheus + Grafana — platform team or devops?
6. Can we break existing AgentCard schema via nullable additions,
   or do we need a compat shim?
7. Seed pool size for the first Resonant Fit-Off?
8. Legal review on "CARL Resonants are auditable by construction"
   copy — is that strong enough to survive a trademark challenge
   from Anthropic's "Constitutional AI"?
9. Which analytics stack? (PostHog / Mixpanel / custom)
10. Should `compose_resonants` run server-side or stay client-only
    in Phase B/C? (Server-side is cheaper per user; client-side
    surfaces the composition math in the browser which is
    visually better.)

## 10. Out of scope (do NOT build in v0.9.x)

- On-chain ledger inscription (Ethereum/Solana bridge). Ledger is
  a Supabase hash-chain for now; on-chain is a separate initiative.
- ZK-SNARK proof generation for policy eval. Paper sketches the
  architecture; circuit not written.
- The full drag-drop magma-tile UI. Phase A/B uses a simpler
  JSON-editor; tiles come later.
- Hardware device enrollment in production. Phase E only.
- Token / NFT anything. Resonants have a hash; that's all.

## 11. How to check in

Write back after each sprint with:
- Migration IDs + endpoint URLs deployed
- Any blocker on §9 questions
- Divergences from this brief (flag for discussion)
- Metrics: signup rate, fit rate, marketplace volume, leaderboard
  submissions

If something here contradicts a newer file in `docs/`, the newer
file wins — ping back to reconcile.

Ship clean. Sign everything. Keep the math visible. Respect the
depth cap. Make it feel fun without making it feel hyped.

===END===

## Usage notes for Tej

- Paste everything between the markers into the carl.camp agent's
  first turn.
- Expect Sprint 1 to take 1–2 sessions depending on Supabase
  starting state.
- Have answers to §9 questions ready — they'll block early
  progress.
- The content in §5–§8 is editorial — the carl.camp agent can
  refine copy but the frame (tweet-sized AI, 300-byte signed
  artifact, auditable by construction, composable like lego)
  should hold.
- Phase A–E timing is a recommendation, not a contract. Let the
  agent compress or extend based on what ships.

## Changelog

- 2026-04-20 — initial consolidated prompt authored. Supersedes
  `docs/carlcamp_agent_handoff_prompt.md` (narrower platform-only
  version kept for reference).
