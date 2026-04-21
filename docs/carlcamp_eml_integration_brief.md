---
last_updated: 2026-04-21
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.9.0
audience: carl.camp platform agent
status: actionable brief
---

# carl.camp × EML integration brief

> **v0.9.x signing-tier note (2026-04-21 refresh).** The v0.9.0 wire
> protocol is **HMAC-SHA256 software tier**, not ed25519. Ed25519
> remains the primitive for `constitutional_ledger_blocks` (§3.2)
> and is NOT used for Resonant attestation. The hardware-attested
> variant (ed25519 + device key) is explicitly deferred to v0.10 per
> §9.3 of the agent-handoff prompt. Where this brief still reads
> "Ed25519" against `resonants` or the purchase countersig, treat
> those as forward references; the shipped contract is in
> `docs/eml_signing_protocol.md` §2.1 and §4. Surgical corrections
> below.

This is a scoped brief for the carl.camp platform agent. It tells the
platform side exactly what changes are needed to host, sell, and route
EML-trained `Resonant` artifacts produced by carl-studio clients. The
client side (carl-studio) and the private runtime (terminals-runtime)
are already shipping the primitive; everything below is the platform
work that closes the loop.

## 1 · EML in one paragraph

EML is the **exponential-minus-log magma** from Odrzywolek 2026
(arXiv 2603.21852): a single binary operation

```
eml(x, y) = exp(x) - ln(y)
S -> 1 | eml(S, S)
```

From the distinguished constant `1`, the grammar generates closed-form
expressions for `e, pi, i, +, -, *, /, ^, sin, cos, sqrt, log`. It is a
magma only (not associative, not commutative, no identity). Adam is
empirically trainable up to tree depth 4 (100% convergence at d=2,
~25% at d=3-4, 0/448 at d=6). In CARL, we wrap a fitted EML tree with
linear projection/readout matrices to get a **`Resonant`**: the
perceive → cognize → act triple that is the smallest composable agent
primitive. Resonants close under `compose_resonants(r1, r2)` as
`eml(r1.tree, r2.tree)` — which is precisely what the marketplace
monetizes.

## 2 · Split of responsibilities

| Layer                | License | Ships                                                                                                                                        |
| -------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `carl-core`          | MIT     | `carl_core.eml` primitive (`EMLNode`, `EMLTree`, `EMLOp`, `eml`, `eml_scalar_reward`), `Resonant` dataclass, `compose_resonants`, trace hash |
| `carl-studio`        | MIT     | `EMLCompositeReward`, smooth coherence gate, Constitutional ledger format, `carl publish` and `carl push resonant` CLIs                      |
| `terminals-runtime`  | BUSL    | Private fitter, closed-form eval optimizer, HMAC-SHA256 hardware-tier signer (`hw_fp XOR user_secret`), Ed25519 for ledger blocks only       |
| **carl.camp** (you)  | —       | Supabase schema, x402 settlement, marketplace API, a2a discovery, MCP tools for `eml_evaluate` / `eml_fit_request`, training-run hooks       |

**Do not** re-implement the fitter on the platform side. The platform
only moves signed blobs; the fit itself runs on the user's hardware
(CPU for depth ≤ 3, optional GPU for d=4 sweeps).

## 3 · Supabase schema changes (additive)

All changes are additive. No existing column on `org_members`,
`agent_cards`, or `training_runs` changes shape.

### 3.1 · `resonants` (new)

```sql
create table public.resonants (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid not null references auth.users(id) on delete cascade,
    tree_bytes      bytea not null,                 -- serialized EMLTree (carl_core.eml.encode_tree)
    signature       bytea not null,                 -- HMAC-SHA256 over inner tree bytes (software tier, v0.9); 32 bytes fixed
    trust_tier      text not null default 'software' check (trust_tier in ('software', 'hardware')),
    sig_public_component text not null,             -- sha256(user_secret)[0:16] hex; identity fingerprint, not the secret
    input_dim       int  not null check (input_dim  > 0),
    output_dim      int  not null check (output_dim > 0),
    projection_shape int[] not null,                 -- [k, d]
    readout_shape    int[] not null,                 -- [a, k]
    depth           int  not null check (depth between 1 and 4),
    hash            text not null,                  -- sha256 hex (stable content id)
    domain          text,                            -- free-form: "audio", "trading", "biofeedback"
    metadata        jsonb not null default '{}',
    created_at      timestamptz not null default now()
);

create unique index resonants_hash_idx   on public.resonants (hash);
create        index resonants_user_idx   on public.resonants (user_id);
create        index resonants_domain_idx on public.resonants (domain);

alter table public.resonants enable row level security;

create policy "resonants are visible to owner or if listed in marketplace"
    on public.resonants for select
    using (
        user_id = auth.uid()
        or exists (select 1 from public.eml_marketplace m where m.resonant_id = resonants.id and m.listed = true)
    );

create policy "resonants are insertable by owner" on public.resonants
    for insert with check (user_id = auth.uid());
```

### 3.2 · `constitutional_ledger_blocks` (new)

Append-only hash chain of policy decisions. Every Resonant evaluation
that happened under a policy contract produces a block.

```sql
create table public.constitutional_ledger_blocks (
    block_id     uuid primary key default gen_random_uuid(),
    user_id      uuid not null references auth.users(id) on delete cascade,
    prev_hash    text,                                 -- null for genesis block only
    policy_id    text not null,                        -- carl.yaml policy identifier
    resonant_id  uuid references public.resonants(id) on delete set null,
    verdict      double precision not null,            -- signed scalar, typically in [-5, 5]
    inputs_hash  text not null,                        -- sha256 of canonical JSON inputs
    outputs_hash text not null,
    signature    bytea not null,                       -- Ed25519 over canonical JSON of block
    timestamp    timestamptz not null default now()
);

create index ledger_user_time_idx on public.constitutional_ledger_blocks (user_id, timestamp);
create index ledger_prev_hash_idx on public.constitutional_ledger_blocks (prev_hash);
```

### 3.3 · `eml_marketplace` (new)

```sql
create table public.eml_marketplace (
    resonant_id  uuid primary key references public.resonants(id) on delete cascade,
    seller_id    uuid not null references auth.users(id) on delete cascade,
    tier         text not null check (tier in ('free', 'paid')) default 'free',
    price_cents  int  not null default 0 check (price_cents >= 0),
    meter        text not null check (meter in ('per_download', 'per_evaluation')) default 'per_download',
    downloads    bigint not null default 0,
    rating       double precision,                     -- EMA over user ratings
    listed       boolean not null default true,
    listed_at    timestamptz not null default now()
);

create index marketplace_tier_idx   on public.eml_marketplace (tier);
create index marketplace_rating_idx on public.eml_marketplace (rating desc nulls last);
```

### 3.4 · `training_runs` extension (additive column)

```sql
alter table public.training_runs
    add column if not exists eml_head_id uuid references public.resonants(id) on delete set null;

create index if not exists training_runs_eml_idx on public.training_runs (eml_head_id);
```

No existing row needs backfill; column is nullable.

## 4 · x402 payment integration

carl-studio already ships `src/carl_studio/x402.py` as a pure-stdlib
HTTP 402 client gated by the `contract_witnessing` consent flag. The
platform needs one new endpoint on the facilitator path.

### 4.1 · `POST /api/resonants/purchase`

x402-flavored settlement. Typical unit price is `$0.50 – $5.00`.

**Happy path:**

1. Client sends `POST /api/resonants/purchase` with
   `{ "resonant_id": "...", "buyer_id": "..." }` and *no* payment header.
2. Platform responds `402 Payment Required` with the x402 negotiation
   body (`pay_to`, `amount`, `asset`, `network`, `nonce`, `expires_at`,
   `description`).
3. Client signs payment and re-POSTs with the `X-PAYMENT` header.
4. Platform verifies via the existing facilitator, on success returns
   `200 OK` with `Content-Type: application/octet-stream` carrying the
   signed `.emlt` blob and the platform countersignature headers
   defined in `docs/eml_signing_protocol.md` §4.1:
   `X-Carl-Platform-Countersig` (base64 HMAC-SHA256 over the §4.2
   payload), `X-Carl-Platform-Countersig-Timestamp`, and
   `X-Carl-Platform-Countersig-Txid`. Use
   `carl_core.signing.sign_platform_countersig()` as the reference
   impl; the TS mirror ships in `@terminals-tech/emlt-codec`.
5. Platform writes a marketplace-income row and increments
   `eml_marketplace.downloads`.

**Meter switch.** If `meter = 'per_evaluation'`, step 4 instead returns
a short-lived HMAC token the client presents to `/api/resonants/evaluate`
on each forward pass. The platform bills the buyer per call, settling
on a 24-hour tumbling window.

**Attestation rule.** When the buyer has admin tier, the response MUST
include a hardware attestation chain from `carl admin attest-device`
(see the hardware spec). Free-tier and paid-tier users get the
HMAC-SHA256 software-tier signature (v0.9.x). Admin-tier adds the
hardware attestation envelope on top (v0.10+).

### 4.2 · Price derivation

The seller picks price + meter at listing time. Platform takes 10%.
Use Stripe Connect for fiat payout; x402 settles on-chain. Do not
expose the fitter cost on the platform — that's user-side.

## 5 · A2A marketplace card extensions

Add these fields to `CARLAgentCard` (carl-studio already exposes the
dataclass; platform mirrors them in the `agent_cards` Supabase table):

```python
supports_eml:     bool          # default False
resonant_ids:     list[str]     # hashes the agent can serve (attached or composed)
eml_policy_hash:  str | None    # latest Constitutional-ledger block hash
eml_depth_cap:    int           # typically 4
```

**Discovery filter.** Expose:

```
GET /api/agents/discover?supports_eml=1&domain=audio&min_rating=4
```

Results must also include `agent_cards` whose attached `resonant_ids`
have an active `eml_marketplace` listing so a buyer's client can deep-link
into purchase.

## 6 · MCP client requirements

carl-studio hosts a FastMCP server at `src/carl_studio/mcp/server.py`.
The platform should register these two new tools on the hosted MCP
bridge so agents that consume Resonants can reach them over MCP.

### 6.1 · `eml_evaluate`

Server-side EML tree evaluation for clients that lack numpy.

```json
{
  "name": "eml_evaluate",
  "description": "Evaluate a registered Resonant against an input vector.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "resonant_id": { "type": "string", "format": "uuid" },
      "observation": { "type": "array", "items": { "type": "number" } }
    },
    "required": ["resonant_id", "observation"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "action":    { "type": "array", "items": { "type": "number" } },
      "latent":    { "type": "array", "items": { "type": "number" } },
      "tree_hash": { "type": "string" },
      "billed":    { "type": "boolean" }
    }
  }
}
```

### 6.2 · `eml_fit_request`

Dispatches a remote fit job. The fitter itself is private; the platform
only brokers the request and returns a `job_id` and a WebSocket-
compatible streaming URL.

```json
{
  "name": "eml_fit_request",
  "inputSchema": {
    "type": "object",
    "properties": {
      "dataset_uri":  { "type": "string" },
      "target_depth": { "type": "integer", "minimum": 1, "maximum": 4 },
      "seed":         { "type": "integer" },
      "budget_cents": { "type": "integer" }
    },
    "required": ["dataset_uri", "target_depth"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "job_id": { "type": "string" },
      "stream": { "type": "string", "format": "uri" },
      "eta_s":  { "type": "number" }
    }
  }
}
```

## 7 · Provider / training-run hooks

Dynamic interactive research environments (Sandbox, HF Spaces, Colab,
on-prem) need a consistent way to register a freshly fitted Resonant.

```python
# pseudocode executed by provider at run completion
from carl_studio.hooks import on_run_complete

@on_run_complete
def publish_resonant(run):
    if run.eml_head_id:
        push_to_marketplace(
            resonant_id=run.eml_head_id,
            tier=run.user.tier,
            listed=run.user_consent.publish_marketplace,
        )
```

The platform must:

- Accept webhook `POST /api/hooks/run_complete` with body
  `{ run_id, resonant_hash, signature }`.
- Verify the signature against the user's registered device key.
- Insert a row into `resonants` **only if** the uploaded signature
  verifies; otherwise respond `422` with code `carl.eml.attestation_failed`.
- Offer a Grafana board stub at `/dashboards/resonants/{id}` showing
  fitness trajectory: loss, R, phi_mean, lyapunov proxy over epochs.

## 8 · Constraints (non-negotiable)

1. **Respect MIT / BUSL split.** The platform must never host the
   private fitter source. `eml_fit_request` routes to the user's own
   compute; the platform is a dispatcher, not the trainer.
2. **Signature-gated uploads.** Every `resonants` insert must verify
   the HMAC-SHA256 signature over the **inner tree bytes** (see
   `docs/eml_signing_protocol.md` §2.1) using the user's `user_secret`
   obtained from the `X-Carl-User-Secret` header on the upload
   request. Reject unsigned or mis-signed uploads at the edge function
   with `carl.eml.attestation_failed`. Projection + readout matrices
   are not part of the signed scope; their integrity is enforced via
   the Resonant identity hash (content-addressing) at insert time.
3. **Hardware-attested downloads for admin tier.** Admin-tier buyers
   receive an attestation chain with the blob; free/paid do not.
4. **Rate limits.** `eml_fit_request`: 10/hr on free, 100/hr on paid,
   unlimited on admin. `eml_evaluate`: 1k/hr on free, 100k/hr on paid.
5. **Privacy-first.** No Resonant payload is indexed by content in
   search until the user has flipped `consent.publish_marketplace`.
6. **Depth cap.** Reject any tree with depth > 4 with
   `carl.eml.depth_exceeded` — matches `carl_core.eml.MAX_DEPTH`.

## 9 · Example API shapes

### 9.1 · Upload

```http
POST /api/resonants
Content-Type: application/octet-stream
Authorization: Bearer <supabase_jwt>
X-Carl-User-Secret: <base64(user_secret, 16-64 bytes)>
X-Carl-Input-Dim: 3
X-Carl-Output-Dim: 2
X-Carl-Projection: <base64(projection_matrix_bytes)>
X-Carl-Readout: <base64(readout_matrix_bytes)>

<raw .emlt envelope bytes: EMLT magic | version | inner tree | 32-byte HMAC sig>
```

The `X-Carl-User-Secret` header MUST be added to the logger redaction
list, never written to the database, and used only transiently for the
HMAC verify (GC'd after the `verifySoftware()` call). TLS-only
transport. Identity fingerprint stored in the DB is
`sha256(user_secret)[0:16]` hex — a 32-char one-way fingerprint that
lets consumers recognize the signing identity without ever seeing the
secret. **Upgrade path (v0.10):** `sig_public_component` becomes the
raw ed25519 pubkey hex; schema unchanged.

### 9.2 · Evaluate

```http
POST /api/resonants/abcd.../evaluate
Content-Type: application/json

{ "observation": [0.4, 0.1, 0.7] }
```

Response:

```json
{
  "action":   [0.81, -0.12],
  "latent":   [0.55, -0.02],
  "tree_hash": "3f1c...",
  "billed":    true
}
```

### 9.3 · List marketplace

```http
GET /api/marketplace/resonants?domain=audio&tier=paid&sort=rating
```

Response envelope mirrors existing `agent_cards` list endpoints for
consistency (same pagination, same `next_cursor` token).

## 10 · Migration notes

- **Zero breaking changes.** Everything is additive: new tables, new
  endpoint, new MCP tools, new `agent_cards` fields, new
  `training_runs.eml_head_id` column.
- **Rollout order.**
  1. Deploy migrations 3.1–3.4 (can go in one pack).
  2. Ship `/api/resonants` upload, evaluate, purchase endpoints.
  3. Register MCP tools on the hosted bridge.
  4. Enable agent-card extensions in discovery.
  5. Flip the hook endpoint on — at this point carl-studio v0.9.0
     clients start publishing.
- **Backfill.** None. Existing users get an empty `resonants` list
  until they fit their first tree.
- **Observability.** Add Prometheus counters
  `carl_eml_uploads_total`, `carl_eml_purchases_total`,
  `carl_eml_evaluate_ms` (histogram), `carl_eml_fit_requests_total`.
- **Rollback.** The marketplace listing can be disabled by flipping
  `eml_marketplace.listed = false` rows; Resonants remain private.
  Training runs continue to work with `eml_head_id = null`.

## 11 · Open questions for principal

- Payout rail for fiat sellers: Stripe Connect vs direct ACH?
- Default price: $0.50 flat, or seller-set with a floor of $0.10?
- Rating moderation: do we ship reviews in v0.9.0 or defer?
- Should `eml_fit_request` accept private dataset URIs (HF private
  repo, S3 presigned), or require public first?

Drop answers into `docs/v10_master_plan.md` once decided.
