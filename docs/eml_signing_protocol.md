---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.9.0
---

# EML Signing Protocol v1

Canonical spec for how Resonants (and LedgerBlocks) are serialized,
signed, and verified across language boundaries. This is the
source-of-truth doc for platform integrators (carl.camp), third-party
clients, and the TypeScript `@carl/emlt-codec` package.

If this file disagrees with the Python source in
`packages/carl-core/src/carl_core/eml.py` or
`packages/carl-core/src/carl_core/signing.py`, **the Python source
wins** — file a PR to update this doc.

## 1. Two wire formats

There are two distinct byte formats in play. Do not confuse them.

### 1.1 Inner tree format (canonical for hashing)

Produced by `EMLTree.to_bytes()` (carl_core) and
`EmlTree::encode()` (terminals-core Rust, different layout — see §1.3).
The Python format is the canonical one for platform storage.

```
byte offset   | content
--------------|--------------------------------------------
0..3          | MAGIC: b"EML\x01" (four bytes, literal)
4..5          | input_dim: uint16 little-endian
6..N          | postfix tag stream; tags below
```

Postfix tag stream (a sequence of nodes read in postorder):

| tag byte | name  | payload                              |
|----------|-------|--------------------------------------|
| 0x01     | CONST | float64 little-endian (8 bytes)      |
| 0x02     | VAR_X | uint16 little-endian var_idx (2 B)   |
| 0x03     | EML   | (none — binary op, pops two children)|

Depth is computed during decode by tracking stack height; enforce
`depth <= 4` (Odrzywolek ceiling; Adam trainability degrades above).

### 1.2 Envelope format (wire-level container)

Produced by `terminals_runtime.eml.codec_impl.encode()` (signed) or
equivalently by the TS codec `@carl/emlt-codec` (unsigned or signed).

```
byte offset   | content
--------------|--------------------------------------------
0..3          | MAGIC: b"EMLT" (four bytes, literal)
4             | VERSION: 0x01
5..5+M-1      | inner tree bytes (M bytes, §1.1)
[5+M..5+M+31] | optional signature (32 bytes, §2)
```

Total size for a depth-4 tree ~ 14 (inner) + 5 (header) + 32 (sig) =
**~51 bytes**. Well under the §0 pitch's "~300 bytes" target; the
headroom is for projection + readout matrices attached server-side.

### 1.3 Rust wire format (separate, not platform-facing)

The Rust implementation in `terminals-core/src/primitives/eml.rs`
uses a different layout: magic `b"EML1"` (ASCII '1', not 0x01),
uint32 dims, and a separate constant pool with index-referenced
leaves. This is intentional divergence — see `docs/eml_wire_format.md`
for the full bijection table. **Platform always uses §1.1/§1.2
(Python-flavored). Rust blobs do NOT travel across the platform.**

## 2. Signing

All signatures are 32-byte HMAC-SHA256 over the **inner tree bytes**
(§1.1), NOT over the envelope. The envelope magic + version + signature
tail are NOT part of what the HMAC covers.

Two tiers exist.

### 2.1 Software tier (platform-verifiable)

**Key:** `user_secret` — a per-user 32-byte secret the user holds and
the platform also holds (issued at signup, rotatable).

**Construction:**
```
sig = HMAC-SHA256(key=user_secret, msg=inner_tree_bytes)
```

**Verify (platform):**
```python
from carl_core.signing import verify_software_signature
ok = verify_software_signature(inner_tree_bytes, sig, user_secret)
```

Constant-time HMAC compare. Reject on False with error code
`carl.eml.attestation_failed`.

**Python API:** `carl_core.signing.sign_tree_software()`, `verify_software_signature()` — both MIT, stdlib only.

**TS API:** `@carl/emlt-codec`'s `signSoftware()` / `verifySoftware()`
mirror the Python signatures exactly.

### 2.2 Hardware tier (machine-bound, platform cannot verify)

**Key:** `hw_fingerprint(machine) XOR user_secret` — mixes a
machine-specific fingerprint into the HMAC key.

**Construction:** see `terminals_runtime/eml/sign_impl.py` (BUSL-1.1,
hardware-gated).

**Verify:** only on the signing machine via the private runtime.
Platform cannot verify hardware-tier blobs; instead, it stores
`trust_tier="hardware"` and trusts the client-side attestation.

### 2.3 Trust tier field

Every Resonant stored on the platform carries a `trust_tier`:

- `"software"` — HMAC with `user_secret` only. Platform verifies
  before accepting. Viable for marketplace.
- `"hardware"` — HMAC with `hw_fp XOR user_secret`. Platform cannot
  verify; trusts client-side attestation. Reserved for admin tier
  (deferred to v0.10+).

For v0.9.x, **all platform-accepted Resonants are software-tier.**
Hardware-tier is a forward-compat slot.

## 3. Constitutional ledger blocks

Separate primitive from Resonants. Uses ed25519, not HMAC.

### 3.1 Canonical signing bytes

`LedgerBlock.signing_bytes()` is the authoritative byte construction
that the ed25519 signature covers. From
`packages/carl-core/src/carl_core/constitutional.py:229-248`:

```
signing_bytes =
    prev_block_hash (ascii) || b"|" ||
    policy_id (ascii)       || b"|" ||
    action_digest (ascii)   || b"|" ||
    verdict (float64 LE)    ||
    timestamp_ns (int64 LE) ||
    signer_pubkey (raw 32 bytes)
```

Platform-side verify:
```python
from nacl.signing import VerifyKey
VerifyKey(signer_pubkey).verify(signing_bytes, signature)
```

### 3.2 Canonical block hash (for chain integrity)

`LedgerBlock.block_hash()` returns hex sha256 of
`canonical_json(block)`. The canonical JSON is produced via
`carl_core.hashing.canonical_json` with:

- `sort_keys=True`
- `ensure_ascii=True`
- `separators=(",", ":")` (no whitespace)
- bytes fields rendered as `.hex()` strings
- float64 preserved (raw JSON number; NaN/inf rejected)
- datetime / Path / Decimal stringified deterministically

The JSON shape for a block:
```json
{
  "action_digest": "<hex>",
  "block_id": <int>,
  "policy_id": "<str>",
  "prev_block_hash": "<hex>",
  "signature": "<hex>",
  "signer_pubkey": "<hex>",
  "timestamp_ns": <int>,
  "verdict": <float>
}
```

Platform-side chain-integrity check:

1. On append, decode the incoming `LedgerBlock`.
2. Compute `candidate_prev = prev_block.block_hash()` where
   `prev_block` is the current chain head for this user.
3. Reject if `incoming.prev_block_hash != candidate_prev` with
   `carl.constitutional.chain_invalid`.
4. Verify ed25519 signature with `incoming.verify()`.
5. Insert row; update chain head pointer.

Use `canonical_json` (same library, port to TS via `@carl/emlt-codec`)
on the platform — do NOT roll your own JSON canonicalization. Any
drift breaks chains.

## 4. Platform countersignature (purchase delivery)

When the platform delivers a purchased Resonant to the buyer, it
attaches a platform countersignature as proof-of-purchase. This does
NOT replace the seller's signature; it's an additional envelope.

### 4.1 Header format

```
X-Carl-Platform-Countersig: <base64(sig_32)>
X-Carl-Platform-Countersig-Timestamp: <iso8601>
X-Carl-Platform-Countersig-Txid: <purchase_tx_id>
```

### 4.2 Signed payload construction

```
countersig_payload =
    b"carl-platform-countersig-v1|" ||
    resonant_content_hash (hex ascii) || b"|" ||
    purchase_tx_id (ascii) || b"|" ||
    buyer_user_id (ascii) || b"|" ||
    timestamp_ns (int64 LE, 8 bytes)

countersig = HMAC-SHA256(
    key=platform_signing_secret_v1,
    msg=countersig_payload,
)
```

### 4.3 Platform signing secret

- Lives in platform env as `CARL_PLATFORM_COUNTERSIG_SECRET_V1`
- Rotated on a schedule; `_v1` suffix denotes version
- Buyers can verify via `GET /api/platform/countersig-keys`
  (returns all active versions + their pubkey material where
  applicable — for HMAC, returns verification path, not the secret)

### 4.4 Buyer-side verify (reference impl in TS codec)

```typescript
import { verifyCountersig } from "@carl/emlt-codec";
const ok = await verifyCountersig(
  resonantContentHash, txId, userId, timestampNs, sigBase64,
);
```

## 5. Software tier onboarding flow

1. User signs up → platform issues 32-byte `user_secret`
2. User stores `user_secret` in their client (carl CLI writes to
   `~/.carl/credentials/user_secret`, mode 0600)
3. User trains a tree; client signs with `sign_tree_software(bytes, user_secret)`
4. Client uploads envelope (inner + sig) to `POST /api/resonants`
5. Platform retrieves `user_secret` for the authenticated user,
   verifies, and on success inserts
6. Success response includes the Resonant's `id` and
   `content_hash` for reference

## 6. Rotation + revocation

- `user_secret` can be rotated by the user via
  `POST /api/auth/rotate-user-secret`. Platform keeps the last N
  versions active for 30 days (configurable) so in-flight blobs
  signed with the previous secret still verify.
- Blob revocation is soft: mark the marketplace row as `revoked=true`;
  platform refuses to serve on purchase.
- The platform never logs or exposes `user_secret` in any surface.

## 7. Reference implementations

- **Python:** `packages/carl-core/src/carl_core/signing.py`
- **TypeScript:** `@carl/emlt-codec` npm package (authored in
  `packages/emlt-codec-ts/`, published on npm by the carl-studio
  team)
- **Rust:** not exposed across the platform boundary; lives privately
  in `terminals-runtime/src/terminals_runtime/eml/sign_impl.py` and
  in `terminals-core` for local evaluation only.

## 8. Test vectors

The TS codec and Python reference impl both validate against a shared
test-vector file:
- `packages/emlt-codec-ts/test/vectors.json` — 10 fixed
  (tree, inputs, expected_output, sig, user_secret) tuples
- Python test: `tests/test_eml_py_ts_vectors.py` (run against the
  same JSON)

Any impl that decodes + verifies all 10 passes the interop bar.

## 9. Changelog

- 2026-04-20 v1 — initial protocol spec, paired with v0.9.0 ship.
