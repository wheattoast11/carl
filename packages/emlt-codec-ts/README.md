# `@terminals-tech/emlt-codec`

TypeScript port of the **EML wire format** and **software-tier HMAC-SHA256
signature verification** used by the CARL / `carl.camp` platform. Byte- and
bit-identical with the Python reference in
[`carl-core`](https://github.com/terminals-tech/carl-studio/tree/main/packages/carl-core).

Zero runtime dependencies. Node 18+. Works in Bun, Deno (via the `node:` prefix),
and Vercel Edge (via the built-in `node:crypto` shim).

## Install

```bash
npm install @terminals-tech/emlt-codec
# or
bun add @terminals-tech/emlt-codec
```

## Quick start

### Encode / decode an EML tree

```ts
import {
  EMLOp,
  encodeInner,
  decodeInner,
  type EMLTree,
} from "@terminals-tech/emlt-codec";

const tree: EMLTree = {
  root: {
    op: EMLOp.EML,
    left: { op: EMLOp.VAR_X, varIdx: 0 },
    right: { op: EMLOp.CONST, const_: 1 },
  },
  inputDim: 1,
};

const innerBytes = encodeInner(tree); // §1.1 canonical bytes
const decoded = decodeInner(innerBytes); // round-trips exactly
```

### Envelope with optional signature

```ts
import {
  encodeEnvelope,
  decodeEnvelope,
  signSoftware,
  verifySoftware,
} from "@terminals-tech/emlt-codec";

const userSecret = new Uint8Array(32); // per-user 32-byte secret, platform-held
const inner = encodeInner(tree);
const sig = signSoftware(inner, userSecret);

const envelope = encodeEnvelope(tree, { signature: sig });
// ... send to platform ...

const { tree: decodedTree, signature } = decodeEnvelope(envelope);
if (!signature) throw new Error("missing sig");
const ok = verifySoftware(encodeInner(decodedTree), signature, userSecret);
```

### Platform-side countersignature (purchase delivery, §4)

```ts
import {
  signPlatformCountersig,
  verifyPlatformCountersig,
} from "@terminals-tech/emlt-codec";

const platformSecret = Buffer.from(process.env.CARL_PLATFORM_COUNTERSIG_SECRET_V1!, "hex");

const sig = signPlatformCountersig({
  contentHashHex: "<64-char hex>",
  purchaseTxId: "tx_abc123",
  buyerUserId: "user_42",
  timestampNs: BigInt(Date.now()) * 1_000_000n,
  platformSecret,
});

// Buyer-side verify:
const ok = verifyPlatformCountersig({
  contentHashHex, purchaseTxId, buyerUserId, timestampNs,
  platformSecret, sig,
});
```

### Constitutional ledger canonicalization + signing bytes (0.2.0+)

```ts
import {
  canonicalJson,
  ledgerBlockSigningBytes,
  ledgerBlockHash,
  type LedgerBlock,
} from "@terminals-tech/emlt-codec";

const block: LedgerBlock = {
  block_id: 1,
  prev_block_hash: "00".repeat(32),
  policy_id: "policy.gate",
  action_digest: "ab".repeat(32),
  verdict: 0.42,                               // float; canonicalized as "0.42"
  timestamp_ns: BigInt(Date.now()) * 1_000_000n,
  signer_pubkey: pubkeyBytes,                  // 32 bytes (or hex string)
  signature: sigBytes,                         // 64 bytes (or hex string)
};

// Binary bytes ed25519 signs over (§3.1 of the signing protocol).
const toSign = ledgerBlockSigningBytes(block);

// Hex sha256 for chain integrity (§3.2). Next block's prev_block_hash.
const hash = ledgerBlockHash(block);

// Low-level canonical JSON for custom shapes. Defaults to the ledger
// float-key set (`{"verdict"}`); override via options for other schemas.
const json = canonicalJson(someObject, { floatKeys: ["score", "reward"] });
```

**Parity lock:** every assertion in `test/ledger.test.ts` is mirrored
in `tests/test_ledger_parity_vectors.py` on the Python side. Both
sides load the same `test/ledger_vectors.json` (5 fixtures generated
from `carl_core.constitutional.LedgerBlock`). If Python and TS ever
drift, one side's tests fail before the other's ship. Regenerate with
`npm run gen-ledger-vectors`.

**Scope note:** `canonicalJson` here targets the LedgerBlock schema
(str / int / float / bool / null / array / object). It does NOT try
to mirror `carl_core.hashing.canonical_json`'s full breadth (Decimal
/ datetime / Path / bytes coercion). That would belong in a separate
package.

## API surface

```ts
// Types
export enum EMLOp { CONST, VAR_X, EML }
export interface EMLNode {
  op: EMLOp;
  const_?: number;
  varIdx?: number;
  left?: EMLNode;
  right?: EMLNode;
}
export interface EMLTree {
  root: EMLNode;
  inputDim: number;
  leafParams?: Float64Array;
}

// Constants
export const MAX_DEPTH = 4;
export const SIG_LEN = 32;
export const MIN_SECRET_LEN = 16;
export const INNER_MAGIC: Uint8Array;    // b"EML\x01"
export const ENVELOPE_MAGIC: Uint8Array; // b"EMLT"
export const ENVELOPE_VERSION = 1;

// Codec
export function encodeInner(tree: EMLTree): Uint8Array;
export function decodeInner(data: Uint8Array): EMLTree;
export function encodeEnvelope(
  tree: EMLTree,
  options?: { signature?: Uint8Array },
): Uint8Array;
export function decodeEnvelope(data: Uint8Array): {
  tree: EMLTree;
  signature?: Uint8Array;
};
export function evalTree(tree: EMLTree, inputs: Float64Array): number;
export function computeDepth(tree: EMLTree): number;

// Constitutional ledger (0.2.0+)
export interface LedgerBlock {
  block_id: number | bigint;
  prev_block_hash: string;
  policy_id: string;
  action_digest: string;
  verdict: number;
  timestamp_ns: number | bigint;
  signer_pubkey: string | Uint8Array;
  signature: string | Uint8Array;
}
export interface CanonicalJsonOptions {
  floatKeys?: ReadonlySet<string> | readonly string[];
}
export class CanonicalizationError extends Error {}
export function canonicalJson(value: unknown, options?: CanonicalJsonOptions): string;
export function ledgerBlockSigningBytes(block: LedgerBlock): Uint8Array;
export function ledgerBlockHash(block: LedgerBlock): string;

// Signing
export function signSoftware(treeBytes: Uint8Array, userSecret: Uint8Array): Uint8Array;
export function verifySoftware(treeBytes: Uint8Array, sig: Uint8Array, userSecret: Uint8Array): boolean;
export function signPlatformCountersig(params: PlatformCountersigParams): Uint8Array;
export function verifyPlatformCountersig(params: PlatformCountersigParams & { sig: Uint8Array }): boolean;
```

## Wire format

See [`docs/eml_signing_protocol.md`](../../docs/eml_signing_protocol.md) in the
carl-studio repo — that's the source of truth. Summary:

```
Inner:    b"EML\x01" (4)  | input_dim (uint16 LE) | postfix tag stream
Envelope: b"EMLT"    (4)  | VERSION (0x01)        | inner | [sig 32 bytes]

CONST tag = 0x01 + float64 LE (8 bytes)
VAR_X tag = 0x02 + uint16 LE  (2 bytes)
EML   tag = 0x03  (no payload)
```

Signatures are HMAC-SHA256 over **inner bytes only**, never over the envelope.

## Development

```bash
# From this package root:
npm install
npm run build    # emits dist/{esm,cjs,types}
npm test         # node --test with tsx

# Regenerate cross-language test vectors from the Python reference:
npm run gen-vectors   # requires python3 + numpy + carl-core source
```

If `npm` is unavailable, `bun install && bun test` works too — Bun runs the
same `tsx` loader under the hood.

## Parity guarantee

Every test in `test/vectors.test.ts` is generated from the Python reference
(`scripts/gen_vectors.py`) and asserts byte-identical encoding + matching HMAC
output. Any change that breaks parity breaks the test.

## License

MIT (matches `carl-core`). No hardware-tier signing in this package — that
primitive lives in the BUSL-licensed `terminals-runtime` and is not needed for
platform-side verification.
