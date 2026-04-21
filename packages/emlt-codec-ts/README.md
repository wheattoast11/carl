# `@carl/emlt-codec`

TypeScript port of the **EML wire format** and **software-tier HMAC-SHA256
signature verification** used by the CARL / `carl.camp` platform. Byte- and
bit-identical with the Python reference in
[`carl-core`](https://github.com/terminals-tech/carl-studio/tree/main/packages/carl-core).

Zero runtime dependencies. Node 18+. Works in Bun, Deno (via the `node:` prefix),
and Vercel Edge (via the built-in `node:crypto` shim).

## Install

```bash
npm install @carl/emlt-codec
# or
bun add @carl/emlt-codec
```

## Quick start

### Encode / decode an EML tree

```ts
import {
  EMLOp,
  encodeInner,
  decodeInner,
  type EMLTree,
} from "@carl/emlt-codec";

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
} from "@carl/emlt-codec";

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
} from "@carl/emlt-codec";

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
