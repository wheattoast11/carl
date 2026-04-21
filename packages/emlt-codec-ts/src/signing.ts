/**
 * Software-tier HMAC-SHA256 signing + platform countersignature.
 *
 * Byte-identical to the Python reference in
 * packages/carl-core/src/carl_core/signing.py (MIT, stdlib only).
 *
 * Uses Node's `node:crypto` for HMAC + constant-time compare. Works on
 * Node 18+ and any runtime that ships the Node crypto API (Bun, Deno with
 * `node:` prefix, Vercel Edge via the node:crypto shim, etc.).
 */

import { createHmac, timingSafeEqual } from "node:crypto";

export const SIG_LEN = 32 as const;
export const MIN_SECRET_LEN = 16 as const;

function validateSecret(secret: Uint8Array): void {
  if (secret.length < MIN_SECRET_LEN) {
    throw new Error(
      `secret must be at least ${MIN_SECRET_LEN} bytes; got ${secret.length}`,
    );
  }
}

function hmacSha256(key: Uint8Array, msg: Uint8Array): Uint8Array {
  // Buffer.from over a Uint8Array is a view (no copy) on Node / Bun.
  const h = createHmac("sha256", Buffer.from(key.buffer, key.byteOffset, key.byteLength));
  h.update(Buffer.from(msg.buffer, msg.byteOffset, msg.byteLength));
  return new Uint8Array(h.digest());
}

function constantTimeEquals(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  return timingSafeEqual(
    Buffer.from(a.buffer, a.byteOffset, a.byteLength),
    Buffer.from(b.buffer, b.byteOffset, b.byteLength),
  );
}

/**
 * Sign the §1.1 inner tree bytes. Matches
 * `carl_core.signing.sign_tree_software()`.
 *
 * `treeBytes` MUST be the output of `encodeInner()`, NOT the full envelope.
 */
export function signSoftware(treeBytes: Uint8Array, userSecret: Uint8Array): Uint8Array {
  validateSecret(userSecret);
  return hmacSha256(userSecret, treeBytes);
}

/**
 * Constant-time verify. Returns `false` (does NOT throw) on length mismatch
 * or undersize secret — the caller decides how to surface attestation failure.
 */
export function verifySoftware(
  treeBytes: Uint8Array,
  sig: Uint8Array,
  userSecret: Uint8Array,
): boolean {
  if (sig.length !== SIG_LEN) return false;
  if (userSecret.length < MIN_SECRET_LEN) return false;
  const expected = hmacSha256(userSecret, treeBytes);
  return constantTimeEquals(expected, sig);
}

// -------- platform countersignature (§4.2) -----------------------------------

export interface PlatformCountersigParams {
  /** Hex-encoded content hash of the resonant (as stored in the marketplace). */
  contentHashHex: string;
  /** Purchase transaction id (opaque ASCII string). */
  purchaseTxId: string;
  /** Buyer user id (opaque ASCII string). */
  buyerUserId: string;
  /** Timestamp in nanoseconds since epoch. */
  timestampNs: bigint;
  /** Platform's active HMAC secret (e.g. CARL_PLATFORM_COUNTERSIG_SECRET_V1). */
  platformSecret: Uint8Array;
}

function countersigPayload(
  contentHashHex: string,
  purchaseTxId: string,
  buyerUserId: string,
  timestampNs: bigint,
): Uint8Array {
  const te = new TextEncoder();
  const prefix = te.encode("carl-platform-countersig-v1|");
  const contentHash = te.encode(contentHashHex);
  const sep = te.encode("|");
  const txId = te.encode(purchaseTxId);
  const userId = te.encode(buyerUserId);
  const ts = new Uint8Array(8);
  // little-endian int64
  const view = new DataView(ts.buffer);
  view.setBigInt64(0, timestampNs, true);

  const total =
    prefix.length +
    contentHash.length +
    sep.length +
    txId.length +
    sep.length +
    userId.length +
    sep.length +
    ts.length;
  const out = new Uint8Array(total);
  let o = 0;
  out.set(prefix, o);
  o += prefix.length;
  out.set(contentHash, o);
  o += contentHash.length;
  out.set(sep, o);
  o += sep.length;
  out.set(txId, o);
  o += txId.length;
  out.set(sep, o);
  o += sep.length;
  out.set(userId, o);
  o += userId.length;
  out.set(sep, o);
  o += sep.length;
  out.set(ts, o);
  return out;
}

/**
 * Platform-side countersignature. Matches
 * `carl_core.signing.sign_platform_countersig()`.
 *
 * Payload (§4.2):
 *   b"carl-platform-countersig-v1|" ||
 *   content_hash_hex (ascii) || b"|" ||
 *   purchase_tx_id (ascii)   || b"|" ||
 *   buyer_user_id (ascii)    || b"|" ||
 *   timestamp_ns (int64 LE, 8 bytes)
 */
export function signPlatformCountersig(params: PlatformCountersigParams): Uint8Array {
  validateSecret(params.platformSecret);
  const payload = countersigPayload(
    params.contentHashHex,
    params.purchaseTxId,
    params.buyerUserId,
    params.timestampNs,
  );
  return hmacSha256(params.platformSecret, payload);
}

/** Buyer-side verify of a platform countersignature. Constant-time. */
export function verifyPlatformCountersig(
  params: PlatformCountersigParams & { sig: Uint8Array },
): boolean {
  if (params.sig.length !== SIG_LEN) return false;
  if (params.platformSecret.length < MIN_SECRET_LEN) return false;
  const expected = signPlatformCountersig(params);
  return constantTimeEquals(expected, params.sig);
}
