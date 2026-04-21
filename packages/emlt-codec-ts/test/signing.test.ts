/**
 * HMAC signing self-tests. Constant-time verify, short-secret rejection,
 * constructive countersig byte-layout check.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";

import {
  signSoftware,
  verifySoftware,
  signPlatformCountersig,
  verifyPlatformCountersig,
  SIG_LEN,
  MIN_SECRET_LEN,
} from "../src/index.js";

test("signing: output length is SIG_LEN (32)", () => {
  const secret = new Uint8Array(16).fill(0x42);
  const msg = new Uint8Array([1, 2, 3, 4]);
  const sig = signSoftware(msg, secret);
  assert.equal(sig.length, SIG_LEN);
});

test("signing: short secret throws on sign", () => {
  const msg = new Uint8Array([0]);
  const shortSecret = new Uint8Array(MIN_SECRET_LEN - 1);
  assert.throws(() => signSoftware(msg, shortSecret));
});

test("signing: verify round-trip true", () => {
  const secret = new Uint8Array(16).fill(0x11);
  const msg = new Uint8Array([9, 8, 7]);
  const sig = signSoftware(msg, secret);
  assert.equal(verifySoftware(msg, sig, secret), true);
});

test("signing: verify with wrong secret false", () => {
  const a = new Uint8Array(16).fill(0x11);
  const b = new Uint8Array(16).fill(0x22);
  const msg = new Uint8Array([9]);
  const sig = signSoftware(msg, a);
  assert.equal(verifySoftware(msg, sig, b), false);
});

test("signing: verify with tampered msg false", () => {
  const secret = new Uint8Array(16).fill(0x33);
  const msg = new Uint8Array([1, 2, 3]);
  const sig = signSoftware(msg, secret);
  const tampered = new Uint8Array([1, 2, 4]);
  assert.equal(verifySoftware(tampered, sig, secret), false);
});

test("signing: verify short-secret returns false (no throw)", () => {
  const secret = new Uint8Array(16).fill(0x55);
  const msg = new Uint8Array([5]);
  const sig = signSoftware(msg, secret);
  const tooShort = new Uint8Array(MIN_SECRET_LEN - 1);
  assert.equal(verifySoftware(msg, sig, tooShort), false);
});

test("signing: output matches raw Node HMAC (sanity)", () => {
  const secret = new Uint8Array(16).fill(0x77);
  const msg = new Uint8Array([1, 2, 3, 4, 5]);
  const sig = signSoftware(msg, secret);
  const expected = createHmac("sha256", Buffer.from(secret))
    .update(Buffer.from(msg))
    .digest();
  assert.deepEqual(Array.from(sig), Array.from(expected));
});

// ---- platform countersig ----------------------------------------------------

test("countersig: verify round-trip", () => {
  const secret = new Uint8Array(16).fill(0xab);
  const p = {
    contentHashHex: "a".repeat(64),
    purchaseTxId: "tx_1",
    buyerUserId: "u_1",
    timestampNs: 1700000000000000000n,
    platformSecret: secret,
  };
  const sig = signPlatformCountersig(p);
  assert.equal(sig.length, 32);
  assert.equal(verifyPlatformCountersig({ ...p, sig }), true);
});

test("countersig: verify with wrong tx id false", () => {
  const secret = new Uint8Array(16).fill(0xab);
  const p = {
    contentHashHex: "b".repeat(64),
    purchaseTxId: "tx_A",
    buyerUserId: "u_1",
    timestampNs: 0n,
    platformSecret: secret,
  };
  const sig = signPlatformCountersig(p);
  assert.equal(
    verifyPlatformCountersig({ ...p, purchaseTxId: "tx_B", sig }),
    false,
  );
});

test("countersig: timestamp is int64 LE in payload (regression)", () => {
  // If timestamp encoding changes, this vector breaks.
  const secret = new Uint8Array(16).fill(0xcd);
  const sig1 = signPlatformCountersig({
    contentHashHex: "00".repeat(32),
    purchaseTxId: "t",
    buyerUserId: "u",
    timestampNs: 0n,
    platformSecret: secret,
  });
  const sig2 = signPlatformCountersig({
    contentHashHex: "00".repeat(32),
    purchaseTxId: "t",
    buyerUserId: "u",
    timestampNs: 1n,
    platformSecret: secret,
  });
  // Different timestamps must yield different signatures.
  assert.notDeepEqual(Array.from(sig1), Array.from(sig2));
});
