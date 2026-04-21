/**
 * Ledger canonicalization + signing-bytes parity tests.
 *
 * Every assertion here is asserted on both sides of the Py↔TS boundary.
 * See `tests/test_ledger_parity_vectors.py` for the Python mirror.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, test } from "node:test";
import assert from "node:assert/strict";

import {
  CanonicalizationError,
  canonicalJson,
  ledgerBlockHash,
  ledgerBlockSigningBytes,
  type LedgerBlock,
} from "../src/ledger.js";

function vectorsPath(): string {
  const here = fileURLToPath(import.meta.url);
  return here.replace(/ledger\.test\.[jt]s$/, "ledger_vectors.json");
}

interface VectorFile {
  name: string;
  input: {
    block_id: number;
    prev_block_hash: string;
    policy_id: string;
    action_digest: string;
    verdict: number;
    timestamp_ns: number;
    signer_pubkey_hex: string;
    signature_hex: string;
  };
  expected: {
    signing_bytes_hex: string;
    block_hash: string;
  };
}

function loadVectors(): VectorFile[] {
  return JSON.parse(readFileSync(vectorsPath(), "utf8"));
}

function bytesToHex(b: Uint8Array): string {
  let out = "";
  for (let i = 0; i < b.length; i++) out += b[i].toString(16).padStart(2, "0");
  return out;
}

function toLedgerBlock(v: VectorFile["input"]): LedgerBlock {
  return {
    block_id: v.block_id,
    prev_block_hash: v.prev_block_hash,
    policy_id: v.policy_id,
    action_digest: v.action_digest,
    verdict: v.verdict,
    timestamp_ns: BigInt(v.timestamp_ns),
    signer_pubkey: v.signer_pubkey_hex,
    signature: v.signature_hex,
  };
}

// ---------------------------------------------------------------------------

describe("canonicalJson — shape + ordering", () => {
  test("sorts object keys", () => {
    const s = canonicalJson({ b: 2, a: 1 });
    assert.equal(s, '{"a":1,"b":2}');
  });

  test("no whitespace", () => {
    const s = canonicalJson({ x: [1, 2, { y: 3 }] });
    assert.equal(s, '{"x":[1,2,{"y":3}]}');
  });

  test("strings are ASCII-escaped", () => {
    const s = canonicalJson({ s: "hé" });
    assert.equal(s, '{"s":"h\\u00e9"}');
  });

  test("null / bool primitives", () => {
    assert.equal(canonicalJson({ n: null, t: true, f: false }), '{"f":false,"n":null,"t":true}');
  });

  test("integer-valued number in verdict slot keeps .0 suffix", () => {
    const s = canonicalJson({ verdict: 1 });
    assert.equal(s, '{"verdict":1.0}');
  });

  test("integer in non-float slot stays bare", () => {
    const s = canonicalJson({ block_id: 42 });
    assert.equal(s, '{"block_id":42}');
  });

  test("negative zero verdict emits -0.0", () => {
    const s = canonicalJson({ verdict: -0 });
    assert.equal(s, '{"verdict":-0.0}');
  });

  test("rejects NaN verdict", () => {
    assert.throws(
      () => canonicalJson({ verdict: NaN }),
      CanonicalizationError,
    );
  });

  test("rejects Infinity verdict", () => {
    assert.throws(
      () => canonicalJson({ verdict: Infinity }),
      CanonicalizationError,
    );
  });

  test("explicit floatKeys option", () => {
    const s = canonicalJson({ score: 2 }, { floatKeys: ["score"] });
    assert.equal(s, '{"score":2.0}');
  });

  test("bigint serializes as bare integer", () => {
    const s = canonicalJson({ big: 9007199254740993n });
    assert.equal(s, '{"big":9007199254740993}');
  });
});

describe("ledgerBlockSigningBytes — parity with Python", () => {
  for (const v of loadVectors()) {
    test(`vector[${v.name}]: signing_bytes matches Python`, () => {
      const block = toLedgerBlock(v.input);
      const got = bytesToHex(ledgerBlockSigningBytes(block));
      assert.equal(got, v.expected.signing_bytes_hex);
    });
  }

  test("rejects non-32-byte pubkey", () => {
    const block = toLedgerBlock(loadVectors()[0].input);
    block.signer_pubkey = new Uint8Array(31);
    assert.throws(() => ledgerBlockSigningBytes(block), CanonicalizationError);
  });

  test("accepts Uint8Array pubkey + signature directly", () => {
    const v = loadVectors()[0];
    const pk = Buffer.from(v.input.signer_pubkey_hex, "hex");
    const sig = Buffer.from(v.input.signature_hex, "hex");
    const block: LedgerBlock = {
      ...toLedgerBlock(v.input),
      signer_pubkey: new Uint8Array(pk),
      signature: new Uint8Array(sig),
    };
    const got = bytesToHex(ledgerBlockSigningBytes(block));
    assert.equal(got, v.expected.signing_bytes_hex);
  });
});

describe("ledgerBlockHash — parity with Python", () => {
  for (const v of loadVectors()) {
    test(`vector[${v.name}]: block_hash matches Python`, () => {
      const block = toLedgerBlock(v.input);
      const got = ledgerBlockHash(block);
      assert.equal(got, v.expected.block_hash);
    });
  }

  test("hash changes when verdict changes", () => {
    const v = loadVectors()[0];
    const a = ledgerBlockHash({ ...toLedgerBlock(v.input), verdict: 0.0 });
    const b = ledgerBlockHash({ ...toLedgerBlock(v.input), verdict: 0.1 });
    assert.notEqual(a, b);
  });

  test("hash stable across bigint / number timestamp_ns inputs", () => {
    const v = loadVectors()[0];
    const a = ledgerBlockHash({
      ...toLedgerBlock(v.input),
      timestamp_ns: BigInt(v.input.timestamp_ns),
    });
    const b = ledgerBlockHash({
      ...toLedgerBlock(v.input),
      timestamp_ns: v.input.timestamp_ns,
    });
    assert.equal(a, b);
  });

  test("hash stable across hex / Uint8Array pubkey inputs", () => {
    const v = loadVectors()[0];
    const hexInput = toLedgerBlock(v.input);
    const pk = Buffer.from(v.input.signer_pubkey_hex, "hex");
    const sig = Buffer.from(v.input.signature_hex, "hex");
    const bytesInput: LedgerBlock = {
      ...hexInput,
      signer_pubkey: new Uint8Array(pk),
      signature: new Uint8Array(sig),
    };
    assert.equal(ledgerBlockHash(hexInput), ledgerBlockHash(bytesInput));
  });
});
