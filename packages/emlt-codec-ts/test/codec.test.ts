/**
 * Round-trip codec tests — exercise the inner tag stream + envelope heuristics
 * without depending on Python-generated vectors. The vectors test in
 * ./vectors.test.ts covers cross-language byte parity.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  EMLOp,
  MAX_DEPTH,
  INNER_MAGIC,
  ENVELOPE_MAGIC,
  ENVELOPE_VERSION,
  SIG_LEN,
  computeDepth,
  encodeInner,
  decodeInner,
  encodeEnvelope,
  decodeEnvelope,
  evalTree,
  EmltDepthExceededError,
  EmltDecodeError,
  type EMLNode,
  type EMLTree,
} from "../src/index.js";

// ---- helpers ----------------------------------------------------------------

function constNode(v: number): EMLNode {
  return { op: EMLOp.CONST, const_: v };
}
function varNode(i: number): EMLNode {
  return { op: EMLOp.VAR_X, varIdx: i };
}
function eml(l: EMLNode, r: EMLNode): EMLNode {
  return { op: EMLOp.EML, left: l, right: r };
}

// ---- inner codec ------------------------------------------------------------

test("inner: magic is b'EML\\x01'", () => {
  assert.deepEqual(
    Array.from(INNER_MAGIC),
    [0x45, 0x4d, 0x4c, 0x01],
    "inner magic constant",
  );
});

test("inner: envelope magic is b'EMLT'", () => {
  assert.deepEqual(
    Array.from(ENVELOPE_MAGIC),
    [0x45, 0x4d, 0x4c, 0x54],
    "envelope magic constant",
  );
  assert.equal(ENVELOPE_VERSION, 1);
});

test("inner: SIG_LEN is 32", () => {
  assert.equal(SIG_LEN, 32);
});

test("inner: VAR_X root encode + decode roundtrip", () => {
  const tree: EMLTree = { root: varNode(0), inputDim: 1 };
  const bytes = encodeInner(tree);
  // header(4) + input_dim(2) + VAR_X tag(1) + varIdx(2) = 9
  assert.equal(bytes.length, 9);
  assert.equal(bytes[0], 0x45);
  assert.equal(bytes[4], 0x01); // input_dim low
  assert.equal(bytes[5], 0x00);
  assert.equal(bytes[6], 0x02); // VAR_X tag

  const decoded = decodeInner(bytes);
  assert.equal(decoded.inputDim, 1);
  assert.equal(decoded.root.op, EMLOp.VAR_X);
  assert.equal(decoded.root.varIdx, 0);
});

test("inner: CONST root encode + decode", () => {
  const tree: EMLTree = { root: constNode(3.14), inputDim: 1 };
  const bytes = encodeInner(tree);
  // header(4) + input_dim(2) + CONST tag(1) + float64(8) = 15
  assert.equal(bytes.length, 15);

  const decoded = decodeInner(bytes);
  assert.equal(decoded.root.op, EMLOp.CONST);
  // round-trip via leafParams since decoder stores there
  assert.ok(decoded.leafParams);
  assert.equal(decoded.leafParams![0], 3.14);
});

test("inner: depth-1 EML roundtrip eml(x0, 1)", () => {
  const tree: EMLTree = { root: eml(varNode(0), constNode(1)), inputDim: 1 };
  const bytes = encodeInner(tree);
  const decoded = decodeInner(bytes);
  assert.equal(decoded.root.op, EMLOp.EML);
  assert.equal(decoded.root.left!.op, EMLOp.VAR_X);
  assert.equal(decoded.root.right!.op, EMLOp.CONST);

  // round-trip preservation: re-encoding produces the same bytes.
  const bytes2 = encodeInner(decoded);
  assert.deepEqual(Array.from(bytes2), Array.from(bytes));
});

test("inner: depth-4 identity_deep encodes + decodes", () => {
  // exp(ln(x)) at depth exactly 4, max allowed.
  const one = constNode(1);
  const x = varNode(0);
  const emlEml = (l: EMLNode, r: EMLNode): EMLNode => eml(l, r);
  const ln = emlEml(
    one,
    emlEml(emlEml(one, x), one),
  );
  const root = emlEml(ln, one);
  const tree: EMLTree = { root, inputDim: 1 };
  assert.equal(computeDepth(tree), MAX_DEPTH);
  const bytes = encodeInner(tree);
  const decoded = decodeInner(bytes);
  assert.equal(computeDepth(decoded), MAX_DEPTH);
});

test("inner: depth-5 tree rejected at encode", () => {
  // Build a linear spine 5 deep.
  let node: EMLNode = varNode(0);
  for (let i = 0; i < 5; i++) node = eml(node, constNode(1));
  const tree: EMLTree = { root: node, inputDim: 1 };
  assert.throws(() => encodeInner(tree), EmltDepthExceededError);
});

test("inner: bad magic on decode raises EmltDecodeError", () => {
  const bogus = new Uint8Array([0x00, 0x00, 0x00, 0x00, 0x01, 0x00]);
  assert.throws(() => decodeInner(bogus), EmltDecodeError);
});

test("inner: truncated buffer raises", () => {
  const bytes = encodeInner({ root: varNode(0), inputDim: 1 });
  assert.throws(() => decodeInner(bytes.subarray(0, 5)), EmltDecodeError);
});

test("inner: unknown tag raises", () => {
  const bad = new Uint8Array([0x45, 0x4d, 0x4c, 0x01, 0x01, 0x00, 0xff]);
  assert.throws(() => decodeInner(bad), EmltDecodeError);
});

test("inner: re-encode is a fixed point", () => {
  const tree: EMLTree = {
    root: eml(eml(varNode(0), constNode(2.5)), constNode(-1.25)),
    inputDim: 1,
  };
  const a = encodeInner(tree);
  const decoded = decodeInner(a);
  const b = encodeInner(decoded);
  assert.deepEqual(Array.from(a), Array.from(b));
});

// ---- envelope codec ---------------------------------------------------------

test("envelope: unsigned roundtrip", () => {
  const tree: EMLTree = { root: eml(varNode(0), constNode(1)), inputDim: 1 };
  const env = encodeEnvelope(tree);
  assert.equal(env[0], 0x45);
  assert.equal(env[3], 0x54);
  assert.equal(env[4], 0x01);
  const { tree: decoded, signature } = decodeEnvelope(env);
  assert.equal(signature, undefined);
  assert.equal(decoded.root.op, EMLOp.EML);
});

test("envelope: signed roundtrip preserves sig", () => {
  const tree: EMLTree = { root: varNode(0), inputDim: 1 };
  const fakeSig = new Uint8Array(32);
  for (let i = 0; i < 32; i++) fakeSig[i] = i;
  const env = encodeEnvelope(tree, { signature: fakeSig });
  const { tree: decoded, signature } = decodeEnvelope(env);
  assert.ok(signature);
  assert.deepEqual(Array.from(signature!), Array.from(fakeSig));
  assert.equal(decoded.inputDim, 1);
});

test("envelope: sig length != 32 throws on encode", () => {
  const tree: EMLTree = { root: varNode(0), inputDim: 1 };
  const badSig = new Uint8Array(16);
  assert.throws(
    () => encodeEnvelope(tree, { signature: badSig }),
    EmltDecodeError,
  );
});

test("envelope: bad magic throws", () => {
  const bogus = new Uint8Array([0x00, 0x00, 0x00, 0x00, 0x01]);
  assert.throws(() => decodeEnvelope(bogus), EmltDecodeError);
});

test("envelope: unsupported version throws", () => {
  const bad = new Uint8Array([0x45, 0x4d, 0x4c, 0x54, 0x02]);
  assert.throws(() => decodeEnvelope(bad), EmltDecodeError);
});

test("envelope: decodeInner is pure (does not allocate leafParams for VAR-only tree)", () => {
  const tree: EMLTree = { root: varNode(2), inputDim: 3 };
  const bytes = encodeInner(tree);
  const decoded = decodeInner(bytes);
  assert.equal(decoded.leafParams, undefined);
  assert.equal(decoded.root.varIdx, 2);
});

// ---- reference evaluator ----------------------------------------------------

test("eval: VAR_X returns the indexed input", () => {
  const tree: EMLTree = { root: varNode(0), inputDim: 1 };
  assert.equal(evalTree(tree, Float64Array.from([7.5])), 7.5);
});

test("eval: CONST returns the constant", () => {
  const tree: EMLTree = { root: constNode(42), inputDim: 1 };
  assert.equal(evalTree(tree, Float64Array.from([0])), 42);
});

test("eval: exp_single = exp(x) via eml(x, 1)", () => {
  const tree: EMLTree = { root: eml(varNode(0), constNode(1)), inputDim: 1 };
  const out = evalTree(tree, Float64Array.from([0.5]));
  // eml(0.5, 1) = exp(0.5) - ln(1) = exp(0.5)
  assert.ok(Math.abs(out - Math.exp(0.5)) < 1e-12);
});
