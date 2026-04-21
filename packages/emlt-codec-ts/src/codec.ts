/**
 * Canonical byte codec for EML trees (§1.1 inner format) and the platform
 * envelope (§1.2 outer format). Byte-identical with the Python reference
 * in carl_core.eml.EMLTree.to_bytes / from_bytes and
 * terminals_runtime.eml.codec_impl.encode / decode.
 *
 * Wire format (little-endian):
 *
 *   Inner:  MAGIC(b"EML\x01", 4) | input_dim(uint16 LE) | postfix tag stream
 *     CONST tag = 0x01 + float64 LE (8 bytes)
 *     VAR_X tag = 0x02 + uint16 LE (2 bytes)
 *     EML   tag = 0x03 (no payload)
 *
 *   Envelope: MAGIC(b"EMLT", 4) | VERSION(0x01, 1) | inner | [sig 32 bytes]
 */

import { EMLOp, type EMLNode, type EMLTree, EmltDecodeError } from "./types.js";
import { assertDepthOk, inputDimRequired } from "./depth.js";

// -------- constants -----------------------------------------------------------

export const SIG_LEN = 32 as const;
export const ENVELOPE_VERSION = 1 as const;

/** b"EML\x01" — inner format magic. */
export const INNER_MAGIC: Uint8Array = new Uint8Array([0x45, 0x4d, 0x4c, 0x01]);

/** b"EMLT" — envelope format magic. */
export const ENVELOPE_MAGIC: Uint8Array = new Uint8Array([0x45, 0x4d, 0x4c, 0x54]);

const TAG_CONST = 0x01;
const TAG_VAR_X = 0x02;
const TAG_EML = 0x03;

const INNER_HEADER_LEN = INNER_MAGIC.length + 2; // magic + uint16 input_dim
const ENVELOPE_HEADER_LEN = ENVELOPE_MAGIC.length + 1; // magic + version byte

// -------- helpers -------------------------------------------------------------

function eqBytes(a: Uint8Array, b: Uint8Array, len: number): boolean {
  if (a.length < len || b.length < len) return false;
  for (let i = 0; i < len; i++) if (a[i] !== b[i]) return false;
  return true;
}

function hex(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i]!;
    s += (b < 16 ? "0" : "") + b.toString(16);
  }
  return s;
}

// -------- inner encode --------------------------------------------------------

/**
 * Encode an EML tree into the §1.1 canonical inner format.
 *
 * Byte-identical with `EMLTree.to_bytes()` in Python. If `tree.leafParams` is
 * set, CONST leaves read their value from that flat array in in-order
 * traversal; otherwise each CONST node's own `const_` is used.
 */
export function encodeInner(tree: EMLTree): Uint8Array {
  assertDepthOk(tree);
  if (tree.inputDim < 0 || !Number.isInteger(tree.inputDim) || tree.inputDim > 0xffff) {
    throw new EmltDecodeError(
      `input_dim out of range for uint16: ${tree.inputDim}`,
      "carl.eml.domain_error",
      { inputDim: tree.inputDim },
    );
  }
  const required = inputDimRequired(tree.root);
  if (required > tree.inputDim) {
    throw new EmltDecodeError(
      `input_dim=${tree.inputDim} insufficient for VAR_X references (need ${required})`,
      "carl.eml.domain_error",
      { inputDim: tree.inputDim, required },
    );
  }

  // Two-pass: first count nodes to preallocate, then write.
  const buf: number[] = [];
  // Header: magic (4) + input_dim (uint16 LE)
  for (let i = 0; i < INNER_MAGIC.length; i++) buf.push(INNER_MAGIC[i]!);
  buf.push(tree.inputDim & 0xff);
  buf.push((tree.inputDim >>> 8) & 0xff);

  // Postfix walk.
  const leafParams = tree.leafParams;
  const counter = { i: 0 };
  const scratch = new ArrayBuffer(8);
  const scratchView = new DataView(scratch);

  function visit(node: EMLNode): void {
    if (node.op === EMLOp.CONST) {
      const idx = counter.i++;
      let val: number;
      if (leafParams !== undefined && idx < leafParams.length) {
        val = leafParams[idx]!;
      } else if (node.const_ === undefined) {
        throw new EmltDecodeError(
          "CONST node requires `const_` value",
          "carl.eml.domain_error",
        );
      } else {
        val = node.const_;
      }
      buf.push(TAG_CONST);
      scratchView.setFloat64(0, val, true /* little-endian */);
      for (let i = 0; i < 8; i++) buf.push(scratchView.getUint8(i));
      return;
    }
    if (node.op === EMLOp.VAR_X) {
      const idx = node.varIdx;
      if (idx === undefined || idx < 0 || !Number.isInteger(idx) || idx > 0xffff) {
        throw new EmltDecodeError(
          `VAR_X requires varIdx in [0, 65535]; got ${String(idx)}`,
          "carl.eml.domain_error",
        );
      }
      buf.push(TAG_VAR_X);
      buf.push(idx & 0xff);
      buf.push((idx >>> 8) & 0xff);
      return;
    }
    if (node.op === EMLOp.EML) {
      if (!node.left || !node.right) {
        throw new EmltDecodeError(
          "EML node requires both `left` and `right`",
          "carl.eml.domain_error",
        );
      }
      visit(node.left);
      visit(node.right);
      buf.push(TAG_EML);
      return;
    }
    throw new EmltDecodeError(`unknown op: ${String(node.op)}`, "carl.eml.domain_error");
  }

  visit(tree.root);
  return Uint8Array.from(buf);
}

// -------- inner decode --------------------------------------------------------

/**
 * Decode the §1.1 inner format back into an EML tree.
 * Byte-identical with `EMLTree.from_bytes()` in Python.
 *
 * Collected CONST values are returned as `leafParams` on the tree, so a
 * re-encode is a fixed point.
 */
export function decodeInner(data: Uint8Array): EMLTree {
  if (data.length < INNER_HEADER_LEN) {
    throw new EmltDecodeError("buffer too short for header", "carl.eml.decode_error", {
      size: data.length,
    });
  }
  if (!eqBytes(data, INNER_MAGIC, INNER_MAGIC.length)) {
    throw new EmltDecodeError("bad magic", "carl.eml.decode_error", {
      magic: hex(data.subarray(0, INNER_MAGIC.length)),
    });
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const inputDim = view.getUint16(INNER_MAGIC.length, true);
  let cursor = INNER_HEADER_LEN;

  const stack: EMLNode[] = [];
  const leafValues: number[] = [];

  while (cursor < data.length) {
    const tag = data[cursor]!;
    cursor += 1;
    if (tag === TAG_CONST) {
      if (cursor + 8 > data.length) {
        throw new EmltDecodeError("truncated CONST payload", "carl.eml.decode_error");
      }
      const val = view.getFloat64(cursor, true);
      cursor += 8;
      stack.push({ op: EMLOp.CONST, const_: val });
      leafValues.push(val);
    } else if (tag === TAG_VAR_X) {
      if (cursor + 2 > data.length) {
        throw new EmltDecodeError("truncated VAR_X payload", "carl.eml.decode_error");
      }
      const idx = view.getUint16(cursor, true);
      cursor += 2;
      stack.push({ op: EMLOp.VAR_X, varIdx: idx });
    } else if (tag === TAG_EML) {
      if (stack.length < 2) {
        throw new EmltDecodeError(
          "EML tag but fewer than 2 stack entries",
          "carl.eml.decode_error",
        );
      }
      const right = stack.pop()!;
      const left = stack.pop()!;
      stack.push({ op: EMLOp.EML, left, right });
    } else {
      throw new EmltDecodeError(
        `unknown tag 0x${tag.toString(16).padStart(2, "0")}`,
        "carl.eml.decode_error",
        { tag },
      );
    }
  }

  if (stack.length !== 1) {
    throw new EmltDecodeError(
      `postfix stream left ${stack.length} nodes on stack (expected 1)`,
      "carl.eml.decode_error",
      { stackSize: stack.length },
    );
  }

  const tree: EMLTree = {
    root: stack[0]!,
    inputDim,
    leafParams: leafValues.length > 0 ? Float64Array.from(leafValues) : undefined,
  };

  // Enforce depth after structural decode, mirroring Python's EMLTree invariants.
  assertDepthOk(tree);
  return tree;
}

// -------- envelope encode / decode -------------------------------------------

/**
 * Encode tree into a §1.2 envelope. If `options.signature` is provided, it is
 * appended as the final 32 bytes. Matches
 * `terminals_runtime.eml.codec_impl.encode()`.
 */
export function encodeEnvelope(
  tree: EMLTree,
  options?: { signature?: Uint8Array },
): Uint8Array {
  const sig = options?.signature;
  if (sig !== undefined && sig.length !== SIG_LEN) {
    throw new EmltDecodeError(
      `signature must be ${SIG_LEN} bytes; got ${sig.length}`,
      "carl.eml.domain_error",
      { sigLen: sig.length },
    );
  }
  const inner = encodeInner(tree);
  const total = ENVELOPE_HEADER_LEN + inner.length + (sig ? SIG_LEN : 0);
  const out = new Uint8Array(total);
  out.set(ENVELOPE_MAGIC, 0);
  out[ENVELOPE_MAGIC.length] = ENVELOPE_VERSION;
  out.set(inner, ENVELOPE_HEADER_LEN);
  if (sig) out.set(sig, ENVELOPE_HEADER_LEN + inner.length);
  return out;
}

/**
 * Decode a §1.2 envelope. Implements the same heuristic as
 * `codec_impl.decode`:
 *   - if stripping the trailing 32 bytes yields a parseable body AND the full
 *     remainder does NOT parse, those trailing bytes are a signature;
 *   - if the full remainder parses, there is no signature (the last 32 bytes
 *     were tree-internal data).
 */
export function decodeEnvelope(data: Uint8Array): {
  tree: EMLTree;
  signature?: Uint8Array;
} {
  if (data.length < ENVELOPE_HEADER_LEN) {
    throw new EmltDecodeError("buffer too short for envelope header", "carl.eml.decode_error", {
      size: data.length,
    });
  }
  if (!eqBytes(data, ENVELOPE_MAGIC, ENVELOPE_MAGIC.length)) {
    throw new EmltDecodeError("bad envelope magic", "carl.eml.decode_error", {
      magic: hex(data.subarray(0, ENVELOPE_MAGIC.length)),
    });
  }
  const version = data[ENVELOPE_MAGIC.length]!;
  if (version !== ENVELOPE_VERSION) {
    throw new EmltDecodeError("unsupported envelope version", "carl.eml.decode_error", {
      version,
      supported: ENVELOPE_VERSION,
    });
  }
  const remainder = data.subarray(ENVELOPE_HEADER_LEN);

  // Remainder too short for a signature — must be pure inner body.
  if (remainder.length <= SIG_LEN) {
    return { tree: decodeInner(remainder) };
  }

  const candidateBody = remainder.subarray(0, remainder.length - SIG_LEN);
  const candidateSig = remainder.subarray(remainder.length - SIG_LEN);

  let signedTree: EMLTree | undefined;
  try {
    signedTree = decodeInner(candidateBody);
  } catch {
    signedTree = undefined;
  }

  if (signedTree !== undefined) {
    // Ambiguity check: if the full remainder also parses, no signature is attached.
    let fullTree: EMLTree | undefined;
    try {
      fullTree = decodeInner(remainder);
    } catch {
      fullTree = undefined;
    }
    if (fullTree !== undefined) {
      return { tree: fullTree };
    }
    // Return a fresh copy of the sig to decouple from the input buffer.
    return { tree: signedTree, signature: new Uint8Array(candidateSig) };
  }

  // Stripped body did not parse; the full remainder must be the inner body.
  return { tree: decodeInner(remainder) };
}

// -------- reference evaluator ------------------------------------------------

/** Exp overflow guard (matches `carl_core.eml.CLAMP_X`). */
const CLAMP_X = 20.0;
/** Additive floor guarding ln (matches `carl_core.eml.EPS`). */
const EPS = 1e-12;

function emlScalar(x: number, y: number): number {
  const xc = x > CLAMP_X ? CLAMP_X : x < -CLAMP_X ? -CLAMP_X : x;
  const yc = y > EPS ? y : EPS;
  return Math.exp(xc) - Math.log(yc);
}

/**
 * Reference evaluator for a decoded tree.
 *
 * This is a small scalar evaluator — it is NOT the full Adam-trainable
 * evaluator in the Python reference. Use it to sanity-check a decoded tree.
 *
 * If `tree.leafParams` is set, CONST leaves read from that flat array in
 * in-order traversal (same binding as `encodeInner`); otherwise each CONST
 * node's `const_` is used.
 */
export function evalTree(tree: EMLTree, inputs: Float64Array): number {
  if (inputs.length < tree.inputDim) {
    throw new EmltDecodeError(
      `inputs length ${inputs.length} < inputDim ${tree.inputDim}`,
      "carl.eml.domain_error",
    );
  }
  const params = tree.leafParams;
  const counter = { i: 0 };

  function visit(node: EMLNode): number {
    if (node.op === EMLOp.CONST) {
      const idx = counter.i++;
      if (params !== undefined && idx < params.length) return params[idx]!;
      if (node.const_ === undefined) {
        throw new EmltDecodeError("CONST without const_", "carl.eml.domain_error");
      }
      return node.const_;
    }
    if (node.op === EMLOp.VAR_X) {
      const vi = node.varIdx;
      if (vi === undefined) {
        throw new EmltDecodeError("VAR_X without varIdx", "carl.eml.domain_error");
      }
      return inputs[vi]!;
    }
    if (!node.left || !node.right) {
      throw new EmltDecodeError("EML without children", "carl.eml.domain_error");
    }
    const x = visit(node.left);
    const y = visit(node.right);
    return emlScalar(x, y);
  }

  return visit(tree.root);
}
