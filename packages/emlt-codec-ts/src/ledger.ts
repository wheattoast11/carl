/**
 * Constitutional ledger canonicalization + ed25519 signing bytes.
 *
 * Byte-identical to the Python reference in
 *   packages/carl-core/src/carl_core/constitutional.py (LedgerBlock.signing_bytes, .block_hash)
 *   packages/carl-core/src/carl_core/hashing.py       (canonical_json)
 *
 * Spec: docs/eml_signing_protocol.md §3 in the carl-studio repo.
 *
 * Scope note: `canonicalJson` here is scoped to the LedgerBlock schema
 * — it handles strings, integers (number or bigint), booleans, null,
 * floats (with an explicit `floatKeys` opt-in for the single `verdict`
 * field), nested objects, and arrays. It does NOT try to match the full
 * `carl_core.hashing.canonical_json` breadth (no Decimal / datetime /
 * Path / bytes coercion). Keep it that way — the ledger schema is
 * fixed; broader canonicalization belongs in a separate package.
 */

import { createHash } from "node:crypto";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export class CanonicalizationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CanonicalizationError";
  }
}

/**
 * Constitutional ledger block. Field order / types match
 * `carl_core.constitutional.LedgerBlock`.
 *
 * `signer_pubkey` and `signature` accept either hex strings (as stored
 * in JSON) or raw bytes (as produced by ed25519 libraries). Functions
 * normalize internally to the canonical form required by each spec
 * path.
 */
export interface LedgerBlock {
  /** 0-based block index in the chain for this user. */
  block_id: number | bigint;
  /** Previous block's hash (hex, 64 chars); zero-sha256 for genesis. */
  prev_block_hash: string;
  /** Policy identifier (ASCII; carl.yaml id or similar). */
  policy_id: string;
  /** sha256 hex of canonicalized action inputs. */
  action_digest: string;
  /** Signed scalar verdict from policy evaluation. */
  verdict: number;
  /** Nanoseconds since epoch. Accepts number or bigint; int64 on the wire. */
  timestamp_ns: number | bigint;
  /** 32-byte ed25519 public key, hex-encoded (64 chars) or raw. */
  signer_pubkey: string | Uint8Array;
  /** 64-byte ed25519 signature, hex-encoded (128 chars) or raw. */
  signature: string | Uint8Array;
}

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0) {
    throw new CanonicalizationError(`hex string has odd length: ${hex.length}`);
  }
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    const byte = parseInt(hex.substr(i * 2, 2), 16);
    if (Number.isNaN(byte)) {
      throw new CanonicalizationError(`invalid hex at offset ${i * 2}`);
    }
    out[i] = byte;
  }
  return out;
}

function bytesToHex(b: Uint8Array): string {
  let out = "";
  for (let i = 0; i < b.length; i++) {
    out += b[i].toString(16).padStart(2, "0");
  }
  return out;
}

function toBytes(v: string | Uint8Array): Uint8Array {
  return typeof v === "string" ? hexToBytes(v) : v;
}

function toHex(v: string | Uint8Array): string {
  return typeof v === "string" ? v : bytesToHex(v);
}

function asciiBytes(s: string): Uint8Array {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    if (code > 0x7f) {
      throw new CanonicalizationError(`non-ASCII character at index ${i} in "${s}"`);
    }
    out[i] = code;
  }
  return out;
}

function concatBytes(chunks: Uint8Array[]): Uint8Array {
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

// ---------------------------------------------------------------------------
// JSON primitives — match Python json.dumps under the LedgerBlock scope
// ---------------------------------------------------------------------------

/** JSON string quoted with ASCII-only escape. Matches ensure_ascii=True. */
function jsonStringAscii(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    const ch = s[i];
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (code === 0x08) out += "\\b";
    else if (code === 0x09) out += "\\t";
    else if (code === 0x0a) out += "\\n";
    else if (code === 0x0c) out += "\\f";
    else if (code === 0x0d) out += "\\r";
    else if (code < 0x20 || code >= 0x7f) {
      out += "\\u" + code.toString(16).padStart(4, "0");
    } else {
      out += ch;
    }
  }
  return out + '"';
}

/** Match Python `json.dumps(int)` — bare integer literal. */
function intJson(n: number | bigint): string {
  if (typeof n === "bigint") return n.toString();
  if (!Number.isFinite(n) || !Number.isInteger(n)) {
    throw new CanonicalizationError(`expected integer, got ${n}`);
  }
  return n.toString();
}

/**
 * Match Python `json.dumps(float)` for the LedgerBlock `verdict` scope.
 *
 * Key parity rules (verified against Python's float.__repr__ on 64-bit
 * CPython 3.11+):
 *   - NaN / inf → throw CanonicalizationError (matches Python's refusal
 *     to emit these in strict JSON mode).
 *   - Integer-valued floats get a trailing ".0"  (TS drops it by default).
 *   - `Object.is(n, -0)` → "-0.0"  (matches Python repr(-0.0)).
 *   - Non-integer finite floats → TS Number.prototype.toString(), which
 *     matches Python's shortest-round-trip repr for values in the
 *     typical ledger range of [-1e15, 1e15] excluding extreme subnormals.
 *     Values outside this range are REJECTED — the LedgerBlock verdict
 *     is constrained to a small scalar in practice.
 */
function floatJson(n: number): string {
  if (Number.isNaN(n)) {
    throw new CanonicalizationError("cannot canonicalize NaN");
  }
  if (!Number.isFinite(n)) {
    throw new CanonicalizationError("cannot canonicalize Infinity");
  }
  // Guard against ambiguous exponent formatting at the extremes.
  if (n !== 0 && (Math.abs(n) < 1e-15 || Math.abs(n) >= 1e16)) {
    throw new CanonicalizationError(
      `verdict magnitude ${n} outside the safe canonicalization range; ` +
        "refusing to emit ambiguous float",
    );
  }
  if (Object.is(n, -0)) return "-0.0";
  if (Number.isInteger(n)) return `${n}.0`;
  return n.toString();
}

// ---------------------------------------------------------------------------
// canonicalJson — scoped to LedgerBlock shape
// ---------------------------------------------------------------------------

export interface CanonicalJsonOptions {
  /**
   * Set of dotted-path keys to render as floats. Defaults to `["verdict"]`
   * (the LedgerBlock float field). Paths are simple JSON keys — not
   * nested paths — so unique field names are fine.
   *
   * Example: `{ floatKeys: ["verdict", "score"] }`.
   */
  floatKeys?: ReadonlySet<string> | readonly string[];
}

const LEDGER_FLOAT_KEYS: ReadonlySet<string> = new Set(["verdict"]);

type CanonicalValue =
  | null
  | boolean
  | string
  | number
  | bigint
  | readonly CanonicalValue[]
  | { readonly [k: string]: CanonicalValue };

function encodeValue(
  value: CanonicalValue,
  floatKeys: ReadonlySet<string>,
  parentKey: string | null,
): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return jsonStringAscii(value);
  if (typeof value === "bigint") return intJson(value);
  if (typeof value === "number") {
    if (parentKey !== null && floatKeys.has(parentKey)) return floatJson(value);
    // Default: treat whole numbers as ints, fractional as floats. This
    // matches Python's type-aware json.dumps for dict values that were
    // originally ints. Consumers who need a float should declare the
    // key in `floatKeys`.
    if (Number.isInteger(value)) return intJson(value);
    return floatJson(value);
  }
  if (Array.isArray(value)) {
    const parts: string[] = [];
    for (const item of value) parts.push(encodeValue(item, floatKeys, null));
    return "[" + parts.join(",") + "]";
  }
  if (typeof value === "object") {
    const obj = value as { [k: string]: CanonicalValue };
    const keys = Object.keys(obj).sort();
    const parts: string[] = [];
    for (const k of keys) {
      parts.push(`${jsonStringAscii(k)}:${encodeValue(obj[k], floatKeys, k)}`);
    }
    return "{" + parts.join(",") + "}";
  }
  throw new CanonicalizationError(
    `unsupported value of type ${typeof value}`,
  );
}

/**
 * Deterministic JSON encoding for the LedgerBlock schema scope.
 *
 *   - Object keys sorted lexicographically
 *   - No whitespace (separators `(",", ":")`)
 *   - ASCII-only (non-ASCII chars escaped as `\uXXXX`)
 *   - NaN / Infinity rejected
 *   - `floatKeys` forces integer-valued numbers to emit with ".0" suffix
 *
 * Matches `carl_core.hashing.canonical_json` output exactly for dict
 * values composed of str / int / float / bool / None / list / dict.
 */
export function canonicalJson(
  value: CanonicalValue,
  options: CanonicalJsonOptions = {},
): string {
  const fk = options.floatKeys ?? LEDGER_FLOAT_KEYS;
  const floatKeys: ReadonlySet<string> = fk instanceof Set ? fk : new Set(fk);
  return encodeValue(value, floatKeys, null);
}

// ---------------------------------------------------------------------------
// LedgerBlock-specific helpers
// ---------------------------------------------------------------------------

/**
 * Binary bytes the ed25519 signature covers. Byte-identical to
 * `LedgerBlock.signing_bytes()` in Python.
 *
 * Layout (§3.1):
 *   prev_block_hash (ascii) || b"|" ||
 *   policy_id (ascii)       || b"|" ||
 *   action_digest (ascii)   || b"|" ||
 *   verdict (float64 LE, 8 bytes) ||
 *   timestamp_ns (int64 LE, 8 bytes) ||
 *   signer_pubkey (raw 32 bytes)
 */
export function ledgerBlockSigningBytes(block: LedgerBlock): Uint8Array {
  const verdictBuf = new ArrayBuffer(8);
  new DataView(verdictBuf).setFloat64(0, block.verdict, true);
  const tsBuf = new ArrayBuffer(8);
  const tsBigInt =
    typeof block.timestamp_ns === "bigint"
      ? block.timestamp_ns
      : BigInt(block.timestamp_ns);
  new DataView(tsBuf).setBigInt64(0, tsBigInt, true);

  const pubkey = toBytes(block.signer_pubkey);
  if (pubkey.length !== 32) {
    throw new CanonicalizationError(
      `signer_pubkey must be 32 bytes; got ${pubkey.length}`,
    );
  }

  return concatBytes([
    asciiBytes(block.prev_block_hash),
    asciiBytes("|"),
    asciiBytes(block.policy_id),
    asciiBytes("|"),
    asciiBytes(block.action_digest),
    asciiBytes("|"),
    new Uint8Array(verdictBuf),
    new Uint8Array(tsBuf),
    pubkey,
  ]);
}

/**
 * sha256 hex of canonical JSON of the full block (including signature).
 * Byte-identical to `LedgerBlock.block_hash()` in Python.
 *
 * Used for chain integrity (`prev_block_hash` on the next block).
 */
export function ledgerBlockHash(block: LedgerBlock): string {
  const asDict = {
    action_digest: block.action_digest,
    block_id:
      typeof block.block_id === "bigint" ? block.block_id : BigInt(block.block_id),
    policy_id: block.policy_id,
    prev_block_hash: block.prev_block_hash,
    signature: toHex(block.signature),
    signer_pubkey: toHex(block.signer_pubkey),
    timestamp_ns:
      typeof block.timestamp_ns === "bigint"
        ? block.timestamp_ns
        : BigInt(block.timestamp_ns),
    verdict: block.verdict,
  };
  const json = canonicalJson(asDict);
  return createHash("sha256").update(json, "utf8").digest("hex");
}
