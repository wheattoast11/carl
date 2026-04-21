/**
 * Core types for the @carl/emlt-codec package.
 *
 * These types mirror the Python reference types in
 * packages/carl-core/src/carl_core/eml.py (EMLOp, EMLNode, EMLTree).
 * The wire format is canonical — see docs/eml_signing_protocol.md §1.1/§1.2.
 */

/** Node operation kind. Values are stable; used in the wire encoding. */
export enum EMLOp {
  CONST = 0,
  VAR_X = 1,
  EML = 2,
}

/**
 * A node in an EML expression tree.
 *
 * Invariants (enforced at construction via helper builders / decoder):
 *   - CONST: `const_` is a finite number; `varIdx`, `left`, `right` unset.
 *   - VAR_X: `varIdx` is a non-negative integer; `const_`, `left`, `right` unset.
 *   - EML  : `left` and `right` are populated; `const_`, `varIdx` unset.
 */
export interface EMLNode {
  readonly op: EMLOp;
  readonly const_?: number;
  readonly varIdx?: number;
  readonly left?: EMLNode;
  readonly right?: EMLNode;
}

/**
 * An EML expression tree with input binding and optional leaf-params override.
 *
 * When `leafParams` is provided, the encoder emits those values for CONST
 * leaves in in-order traversal (matching the Python `_encode_postfix` helper).
 * When absent, each CONST leaf's own `const_` is used verbatim.
 */
export interface EMLTree {
  readonly root: EMLNode;
  readonly inputDim: number;
  readonly leafParams?: Float64Array;
}

/** Semantic decode error. Carries a stable `code` that mirrors Python's. */
export class EmltDecodeError extends Error {
  public readonly code: string;
  public readonly context: Record<string, unknown>;
  constructor(
    message: string,
    code: string = "carl.eml.decode_error",
    context: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "EmltDecodeError";
    this.code = code;
    this.context = context;
  }
}

/** Thrown when tree depth exceeds the Odrzywolek ceiling of 4. */
export class EmltDepthExceededError extends Error {
  public readonly code: string;
  public readonly depth: number;
  public readonly maxDepth: number;
  constructor(depth: number, maxDepth: number) {
    super(`EML tree depth ${depth} exceeds MAX_DEPTH=${maxDepth}`);
    this.name = "EmltDepthExceededError";
    this.code = "carl.eml.depth_exceeded";
    this.depth = depth;
    this.maxDepth = maxDepth;
  }
}
