/**
 * Tree depth utilities. Depth is capped at MAX_DEPTH = 4 (Odrzywolek ceiling);
 * the Python encoder/decoder enforces this invariant, and so do we.
 */

import { EMLOp, type EMLNode, type EMLTree, EmltDepthExceededError } from "./types.js";

export const MAX_DEPTH = 4 as const;

/** Depth of a single node: leaves = 0, EML(l, r) = 1 + max(depth(l), depth(r)). */
export function nodeDepth(node: EMLNode): number {
  if (node.op === EMLOp.CONST || node.op === EMLOp.VAR_X) return 0;
  if (!node.left || !node.right) {
    // Structural guarantee violated; callers should have validated first.
    return 0;
  }
  return 1 + Math.max(nodeDepth(node.left), nodeDepth(node.right));
}

/** Depth of a full tree. */
export function computeDepth(tree: EMLTree): number {
  return nodeDepth(tree.root);
}

/** Throws `EmltDepthExceededError` if `depth(tree) > MAX_DEPTH`. */
export function assertDepthOk(tree: EMLTree): void {
  const d = computeDepth(tree);
  if (d > MAX_DEPTH) {
    throw new EmltDepthExceededError(d, MAX_DEPTH);
  }
}

/** Smallest `inputDim` that satisfies all VAR_X references in the tree. */
export function inputDimRequired(node: EMLNode): number {
  if (node.op === EMLOp.CONST) return 0;
  if (node.op === EMLOp.VAR_X) return (node.varIdx ?? 0) + 1;
  if (!node.left || !node.right) return 0;
  return Math.max(inputDimRequired(node.left), inputDimRequired(node.right));
}

/** Total node count (leaves + EML nodes). */
export function nodeCount(node: EMLNode): number {
  if (node.op === EMLOp.CONST || node.op === EMLOp.VAR_X) return 1;
  if (!node.left || !node.right) return 1;
  return 1 + nodeCount(node.left) + nodeCount(node.right);
}
