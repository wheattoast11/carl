/**
 * @terminals-tech/emlt-codec — public API surface.
 *
 * TypeScript port of the EML wire format + software-tier HMAC signature
 * verification for the CARL / carl.camp platform. Spec:
 * docs/eml_signing_protocol.md in the carl-studio repo.
 */

export { EMLOp, type EMLNode, type EMLTree, EmltDecodeError, EmltDepthExceededError } from "./types.js";

export {
  MAX_DEPTH,
  computeDepth,
  nodeDepth,
  nodeCount,
  inputDimRequired,
  assertDepthOk,
} from "./depth.js";

export {
  SIG_LEN,
  ENVELOPE_VERSION,
  INNER_MAGIC,
  ENVELOPE_MAGIC,
  encodeInner,
  decodeInner,
  encodeEnvelope,
  decodeEnvelope,
  evalTree,
} from "./codec.js";

export {
  MIN_SECRET_LEN,
  signSoftware,
  verifySoftware,
  signPlatformCountersig,
  verifyPlatformCountersig,
  type PlatformCountersigParams,
} from "./signing.js";
