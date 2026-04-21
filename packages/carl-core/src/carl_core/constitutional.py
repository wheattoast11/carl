"""Constitutional FSM Ledger — EML-policy-gated, hash-chained, ed25519-signed.

Two EML trees govern the runtime:

- **C (constitutional)** — immutable, signed at genesis. Encodes the policy
  contract: "action is allowed iff eml_eval(C, features(s,a)) > tau".
- **B_t (behavioral)** — mutable, updated per step. Tracks the current
  behavioral fingerprint so a verifier can tell what the agent looks like now
  vs. what the constitution permits.

Every authorized action appends a ``LedgerBlock`` whose hash folds in the
previous block, the policy id, the action digest, the verdict score, the
timestamp, and an ed25519 signature. Tampering with any prior block breaks
the chain.

The ``pynacl`` dependency is optional — signing/verification imports are
lazy. Importing this module (e.g. to read ``ConstitutionalPolicy`` metadata
from disk) never requires pynacl.
"""
from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from carl_core.eml import EMLTree
from carl_core.errors import CredentialError, ValidationError
from carl_core.hashing import content_hash

# ---------------------------------------------------------------------------
# Error helpers (all under carl.constitutional.*)
# ---------------------------------------------------------------------------


def _missing_pynacl() -> ImportError:
    return ImportError(
        "Constitutional ledger requires 'pynacl' (ed25519). "
        "Install: pip install 'carl-studio[constitutional]' "
        "or: pip install pynacl"
    )


def _lazy_nacl() -> tuple[Any, Any]:
    """Return (SigningKey, VerifyKey) classes or raise a typed ImportError."""
    try:
        from nacl.signing import SigningKey, VerifyKey  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without pynacl
        raise _missing_pynacl() from exc
    return SigningKey, VerifyKey


def _bad_chain(msg: str, **ctx: Any) -> ValidationError:
    return ValidationError(
        msg,
        code="carl.constitutional.bad_chain",
        context=ctx,
    )


# ---------------------------------------------------------------------------
# Feature encoding — 25-dim action vector convention.
# ---------------------------------------------------------------------------


FEATURE_DIM: int = 25
"""Fixed feature dimension for constitutional policy evaluation."""

_ACTION_TYPES: tuple[str, ...] = (
    "GATE",          # 0
    "EXTERNAL",      # 1
    "PAYMENT",       # 2
    "TRAINING",      # 3
    "EVAL",          # 4
    "INFER",         # 5
    "TOOL",          # 6
    "LLM",           # 7
    "REWARD",        # 8
    "CONSENT",       # 9
    "CONFIG",        # 10
    "READ",          # 11
    "WRITE",         # 12
    "NETWORK",       # 13
    "PERMISSION",    # 14
    "OTHER",         # 15
)


def encode_action_features(action: dict[str, Any]) -> NDArray[np.float64]:
    """Encode an action dict into the fixed 25-dim feature vector.

    Layout (all floats):
      [0:16]   action_type onehot (GATE..OTHER)
      [16]     amount (0.0 if absent)
      [17:21]  consent flags (4 booleans as 0/1)
      [21]     coherence_phi
      [22]     kuramoto_R
      [23:25]  tier onehot (FREE=0, PAID=1)
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)

    atype = str(action.get("type", "OTHER")).upper()
    try:
        idx = _ACTION_TYPES.index(atype)
    except ValueError:
        idx = _ACTION_TYPES.index("OTHER")
    vec[idx] = 1.0

    vec[16] = float(action.get("amount", 0.0) or 0.0)

    consent_raw: Any = action.get("consent_flags") or {}
    if isinstance(consent_raw, dict):
        consent: dict[str, Any] = cast(dict[str, Any], consent_raw)
        vec[17] = 1.0 if consent.get("telemetry") else 0.0
        vec[18] = 1.0 if consent.get("contract_witnessing") else 0.0
        vec[19] = 1.0 if consent.get("coherence_probe") else 0.0
        vec[20] = 1.0 if consent.get("mcp_share") else 0.0
    elif isinstance(consent_raw, (list, tuple)):
        consent_seq: list[Any] = list(cast("list[Any] | tuple[Any, ...]", consent_raw))
        for i, flag in enumerate(consent_seq[:4]):
            vec[17 + i] = 1.0 if flag else 0.0

    vec[21] = float(action.get("coherence_phi", 0.0) or 0.0)
    vec[22] = float(action.get("kuramoto_R", 0.0) or 0.0)

    tier = str(action.get("tier", "FREE")).upper()
    if tier == "FREE":
        vec[23] = 1.0
    elif tier == "PAID":
        vec[24] = 1.0
    return vec


# ---------------------------------------------------------------------------
# ConstitutionalPolicy — EML tree + threshold, identified by sha256.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstitutionalPolicy:
    """Immutable policy: an EML tree, a threshold, and metadata.

    ``policy_id`` is sha256 of (tree.hash + threshold + metadata). Two policies
    with the same tree but different thresholds produce different ids.
    """

    tree: EMLTree
    threshold: float
    policy_id: str
    metadata: dict[str, Any] = field(default_factory=lambda: cast("dict[str, Any]", {}))

    @classmethod
    def create(
        cls,
        tree: EMLTree,
        threshold: float,
        metadata: dict[str, Any] | None = None,
    ) -> ConstitutionalPolicy:
        md: dict[str, Any] = dict(metadata or {})
        pid = content_hash(
            {
                "tree_hash": tree.hash(),
                "threshold": float(threshold),
                "metadata": md,
            }
        )
        return cls(tree=tree, threshold=float(threshold), policy_id=pid, metadata=md)

    def evaluate(
        self, action_features: NDArray[np.floating[Any]]
    ) -> tuple[bool, float]:
        """Run the EML tree and compare against the threshold."""
        score = float(self.tree.forward(np.asarray(action_features, dtype=np.float64)))
        return score > self.threshold, score

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "threshold": self.threshold,
            "metadata": self.metadata,
            "tree_bytes_hex": self.tree.to_bytes().hex(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConstitutionalPolicy:
        tree = EMLTree.from_bytes(bytes.fromhex(d["tree_bytes_hex"]))
        return cls(
            tree=tree,
            threshold=float(d["threshold"]),
            policy_id=str(d["policy_id"]),
            metadata=dict(d.get("metadata") or {}),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2))

    @classmethod
    def load(cls, path: Path) -> ConstitutionalPolicy:
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# LedgerBlock — hash-chained + ed25519-signed record of an authorized action.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerBlock:
    block_id: int
    prev_block_hash: str
    policy_id: str
    action_digest: str
    verdict: float
    timestamp_ns: int
    signer_pubkey: bytes
    signature: bytes

    def signing_bytes(self) -> bytes:
        """Deterministic bytes that the signature covers.

        Layout: prev_block_hash || policy_id || action_digest ||
        verdict (float64 little-endian) || timestamp_ns (int64 LE) ||
        signer_pubkey.
        """
        return b"".join(
            [
                self.prev_block_hash.encode("ascii"),
                b"|",
                self.policy_id.encode("ascii"),
                b"|",
                self.action_digest.encode("ascii"),
                b"|",
                struct.pack("<d", float(self.verdict)),
                struct.pack("<q", int(self.timestamp_ns)),
                self.signer_pubkey,
            ]
        )

    def block_hash(self) -> str:
        """sha256 of the full block including signature."""
        return content_hash(
            {
                "block_id": int(self.block_id),
                "prev_block_hash": self.prev_block_hash,
                "policy_id": self.policy_id,
                "action_digest": self.action_digest,
                "verdict": float(self.verdict),
                "timestamp_ns": int(self.timestamp_ns),
                "signer_pubkey": self.signer_pubkey.hex(),
                "signature": self.signature.hex(),
            }
        )

    def verify(self, pubkey: bytes | None = None) -> bool:
        """Verify the ed25519 signature over ``signing_bytes``."""
        _, VerifyKey = _lazy_nacl()
        pk = pubkey if pubkey is not None else self.signer_pubkey
        try:
            vk = VerifyKey(pk)
            vk.verify(self.signing_bytes(), self.signature)
            return True
        except Exception:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": int(self.block_id),
            "prev_block_hash": self.prev_block_hash,
            "policy_id": self.policy_id,
            "action_digest": self.action_digest,
            "verdict": float(self.verdict),
            "timestamp_ns": int(self.timestamp_ns),
            "signer_pubkey_hex": self.signer_pubkey.hex(),
            "signature_hex": self.signature.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LedgerBlock:
        return cls(
            block_id=int(d["block_id"]),
            prev_block_hash=str(d["prev_block_hash"]),
            policy_id=str(d["policy_id"]),
            action_digest=str(d["action_digest"]),
            verdict=float(d["verdict"]),
            timestamp_ns=int(d["timestamp_ns"]),
            signer_pubkey=bytes.fromhex(str(d["signer_pubkey_hex"])),
            signature=bytes.fromhex(str(d["signature_hex"])),
        )


# ---------------------------------------------------------------------------
# ConstitutionalLedger — append-only, hash-chained, persisted on disk.
# ---------------------------------------------------------------------------


_BLOCKS_FILENAME: str = "chain.jsonl"
_POLICY_FILENAME: str = "policy.json"
_KEY_FILENAME: str = "signing_key.bin"


class ConstitutionalLedger:
    """Append-only hash-chained ledger of policy-authorized actions.

    On disk layout under ``root``::

        policy.json          # ConstitutionalPolicy JSON
        chain.jsonl          # one LedgerBlock JSON per line
        signing_key.bin      # raw 32-byte ed25519 seed (0600 perms)

    ``signing_key`` may be provided directly (tests) or loaded from disk.
    When no key is present on disk and none is supplied, a fresh key is
    generated at genesis and persisted.
    """

    def __init__(self, root: Path, signing_key: bytes | None = None) -> None:
        self.root = Path(root)
        self._signing_seed: bytes | None = signing_key
        self._pubkey: bytes | None = None
        self._policy: ConstitutionalPolicy | None = None
        # In-memory cache of loaded blocks (rehydrated on demand).
        self._blocks: list[LedgerBlock] | None = None

    # ----- key management -------------------------------------------------

    def _key_path(self) -> Path:
        return self.root / _KEY_FILENAME

    def _load_or_create_key(self) -> bytes:
        if self._signing_seed is not None:
            return self._signing_seed
        kp = self._key_path()
        if kp.exists():
            self._signing_seed = kp.read_bytes()
            return self._signing_seed
        SigningKey, _ = _lazy_nacl()
        seed = bytes(SigningKey.generate())  # 32 raw bytes
        # nacl's SigningKey.__bytes__ returns the raw seed.
        self.root.mkdir(parents=True, exist_ok=True)
        kp.write_bytes(seed)
        try:
            os.chmod(kp, 0o600)
        except OSError:  # pragma: no cover - best-effort on non-POSIX
            pass
        self._signing_seed = seed
        return seed

    def _signer(self) -> Any:
        SigningKey, _ = _lazy_nacl()
        seed = self._load_or_create_key()
        return SigningKey(seed)

    def pubkey(self) -> bytes:
        if self._pubkey is not None:
            return self._pubkey
        sk = self._signer()
        self._pubkey = bytes(sk.verify_key)
        return self._pubkey

    # ----- policy management ----------------------------------------------

    def _policy_path(self) -> Path:
        return self.root / _POLICY_FILENAME

    def policy(self) -> ConstitutionalPolicy:
        if self._policy is None:
            p = self._policy_path()
            if not p.exists():
                raise _bad_chain(
                    "ledger has no policy — call genesis() first", root=str(self.root)
                )
            self._policy = ConstitutionalPolicy.load(p)
        return self._policy

    # ----- chain I/O ------------------------------------------------------

    def _chain_path(self) -> Path:
        return self.root / _BLOCKS_FILENAME

    def _append_block_file(self, block: LedgerBlock) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._chain_path().open("a") as fh:
            fh.write(json.dumps(block.to_dict(), sort_keys=True) + "\n")

    def _load_blocks(self) -> list[LedgerBlock]:
        if self._blocks is not None:
            return self._blocks
        p = self._chain_path()
        if not p.exists():
            self._blocks = []
            return self._blocks
        blocks: list[LedgerBlock] = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            blocks.append(LedgerBlock.from_dict(json.loads(line)))
        self._blocks = blocks
        return blocks

    # ----- public operations ---------------------------------------------

    def genesis(self, policy: ConstitutionalPolicy) -> LedgerBlock:
        """Install ``policy`` and sign the genesis block.

        Raises ``ValidationError`` if a policy already exists at this root.
        """
        if self._policy_path().exists():
            raise _bad_chain(
                "genesis already exists at this root", root=str(self.root)
            )
        policy.save(self._policy_path())
        self._policy = policy

        # Genesis block: prev_block_hash is a zero-sha256 marker, action_digest
        # is the hash of the policy, verdict is the threshold itself.
        genesis_action = {"type": "GENESIS", "policy_id": policy.policy_id}
        block = self._sign_block(
            block_id=0,
            prev_block_hash="0" * 64,
            policy_id=policy.policy_id,
            action_digest=content_hash(genesis_action),
            verdict=float(policy.threshold),
        )
        self._append_block_file(block)
        if self._blocks is None:
            self._blocks = []
        self._blocks.append(block)
        return block

    def append(self, action: dict[str, Any], policy_id: str) -> LedgerBlock:
        """Append a new block recording an authorized action.

        The caller is responsible for having already evaluated the policy
        (via ``ConstitutionalPolicy.evaluate``) and decided to proceed.
        """
        policy = self.policy()
        if policy_id != policy.policy_id:
            raise _bad_chain(
                "policy_id mismatch", given=policy_id, expected=policy.policy_id
            )
        blocks = self._load_blocks()
        if not blocks:
            raise _bad_chain("no genesis block — call genesis() first")

        features = encode_action_features(action)
        _, score = policy.evaluate(features)

        prev = blocks[-1]
        block = self._sign_block(
            block_id=prev.block_id + 1,
            prev_block_hash=prev.block_hash(),
            policy_id=policy.policy_id,
            action_digest=content_hash(action),
            verdict=float(score),
        )
        self._append_block_file(block)
        blocks.append(block)
        return block

    def verify_chain(self) -> tuple[bool, list[int]]:
        """Walk the chain end-to-end, validating hashes + signatures."""
        blocks = self._load_blocks()
        bad: list[int] = []
        expected_prev = "0" * 64
        for i, blk in enumerate(blocks):
            if blk.block_id != i:
                bad.append(i)
                continue
            if blk.prev_block_hash != expected_prev:
                bad.append(i)
                expected_prev = blk.block_hash()
                continue
            if not blk.verify():
                bad.append(i)
                expected_prev = blk.block_hash()
                continue
            expected_prev = blk.block_hash()
        return (len(bad) == 0, bad)

    def head(self) -> LedgerBlock | None:
        blocks = self._load_blocks()
        return blocks[-1] if blocks else None

    def replay(self, until_block: int | None = None) -> list[LedgerBlock]:
        blocks = self._load_blocks()
        if until_block is None:
            return list(blocks)
        return [b for b in blocks if b.block_id <= until_block]

    def height(self) -> int:
        return len(self._load_blocks())

    # ----- internals -----------------------------------------------------

    def _sign_block(
        self,
        *,
        block_id: int,
        prev_block_hash: str,
        policy_id: str,
        action_digest: str,
        verdict: float,
    ) -> LedgerBlock:
        try:
            sk = self._signer()
        except ImportError:
            raise
        except Exception as exc:
            raise CredentialError(
                f"failed to load signing key: {exc}",
                code="carl.constitutional.bad_key",
            ) from exc

        pk = bytes(sk.verify_key)
        ts = time.time_ns()
        pending = LedgerBlock(
            block_id=block_id,
            prev_block_hash=prev_block_hash,
            policy_id=policy_id,
            action_digest=action_digest,
            verdict=float(verdict),
            timestamp_ns=int(ts),
            signer_pubkey=pk,
            signature=b"",
        )
        signed = sk.sign(pending.signing_bytes())
        return LedgerBlock(
            block_id=block_id,
            prev_block_hash=prev_block_hash,
            policy_id=policy_id,
            action_digest=action_digest,
            verdict=float(verdict),
            timestamp_ns=int(ts),
            signer_pubkey=pk,
            signature=bytes(signed.signature),
        )


__all__ = [
    "FEATURE_DIM",
    "encode_action_features",
    "ConstitutionalPolicy",
    "LedgerBlock",
    "ConstitutionalLedger",
]
