"""Constitutional Ledger — public client façade.

Public surface (MIT) for the constitutional ledger primitive. The full
proprietary implementation — genesis/append lifecycle, ed25519 signing,
hash-chain construction, 25-dim action feature encoder — lives in the
private ``resonance.signals.constitutional`` module and is loaded
lazily via the admin-gate pattern (see ``carl_studio.admin.load_private``).

**What stays public** (required for TS ``@terminals-tech/emlt-codec``
parity per ``docs/eml_signing_protocol.md``):

- ``LedgerBlock`` dataclass shape + ``signing_bytes`` / ``block_hash``
  / ``to_dict`` / ``from_dict`` / ``verify``. These are pure-data or
  pure-crypto operations shared with the TypeScript sibling; hiding
  them would break wire-format parity (the ledger_vectors.json
  regression tests cover this).
- ``ConstitutionalPolicy`` dataclass shape + ``evaluate`` / ``to_dict``
  / ``from_dict`` / ``save`` / ``load``. Pure numpy against an EMLTree.
- ``encode_action_features`` — the 25-dim feature encoder function
  shape is a wire-format concern (policies live on disk and are
  shared across systems).

**What's gated** (routed through ``resonance.signals.constitutional``):

- ``ConstitutionalLedger.genesis`` / ``ConstitutionalLedger.append``
  / ``ConstitutionalLedger.verify_chain`` — the append-only,
  signing-key-managing, hash-chain-walking lifecycle. These are the
  proprietary combinations; without admin unlock they raise
  ``ConstitutionalLedgerError(code='carl.constitutional.private_required')``.

The pure-read methods (``head`` / ``replay`` / ``height`` / ``policy``)
work locally against persisted ledger data — useful for read-only
verification workflows that don't need to mint new blocks.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from carl_core.eml import EMLTree
from carl_core.errors import CARLError, ValidationError
from carl_core.hashing import content_hash

# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class ConstitutionalLedgerError(CARLError):
    """Raised when a mutating ledger operation runs without private runtime."""

    code = "carl.constitutional.private_required"


def _missing_pynacl() -> ImportError:
    return ImportError(
        "Constitutional ledger requires 'pynacl' (ed25519). "
        "Install: pip install 'carl-studio[constitutional]' "
        "or: pip install pynacl"
    )


def _lazy_nacl() -> tuple[Any, Any]:
    """Return (SigningKey, VerifyKey) classes or raise a typed ImportError.

    Kept public: ed25519 verify is standard crypto, no proprietary math,
    and the TS sibling's parity vectors round-trip through it.
    """
    try:
        from nacl.signing import SigningKey, VerifyKey  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise _missing_pynacl() from exc
    return SigningKey, VerifyKey


def _bad_chain(msg: str, **ctx: Any) -> ValidationError:
    return ValidationError(
        msg,
        code="carl.constitutional.bad_chain",
        context=ctx,
    )


# ---------------------------------------------------------------------------
# Private-runtime resolution — lazy, admin-gated.
# ---------------------------------------------------------------------------


def _load_private_impl() -> ModuleType | None:
    """Resolve the private resonance.signals.constitutional implementation.

    Uses a runtime import of ``carl_studio.admin`` to invoke ``load_private``.
    carl-core cannot import carl-studio at module-load time (carl-core is
    upstream), so this dance stays inside a function body.

    Returns the private module on admin unlock, or ``None`` when the gate
    is locked. Any underlying ImportError is caught — callers decide how
    to surface the locked state (typically by raising
    ``ConstitutionalLedgerError``).
    """
    try:
        admin = import_module("carl_studio.admin")
    except ImportError:
        return None
    try:
        is_admin_fn = getattr(admin, "is_admin", None)
        load_private_fn = getattr(admin, "load_private", None)
        if is_admin_fn is None or load_private_fn is None:
            return None
        if not is_admin_fn():
            return None
        mod = load_private_fn("signals.constitutional")
        return cast(ModuleType, mod)
    except ImportError:
        return None
    except Exception:
        return None


def _require_private(op: str) -> ModuleType:
    """Return the private impl or raise ``ConstitutionalLedgerError``."""
    mod = _load_private_impl()
    if mod is None:
        raise ConstitutionalLedgerError(
            f"ConstitutionalLedger.{op} requires the private resonance runtime. "
            "Unlock admin mode (`carl admin unlock`) or install the resonance "
            "package. Public carl-core ships read-only ledger access.",
            code="carl.constitutional.private_required",
            context={"op": op},
        )
    return mod


# ---------------------------------------------------------------------------
# Feature encoding — 25-dim action vector convention.
#
# Kept public because policies live on disk (``policy.json``) and are
# exchanged across runtimes; the feature layout is a wire-format
# concern, not a proprietary secret.
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
# ConstitutionalPolicy — pure-data + pure-math. Kept public.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstitutionalPolicy:
    """Immutable policy: an EML tree, a threshold, and metadata.

    ``policy_id`` is sha256 of (tree.hash + threshold + metadata). Two policies
    with the same tree but different thresholds produce different ids.

    Kept public so ``policy.json`` files remain exchangeable across runtimes
    and the TS parity vectors continue to round-trip.
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
# LedgerBlock — pure-data + pure-crypto. Kept public for TS parity.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerBlock:
    """Hash-chained + ed25519-signed record of an authorized action.

    The dataclass shape + ``signing_bytes`` / ``block_hash`` layout is
    shared with ``@terminals-tech/emlt-codec`` (see
    ``packages/emlt-codec-ts/test/ledger_vectors.json``). Changing this
    surface breaks the npm sibling; keep parity by updating both.
    """

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
        """Verify the ed25519 signature over ``signing_bytes``.

        Kept public: standard crypto verify with no proprietary component,
        shared with the TS parity path. Requires pynacl at call time.
        """
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
# ConstitutionalLedger — client façade.
#
# Mutating + signing operations (genesis, append, verify_chain) route
# through the private ``resonance.signals.constitutional`` implementation
# via the admin-gate pattern. Read-only operations (head, replay, height,
# policy) work locally against any already-persisted ledger data so that
# verifiers without admin unlock can still inspect a chain that was minted
# on an unlocked host.
# ---------------------------------------------------------------------------


_BLOCKS_FILENAME: str = "chain.jsonl"
_POLICY_FILENAME: str = "policy.json"
_KEY_FILENAME: str = "signing_key.bin"


class ConstitutionalLedger:
    """Public façade for the append-only, hash-chained, ed25519-signed ledger.

    With the private ``resonance`` runtime available (admin-unlocked host),
    ``genesis`` / ``append`` / ``verify_chain`` delegate to the real
    implementation. Without the runtime they raise
    ``ConstitutionalLedgerError`` (``code='carl.constitutional.private_required'``).

    Read-only operations (``head`` / ``replay`` / ``height`` / ``policy``)
    work locally against any already-persisted ledger data.
    """

    def __init__(self, root: Path, signing_key: bytes | None = None) -> None:
        self.root = Path(root)
        self._signing_seed: bytes | None = signing_key
        self._pubkey: bytes | None = None
        self._policy: ConstitutionalPolicy | None = None
        # In-memory cache of loaded blocks (rehydrated on demand).
        self._blocks: list[LedgerBlock] | None = None
        # Lazily-bound private impl. Only populated if a mutating call
        # succeeds in loading it.
        self._priv_ledger: Any | None = None

    # ----- private-runtime binding ----------------------------------------

    def _bind_private(self, op: str) -> Any:
        """Return a private ``ConstitutionalLedger`` bound to the same root.

        Raises ``ConstitutionalLedgerError`` when the admin gate is locked.
        """
        if self._priv_ledger is not None:
            return self._priv_ledger
        mod = _require_private(op)
        priv_cls = getattr(mod, "ConstitutionalLedger", None)
        if priv_cls is None:  # pragma: no cover - defensive
            raise ConstitutionalLedgerError(
                "private resonance.signals.constitutional missing "
                "ConstitutionalLedger",
                code="carl.constitutional.private_required",
                context={"op": op},
            )
        self._priv_ledger = priv_cls(self.root, signing_key=self._signing_seed)
        return self._priv_ledger

    # ----- key + pubkey accessors (routed) --------------------------------

    def pubkey(self) -> bytes:
        if self._pubkey is not None:
            return self._pubkey
        priv = self._bind_private("pubkey")
        pk = bytes(priv.pubkey())
        self._pubkey = pk
        return pk

    # ----- policy management (read-only, no gating) ----------------------

    def _policy_path(self) -> Path:
        return self.root / _POLICY_FILENAME

    def policy(self) -> ConstitutionalPolicy:
        """Load the on-disk policy. Read-only; no admin gate required."""
        if self._policy is None:
            p = self._policy_path()
            if not p.exists():
                raise _bad_chain(
                    "ledger has no policy — call genesis() first",
                    root=str(self.root),
                )
            self._policy = ConstitutionalPolicy.load(p)
        return self._policy

    # ----- chain I/O (read-only, no gating) -------------------------------

    def _chain_path(self) -> Path:
        return self.root / _BLOCKS_FILENAME

    def _load_blocks(self) -> list[LedgerBlock]:
        """Load blocks from ``chain.jsonl``. Pure-data; no gating."""
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

    # ----- mutating / signing operations (gated) --------------------------

    def genesis(self, policy: ConstitutionalPolicy) -> LedgerBlock:
        """Install ``policy`` and sign the genesis block.

        Requires the private resonance runtime (admin unlock). Raises
        ``ConstitutionalLedgerError`` when locked.
        """
        priv = self._bind_private("genesis")
        block = priv.genesis(policy)
        # Warm our local caches so subsequent read-only accessors work.
        self._policy = policy
        self._blocks = None  # force reload from disk
        return cast(LedgerBlock, block)

    def append(self, action: dict[str, Any], policy_id: str) -> LedgerBlock:
        """Append a new block recording an authorized action.

        Requires the private resonance runtime.
        """
        priv = self._bind_private("append")
        block = priv.append(action, policy_id)
        self._blocks = None  # force reload
        return cast(LedgerBlock, block)

    def verify_chain(self) -> tuple[bool, list[int]]:
        """Walk the chain end-to-end, validating hashes + signatures.

        Requires the private resonance runtime for full signature checks.
        """
        priv = self._bind_private("verify_chain")
        ok, bad = priv.verify_chain()
        return (bool(ok), list(bad))

    # ----- read-only accessors (no gating) --------------------------------

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


# ---------------------------------------------------------------------------
# Backwards-compat: file-layout constants still exported so callers that
# poke at ``ledger.root / chain.jsonl`` directly keep working.
# ---------------------------------------------------------------------------


# Used by some tests + the CLI status path.
_ = (_BLOCKS_FILENAME, _POLICY_FILENAME, _KEY_FILENAME)


__all__ = [
    "ConstitutionalLedger",
    "ConstitutionalLedgerError",
    "ConstitutionalPolicy",
    "FEATURE_DIM",
    "LedgerBlock",
    "encode_action_features",
]
