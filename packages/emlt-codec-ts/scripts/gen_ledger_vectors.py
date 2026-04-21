#!/usr/bin/env python3
"""Generate shared LedgerBlock test vectors.

Writes packages/emlt-codec-ts/test/ledger_vectors.json with 5 fully-specified
LedgerBlock fixtures plus their expected signing_bytes (hex) and block_hash
(hex) computed by the Python reference. Both the TS tests and
tests/test_ledger_parity_vectors.py (Python side) load this file and assert
their own outputs match it.

Re-run after any change to LedgerBlock.signing_bytes or LedgerBlock.block_hash:

    python packages/emlt-codec-ts/scripts/gen_ledger_vectors.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent  # carl-studio/
CARL_CORE = REPO_ROOT / "packages" / "carl-core" / "src"
if str(CARL_CORE) not in sys.path:
    sys.path.insert(0, str(CARL_CORE))

from carl_core.constitutional import LedgerBlock  # noqa: E402


def _pubkey(tag: int) -> bytes:
    return hashlib.sha256(f"pk-{tag}".encode()).digest()  # 32 bytes


def _signature(tag: int) -> bytes:
    # Not a real ed25519 signature — just 64 deterministic bytes so the
    # block_hash is reproducible. Real chains carry real signatures.
    h = hashlib.sha512(f"sig-{tag}".encode()).digest()
    return h  # 64 bytes


def _make(
    block_id: int,
    prev: str,
    policy: str,
    action: str,
    verdict: float,
    ts_ns: int,
    tag: int,
) -> LedgerBlock:
    return LedgerBlock(
        block_id=block_id,
        prev_block_hash=prev,
        policy_id=policy,
        action_digest=action,
        verdict=verdict,
        timestamp_ns=ts_ns,
        signer_pubkey=_pubkey(tag),
        signature=_signature(tag),
    )


def _vector(name: str, block: LedgerBlock) -> dict:
    return {
        "name": name,
        "input": {
            "block_id": block.block_id,
            "prev_block_hash": block.prev_block_hash,
            "policy_id": block.policy_id,
            "action_digest": block.action_digest,
            "verdict": block.verdict,
            "timestamp_ns": block.timestamp_ns,
            "signer_pubkey_hex": block.signer_pubkey.hex(),
            "signature_hex": block.signature.hex(),
        },
        "expected": {
            "signing_bytes_hex": block.signing_bytes().hex(),
            "block_hash": block.block_hash(),
        },
    }


def main() -> None:
    ZERO = "0" * 64
    vectors = [
        _vector(
            "genesis_zero_verdict",
            _make(0, ZERO, "policy.genesis", "a" * 64, 0.0, 1_700_000_000_000_000_000, 1),
        ),
        _vector(
            "positive_integer_verdict",
            _make(1, "b" * 64, "policy.gate", "c" * 64, 1.0, 1_700_000_001_000_000_000, 2),
        ),
        _vector(
            "negative_fractional_verdict",
            _make(2, "d" * 64, "policy.cost", "e" * 64, -0.75, 1_700_000_002_000_000_000, 3),
        ),
        _vector(
            "small_fractional_verdict",
            _make(3, "f" * 64, "policy.audit", "1" * 64, 0.001, 1_700_000_003_000_000_000, 4),
        ),
        _vector(
            "larger_scalar_verdict",
            _make(4, "a" * 64, "policy.reward", "2" * 64, 4.25, 1_700_000_004_000_000_000, 5),
        ),
    ]

    out_path = HERE.parent / "test" / "ledger_vectors.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vectors, indent=2, sort_keys=False) + "\n")
    print(f"wrote {len(vectors)} vectors -> {out_path}")


if __name__ == "__main__":
    main()
