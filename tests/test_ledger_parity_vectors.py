"""Python ↔ TypeScript ledger parity via shared test vectors.

Both sides load ``packages/emlt-codec-ts/test/ledger_vectors.json`` and
assert their own ``LedgerBlock.signing_bytes()`` / ``block_hash()``
outputs match the expected hex recorded in the file. The file is
generated from this same Python code (see
``packages/emlt-codec-ts/scripts/gen_ledger_vectors.py``), so this test
is effectively a round-trip regression: if someone changes the Python
hashing logic without re-running the generator, the TS side breaks
immediately. If someone changes only the JSON by hand, the Python side
breaks immediately. No drift window.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("nacl", reason="constitutional ledger requires pynacl extra")

from carl_core.constitutional import LedgerBlock  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = (
    REPO_ROOT / "packages" / "emlt-codec-ts" / "test" / "ledger_vectors.json"
)


def _load_vectors() -> list[dict[str, Any]]:
    return json.loads(VECTORS_PATH.read_text())


def _block(v: dict[str, Any]) -> LedgerBlock:
    return LedgerBlock(
        block_id=int(v["block_id"]),
        prev_block_hash=str(v["prev_block_hash"]),
        policy_id=str(v["policy_id"]),
        action_digest=str(v["action_digest"]),
        verdict=float(v["verdict"]),
        timestamp_ns=int(v["timestamp_ns"]),
        signer_pubkey=bytes.fromhex(v["signer_pubkey_hex"]),
        signature=bytes.fromhex(v["signature_hex"]),
    )


def test_vectors_file_exists() -> None:
    assert VECTORS_PATH.is_file(), (
        f"missing shared vectors file at {VECTORS_PATH}. "
        "regenerate with: python packages/emlt-codec-ts/scripts/gen_ledger_vectors.py"
    )


@pytest.mark.parametrize("vector", _load_vectors(), ids=lambda v: v["name"])
def test_signing_bytes_matches_expected(vector: dict[str, Any]) -> None:
    block = _block(vector["input"])
    got = block.signing_bytes().hex()
    assert got == vector["expected"]["signing_bytes_hex"], (
        f"signing_bytes drift for {vector['name']}: "
        f"regenerate ledger_vectors.json"
    )


@pytest.mark.parametrize("vector", _load_vectors(), ids=lambda v: v["name"])
def test_block_hash_matches_expected(vector: dict[str, Any]) -> None:
    block = _block(vector["input"])
    got = block.block_hash()
    assert got == vector["expected"]["block_hash"], (
        f"block_hash drift for {vector['name']}: "
        f"regenerate ledger_vectors.json"
    )


def test_vector_count_is_sensible() -> None:
    vectors = _load_vectors()
    assert len(vectors) >= 5, "expected at least 5 LedgerBlock vectors"


def test_every_vector_has_complete_shape() -> None:
    required_input = {
        "block_id", "prev_block_hash", "policy_id", "action_digest",
        "verdict", "timestamp_ns", "signer_pubkey_hex", "signature_hex",
    }
    required_expected = {"signing_bytes_hex", "block_hash"}
    for v in _load_vectors():
        assert set(v["input"].keys()) >= required_input, v["name"]
        assert set(v["expected"].keys()) >= required_expected, v["name"]
