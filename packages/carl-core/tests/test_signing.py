"""Tests for the public software-tier signing module."""
from __future__ import annotations

import pytest

from carl_core.eml import EMLNode, EMLOp, EMLTree
from carl_core.signing import (
    MIN_SECRET_LEN,
    SIG_LEN,
    sign_platform_countersig,
    sign_tree_software,
    verify_platform_countersig,
    verify_software_signature,
)


def _tree() -> EMLTree:
    """A small deterministic tree for tests."""
    root = EMLNode(op=EMLOp.EML,
                   left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
                   right=EMLNode(op=EMLOp.CONST, const=1.0))
    return EMLTree(root=root, input_dim=1)


class TestSoftwareSignature:
    def test_signature_is_32_bytes(self) -> None:
        tree_bytes = _tree().to_bytes()
        secret = b"x" * MIN_SECRET_LEN
        sig = sign_tree_software(tree_bytes, secret)
        assert len(sig) == SIG_LEN == 32

    def test_verify_roundtrip(self) -> None:
        tree_bytes = _tree().to_bytes()
        secret = b"a secret with enough bytes!!!"
        sig = sign_tree_software(tree_bytes, secret)
        assert verify_software_signature(tree_bytes, sig, secret) is True

    def test_verify_rejects_wrong_secret(self) -> None:
        tree_bytes = _tree().to_bytes()
        secret_a = b"a secret with enough bytes!!!"
        secret_b = b"different secret with enough!!"
        sig = sign_tree_software(tree_bytes, secret_a)
        assert verify_software_signature(tree_bytes, sig, secret_b) is False

    def test_verify_rejects_tampered_tree(self) -> None:
        tree_bytes = _tree().to_bytes()
        secret = b"a secret with enough bytes!!!"
        sig = sign_tree_software(tree_bytes, secret)
        tampered = bytearray(tree_bytes)
        tampered[-1] ^= 0x01
        assert (
            verify_software_signature(bytes(tampered), sig, secret) is False
        )

    def test_verify_rejects_wrong_length_sig(self) -> None:
        tree_bytes = _tree().to_bytes()
        secret = b"a secret with enough bytes!!!"
        assert verify_software_signature(tree_bytes, b"\x00" * 16, secret) is False
        assert verify_software_signature(tree_bytes, b"\x00" * 64, secret) is False

    def test_signing_rejects_short_secret(self) -> None:
        from carl_core.errors import ValidationError

        tree_bytes = _tree().to_bytes()
        with pytest.raises(ValidationError) as exc:
            sign_tree_software(tree_bytes, b"too short")
        assert exc.value.code == "carl.eml.domain_error"

    def test_verify_accepts_bytearray_secret(self) -> None:
        tree_bytes = _tree().to_bytes()
        secret = bytearray(b"a secret with enough bytes!!!")
        sig = sign_tree_software(tree_bytes, secret)
        assert verify_software_signature(tree_bytes, sig, secret) is True

    def test_deterministic(self) -> None:
        tree_bytes = _tree().to_bytes()
        secret = b"a secret with enough bytes!!!"
        sigs = [sign_tree_software(tree_bytes, secret) for _ in range(5)]
        assert all(s == sigs[0] for s in sigs)


class TestPlatformCountersig:
    def test_roundtrip(self) -> None:
        platform_secret = b"platform-secret-with-enough-bytes!"
        sig = sign_platform_countersig(
            content_hash_hex="deadbeef" * 8,
            purchase_tx_id="tx_abc123",
            buyer_user_id="user_42",
            timestamp_ns=1_700_000_000_000_000_000,
            platform_secret=platform_secret,
        )
        assert len(sig) == SIG_LEN
        assert verify_platform_countersig(
            content_hash_hex="deadbeef" * 8,
            purchase_tx_id="tx_abc123",
            buyer_user_id="user_42",
            timestamp_ns=1_700_000_000_000_000_000,
            platform_secret=platform_secret,
            sig=sig,
        ) is True

    def test_rejects_wrong_buyer(self) -> None:
        secret = b"platform-secret-with-enough-bytes!"
        sig = sign_platform_countersig(
            content_hash_hex="ab" * 32,
            purchase_tx_id="tx_1",
            buyer_user_id="alice",
            timestamp_ns=0,
            platform_secret=secret,
        )
        assert verify_platform_countersig(
            content_hash_hex="ab" * 32,
            purchase_tx_id="tx_1",
            buyer_user_id="bob",
            timestamp_ns=0,
            platform_secret=secret,
            sig=sig,
        ) is False

    def test_rejects_tampered_timestamp(self) -> None:
        secret = b"platform-secret-with-enough-bytes!"
        sig = sign_platform_countersig(
            content_hash_hex="ab" * 32,
            purchase_tx_id="tx_1",
            buyer_user_id="alice",
            timestamp_ns=100,
            platform_secret=secret,
        )
        assert verify_platform_countersig(
            content_hash_hex="ab" * 32,
            purchase_tx_id="tx_1",
            buyer_user_id="alice",
            timestamp_ns=101,
            platform_secret=secret,
            sig=sig,
        ) is False
