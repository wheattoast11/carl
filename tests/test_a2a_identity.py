"""Tests for :mod:`carl_studio.a2a.identity`.

Covers ES256 keypair generation, JWS sign/verify round-trip, tamper
detection, algorithm / kid mismatch, missing-key load fallback.

Every test uses ``tmp_path`` as an isolated keystore; nothing touches
the real ``~/.carl/keys`` directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carl_core.connection import ConnectionAuthError

from carl_studio.a2a import AgentIdentity, CARLAgentCard


@pytest.fixture(autouse=True)
def _isolate_keyring(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never touch the real OS keyring during tests.

    The identity module tries the keyring as a fallback; any leftover key
    from a previous test run or a developer's real keyring would leak
    across tests, breaking isolation assumptions. We replace both helpers
    with no-ops so the only key material in play is the filesystem copy
    under ``tmp_path``.
    """
    from carl_studio.a2a import identity as identity_mod

    def _no_keyring_load() -> None:
        return None

    def _no_keyring_save(_pem: bytes) -> None:
        return None

    monkeypatch.setattr(identity_mod, "_try_keyring_load", _no_keyring_load)
    monkeypatch.setattr(identity_mod, "_try_keyring_save", _no_keyring_save)


# ---------------------------------------------------------------------------
# Key generation & persistence.
# ---------------------------------------------------------------------------


class TestKeyLifecycle:
    def test_load_generates_keypair_when_missing(self, tmp_path: Path) -> None:
        identity = AgentIdentity.load(keystore_dir=tmp_path)
        assert identity.private_key_pem is not None
        assert identity.public_key_pem is not None
        assert (tmp_path / "agent.key").exists()
        assert (tmp_path / "agent.pub").exists()

    def test_private_key_has_600_perms_on_posix(self, tmp_path: Path) -> None:
        import os
        import sys

        AgentIdentity.load(keystore_dir=tmp_path)
        priv = tmp_path / "agent.key"
        if sys.platform == "win32":
            pytest.skip("Windows does not honor POSIX 0600 perms")
        mode = os.stat(priv).st_mode & 0o777
        assert mode == 0o600

    def test_load_reuses_existing_keypair(self, tmp_path: Path) -> None:
        a = AgentIdentity.load(keystore_dir=tmp_path)
        b = AgentIdentity.load(keystore_dir=tmp_path)
        # Byte-identical PEM => same keypair on disk.
        assert a.private_key_pem == b.private_key_pem
        assert a.public_key_pem == b.public_key_pem

    def test_load_without_create_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ConnectionAuthError):
            AgentIdentity.load(keystore_dir=tmp_path, create_if_missing=False)

    def test_rejects_unsupported_algorithm(self) -> None:
        with pytest.raises(ValueError):
            AgentIdentity(
                private_key_pem=b"fake",
                public_key_pem=b"fake",
                algorithm="RS256",
            )

    def test_rejects_empty_key_material(self) -> None:
        with pytest.raises(ValueError):
            AgentIdentity()


# ---------------------------------------------------------------------------
# Sign / verify round-trip.
# ---------------------------------------------------------------------------


class TestSignVerify:
    def test_round_trip_valid_signature(self, tmp_path: Path) -> None:
        identity = AgentIdentity.load(keystore_dir=tmp_path)
        card = CARLAgentCard(
            name="carl-test", version="0.4.2", skills=["train", "eval"]
        )
        token = identity.sign_card(card)
        # Compact JWS has three dot-separated parts.
        assert token.count(".") == 2

        # Verify with the same identity.
        parsed = identity.verify_card(token)
        assert isinstance(parsed, CARLAgentCard)
        assert parsed.name == "carl-test"
        assert parsed.version == "0.4.2"
        assert parsed.skills == ["train", "eval"]

    def test_verify_with_public_key_only(self, tmp_path: Path) -> None:
        signer = AgentIdentity.load(keystore_dir=tmp_path)
        # Counterparty knows only the public key.
        verifier = AgentIdentity(public_key_pem=signer.public_key_bytes())
        card = CARLAgentCard(name="x", version="1")
        token = signer.sign_card(card)
        parsed = verifier.verify_card(token)
        assert parsed.name == "x"

    def test_signature_with_different_key_fails(self, tmp_path: Path) -> None:
        tmp_a = tmp_path / "a"
        tmp_b = tmp_path / "b"
        tmp_a.mkdir()
        tmp_b.mkdir()
        signer = AgentIdentity.load(keystore_dir=tmp_a)
        other = AgentIdentity.load(keystore_dir=tmp_b)
        token = signer.sign_card(CARLAgentCard(name="foo"))
        with pytest.raises(ConnectionAuthError):
            other.verify_card(token)

    def test_tamper_detection(self, tmp_path: Path) -> None:
        identity = AgentIdentity.load(keystore_dir=tmp_path)
        token = identity.sign_card(CARLAgentCard(name="foo"))
        header_b64, payload_b64, sig_b64 = token.split(".")
        import base64

        # Flip a payload byte.
        padding = "=" * (-len(payload_b64) % 4)
        raw = bytearray(base64.urlsafe_b64decode(payload_b64 + padding))
        raw[0] ^= 0x01
        tampered_payload = (
            base64.urlsafe_b64encode(bytes(raw)).rstrip(b"=").decode("ascii")
        )
        tampered = f"{header_b64}.{tampered_payload}.{sig_b64}"
        with pytest.raises(ConnectionAuthError):
            identity.verify_card(tampered)

    def test_malformed_jws_rejected(self, tmp_path: Path) -> None:
        identity = AgentIdentity.load(keystore_dir=tmp_path)
        with pytest.raises(ConnectionAuthError):
            identity.verify_card("not.a.jws.token")
        with pytest.raises(ConnectionAuthError):
            identity.verify_card("nodots")

    def test_alg_mismatch_rejected(self, tmp_path: Path) -> None:
        identity = AgentIdentity.load(keystore_dir=tmp_path)
        token = identity.sign_card(CARLAgentCard(name="foo"))

        # Rewrite the header alg to RS256.
        import base64
        import json

        h_b64, p_b64, s_b64 = token.split(".")
        header = json.loads(
            base64.urlsafe_b64decode(h_b64 + "==").decode("utf-8")
        )
        header["alg"] = "RS256"
        new_header = (
            base64.urlsafe_b64encode(
                json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
            )
            .rstrip(b"=")
            .decode("ascii")
        )
        tampered = f"{new_header}.{p_b64}.{s_b64}"
        with pytest.raises(ConnectionAuthError):
            identity.verify_card(tampered)

    def test_kid_mismatch_rejected(self, tmp_path: Path) -> None:
        identity = AgentIdentity.load(keystore_dir=tmp_path)
        token = identity.sign_card(CARLAgentCard(name="foo"))
        with pytest.raises(ConnectionAuthError):
            identity.verify_card(token, expected_kid="different-kid")

    def test_kid_match_accepted(self, tmp_path: Path) -> None:
        identity = AgentIdentity.load(keystore_dir=tmp_path)
        token = identity.sign_card(CARLAgentCard(name="foo"))
        parsed = identity.verify_card(token, expected_kid=identity.kid)
        assert parsed.name == "foo"


# ---------------------------------------------------------------------------
# Canonicalization.
# ---------------------------------------------------------------------------


class TestCanonicalization:
    def test_signatures_are_deterministic_for_ecdsa_only_on_payload(
        self, tmp_path: Path
    ) -> None:
        """ES256 signatures are non-deterministic (RFC 6979 is not required),
        but the signing input (header+payload canonical bytes) MUST be
        identical across calls for the same card.

        So we verify by re-signing and confirming that both tokens verify
        back to the same card — not by byte-comparing the signatures.
        """
        identity = AgentIdentity.load(keystore_dir=tmp_path)
        card = CARLAgentCard(name="det", version="0.1")
        t1 = identity.sign_card(card)
        t2 = identity.sign_card(card)
        assert identity.verify_card(t1).name == "det"
        assert identity.verify_card(t2).name == "det"

    def test_public_key_bytes_derived_from_private(self, tmp_path: Path) -> None:
        identity = AgentIdentity.load(keystore_dir=tmp_path)
        priv_only = AgentIdentity(
            private_key_pem=identity.private_key_pem,
            algorithm="ES256",
        )
        assert priv_only.public_key_pem is None
        derived = priv_only.public_key_bytes()
        assert derived == identity.public_key_pem


# ---------------------------------------------------------------------------
# Signing requires a private key.
# ---------------------------------------------------------------------------


def test_sign_without_private_key_raises(tmp_path: Path) -> None:
    signer = AgentIdentity.load(keystore_dir=tmp_path)
    verifier = AgentIdentity(public_key_pem=signer.public_key_bytes())
    with pytest.raises(ConnectionAuthError):
        verifier.sign_card(CARLAgentCard(name="x"))
