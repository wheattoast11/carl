"""Tests for the local Resonant storage + user_secret layer."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from carl_core.eml import EMLNode, EMLOp, EMLTree
from carl_core.resonant import make_resonant
from carl_core.signing import verify_software_signature


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``~`` at a temp dir so every test writes into a clean sandbox."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Force re-import so the module-level paths re-resolve against new HOME.
    import importlib

    import carl_studio.resonant_store as mod

    importlib.reload(mod)
    return tmp_path


def _tree(depth: int = 2) -> EMLTree:
    """Build a small deterministic tree for the tests."""
    if depth <= 0:
        leaf = EMLNode(op=EMLOp.CONST, const=1.0)
        return EMLTree(root=leaf, input_dim=1)
    # depth 2: eml( var0, const(1) )
    root = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
        right=EMLNode(op=EMLOp.CONST, const=1.0),
    )
    return EMLTree(root=root, input_dim=1)


def _resonant(obs_dim: int = 3, latent: int = 2, action: int = 2):  # noqa: ANN202
    tree = _tree()
    projection = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])[:latent, :obs_dim]
    readout = np.eye(action, latent)
    return make_resonant(tree, projection, readout, metadata={"kind": "test"})


class TestUserSecret:
    def test_create_on_first_read(self, tmp_path: Path) -> None:
        from carl_studio.resonant_store import USER_SECRET_PATH, read_or_create_user_secret

        assert not USER_SECRET_PATH.exists()
        secret = read_or_create_user_secret()
        assert len(secret) == 32
        assert USER_SECRET_PATH.exists()

    def test_mode_is_0600(self) -> None:
        from carl_studio.resonant_store import USER_SECRET_PATH, read_or_create_user_secret

        read_or_create_user_secret()
        mode = stat.S_IMODE(os.stat(USER_SECRET_PATH).st_mode)
        assert mode == 0o600

    def test_reread_is_stable(self) -> None:
        from carl_studio.resonant_store import read_or_create_user_secret

        a = read_or_create_user_secret()
        b = read_or_create_user_secret()
        assert a == b

    def test_rejects_truncated_secret(self) -> None:
        from carl_studio.resonant_store import USER_SECRET_PATH, read_or_create_user_secret
        from carl_core.errors import ValidationError

        USER_SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
        USER_SECRET_PATH.write_bytes(b"too short")
        with pytest.raises(ValidationError) as exc:
            read_or_create_user_secret()
        assert exc.value.code == "carl.credential.invalid"


class TestIdentityFingerprint:
    def test_length_32_hex_chars(self) -> None:
        from carl_studio.resonant_store import identity_fingerprint

        fp = identity_fingerprint(b"x" * 32)
        assert len(fp) == 32
        assert all(c in "0123456789abcdef" for c in fp)

    def test_stable_for_same_secret(self) -> None:
        from carl_studio.resonant_store import identity_fingerprint

        s = b"x" * 32
        assert identity_fingerprint(s) == identity_fingerprint(s)

    def test_differs_for_different_secrets(self) -> None:
        from carl_studio.resonant_store import identity_fingerprint

        assert identity_fingerprint(b"x" * 32) != identity_fingerprint(b"y" * 32)

    def test_uses_disk_secret_when_none_passed(self) -> None:
        from carl_studio.resonant_store import identity_fingerprint, read_or_create_user_secret

        secret = read_or_create_user_secret()
        assert identity_fingerprint() == identity_fingerprint(secret)


class TestEnvelopeRoundtrip:
    def test_encode_decode_unsigned(self) -> None:
        from carl_studio.resonant_store import decode_envelope, encode_envelope

        inner = _tree().to_bytes()
        env = encode_envelope(inner)
        got_inner, sig = decode_envelope(env)
        assert got_inner == inner
        assert sig is None

    def test_encode_decode_signed(self) -> None:
        from carl_studio.resonant_store import decode_envelope, encode_envelope

        inner = _tree().to_bytes()
        sig = b"\x01" * 32
        env = encode_envelope(inner, sig)
        got_inner, got_sig = decode_envelope(env)
        assert got_inner == inner
        assert got_sig == sig

    def test_bad_magic_rejected(self) -> None:
        from carl_core.errors import ValidationError

        from carl_studio.resonant_store import decode_envelope

        with pytest.raises(ValidationError) as exc:
            decode_envelope(b"XXXX\x01whatever")
        assert exc.value.code == "carl.eml.decode_error"

    def test_sig_length_enforced_on_encode(self) -> None:
        from carl_core.errors import ValidationError

        from carl_studio.resonant_store import encode_envelope

        inner = _tree().to_bytes()
        with pytest.raises(ValidationError) as exc:
            encode_envelope(inner, sig=b"\x00" * 16)
        assert exc.value.code == "carl.eml.domain_error"


class TestSaveLoadResonant:
    def test_roundtrip_preserves_structure_and_matrices(self) -> None:
        from carl_studio.resonant_store import load_resonant, save_resonant

        r = _resonant()
        path = save_resonant("r1", r)
        assert path.is_dir()
        for stem in ("tree.emlt", "projection.npy", "readout.npy", "metadata.json"):
            assert (path / stem).is_file(), f"missing {stem}"

        r2, envelope, meta = load_resonant("r1")
        np.testing.assert_array_equal(r.projection, r2.projection)
        np.testing.assert_array_equal(r.readout, r2.readout)
        assert r.tree.hash() == r2.tree.hash()
        assert meta["name"] == "r1"
        assert meta["tree_hash"] == r.tree.hash()
        assert meta["depth"] == r.tree.depth()
        assert isinstance(meta["sig_public_component"], str)
        assert len(meta["sig_public_component"]) == 32

    def test_signature_verifies(self) -> None:
        from carl_studio.resonant_store import (
            decode_envelope,
            load_resonant,
            read_or_create_user_secret,
            save_resonant,
        )

        r = _resonant()
        save_resonant("r_sig", r)
        secret = read_or_create_user_secret()

        _loaded, envelope, _meta = load_resonant("r_sig")
        inner, sig = decode_envelope(envelope)
        assert sig is not None
        assert verify_software_signature(inner, sig, secret) is True

    def test_invalid_name_rejected(self) -> None:
        from carl_core.errors import ValidationError

        from carl_studio.resonant_store import resonant_path

        with pytest.raises(ValidationError) as exc:
            resonant_path("../evil")
        assert exc.value.code == "carl.validation.name_invalid"

    def test_load_missing_raises(self) -> None:
        from carl_core.errors import ValidationError

        from carl_studio.resonant_store import load_resonant

        with pytest.raises(ValidationError) as exc:
            load_resonant("not_there")
        assert exc.value.code == "carl.validation.not_found"


class TestListAndDelete:
    def test_empty_list(self) -> None:
        from carl_studio.resonant_store import list_resonants

        assert list_resonants() == []

    def test_list_returns_saved(self) -> None:
        from carl_studio.resonant_store import list_resonants, save_resonant

        save_resonant("alpha", _resonant())
        save_resonant("beta", _resonant())
        entries = list_resonants()
        names = [e["name"] for e in entries]
        assert names == sorted(names)
        assert {"alpha", "beta"}.issubset(names)
        for e in entries:
            assert e["depth"] in (0, 1, 2, 3, 4)
            assert isinstance(e["tree_hash"], str)

    def test_delete_removes_dir(self) -> None:
        from carl_studio.resonant_store import (
            RESONANTS_DIR,
            delete_resonant,
            list_resonants,
            save_resonant,
        )

        save_resonant("to_delete", _resonant())
        assert (RESONANTS_DIR / "to_delete").is_dir()
        assert delete_resonant("to_delete") is True
        assert not (RESONANTS_DIR / "to_delete").exists()
        assert "to_delete" not in [e["name"] for e in list_resonants()]

    def test_delete_missing_returns_false(self) -> None:
        from carl_studio.resonant_store import delete_resonant

        assert delete_resonant("never_existed") is False
