"""Tests for the encrypted wallet store."""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from carl_studio.wallet_store import (
    ENVELOPE_VERSION,
    KEYRING_ACCOUNT,
    KEYRING_SERVICE,
    SALT_LENGTH,
    WalletCorrupted,
    WalletLocked,
    WalletStore,
    keyring_available,
    set_keyring_master_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip env passphrase and stub keyring to empty for isolation."""
    monkeypatch.delenv("CARL_WALLET_PASSPHRASE", raising=False)
    monkeypatch.setattr(
        "carl_studio.wallet_store._keyring_get_password",
        lambda service, account: None,
    )
    monkeypatch.setattr(
        "carl_studio.wallet_store._keyring_set_password",
        lambda service, account, value: False,
    )
    yield


@pytest.fixture
def tmp_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    return home


# ---------------------------------------------------------------------------
# Unlock + roundtrip
# ---------------------------------------------------------------------------


class TestPassphraseUnlock:
    def test_unlock_with_correct_passphrase_roundtrip(
        self, tmp_home: Path, clean_env: None
    ) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(passphrase="correct-horse-battery")
        assert not store.is_locked
        store.put("private_key", "0xDEADBEEF")
        store.put("mnemonic", "one two three four")
        assert store.get("private_key") == "0xDEADBEEF"
        assert store.get("mnemonic") == "one two three four"

        # Re-open and decrypt with the same passphrase.
        reopened = WalletStore(home=tmp_home)
        reopened.unlock(passphrase="correct-horse-battery")
        assert reopened.get("private_key") == "0xDEADBEEF"
        assert reopened.get("mnemonic") == "one two three four"

    def test_unlock_with_wrong_passphrase_raises_locked(
        self, tmp_home: Path, clean_env: None
    ) -> None:
        first = WalletStore(home=tmp_home)
        first.unlock(passphrase="right-passphrase")
        first.put("private_key", "0xSECRET")

        other = WalletStore(home=tmp_home)
        with pytest.raises(WalletLocked):
            other.unlock(passphrase="wrong-passphrase")

    def test_no_passphrase_and_no_keyring_raises(
        self, tmp_home: Path, clean_env: None
    ) -> None:
        store = WalletStore(home=tmp_home)
        with pytest.raises(WalletLocked):
            store.unlock()

    def test_env_passphrase_is_picked_up(
        self, tmp_home: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CARL_WALLET_PASSPHRASE", "env-pass-1234")
        store = WalletStore(home=tmp_home)
        store.unlock()
        assert not store.is_locked
        assert store.backend == "fernet"

    def test_short_passphrase_rejected(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        with pytest.raises(Exception) as ei:
            store.unlock(passphrase="short")
        assert "at least 8" in str(ei.value) or "Passphrase" in str(ei.value)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


class TestRotate:
    def test_rotate_passphrase(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(passphrase="old-passphrase-123")
        store.put("private_key", "0xABC")

        store.rotate_passphrase("old-passphrase-123", "new-passphrase-456")

        # New passphrase works on a fresh instance.
        after = WalletStore(home=tmp_home)
        after.unlock(passphrase="new-passphrase-456")
        assert after.get("private_key") == "0xABC"

        # Old passphrase no longer works.
        old = WalletStore(home=tmp_home)
        with pytest.raises(WalletLocked):
            old.unlock(passphrase="old-passphrase-123")

    def test_rotate_wrong_old_raises(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(passphrase="real-old-pass-123")
        store.put("k", "v")
        with pytest.raises(WalletLocked):
            store.rotate_passphrase("not-the-old-pass", "new-secure-pass")

    def test_rotate_requires_min_length(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(passphrase="old-passphrase-123")
        with pytest.raises(Exception):
            store.rotate_passphrase("old-passphrase-123", "bad")


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------


class TestCorruption:
    def test_corrupted_ciphertext_raises(
        self, tmp_home: Path, clean_env: None
    ) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(passphrase="pp-pp-pp-pp")
        store.put("private_key", "0xVAL")

        # Overwrite ct bytes with garbage but keep envelope parseable.
        envelope = json.loads(store.path.read_text(encoding="utf-8"))
        envelope["ct"] = base64.urlsafe_b64encode(b"not-real-ciphertext").decode(
            "ascii"
        )
        store.path.write_text(json.dumps(envelope), encoding="utf-8")

        other = WalletStore(home=tmp_home)
        with pytest.raises(WalletLocked):
            # InvalidToken is mapped to WalletLocked (wrong-passphrase-like).
            other.unlock(passphrase="pp-pp-pp-pp")

    def test_unparseable_envelope_raises(
        self, tmp_home: Path, clean_env: None
    ) -> None:
        enc_path = tmp_home / "wallet.enc"
        enc_path.write_text("{not valid json", encoding="utf-8")
        store = WalletStore(home=tmp_home)
        with pytest.raises(WalletCorrupted):
            store.unlock(passphrase="any-8chars")

    def test_unsupported_version_raises(
        self, tmp_home: Path, clean_env: None
    ) -> None:
        enc_path = tmp_home / "wallet.enc"
        enc_path.write_text(
            json.dumps({"v": 99, "salt": "AAAAAAAAAAAAAAAAAAAAAA==", "ct": "zzz"}),
            encoding="utf-8",
        )
        store = WalletStore(home=tmp_home)
        with pytest.raises(WalletCorrupted):
            store.unlock(passphrase="any-8chars")

    def test_envelope_missing_fields(self, tmp_home: Path, clean_env: None) -> None:
        enc_path = tmp_home / "wallet.enc"
        enc_path.write_text(
            json.dumps({"v": ENVELOPE_VERSION, "salt": "AAA="}),
            encoding="utf-8",
        )
        store = WalletStore(home=tmp_home)
        with pytest.raises(WalletCorrupted):
            store.unlock(passphrase="any-8chars")


# ---------------------------------------------------------------------------
# Keyring fallback
# ---------------------------------------------------------------------------


class TestKeyringFallback:
    def test_keyring_provides_master_key(
        self, tmp_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CARL_WALLET_PASSPHRASE", raising=False)

        # Seed: write an envelope under the keyring-provided passphrase.
        seed = WalletStore(home=tmp_home)
        seed.unlock(passphrase="keyring-master-8chars")
        seed.put("private_key", "0xKR")

        calls: list[tuple[str, str]] = []

        def fake_get(service: str, account: str) -> str | None:
            calls.append((service, account))
            if service == KEYRING_SERVICE and account == KEYRING_ACCOUNT:
                return "keyring-master-8chars"
            return None

        monkeypatch.setattr("carl_studio.wallet_store._keyring_get_password", fake_get)

        store = WalletStore(home=tmp_home)
        store.unlock()  # no passphrase — should fall through to keyring
        assert store.backend == "keyring"
        assert store.get("private_key") == "0xKR"
        assert (KEYRING_SERVICE, KEYRING_ACCOUNT) in calls

    def test_keyring_missing_falls_through_to_locked(
        self, tmp_home: Path, clean_env: None
    ) -> None:
        store = WalletStore(home=tmp_home)
        with pytest.raises(WalletLocked):
            store.unlock()

    def test_keyring_available_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "carl_studio.wallet_store._keyring_get_password",
            lambda s, a: "x" * 10,
        )
        assert keyring_available() is True

        monkeypatch.setattr(
            "carl_studio.wallet_store._keyring_get_password",
            lambda s, a: None,
        )
        assert keyring_available() is False

    def test_set_keyring_master_key_uses_keyring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, str] = {}

        def fake_set(service: str, account: str, value: str) -> bool:
            captured["svc"] = service
            captured["acct"] = account
            captured["val"] = value
            return True

        monkeypatch.setattr("carl_studio.wallet_store._keyring_set_password", fake_set)
        ok = set_keyring_master_key("super-secure-12")
        assert ok is True
        assert captured == {
            "svc": KEYRING_SERVICE,
            "acct": KEYRING_ACCOUNT,
            "val": "super-secure-12",
        }

    def test_set_keyring_master_key_rejects_short(self) -> None:
        with pytest.raises(Exception):
            set_keyring_master_key("short")


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------


class TestFileMode:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX perms only")
    def test_enc_file_is_0600(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(passphrase="pp-pp-pp-pp")
        store.put("k", "v")
        mode = stat.S_IMODE(os.stat(store.path).st_mode)
        assert mode == 0o600

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX perms only")
    def test_plaintext_file_is_0600(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(allow_plaintext=True)
        store.put("address", "0xPT")
        plain_path = tmp_home / "wallet.json"
        assert plain_path.exists()
        mode = stat.S_IMODE(os.stat(plain_path).st_mode)
        assert mode == 0o600


# ---------------------------------------------------------------------------
# Plaintext fallback policy
# ---------------------------------------------------------------------------


class TestPlaintextFallback:
    def test_plaintext_requires_opt_in(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        with pytest.raises(WalletLocked):
            # allow_plaintext defaults to False.
            store.unlock()

    def test_plaintext_unlocks_and_saves(
        self, tmp_home: Path, clean_env: None
    ) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(allow_plaintext=True)
        store.put("address", "0xABC")
        assert store.backend == "plaintext"
        raw = json.loads((tmp_home / "wallet.json").read_text(encoding="utf-8"))
        assert raw == {"address": "0xABC"}

    def test_plaintext_blocked_when_encrypted_exists(
        self, tmp_home: Path, clean_env: None
    ) -> None:
        # Seed an encrypted envelope.
        enc = WalletStore(home=tmp_home)
        enc.unlock(passphrase="pp-pp-pp-pp")
        enc.put("x", "y")

        # Plaintext fallback must NOT silently downgrade.
        store = WalletStore(home=tmp_home)
        with pytest.raises(WalletLocked):
            store.unlock(allow_plaintext=True)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


class TestAPI:
    def test_get_before_unlock_raises(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        with pytest.raises(WalletLocked):
            store.get("anything")

    def test_put_validates_inputs(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(passphrase="pp-pp-pp-pp")
        with pytest.raises(Exception):
            store.put("", "v")
        with pytest.raises(Exception):
            store.put("k", 123)  # type: ignore[arg-type]

    def test_keys_listing(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(passphrase="pp-pp-pp-pp")
        store.put("b", "2")
        store.put("a", "1")
        assert store.keys() == ["a", "b"]

    def test_delete_removes_key(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(passphrase="pp-pp-pp-pp")
        store.put("a", "1")
        assert store.delete("a") is True
        assert store.get("a") is None
        assert store.delete("a") is False


# ---------------------------------------------------------------------------
# Salt uniqueness
# ---------------------------------------------------------------------------


class TestSaltEntropy:
    def test_fresh_stores_use_unique_salts(
        self, tmp_path: Path, clean_env: None
    ) -> None:
        salts: set[str] = set()
        for i in range(5):
            home = tmp_path / f"home-{i}"
            home.mkdir()
            s = WalletStore(home=home)
            s.unlock(passphrase=f"pp-pp-pp-pp-{i}")
            s.put("k", "v")
            envelope = json.loads(s.path.read_text(encoding="utf-8"))
            salts.add(envelope["salt"])
            salt_bytes = base64.urlsafe_b64decode(envelope["salt"].encode("ascii"))
            assert len(salt_bytes) == SALT_LENGTH
        assert len(salts) == 5

    def test_rotate_changes_salt(self, tmp_home: Path, clean_env: None) -> None:
        store = WalletStore(home=tmp_home)
        store.unlock(passphrase="pp-pp-pp-pp")
        store.put("k", "v")
        old_salt = json.loads(store.path.read_text(encoding="utf-8"))["salt"]
        store.rotate_passphrase("pp-pp-pp-pp", "pp-pp-pp-qq")
        new_salt = json.loads(store.path.read_text(encoding="utf-8"))["salt"]
        assert old_salt != new_salt
