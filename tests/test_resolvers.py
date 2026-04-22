"""Tests for the public resolvers (env / keyring / fernet-file).

Covers:
- URI parsing (scheme match + body extraction)
- Each resolver's happy path + not-found + backend-unavailable branches
- Integration with Vault.register_resolver + Vault.resolve
- Fernet write + read round-trip
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

from carl_core.secrets import SecretRef, SecretVault
from carl_core.vault import VaultError

from carl_studio.handles.resolvers import (
    EnvResolver,
    FernetFileResolver,
    KeyringResolver,
    ResolverError,
    parse_uri,
)


# ---------------------------------------------------------------------------
# parse_uri
# ---------------------------------------------------------------------------


def test_parse_uri_happy_path() -> None:
    assert parse_uri("env://GITHUB_TOKEN", "env") == "GITHUB_TOKEN"
    assert parse_uri("keyring://gmail/tej", "keyring") == "gmail/tej"
    assert parse_uri("fernet-file://vault/key", "fernet-file") == "vault/key"


def test_parse_uri_scheme_mismatch() -> None:
    with pytest.raises(ResolverError) as exc:
        parse_uri("env://X", "keyring")
    assert exc.value.code == "carl.resolver.keyring.scheme_mismatch"


# ---------------------------------------------------------------------------
# EnvResolver
# ---------------------------------------------------------------------------


def _fresh_ref(uri: str, kind: str = "env") -> SecretRef:
    return SecretRef(kind=cast(Any, kind), uri=uri)


def test_env_resolver_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARL_TEST_VAR_123", "hello")
    ref = _fresh_ref("env://CARL_TEST_VAR_123")
    got = EnvResolver()(ref)
    assert got == b"hello"


def test_env_resolver_missing_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARL_TEST_MISSING_42", raising=False)
    ref = _fresh_ref("env://CARL_TEST_MISSING_42")
    with pytest.raises(ResolverError) as exc:
        EnvResolver()(ref)
    assert exc.value.code == "carl.resolver.env.not_found"


def test_env_resolver_empty_var_returns_empty_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARL_TEST_EMPTY", "")
    ref = _fresh_ref("env://CARL_TEST_EMPTY")
    assert EnvResolver()(ref) == b""


def test_env_resolver_via_vault_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARL_TEST_CHAIN_VAL", "through-vault")
    vault = SecretVault()
    vault.register_resolver("env", EnvResolver())
    ref = SecretRef(kind="env", uri="env://CARL_TEST_CHAIN_VAL")
    vault.put_ref_only(ref)
    # Privileged required on SecretVault.
    got = vault.resolve(ref, privileged=True)
    assert got == b"through-vault"


# ---------------------------------------------------------------------------
# KeyringResolver — fake the `keyring` module via sys.modules
# ---------------------------------------------------------------------------


class _FakeKeyring:
    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.entries.get((service, account))


@pytest.fixture
def fake_keyring() -> Any:  # pyright: ignore[reportUnusedFunction]
    fk = _FakeKeyring()
    mod = types.ModuleType("keyring")
    mod.get_password = fk.get_password  # type: ignore[attr-defined]
    sys.modules["keyring"] = mod
    yield fk
    sys.modules.pop("keyring", None)


def test_keyring_resolver_happy_path(fake_keyring: _FakeKeyring) -> None:
    fake_keyring.entries[("gmail", "tej@x.tech")] = "sup3r-secret"
    ref = _fresh_ref("keyring://gmail/tej@x.tech", kind="keychain")
    got = KeyringResolver()(ref)
    assert got == b"sup3r-secret"


def test_keyring_resolver_missing_entry(fake_keyring: _FakeKeyring) -> None:
    ref = _fresh_ref("keyring://ghost-service/ghost", kind="keychain")
    with pytest.raises(ResolverError) as exc:
        KeyringResolver()(ref)
    assert exc.value.code == "carl.resolver.keyring.not_found"


def test_keyring_resolver_bad_uri(fake_keyring: _FakeKeyring) -> None:
    ref = _fresh_ref("keyring://no-slash", kind="keychain")
    with pytest.raises(ResolverError) as exc:
        KeyringResolver()(ref)
    assert exc.value.code == "carl.resolver.keyring.bad_uri"


def test_keyring_resolver_backend_unavailable() -> None:
    # No fake_keyring fixture — sys.modules has no 'keyring' entry
    sys.modules.pop("keyring", None)
    ref = _fresh_ref("keyring://svc/user", kind="keychain")
    # When the real keyring is installed in the dev env, the import succeeds.
    # Only the not_found / bad_uri paths are exercised in that case.
    import importlib.util

    if importlib.util.find_spec("keyring") is not None:
        pytest.skip("real keyring installed; backend_unavailable can't be asserted")
    with pytest.raises(ResolverError) as exc:
        KeyringResolver()(ref)
    assert exc.value.code == "carl.resolver.keyring.backend_unavailable"


# ---------------------------------------------------------------------------
# FernetFileResolver
# ---------------------------------------------------------------------------


def _skip_if_no_cryptography() -> None:
    import importlib.util

    if importlib.util.find_spec("cryptography.fernet") is None:
        pytest.skip("cryptography not installed; fernet-file tests skipped")


def test_fernet_file_write_then_read_roundtrip(tmp_path: Path) -> None:
    _skip_if_no_cryptography()
    resolver = FernetFileResolver(
        key_path=tmp_path / "vault.key",
        vault_dir=tmp_path / "vault",
    )
    resolver.write("github_token", b"ghp_example_payload")
    ref = _fresh_ref("fernet-file://github_token", kind="vault")
    got = resolver(ref)
    assert got == b"ghp_example_payload"


def test_fernet_file_missing(tmp_path: Path) -> None:
    _skip_if_no_cryptography()
    resolver = FernetFileResolver(
        key_path=tmp_path / "vault.key",
        vault_dir=tmp_path / "vault",
    )
    ref = _fresh_ref("fernet-file://missing", kind="vault")
    with pytest.raises(ResolverError) as exc:
        resolver(ref)
    assert exc.value.code == "carl.resolver.fernet_file.not_found"


def test_fernet_file_wrong_key_raises_decrypt_failed(tmp_path: Path) -> None:
    _skip_if_no_cryptography()
    # Write with one key
    r1 = FernetFileResolver(
        key_path=tmp_path / "key1.key",
        vault_dir=tmp_path / "vault",
    )
    r1.write("entry", b"hello")
    # Read with a different key — should fail
    r2 = FernetFileResolver(
        key_path=tmp_path / "key2.key",  # will auto-create a different key
        vault_dir=tmp_path / "vault",
    )
    ref = _fresh_ref("fernet-file://entry", kind="vault")
    with pytest.raises(ResolverError) as exc:
        r2(ref)
    assert exc.value.code == "carl.resolver.fernet_file.decrypt_failed"


def test_fernet_file_key_is_0600(tmp_path: Path) -> None:
    _skip_if_no_cryptography()
    resolver = FernetFileResolver(
        key_path=tmp_path / "vault.key",
        vault_dir=tmp_path / "vault",
    )
    resolver.write("x", b"y")
    mode = (tmp_path / "vault.key").stat().st_mode & 0o777
    assert mode == 0o600


def test_fernet_file_resolver_via_vault_register(tmp_path: Path) -> None:
    _skip_if_no_cryptography()
    resolver = FernetFileResolver(
        key_path=tmp_path / "vault.key",
        vault_dir=tmp_path / "vault",
    )
    resolver.write("github_token", b"ghp_example")
    vault = SecretVault()
    vault.register_resolver("vault", resolver)
    ref = SecretRef(kind="vault", uri="fernet-file://github_token")
    vault.put_ref_only(ref)
    got = vault.resolve(ref, privileged=True)
    assert got == b"ghp_example"


# ---------------------------------------------------------------------------
# End-to-end: resolver chain within SecretVault.resolve() wraps errors
# ---------------------------------------------------------------------------


def test_resolver_failure_wraps_as_vault_resolver_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CARL_NONEXISTENT_VAR_E2E", raising=False)
    vault = SecretVault()
    vault.register_resolver("env", EnvResolver())
    ref = SecretRef(kind="env", uri="env://CARL_NONEXISTENT_VAR_E2E")
    vault.put_ref_only(ref)
    # EnvResolver raises ResolverError; vault.resolve wraps via resolver_failed.
    # The ResolverError is NOT a VaultError subclass, so wrapping kicks in.
    with pytest.raises(VaultError) as exc:
        vault.resolve(ref, privileged=True)
    # The outer code is carl.secrets.resolver_failed (vault's prefix)
    assert exc.value.code == "carl.secrets.resolver_failed"
    # And the cause preserves the original ResolverError
    assert isinstance(exc.value.__cause__, ResolverError)
    orig = exc.value.__cause__
    assert orig.code == "carl.resolver.env.not_found"


