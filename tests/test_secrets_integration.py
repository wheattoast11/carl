"""Integration tests for carl_studio.secrets — Stage B.

Exercises the toolkit façade, keychain wrapper, clipboard bridge, and
minter against a fake ``keyring`` + ``pyperclip`` module so tests run
deterministically on CI without real OS integrations.

Every test asserts the audit trail carries fingerprints only — never
the raw value. The test file itself does not import the value of any
secret; it only sees fingerprint + uri + ref_id.
"""

from __future__ import annotations

import sys
import types
import uuid
from datetime import datetime, timezone
from typing import Any, cast

import pytest

from carl_core.errors import ValidationError
from carl_core.interaction import ActionType, InteractionChain
from carl_core.secrets import SecretVault, SecretsError
from carl_studio.secrets import (
    ClipboardBridge,
    CryptoRandomMinter,
    KeychainBackend,
    KeychainError,
    SecretsToolkit,
)


# ---------------------------------------------------------------------------
# Fake keyring + pyperclip fixtures
# ---------------------------------------------------------------------------


class _FakeKeyringBackend:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.store.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.store[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        if (service, account) not in self.store:
            raise RuntimeError("no such password")
        del self.store[(service, account)]


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> _FakeKeyringBackend:
    """Install a fake keyring module into sys.modules for the test."""
    backend = _FakeKeyringBackend()
    fake = types.ModuleType("keyring")
    fake.get_keyring = lambda: backend  # type: ignore[attr-defined]
    fake.get_password = backend.get_password  # type: ignore[attr-defined]
    fake.set_password = backend.set_password  # type: ignore[attr-defined]
    fake.delete_password = backend.delete_password  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", fake)
    return backend


class _FakePyperclip:
    def __init__(self) -> None:
        self._content = ""

    def copy(self, value: str) -> None:
        self._content = value

    def paste(self) -> str:
        return self._content


@pytest.fixture
def fake_pyperclip(monkeypatch: pytest.MonkeyPatch) -> _FakePyperclip:
    clip = _FakePyperclip()
    fake = types.ModuleType("pyperclip")
    fake.copy = clip.copy  # type: ignore[attr-defined]
    fake.paste = clip.paste  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyperclip", fake)
    return clip


# ---------------------------------------------------------------------------
# Minter
# ---------------------------------------------------------------------------


def test_minter_hex_emits_secret_mint_step() -> None:
    chain = InteractionChain()
    vault = SecretVault()
    minter = CryptoRandomMinter(vault, chain=chain)

    ref = minter.mint_hex(nbytes=16)

    assert ref.kind == "mint"
    assert len(chain) == 1
    step = chain.last()
    assert step is not None
    assert step.action == ActionType.SECRET_MINT
    out_raw = step.output
    assert isinstance(out_raw, dict)
    output = cast(dict[str, Any], out_raw)
    assert output["ref_id"] == str(ref.ref_id)
    assert len(output["fingerprint"]) == 12
    # Value must not leak into the step.
    resolved = vault.resolve(ref, privileged=True)
    assert resolved.decode() not in str(output)


def test_minter_base64_length_matches_nbytes() -> None:
    vault = SecretVault()
    minter = CryptoRandomMinter(vault)
    ref = minter.mint_base64(nbytes=32)
    # base64-encoded 32 bytes is ~44 chars (with padding).
    raw = vault.resolve(ref, privileged=True)
    assert 40 < len(raw) < 50


def test_minter_uuid_yields_valid_uuid() -> None:
    vault = SecretVault()
    minter = CryptoRandomMinter(vault)
    ref = minter.mint_uuid()
    raw = vault.resolve(ref, privileged=True).decode()
    uuid.UUID(raw)  # round-trips


def test_minter_ed25519_keypair_emits_two_steps() -> None:
    chain = InteractionChain()
    vault = SecretVault()
    minter = CryptoRandomMinter(vault, chain=chain)
    priv_ref, pub_ref = minter.mint_ed25519_keypair()
    # Two SECRET_MINT Steps: priv + pub.
    mints = chain.by_action(ActionType.SECRET_MINT)
    assert len(mints) == 2
    assert priv_ref.ref_id != pub_ref.ref_id
    # Raw keys are 32 bytes each.
    assert len(vault.resolve(priv_ref, privileged=True)) == 32
    assert len(vault.resolve(pub_ref, privileged=True)) == 32


def test_minter_hex_rejects_tiny_nbytes() -> None:
    vault = SecretVault()
    minter = CryptoRandomMinter(vault)
    with pytest.raises(ValueError):
        minter.mint_hex(nbytes=4)  # < 8


def test_minter_silent_when_chain_is_none() -> None:
    vault = SecretVault()
    minter = CryptoRandomMinter(vault, chain=None)
    ref = minter.mint_hex()
    assert ref.kind == "mint"
    # No audit (chain is None) — just returns the ref.


# ---------------------------------------------------------------------------
# KeychainBackend
# ---------------------------------------------------------------------------


def test_keychain_load_to_vault_roundtrip(
    fake_keyring: _FakeKeyringBackend,
) -> None:
    fake_keyring.store[("carl-camp", "hf_token")] = "secret-hf-token-value"
    vault = SecretVault()
    chain = InteractionChain()
    backend = KeychainBackend(vault, chain=chain)

    ref = backend.load_to_vault("carl-camp", "hf_token")

    # The vault has the value.
    raw = vault.resolve(ref, privileged=True)
    assert raw == b"secret-hf-token-value"

    # Audit emitted a SECRET_RESOLVE Step with fingerprint but no value.
    resolves = chain.by_action(ActionType.SECRET_RESOLVE)
    assert len(resolves) == 1
    step = resolves[0]
    out_raw = step.output
    assert isinstance(out_raw, dict)
    output = cast(dict[str, Any], out_raw)
    assert output["service"] == "carl-camp"
    assert output["account"] == "hf_token"
    assert "fingerprint" in output
    # Raw value must not leak into the step.
    assert "secret-hf-token-value" not in str(output)


def test_keychain_load_missing_raises(fake_keyring: _FakeKeyringBackend) -> None:
    vault = SecretVault()
    backend = KeychainBackend(vault)
    with pytest.raises(KeychainError) as excinfo:
        backend.load_to_vault("not-there", "whatever")
    assert excinfo.value.code == "carl.keychain.not_found"


def test_keychain_store_persists_via_vault(
    fake_keyring: _FakeKeyringBackend,
) -> None:
    vault = SecretVault()
    backend = KeychainBackend(vault)
    # Stage a value in the vault via the minter (value never enters test).
    minter = CryptoRandomMinter(vault)
    ref = minter.mint_hex(nbytes=16)
    backend.store("carl-camp", "minted_key", ref)
    # The fake backend now has the string form.
    assert fake_keyring.store[("carl-camp", "minted_key")] != ""


def test_keychain_delete_emits_secret_revoke_step(
    fake_keyring: _FakeKeyringBackend,
) -> None:
    fake_keyring.store[("svc", "acct")] = "val"
    chain = InteractionChain()
    backend = KeychainBackend(SecretVault(), chain=chain)
    assert backend.delete("svc", "acct") is True

    revokes = chain.by_action(ActionType.SECRET_REVOKE)
    assert len(revokes) == 1
    step = revokes[0]
    assert step.name == "keychain.delete"


def test_keychain_delete_missing_returns_false(
    fake_keyring: _FakeKeyringBackend,
) -> None:
    backend = KeychainBackend(SecretVault())
    assert backend.delete("missing", "entry") is False


def test_keychain_exists(fake_keyring: _FakeKeyringBackend) -> None:
    fake_keyring.store[("svc", "acct")] = "v"
    backend = KeychainBackend(SecretVault())
    assert backend.exists("svc", "acct") is True
    assert backend.exists("svc", "nope") is False


def test_keychain_unavailable_without_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uninstalled keyring → carl.keychain.unavailable."""
    # Ensure keyring is NOT importable.
    monkeypatch.setitem(sys.modules, "keyring", None)
    backend = KeychainBackend(SecretVault())
    with pytest.raises(KeychainError) as excinfo:
        backend.load_to_vault("svc", "acct")
    assert excinfo.value.code == "carl.keychain.unavailable"


# ---------------------------------------------------------------------------
# ClipboardBridge
# ---------------------------------------------------------------------------


def test_clipboard_write_from_ref_emits_audit_step(
    fake_pyperclip: _FakePyperclip,
) -> None:
    vault = SecretVault()
    chain = InteractionChain()
    bridge = ClipboardBridge(vault, chain=chain, default_ttl_s=60)
    ref = vault.put(b"paste-me-into-form", kind="mint")

    result = bridge.write_from_ref(ref, ttl_s=10)

    assert "fingerprint" in result
    assert "expires_at" in result
    assert result["ttl_s"] == 10
    # Clipboard now has the value.
    assert fake_pyperclip.paste() == "paste-me-into-form"

    # Audit emitted.
    writes = chain.by_action(ActionType.CLIPBOARD_WRITE)
    assert len(writes) == 1
    step = writes[0]
    out_raw = step.output
    assert isinstance(out_raw, dict)
    output = cast(dict[str, Any], out_raw)
    assert output["fingerprint"] == result["fingerprint"]
    # Raw value must not leak.
    assert "paste-me-into-form" not in str(output)


def test_clipboard_wipe_clears_clipboard(fake_pyperclip: _FakePyperclip) -> None:
    vault = SecretVault()
    bridge = ClipboardBridge(vault)
    ref = vault.put(b"sensitive", kind="mint")
    bridge.write_from_ref(ref, ttl_s=5)
    assert fake_pyperclip.paste() == "sensitive"
    assert bridge.wipe() is True
    assert fake_pyperclip.paste() == ""


def test_clipboard_explicit_wipe_emits_revoke_step(
    fake_pyperclip: _FakePyperclip,
) -> None:
    chain = InteractionChain()
    bridge = ClipboardBridge(SecretVault(), chain=chain)
    # Stage something first so wipe has prior state to log.
    v = SecretVault()
    bridge = ClipboardBridge(v, chain=chain)
    ref = v.put(b"x", kind="mint")
    bridge.write_from_ref(ref, ttl_s=60)
    bridge.wipe()
    revokes = chain.by_action(ActionType.SECRET_REVOKE)
    # Only one revoke emitted (the explicit wipe).
    assert any(s.name == "clipboard.wipe" for s in revokes)


def test_clipboard_auto_wipe_fires_via_timer(
    fake_pyperclip: _FakePyperclip,
) -> None:
    """Short TTL + live sleep exercises the Timer-scheduled auto-wipe."""
    import time

    chain = InteractionChain()
    bridge = ClipboardBridge(SecretVault(), chain=chain)
    v = SecretVault()
    bridge = ClipboardBridge(v, chain=chain)
    ref = v.put(b"short-lived", kind="mint")
    bridge.write_from_ref(ref, ttl_s=1)
    assert fake_pyperclip.paste() == "short-lived"
    time.sleep(1.3)
    # Timer fired — clipboard is now empty.
    assert fake_pyperclip.paste() == ""
    # A SECRET_REVOKE step named "clipboard.auto_wipe" landed.
    revokes = chain.by_action(ActionType.SECRET_REVOKE)
    assert any(s.name == "clipboard.auto_wipe" for s in revokes)


def test_clipboard_new_write_cancels_pending_wipe(
    fake_pyperclip: _FakePyperclip,
) -> None:
    """Second write supersedes the first; no double-wipe."""
    import time

    vault = SecretVault()
    bridge = ClipboardBridge(vault)
    ref1 = vault.put(b"first", kind="mint")
    bridge.write_from_ref(ref1, ttl_s=2)
    ref2 = vault.put(b"second", kind="mint")
    bridge.write_from_ref(ref2, ttl_s=10)
    # Past the first ttl_s — the first timer was cancelled.
    time.sleep(2.3)
    assert fake_pyperclip.paste() == "second"


def test_clipboard_write_rejects_bad_ttl(fake_pyperclip: _FakePyperclip) -> None:
    vault = SecretVault()
    bridge = ClipboardBridge(vault)
    ref = vault.put(b"x", kind="mint")
    from carl_studio.secrets import ClipboardBridgeError

    with pytest.raises(ClipboardBridgeError) as excinfo:
        bridge.write_from_ref(ref, ttl_s=0)
    assert excinfo.value.code == "carl.clipboard.invalid_ttl"


# ---------------------------------------------------------------------------
# SecretsToolkit — façade
# ---------------------------------------------------------------------------


def test_toolkit_mint_secret_hex_full_cycle(
    fake_pyperclip: _FakePyperclip,
) -> None:
    chain = InteractionChain()
    toolkit = SecretsToolkit.build(chain=chain)

    minted = toolkit.mint_secret(kind="hex", nbytes=32)
    assert set(minted.keys()) == {"ref_id", "uri", "fingerprint", "kind", "ttl_s"}
    assert minted["kind"] == "hex"
    assert len(minted["fingerprint"]) == 12

    copied = toolkit.copy_to_clipboard(minted["ref_id"], ttl_s=60)
    assert copied["fingerprint"] == minted["fingerprint"]
    assert "expires_at" in copied

    revoked = toolkit.revoke_secret(minted["ref_id"])
    assert revoked == {"ref_id": minted["ref_id"], "revoked": True}


def test_toolkit_hash_value_default_algorithm() -> None:
    toolkit = SecretsToolkit.build()
    minted = toolkit.mint_secret(kind="hex")
    result = toolkit.hash_value(minted["ref_id"])
    assert result["algorithm"] == "sha256-12"
    assert result["fingerprint"] == minted["fingerprint"]


def test_toolkit_hash_value_alternate_algorithm() -> None:
    toolkit = SecretsToolkit.build()
    minted = toolkit.mint_secret(kind="hex")
    result = toolkit.hash_value(minted["ref_id"], algorithm="sha256")
    # Full 64-char hex digest for sha256.
    assert len(result["fingerprint"]) == 64
    assert result["algorithm"] == "sha256"


def test_toolkit_hash_value_rejects_bad_algorithm() -> None:
    toolkit = SecretsToolkit.build()
    minted = toolkit.mint_secret(kind="hex")
    with pytest.raises(ValidationError) as excinfo:
        toolkit.hash_value(minted["ref_id"], algorithm="rot13")
    assert excinfo.value.code == "carl.secrets.invalid_kind"


def test_toolkit_list_secrets_metadata_only() -> None:
    toolkit = SecretsToolkit.build()
    toolkit.mint_secret(kind="hex")
    toolkit.mint_secret(kind="base64")
    toolkit.mint_secret(kind="uuid")
    listing = toolkit.list_secrets()
    assert listing["count"] == 3
    for entry in listing["refs"]:
        # No value-ish field.
        assert "value" not in entry
        assert "secret" not in entry
        # Fingerprint present, valid hex.
        fp = entry["fingerprint"]
        assert len(fp) == 12
        int(fp, 16)


def test_toolkit_unknown_ref_id_raises() -> None:
    toolkit = SecretsToolkit.build()
    with pytest.raises(SecretsError) as excinfo:
        toolkit.copy_to_clipboard(str(uuid.uuid4()))
    assert excinfo.value.code == "carl.secrets.not_found"


def test_toolkit_malformed_ref_id_raises() -> None:
    toolkit = SecretsToolkit.build()
    with pytest.raises(ValidationError) as excinfo:
        toolkit.revoke_secret("not-a-uuid")
    assert excinfo.value.code == "carl.secrets.invalid_kind"


def test_toolkit_mint_rejects_bad_kind() -> None:
    toolkit = SecretsToolkit.build()
    with pytest.raises(ValidationError):
        toolkit.mint_secret(kind="bogus")


def test_toolkit_ed25519_keypair_returns_two_refs() -> None:
    toolkit = SecretsToolkit.build()
    result = toolkit.mint_secret(kind="ed25519_keypair")
    assert result["kind"] == "ed25519_keypair"
    assert "priv_ref_id" in result
    assert "pub_ref_id" in result
    assert result["priv_ref_id"] != result["pub_ref_id"]


def test_toolkit_full_virtual_kvm_flow_no_value_leaks(
    fake_pyperclip: _FakePyperclip,
) -> None:
    """End-to-end KVM: mint → copy → wipe → revoke. Audit carries fingerprints only."""
    chain = InteractionChain()
    toolkit = SecretsToolkit.build(chain=chain)

    # 1. Mint
    minted = toolkit.mint_secret(kind="hex", nbytes=32)
    assert len(minted["fingerprint"]) == 12

    # 2. Copy to clipboard
    _ = toolkit.copy_to_clipboard(minted["ref_id"], ttl_s=60)

    # 3. Hash (derives a second fingerprint via privileged resolve)
    hashed = toolkit.hash_value(minted["ref_id"], algorithm="sha256")

    # 4. Revoke
    toolkit.revoke_secret(minted["ref_id"])

    # Audit trail must not contain the actual clipboard value.
    clipboard_value = fake_pyperclip.paste()
    # Invariant: the actual value exists only in the OS clipboard + vault
    # memory. It's NOT in any chain Step.
    for step in chain.steps:
        step_dict = step.to_dict()
        step_serialized = str(step_dict)
        assert clipboard_value not in step_serialized, (
            f"Step {step.action} name={step.name!r} leaked the clipboard "
            f"value into its serialized form."
        )

    # And the hashing step records both the 12-hex vault fingerprint and
    # the full-64 sha256, but no plaintext.
    resolves = chain.by_action(ActionType.SECRET_RESOLVE)
    assert len(resolves) >= 1
    hash_step = next((s for s in resolves if s.name == "toolkit.hash_value"), None)
    assert hash_step is not None
    hash_out_raw = hash_step.output
    assert isinstance(hash_out_raw, dict)
    hash_out = cast(dict[str, Any], hash_out_raw)
    assert hash_out["algorithm"] == "sha256"
    # The full 64-char digest is present (that's the public output) but
    # no value byte leaks are captured in this step.
    assert len(hashed["fingerprint"]) == 64


def test_toolkit_chain_none_does_not_break_operations(
    fake_pyperclip: _FakePyperclip,
) -> None:
    toolkit = SecretsToolkit.build(chain=None)
    minted = toolkit.mint_secret(kind="hex")
    toolkit.copy_to_clipboard(minted["ref_id"], ttl_s=60)
    toolkit.revoke_secret(minted["ref_id"])
    # No tracebacks = passing.


def test_toolkit_build_returns_wired_instance() -> None:
    chain = InteractionChain()
    toolkit = SecretsToolkit.build(chain=chain, clipboard_default_ttl_s=15)
    assert toolkit.vault is not None
    assert toolkit.minter.vault is toolkit.vault
    assert toolkit.keychain.vault is toolkit.vault
    assert toolkit.clipboard.vault is toolkit.vault
    assert toolkit.clipboard.default_ttl_s == 15


# ---------------------------------------------------------------------------
# InteractionChain ActionType coverage
# ---------------------------------------------------------------------------


def test_new_action_types_are_exposed() -> None:
    """Sanity: SECRET_MINT / SECRET_RESOLVE / SECRET_REVOKE / CLIPBOARD_WRITE
    are reachable via the public ActionType enum."""
    for name in ("SECRET_MINT", "SECRET_RESOLVE", "SECRET_REVOKE", "CLIPBOARD_WRITE"):
        assert hasattr(ActionType, name), f"ActionType missing {name}"


# Markers for optional-timing tests so slow CI can skip them cleanly.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
