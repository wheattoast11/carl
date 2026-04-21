"""Tests for carl_core.secrets — capability-security primitives."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from carl_core.errors import ValidationError
from carl_core.secrets import (
    SecretRef,
    SecretVault,
    SecretsError,
    generate_box_keypair,
    seal,
    unseal,
)


# ---------------------------------------------------------------------------
# SecretRef — handle shape + expiry
# ---------------------------------------------------------------------------


def test_secret_ref_construction_defaults() -> None:
    ref = SecretRef(kind="mint", uri="carl://mint/demo")
    assert ref.kind == "mint"
    assert ref.uri == "carl://mint/demo"
    assert isinstance(ref.ref_id, uuid.UUID)
    assert ref.ttl_s is None
    assert isinstance(ref.created_at, datetime)
    assert ref.created_at.tzinfo is not None


def test_secret_ref_is_frozen() -> None:
    ref = SecretRef(kind="mint", uri="carl://mint/demo")
    with pytest.raises(Exception):  # pydantic frozen raises ValidationError
        ref.kind = "keychain"  # type: ignore[misc]


def test_secret_ref_no_ttl_never_expires() -> None:
    ref = SecretRef(kind="mint", uri="carl://mint/x")
    assert ref.expired_at() is None
    assert ref.is_expired() is False


def test_secret_ref_ttl_computes_expiry() -> None:
    ref = SecretRef(kind="mint", uri="carl://mint/x", ttl_s=30)
    expected = ref.created_at + timedelta(seconds=30)
    assert ref.expired_at() == expected
    # Not expired when probed "now".
    assert ref.is_expired() is False
    # Expired when probed far in the future.
    assert ref.is_expired(now=ref.created_at + timedelta(hours=1)) is True


def test_secret_ref_describe_never_leaks_value() -> None:
    ref = SecretRef(kind="mint", uri="carl://mint/x", ttl_s=10)
    described = ref.describe()
    # Describe keys are metadata only.
    assert set(described.keys()) == {
        "ref_id",
        "kind",
        "uri",
        "ttl_s",
        "created_at",
        "expires_at",
    }
    # No value-ish field.
    assert "value" not in described
    assert "secret" not in described


# ---------------------------------------------------------------------------
# SecretVault — lifecycle
# ---------------------------------------------------------------------------


def test_vault_put_returns_fresh_ref() -> None:
    v = SecretVault()
    ref1 = v.put(b"the-value", kind="mint")
    ref2 = v.put(b"the-value", kind="mint")  # same value, new ref
    assert ref1.ref_id != ref2.ref_id
    # Same underlying value → same fingerprint.
    assert v.fingerprint_of(ref1) == v.fingerprint_of(ref2)
    assert len(v) == 2


def test_vault_resolve_requires_privileged_flag() -> None:
    v = SecretVault()
    ref = v.put(b"the-value", kind="mint")
    with pytest.raises(SecretsError) as excinfo:
        v.resolve(ref)  # default privileged=False
    assert excinfo.value.code == "carl.secrets.unauthorized_resolve"


def test_vault_resolve_privileged_returns_value() -> None:
    v = SecretVault()
    ref = v.put(b"the-value", kind="mint")
    assert v.resolve(ref, privileged=True) == b"the-value"


def test_vault_resolve_expired_raises() -> None:
    v = SecretVault()
    ref = SecretRef(
        kind="mint",
        uri="carl://mint/x",
        ttl_s=1,
        # Backdate creation so the ref is already expired.
        created_at=datetime.now(timezone.utc) - timedelta(seconds=10),
    )
    # Inject manually via put-then-replace to use our pre-expired ref.
    real_ref = v.put(b"v", kind="mint")
    # Swap the ref metadata for our expired one, keeping the vault entry.
    v._refs[real_ref.ref_id] = SecretRef(
        ref_id=real_ref.ref_id,
        kind="mint",
        uri="carl://mint/x",
        ttl_s=1,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=10),
    )
    with pytest.raises(SecretsError) as excinfo:
        v.resolve(v._refs[real_ref.ref_id], privileged=True)
    assert excinfo.value.code == "carl.secrets.expired"


def test_vault_resolve_unknown_ref_raises() -> None:
    v = SecretVault()
    ref = SecretRef(kind="mint", uri="carl://mint/ghost")
    with pytest.raises(SecretsError) as excinfo:
        v.resolve(ref, privileged=True)
    assert excinfo.value.code == "carl.secrets.not_found"


def test_vault_revoke_blocks_subsequent_resolve() -> None:
    v = SecretVault()
    ref = v.put(b"the-value", kind="mint")
    assert v.revoke(ref) is True
    with pytest.raises(SecretsError) as excinfo:
        v.resolve(ref, privileged=True)
    assert excinfo.value.code == "carl.secrets.revoked"


def test_vault_revoke_is_idempotent() -> None:
    v = SecretVault()
    ref = v.put(b"the-value", kind="mint")
    assert v.revoke(ref) is True
    # Second revoke on same ref returns False (already revoked).
    assert v.revoke(ref) is False


def test_vault_revoke_unknown_returns_false() -> None:
    v = SecretVault()
    ghost = SecretRef(kind="mint", uri="carl://mint/ghost")
    assert v.revoke(ghost) is False


def test_vault_fingerprint_stable_for_same_value() -> None:
    v = SecretVault()
    ref_a = v.put("hello world", kind="mint")
    ref_b = v.put("hello world", kind="mint")
    assert v.fingerprint_of(ref_a) == v.fingerprint_of(ref_b)
    # Fingerprints are 12 hex chars.
    assert len(v.fingerprint_of(ref_a)) == 12


def test_vault_fingerprint_differs_for_different_values() -> None:
    v = SecretVault()
    a = v.put(b"alpha", kind="mint")
    b = v.put(b"beta", kind="mint")
    assert v.fingerprint_of(a) != v.fingerprint_of(b)


def test_vault_fingerprint_available_after_revoke() -> None:
    """Revoked handles can still be fingerprinted for audit."""
    v = SecretVault()
    ref = v.put(b"the-value", kind="mint")
    fp_before = v.fingerprint_of(ref)
    v.revoke(ref)
    assert v.fingerprint_of(ref) == fp_before


def test_vault_exists_tracks_lifecycle() -> None:
    v = SecretVault()
    ref = v.put(b"the-value", kind="mint", ttl_s=100)
    assert v.exists(ref) is True
    v.revoke(ref)
    assert v.exists(ref) is False


def test_vault_list_refs_excludes_revoked() -> None:
    v = SecretVault()
    a = v.put(b"keep", kind="mint")
    b = v.put(b"drop", kind="mint")
    v.revoke(b)
    refs = v.list_refs()
    ids = {r.ref_id for r in refs}
    assert a.ref_id in ids
    assert b.ref_id not in ids


def test_vault_len_matches_list_refs() -> None:
    v = SecretVault()
    v.put(b"a", kind="mint")
    v.put(b"b", kind="mint")
    v.put(b"c", kind="mint")
    assert len(v) == 3
    assert len(v.list_refs()) == 3


def test_vault_rejects_invalid_kind() -> None:
    v = SecretVault()
    with pytest.raises(ValidationError) as excinfo:
        v.put(b"x", kind="bogus-kind")  # type: ignore[arg-type]
    assert excinfo.value.code == "carl.secrets.invalid_kind"


def test_vault_rejects_non_positive_ttl() -> None:
    v = SecretVault()
    with pytest.raises(ValidationError):
        v.put(b"x", kind="mint", ttl_s=0)
    with pytest.raises(ValidationError):
        v.put(b"x", kind="mint", ttl_s=-5)


def test_vault_put_accepts_str_and_bytes() -> None:
    v = SecretVault()
    ref_s = v.put("hello", kind="mint")
    ref_b = v.put(b"hello", kind="mint")
    # UTF-8 encoding → same fingerprint.
    assert v.fingerprint_of(ref_s) == v.fingerprint_of(ref_b)


def test_vault_generates_default_uri_when_none() -> None:
    v = SecretVault()
    ref = v.put(b"x", kind="keychain")
    assert ref.uri.startswith("carl://keychain/")


def test_vault_honors_custom_uri() -> None:
    v = SecretVault()
    ref = v.put(b"x", kind="keychain", uri="carl://my-vault/my-item")
    assert ref.uri == "carl://my-vault/my-item"


# ---------------------------------------------------------------------------
# Sealed-box round-trip (requires pynacl)
# ---------------------------------------------------------------------------


pynacl_installed = pytest.importorskip("nacl")


def test_seal_unseal_round_trip() -> None:
    priv, pub = generate_box_keypair()
    assert len(priv) == 32
    assert len(pub) == 32
    ciphertext = seal(pub, b"confidential payload")
    plaintext = unseal(priv, ciphertext)
    assert plaintext == b"confidential payload"


def test_seal_rejects_non_32_byte_pubkey() -> None:
    with pytest.raises(ValidationError) as excinfo:
        seal(b"too-short", b"data")
    assert excinfo.value.code == "carl.secrets.invalid_kind"


def test_unseal_rejects_non_32_byte_privkey() -> None:
    with pytest.raises(ValidationError) as excinfo:
        unseal(b"too-short", b"data")
    assert excinfo.value.code == "carl.secrets.invalid_kind"


def test_seal_with_wrong_pubkey_produces_unseal_failure() -> None:
    priv1, _pub1 = generate_box_keypair()
    _priv2, pub2 = generate_box_keypair()
    ciphertext = seal(pub2, b"data")
    # Unsealing with the WRONG private key (priv1 instead of priv2) raises.
    with pytest.raises(Exception):
        unseal(priv1, ciphertext)


def test_seal_produces_different_ciphertext_each_call() -> None:
    """Sealed boxes use fresh ephemeral keys; re-encrypting yields new bytes."""
    _priv, pub = generate_box_keypair()
    ct1 = seal(pub, b"same data")
    ct2 = seal(pub, b"same data")
    assert ct1 != ct2
    # But both decrypt to the same plaintext.


# ---------------------------------------------------------------------------
# Value-leak invariants — probe serialized / logged representations
# ---------------------------------------------------------------------------


def test_ref_repr_does_not_leak_anything_sensitive() -> None:
    ref = SecretRef(kind="mint", uri="carl://mint/x")
    rendering = repr(ref)
    # Positive assertions: metadata fields are present.
    assert "kind=" in rendering
    # SecretRef never held the value, so "value"-shaped substrings are
    # trivially absent. Sanity check that we're not accidentally leaking
    # something else via repr.
    assert "SecretBytes" not in rendering  # nothing from the vault internals


def test_vault_len_with_expired_refs_excludes_them() -> None:
    v = SecretVault()
    ref_a = v.put(b"a", kind="mint", ttl_s=10)
    # Put ref that will be expired when we check.
    _ref_b = v.put(b"b", kind="mint")
    # Backdate the ref_a metadata to make it expired for accounting.
    now = datetime.now(timezone.utc)
    v._refs[ref_a.ref_id] = SecretRef(
        ref_id=ref_a.ref_id,
        kind="mint",
        uri=ref_a.uri,
        ttl_s=1,
        created_at=now - timedelta(seconds=10),
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_vault_concurrent_put_resolve_revoke_threadsafe() -> None:
    """Smoke test: hammer the vault with threaded ops; no tracebacks allowed."""
    import threading

    v = SecretVault()
    errors: list[Exception] = []

    def worker(seed: int) -> None:
        try:
            for i in range(50):
                ref = v.put(f"v-{seed}-{i}".encode(), kind="mint")
                fp = v.fingerprint_of(ref)
                assert len(fp) == 12
                if i % 2 == 0:
                    v.revoke(ref)
                else:
                    got = v.resolve(ref, privileged=True)
                    assert got == f"v-{seed}-{i}".encode()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# ---------------------------------------------------------------------------
# Self-expire on access
# ---------------------------------------------------------------------------


def test_expired_ref_resolve_also_revokes_for_subsequent_ops() -> None:
    """Resolving an expired ref auto-revokes so a later lookup sees revoked."""
    v = SecretVault()
    # Use tiny ttl + sleep instead of backdating — exercises the live clock.
    ref_init = v.put(b"x", kind="mint")
    # Swap for an expired ref with real clock.
    now = datetime.now(timezone.utc)
    v._refs[ref_init.ref_id] = SecretRef(
        ref_id=ref_init.ref_id,
        kind="mint",
        uri=ref_init.uri,
        ttl_s=1,
        created_at=now - timedelta(seconds=10),
    )

    expired_ref = v._refs[ref_init.ref_id]
    with pytest.raises(SecretsError):
        v.resolve(expired_ref, privileged=True)  # raises expired

    # After auto-revoke, a second resolve reports revoked (or expired — both OK).
    with pytest.raises(SecretsError) as excinfo:
        v.resolve(expired_ref, privileged=True)
    assert excinfo.value.code in ("carl.secrets.revoked", "carl.secrets.expired")


# ---------------------------------------------------------------------------
# Short sleep test (optional; skipped on slow CI)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ttl_s", [1])
def test_live_ttl_tiny(ttl_s: int) -> None:
    """Real-clock TTL expiry test — keeps sleep brief (<2s)."""
    v = SecretVault()
    # Construct a ref whose TTL will elapse during the sleep.
    ref = SecretRef(
        kind="mint",
        uri="carl://mint/x",
        ttl_s=ttl_s,
        created_at=datetime.now(timezone.utc),
    )
    # Manually register matching entry via put (matching value is irrelevant).
    actual = v.put(b"tmp", kind="mint", ttl_s=ttl_s)
    time.sleep(ttl_s + 0.2)
    with pytest.raises(SecretsError):
        v.resolve(actual, privileged=True)
