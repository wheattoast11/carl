"""Tests for the generic `Vault[H, V]` base class + resolver chain.

Covers:
- Handle lifecycle (put → resolve → revoke → error paths)
- TTL self-revoke at resolve time
- Privileged resolve guard (class-level opt-in)
- Resolver chain fall-through (local value → registered resolver → not_found)
- Resolver TTL caching
- Thread safety under concurrent put/revoke
- Property-based invariants via hypothesis
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from carl_core.vault import (
    HandleRef,
    Vault,
    VaultError,
)


# ---------------------------------------------------------------------------
# Test handle + vault (minimal shapes for base-class verification)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TestRef:
    """Minimal HandleRef satisfying the Protocol."""

    ref_id: uuid.UUID
    kind: str
    ttl_s: int | None = None
    created_at: datetime = datetime.now(timezone.utc)

    def is_expired(self, *, now: datetime | None = None) -> bool:
        if self.ttl_s is None:
            return False
        current = now if now is not None else datetime.now(timezone.utc)
        return current >= self.created_at + timedelta(seconds=self.ttl_s)

    def describe(self) -> dict[str, Any]:
        return {
            "ref_id": str(self.ref_id),
            "kind": self.kind,
            "ttl_s": self.ttl_s,
        }


class _TestVault(Vault[_TestRef, bytes]):
    """Non-privileged vault over test handles."""

    _ref_class = _TestRef


class _PrivilegedTestVault(Vault[_TestRef, bytes]):
    """Privileged vault — resolve() demands privileged=True."""

    _ref_class = _TestRef
    _require_privileged_resolve = True


class _CachedTestVault(Vault[_TestRef, bytes]):
    """Test vault with 0.5-second resolver cache for TTL tests."""

    _ref_class = _TestRef
    _resolver_ttl_s = 1


def _fresh_ref(kind: str = "test", ttl_s: int | None = None) -> _TestRef:
    return _TestRef(
        ref_id=uuid.uuid4(),
        kind=kind,
        ttl_s=ttl_s,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# HandleRef protocol verification
# ---------------------------------------------------------------------------


def test_testref_satisfies_handleref_protocol() -> None:
    ref = _fresh_ref()
    assert isinstance(ref, HandleRef)


# ---------------------------------------------------------------------------
# put_value + resolve
# ---------------------------------------------------------------------------


def test_put_value_then_resolve_roundtrip() -> None:
    vault = _TestVault()
    ref = _fresh_ref()
    vault.put_value(ref, b"hello")
    assert vault.resolve(ref) == b"hello"


def test_resolve_unknown_ref_raises_not_found() -> None:
    vault = _TestVault()
    ghost = _fresh_ref()
    with pytest.raises(VaultError) as exc:
        vault.resolve(ghost)
    assert exc.value.code == "carl.vault.not_found"


# ---------------------------------------------------------------------------
# Privileged-resolve guard
# ---------------------------------------------------------------------------


def test_privileged_vault_requires_privileged_true() -> None:
    vault = _PrivilegedTestVault()
    ref = _fresh_ref()
    vault.put_value(ref, b"secret")
    with pytest.raises(VaultError) as exc:
        vault.resolve(ref)
    assert exc.value.code == "carl.vault.unauthorized_resolve"
    # Opt-in resolves succeed
    assert vault.resolve(ref, privileged=True) == b"secret"


def test_non_privileged_vault_resolves_without_opt_in() -> None:
    vault = _TestVault()
    ref = _fresh_ref()
    vault.put_value(ref, b"public")
    # `privileged=False` is the default; resolve succeeds.
    assert vault.resolve(ref) == b"public"


# ---------------------------------------------------------------------------
# Revoke + TTL self-revoke
# ---------------------------------------------------------------------------


def test_revoke_is_idempotent_and_blocks_resolve() -> None:
    vault = _TestVault()
    ref = _fresh_ref()
    vault.put_value(ref, b"x")
    assert vault.revoke(ref) is True
    assert vault.revoke(ref) is False  # second revoke returns False
    with pytest.raises(VaultError) as exc:
        vault.resolve(ref)
    assert exc.value.code == "carl.vault.revoked"


def test_ttl_self_revoke_on_resolve() -> None:
    vault = _TestVault()
    ref = _fresh_ref(ttl_s=1)
    vault.put_value(ref, b"tick")
    time.sleep(1.05)
    with pytest.raises(VaultError) as exc:
        vault.resolve(ref)
    assert exc.value.code == "carl.vault.expired"
    # Expired refs also disappear from list_refs
    assert vault.list_refs() == []


def test_invalid_ttl_rejected_by_subclass_using_helper() -> None:
    """Subclass put methods call `_validate_ttl` to reject non-positive TTLs.

    We verify the helper here via a minimal subclass that exposes it through
    a typed `put_with_ttl` method — matches how real specializations (Secret,
    Data, Resource) gate their `put_*` entry points.
    """

    class _TTLEnforcingVault(Vault[_TestRef, bytes]):
        _ref_class = _TestRef

        def put_with_ttl(self, ref: _TestRef, value: bytes, *, ttl_s: int | None) -> _TestRef:
            self._validate_ttl(ttl_s)
            return self.put_value(ref, value)

    vault = _TTLEnforcingVault()
    ref = _fresh_ref()
    with pytest.raises(Exception) as exc:
        vault.put_with_ttl(ref, b"x", ttl_s=0)
    assert "ttl_s" in str(exc.value)


# ---------------------------------------------------------------------------
# list_refs / __len__ / exists
# ---------------------------------------------------------------------------


def test_list_refs_omits_revoked_and_expired() -> None:
    vault = _TestVault()
    live = _fresh_ref()
    gone = _fresh_ref()
    vault.put_value(live, b"a")
    vault.put_value(gone, b"b")
    vault.revoke(gone)
    short = _fresh_ref(ttl_s=1)
    vault.put_value(short, b"c")
    time.sleep(1.05)
    listed = {r.ref_id for r in vault.list_refs()}
    assert listed == {live.ref_id}
    assert len(vault) == 1


def test_exists_reflects_lifecycle() -> None:
    vault = _TestVault()
    ref = _fresh_ref()
    vault.put_value(ref, b"x")
    assert vault.exists(ref) is True
    vault.revoke(ref)
    assert vault.exists(ref) is False


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_of_bytes_value_is_12_hex() -> None:
    vault = _TestVault()
    ref = _fresh_ref()
    vault.put_value(ref, b"fingerprint-me")
    fp = vault.fingerprint_of(ref)
    assert len(fp) == 12
    assert all(c in "0123456789abcdef" for c in fp)
    # Deterministic
    assert vault.fingerprint_of(ref) == fp


def test_fingerprint_of_unknown_ref_raises_not_found() -> None:
    vault = _TestVault()
    with pytest.raises(VaultError) as exc:
        vault.fingerprint_of(_fresh_ref())
    assert exc.value.code == "carl.vault.not_found"


# ---------------------------------------------------------------------------
# Resolver chain — registration + fall-through
# ---------------------------------------------------------------------------


def test_resolver_falls_through_when_no_local_value() -> None:
    vault = _TestVault()

    def env_resolver(ref: HandleRef) -> bytes:
        return f"resolved:{ref.ref_id}".encode()

    vault.register_resolver("env", env_resolver)
    ref = _fresh_ref(kind="env")
    vault.put_ref_only(ref)
    got = vault.resolve(ref)
    assert got == f"resolved:{ref.ref_id}".encode()


def test_no_value_no_resolver_raises_resolver_unavailable() -> None:
    vault = _TestVault()
    ref = _fresh_ref(kind="stranger")
    vault.put_ref_only(ref)
    with pytest.raises(VaultError) as exc:
        vault.resolve(ref)
    assert exc.value.code == "carl.vault.resolver_unavailable"


def test_resolver_exception_wrapped_as_resolver_failed() -> None:
    vault = _TestVault()

    def broken(_ref: HandleRef) -> bytes:
        raise RuntimeError("backend down")

    vault.register_resolver("env", broken)
    ref = _fresh_ref(kind="env")
    vault.put_ref_only(ref)
    with pytest.raises(VaultError) as exc:
        vault.resolve(ref)
    assert exc.value.code == "carl.vault.resolver_failed"


def test_resolver_vault_error_propagates_unchanged() -> None:
    vault = _TestVault()

    def vault_err(_ref: HandleRef) -> bytes:
        raise VaultError(
            "not-so-resolvable",
            code="carl.vault.custom",
            context={"detail": "x"},
        )

    vault.register_resolver("env", vault_err)
    ref = _fresh_ref(kind="env")
    vault.put_ref_only(ref)
    with pytest.raises(VaultError) as exc:
        vault.resolve(ref)
    # VaultErrors bypass the wrapper; the original code survives
    assert exc.value.code == "carl.vault.custom"


def test_register_resolver_rejects_non_callable() -> None:
    vault = _TestVault()
    with pytest.raises(Exception) as exc:
        vault.register_resolver("env", "not a function")  # type: ignore[arg-type]
    assert "carl.vault.invalid_resolver" in str(
        getattr(exc.value, "code", exc.value)
    )


def test_unregister_resolver_reports_presence() -> None:
    vault = _TestVault()
    vault.register_resolver("env", lambda _r: b"ok")
    assert vault.unregister_resolver("env") is True
    assert vault.unregister_resolver("env") is False  # no-op second time


def test_registered_kinds_is_sorted() -> None:
    vault = _TestVault()
    vault.register_resolver("zeta", lambda _r: b"z")
    vault.register_resolver("alpha", lambda _r: b"a")
    vault.register_resolver("mu", lambda _r: b"m")
    assert vault.registered_kinds() == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# Resolver TTL caching
# ---------------------------------------------------------------------------


def test_resolver_cache_hits_within_ttl() -> None:
    vault = _CachedTestVault()  # _resolver_ttl_s = 1
    calls = {"count": 0}

    def counting_resolver(_ref: HandleRef) -> bytes:
        calls["count"] += 1
        return b"v"

    vault.register_resolver("env", counting_resolver)
    ref = _fresh_ref(kind="env")
    vault.put_ref_only(ref)
    vault.resolve(ref)
    vault.resolve(ref)
    vault.resolve(ref)
    assert calls["count"] == 1  # cached after first call


def test_resolver_cache_expires() -> None:
    vault = _CachedTestVault()  # _resolver_ttl_s = 1
    calls = {"count": 0}

    def counting_resolver(_ref: HandleRef) -> bytes:
        calls["count"] += 1
        return b"v"

    vault.register_resolver("env", counting_resolver)
    ref = _fresh_ref(kind="env")
    vault.put_ref_only(ref)
    vault.resolve(ref)
    time.sleep(1.1)
    vault.resolve(ref)
    assert calls["count"] == 2


def test_revoke_clears_resolver_cache() -> None:
    vault = _CachedTestVault()
    calls = {"count": 0}

    def counting_resolver(_ref: HandleRef) -> bytes:
        calls["count"] += 1
        return b"v"

    vault.register_resolver("env", counting_resolver)
    ref = _fresh_ref(kind="env")
    vault.put_ref_only(ref)
    vault.resolve(ref)
    vault.revoke(ref)
    # The ref is revoked — resolving now raises, not using the cached value.
    with pytest.raises(VaultError) as exc:
        vault.resolve(ref)
    assert exc.value.code == "carl.vault.revoked"


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------


@given(
    value=st.binary(min_size=0, max_size=1024),
)
@settings(max_examples=50, deadline=None)
def test_property_put_then_resolve_returns_value(value: bytes) -> None:
    vault = _TestVault()
    ref = _fresh_ref()
    vault.put_value(ref, value)
    assert vault.resolve(ref) == value


@given(value=st.binary(min_size=1, max_size=256))
@settings(max_examples=50, deadline=None)
def test_property_put_then_revoke_then_resolve_errors(value: bytes) -> None:
    vault = _TestVault()
    ref = _fresh_ref()
    vault.put_value(ref, value)
    vault.revoke(ref)
    with pytest.raises(VaultError) as exc:
        vault.resolve(ref)
    assert exc.value.code == "carl.vault.revoked"


@given(values=st.lists(st.binary(min_size=0, max_size=64), min_size=0, max_size=20))
@settings(max_examples=20, deadline=None)
def test_property_len_matches_live_refs(values: list[bytes]) -> None:
    vault = _TestVault()
    refs: list[_TestRef] = []
    for v in values:
        ref = _fresh_ref()
        vault.put_value(ref, v)
        refs.append(ref)
    assert len(vault) == len(values)
    # Revoke half
    for r in refs[::2]:
        vault.revoke(r)
    expected_live = len(values) - (len(values) + 1) // 2
    assert len(vault) == expected_live


# ---------------------------------------------------------------------------
# Thread safety — concurrent put/revoke
# ---------------------------------------------------------------------------


def test_concurrent_put_revoke_maintains_consistency() -> None:
    vault = _TestVault()
    refs: list[_TestRef] = []
    lock = threading.Lock()

    def producer() -> None:
        for _ in range(20):
            ref = _fresh_ref()
            vault.put_value(ref, b"x")
            with lock:
                refs.append(ref)

    def revoker() -> None:
        time.sleep(0.01)
        for r in list(refs):
            vault.revoke(r)

    threads = [threading.Thread(target=producer) for _ in range(3)]
    threads.append(threading.Thread(target=revoker))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Final invariant: the counts are consistent (no orphaned or double-state entries)
    live_count = len(vault)
    # It's fine to have non-zero live count — the revoker may have missed late
    # producers. What must hold: vault.list_refs() is internally consistent.
    assert live_count == len(vault.list_refs())


