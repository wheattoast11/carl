"""Tests for carl_core.resource_handles — ResourceRef, ResourceVault lifecycle."""

from __future__ import annotations

import threading
import uuid
from typing import Any, cast

import pytest

from carl_core.resource_handles import (
    ResourceError,
    ResourceRef,
    ResourceVault,
)


class _FakeBackend:
    """Minimal resource backend for test purposes."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.closed = False


def _closer(backend: _FakeBackend) -> None:
    backend.closed = True


# ---------------------------------------------------------------------------
# ResourceRef metadata
# ---------------------------------------------------------------------------


def test_resourceref_describes_without_backend() -> None:
    ref = ResourceRef(
        kind="browser_page",
        provider="playwright",
        uri="https://example.com",
        labels={"role": "login"},
    )
    d = ref.describe()
    assert d["kind"] == "browser_page"
    assert d["provider"] == "playwright"
    assert d["uri"] == "https://example.com"
    assert d["labels"] == {"role": "login"}


def test_resourceref_is_frozen() -> None:
    ref = ResourceRef(kind="subprocess", provider="subprocess", uri="pid:100")
    with pytest.raises(Exception):
        ref.uri = "pid:200"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ResourceVault put / resolve / revoke
# ---------------------------------------------------------------------------


def test_put_and_resolve_requires_privilege() -> None:
    vault = ResourceVault()
    backend = _FakeBackend("a")
    ref = vault.put(
        backend,
        kind="browser_page",
        provider="playwright",
        uri="about:blank",
    )
    with pytest.raises(ResourceError) as exc:
        vault.resolve(ref)
    assert exc.value.code == "carl.resource.unauthorized_resolve"
    got = vault.resolve(ref, privileged=True)
    assert got is backend


def test_revoke_runs_closer() -> None:
    vault = ResourceVault()
    backend = _FakeBackend("a")
    ref = vault.put(
        backend,
        kind="subprocess",
        provider="subprocess",
        uri="pid:42",
        closer=_closer,
    )
    assert vault.revoke(ref) is True
    assert backend.closed is True
    # Resolve after revoke raises
    with pytest.raises(ResourceError) as exc:
        vault.resolve(ref, privileged=True)
    assert exc.value.code == "carl.resource.revoked"


def test_revoke_is_idempotent() -> None:
    vault = ResourceVault()
    backend = _FakeBackend()
    ref = vault.put(
        backend, kind="subprocess", provider="subprocess", uri="pid:1"
    )
    assert vault.revoke(ref) is True
    assert vault.revoke(ref) is False


def test_revoke_with_broken_closer_raises_resource_error() -> None:
    vault = ResourceVault()
    backend = _FakeBackend()

    def boom(_: Any) -> None:
        raise RuntimeError("cleanup died")

    ref = vault.put(
        backend,
        kind="subprocess",
        provider="subprocess",
        uri="pid:broken",
        closer=boom,
    )
    with pytest.raises(ResourceError) as exc:
        vault.revoke(ref)
    assert exc.value.code == "carl.resource.backend_error"


def test_invalid_kind_rejected() -> None:
    vault = ResourceVault()
    with pytest.raises(Exception) as exc:
        vault.put(_FakeBackend(), kind=cast(Any, "hologram"), provider="x", uri="x")
    assert "invalid_kind" in str(cast(Any, exc.value).__dict__.get("code", exc.value))


def test_invalid_ttl_rejected() -> None:
    vault = ResourceVault()
    with pytest.raises(Exception) as exc:
        vault.put(
            _FakeBackend(),
            kind="subprocess",
            provider="subprocess",
            uri="pid:1",
            ttl_s=0,
        )
    assert "ttl_s" in str(exc.value) or "invalid_kind" in str(exc.value)


# ---------------------------------------------------------------------------
# list_refs / exists / len
# ---------------------------------------------------------------------------


def test_list_refs_omits_revoked() -> None:
    vault = ResourceVault()
    live = vault.put(
        _FakeBackend("a"), kind="subprocess", provider="subprocess", uri="pid:1"
    )
    gone = vault.put(
        _FakeBackend("b"), kind="subprocess", provider="subprocess", uri="pid:2"
    )
    vault.revoke(gone)
    listed = {r.ref_id for r in vault.list_refs()}
    assert live.ref_id in listed
    assert gone.ref_id not in listed
    assert len(vault) == 1


def test_unknown_ref_resolve_raises_not_found() -> None:
    vault = ResourceVault()
    phantom = ResourceRef(
        ref_id=uuid.uuid4(),
        kind="browser_page",
        provider="playwright",
        uri="x",
    )
    with pytest.raises(ResourceError) as exc:
        vault.resolve(phantom, privileged=True)
    assert exc.value.code == "carl.resource.not_found"


# ---------------------------------------------------------------------------
# Thread safety sanity check
# ---------------------------------------------------------------------------


def test_concurrent_put_and_revoke_safe() -> None:
    vault = ResourceVault()
    refs: list[ResourceRef] = []

    def producer() -> None:
        for i in range(20):
            refs.append(
                vault.put(
                    _FakeBackend(str(i)),
                    kind="subprocess",
                    provider="subprocess",
                    uri=f"pid:{i}",
                )
            )

    threads = [threading.Thread(target=producer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(refs) == 80
    # Revoke all; lengths consistent
    for ref in refs:
        vault.revoke(ref)
    assert len(vault) == 0
