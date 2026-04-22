"""Tests for carl_core.data_handles — DataRef, DataVault, put/open/read lifecycle."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, cast

import pytest

from carl_core.data_handles import (
    DataError,
    DataRef,
    DataVault,
    content_type_from_path,
)


# ---------------------------------------------------------------------------
# DataRef metadata
# ---------------------------------------------------------------------------


def test_dataref_describes_without_value() -> None:
    ref = DataRef(kind="bytes", uri="carl-data://bytes/x", size_bytes=3)
    d = ref.describe()
    assert d["kind"] == "bytes"
    assert d["uri"] == "carl-data://bytes/x"
    assert d["size_bytes"] == 3
    assert d["ttl_s"] is None
    assert d["expires_at"] is None
    assert "sha256" in d


def test_dataref_expiry() -> None:
    ref = DataRef(kind="bytes", uri="carl-data://bytes/x", ttl_s=1)
    assert not ref.is_expired()
    time.sleep(1.05)
    assert ref.is_expired()


def test_dataref_frozen() -> None:
    ref = DataRef(kind="bytes", uri="u")
    with pytest.raises(Exception):
        ref.kind = "file"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# put_bytes — inline payload
# ---------------------------------------------------------------------------


def test_put_bytes_registers_with_full_hash() -> None:
    vault = DataVault()
    ref = vault.put_bytes(b"hello world")
    assert ref.kind == "bytes"
    assert ref.size_bytes == 11
    assert ref.sha256 is not None and len(ref.sha256) == 64
    assert vault.read_all(ref) == b"hello world"


def test_put_bytes_supports_content_type() -> None:
    vault = DataVault()
    ref = vault.put_bytes(b"{}", content_type="application/json")
    assert ref.content_type == "application/json"


def test_put_bytes_offset_length_read() -> None:
    vault = DataVault()
    ref = vault.put_bytes(b"abcdefg")
    assert vault.read(ref, offset=2, length=3) == b"cde"
    assert vault.read(ref, offset=5) == b"fg"
    assert vault.read(ref, offset=10) == b""  # past-end tolerated


# ---------------------------------------------------------------------------
# open_file — path-backed
# ---------------------------------------------------------------------------


def test_open_file_lazy_sha256(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"file contents here")

    vault = DataVault()
    ref = vault.open_file(p)
    assert ref.kind == "file"
    assert ref.size_bytes == 18
    assert ref.sha256 is None  # lazy
    data = vault.read_all(ref)
    assert data == b"file contents here"
    # After full read, sha256 is populated on the stored ref
    stored = next(r for r in vault.list_refs() if r.ref_id == ref.ref_id)
    assert stored.sha256 is not None and len(stored.sha256) == 64


def test_open_file_content_type_inferred(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    p.write_text("{}")
    vault = DataVault()
    ref = vault.open_file(p)
    assert ref.content_type == "application/json"


def test_open_file_rejects_missing(tmp_path: Path) -> None:
    vault = DataVault()
    with pytest.raises(DataError) as exc:
        vault.open_file(tmp_path / "does-not-exist.txt")
    assert exc.value.code == "carl.data.backend_unavailable"


# ---------------------------------------------------------------------------
# open_stream — consumable iterator
# ---------------------------------------------------------------------------


def test_open_stream_consumes_lazily() -> None:
    chunks = [b"aa", b"bb", b"cc"]
    vault = DataVault()
    ref = vault.open_stream(iter(chunks), uri="carl-data://stream/x")
    assert ref.kind == "stream"
    assert vault.read(ref, offset=0, length=4) == b"aabb"
    # Read past already-buffered portion pulls more chunks
    assert vault.read(ref, offset=0, length=6) == b"aabbcc"


def test_open_stream_size_hint_preserved() -> None:
    vault = DataVault()
    ref = vault.open_stream(iter([b"xy"]), uri="carl-data://stream/x", size_hint=2)
    assert ref.size_bytes == 2


# ---------------------------------------------------------------------------
# put_external — query / url / derived
# ---------------------------------------------------------------------------


def test_put_external_query() -> None:
    vault = DataVault()
    ref = vault.put_external(
        "duckdb://select * from t", kind="query", size_hint=None
    )
    assert ref.kind == "query"
    # Reading an external ref should raise — resolver lives above
    with pytest.raises(DataError) as exc:
        vault.read(ref)
    assert exc.value.code == "carl.data.backend_unavailable"


def test_put_external_rejects_invalid_kind() -> None:
    vault = DataVault()
    with pytest.raises(Exception) as exc:
        vault.put_external("x", kind="bytes")  # bytes isn't external
    # carl.data.invalid_kind is the expected code
    assert "invalid_kind" in str(cast(Any, exc.value).__dict__.get("code", exc.value))


# ---------------------------------------------------------------------------
# fingerprint / sha256 / size
# ---------------------------------------------------------------------------


def test_fingerprint_and_sha256_consistent() -> None:
    vault = DataVault()
    ref = vault.put_bytes(b"hello world")
    fp = vault.fingerprint_of(ref)
    sha = vault.sha256_of(ref)
    assert sha.startswith(fp)
    assert len(fp) == 12
    assert len(sha) == 64


def test_fingerprint_lazy_for_file(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"abc")
    vault = DataVault()
    ref = vault.open_file(p)
    # No sha at open time
    assert ref.sha256 is None
    # Requesting fingerprint forces a read + populates internally
    fp = vault.fingerprint_of(ref)
    assert len(fp) == 12


# ---------------------------------------------------------------------------
# Revoke / expiry / lifecycle
# ---------------------------------------------------------------------------


def test_revoke_is_idempotent() -> None:
    vault = DataVault()
    ref = vault.put_bytes(b"x")
    assert vault.revoke(ref) is True
    assert vault.revoke(ref) is False
    with pytest.raises(DataError) as exc:
        vault.read(ref)
    assert exc.value.code == "carl.data.revoked"


def test_expiry_self_revokes() -> None:
    vault = DataVault()
    ref = vault.put_bytes(b"x", ttl_s=1)
    time.sleep(1.05)
    with pytest.raises(DataError) as exc:
        vault.read(ref)
    assert exc.value.code == "carl.data.expired"
    assert vault.exists(ref) is False


def test_list_refs_omits_revoked_and_expired() -> None:
    vault = DataVault()
    live = vault.put_bytes(b"a")
    revoked = vault.put_bytes(b"b")
    vault.revoke(revoked)
    vault.put_bytes(b"c", ttl_s=1)
    time.sleep(1.05)
    listed = {r.ref_id for r in vault.list_refs()}
    assert live.ref_id in listed
    assert revoked.ref_id not in listed
    assert len(listed) == 1
    assert len(vault) == 1


def test_unknown_ref_raises_not_found() -> None:
    vault = DataVault()
    phantom = DataRef(ref_id=uuid.uuid4(), kind="bytes", uri="x")
    with pytest.raises(DataError) as exc:
        vault.read(phantom)
    assert exc.value.code == "carl.data.not_found"


def test_invalid_ttl_rejected() -> None:
    vault = DataVault()
    with pytest.raises(Exception) as exc:
        vault.put_bytes(b"x", ttl_s=0)
    assert "ttl_s" in str(exc.value) or "invalid_kind" in str(exc.value)


def test_out_of_range_offset_rejected() -> None:
    vault = DataVault()
    ref = vault.put_bytes(b"abc")
    with pytest.raises(DataError) as exc:
        vault.read(ref, offset=-1)
    assert exc.value.code == "carl.data.out_of_range"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_content_type_from_path(tmp_path: Path) -> None:
    assert content_type_from_path(tmp_path / "x.json") == "application/json"
    assert content_type_from_path(tmp_path / "x.txt") == "text/plain"
    assert content_type_from_path(tmp_path / "x.unknown-suffix-xyz") is None
