"""Tests for ``src/carl_studio/entitlements.py`` (v0.10 remote tier verification)."""

from __future__ import annotations

import base64
import json
import os
import stat
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from carl_core.errors import (
    EntitlementsCacheError,
    EntitlementsNetworkError,
    EntitlementsSignatureError,
    JWKSStaleError,
)
from carl_studio.entitlements import (
    DEFAULT_CARL_CAMP_BASE,
    ENTITLEMENTS_PATH,
    JWKS_PATH,
    EntitlementsClient,
    reset_default_client,
)
from nacl.signing import SigningKey


# ---------------------------------------------------------------------------
# JWT/JWKS fixture helpers
# ---------------------------------------------------------------------------


def _b64url_no_pad(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _make_keypair(kid: str = "cc-test-1") -> tuple[SigningKey, dict[str, Any]]:
    """Return ``(signing_key, jwks_entry)`` for a single Ed25519 keypair."""
    sk = SigningKey.generate()
    pub_b64url = _b64url_no_pad(bytes(sk.verify_key))
    entry = {
        "kty": "OKP",
        "crv": "Ed25519",
        "kid": kid,
        "alg": "EdDSA",
        "use": "sig",
        "x": pub_b64url,
    }
    return sk, entry


def _make_jwt(
    payload: dict[str, Any] | None = None,
    *,
    kid: str = "cc-test-1",
    signing_key: SigningKey | None = None,
    exp_offset_s: int = 15 * 60,
) -> tuple[str, dict[str, Any]]:
    """Sign a fresh JWT, returning ``(token, jwks_dict_with_only_this_kid)``."""
    sk: SigningKey
    if signing_key is None:
        sk_local, entry = _make_keypair(kid=kid)
        sk = sk_local
    else:
        sk = signing_key
        entry = _make_keypair(kid=kid)[1]
        # Replace `x` with the actual signing key's public part
        entry["x"] = _b64url_no_pad(bytes(sk.verify_key))

    now = int(time.time())
    body: dict[str, Any] = {
        "iss": "carl.camp",
        "aud": "carl-studio",
        "sub": "user-123",
        "tier": "PAID",
        "tier_label": "managed_payg",
        "entitlements": [
            {"key": "agent.publish", "granted_at": now},
            {"key": "train.slime", "granted_at": now},
        ],
        "iat": now,
        "exp": now + exp_offset_s,
    }
    if payload is not None:
        body.update(payload)

    header = {"alg": "EdDSA", "typ": "JWT", "kid": kid}
    h_b64 = _b64url_no_pad(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p_b64 = _b64url_no_pad(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    sig = sk.sign(signing_input).signature
    sig_b64 = _b64url_no_pad(sig)
    token = f"{h_b64}.{p_b64}.{sig_b64}"
    jwks = {"keys": [entry]}
    return token, jwks


def _client(tmp_path: Path) -> EntitlementsClient:
    """Construct an EntitlementsClient with cache paths under tmp_path."""
    return EntitlementsClient(
        base_url=DEFAULT_CARL_CAMP_BASE,
        cache_path=tmp_path / "entitlements_cache.json",
        jwks_path_local=tmp_path / "jwks_cache.json",
    )


# ---------------------------------------------------------------------------
# Autouse: reset the singleton/background executor between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    reset_default_client()
    yield
    reset_default_client()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_remote_happy_path_caches_jwt(tmp_path: Path) -> None:
    """Fetch returns Entitlements; second call within TTL hits cache (no net)."""
    sk, entry = _make_keypair()
    token, _jwks = _make_jwt(signing_key=sk)
    jwks = {"keys": [entry]}

    ent_route = respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        return_value=httpx.Response(200, json={"jwt": token})
    )
    jwks_route = respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json=jwks)
    )

    client = _client(tmp_path)
    result = client.fetch_remote("user-bearer-token")

    assert result.tier == "PAID"
    assert result.tier_label == "managed_payg"
    assert result.key_id == "cc-test-1"
    assert {g.key for g in result.entitlements} == {"agent.publish", "train.slime"}
    assert ent_route.call_count == 1
    assert jwks_route.call_count == 1

    # Second call within TTL must NOT touch the network.
    again = client.fetch_remote("user-bearer-token")
    assert again.tier == "PAID"
    assert ent_route.call_count == 1  # still 1, served from cache
    assert jwks_route.call_count == 1


@respx.mock
def test_fetch_remote_signature_invalid_raises(tmp_path: Path) -> None:
    """Token signed with wrong key raises EntitlementsSignatureError."""
    _sk_real, entry_real = _make_keypair()
    sk_other, _entry_other = _make_keypair()
    # Sign the token with sk_other but advertise the public component of sk_real.
    token, _ = _make_jwt(signing_key=sk_other)
    advertised_jwks = {"keys": [entry_real]}

    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        return_value=httpx.Response(200, json={"jwt": token})
    )
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json=advertised_jwks)
    )

    client = _client(tmp_path)
    with pytest.raises(EntitlementsSignatureError) as info:
        client.fetch_remote("user-bearer-token")
    assert info.value.code == "carl.entitlements.signature_invalid"


@respx.mock
def test_offline_grace_serves_cache_within_24h(tmp_path: Path) -> None:
    """Cache fresh-but-stale (TTL elapsed, within 24h grace) + network down → cached."""
    sk, entry = _make_keypair()
    token, _ = _make_jwt(signing_key=sk)
    jwks = {"keys": [entry]}

    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        return_value=httpx.Response(200, json={"jwt": token})
    )
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json=jwks)
    )

    client = _client(tmp_path)
    seeded = client.fetch_remote("bearer")
    assert seeded.tier == "PAID"

    # Move the cached_at timestamp into the past so the cache is stale-by-TTL
    # but still inside the 24h offline-grace window.
    cache_path = tmp_path / "entitlements_cache.json"
    raw = json.loads(cache_path.read_text())
    raw["cached_at"] = (
        datetime.fromtimestamp(time.time() - (60 * 60), tz=timezone.utc).isoformat()
    )
    cache_path.write_text(json.dumps(raw))

    # Now make the network throw on the entitlements endpoint.
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        side_effect=httpx.ConnectError("simulated outage")
    )
    cached = client.fetch_remote("bearer", force=True)
    assert cached.tier == "PAID"
    assert cached.key_id == "cc-test-1"


@respx.mock
def test_offline_grace_expires_after_24h_raises(tmp_path: Path) -> None:
    """Cache older than offline-grace + network down → raises NetworkError."""
    sk, entry = _make_keypair()
    token, _ = _make_jwt(signing_key=sk)
    jwks = {"keys": [entry]}

    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        return_value=httpx.Response(200, json={"jwt": token})
    )
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json=jwks)
    )

    client = _client(tmp_path)
    client.fetch_remote("bearer")

    cache_path = tmp_path / "entitlements_cache.json"
    raw = json.loads(cache_path.read_text())
    # 25h ago — past the 24h grace window.
    raw["cached_at"] = (
        datetime.fromtimestamp(
            time.time() - (25 * 60 * 60), tz=timezone.utc
        ).isoformat()
    )
    cache_path.write_text(json.dumps(raw))

    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        side_effect=httpx.ConnectError("still offline")
    )
    with pytest.raises(EntitlementsNetworkError) as info:
        client.fetch_remote("bearer", force=True)
    assert info.value.code == "carl.entitlements.network_unavailable"


def test_cache_corrupt_returns_none_then_refetches(tmp_path: Path) -> None:
    """Corrupt cache JSON raises EntitlementsCacheError and the file is quarantined."""
    cache_path = tmp_path / "entitlements_cache.json"
    cache_path.write_text("{not really json")

    client = _client(tmp_path)

    with pytest.raises(EntitlementsCacheError) as info:
        client.cache_get()
    assert info.value.code == "carl.entitlements.cache_corrupt"

    # Live cache path must have been moved aside.
    assert not cache_path.exists()
    quarantines = list(tmp_path.glob("entitlements_cache.json.corrupt-*.json"))
    assert len(quarantines) == 1


@respx.mock
def test_jwks_kid_unknown_raises_jwks_stale(tmp_path: Path) -> None:
    """Token kid not present in fetched JWKS → JWKSStaleError."""
    sk_signing, _ = _make_keypair(kid="cc-rotated")
    token, _ = _make_jwt(signing_key=sk_signing, kid="cc-rotated")

    # JWKS only advertises a different kid.
    _other_sk, other_entry = _make_keypair(kid="cc-other")
    advertised = {"keys": [other_entry]}

    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        return_value=httpx.Response(200, json={"jwt": token})
    )
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json=advertised)
    )

    client = _client(tmp_path)
    with pytest.raises(JWKSStaleError) as info:
        client.fetch_remote("bearer")
    assert info.value.code == "carl.entitlements.jwks_stale"


@respx.mock
def test_jwt_iat_skew_5min_rejects_expired_outside_window(tmp_path: Path) -> None:
    """exp 6 min in the past + 5-min skew → still expired."""
    sk, entry = _make_keypair()
    # exp_offset_s = -360 → exp is 6 minutes in the past.
    token, _ = _make_jwt(signing_key=sk, exp_offset_s=-360)
    jwks = {"keys": [entry]}

    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        return_value=httpx.Response(200, json={"jwt": token})
    )
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json=jwks)
    )

    client = _client(tmp_path)
    with pytest.raises(EntitlementsSignatureError) as info:
        client.fetch_remote("bearer")
    assert info.value.code == "carl.entitlements.signature_invalid"
    assert "expired" in str(info.value).lower()


@respx.mock
def test_cache_set_writes_mode_0600(tmp_path: Path) -> None:
    """After cache_set, file permissions are 0600 (rw owner only)."""
    sk, entry = _make_keypair()
    token, _ = _make_jwt(signing_key=sk)
    jwks = {"keys": [entry]}

    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        return_value=httpx.Response(200, json={"jwt": token})
    )
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json=jwks)
    )

    client = _client(tmp_path)
    client.fetch_remote("bearer")

    cache_path = tmp_path / "entitlements_cache.json"
    assert cache_path.exists()
    mode = stat.S_IMODE(os.stat(cache_path).st_mode)
    assert mode == 0o600


@respx.mock
def test_jwks_pofu_rejects_silent_key_swap(tmp_path: Path) -> None:
    """Cached JWKS has kid A; refresh returns ONLY kid B → JWKSStaleError."""
    sk_a, entry_a = _make_keypair(kid="kid-a")
    sk_b, entry_b = _make_keypair(kid="kid-b")
    token_a, _ = _make_jwt(signing_key=sk_a, kid="kid-a")

    # Round 1: JWKS advertises kid-a; client caches it.
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json={"keys": [entry_a]})
    )
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        return_value=httpx.Response(200, json={"jwt": token_a})
    )

    client = _client(tmp_path)
    client.fetch_remote("bearer")

    # Round 2: silent swap — JWKS now ONLY advertises kid-b.
    # Force a fresh JWKS fetch by expiring the JWKS cache.
    jwks_cache = tmp_path / "jwks_cache.json"
    raw = json.loads(jwks_cache.read_text())
    raw["cached_at"] = time.time() - (3 * 60 * 60)  # 3h ago, past 1h TTL
    jwks_cache.write_text(json.dumps(raw))

    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json={"keys": [entry_b]})
    )

    with pytest.raises(JWKSStaleError) as info:
        client.fetch_jwks()
    assert info.value.code == "carl.entitlements.jwks_stale"
    assert "silent swap" in str(info.value).lower()
    # Sanity: sk_b is referenced by the new entry.
    assert entry_b["x"] == base64.urlsafe_b64encode(bytes(sk_b.verify_key)).rstrip(b"=").decode()


@respx.mock
def test_jwks_pofu_accepts_additive_rotation(tmp_path: Path) -> None:
    """Cached JWKS has kid A; refresh returns kid A + kid B → accepts."""
    sk_a, entry_a = _make_keypair(kid="kid-a")
    _sk_b, entry_b = _make_keypair(kid="kid-b")
    token_a, _ = _make_jwt(signing_key=sk_a, kid="kid-a")

    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json={"keys": [entry_a]})
    )
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{ENTITLEMENTS_PATH}").mock(
        return_value=httpx.Response(200, json={"jwt": token_a})
    )

    client = _client(tmp_path)
    client.fetch_remote("bearer")

    # Force fresh JWKS fetch.
    jwks_cache = tmp_path / "jwks_cache.json"
    raw = json.loads(jwks_cache.read_text())
    raw["cached_at"] = time.time() - (3 * 60 * 60)
    jwks_cache.write_text(json.dumps(raw))

    # Round 2: additive — both kid-a AND kid-b advertised.
    additive = {"keys": [entry_a, entry_b]}
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{JWKS_PATH}").mock(
        return_value=httpx.Response(200, json=additive)
    )

    fetched = client.fetch_jwks()
    kids = {k["kid"] for k in fetched["keys"]}
    assert kids == {"kid-a", "kid-b"}


# ---------------------------------------------------------------------------
# Lightweight sanity: import-time cheapness — see Self-Review checklist.
# ---------------------------------------------------------------------------


def test_module_import_does_not_pull_pynacl_eagerly() -> None:
    """``import carl_studio.entitlements`` must not import pynacl at module load.

    pynacl is only required to verify a JWT signature — callers that just
    need the typed models or error codes shouldn't pay that cost.
    """
    import importlib
    import sys

    # Drop both modules so a clean reimport reflects current state.
    for mod_name in list(sys.modules):
        if mod_name == "carl_studio.entitlements" or mod_name.startswith("nacl"):
            sys.modules.pop(mod_name, None)

    importlib.import_module("carl_studio.entitlements")
    nacl_loaded = any(name == "nacl" or name.startswith("nacl.") for name in sys.modules)
    assert not nacl_loaded, (
        "carl_studio.entitlements pulled pynacl at import time; verify_jwt() "
        "should import it lazily."
    )


# Defensive: keeps pyright happy on Any imports used only inside fixture helpers.
_unused_any: Any = None
_ = _unused_any
