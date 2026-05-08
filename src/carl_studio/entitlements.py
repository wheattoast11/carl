"""carl-studio remote entitlements client for v0.10 tier verification.

Fetches the Ed25519-signed JWT from carl.camp's ``GET /api/platform/entitlements``,
verifies the signature against the published JWKS at
``/.well-known/carl-camp-jwks.json``, and caches the parsed result locally
with a 15-minute fresh-cache TTL and a 24h offline-grace window.

The local-fast-path doctrine (per CLAUDE.md AP-1): ``tier_gate``'s existing
local check runs FIRST. :func:`fetch_remote_async` fires after a successful
local allow to verify in the background; only blocks if local denies a
paid feature and we need the cached grant to override.

Cryptography
------------

* Ed25519 signature verification via ``pynacl`` (``[constitutional]`` extra).
* JWT segment parsing with ``base64url`` (Python stdlib + a tiny pad helper).
* JWKS pin-on-first-use: a SHA-256 fingerprint of the keys array is stored
  on first fetch. A subsequent fetch that drops every previously-seen kid
  raises :class:`JWKSStaleError`; additive rotations (old kid + new kid)
  are accepted silently.
* 5-minute clock-skew tolerance on ``exp``/``nbf`` per the JOSE
  best-practice.

Caching
-------

* Cache file lives at ``~/.carl/entitlements_cache.json`` (mode ``0600``).
* Atomic writes via ``tmp + os.replace``; permissions set on tmp BEFORE
  the data is written so umask leaks aren't a window.
* JWKS cache lives at ``~/.carl/jwks_cache.json``; reused across
  verification calls.

This module keeps imports light at module load — ``pynacl`` is imported
lazily inside :meth:`EntitlementsClient.verify_jwt` so that
``import carl_studio.entitlements`` stays cheap for callers who only
need the typed models or error codes.
"""

from __future__ import annotations

import atexit
import base64
import concurrent.futures
import contextlib
import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from carl_core.errors import (
    EntitlementsCacheError,
    EntitlementsNetworkError,
    EntitlementsSignatureError,
    JWKSStaleError,
)
from carl_core.interaction import ActionType, InteractionChain
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DEFAULT_CARL_CAMP_BASE: str = "https://carl.camp"
ENTITLEMENTS_PATH: str = "/api/platform/entitlements"
JWKS_PATH: str = "/.well-known/carl-camp-jwks.json"

DEFAULT_CACHE_TTL_S: int = 15 * 60          # 15 min — matches JWT exp.
DEFAULT_OFFLINE_GRACE_S: int = 24 * 60 * 60  # 24 h — soft fall-back during outages.
DEFAULT_JWKS_TTL_S: int = 60 * 60            # 1 h — JWKS rotation is rare.

CACHE_FILE_MODE: int = 0o600                 # private; never let other users read.

# Clock-skew tolerance applied to ``exp``/``nbf`` checks (JOSE best practice).
_CLOCK_SKEW_S: int = 5 * 60

_TIER_VALUES: tuple[str, ...] = ("FREE", "PAID")
_TIER_LABELS: tuple[str, ...] = ("free", "managed_payg", "managed_lodge")


# ---------------------------------------------------------------------------
# Typed models
# ---------------------------------------------------------------------------


class EntitlementGrant(BaseModel):
    """A single ``feature_key + grant_timestamp`` row from the platform JWT."""

    key: str
    granted_at: datetime


class Entitlements(BaseModel):
    """Parsed, signature-verified entitlements snapshot.

    ``cached_at`` is when the studio retrieved + verified this JWT;
    ``expires_at`` is the JWT's own ``exp`` claim. The studio uses
    ``cached_at`` for offline-grace math (it survives clock skew on the
    server side) and ``expires_at`` for fresh-cache TTL.
    """

    tier: Literal["FREE", "PAID"]
    tier_label: Literal["free", "managed_payg", "managed_lodge"]
    entitlements: list[EntitlementGrant] = Field(default_factory=lambda: [])
    cached_at: datetime
    expires_at: datetime
    key_id: str
    org_id: str | None = None
    sub: str | None = None
    jwt: str  # raw token retained for replay/audit; not redacted because it's verifiable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64url_decode(s: str) -> bytes:
    """Decode a base64url-encoded string, handling missing padding."""
    padding = (-len(s)) % 4
    if padding:
        s = s + ("=" * padding)
    return base64.urlsafe_b64decode(s.encode("ascii"))


def _utc_from_epoch(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _jwks_fingerprint(keys: list[dict[str, Any]]) -> str:
    """Deterministic SHA-256 over the JWKS keys array.

    Sorted-key JSON ensures the digest is stable regardless of key order.
    """
    payload = json.dumps(keys, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _kid_set(keys: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for entry in keys:
        kid = entry.get("kid")
        if isinstance(kid, str):
            out.add(kid)
    return out


def _record_step(
    chain: InteractionChain | None,
    name: str,
    *,
    output: dict[str, Any] | None = None,
    success: bool = True,
) -> None:
    """Best-effort step record; failures never propagate to the caller."""
    if chain is None:
        return
    try:
        chain.record(
            ActionType.EXTERNAL,
            name,
            output=output,
            success=success,
        )
    except Exception:  # pragma: no cover - defensive
        pass


# ---------------------------------------------------------------------------
# EntitlementsClient
# ---------------------------------------------------------------------------


class EntitlementsClient:
    """Fetches, verifies, and caches v0.10 entitlements from carl.camp.

    Thread-safe for read; writes serialized via the file-replace primitive
    — last-writer-wins on disk. Network errors fall through to the
    offline-grace cache (24h) before being surfaced as
    :class:`EntitlementsNetworkError`.

    The client is cheap to construct — it does no work at ``__init__`` —
    so a process-wide singleton from :func:`default_client` is fine.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        cache_path: Path | None = None,
        jwks_path_local: Path | None = None,
        cache_ttl_s: int = DEFAULT_CACHE_TTL_S,
        offline_grace_s: int = DEFAULT_OFFLINE_GRACE_S,
        jwks_ttl_s: int = DEFAULT_JWKS_TTL_S,
        http: httpx.Client | None = None,
    ) -> None:
        env_base = os.environ.get("CARL_CAMP_BASE")
        self._base_url = (base_url or env_base or DEFAULT_CARL_CAMP_BASE).rstrip("/")
        self._cache_path = cache_path or (Path.home() / ".carl" / "entitlements_cache.json")
        self._jwks_path_local = jwks_path_local or (Path.home() / ".carl" / "jwks_cache.json")
        self._cache_ttl_s = int(cache_ttl_s)
        self._offline_grace_s = int(offline_grace_s)
        self._jwks_ttl_s = int(jwks_ttl_s)
        self._http: httpx.Client = http or httpx.Client(timeout=httpx.Timeout(10.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_remote(
        self,
        jwt: str,
        *,
        force: bool = False,
        chain: InteractionChain | None = None,
    ) -> Entitlements:
        """Return signed/verified entitlements, refreshing as needed.

        Decision ladder:

        1. ``force=False`` AND fresh cache (within ``cache_ttl_s``) → cache.
        2. Network reachable + JWKS resolvable → fetch + verify + cache.
        3. Network failure within offline-grace window → cache (with audit).
        4. Cache absent / stale beyond grace → :class:`EntitlementsNetworkError`.
        """
        if not jwt:
            raise EntitlementsSignatureError(
                "Bearer JWT required for /api/platform/entitlements.",
                context={"hint": "run: carl camp login"},
            )

        if not force:
            cached = self.cache_get()
            if cached is not None and self._cache_is_fresh(cached):
                return cached

        url = f"{self._base_url}{ENTITLEMENTS_PATH}"
        try:
            response = self._http.get(
                url,
                headers={
                    "Authorization": f"Bearer {jwt}",
                    "Accept": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            return self._fall_back_to_cache(
                reason=f"transport: {exc!r}", chain=chain, cause=exc
            )

        status = int(getattr(response, "status_code", 0) or 0)
        if status >= 400:
            return self._fall_back_to_cache(
                reason=f"http_{status}", chain=chain, cause=None
            )

        try:
            payload: Any = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise EntitlementsNetworkError(
                "carl.camp returned a non-JSON entitlements payload.",
                context={"status": status},
                cause=exc,
            ) from exc

        if not isinstance(payload, dict):
            raise EntitlementsNetworkError(
                "carl.camp returned an entitlements payload that wasn't an object.",
                context={"status": status},
            )

        body = cast(dict[str, Any], payload)
        # carl.camp's GET /api/platform/entitlements returns {ok, token, ...}.
        # Accept the modern `token` field first; fall back to `jwt` /
        # `access_token` for forward-compat with future renames or any
        # alternative deployment that surfaces a different name.
        token = body.get("token") or body.get("jwt") or body.get("access_token")
        if not isinstance(token, str) or not token:
            raise EntitlementsNetworkError(
                "carl.camp entitlements payload missing 'token' field.",
                context={"status": status},
            )

        jwks = self.fetch_jwks()
        claims = self.verify_jwt(token, jwks)
        ent = self._claims_to_entitlements(claims, raw_token=token)
        self.cache_set(ent)
        _record_step(
            chain,
            "entitlements.fetch",
            output={
                "tier": ent.tier,
                "tier_label": ent.tier_label,
                "entitlement_count": len(ent.entitlements),
                "key_id": ent.key_id,
                "source": "remote",
            },
            success=True,
        )
        return ent

    def verify_jwt(self, token: str, jwks: dict[str, Any]) -> dict[str, Any]:
        """Verify an Ed25519-signed JWT against the supplied JWKS.

        Returns the decoded payload dict on success. Raises
        :class:`EntitlementsSignatureError` on:

        * malformed token (wrong segment count, undecodable segments)
        * ``alg != EdDSA`` in header
        * ``kid`` not in the JWKS keys list
        * key not Ed25519 (``kty != "OKP"`` or ``crv != "Ed25519"``)
        * signature verification failure
        * ``exp`` past + skew or ``nbf`` future + skew
        """
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey

        if token.count(".") != 2:
            raise EntitlementsSignatureError(
                "Malformed JWT: expected 3 dot-separated segments.",
                context={"segments": token.count(".")},
            )

        h_b64, p_b64, s_b64 = token.split(".")
        try:
            header_raw = _b64url_decode(h_b64)
            payload_raw = _b64url_decode(p_b64)
            sig_bytes = _b64url_decode(s_b64)
        except ValueError as exc:
            raise EntitlementsSignatureError(
                "Malformed JWT: base64url decode failed.",
                cause=exc,
            ) from exc

        try:
            header_any: Any = json.loads(header_raw)
            claims_any: Any = json.loads(payload_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise EntitlementsSignatureError(
                "Malformed JWT: header or payload was not valid JSON.",
                cause=exc,
            ) from exc

        if not isinstance(header_any, dict) or not isinstance(claims_any, dict):
            raise EntitlementsSignatureError(
                "Malformed JWT: header or payload was not a JSON object.",
            )
        header = cast(dict[str, Any], header_any)
        claims = cast(dict[str, Any], claims_any)

        alg = header.get("alg")
        if alg != "EdDSA":
            raise EntitlementsSignatureError(
                f"Unsupported JWT alg '{alg}'; expected 'EdDSA'.",
                context={"alg": alg},
            )

        kid_any = header.get("kid")
        if not isinstance(kid_any, str) or not kid_any:
            raise EntitlementsSignatureError(
                "JWT header missing 'kid' — cannot select verification key.",
            )
        kid: str = kid_any

        keys_raw = jwks.get("keys")
        if not isinstance(keys_raw, list):
            raise JWKSStaleError(
                "JWKS payload has no 'keys' array.",
                context={"kid_requested": kid},
            )
        keys = cast(list[Any], keys_raw)

        match: dict[str, Any] | None = None
        for entry in keys:
            if not isinstance(entry, dict):
                continue
            entry_dict = cast(dict[str, Any], entry)
            if entry_dict.get("kid") == kid:
                match = entry_dict
                break

        if match is None:
            raise JWKSStaleError(
                f"JWT signed with kid '{kid}' but local JWKS has no matching key.",
                context={"kid_requested": kid, "kid_known": sorted(_kid_set(keys))},
            )

        if match.get("kty") != "OKP" or match.get("crv") != "Ed25519":
            raise EntitlementsSignatureError(
                "JWKS entry kty/crv mismatch — expected OKP/Ed25519.",
                context={
                    "kid": kid,
                    "kty": match.get("kty"),
                    "crv": match.get("crv"),
                },
            )

        if match.get("alg") not in (None, "EdDSA"):
            raise EntitlementsSignatureError(
                "JWKS entry advertises a non-EdDSA algorithm.",
                context={"kid": kid, "alg": match.get("alg")},
            )

        x_field = match.get("x")
        if not isinstance(x_field, str):
            raise EntitlementsSignatureError(
                "JWKS entry missing 'x' base64url public-key field.",
                context={"kid": kid},
            )

        try:
            pubkey_raw = _b64url_decode(x_field)
        except ValueError as exc:
            raise EntitlementsSignatureError(
                "JWKS entry 'x' is not valid base64url.",
                context={"kid": kid},
                cause=exc,
            ) from exc

        if len(pubkey_raw) != 32:
            raise EntitlementsSignatureError(
                "JWKS entry 'x' must decode to a 32-byte Ed25519 public key.",
                context={"kid": kid, "x_bytes": len(pubkey_raw)},
            )

        signing_input = f"{h_b64}.{p_b64}".encode("ascii")
        try:
            VerifyKey(pubkey_raw).verify(signing_input, sig_bytes)
        except BadSignatureError as exc:
            raise EntitlementsSignatureError(
                "JWT signature did not verify against the JWKS public key.",
                context={"kid": kid},
                cause=exc,
            ) from exc

        # Time-bound checks with a small skew tolerance.
        now = int(time.time())
        exp_claim = claims.get("exp")
        if isinstance(exp_claim, (int, float)) and int(exp_claim) + _CLOCK_SKEW_S < now:
            raise EntitlementsSignatureError(
                "JWT expired (with 5-min clock-skew tolerance applied).",
                context={"exp": int(exp_claim), "now": now},
            )
        nbf_claim = claims.get("nbf")
        if isinstance(nbf_claim, (int, float)) and int(nbf_claim) - _CLOCK_SKEW_S > now:
            raise EntitlementsSignatureError(
                "JWT not yet valid (nbf in the future).",
                context={"nbf": int(nbf_claim), "now": now},
            )

        # Stash the kid we just verified against — useful for cache provenance.
        claims["__verified_kid"] = kid
        return claims

    def fetch_jwks(self, *, force: bool = False) -> dict[str, Any]:
        """Return the JWKS dict, fetching from carl.camp when stale.

        Caches at ``~/.carl/jwks_cache.json`` for ``jwks_ttl_s`` seconds.
        Pin-on-first-use:

        * On first fetch, store ``sha256(json.dumps(keys, sort_keys=True))``
          alongside the cached payload.
        * On subsequent fetches that change the fingerprint, *every*
          previously-known kid must still be present (additive rotation).
          A "silent swap" — every old kid disappears in one fetch — raises
          :class:`JWKSStaleError`.

        ``force=True`` bypasses both the TTL and the rotation check.
        """
        if not force:
            cached = self._jwks_load_cached()
            if cached is not None and self._jwks_is_fresh(cached):
                cached_payload = cached.get("payload")
                if isinstance(cached_payload, dict):
                    return cast(dict[str, Any], cached_payload)

        url = f"{self._base_url}{JWKS_PATH}"
        try:
            response = self._http.get(url, headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            cached = self._jwks_load_cached()
            if cached is not None:
                cached_payload = cached.get("payload")
                if isinstance(cached_payload, dict):
                    return cast(dict[str, Any], cached_payload)
            raise JWKSStaleError(
                "JWKS fetch failed and no cached JWKS is available.",
                context={"url": url},
                cause=exc,
            ) from exc

        status = int(getattr(response, "status_code", 0) or 0)
        if status >= 400:
            cached = self._jwks_load_cached()
            if cached is not None:
                cached_payload = cached.get("payload")
                if isinstance(cached_payload, dict):
                    return cast(dict[str, Any], cached_payload)
            raise JWKSStaleError(
                f"JWKS endpoint returned HTTP {status}.",
                context={"url": url, "status": status},
            )

        try:
            payload_any: Any = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise JWKSStaleError(
                "JWKS endpoint returned a non-JSON body.",
                context={"url": url},
                cause=exc,
            ) from exc
        if not isinstance(payload_any, dict):
            raise JWKSStaleError(
                "JWKS endpoint returned a JSON value that wasn't an object.",
                context={"url": url},
            )
        payload = cast(dict[str, Any], payload_any)

        keys_any = payload.get("keys")
        if not isinstance(keys_any, list):
            raise JWKSStaleError(
                "JWKS payload missing 'keys' array.",
                context={"url": url},
            )
        keys = cast(list[Any], keys_any)
        keys_normalized: list[dict[str, Any]] = [
            cast(dict[str, Any], k) for k in keys if isinstance(k, dict)
        ]

        new_fp = _jwks_fingerprint(keys_normalized)
        new_kids = _kid_set(keys_normalized)

        if not force:
            cached = self._jwks_load_cached()
            if cached is not None:
                cached_payload_any = cached.get("payload")
                cached_keys: list[dict[str, Any]] = []
                if isinstance(cached_payload_any, dict):
                    raw_keys = cast(dict[str, Any], cached_payload_any).get("keys")
                    if isinstance(raw_keys, list):
                        cached_keys = [
                            cast(dict[str, Any], k)
                            for k in cast(list[Any], raw_keys)
                            if isinstance(k, dict)
                        ]
                cached_kids = _kid_set(cached_keys)
                cached_fp = cached.get("fingerprint")
                if cached_kids and cached_fp != new_fp:
                    if cached_kids and cached_kids.isdisjoint(new_kids):
                        raise JWKSStaleError(
                            "JWKS rotation rejected: every previously-known kid "
                            "vanished in a single fetch (silent swap).",
                            context={
                                "kid_known": sorted(cached_kids),
                                "kid_received": sorted(new_kids),
                            },
                        )

        self._jwks_save(payload, fingerprint=new_fp)
        return payload

    # ------------------------------------------------------------------
    # Cache primitives
    # ------------------------------------------------------------------

    def cache_get(self) -> Entitlements | None:
        """Return cached entitlements, or ``None`` on miss.

        A corrupt cache (parse error, missing fields) raises
        :class:`EntitlementsCacheError`. The corrupt file is moved aside
        to ``<name>.corrupt-<timestamp>.json`` for forensic review and
        removed from the live path so the next fetch can repopulate it.
        """
        path = self._cache_path
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise EntitlementsCacheError(
                "Cache file unreadable.",
                context={"path": str(path)},
                cause=exc,
            ) from exc

        try:
            obj_any: Any = json.loads(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            self._quarantine_cache()
            raise EntitlementsCacheError(
                "Cache file contained invalid JSON.",
                context={"path": str(path)},
                cause=exc,
            ) from exc

        if not isinstance(obj_any, dict):
            self._quarantine_cache()
            raise EntitlementsCacheError(
                "Cache file root was not a JSON object.",
                context={"path": str(path)},
            )

        try:
            return Entitlements.model_validate(obj_any)
        except Exception as exc:
            self._quarantine_cache()
            raise EntitlementsCacheError(
                "Cache file did not match the Entitlements schema.",
                context={"path": str(path)},
                cause=exc,
            ) from exc

    def cache_set(self, ent: Entitlements) -> None:
        """Persist entitlements to ``cache_path`` with mode 0600 atomically.

        Mode is set on the temporary file BEFORE it's written so a crash
        between create and chmod cannot leave a world-readable secret on
        disk. Final placement is ``os.replace`` for atomicity.
        """
        path = self._cache_path
        path.parent.mkdir(parents=True, exist_ok=True)

        token = secrets.token_hex(8)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{token}")

        # Pre-create with strict mode, then write — never leaks at default umask.
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, CACHE_FILE_MODE)
        try:
            os.write(fd, ent.model_dump_json().encode("utf-8"))
        finally:
            os.close(fd)

        try:
            os.chmod(tmp, CACHE_FILE_MODE)
            os.replace(tmp, path)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def is_offline_grace_valid(self, ent: Entitlements) -> bool:
        """True when ``cached_at`` is within ``offline_grace_s`` of now."""
        cached_epoch = ent.cached_at.timestamp()
        return (time.time() - cached_epoch) <= self._offline_grace_s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_is_fresh(self, ent: Entitlements) -> bool:
        cached_epoch = ent.cached_at.timestamp()
        return (time.time() - cached_epoch) <= self._cache_ttl_s

    def _claims_to_entitlements(
        self,
        claims: dict[str, Any],
        *,
        raw_token: str,
    ) -> Entitlements:
        tier_raw = claims.get("tier")
        if tier_raw not in _TIER_VALUES:
            raise EntitlementsSignatureError(
                "JWT 'tier' claim must be 'FREE' or 'PAID'.",
                context={"tier": tier_raw},
            )
        tier_label_raw = claims.get("tier_label", "free")
        if tier_label_raw not in _TIER_LABELS:
            raise EntitlementsSignatureError(
                "JWT 'tier_label' claim is not in the canonical set.",
                context={"tier_label": tier_label_raw},
            )

        raw_grants_any: Any = claims.get("entitlements") or []
        if not isinstance(raw_grants_any, list):
            raise EntitlementsSignatureError(
                "JWT 'entitlements' claim must be a list.",
                context={"type": type(raw_grants_any).__name__},
            )
        ent_grants_list = cast(list[Any], raw_grants_any)

        grants: list[EntitlementGrant] = []
        for raw_grant in ent_grants_list:
            grant: dict[str, Any]
            if isinstance(raw_grant, dict):
                grant = cast(dict[str, Any], raw_grant)
            elif isinstance(raw_grant, str):
                grant = {"key": raw_grant, "granted_at": int(time.time())}
            else:
                raise EntitlementsSignatureError(
                    "JWT 'entitlements' entries must be objects or feature-key strings.",
                    context={"type": type(raw_grant).__name__},
                )
            key = grant.get("key") or grant.get("feature_key") or grant.get("name")
            if not isinstance(key, str) or not key:
                raise EntitlementsSignatureError(
                    "Entitlement entry missing 'key' field.",
                )
            granted_raw = grant.get("granted_at") or grant.get("issued_at") or int(time.time())
            granted_at = self._coerce_timestamp(granted_raw)
            grants.append(EntitlementGrant(key=key, granted_at=granted_at))

        exp_claim = claims.get("exp")
        if not isinstance(exp_claim, (int, float)):
            raise EntitlementsSignatureError(
                "JWT missing or non-numeric 'exp' claim.",
                context={"exp": exp_claim},
            )
        expires_at = _utc_from_epoch(float(exp_claim))

        kid = claims.get("__verified_kid")
        if not isinstance(kid, str) or not kid:
            raise EntitlementsSignatureError(
                "verify_jwt() did not stamp '__verified_kid' on the claims.",
            )

        org_raw = claims.get("org_id")
        sub_raw = claims.get("sub")
        org_id = org_raw if isinstance(org_raw, str) else None
        sub = sub_raw if isinstance(sub_raw, str) else None

        return Entitlements(
            tier=cast(Literal["FREE", "PAID"], tier_raw),
            tier_label=cast(
                Literal["free", "managed_payg", "managed_lodge"], tier_label_raw
            ),
            entitlements=grants,
            cached_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            key_id=kid,
            org_id=org_id,
            sub=sub,
            jwt=raw_token,
        )

    @staticmethod
    def _coerce_timestamp(raw: Any) -> datetime:
        if isinstance(raw, (int, float)):
            return _utc_from_epoch(float(raw))
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise EntitlementsSignatureError(
                    "Could not parse entitlement timestamp.",
                    context={"value": raw},
                    cause=exc,
                ) from exc
        raise EntitlementsSignatureError(
            "Entitlement timestamp must be a number or ISO string.",
            context={"type": type(raw).__name__},
        )

    def _fall_back_to_cache(
        self,
        *,
        reason: str,
        chain: InteractionChain | None,
        cause: BaseException | None,
    ) -> Entitlements:
        cached = None
        with contextlib.suppress(EntitlementsCacheError):
            cached = self.cache_get()
        if cached is not None and self.is_offline_grace_valid(cached):
            _record_step(
                chain,
                "entitlements.fetch",
                output={
                    "tier": cached.tier,
                    "tier_label": cached.tier_label,
                    "entitlement_count": len(cached.entitlements),
                    "key_id": cached.key_id,
                    "source": "cache_grace",
                    "reason": reason,
                },
                success=True,
            )
            return cached
        raise EntitlementsNetworkError(
            "carl.camp /api/platform/entitlements is unreachable and no "
            "cached entitlements within the offline-grace window are "
            "available.",
            context={"reason": reason},
            cause=cause,
        )

    # JWKS cache helpers ------------------------------------------------

    def _jwks_load_cached(self) -> dict[str, Any] | None:
        path = self._jwks_path_local
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            obj_any: Any = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return None
        if not isinstance(obj_any, dict):
            return None
        return cast(dict[str, Any], obj_any)

    def _jwks_is_fresh(self, cached: dict[str, Any]) -> bool:
        ts = cached.get("cached_at")
        if not isinstance(ts, (int, float)):
            return False
        return (time.time() - float(ts)) <= self._jwks_ttl_s

    def _jwks_save(self, payload: dict[str, Any], *, fingerprint: str) -> None:
        path = self._jwks_path_local
        path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "payload": payload,
            "fingerprint": fingerprint,
            "cached_at": time.time(),
        }
        token = secrets.token_hex(8)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{token}")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, CACHE_FILE_MODE)
        try:
            os.write(fd, json.dumps(record).encode("utf-8"))
        finally:
            os.close(fd)
        try:
            os.chmod(tmp, CACHE_FILE_MODE)
            os.replace(tmp, path)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _quarantine_cache(self) -> None:
        path = self._cache_path
        if not path.exists():
            return
        ts = int(time.time())
        target = path.with_name(f"{path.name}.corrupt-{ts}.json")
        with contextlib.suppress(OSError):
            os.replace(path, target)


# ---------------------------------------------------------------------------
# Module-level singleton + background fetch executor
# ---------------------------------------------------------------------------


_default_client: EntitlementsClient | None = None
_bg_executor: concurrent.futures.ThreadPoolExecutor | None = None


def default_client() -> EntitlementsClient:
    """Return (or lazily build) the process-wide default client."""
    global _default_client
    if _default_client is None:
        _default_client = EntitlementsClient()
    return _default_client


def _shutdown_executor() -> None:  # pragma: no cover - atexit hook
    global _bg_executor
    executor = _bg_executor
    if executor is not None:
        _bg_executor = None
        with contextlib.suppress(Exception):
            executor.shutdown(wait=False)


def fetch_remote_async(
    jwt: str,
    *,
    chain: InteractionChain | None = None,
) -> concurrent.futures.Future[Entitlements]:
    """Schedule :meth:`EntitlementsClient.fetch_remote` on a background worker.

    Returns the :class:`~concurrent.futures.Future`; callers may discard
    it. The fast-path tier-gate caller never blocks on the returned
    future and never sees its exceptions — the future captures them
    instead. Callers that want an error signal can attach
    :meth:`Future.add_done_callback` and inspect ``future.exception()``.
    """
    global _bg_executor
    if _bg_executor is None:
        _bg_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="entitlements"
        )
        atexit.register(_shutdown_executor)
    return _bg_executor.submit(default_client().fetch_remote, jwt, chain=chain)


# Test-only helper: drop the singleton so each test can install its own.
def reset_default_client() -> None:
    """Reset the process-wide singleton + shut down the bg executor.

    Test-only seam — production code should never need this. Public
    (rather than ``_reset_default_client``) so tests don't fire
    ``reportPrivateUsage`` under pyright strict mode.
    """
    global _default_client, _bg_executor
    _default_client = None
    if _bg_executor is not None:
        with contextlib.suppress(Exception):
            _bg_executor.shutdown(wait=False)
        _bg_executor = None


__all__ = [
    "DEFAULT_CARL_CAMP_BASE",
    "ENTITLEMENTS_PATH",
    "JWKS_PATH",
    "DEFAULT_CACHE_TTL_S",
    "DEFAULT_OFFLINE_GRACE_S",
    "DEFAULT_JWKS_TTL_S",
    "CACHE_FILE_MODE",
    "Entitlements",
    "EntitlementGrant",
    "EntitlementsClient",
    "default_client",
    "fetch_remote_async",
    "reset_default_client",
]
