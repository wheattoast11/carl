"""``carl entitlements`` — inspect cached v0.10 platform entitlements.

User-facing surface for the entitlements cache built by S1a / S1b. The
single subcommand here (`show`) lets operators see what carl.camp last
told us about their tier without having to attempt a remote-verified
gated feature.

Status taxonomy
---------------

Returned by both this command and the ``entitlements`` block in
``carl doctor``:

* ``fresh``                  — ``cached_at`` within 15min of now (cache is
                                canonical; no fetch needed).
* ``stale``                  — ``cached_at`` between 15min and 1h
                                (network may help; cache is still trusted).
* ``offline_grace``          — ``cached_at`` between 1h and 24h
                                (network is unreachable; cache is the
                                source of truth per offline-grace
                                doctrine).
* ``offline_grace_expired``  — ``cached_at`` > 24h ago (cache rejected;
                                a network fetch is mandatory before
                                the next gated call).
* ``missing``                — no cache file present.
* ``corrupt``                — cache file present but unparseable
                                (will be quarantined on next read).

Imports of ``carl_studio.entitlements`` are kept lazy at the function
level so that ``carl --version`` and ``carl --help`` don't drag the
entitlements client into the import graph.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import typer

# Local convenience constants. Mirroring DEFAULT_CACHE_TTL_S /
# DEFAULT_OFFLINE_GRACE_S from ``carl_studio.entitlements`` would force
# an eager import and defeat the lazy-load doctrine — these are stable
# by spec and unit-tested in S1a, so duplicating two ints is the lesser
# sin.
_FRESH_TTL_S: int = 15 * 60
_STALE_TTL_S: int = 60 * 60
_OFFLINE_GRACE_TTL_S: int = 24 * 60 * 60


entitlements_app = typer.Typer(
    name="entitlements",
    help="Inspect cached v0.10 platform entitlements (carl.camp).",
    no_args_is_help=True,
)


def _classify_status(age_s: int) -> str:
    """Map cache age to a status label using the v0.10 taxonomy."""
    if age_s <= _FRESH_TTL_S:
        return "fresh"
    if age_s <= _STALE_TTL_S:
        return "stale"
    if age_s <= _OFFLINE_GRACE_TTL_S:
        return "offline_grace"
    return "offline_grace_expired"


@entitlements_app.command("show")
def show_cmd(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output the cache snapshot as JSON instead of human-readable text.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Force a fetch from carl.camp before printing (bearer required).",
    ),
) -> None:
    """Print the currently-cached entitlements with age + key_id.

    With ``--refresh``, fetches first and prints whatever the fetch left
    in cache. Without a bearer, ``--refresh`` is best-effort and falls
    through to the existing cache (warning printed to stderr).
    """
    from carl_core.errors import (
        EntitlementsCacheError,
        EntitlementsNetworkError,
        EntitlementsSignatureError,
        JWKSStaleError,
    )
    from carl_studio.entitlements import default_client
    from carl_studio.tier import (
        _resolve_bearer_token_for_verify,  # pyright: ignore[reportPrivateUsage]
    )

    client = default_client()

    if refresh:
        bearer = _resolve_bearer_token_for_verify()
        if not bearer:
            typer.echo(
                "warning: --refresh requested but no bearer token available "
                "(set CARL_CAMP_TOKEN or run `carl camp login`); "
                "showing cached state.",
                err=True,
            )
        else:
            try:
                client.fetch_remote(bearer, force=True)
            except (
                EntitlementsCacheError,
                EntitlementsNetworkError,
                EntitlementsSignatureError,
                JWKSStaleError,
            ) as exc:
                # Best-effort refresh — never escalate a refresh failure
                # past the user's actual request, which is "show me the
                # cache." Surface the reason on stderr so it's visible
                # but doesn't pollute --json stdout.
                typer.echo(f"warning: refresh failed ({exc.code}): {exc}", err=True)

    try:
        ent = client.cache_get()
    except EntitlementsCacheError as exc:
        if json_output:
            typer.echo(
                json.dumps(
                    {"status": "corrupt", "code": exc.code, "error": str(exc)},
                    indent=2,
                )
            )
        else:
            typer.echo(f"entitlements cache: corrupt ({exc.code}) — {exc}", err=True)
        raise typer.Exit(code=1)

    if ent is None:
        if json_output:
            typer.echo(json.dumps({"status": "missing"}, indent=2))
        else:
            typer.echo("entitlements cache: missing")
            typer.echo(
                "  Run `carl camp login` then `carl entitlements show --refresh`."
            )
        return

    age_s = int(
        (datetime.now(tz=timezone.utc) - ent.cached_at).total_seconds()
    )
    status = _classify_status(age_s)

    if json_output:
        payload: dict[str, Any] = {
            "status": status,
            "age_s": age_s,
            "key_id": ent.key_id,
            "tier": ent.tier,
            "tier_label": ent.tier_label,
            "entitlements": [
                {"key": g.key, "granted_at": g.granted_at.isoformat()}
                for g in ent.entitlements
            ],
            "cached_at": ent.cached_at.isoformat(),
            "expires_at": ent.expires_at.isoformat(),
        }
        if ent.org_id is not None:
            payload["org_id"] = ent.org_id
        if ent.sub is not None:
            payload["sub"] = ent.sub
        typer.echo(json.dumps(payload, indent=2))
        return

    # Human-readable rendering — keep it dependency-light. The Resonant
    # CLI uses ``CampConsole`` for header/kv blocks; we mirror that
    # vocabulary so operators recognise the shape, but raw ``typer.echo``
    # works fine on non-TTY too which is the common ``carl ... | jq``
    # pipeline path even without ``--json``.
    typer.echo(f"entitlements cache: {status}")
    typer.echo(f"  tier         : {ent.tier} ({ent.tier_label})")
    typer.echo(f"  key_id       : {ent.key_id}")
    typer.echo(f"  cached_at    : {ent.cached_at.isoformat()}")
    typer.echo(f"  expires_at   : {ent.expires_at.isoformat()}")
    typer.echo(f"  age          : {age_s}s")
    if ent.org_id is not None:
        typer.echo(f"  org_id       : {ent.org_id}")
    if ent.sub is not None:
        typer.echo(f"  sub          : {ent.sub}")
    if ent.entitlements:
        typer.echo(f"  entitlements ({len(ent.entitlements)}):")
        for g in ent.entitlements:
            typer.echo(f"    - {g.key} (granted {g.granted_at.isoformat()})")
    else:
        typer.echo("  entitlements : (none)")


def doctor_entitlements_state() -> dict[str, Any]:
    """Return the entitlements cache state for the ``carl doctor`` payload.

    Defensive: never raises. The doctor block is a side-of-the-road
    informational stanza — if the entitlements module isn't importable
    or the cache is wedged, we surface that as a status string rather
    than letting it tear down ``carl doctor``.

    Always returns a dict with at minimum:

    * ``status``  — one of fresh/stale/offline_grace/offline_grace_expired/
                    missing/corrupt/unavailable
    * ``key_id``  — str or None
    * ``age_s``   — int or None
    """
    try:
        from carl_core.errors import EntitlementsCacheError
        from carl_studio.entitlements import default_client
    except ImportError as exc:
        return {
            "status": "unavailable",
            "key_id": None,
            "age_s": None,
            "reason": f"import failed: {exc}",
        }

    try:
        client = default_client()
    except Exception as exc:  # noqa: BLE001 — doctor must never crash
        return {
            "status": "unavailable",
            "key_id": None,
            "age_s": None,
            "reason": f"client init failed: {exc}",
        }

    try:
        ent = client.cache_get()
    except EntitlementsCacheError as exc:
        return {
            "status": "corrupt",
            "key_id": None,
            "age_s": None,
            "code": exc.code,
        }
    except Exception as exc:  # noqa: BLE001 — defensive
        return {
            "status": "unavailable",
            "key_id": None,
            "age_s": None,
            "reason": str(exc),
        }

    if ent is None:
        return {"status": "missing", "key_id": None, "age_s": None}

    age_s = int(
        (datetime.now(tz=timezone.utc) - ent.cached_at).total_seconds()
    )
    return {
        "status": _classify_status(age_s),
        "age_s": age_s,
        "key_id": ent.key_id,
        "tier": ent.tier,
        "tier_label": ent.tier_label,
    }


__all__ = ["entitlements_app", "doctor_entitlements_state"]
