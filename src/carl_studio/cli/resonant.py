"""Resonant CLI — ``carl resonant [publish|list|whoami|eval]``.

v0.9.x surface per `docs/v0_9_deferred_items.md` §1.1. The ``fit``
command is deliberately omitted in this release because it needs the
private terminals-runtime fitter or the resonance package's torch
optimizer — both behind the admin gate. When wired, ``fit`` follows
the ``ttt/eml_head.py`` pattern (public stub → lazy admin import).

All subcommands here use only public primitives:
    carl_core.eml, carl_core.resonant, carl_core.signing,
    carl_studio.resonant_store.

Wire contract: docs/eml_signing_protocol.md §5.1.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import typer

resonant_app = typer.Typer(
    name="resonant",
    help="Manage local Resonants: publish to carl.camp, list, eval, identity.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_REDACTED = "<redacted>"


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of headers with secret-shaped fields redacted."""
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in {"x-carl-user-secret", "authorization"}:
            out[k] = _REDACTED
        else:
            out[k] = v
    return out


def _resolve_bearer_token() -> str | None:
    """Reuse the same resolution order as a2a._cli._resolve_bearer_token.

    Order: ``CARL_CAMP_TOKEN`` env → ``~/.carl/camp_token`` legacy file →
    ``LocalDB.get_auth("jwt")`` (what ``carl camp login`` writes).
    """
    explicit = os.environ.get("CARL_CAMP_TOKEN")
    if explicit:
        return explicit
    token_file = Path.home() / ".carl" / "camp_token"
    if token_file.exists():
        try:
            text = token_file.read_text().strip()
            if text:
                return text
        except OSError:
            pass
    try:
        from carl_studio.db import LocalDB

        return LocalDB().get_auth("jwt")
    except Exception:
        return None


def _http_post_bytes(
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float = 20.0,
) -> tuple[int, bytes, dict[str, str]]:
    """POST raw bytes. Uses requests if available; stdlib urllib otherwise.

    Returns ``(status_code, response_body, response_headers)``. Never
    logs ``X-Carl-User-Secret`` or ``Authorization``.
    """
    try:
        import requests  # type: ignore[import-untyped]

        resp = requests.post(url, data=body, headers=headers, timeout=timeout)
        return resp.status_code, resp.content, dict(resp.headers)
    except ImportError:
        pass

    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (
                resp.getcode(),
                resp.read(),
                dict(resp.getheaders()),
            )
    except urllib.error.HTTPError as e:
        return (
            e.code,
            (e.read() if e.fp else b""),
            dict(e.headers.items()) if e.headers else {},
        )


def _console() -> Any:
    """Lazy import to avoid pulling heavy deps on --help."""
    from carl_studio.console import get_console

    return get_console()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@resonant_app.command("whoami")
def whoami_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show the identity fingerprint for the local user_secret.

    Creates the secret on first run (32 bytes at
    ``~/.carl/credentials/user_secret``, mode 0600).
    """
    from carl_studio.resonant_store import identity_fingerprint, read_or_create_user_secret

    secret = read_or_create_user_secret()
    fp = identity_fingerprint(secret)
    short = fp[:8]

    if json_output:
        typer.echo(json.dumps({"sig_public_component": fp, "short": short}))
        return

    c = _console()
    c.header("Identity", short)
    c.kv("Fingerprint", fp, key_width=20)
    c.kv("Short", short, key_width=20)
    c.info("Share the short form; it's how marketplace consumers recognize your Resonants.")


@resonant_app.command("list")
def list_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Enumerate local Resonants in ``~/.carl/resonants/``."""
    from carl_studio.resonant_store import list_resonants

    entries = list_resonants()
    if json_output:
        typer.echo(json.dumps(entries, indent=2))
        return

    c = _console()
    if not entries:
        c.info("No local Resonants. Train one, then call carl resonant publish <name>.")
        return
    c.header("Local Resonants", f"{len(entries)} total")
    for e in entries:
        name = e.get("name", "(unnamed)")
        depth = e.get("depth")
        inp = e.get("input_dim")
        out = e.get("output_dim")
        tree_hash = e.get("tree_hash", "") or ""
        c.kv(
            name,
            f"depth={depth} dim={inp}→{out} hash={tree_hash[:12]}",
            key_width=24,
        )


@resonant_app.command("eval")
def eval_cmd(
    name: str = typer.Argument(..., help="Local Resonant name"),
    inputs: str = typer.Option(
        ..., "--inputs", "-i",
        help="JSON array of floats, e.g. '[0.4, 0.1, 0.7]'",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Evaluate a local Resonant on the given observation vector.

    Runs ``perceive → cognize → act`` end-to-end. Useful sanity check
    before publishing to the marketplace.
    """
    from carl_core.errors import ValidationError
    from carl_core.resonant import ResonantError

    from carl_studio.resonant_store import load_resonant

    try:
        obs_list = json.loads(inputs)
    except json.JSONDecodeError as exc:
        typer.echo(f"invalid --inputs JSON: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not isinstance(obs_list, list):
        typer.echo("--inputs must be a JSON array", err=True)
        raise typer.Exit(1)

    try:
        resonant, _envelope, _meta = load_resonant(name)
    except ValidationError as exc:
        typer.echo(f"load failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    obs = np.asarray(obs_list, dtype=np.float64)
    try:
        action = resonant.forward(obs)
    except ValidationError as exc:
        typer.echo(f"eval failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    except ResonantError as exc:
        # Joint-mode Resonant + admin-locked host: surface the private-required
        # message instead of a bare traceback. Exit code 2 distinguishes the
        # gated-feature case from ValidationError shape/decode failures (1).
        typer.echo(f"eval blocked: {exc}", err=True)
        raise typer.Exit(2) from exc

    latent = resonant.perceive(obs)
    action_list = [float(x) for x in np.asarray(action).ravel()]
    latent_list = [float(x) for x in np.asarray(latent).ravel()]

    if json_output:
        typer.echo(json.dumps({
            "name": name,
            "action": action_list,
            "latent": latent_list,
            "tree_hash": resonant.tree.hash(),
        }))
        return

    c = _console()
    c.header("Resonant.eval", name)
    c.kv("Input", json.dumps(obs_list), key_width=16)
    c.kv("Latent", json.dumps([round(x, 4) for x in latent_list]), key_width=16)
    c.kv("Action", json.dumps([round(x, 4) for x in action_list]), key_width=16)
    c.kv("Tree hash", resonant.tree.hash()[:16] + "...", key_width=16)


@resonant_app.command("publish")
def publish_cmd(
    name: str = typer.Argument(..., help="Local Resonant name to upload"),
    base_url: str = typer.Option(
        "https://carl.camp",
        "--base-url",
        envvar="CARL_CAMP_BASE",
        help="carl.camp platform base URL",
    ),
    domain: str | None = typer.Option(
        None, "--domain", help="Optional domain tag for discovery (e.g. 'audio')"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Build the request but do not send it"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Upload a local Resonant to carl.camp.

    Wire contract: ``docs/eml_signing_protocol.md`` §5.1.
      * Body = signed .emlt envelope (codec §1.2).
      * ``X-Carl-User-Secret`` header carries base64 of the raw
        ``~/.carl/credentials/user_secret``.
      * Projection + readout matrices travel as separate base64 headers.

    The user_secret is NEVER logged, NEVER stored server-side
    long-term, and NEVER sent over plain HTTP (we refuse ``http://``
    base URLs unless explicitly opted in via ``--dry-run``).
    """
    # v0.18 Track B: publishing a Resonant mints a marketplace entry
    # tied to the enclosing project — require a project context so the
    # signed envelope's provenance matches an on-disk project root.
    from carl_studio.project_context import require as _require_project

    _require_project("resonant publish")

    from carl_core.errors import ValidationError

    from carl_studio.resonant_store import (
        identity_fingerprint,
        load_resonant,
        read_or_create_user_secret,
    )

    c = _console()

    if not (base_url.startswith("https://") or dry_run):
        typer.echo(
            f"refusing to POST a user_secret over {base_url!r} (non-HTTPS). "
            "Use --dry-run to inspect the request shape without sending.",
            err=True,
        )
        raise typer.Exit(2)

    try:
        resonant, envelope, meta = load_resonant(name)
    except ValidationError as exc:
        typer.echo(f"load failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    secret = read_or_create_user_secret()
    fp_short = identity_fingerprint(secret)[:8]

    proj_b64 = base64.b64encode(
        np.asarray(resonant.projection, dtype=np.float64).tobytes()
    ).decode("ascii")
    read_b64 = base64.b64encode(
        np.asarray(resonant.readout, dtype=np.float64).tobytes()
    ).decode("ascii")

    headers: dict[str, str] = {
        "Content-Type": "application/octet-stream",
        "X-Carl-User-Secret": base64.b64encode(secret).decode("ascii"),
        "X-Carl-Input-Dim": str(resonant.tree.input_dim),
        "X-Carl-Output-Dim": str(resonant.readout.shape[0]),
        "X-Carl-Latent-Dim": str(resonant.projection.shape[0]),
        "X-Carl-Projection": proj_b64,
        "X-Carl-Readout": read_b64,
        "X-Carl-Identity-Short": fp_short,
    }
    if domain:
        headers["X-Carl-Domain"] = domain

    bearer = _resolve_bearer_token()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    url = f"{base_url.rstrip('/')}/api/resonants"

    if dry_run:
        summary = {
            "url": url,
            "headers": _redact_headers(headers),
            "body_bytes": len(envelope),
            "identity_short": fp_short,
            "tree_hash": resonant.tree.hash(),
            "depth": resonant.tree.depth(),
            "input_dim": resonant.tree.input_dim,
            "output_dim": int(resonant.readout.shape[0]),
        }
        if json_output:
            typer.echo(json.dumps(summary, indent=2))
        else:
            c.header("Resonant.publish (dry-run)", name)
            c.kv("POST", url, key_width=20)
            c.kv("Identity", fp_short, key_width=20)
            c.kv("Body bytes", str(len(envelope)), key_width=20)
            c.kv("Headers", json.dumps(summary["headers"], indent=2), key_width=20)
            c.info("No request sent. Drop --dry-run to ship.")
        return

    if not bearer:
        typer.echo(
            "no CARL_CAMP_TOKEN and no ~/.carl/camp_token file — run `carl camp login` first",
            err=True,
        )
        raise typer.Exit(1)

    try:
        status, body, resp_headers = _http_post_bytes(url, envelope, headers)
    except Exception as exc:  # noqa: BLE001 — CLI boundary; print redacted error
        typer.echo(f"POST failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if status == 200:
        try:
            parsed = json.loads(body.decode("utf-8", errors="replace"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = {"raw": body.decode("utf-8", errors="replace")}
        resonant_id = parsed.get("id", "?")
        content_hash = parsed.get("content_hash", resonant.tree.hash())
        if json_output:
            typer.echo(json.dumps({
                "status": "published",
                "name": name,
                "resonant_id": resonant_id,
                "content_hash": content_hash,
                "url": url,
            }))
        else:
            c.header("Resonant.publish", name)
            c.kv("Status", f"✓ published (HTTP 200)", key_width=16)
            c.kv("ID", str(resonant_id), key_width=16)
            c.kv("Hash", content_hash[:16] + "...", key_width=16)
            c.kv("Identity", fp_short, key_width=16)
        return

    if status == 402:
        typer.echo(
            "402 Payment Required — publish path is free-tier; this indicates "
            "tier misconfiguration on carl.camp. Response body:\n"
            + body.decode("utf-8", errors="replace"),
            err=True,
        )
        raise typer.Exit(1)

    if status == 409:
        # v0.18 Track D — per docs/platform-parity-reply-2026-04-22.md §Q2
        # and carl.camp agent's ack (fc535e5, 2026-04-22): the 409 body
        # deliberately does NOT include the existing resonant id —
        # ownership-privacy, no leak of another user's id to a
        # hash-collision visitor. No retry, no auto-rename.
        typer.echo(
            "409 Conflict — this content-hash is already published under "
            "a different signer. Your copy has the same bytes as an "
            "existing Resonant owned by another user; we refuse to dedupe "
            "silently because that would strip authorship.\n"
            "To proceed: rename / tweak your Resonant so the hash differs.",
            err=True,
        )
        raise typer.Exit(1)

    if status == 422:
        typer.echo(
            f"422 attestation failed — {body.decode('utf-8', errors='replace')}\n"
            "Regenerate the Resonant and re-save, or rotate your user_secret via "
            "carl whoami.",
            err=True,
        )
        raise typer.Exit(1)

    # Other error path — surface response without the secret headers
    snippet = body.decode("utf-8", errors="replace")[:500]
    typer.echo(f"POST {url} → HTTP {status}\n{snippet}", err=True)
    raise typer.Exit(1)


__all__ = ["resonant_app"]
