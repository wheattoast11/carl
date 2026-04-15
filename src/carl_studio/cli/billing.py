"""Billing CLI commands for carl.camp subscription management.

Commands: upgrade, billing (portal), subscription, account.
Wired in cli/wiring.py.
"""

from __future__ import annotations

import json
import webbrowser

import typer

from carl_studio.billing import (
    BILLING_PORTAL_URL,
    CHECKOUT_ANNUAL_URL,
    CHECKOUT_MONTHLY_URL,
    BillingError,
    get_subscription_status,
)
from carl_studio.camp import CampError, resolve_camp_profile
from carl_studio.console import get_console
from carl_studio.db import LocalDB

from .shared import _warn_legacy_command_alias


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAID_FEATURES = [
    ("--send-it", "Autonomous SFT → eval → GRPO → eval → push pipeline"),
    ("Experiments", "Discovery engine: auto-judgment, pre-registered hypotheses"),
    ("Cloud sync", "carl push / carl pull — Supabase-backed history"),
    ("carl.camp", "Web dashboard: run history, metrics, gate status"),
    ("MCP server", "Multi-agent integration via Model Context Protocol"),
    ("Resonance rewards", "terminals-runtime reward layer (SLOT, Kuramoto LR)"),
]


def _effective_tier() -> str:
    """Return the current effective tier string without raising."""
    try:
        from carl_studio.settings import CARLSettings
        from carl_studio.tier import detect_effective_tier

        settings = CARLSettings.load()
        return detect_effective_tier(settings.tier).value
    except Exception:
        return "free"


# ---------------------------------------------------------------------------
# carl upgrade
# ---------------------------------------------------------------------------


def upgrade(
    ctx: typer.Context = typer.Option(None, hidden=True),
    annual: bool = typer.Option(False, "--annual", help="Annual plan ($290/year, save 17%)"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser to checkout"),
) -> None:
    """Upgrade to CARL Paid. Opens carl.camp checkout."""
    c = get_console()
    _warn_legacy_command_alias(c, ctx, "carl camp upgrade")
    c.header("CARL Upgrade", "carl.camp")

    tier = _effective_tier()

    if tier == "paid":
        # Already subscribed — show status and portal hint
        db = LocalDB()
        jwt = db.get_auth("jwt")
        supabase_url = db.get_config("supabase_url")
        days_info = ""
        if jwt and supabase_url:
            try:
                status = get_subscription_status(jwt, supabase_url)
                dr = status.days_remaining
                if dr is not None:
                    days_info = f" ({dr} days remaining)"
                if status.cancel_at_period_end:
                    c.warn(f"Subscription cancels at period end{days_info}.")
                else:
                    c.ok(f"Already CARL Paid{days_info}.")
            except BillingError:
                c.ok("Already CARL Paid.")
        else:
            c.ok("Already CARL Paid.")
        c.info("Manage: carl camp billing")
        raise typer.Exit(0)

    # FREE → show what PAID unlocks
    c.voice("send_it")
    c.blank()
    c.print("  [camp.primary]CARL Paid unlocks:[/]")
    for feature, description in _PAID_FEATURES:
        c.kv(feature, description, key_width=18)
    c.blank()

    plan_label = "annual ($290/yr)" if annual else "monthly ($29/mo)"
    url = CHECKOUT_ANNUAL_URL if annual else CHECKOUT_MONTHLY_URL

    c.info(f"Plan: {plan_label}")
    c.info(f"Opening: {url}")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass  # Non-interactive environments — URL is already printed

    c.blank()
    c.info("After checkout, run: carl camp login  to update your session.")


# ---------------------------------------------------------------------------
# carl billing
# ---------------------------------------------------------------------------


def billing_portal(
    ctx: typer.Context = typer.Option(None, hidden=True),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open browser to billing portal"
    ),
) -> None:
    """Open carl.camp billing portal to manage subscription."""
    c = get_console()
    _warn_legacy_command_alias(c, ctx, "carl camp billing")
    c.header("CARL Billing", "carl.camp/billing")

    db = LocalDB()
    jwt = db.get_auth("jwt")

    if not jwt:
        c.warn("Not logged in. Run: carl camp login")
        raise typer.Exit(1)

    c.ok("Opening billing portal...")
    c.info(f"URL: {BILLING_PORTAL_URL}")

    if open_browser:
        try:
            webbrowser.open(BILLING_PORTAL_URL)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# carl subscription
# ---------------------------------------------------------------------------


def subscription_status(
    ctx: typer.Context = typer.Option(None, hidden=True),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show current subscription status."""
    c = get_console()
    if not json_output:
        _warn_legacy_command_alias(c, ctx, "carl camp subscription")

    db = LocalDB()
    jwt = db.get_auth("jwt")
    supabase_url = db.get_config("supabase_url")

    if not jwt:
        c.warn("Not logged in.")
        c.info("Run: carl camp login")
        raise typer.Exit(0)

    if not supabase_url:
        # No supabase URL configured — show cached tier only
        tier = _effective_tier()
        if json_output:
            typer.echo(json.dumps({"tier": tier, "status": "cached", "plan": None}))
        else:
            c.header("Subscription", "cached")
            c.kv("Tier", tier)
            c.warn("No Supabase URL configured. Run: carl camp login")
        raise typer.Exit(0)

    try:
        status = get_subscription_status(jwt, supabase_url)

        if json_output:
            typer.echo(json.dumps(status.to_dict(), indent=2))
            raise typer.Exit(0)

        c.header("Subscription", "carl.camp")
        c.kv("Tier", status.tier.title())
        c.kv("Plan", status.plan or "—")
        c.kv("Status", status.status)

        dr = status.days_remaining
        if dr is not None:
            c.kv("Renews in", f"{dr} days")

        if status.cancel_at_period_end:
            c.warn("Subscription cancels at end of current period.")

        if status.is_active_paid:
            c.ok("CARL Paid — full autonomy unlocked.")
        elif status.tier == "free":
            c.info("Free tier. Run: carl camp upgrade  to unlock autonomy.")

    except BillingError:
        # Graceful offline fallback
        tier = _effective_tier()
        if json_output:
            typer.echo(json.dumps({"tier": tier, "status": "cached", "plan": None}))
        else:
            c.warn("Could not reach carl.camp — showing cached tier.")
            c.kv("Tier", tier)
            c.info("Check your connection or run: carl camp login")


# ---------------------------------------------------------------------------
# carl camp account
# ---------------------------------------------------------------------------


def account_status(
    json_output: bool = typer.Option(False, "--json", help="Output account profile as JSON"),
    refresh: bool = typer.Option(
        True, "--refresh/--cached", help="Refresh from carl.camp before rendering"
    ),
) -> None:
    """Show the unified managed account profile and payment capabilities."""
    c = get_console()
    db = LocalDB()

    try:
        session, profile, source = resolve_camp_profile(refresh=refresh, db=db)
    except CampError as exc:
        if json_output:
            typer.echo(
                json.dumps({"authenticated": True, "error": str(exc), "source": "error"}, indent=2)
            )
            raise typer.Exit(0)
        c.header("Camp Account", "error")
        c.warn(str(exc))
        c.info("Use cached mode with: carl camp account --cached")
        raise typer.Exit(0)

    payload = {
        "authenticated": bool(session.jwt),
        "supabase_configured": bool(session.supabase_url),
        "cached_tier": session.cached_tier,
        "source": source,
        "profile": profile.to_dict() if profile is not None else None,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(0)

    c.header("Camp Account", source)
    if not session.jwt:
        c.warn("Not logged in. CARL Studio still works fully in local-first FREE mode.")
        c.info("Attach the managed platform later with: carl camp login")
        c.info("Free/core path stays available: carl train --config carl.yaml")
        raise typer.Exit(0)

    c.kv("Authenticated", "yes")
    c.kv("Supabase", session.supabase_url or "(not set)")

    if profile is None:
        c.warn("Authenticated, but no account profile is cached yet.")
        c.info("Run: carl camp subscription")
        raise typer.Exit(0)

    c.kv("Tier", profile.tier.title())
    c.kv("Plan", profile.plan or "free")
    c.kv("Status", profile.status)
    if profile.days_remaining is not None:
        c.kv("Renews in", f"{profile.days_remaining} days")
    c.kv("Credits", f"{profile.credits_remaining} remaining / {profile.credits_total} total")
    c.kv("Payments", profile.payment_summary)
    c.kv("Wallet login", "enabled" if profile.wallet_auth_enabled else "not enabled")
    c.kv("x402 rail", "enabled" if profile.x402_enabled else "planned")
    c.kv("Observability", "opt-in" if profile.observability_opt_in else "off by default")
    c.kv("Telemetry", "opt-in" if profile.telemetry_opt_in else "off by default")
    c.kv("Usage tracking", "minimal" if profile.usage_tracking_enabled else "off")
    c.kv("Contract witnesses", "enabled" if profile.contract_witnessing else "planned")
    if profile.contract_terms_url:
        c.kv("Terms", profile.contract_terms_url)

    c.blank()
    if profile.tier == "paid":
        c.info("Manage subscription: carl camp billing")
        c.info("Inspect credits: carl camp credits show")
        c.info("Sync managed state: carl camp sync push")
    else:
        c.info("Stay local-first on FREE: carl train --config carl.yaml")
        c.info("Attach managed features later: carl camp upgrade")
        c.info("Lead-safe default: observability and telemetry remain opt-in")
