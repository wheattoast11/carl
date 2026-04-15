"""Credits CLI sub-app for carl-studio.

Registration in cli.py (integration step):

    from carl_studio.credits._cli import credits_app
    app.add_typer(credits_app)

Commands are a Typer sub-app registered as ``carl credits``.
"""

from __future__ import annotations

import json
import webbrowser
from typing import Any

import typer

from carl_studio.console import get_console
from carl_studio.credits.balance import CreditBalance, CreditError, get_credit_balance
from carl_studio.credits.estimate import (
    BUNDLES,
    METHOD_STEP_SECONDS,
    CreditEstimate,
    best_bundle,
    estimate_job_cost,
)
from carl_studio.db import LocalDB

credits_app = typer.Typer(
    name="credits",
    help="Manage compute credits -- balance, estimates, and purchases.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _effective_tier() -> str:
    """Return the current effective tier string without raising."""
    try:
        from carl_studio.settings import CARLSettings
        from carl_studio.tier import detect_effective_tier

        settings = CARLSettings.load()
        return detect_effective_tier(settings.tier).value
    except Exception:
        return "free"


def _require_paid(c: Any) -> bool:
    """Check that user is on PAID tier. Print warning if not. Returns True if paid."""
    tier = _effective_tier()
    if tier != "paid":
        c.warn("Credits require CARL Paid tier.")
        c.info("Run: carl upgrade")
        return False
    return True


def _get_auth() -> tuple[str | None, str | None]:
    """Get jwt and supabase_url from local DB."""
    try:
        db = LocalDB()
        jwt = db.get_auth("jwt")
        supabase_url = db.get_config("supabase_url")
        return jwt, supabase_url
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# carl credits show
# ---------------------------------------------------------------------------


@credits_app.command(name="show")
def credits_show(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show current credit balance."""
    c = get_console()

    if not _require_paid(c):
        raise typer.Exit(0)

    jwt, supabase_url = _get_auth()

    if not jwt:
        c.warn("Not logged in. Credits require carl.camp account.")
        c.info("Run: carl login")
        raise typer.Exit(0)

    if not supabase_url:
        c.warn("No Supabase URL configured.")
        c.info("Run: carl login")
        raise typer.Exit(0)

    try:
        balance = get_credit_balance(jwt, supabase_url)
    except CreditError:
        if json_output:
            typer.echo(json.dumps({"error": "offline", "remaining": None}))
        else:
            c.warn("Could not fetch balance (offline).")
            c.info("Run: carl login")
        raise typer.Exit(0)

    if json_output:
        typer.echo(json.dumps(balance.model_dump(), indent=2))
        raise typer.Exit(0)

    c.header("Credits", "carl.camp")
    c.kv("Remaining", str(balance.remaining))
    c.kv("Used", str(balance.used))
    c.kv("Total", str(balance.total))
    if balance.included_monthly > 0:
        c.kv("Included/mo", str(balance.included_monthly))

    if balance.remaining < 50:
        c.blank()
        c.warn("Low balance -- consider: carl credits buy")


# ---------------------------------------------------------------------------
# carl credits estimate
# ---------------------------------------------------------------------------


@credits_app.command(name="estimate")
def credits_estimate(
    hardware: str = typer.Option("a100-large", help="Hardware flavor"),
    steps: int = typer.Option(80, help="Max training steps"),
    method: str = typer.Option(
        "grpo-env",
        help="Training method: sft, grpo-text, grpo-env, grpo-vision",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Estimate credit cost for a training job."""
    c = get_console()

    per_step = METHOD_STEP_SECONDS.get(method)
    if per_step is None:
        valid = ", ".join(sorted(METHOD_STEP_SECONDS.keys()))
        c.error(f"Unknown method '{method}'. Valid: {valid}")
        raise typer.Exit(1)

    try:
        est = estimate_job_cost(hardware, steps, per_step)
    except ValueError as e:
        c.error(str(e))
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(est.model_dump(), indent=2))
        raise typer.Exit(0)

    c.header("Credit Estimate", method)

    table = c.make_table("Parameter", "Value")
    table.add_row("Hardware", est.hardware)
    table.add_row("Rate", f"{est.rate_per_min} credits/min")
    table.add_row("Steps", str(steps))
    table.add_row("Est. time", f"{est.estimated_minutes} min")
    table.add_row("Est. credits", str(est.estimated_credits))
    table.add_row(f"Buffer ({int(est.buffer_pct * 100)}%)", f"+{est.total_with_buffer - est.estimated_credits}")
    table.add_row("Pre-deducted", str(est.total_with_buffer))
    table.add_row("Est. cost", f"${est.estimated_usd:.2f}")
    c.print(table)

    # Bundle recommendation
    rec = best_bundle(est.total_with_buffer)
    if rec:
        bundle = BUNDLES[rec]
        c.blank()
        c.info(
            f"Recommended bundle: {rec} "
            f"({bundle['credits']} credits, ${bundle['price_usd']})"
        )
    else:
        c.blank()
        c.info("This job exceeds the largest bundle. Buy multiple or contact support.")


# ---------------------------------------------------------------------------
# carl credits buy
# ---------------------------------------------------------------------------


@credits_app.command(name="buy")
def credits_buy(
    bundle: str = typer.Argument(
        "explorer",
        help="Bundle: starter (100/$8), explorer (500/$35), researcher (2000/$120)",
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open browser to checkout"
    ),
) -> None:
    """Buy a credit bundle. Opens carl.camp checkout."""
    c = get_console()

    if not _require_paid(c):
        raise typer.Exit(0)

    bundle_lower = bundle.lower()
    if bundle_lower not in BUNDLES:
        valid = ", ".join(sorted(BUNDLES.keys()))
        c.error(f"Unknown bundle '{bundle}'. Valid: {valid}")
        raise typer.Exit(1)

    info = BUNDLES[bundle_lower]
    credits_count = int(info["credits"])
    price = info["price_usd"]

    c.header("Buy Credits", "carl.camp")
    c.kv("Bundle", bundle_lower)
    c.kv("Credits", str(credits_count))
    c.kv("Price", f"${price}")
    c.kv("Per credit", f"${price / credits_count:.3f}")

    url = f"https://carl.camp/checkout?bundle={bundle_lower}"
    c.blank()
    c.info(f"Opening: {url}")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass  # Non-interactive environments -- URL is already printed

    c.blank()
    c.info("Credits appear in your balance after payment confirms.")


# ---------------------------------------------------------------------------
# carl credits history
# ---------------------------------------------------------------------------


@credits_app.command(name="history")
def credits_history(
    limit: int = typer.Option(10, help="Number of transactions to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show recent credit transactions."""
    c = get_console()

    if not _require_paid(c):
        raise typer.Exit(0)

    jwt, supabase_url = _get_auth()

    if not jwt or not supabase_url:
        c.warn("Not logged in. Run: carl login")
        raise typer.Exit(0)

    # Fetch from Supabase Edge Function
    import urllib.error
    import urllib.request

    url = f"{supabase_url}/functions/v1/credit-history?limit={limit}"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data: dict[str, Any] = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        c.warn(f"Could not fetch history ({e.code}).")
        raise typer.Exit(0)
    except urllib.error.URLError:
        c.warn("Offline -- cannot fetch credit history.")
        raise typer.Exit(0)
    except Exception:
        c.warn("Could not fetch credit history.")
        raise typer.Exit(0)

    transactions: list[dict[str, Any]] = data.get("transactions", [])

    if json_output:
        typer.echo(json.dumps(transactions, indent=2))
        raise typer.Exit(0)

    if not transactions:
        c.header("Credit History", "carl.camp")
        c.info("No transactions yet.")
        raise typer.Exit(0)

    c.header("Credit History", f"last {len(transactions)}")

    table = c.make_table("Date", "Type", "Amount", "Job", "Balance")
    for tx in transactions:
        tx_type = str(tx.get("type", "unknown"))
        amount_val = tx.get("amount", 0)
        amount_str = f"+{amount_val}" if tx_type in ("purchase", "refund", "included") else f"-{amount_val}"
        table.add_row(
            str(tx.get("created_at", ""))[:16],
            tx_type,
            amount_str,
            str(tx.get("job_id", ""))[:12] or "--",
            str(tx.get("balance_after", "")),
        )
    c.print(table)
