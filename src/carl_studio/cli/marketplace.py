"""Marketplace CLI commands for carl.camp.

Register into the main app as::

    marketplace_app = create_marketplace_app()
    app.add_typer(marketplace_app, name="marketplace")
    app.command("publish")(publish_cmd)

Do NOT modify cli.py -- registration happens externally.
"""

from __future__ import annotations

import typer
from typing import cast

from .shared import _warn_legacy_command_alias
from carl_studio.console import get_console
from carl_studio.marketplace import (
    MarketplaceAdapter,
    MarketplaceClient,
    MarketplaceError,
    MarketplaceModel,
)
from carl_studio.tier import check_tier


def _get_client() -> MarketplaceClient:
    """Build a MarketplaceClient, preferring DB auth then settings."""
    try:
        return MarketplaceClient.from_db()
    except Exception:
        try:
            return MarketplaceClient.from_settings()
        except Exception:
            return MarketplaceClient()


def _handle_error(e: MarketplaceError) -> None:
    """Map MarketplaceError to friendly CLI output."""
    c = get_console()
    msg = str(e)
    if "Network error" in msg or "URLError" in msg:
        c.error("Could not reach carl.camp. Check your connection.")
    elif "(401)" in msg or "authentication" in msg.lower():
        c.error("Not authenticated. Run: carl camp login")
    elif "not configured" in msg.lower():
        c.error("Marketplace URL not configured. Run: carl config set supabase_url <URL>")
    else:
        c.error(msg)


# ---------------------------------------------------------------------------
# Marketplace subcommands
# ---------------------------------------------------------------------------

marketplace_app = typer.Typer(
    name="marketplace",
    help="Browse and publish on the carl.camp marketplace.",
    no_args_is_help=True,
)


@marketplace_app.callback()
def marketplace_callback(ctx: typer.Context = typer.Option(None, hidden=True)) -> None:
    """Warn when the legacy top-level marketplace group is used."""
    _warn_legacy_command_alias(get_console(), ctx, "carl camp marketplace")


@marketplace_app.command("models")
def list_models(
    query: str = typer.Option("", "--query", "-q", help="Search query"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """List public models on the marketplace."""
    c = get_console()
    client = _get_client()
    try:
        models = client.list_models(public_only=True, query=query, limit=limit)
    except MarketplaceError as e:
        _handle_error(e)
        raise typer.Exit(1)

    if not models:
        c.info("No models found.")
        return

    table = c.make_table("Name", "Hub ID", "Base", "Downloads", "Stars")
    for m in models:
        table.add_row(m.name, m.hub_id, m.base_model, str(m.downloads), str(m.stars))
    c.print(table)


@marketplace_app.command("adapters")
def list_adapters(
    query: str = typer.Option("", "--query", "-q", help="Search query"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """List public adapters on the marketplace."""
    c = get_console()
    client = _get_client()
    try:
        adapters = client.list_adapters(public_only=True, query=query, limit=limit)
    except MarketplaceError as e:
        _handle_error(e)
        raise typer.Exit(1)

    if not adapters:
        c.info("No adapters found.")
        return

    table = c.make_table("Name", "Hub ID", "Rank", "Downloads", "Stars")
    for a in adapters:
        table.add_row(a.name, a.hub_id, str(a.rank), str(a.downloads), str(a.stars))
    c.print(table)


@marketplace_app.command("recipes")
def list_recipes(
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """List public recipes on the marketplace."""
    c = get_console()
    client = _get_client()
    try:
        recipes = client.list_recipes(limit=limit)
    except MarketplaceError as e:
        _handle_error(e)
        raise typer.Exit(1)

    if not recipes:
        c.info("No recipes found.")
        return

    table = c.make_table("Name", "Slug", "Courses", "Stars", "Description")
    for r in recipes:
        desc = r.description[:60] + "..." if len(r.description) > 60 else r.description
        table.add_row(r.name, r.slug, str(r.courses_count), str(r.stars), desc)
    c.print(table)


@marketplace_app.command("kits")
def list_kits(
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """List public kits on the marketplace."""
    c = get_console()
    client = _get_client()
    try:
        kits = client.list_kits(limit=limit)
    except MarketplaceError as e:
        _handle_error(e)
        raise typer.Exit(1)

    if not kits:
        c.info("No kits found.")
        return

    table = c.make_table("Name", "Slug", "Ingredients", "Stars", "Description")
    for k in kits:
        desc = k.description[:60] + "..." if len(k.description) > 60 else k.description
        table.add_row(k.name, k.slug, str(len(k.ingredients)), str(k.stars), desc)
    c.print(table)


@marketplace_app.command("show")
def show_item(
    hub_id: str = typer.Argument(..., help="Hub ID (e.g. wheattoast11/OmniCoder-9B)"),
) -> None:
    """Show detail for a model or adapter by hub ID."""
    c = get_console()
    client = _get_client()

    # Try model first, then adapter
    item: MarketplaceModel | MarketplaceAdapter | None = None
    item_kind = ""
    try:
        item = client.get_model(hub_id)
        if item:
            item_kind = "Model"
    except MarketplaceError:
        pass

    if item is None:
        try:
            item = client.get_adapter(hub_id)
            if item:
                item_kind = "Adapter"
        except MarketplaceError:
            pass

    if item is None:
        c.error(f"Not found: {hub_id}")
        raise typer.Exit(1)

    c.header(f"{item_kind}: {item.name}")
    c.kv("Hub ID", item.hub_id)
    if item_kind == "Model":
        model = cast(MarketplaceModel, item)
        c.kv("Base Model", model.base_model)
        c.kv("Source Type", model.source_type)
    else:
        adapter = cast(MarketplaceAdapter, item)
        c.kv("Rank", str(adapter.rank))
        c.kv("Compatible", ", ".join(adapter.compatible_bases) or "(any)")
    c.kv("Downloads", str(item.downloads))
    c.kv("Stars", str(item.stars))
    c.kv("Public", str(item.public))
    if item.description:
        c.blank()
        c.print(f"  {item.description}")
    if hasattr(item, "capability_dims") and item.capability_dims:
        c.kv("Capabilities", ", ".join(item.capability_dims))


# ---------------------------------------------------------------------------
# Top-level publish command
# ---------------------------------------------------------------------------


def publish_cmd(
    ctx: typer.Context = typer.Option(None, hidden=True),
    hub_id: str = typer.Argument(..., help="HF Hub repo ID to publish"),
    item_type: str = typer.Option("model", "--type", "-t", help="model or adapter"),
    name: str = typer.Option("", "--name", "-n", help="Display name (defaults to hub_id)"),
    description: str = typer.Option("", "--description", "-d", help="Short description"),
    public: bool = typer.Option(True, "--public/--private", help="Visibility"),
    base_model: str = typer.Option("", "--base", "-b", help="Base model (for models)"),
    rank: int = typer.Option(64, "--rank", "-r", help="LoRA rank (for adapters)"),
) -> None:
    """Publish a model or adapter to the carl.camp marketplace."""
    c = get_console()
    _warn_legacy_command_alias(c, ctx, "carl camp publish")

    # Tier gate: publishing requires PAID
    allowed, effective, _required = check_tier("marketplace.publish")
    if not allowed:
        c.error(
            f"Publishing requires CARL Paid. "
            f"Current tier: {effective.value}. "
            f"Upgrade at https://carl.camp/pricing"
        )
        raise typer.Exit(1)

    client = _get_client()
    display_name = name or hub_id.split("/")[-1]

    try:
        if item_type == "model":
            model = MarketplaceModel(
                hub_id=hub_id,
                name=display_name,
                base_model=base_model,
                description=description,
                public=public,
            )
            result = client.publish_model(model)
            c.ok(f"Published model: {result.hub_id}")
            if result.id:
                c.kv("ID", result.id)

        elif item_type == "adapter":
            adapter = MarketplaceAdapter(
                hub_id=hub_id,
                name=display_name,
                description=description,
                public=public,
                rank=rank,
            )
            result = client.publish_adapter(adapter)
            c.ok(f"Published adapter: {result.hub_id}")
            if result.id:
                c.kv("ID", result.id)

        else:
            c.error(f"Invalid type: {item_type}. Must be 'model' or 'adapter'.")
            raise typer.Exit(1)

    except MarketplaceError as e:
        _handle_error(e)
        raise typer.Exit(1)


marketplace_app.command("publish")(publish_cmd)


# ---------------------------------------------------------------------------
# Factory for registration
# ---------------------------------------------------------------------------


def create_marketplace_app() -> typer.Typer:
    """Return the marketplace sub-app for registration in cli.py."""
    return marketplace_app
