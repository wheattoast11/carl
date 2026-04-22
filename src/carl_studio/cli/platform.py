"""Platform-facing config, auth, and sync command groups."""

from __future__ import annotations

import typer

from carl_studio.camp import (
    DEFAULT_CARL_CAMP_BASE,
    DEFAULT_CARL_CAMP_SUPABASE_URL,
    CampError,
    resolve_camp_profile,
)
from carl_studio.console import get_console

from .apps import app
from .shared import _warn_legacy_command_alias

# ---------------------------------------------------------------------------
# carl config — settings management
# ---------------------------------------------------------------------------

config_app = typer.Typer(
    name="config", help="User settings and tier management", no_args_is_help=True
)
app.add_typer(config_app)


@config_app.command(name="show")
def config_show(
    unmask: bool = typer.Option(False, "--unmask", help="Show full credential values"),
) -> None:
    """Display current settings (credentials masked by default)."""
    from carl_studio.settings import CARLSettings, GLOBAL_CONFIG
    from carl_studio.tier import FEATURE_TIERS, Tier

    c = get_console()
    settings = CARLSettings.load()
    effective = settings.get_effective_tier()
    display = settings.display_dict(mask_secrets=not unmask)

    c.blank()
    c.header("CARL Settings")

    # Tier info with auto-elevation indicator
    tier_label = settings.tier.value.title()
    if effective != settings.tier:
        tier_label += f" -> {effective.value.title()} (auto-elevated)"
    c.kv("Tier", tier_label, key_width=20)
    c.kv("Preset", display["preset"], key_width=20)
    c.blank()

    # Core settings
    pairs = [
        ("default_model", display["default_model"]),
        ("default_compute", display["default_compute"]),
        ("hub_namespace", display["hub_namespace"]),
        ("naming_prefix", display["naming_prefix"]),
        ("log_level", display["log_level"]),
        ("trackio_url", display["trackio_url"]),
    ]
    c.config_block(pairs, title="Defaults")

    # Credentials
    cred_pairs = [
        ("hf_token", display["hf_token"]),
        ("anthropic_api_key", display["anthropic_api_key"]),
    ]
    c.config_block(cred_pairs, title="Credentials")

    # Observe defaults
    obs_pairs = [
        ("entropy", display["observe.entropy"]),
        ("phi", display["observe.phi"]),
        ("sparkline", display["observe.sparkline"]),
        ("poll_interval", display["observe.poll_interval"]),
        ("source", display["observe.source"]),
    ]
    c.config_block(obs_pairs, title="Observe Defaults")

    # Config file locations
    c.blank()
    c.info(f"Global config: {GLOBAL_CONFIG}")
    local = settings.model_config.get("env_prefix", "CARL_")
    c.info(f"Env prefix: {local}")

    # Feature access
    c.blank()
    gated_features = sorted(
        ((f, t) for f, t in FEATURE_TIERS.items() if t > Tier.FREE),
        key=lambda x: (x[1].value, x[0]),
    )
    if gated_features:
        table = c.make_table("Feature", "Required Tier", "Access", title="Gated Features")
        for feat, required in gated_features:
            allowed = effective >= required
            icon = c.theme.icons.ok if allowed else c.theme.icons.fail
            table.add_row(feat, required.value.title(), icon)
        c.print(table)

    c.blank()


@config_app.command(name="set")
def config_set(
    key: str = typer.Argument(..., help="Setting key (e.g. tier, default_model, log_level)"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Set a configuration value. Saves to ~/.carl/config.yaml."""
    from carl_studio.settings import CARLSettings, set_field

    c = get_console()
    settings = CARLSettings.load()

    try:
        settings = set_field(settings, key, value)
    except ValueError as exc:
        c.error(str(exc))
        raise typer.Exit(1)

    path = settings.save()
    c.ok(f"{key} = {value}")
    c.info(f"Saved to {path}")


@config_app.command(name="reset")
def config_reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Reset all settings to defaults."""
    from carl_studio.settings import reset_settings, GLOBAL_CONFIG

    c = get_console()
    if not force:
        if not GLOBAL_CONFIG.is_file():
            c.info("No config file to reset.")
            raise typer.Exit(0)
        from carl_studio.cli import ui

        if not ui.confirm("  Reset all settings to defaults?", default=False):
            raise typer.Exit(0)

    reset_settings()
    c.ok("Settings reset to defaults.")
    c.info(f"Removed {GLOBAL_CONFIG}")


@config_app.command(name="init")
def config_init(
    preset: str = typer.Option(
        "", "--preset", "-p", help="Start with a preset: research, production, quick"
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Interactive setup"
    ),
) -> None:
    """Create ~/.carl/config.yaml with optional interactive prompts."""
    from carl_studio.settings import CARLSettings, GLOBAL_CONFIG, Preset
    from carl_studio.tier import Tier

    c = get_console()

    from carl_studio.cli import ui

    if GLOBAL_CONFIG.is_file() and interactive:
        c.info(f"Config already exists at {GLOBAL_CONFIG}")
        if not ui.confirm("  Overwrite?", default=False):
            raise typer.Exit(0)

    settings = CARLSettings()

    if preset:
        try:
            settings.preset = Preset(preset.lower())
        except ValueError:
            c.error(f"Unknown preset '{preset}'. Use: research, production, quick")
            raise typer.Exit(1)

    if interactive:
        c.blank()
        c.header("CARL Config Setup")
        c.blank()

        # Preset
        preset_choice = ui.select(
            "Configuration preset",
            [
                ui.Choice(
                    value="custom",
                    label="Custom",
                    badge="recommended",
                    hint="configure everything manually",
                ),
                ui.Choice(
                    value="research",
                    label="Research",
                    hint="verbose observe, debug logging, all metrics",
                ),
                ui.Choice(
                    value="production",
                    label="Production",
                    hint="minimal logging, auto-push, eval gating",
                ),
                ui.Choice(
                    value="quick",
                    label="Quick",
                    hint="fast defaults, L4 compute, 20 steps max",
                ),
            ],
            default=0,
        )
        preset_map = {
            "research": Preset.RESEARCH,
            "production": Preset.PRODUCTION,
            "quick": Preset.QUICK,
            "custom": Preset.CUSTOM,
        }
        settings.preset = preset_map.get(preset_choice, Preset.CUSTOM)

        # Model
        c.blank()
        settings.default_model = ui.text(
            "  Default base model",
            default=settings.default_model,
        )

        # Hub namespace
        detected_ns = settings.hub_namespace
        if detected_ns:
            c.info(f"Detected HF namespace: {detected_ns}")
        settings.hub_namespace = ui.text(
            "  Hub namespace",
            default=detected_ns or "",
        )

        # Naming prefix
        settings.naming_prefix = ui.text(
            "  Naming prefix (for runs/repos)",
            default=settings.naming_prefix,
        )

        # Trackio
        c.blank()
        trackio = ui.text("  Trackio dashboard URL (blank to skip)", default="")
        if trackio:
            settings.trackio_url = trackio

        # Tier preference comes last so the workbench stays local-first.
        c.blank()
        tier_choice = ui.select(
            "Preferred tier for upgrade prompts",
            [
                ui.Choice(
                    value="free",
                    label="Free",
                    badge="recommended",
                    hint="Local-first workbench, BYOK compute, manual control",
                ),
                ui.Choice(
                    value="paid",
                    label="Paid",
                    hint="Sync, autonomy, marketplace, fleet, platform",
                ),
            ],
            default=0,
            help="You can stay free and upgrade later with `carl camp upgrade`",
        )
        tier_map = {"free": Tier.FREE, "paid": Tier.PAID}
        settings.tier = tier_map.get(tier_choice, Tier.FREE)

    # Apply preset after interactive to merge
    settings = settings.model_validate(settings.model_dump())

    path = settings.save()
    c.blank()
    c.ok(f"Config saved to {path}")

    # Show summary
    display = settings.display_dict()
    pairs = [(k, v) for k, v in display.items() if k not in ("hf_token", "anthropic_api_key")]
    c.config_block(pairs, title="Your Settings")
    c.blank()
    c.info("Credentials are auto-detected from exported environment variables and HF hub auth.")
    c.info(".env files are not auto-loaded; source them before running `carl` if you use one.")
    c.info("Run 'carl config show' to see your full configuration.")
    c.blank()


@config_app.command(name="path")
def config_path() -> None:
    """Show config file locations."""
    from carl_studio.settings import GLOBAL_CONFIG, CARL_HOME, _find_local_config

    c = get_console()
    c.blank()
    c.kv("Home", str(CARL_HOME), key_width=14)
    c.kv("Global config", str(GLOBAL_CONFIG), key_width=14)
    c.kv("Global exists", "yes" if GLOBAL_CONFIG.is_file() else "no", key_width=14)

    local = _find_local_config()
    if local:
        c.kv("Local config", str(local), key_width=14)
    else:
        c.kv("Local config", "(none found)", key_width=14)
    c.blank()


@config_app.command(name="preset")
def config_preset(
    name: str = typer.Argument(..., help="Preset name: research, production, quick"),
) -> None:
    """Apply a configuration preset."""
    from carl_studio.settings import CARLSettings, Preset

    c = get_console()
    try:
        preset = Preset(name.lower())
    except ValueError:
        c.error(f"Unknown preset '{name}'. Use: research, production, quick")
        raise typer.Exit(1)

    if preset == Preset.CUSTOM:
        c.error("'custom' is not a preset. Use 'carl config set' to customize.")
        raise typer.Exit(1)

    settings = CARLSettings.load()
    settings.preset = preset
    settings = settings.model_validate(settings.model_dump())  # Re-trigger preset application
    path = settings.save()

    c.ok(f"Applied preset: {name}")
    display = settings.display_dict()
    c.config_block(
        [
            (k, v)
            for k, v in display.items()
            if k
            in (
                "default_compute",
                "log_level",
                "observe.entropy",
                "observe.phi",
                "observe.sparkline",
                "observe.poll_interval",
            )
        ],
        title=f"Preset: {name}",
    )
    c.info(f"Saved to {path}")
    c.blank()


# ---------------------------------------------------------------------------
# carl login — authenticate with carl.camp
# ---------------------------------------------------------------------------


@app.command(name="login", hidden=True)
def login(
    ctx: typer.Context = typer.Option(None, hidden=True),
    upgrade: bool = typer.Option(False, "--upgrade", help="Open upgrade checkout after login"),
) -> None:
    """Authenticate with carl.camp. Opens browser for login."""
    import webbrowser
    import http.server
    import urllib.parse
    import threading

    c = get_console()
    _warn_legacy_command_alias(c, ctx, "carl camp login")
    c.blank()
    c.header("CARL Login")

    from carl_studio.db import LocalDB

    db = LocalDB()

    # Check if already logged in
    existing_jwt = db.get_auth("jwt")
    if existing_jwt and not upgrade:
        c.ok("Already authenticated.")
        try:
            _, profile, source = resolve_camp_profile(refresh=True, db=db)
            if profile is not None:
                c.info(f"Account: {profile.tier.title()} ({profile.status}) via {source}")
                if profile.plan:
                    c.info(f"Plan: {profile.plan}")
        except CampError:
            c.info("Could not refresh managed account profile right now.")
        c.info("Run 'carl camp account' to inspect the latest managed state.")
        c.info("Run 'carl camp upgrade' to open the upgrade page.")
        c.blank()
        return

    received_token: dict[str, str] = {}
    server_ready = threading.Event()

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            if "access_token" in params:
                received_token["jwt"] = params["access_token"][0]
                if "refresh_token" in params:
                    received_token["refresh"] = params["refresh_token"][0]

                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"""<html><body style="background:#0A0F0A;color:#E8E8D8;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
                <div style="text-align:center"><h2 style="color:#FF6B35">Welcome to Camp CARL</h2><p>You can close this tab.</p></div>
                </body></html>""")
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing token")

        def log_message(self, format: str, *args: object) -> None:
            pass  # Suppress HTTP logs

    # Start local callback server
    server = http.server.HTTPServer(("127.0.0.1", 0), CallbackHandler)
    port = server.server_address[1]
    callback_url = f"http://127.0.0.1:{port}"

    def serve() -> None:
        server_ready.set()
        server.handle_request()  # Handle one request then stop

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    server_ready.wait()

    # Open browser
    login_url = f"{DEFAULT_CARL_CAMP_BASE}/auth/login?callback={urllib.parse.quote(callback_url)}"

    if upgrade:
        login_url += "&upgrade=true"

    c.info(f"Opening browser: {login_url}")
    webbrowser.open(login_url)
    c.info("Waiting for authentication...")

    # Wait for callback (timeout 120s)
    thread.join(timeout=120)
    server.server_close()

    if received_token.get("jwt"):
        db.set_auth("jwt", received_token["jwt"], ttl_hours=24)
        db.set_config("supabase_url", DEFAULT_CARL_CAMP_SUPABASE_URL)

        c.ok("Authenticated successfully!")
        c.info("Your session is cached locally for 24 hours.")
        try:
            _session, profile, source = resolve_camp_profile(refresh=True, db=db)
            if profile is not None:
                c.info(f"Account: {profile.tier.title()} ({profile.status}) via {source}")
                if profile.plan:
                    c.info(f"Plan: {profile.plan}")
                c.info("Inspect account: carl camp account")
        except CampError:
            c.info("Inspect account later with: carl camp account")
        c.blank()
    else:
        c.error("Authentication timed out or failed.")
        c.info("Try again: carl camp login")
        c.blank()
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# carl logout — clear cached carl.camp session
# ---------------------------------------------------------------------------


@app.command(name="logout", hidden=True)
def logout(
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Clear the cached carl.camp session and return to local-first FREE mode."""
    from carl_studio.db import LocalDB

    c = get_console()
    _warn_legacy_command_alias(c, ctx, "carl camp logout")
    db = LocalDB()

    had_session = bool(db.get_auth("jwt") or db.get_auth("tier") or db.get_config("camp_profile"))
    if not had_session:
        c.info("No local camp session cached.")
        raise typer.Exit(0)

    db.clear_auth()
    db.set_config("supabase_url", "")
    db.set_config("camp_profile", "")
    db.set_config("camp_profile_cached_at", "")

    c.ok("Local camp session cleared.")
    c.info("You are back in local-first FREE mode.")
    c.info("Re-attach managed platform features later with: carl camp login")
    c.blank()


# ---------------------------------------------------------------------------
# carl sync — push/pull data to carl.camp
# ---------------------------------------------------------------------------

sync_app = typer.Typer(
    name="sync",
    help="Push and pull local data with carl.camp.",
    no_args_is_help=True,
)
app.add_typer(sync_app, hidden=True)


@sync_app.callback()
def sync_callback(ctx: typer.Context = typer.Option(None, hidden=True)) -> None:
    """Warn when the legacy top-level sync group is used."""
    _warn_legacy_command_alias(get_console(), ctx, "carl camp sync")


@sync_app.command(name="push")
def sync_push_cmd(
    types: str = typer.Option(
        "runs", "--types", "-t", help="Entity types to push (comma-separated)"
    ),
) -> None:
    """Push local data to carl.camp."""
    c = get_console()
    from carl_studio.tier import check_tier, tier_message

    allowed, _, _ = check_tier("sync.cloud")
    if not allowed:
        c.error_with_hint(
            tier_message("sync.cloud") or "Cloud sync requires CARL Paid.",
            hint="Upgrade with: carl camp upgrade",
            signup_url="https://carl.camp/pricing",
            code="tier:sync.cloud",
        )
        raise typer.Exit(1)
    c.blank()
    c.header("CARL Sync", "Push")

    from carl_studio.sync import push, SyncError

    entity_types = [t.strip() for t in types.split(",")]

    try:
        results = push(entity_types=entity_types)
        for etype, count in results.items():
            if count > 0:
                c.ok(f"{etype}: {count} synced")
            else:
                c.info(f"{etype}: nothing to push")
    except SyncError as e:
        c.error(str(e))
        raise typer.Exit(1)

    c.blank()


@sync_app.command(name="pull")
def sync_pull_cmd(
    since: str = typer.Option("", "--since", "-s", help="Pull updates since ISO timestamp"),
    types: str = typer.Option("", "--types", "-t", help="Entity types to pull (comma-separated)"),
) -> None:
    """Pull updates from carl.camp."""
    c = get_console()
    from carl_studio.tier import check_tier, tier_message

    allowed, _, _ = check_tier("sync.cloud")
    if not allowed:
        c.error_with_hint(
            tier_message("sync.cloud") or "Cloud sync requires CARL Paid.",
            hint="Upgrade with: carl camp upgrade",
            signup_url="https://carl.camp/pricing",
            code="tier:sync.cloud",
        )
        raise typer.Exit(1)
    c.blank()
    c.header("CARL Sync", "Pull")

    from carl_studio.sync import pull, SyncError

    entity_types = [t.strip() for t in types.split(",")] if types else None

    try:
        results = pull(since=since or None, entity_types=entity_types)
        for etype, count in results.items():
            if count > 0:
                c.ok(f"{etype}: {count} pulled")
            else:
                c.info(f"{etype}: up to date")
    except SyncError as e:
        c.error(str(e))
        raise typer.Exit(1)

    c.blank()
