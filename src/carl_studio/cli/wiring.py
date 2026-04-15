"""Alias wiring and optional integration sub-app registration.

All CLI command modules live inside this package (carl_studio.cli.*).
Domain logic stays in carl_studio.* — this package imports from there.
"""

from __future__ import annotations

from .apps import app, camp_app, lab_app
from .lab import (
    admin_app,
    align_cmd,
    bench_cmd,
    chat,
    dev,
    golf_app,
    learn_cmd,
    mcp_serve,
    paper_app,
)
from .platform import login, logout, sync_app

# ---------------------------------------------------------------------------
# Register camp/lab aliases for the converged command map
# ---------------------------------------------------------------------------

camp_app.command(name="login")(login)
camp_app.command(name="logout")(logout)
app.command(name="logout", hidden=True)(logout)
camp_app.add_typer(sync_app, name="sync")

lab_app.command(name="dev")(dev)
lab_app.command(name="chat")(chat)
lab_app.command(name="bench")(bench_cmd)
lab_app.command(name="align")(align_cmd)
lab_app.command(name="learn")(learn_cmd)
lab_app.command(name="mcp")(mcp_serve)
lab_app.add_typer(golf_app, name="golf")
lab_app.add_typer(paper_app, name="paper")
lab_app.add_typer(admin_app, name="admin")


# ---------------------------------------------------------------------------
# Wire skills sub-app
# ---------------------------------------------------------------------------
try:
    from carl_studio.skills._cli import skills_app

    app.add_typer(skills_app, name="skill", hidden=True)
    lab_app.add_typer(skills_app, name="skill")
except ImportError:
    pass  # carl-studio[skills] not installed


# ---------------------------------------------------------------------------
# Wire A2A agent sub-app
# ---------------------------------------------------------------------------
try:
    from carl_studio.a2a._cli import agent_app

    app.add_typer(agent_app, name="agent", hidden=True)
    lab_app.add_typer(agent_app, name="agent")
except ImportError:
    pass  # carl-studio a2a module not installed


# ---------------------------------------------------------------------------
# Wire billing commands
# ---------------------------------------------------------------------------
try:
    from .billing import account_status, billing_portal, subscription_status, upgrade

    app.command(name="upgrade", hidden=True)(upgrade)
    app.command(name="billing", hidden=True)(billing_portal)
    app.command(name="subscription", hidden=True)(subscription_status)
    camp_app.command(name="account")(account_status)
    camp_app.command(name="upgrade")(upgrade)
    camp_app.command(name="billing")(billing_portal)
    camp_app.command(name="subscription")(subscription_status)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Wire credits sub-app
# ---------------------------------------------------------------------------
try:
    from carl_studio.credits._cli import credits_app

    app.add_typer(credits_app, name="credits", hidden=True)
    camp_app.add_typer(credits_app, name="credits")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Wire curriculum sub-app + infer command
# ---------------------------------------------------------------------------
try:
    from .curriculum import curriculum_app

    app.add_typer(curriculum_app, name="curriculum", hidden=True)
    lab_app.add_typer(curriculum_app, name="curriculum")
except ImportError:
    pass

try:
    from .infer import infer_cmd

    app.command(name="infer")(infer_cmd)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Wire marketplace sub-app + publish command
# ---------------------------------------------------------------------------
try:
    from .marketplace import marketplace_app, publish_cmd

    app.add_typer(marketplace_app, name="marketplace", hidden=True)
    app.command(name="publish", hidden=True)(publish_cmd)
    camp_app.add_typer(marketplace_app, name="marketplace")
    camp_app.command(name="publish")(publish_cmd)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Wire consent sub-app
# ---------------------------------------------------------------------------
try:
    from .consent import consent_app

    camp_app.add_typer(consent_app, name="consent")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Wire contract sub-app
# ---------------------------------------------------------------------------
try:
    from .contract import contract_app

    camp_app.add_typer(contract_app, name="contract")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Wire x402 sub-app
# ---------------------------------------------------------------------------
try:
    from .x402 import x402_app

    camp_app.add_typer(x402_app, name="x402")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Wire carlito sub-app
# ---------------------------------------------------------------------------
try:
    from .carlito import carlito_app

    app.add_typer(carlito_app, name="carlito", hidden=True)
    lab_app.add_typer(carlito_app, name="carlito")
except ImportError:
    pass
