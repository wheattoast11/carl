"""Alias wiring and optional integration sub-app registration.

All CLI command modules live inside this package (carl_studio.cli.*).
Domain logic stays in carl_studio.* — this package imports from there.

When an optional sub-module cannot be imported, a lightweight stub command
is registered that tells the user what to install instead of silently
hiding the feature.
"""

from __future__ import annotations

import typer

from . import apps as _apps_mod
from .apps import app, camp_app, lab_app
from .lab import (
    admin_app,
    align_cmd,
    bench_cmd,
    dev,
    golf_app,
    learn_cmd,
    mcp_serve,
    paper_app,
    repl_removed,
)
from .flow import flow_cmd
from .init import init_cmd
from .platform import login, logout, sync_app
from .queue import queue_app


# ---------------------------------------------------------------------------
# Stub factory — keeps each fallback to two lines at the call site
# ---------------------------------------------------------------------------

_BASE_HINT = "pip install 'carl-studio'"
_TRAINING_HINT = "pip install 'carl-studio[training]'"


def _make_stub(
    target_app: typer.Typer,
    name: str,
    *,
    doc: str,
    hint: str = _BASE_HINT,
    hidden: bool = False,
) -> None:
    """Register a lightweight stub command that prints an install hint and exits."""

    @target_app.command(name=name, hidden=hidden)
    def _stub() -> None:  # noqa: D401
        """"""
        typer.echo(f"{doc}")
        typer.echo(f"Install: {hint}")
        raise typer.Exit(1)

    _stub.__doc__ = f"{doc} (not installed)"


# ---------------------------------------------------------------------------
# Register camp/lab aliases for the converged command map
# ---------------------------------------------------------------------------

app.command(name="init")(init_cmd)
app.command(name="flow")(flow_cmd)
# `carl queue` — user-facing sticky-note work inbox. Not optional.
app.add_typer(queue_app, name="queue")
# v0.19 anticipatory coherence (FREE tier) — trinity + substrate health.
from .forecast import forecast_app  # noqa: E402
from .substrate import substrate_app  # noqa: E402

app.add_typer(forecast_app, name="forecast")
app.add_typer(substrate_app, name="substrate")
# Also expose init/flow under `camp` so both `carl init` and `carl camp init` resolve.
camp_app.command(name="init")(init_cmd)
camp_app.command(name="flow")(flow_cmd)
camp_app.command(name="login")(login)
camp_app.command(name="logout")(logout)
app.command(name="logout", hidden=True)(logout)
camp_app.add_typer(sync_app, name="sync")

lab_app.command(name="dev")(dev)
# F1 — `carl lab repl` is removed. Kept as a hidden stub that tells the user
# where the canonical surface lives so existing muscle memory does not dead-end.
lab_app.command(name="repl", hidden=True)(repl_removed)
lab_app.command(name="bench")(bench_cmd)
lab_app.command(name="align")(align_cmd)
lab_app.command(name="learn")(learn_cmd)
lab_app.command(name="mcp")(mcp_serve)
try:
    from .training import backends_cmd as _backends_cmd

    lab_app.command(name="backends")(_backends_cmd)
except ImportError:
    _make_stub(
        lab_app,
        "backends",
        doc="Backend adapter registry requires the full carl-studio package.",
    )
lab_app.add_typer(golf_app, name="golf")
lab_app.add_typer(paper_app, name="paper")
lab_app.add_typer(admin_app, name="admin")


# ---------------------------------------------------------------------------
# Wire research sub-app
# ---------------------------------------------------------------------------
try:
    from carl_studio.research._cli import research_app

    app.add_typer(research_app, name="research")
    lab_app.add_typer(research_app, name="research")
except ImportError:
    _make_stub(
        app,
        "research",
        doc="Research requires the arxiv package.",
        hint="pip install 'carl-studio[research]'",
    )
    _make_stub(
        lab_app,
        "research",
        doc="Research requires the arxiv package.",
        hint="pip install 'carl-studio[research]'",
    )


# ---------------------------------------------------------------------------
# Wire skills sub-app
# ---------------------------------------------------------------------------
try:
    from carl_studio.skills._cli import skills_app

    app.add_typer(skills_app, name="skill", hidden=True)
    lab_app.add_typer(skills_app, name="skill")
except ImportError:
    _make_stub(lab_app, "skill", doc="Skill commands require the full carl-studio package.")


# ---------------------------------------------------------------------------
# Wire A2A agent sub-app
# ---------------------------------------------------------------------------
try:
    from carl_studio.a2a._cli import agent_app

    app.add_typer(agent_app, name="agent", hidden=True)
    lab_app.add_typer(agent_app, name="agent")
except ImportError:
    _make_stub(lab_app, "agent", doc="Agent commands require the full carl-studio package.")


# ---------------------------------------------------------------------------
# Wire billing commands
# ---------------------------------------------------------------------------
try:
    from .billing import account_status, billing_portal, subscription_status, upgrade

    # WS-D4: canonical paths are `carl camp account|upgrade|billing|subscription`.
    # The hidden top-level aliases (carl upgrade / carl billing / carl subscription)
    # were removed to end the duplicate registration.
    camp_app.command(name="account")(account_status)
    camp_app.command(name="upgrade")(upgrade)
    camp_app.command(name="billing")(billing_portal)
    camp_app.command(name="subscription")(subscription_status)
except ImportError:
    _make_stub(camp_app, "account", doc="Billing commands require the full carl-studio package.")
    _make_stub(camp_app, "upgrade", doc="Billing commands require the full carl-studio package.")
    _make_stub(camp_app, "billing", doc="Billing commands require the full carl-studio package.")
    _make_stub(
        camp_app, "subscription", doc="Billing commands require the full carl-studio package."
    )


# ---------------------------------------------------------------------------
# Wire credits sub-app
# ---------------------------------------------------------------------------
try:
    from carl_studio.credits._cli import credits_app

    app.add_typer(credits_app, name="credits", hidden=True)
    camp_app.add_typer(credits_app, name="credits")
except ImportError:
    _make_stub(camp_app, "credits", doc="Credits commands require the full carl-studio package.")


# ---------------------------------------------------------------------------
# Wire curriculum sub-app + infer command
# ---------------------------------------------------------------------------
try:
    from .curriculum import curriculum_app

    app.add_typer(curriculum_app, name="curriculum", hidden=True)
    lab_app.add_typer(curriculum_app, name="curriculum")
except ImportError:
    _make_stub(
        lab_app, "curriculum", doc="Curriculum commands require the full carl-studio package."
    )

try:
    from .infer import infer_cmd

    app.command(name="infer")(infer_cmd)
except ImportError:
    _make_stub(
        app,
        "infer",
        doc="Inference requires the training extras.",
        hint=_TRAINING_HINT,
    )


# ---------------------------------------------------------------------------
# Wire research-cycle verbs: hypothesize + commit
# ---------------------------------------------------------------------------
try:
    from .hypothesize import hypothesize_cmd

    app.command(name="hypothesize")(hypothesize_cmd)
except ImportError:
    _make_stub(app, "hypothesize", doc="Hypothesize requires the full carl-studio.")

try:
    from .commit import commit_cmd

    app.command(name="commit")(commit_cmd)
except ImportError:
    _make_stub(app, "commit", doc="Commit requires the full carl-studio.")


# ---------------------------------------------------------------------------
# Wire carl update (v0.11)
# ---------------------------------------------------------------------------
try:
    from .update import update_cmd

    app.command(name="update")(update_cmd)
except ImportError:
    _make_stub(app, "update", doc="Update requires the full carl-studio.")


# ---------------------------------------------------------------------------
# Wire carl env (v0.12)
# ---------------------------------------------------------------------------
try:
    from .env import env_cmd

    app.command(name="env")(env_cmd)
except ImportError:
    _make_stub(app, "env", doc="Env wizard requires the full carl-studio.")


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
    _make_stub(
        camp_app,
        "marketplace",
        doc="Marketplace commands require the full carl-studio package.",
    )
    _make_stub(
        camp_app,
        "publish",
        doc="Marketplace commands require the full carl-studio package.",
    )


# ---------------------------------------------------------------------------
# Wire consent sub-app
# ---------------------------------------------------------------------------
try:
    from .consent import consent_app

    camp_app.add_typer(consent_app, name="consent")
except ImportError:
    _make_stub(camp_app, "consent", doc="Consent commands require the full carl-studio package.")


# ---------------------------------------------------------------------------
# Wire contract sub-app
# ---------------------------------------------------------------------------
try:
    from .contract import contract_app

    camp_app.add_typer(contract_app, name="contract")
except ImportError:
    _make_stub(
        camp_app, "contract", doc="Contract commands require the full carl-studio package."
    )


# ---------------------------------------------------------------------------
# Wire x402 sub-app
# ---------------------------------------------------------------------------
try:
    from .x402 import x402_app

    camp_app.add_typer(x402_app, name="x402")
except ImportError:
    _make_stub(camp_app, "x402", doc="x402 payment commands require the full carl-studio package.")


# ---------------------------------------------------------------------------
# Wire wallet sub-app
# ---------------------------------------------------------------------------
try:
    from .wallet import wallet_app

    camp_app.add_typer(wallet_app, name="wallet")
except ImportError:
    _make_stub(
        camp_app,
        "wallet",
        doc="Wallet commands require: pip install 'carl-studio[wallet]'",
        hint="pip install 'carl-studio[wallet]'",
    )


# ---------------------------------------------------------------------------
# Wire carlito sub-app
# ---------------------------------------------------------------------------
try:
    from .carlito import carlito_app

    app.add_typer(carlito_app, name="carlito", hidden=True)
    lab_app.add_typer(carlito_app, name="carlito")
except ImportError:
    _make_stub(lab_app, "carlito", doc="Carlito commands require the full carl-studio package.")


# ---------------------------------------------------------------------------
# Wire frame sub-app
# ---------------------------------------------------------------------------
try:
    from .frame import frame_app

    app.add_typer(frame_app, name="frame")
except ImportError:
    _make_stub(app, "frame", doc="Frame commands require the full carl-studio package.")


# ---------------------------------------------------------------------------
# Wire metrics sub-app — Prometheus scrape endpoint
# ---------------------------------------------------------------------------
try:
    from carl_studio.cli.metrics import metrics_app

    app.add_typer(metrics_app, name="metrics")
except ImportError:
    # metrics extra not installed; leave the verb out of the surface so
    # `carl metrics serve` prints its own hint when the user invokes it.
    pass


# ---------------------------------------------------------------------------
# Wire carl session (v0.18 Track D — per-project CLI sessions)
# ---------------------------------------------------------------------------
try:
    from .session_cmd import session_app

    app.add_typer(session_app, name="session")
except ImportError:
    _make_stub(app, "session", doc="Session commands require the full carl-studio package.")


# ---------------------------------------------------------------------------
# Wire carl trust (v0.18 Track B — bare-entry project trust)
# ---------------------------------------------------------------------------
try:
    from .trust import trust_app

    app.add_typer(trust_app, name="trust")
except ImportError:
    _make_stub(app, "trust", doc="Trust commands require the full carl-studio package.")


# ---------------------------------------------------------------------------
# Wire carl chat (top-level agentic chat)
# ---------------------------------------------------------------------------
# F1 (v0.7) — Two canonical surfaces: `carl chat` (interactive) and
# `carl ask "<prompt>"` (one-shot). The legacy `carl lab repl` alias is
# removed.
#
# v0.18 Track A — bare ``carl`` now delegates to
# :func:`carl_studio.cli.entry.route`, which is the single decision point
# for all entry modes:
#
# * empty argv (TTY)            → ``chat_cmd`` REPL
# * empty argv (non-TTY / piped) → help + nudge
# * ``carl "<prompt>"``          → ``chat_cmd(initial_message=prompt)``
# * ``carl -p "<prompt>"``       → ``ask_cmd``
# * ``~/.carl/.initialized`` missing on a TTY → ``init_cmd`` then exit
#
# The router returns ``True`` when it handled the invocation end-to-end;
# ``False`` means "fall through to Typer help". Subcommand dispatch is
# always owned by Typer — the callback only fires when
# ``ctx.invoked_subcommand is None``.
try:
    from .chat import ask_cmd, chat_cmd

    app.command(name="chat")(chat_cmd)
    app.command(name="ask")(ask_cmd)

    @app.callback(invoke_without_command=True)
    def _route_bare_carl(  # pyright: ignore[reportUnusedFunction]
        ctx: typer.Context,
        version: bool = typer.Option(
            False,
            "--version",
            "-V",
            callback=_apps_mod._version_callback,  # pyright: ignore[reportPrivateUsage]
            is_eager=True,
            help="Print carl-studio version and exit.",
        ),
    ) -> None:
        """Delegate bare ``carl`` invocations to the unified router.

        Also attaches the eager ``--version`` / ``-V`` option here because
        Typer only supports one root callback; see ``apps.py`` for the
        callback body.
        """
        del version  # eager callback handles it; suppress pyright unused
        if ctx.invoked_subcommand is not None:
            return

        # Lazy import so the wiring module stays cheap to load when the
        # router is not needed (e.g. on ``carl <verb>``).
        from carl_studio.cli.entry import route

        # The callback fires with ``sys.argv`` already stripped of the
        # program name and of any consumed group-level options. The
        # router inspects the full slice so it can reach the positional
        # prompt / ``-p`` shapes that Typer's callback context does not
        # expose directly.
        import sys as _sys

        handled = route(list(_sys.argv[1:]))
        if handled:
            return

        # Router returned False → fall back to today's help + nudge.
        # This keeps ``carl`` inside a piped script visually identical
        # to v0.17.x for operators who rely on the existing output.
        typer.echo(ctx.get_help())
        typer.echo(
            "\nUse `carl chat` for an interactive session or "
            "`carl ask \"<prompt>\"` for one-shot."
        )

except ImportError:
    _make_stub(app, "chat", doc="Chat requires the full carl-studio package.")
    _make_stub(app, "ask", doc="Ask requires the full carl-studio package.")
