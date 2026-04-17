"""``carl init`` — one-shot onboarding wizard.

Rolls up signup/login, LLM provider detection, optional extras install,
project config, consent, and a freshness baseline into a single command.
Target: a fresh user runs ``pip install carl-studio && carl init`` and
is ready to train in under two minutes.
"""
from __future__ import annotations

import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

import typer

from carl_core.interaction import ActionType, InteractionChain

from carl_studio.console import get_console
from carl_studio.settings import CARL_HOME, GLOBAL_CONFIG

FIRST_RUN_MARKER = CARL_HOME / ".initialized"


def init_cmd(
    skip_extras: bool = typer.Option(
        False, "--skip-extras", help="Don't offer to install optional extras"
    ),
    skip_project: bool = typer.Option(
        False, "--skip-project", help="Don't offer to initialize a project config"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-run even if already initialized"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit a structured summary instead of the wizard UI"
    ),
) -> None:
    """Set you up in under a minute: account, provider, extras, project, consent."""
    c = get_console()
    chain = InteractionChain()
    chain.context["command"] = "carl init"

    if _first_run_complete() and not force:
        c.blank()
        c.info("Already initialized. Use `carl doctor` to audit or pass --force to re-run.")
        if json_output:
            typer.echo(_json_result(chain, status="already_initialized", steps_done=[]))
        raise typer.Exit(0)

    steps_done: list[str] = []

    c.blank()
    c.header("Welcome to Carl")
    c.info("Setting you up in under a minute.")
    c.blank()

    # 1. Account
    if _ensure_camp_account(chain):
        steps_done.append("carl.camp account")

    # 2. LLM provider
    if _ensure_llm_provider(chain):
        steps_done.append("LLM provider")

    # 3. Optional extras
    if not skip_extras and _offer_extras(chain):
        steps_done.append("extras installed")

    # 4. Project
    if not skip_project and _ensure_project(chain):
        steps_done.append("project config")

    # 5. Consent
    if _ensure_consent(chain):
        steps_done.append("consent recorded")

    # 6. Freshness baseline
    _baseline_freshness(chain)
    steps_done.append("freshness baseline")

    # 7. Mark initialized
    _mark_first_run_complete()
    chain.record(ActionType.CLI_CMD, "mark_first_run_complete", success=True)

    c.blank()
    c.ok("Ready.")
    c.info("Try: carl \"train a small model on gsm8k\"")
    c.info("Or:  carl chat")
    c.blank()

    if json_output:
        typer.echo(_json_result(chain, status="initialized", steps_done=steps_done))


# ---------------------------------------------------------------------------
# First-run marker
# ---------------------------------------------------------------------------

def _first_run_complete() -> bool:
    return FIRST_RUN_MARKER.is_file()


def _mark_first_run_complete() -> None:
    CARL_HOME.mkdir(parents=True, exist_ok=True)
    FIRST_RUN_MARKER.touch()


def _clear_first_run_marker() -> None:
    """Test / admin helper — undo ``_mark_first_run_complete()``."""
    try:
        FIRST_RUN_MARKER.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Step: carl.camp account
# ---------------------------------------------------------------------------

def _has_camp_session() -> bool:
    try:
        from carl_studio.db import LocalDB

        return bool(LocalDB().get_auth("jwt"))
    except Exception:  # pragma: no cover - defensive
        return False


def _ensure_camp_account(chain: InteractionChain) -> bool:
    c = get_console()
    if _has_camp_session():
        c.ok("carl.camp session active.")
        chain.record(ActionType.GATE, "camp_session", input={"resolved_via": "cache"}, success=True)
        return True

    c.print("  [camp.primary]carl.camp account[/]")
    has_account = typer.confirm("  Already have one?", default=False)

    if not has_account:
        c.info("Opening carl.camp signup in your browser.")
        try:
            webbrowser.open("https://carl.camp/auth/signup")
        except Exception:  # pragma: no cover - best effort
            pass
        c.info("  When you're done, press Enter to continue with login.")
        try:
            input()
        except EOFError:
            pass

    try:
        from carl_studio.cli.platform import login as login_cmd

        login_cmd()
    except typer.Exit:
        pass  # login raises Exit on completion/failure either way
    except Exception as exc:  # pragma: no cover - defensive
        c.warn(f"Login failed: {exc}")
        chain.record(
            ActionType.GATE, "camp_login", input={"resolved_via": "browser"},
            output={"error": str(exc)}, success=False,
        )
        return False

    success = _has_camp_session()
    chain.record(
        ActionType.GATE,
        "camp_login",
        input={"resolved_via": "browser"},
        success=success,
    )
    return success


# ---------------------------------------------------------------------------
# Step: LLM provider
# ---------------------------------------------------------------------------

def _detect_any_provider() -> str | None:
    """Return a label for whichever provider key we can find, or None."""
    from carl_studio.settings import CARLSettings

    try:
        settings = CARLSettings()
    except Exception:  # pragma: no cover
        return None

    import os

    if os.environ.get("ANTHROPIC_API_KEY") or settings.anthropic_api_key:
        return "Anthropic"
    if os.environ.get("OPENROUTER_API_KEY") or settings.openrouter_api_key:
        return "OpenRouter"
    if os.environ.get("OPENAI_API_KEY") or settings.openai_api_key:
        return "OpenAI"
    return None


def _ensure_llm_provider(chain: InteractionChain) -> bool:
    c = get_console()
    detected = _detect_any_provider()
    if detected:
        c.ok(f"LLM provider: {detected}")
        chain.record(
            ActionType.GATE, "llm_provider",
            input={"resolved_via": "detected"}, output={"provider": detected}, success=True,
        )
        return True

    c.print("  [camp.primary]LLM provider[/]")
    c.print("  [1] Anthropic (Claude)")
    c.print("  [2] OpenRouter (any model)")
    c.print("  [3] OpenAI")
    c.print("  [4] Skip — configure later")
    choice = typer.prompt("  Pick one", default="1")

    try:
        from carl_studio.cli.prompt import require
    except ImportError:  # pragma: no cover
        return False

    try:
        if choice == "1":
            require("ANTHROPIC_API_KEY", chain=chain)
        elif choice == "2":
            require("OPENROUTER_API_KEY", chain=chain)
        elif choice == "3":
            require("OPENAI_API_KEY", chain=chain)
        else:
            c.info("Skipped. Carl will prompt when a command needs a key.")
            chain.record(
                ActionType.GATE, "llm_provider",
                input={"resolved_via": "skipped"}, success=True,
            )
            return True
    except typer.Abort:
        c.info("Skipped. Carl will prompt when a command needs a key.")
        return False
    return True


# ---------------------------------------------------------------------------
# Step: training extras
# ---------------------------------------------------------------------------

def _training_extras_installed() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _offer_extras(chain: InteractionChain) -> bool:
    c = get_console()
    if _training_extras_installed():
        c.ok("Training extras already installed.")
        chain.record(
            ActionType.CLI_CMD, "install_extras",
            input={"already_present": True}, success=True,
        )
        return True

    c.print("  [camp.primary]Training extras[/]")
    c.print("  Torch, transformers, trl, peft — needed for `carl train`.")
    if not typer.confirm("  Install now?", default=False):
        c.info("Skipped. Install later with: pip install 'carl-studio[training]'")
        chain.record(
            ActionType.CLI_CMD, "install_extras",
            input={"answer": "no"}, success=True,
        )
        return False

    c.info("This takes 1–3 minutes.")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "carl-studio[training]"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        c.warn(f"pip install failed: {exc}")
        chain.record(
            ActionType.EXTERNAL, "pip_install",
            input={"target": "carl-studio[training]"},
            output={"error": exc.returncode}, success=False,
        )
        return False

    chain.record(
        ActionType.EXTERNAL, "pip_install",
        input={"target": "carl-studio[training]"}, success=True,
    )
    return True


# ---------------------------------------------------------------------------
# Step: project
# ---------------------------------------------------------------------------

def _has_project_config() -> bool:
    return Path.cwd().joinpath("carl.yaml").is_file()


def _ensure_project(chain: InteractionChain) -> bool:
    c = get_console()
    if _has_project_config():
        c.ok("carl.yaml found in current directory.")
        chain.record(
            ActionType.CLI_CMD, "project_init",
            input={"already_present": True}, success=True,
        )
        return True

    if not typer.confirm("  Initialize carl.yaml in current directory?", default=True):
        chain.record(
            ActionType.CLI_CMD, "project_init",
            input={"answer": "no"}, success=True,
        )
        return False

    try:
        from carl_studio.cli.project_data import project_init

        project_init(
            name="my-carl-project",
            model="",
            method="grpo",
            dataset="",
            output_repo="",
            compute="",
            description="",
            use_case="",
            output="carl.yaml",
            interactive=True,
        )
    except typer.Exit:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        c.warn(f"Project init failed: {exc}")
        chain.record(
            ActionType.CLI_CMD, "project_init",
            output={"error": str(exc)}, success=False,
        )
        return False

    success = _has_project_config()
    chain.record(ActionType.CLI_CMD, "project_init", success=success)
    return success


# ---------------------------------------------------------------------------
# Step: consent
# ---------------------------------------------------------------------------

def _consent_set() -> bool:
    try:
        from carl_studio.consent import ConsentManager

        state = ConsentManager().load()
        return any(
            flag.changed_at is not None
            for flag in (
                state.observability,
                state.telemetry,
                state.usage_analytics,
                state.contract_witnessing,
            )
        )
    except Exception:  # pragma: no cover - defensive
        return False


def _ensure_consent(chain: InteractionChain) -> bool:
    c = get_console()
    if _consent_set():
        c.ok("Consent preferences already on file.")
        chain.record(
            ActionType.GATE, "consent",
            input={"resolved_via": "cache"}, success=True,
        )
        return True

    try:
        from carl_studio.consent import ConsentManager

        ConsentManager().present_first_run()
    except Exception as exc:  # pragma: no cover - defensive
        c.warn(f"Consent prompt failed: {exc}")
        chain.record(
            ActionType.GATE, "consent",
            output={"error": str(exc)}, success=False,
        )
        return False

    chain.record(
        ActionType.GATE, "consent",
        input={"resolved_via": "prompt"}, success=True,
    )
    return True


# ---------------------------------------------------------------------------
# Step: freshness
# ---------------------------------------------------------------------------

def _baseline_freshness(chain: InteractionChain) -> None:
    try:
        from carl_studio.freshness import run_freshness_check

        report = run_freshness_check(force=True)
    except Exception as exc:  # pragma: no cover - defensive
        chain.record(
            ActionType.CLI_CMD, "freshness",
            output={"error": str(exc)}, success=False,
        )
        return
    chain.record(
        ActionType.CLI_CMD, "freshness",
        output={"summary": report.summary}, success=True,
    )


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _json_result(chain: InteractionChain, *, status: str, steps_done: list[str]) -> str:
    import json

    payload: dict[str, Any] = {
        "status": status,
        "steps_done": steps_done,
        "global_config": str(GLOBAL_CONFIG),
        "first_run_marker": str(FIRST_RUN_MARKER),
        "chain": chain.to_dict(),
    }
    return json.dumps(payload, indent=2)


__all__ = [
    "init_cmd",
    "FIRST_RUN_MARKER",
    "_first_run_complete",
    "_mark_first_run_complete",
]
