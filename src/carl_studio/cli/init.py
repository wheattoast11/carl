"""``carl init`` — one-shot onboarding wizard.

Rolls up signup/login, LLM provider detection, optional extras install,
project config, consent, and a freshness baseline into a single command.
Target: a fresh user runs ``pip install carl-studio && carl init`` and
is ready to train in under two minutes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, cast

import typer

from carl_core.interaction import ActionType, InteractionChain

from carl_studio.console import get_console
from carl_studio.settings import CARL_HOME, GLOBAL_CONFIG

FIRST_RUN_MARKER = CARL_HOME / ".initialized"
CONTEXT_FILE = CARL_HOME / "context.json"

# Provider -> default chat model. Keep in sync with
# ``carl_studio.chat_agent._MODEL_PRICING`` and the OpenRouter catalog.
_PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "Anthropic": "claude-sonnet-4-6",
    "OpenRouter": "openrouter/deepseek/deepseek-chat",
    "OpenAI": "gpt-4o-mini",
}


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
    provider_ok, provider_label = _ensure_llm_provider(chain)
    if provider_ok:
        steps_done.append("LLM provider")

    # 2b. Persist default_chat_model so bare `carl` just works.
    if _persist_default_chat_model(provider_label, chain):
        steps_done.append("default chat model")

    # 3. Optional extras
    if not skip_extras and _offer_extras(chain):
        steps_done.append("extras installed")

    # 4. Project
    if not skip_project and _ensure_project(chain):
        steps_done.append("project config")

    # 4b. Context gathering (GitHub repo / HF model). Runs BEFORE the
    # sample scaffold so the scaffold can pin base_model / dataset_repo
    # from context.json.
    if _offer_context_gathering(chain):
        steps_done.append("context gathered")

    # 4c. Optional sample-project scaffold (reads context.json).
    if not skip_project and _offer_sample_project(chain):
        steps_done.append("sample project scaffold")

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

    # Celebration + guidance. Informative only (no prompt); the user
    # picks their next command organically from the listed paths.
    _celebrate_and_guide(c)

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


def _ensure_llm_provider(chain: InteractionChain) -> tuple[bool, str | None]:
    """Resolve an LLM provider. Returns (success, provider_label_or_None).

    ``provider_label`` is the key into :data:`_PROVIDER_DEFAULT_MODEL` — used
    to write a sane ``default_chat_model`` so ``carl`` works immediately
    after ``carl init`` without any further config.
    """
    c = get_console()
    detected = _detect_any_provider()
    if detected:
        c.ok(f"LLM provider: {detected}")
        chain.record(
            ActionType.GATE, "llm_provider",
            input={"resolved_via": "detected"}, output={"provider": detected}, success=True,
        )
        return True, detected

    c.print("  [camp.primary]LLM provider[/]")
    c.print("  [1] Anthropic (Claude)")
    c.print("  [2] OpenRouter (any model)")
    c.print("  [3] OpenAI")
    c.print("  [4] Skip — configure later")
    choice = typer.prompt("  Pick one", default="1")

    try:
        from carl_studio.cli.prompt import require
    except ImportError:  # pragma: no cover
        return False, None

    provider_label: str | None
    try:
        if choice == "1":
            require("ANTHROPIC_API_KEY", chain=chain)
            provider_label = "Anthropic"
        elif choice == "2":
            require("OPENROUTER_API_KEY", chain=chain)
            provider_label = "OpenRouter"
        elif choice == "3":
            require("OPENAI_API_KEY", chain=chain)
            provider_label = "OpenAI"
        else:
            c.info("Skipped. Carl will prompt when a command needs a key.")
            chain.record(
                ActionType.GATE, "llm_provider",
                input={"resolved_via": "skipped"}, success=True,
            )
            return True, None
    except typer.Abort:
        c.info("Skipped. Carl will prompt when a command needs a key.")
        return False, None
    return True, provider_label


def _persisted_default_chat_model() -> str:
    """Read ``default_chat_model`` from the global config file, or ``""``.

    Kept separate from :func:`_persist_default_chat_model` so pyright can
    type the local YAML-unwrap without leaking Any into that function.
    """
    if not GLOBAL_CONFIG.is_file():
        return ""
    try:
        import yaml as _yaml

        parsed: Any = _yaml.safe_load(GLOBAL_CONFIG.read_text())
    except Exception:  # pragma: no cover — defensive
        return ""
    if not isinstance(parsed, dict):
        return ""
    value = cast(dict[str, Any], parsed).get("default_chat_model")
    return value if isinstance(value, str) else ""


def _persist_default_chat_model(provider_label: str | None, chain: InteractionChain) -> bool:
    """Persist ``default_chat_model`` to ~/.carl/config.yaml.

    After a provider is chosen, resolve a sane default chat model for that
    provider and save it via :meth:`CARLSettings.save`. This is what makes
    plain ``carl`` work immediately after ``carl init``.

    Returns ``True`` when a default was written, ``False`` when the step was
    a no-op (unknown provider, user already had a value, save failure).
    """
    if not provider_label:
        return False
    model = _PROVIDER_DEFAULT_MODEL.get(provider_label)
    if not model:
        return False

    try:
        from carl_studio.settings import CARLSettings

        settings = CARLSettings.load()
    except Exception as exc:  # pragma: no cover — defensive
        chain.record(
            ActionType.CLI_CMD, "persist_default_chat_model",
            output={"error": str(exc)}, success=False,
        )
        return False

    # Skip only when the persisted file already has our target value. The
    # schema default ``claude-sonnet-4-6`` is in-memory, not on disk — we
    # still want to write it so bare ``carl`` picks it up via the global
    # config layer without falling back to the schema.
    if _persisted_default_chat_model() == model:
        chain.record(
            ActionType.CLI_CMD, "persist_default_chat_model",
            input={"already_set": True, "model": model}, success=True,
        )
        return False

    try:
        settings.default_chat_model = model
        target = settings.save()
    except Exception as exc:  # pragma: no cover — defensive
        chain.record(
            ActionType.CLI_CMD, "persist_default_chat_model",
            output={"error": str(exc), "model": model}, success=False,
        )
        return False

    chain.record(
        ActionType.CLI_CMD, "persist_default_chat_model",
        input={"provider": provider_label, "model": model},
        output={"path": str(target)}, success=True,
    )
    c = get_console()
    c.info(f"Default chat model set: {model}")
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
# Step: sample project scaffold
# ---------------------------------------------------------------------------

# Minimal carl.yaml schema for the scaffold. Kept intentionally tiny —
# just enough to drive `carl train` end-to-end against a 1.1B TinyLlama
# checkpoint + wikitext. Full schema lives in carl_studio.project.
_SAMPLE_PROJECT_YAML = """\
name: carl-quickstart
description: Sample CARL project — TinyLlama + wikitext. Safe to delete.
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
output_repo: ''
compute_target: local
backend: local
dataset_repo: wikitext
method: sft
max_steps: 50
learning_rate: 2.0e-05
carl_enabled: true
hub_token_env: HF_TOKEN
tracking_url: null
stack:
  tools: []
  frameworks: []
  repos: []
  use_case: quickstart
"""


def _render_sample_project_yaml() -> tuple[str, dict[str, Any]]:
    """Render the scaffold YAML with context-aware overrides applied.

    F4 — when ``~/.carl/context.json`` exists, override the scaffold's
    ``base_model`` and ``dataset_repo`` fields with the user's declared
    ``hf_model`` and ``github_repo``. Empty / missing context leaves the
    hardcoded TinyLlama + wikitext baseline untouched. Returns the
    rendered YAML body and a structured record of which fields were
    overridden (for the InteractionChain trace).
    """
    import yaml as _yaml

    overrides: dict[str, Any] = {}
    # Parse the baseline once so we can mutate a dict and re-dump. The
    # keys here mirror ``_SAMPLE_PROJECT_YAML`` exactly; pyyaml preserves
    # insertion order so the rendered file stays human-readable.
    base: dict[str, Any] = _yaml.safe_load(_SAMPLE_PROJECT_YAML) or {}

    try:
        ctx = _load_context()
    except Exception:
        ctx = {}

    hf_model = ctx.get("hf_model") if isinstance(ctx, dict) else None
    gh_repo = ctx.get("github_repo") if isinstance(ctx, dict) else None

    if isinstance(hf_model, str) and hf_model.strip():
        base["base_model"] = hf_model.strip()
        overrides["base_model"] = hf_model.strip()
    if isinstance(gh_repo, str) and gh_repo.strip():
        # The scaffold's dataset_repo field is a free-form identifier —
        # HF dataset id or a user-supplied label. The GitHub repo is a
        # reasonable default when the user has not explicitly picked a
        # dataset, because it ties the scaffold to the user's real work.
        base["dataset_repo"] = gh_repo.strip()
        overrides["dataset_repo"] = gh_repo.strip()

    body = _yaml.safe_dump(base, sort_keys=False)
    return body, overrides


def _offer_sample_project(chain: InteractionChain) -> bool:
    """Scaffold a minimal ``carl.yaml`` + ``data/`` + ``outputs/`` structure.

    Skipped when the current directory already has a ``carl.yaml`` (the
    main project step already handled that case). Returns ``True`` when
    the scaffold was created, ``False`` on skip / decline / failure.
    """
    c = get_console()
    cwd = Path.cwd()
    if cwd.joinpath("carl.yaml").is_file():
        chain.record(
            ActionType.CLI_CMD, "sample_project",
            input={"skipped": "carl.yaml exists"}, success=True,
        )
        return False

    c.print("  [camp.primary]Sample project[/]")
    c.info("A tiny quickstart to try `carl train` immediately.")
    try:
        wanted = typer.confirm("  Create a sample training project? (quickstart)", default=False)
    except (typer.Abort, EOFError, OSError):
        wanted = False
    if not wanted:
        chain.record(
            ActionType.CLI_CMD, "sample_project",
            input={"answer": "no"}, success=True,
        )
        return False

    try:
        body, overrides = _render_sample_project_yaml()
        target = cwd / "carl.yaml"
        target.write_text(body)
        (cwd / "data").mkdir(exist_ok=True)
        (cwd / "outputs").mkdir(exist_ok=True)
    except OSError as exc:
        c.warn(f"Sample scaffold failed: {exc}")
        chain.record(
            ActionType.CLI_CMD, "sample_project",
            output={"error": str(exc)}, success=False,
        )
        return False

    c.ok(f"Sample project scaffolded in {cwd}")
    if overrides:
        if "base_model" in overrides:
            c.info(f"  base_model pinned from context: {overrides['base_model']}")
        if "dataset_repo" in overrides:
            c.info(f"  dataset_repo pinned from context: {overrides['dataset_repo']}")
    c.info("  Try: carl train --config carl.yaml")
    chain.record(
        ActionType.CLI_CMD, "sample_project",
        input={"answer": "yes", "overrides": overrides or None},
        output={"path": str(target)},
        success=True,
    )
    return True


# ---------------------------------------------------------------------------
# Step: context gathering
# ---------------------------------------------------------------------------

def _load_context() -> dict[str, Any]:
    """Read ``~/.carl/context.json``; return ``{}`` if missing/malformed."""
    empty: dict[str, Any] = {}
    if not CONTEXT_FILE.is_file():
        return empty
    try:
        raw = CONTEXT_FILE.read_text()
    except OSError:
        return empty
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        return empty
    if not isinstance(data, dict):
        return empty
    # Cast narrows json.loads's Any-valued output to the return type.
    return cast(dict[str, Any], data)


def _save_context(data: dict[str, Any]) -> Path:
    """Persist user-supplied context to ``~/.carl/context.json``."""
    CARL_HOME.mkdir(parents=True, exist_ok=True)
    CONTEXT_FILE.write_text(json.dumps(data, indent=2))
    return CONTEXT_FILE


def _offer_context_gathering(chain: InteractionChain) -> bool:
    """Ask the user to link a GitHub repo and/or HF model.

    Idempotent: when ``~/.carl/context.json`` already has values, we prompt
    "keep current context? (Y/n)" and bail unless the user says no. Empty
    answers are never stored — the file is only written when at least one
    field has a real value.
    """
    c = get_console()
    existing = _load_context()

    if existing.get("github_repo") or existing.get("hf_model"):
        c.print("  [camp.primary]Context[/]")
        if existing.get("github_repo"):
            c.info(f"  github_repo: {existing['github_repo']}")
        if existing.get("hf_model"):
            c.info(f"  hf_model:    {existing['hf_model']}")
        try:
            keep = typer.confirm("  Keep current context?", default=True)
        except (typer.Abort, EOFError, OSError):
            keep = True
        if keep:
            chain.record(
                ActionType.CLI_CMD, "context_gathering",
                input={"action": "kept_existing"}, success=True,
            )
            return False
        # fallthrough: user wants to edit

    try:
        wanted = typer.confirm(
            "  Have a GitHub repo or HF model you want to work with?",
            default=False,
        )
    except (typer.Abort, EOFError, OSError):
        # Non-interactive or abort -> skip quietly.
        wanted = False
    if not wanted:
        chain.record(
            ActionType.CLI_CMD, "context_gathering",
            input={"answer": "no"}, success=True,
        )
        return False

    try:
        github_repo = typer.prompt(
            "  GitHub repo (user/repo or URL; blank to skip)",
            default="",
            show_default=False,
        ).strip()
    except (typer.Abort, EOFError, OSError):
        github_repo = ""
    try:
        hf_model = typer.prompt(
            "  HF model (user/model or URL; blank to skip)",
            default="",
            show_default=False,
        ).strip()
    except (typer.Abort, EOFError, OSError):
        hf_model = ""

    data: dict[str, Any] = {}
    if github_repo:
        data["github_repo"] = github_repo
    if hf_model:
        data["hf_model"] = hf_model

    if not data:
        c.info("No context captured (both answers blank).")
        chain.record(
            ActionType.CLI_CMD, "context_gathering",
            input={"answer": "empty"}, success=True,
        )
        return False

    try:
        target = _save_context(data)
    except OSError as exc:
        c.warn(f"Context save failed: {exc}")
        chain.record(
            ActionType.CLI_CMD, "context_gathering",
            output={"error": str(exc)}, success=False,
        )
        return False

    c.ok(f"Context saved: {target}")
    chain.record(
        ActionType.CLI_CMD, "context_gathering",
        input={"fields": sorted(data.keys())},
        output={"path": str(target)},
        success=True,
    )
    return True


# ---------------------------------------------------------------------------
# Step: celebration + guidance
# ---------------------------------------------------------------------------

def _celebrate_and_guide(c: Any) -> None:
    """Print an achievement panel + next-step paths (no input required)."""
    # Compact guidance block — 4 paths covering the main lanes. The user
    # picks one organically; we do not prompt. Mirrors the `carl` intro
    # moves so the mental model stays stable across surfaces.
    c.print("  [camp.accent]You're all set![/]")
    c.info("You earned the \"first-run\" badge. Here's what to try next:")
    c.kv("carl \"explore my repo\"", "start a conversation")
    c.kv("carl train --config carl.yaml", "run training")
    c.kv("carl doctor", "audit your readiness")
    c.kv("carl queue add \"idea\"", "drop a sticky-note for the heartbeat")
    c.blank()


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
    "CONTEXT_FILE",
    "_first_run_complete",
    "_mark_first_run_complete",
    "_persist_default_chat_model",
    "_offer_sample_project",
    "_offer_context_gathering",
    "_render_sample_project_yaml",
    "_celebrate_and_guide",
    "_load_context",
    "_save_context",
]
