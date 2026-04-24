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

from carl_studio.cli import ui
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

    # 6b. v0.18 Track B safety net — if any previous step wrote
    # carl.yaml without going through ``_ensure_project`` /
    # ``_offer_sample_project`` (e.g. a custom ``project_init`` path that
    # bypasses the scaffold helper), make sure .carl/ exists before we
    # mark first-run complete. Idempotent — the inner scaffold call is
    # a no-op when the marker already exists.
    if _has_project_config():
        _scaffold_project_marker(Path.cwd(), chain)

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
    action = ui.select(
        "How do you want to sign in?",
        [
            ui.Choice(value="sign_in", label="Sign in with browser", badge="recommended"),
            ui.Choice(
                value="create_account",
                label="Create an account",
                hint="opens carl.camp/auth/signup",
            ),
            ui.Choice(value="skip", label="Skip — configure later"),
        ],
        default=0,
    )

    if action == "skip":
        chain.record(
            ActionType.GATE,
            "camp_login",
            input={"resolved_via": "skipped"},
            success=True,
        )
        return False

    if action == "create_account":
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
    choice = ui.select(
        "Which provider do you want Carl to use by default?",
        [
            ui.Choice(value="anthropic", label="Anthropic (Claude)", badge="recommended"),
            ui.Choice(value="openrouter", label="OpenRouter", hint="any model, unified API"),
            ui.Choice(value="openai", label="OpenAI"),
            ui.Choice(value="skip", label="Skip — configure later"),
        ],
        default=0,
    )

    try:
        from carl_studio.cli.prompt import require
    except ImportError:  # pragma: no cover
        return False, None

    provider_label: str | None
    try:
        if choice == "anthropic":
            require("ANTHROPIC_API_KEY", chain=chain)
            provider_label = "Anthropic"
        elif choice == "openrouter":
            require("OPENROUTER_API_KEY", chain=chain)
            provider_label = "OpenRouter"
        elif choice == "openai":
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

# Packages the training extra pulls. Keep in sync with pyproject.toml's
# [project.optional-dependencies].training list. `huggingface_hub` isn't
# in that list directly but it's the #1 sibling-corruption source — we
# probe it so we can surface + auto-heal corruption detected via the
# transformers import_value_error path.
_TRAINING_EXTRA_PACKAGES = (
    ("torch", None),
    ("transformers", None),
    ("huggingface_hub", None),
)


def _probe_training_extras() -> list[Any]:
    """Probe every training-extras package and return their DepProbeResults."""
    from carl_core.dependency_probe import probe

    return [probe(name, import_name=import_name) for name, import_name in _TRAINING_EXTRA_PACKAGES]


def _training_extras_installed() -> bool:
    """True iff every training-extras package is healthy.

    Thin bool wrapper over :func:`_probe_training_extras` for callers that
    only want a yes/no (e.g. ``carl doctor`` summary).
    """
    return all(r.healthy for r in _probe_training_extras())


def _offer_extras(chain: InteractionChain) -> bool:
    """Walk the user through fresh-install / auto-heal / skip branches.

    Three states fan out from probing every training-extras package:

    - All healthy → skip install; record + return ``True``.
    - Any corrupt sibling (the HF scenario) → offer to run
      ``pip install --force-reinstall --no-deps <pkg>`` per probe's
      ``repair_command``. Never silent — user must consent. Re-probe after.
    - Any package genuinely missing → offer the bulk
      ``pip install 'carl-studio[training]'`` (current path).
    """
    c = get_console()
    probes = _probe_training_extras()
    unhealthy = [r for r in probes if not r.healthy]

    if not unhealthy:
        c.ok("Training extras already installed.")
        chain.record(
            ActionType.CLI_CMD, "install_extras",
            input={"already_present": True}, success=True,
        )
        return True

    # Auto-heal branch: any probe that needs_repair (installed-but-broken)
    # is distinct from missing. We surface both, but offer the auto-heal
    # first because it's the faster recovery path.
    repair_targets = _collect_repair_targets(probes)
    if repair_targets:
        return _offer_auto_heal(chain, probes, repair_targets)

    # Fresh-install branch — only truly missing packages.
    return _offer_fresh_install(chain, probes)


def _collect_repair_targets(probes: list[Any]) -> list[tuple[str, str]]:
    """Return list of ``(target_name, repair_command)`` for corrupt deps.

    For ``import_value_error`` results (the HF scenario), the probe's
    own ``repair_command`` targets the wrong package — the sibling
    named in the error is what's actually broken. We parse the sibling
    out and substitute. Each ``(target_name, repair_command)`` appears
    at most once even if multiple probes blame the same sibling.
    """
    from carl_core.dependency_probe import extract_corrupt_sibling, probe

    seen: set[str] = set()
    targets: list[tuple[str, str]] = []

    for r in probes:
        if r.healthy or r.is_missing:
            continue

        if r.status == "import_value_error" and r.import_error:
            sibling = extract_corrupt_sibling(r.import_error)
            if sibling:
                sibling_probe = probe(sibling)
                if sibling_probe.needs_repair and sibling_probe.normalized_name not in seen:
                    seen.add(sibling_probe.normalized_name)
                    targets.append(
                        (sibling_probe.normalized_name, sibling_probe.repair_command)
                    )
                continue

        # Other broken states: use the probe's own repair command.
        if r.normalized_name not in seen:
            seen.add(r.normalized_name)
            targets.append((r.normalized_name, r.repair_command))

    return targets


def _offer_auto_heal(
    chain: InteractionChain,
    probes: list[Any],
    targets: list[tuple[str, str]],
) -> bool:
    c = get_console()
    c.print("  [camp.primary]Training extras[/]")
    c.warn("One or more packages appear corrupt (stale dist-info or half-finished upgrade).")
    for r in probes:
        if r.healthy:
            c.ok(f"  ✓ {r.normalized_name:24} {r.import_version or '?'}")
        elif r.is_missing:
            c.info(f"  · {r.normalized_name:24} (not installed)")
        else:
            detail = (r.import_error or r.metadata_error or r.status).splitlines()[0][:80]
            c.warn(f"  ✗ {r.normalized_name:24} {detail}")

    c.blank()
    c.info("Suggested repair:")
    for name, cmd in targets:
        c.print(f"    [camp.primary]{cmd}[/]")

    try:
        run_repair = ui.confirm("  Run the repair now?", default=True)
    except (typer.Abort, EOFError, OSError):
        run_repair = False

    if not run_repair:
        c.info("Skipped. You can run the commands above manually.")
        chain.record(
            ActionType.CLI_CMD, "install_extras",
            input={"action": "auto_heal_declined", "targets": [t[0] for t in targets]},
            success=True,
        )
        return False

    failed: list[str] = []
    for name, cmd in targets:
        c.info(f"Running: {cmd}")
        try:
            subprocess.run(cmd.split(), check=True)
        except subprocess.CalledProcessError as exc:
            c.warn(f"  failed (exit {exc.returncode}): {cmd}")
            failed.append(name)

    chain.record(
        ActionType.EXTERNAL, "pip_auto_heal",
        input={"targets": [t[0] for t in targets]},
        output={"failed": failed},
        success=not failed,
    )

    if failed:
        c.warn(
            "Some repairs failed. Try running the failing commands manually, "
            "then re-run `carl init --force`."
        )
        return False

    # Re-probe to confirm; if still unhealthy, fall through to fresh-install.
    post = _probe_training_extras()
    if all(r.healthy for r in post):
        c.ok("Training extras healthy after repair.")
        return True
    # Still broken — more likely a missing package now; hand off.
    return _offer_fresh_install(chain, post)


def _offer_fresh_install(chain: InteractionChain, probes: list[Any]) -> bool:
    c = get_console()
    c.print("  [camp.primary]Training extras[/]")
    c.print("  Torch, transformers, trl, peft — needed for `carl train`.")
    missing = [r.normalized_name for r in probes if r.is_missing]
    if missing:
        c.info(f"  missing: {', '.join(missing)}")
    try:
        install = ui.confirm("  Install training extras now?", default=False)
    except (typer.Abort, EOFError, OSError):
        install = False

    if not install:
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
        # v0.18: guarantee the .carl/ skeleton exists alongside an
        # already-present carl.yaml. Upgrades from v0.17.x don't have
        # the marker yet; this scaffolds it on first init re-run so
        # the project-context gate resolves correctly.
        _scaffold_project_marker(Path.cwd(), chain)
        chain.record(
            ActionType.CLI_CMD, "project_init",
            input={"already_present": True}, success=True,
        )
        return True

    if not ui.confirm("  Initialize carl.yaml in current directory?", default=True):
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
    if success:
        # v0.18 Track B: mint .carl/ marker so project_context.current()
        # can discover this project from any nested CWD.
        _scaffold_project_marker(Path.cwd(), chain)
    chain.record(ActionType.CLI_CMD, "project_init", success=success)
    return success


def _scaffold_project_marker(cwd: Path, chain: InteractionChain) -> None:
    """Mint the ``.carl/`` skeleton alongside ``carl.yaml``.

    Thin shim over :func:`carl_studio.project_context.scaffold`. Idempotent
    — re-running on an already-scaffolded project is a no-op. Errors are
    recorded on the chain but never propagated: scaffold failure should
    not abort init (the yaml is already on disk; the marker can be
    recovered with ``mkdir .carl`` manually).
    """
    try:
        from carl_studio import project_context

        target = project_context.scaffold(cwd)
    except Exception as exc:  # pragma: no cover — defensive
        chain.record(
            ActionType.CLI_CMD, "scaffold_marker",
            output={"error": str(exc)}, success=False,
        )
        return
    chain.record(
        ActionType.CLI_CMD, "scaffold_marker",
        output={"path": str(target)}, success=True,
    )


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
        wanted = ui.confirm("  Create a sample training project? (quickstart)", default=False)
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

    # v0.18 Track B: also mint .carl/ so project_context.current()
    # can discover the sample project via walk-up.
    _scaffold_project_marker(cwd, chain)

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
            keep = ui.confirm("  Keep current context?", default=True)
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
        wanted = ui.confirm(
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
        github_repo = ui.text(
            "  GitHub repo (user/repo or URL; blank to skip)",
            default="",
        ).strip()
    except (typer.Abort, EOFError, OSError):
        github_repo = ""
    try:
        hf_model = ui.text(
            "  HF model (user/model or URL; blank to skip)",
            default="",
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
    "_training_extras_installed",
    "_probe_training_extras",
    "_offer_extras",
    "_save_context",
]
