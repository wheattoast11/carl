"""In-flow credential + permission prompts for the CLI.

The `require()` helper turns "missing credential" into a resolvable
question instead of a terminal error. Resolution order:

  1. Environment variable
  2. Stored config (``~/.carl/carl.db`` or ``CARLSettings``)
  3. Inline prompt (optionally via browser OAuth)

Every call records a ``GATE`` step on the caller's ``InteractionChain``
when one is provided — so onboarding friction becomes observable.
"""
from __future__ import annotations

import os
import webbrowser
from time import monotonic
from typing import Callable

import typer

from carl_core.interaction import ActionType, InteractionChain

from carl_studio.console import CampConsole, get_console

__all__ = ["require", "RequireSpec", "known_keys"]


def _default_prompt(
    prompt: str,
    *,
    default: str = "",
    hide_input: bool = False,
    show_default: bool = True,
) -> str:
    """Adapter: route the default prompt through ``cli/ui.py``.

    Keeps the signature ``typer.prompt`` compatible (``hide_input``,
    ``show_default``) so ``require()`` callers + tests can keep injecting
    ``prompt_fn=typer.prompt``-shaped lambdas. Lazy-imports to avoid a
    cycle with ``cli/__init__``.
    """
    del show_default  # ui.text handles display itself
    from carl_studio.cli import ui

    return ui.text(prompt, default=default, secret=hide_input)


def _default_confirm(prompt: str, *, default: bool = True) -> bool:
    from carl_studio.cli import ui

    return ui.confirm(prompt, default=default)


class RequireSpec:
    """Configuration for a single credential prompt.

    Most callers should use the pre-baked specs in ``KNOWN_KEYS`` rather
    than constructing this directly.
    """

    __slots__ = ("key_name", "env_var", "signup_url", "hint", "via_login", "storage")

    def __init__(
        self,
        key_name: str,
        *,
        env_var: str,
        signup_url: str,
        hint: str = "",
        via_login: bool = False,
        storage: str = "config",
    ) -> None:
        self.key_name = key_name
        self.env_var = env_var
        self.signup_url = signup_url
        self.hint = hint
        self.via_login = via_login
        # storage: "config" (persistent), "auth" (TTL), "none" (ephemeral)
        self.storage = storage


# Canonical specs for credentials Carl knows about.
KNOWN_KEYS: dict[str, RequireSpec] = {
    "ANTHROPIC_API_KEY": RequireSpec(
        key_name="ANTHROPIC_API_KEY",
        env_var="ANTHROPIC_API_KEY",
        signup_url="https://console.anthropic.com/settings/keys",
        hint="Claude models — used by carl chat, the observer, and diagnose.",
    ),
    "HF_TOKEN": RequireSpec(
        key_name="HF_TOKEN",
        env_var="HF_TOKEN",
        signup_url="https://huggingface.co/settings/tokens",
        hint="Hugging Face token — push models, submit jobs, read gated datasets.",
    ),
    "OPENROUTER_API_KEY": RequireSpec(
        key_name="OPENROUTER_API_KEY",
        env_var="OPENROUTER_API_KEY",
        signup_url="https://openrouter.ai/keys",
        hint="OpenRouter — any model via one API.",
    ),
    "OPENAI_API_KEY": RequireSpec(
        key_name="OPENAI_API_KEY",
        env_var="OPENAI_API_KEY",
        signup_url="https://platform.openai.com/api-keys",
        hint="OpenAI API.",
    ),
    "CARL_CAMP": RequireSpec(
        key_name="CARL_CAMP",
        env_var="CARL_CAMP_JWT",
        signup_url="https://carl.camp/auth/signup",
        hint="carl.camp account — credits, managed training, marketplace.",
        via_login=True,
        storage="auth",
    ),
}


def known_keys() -> dict[str, RequireSpec]:
    """Return a copy of the pre-baked credential specs."""
    return dict(KNOWN_KEYS)


def require(
    key_name: str,
    *,
    env_var: str | None = None,
    signup_url: str | None = None,
    hint: str = "",
    via_login: bool = False,
    storage: str | None = None,
    chain: InteractionChain | None = None,
    console: CampConsole | None = None,
    prompt_fn: Callable[..., str] | None = None,
    confirm_fn: Callable[..., bool] | None = None,
) -> str:
    """Return a credential, prompting inline if missing.

    Parameters
    ----------
    key_name
        Canonical name, e.g. ``"ANTHROPIC_API_KEY"``. If the key is in
        ``KNOWN_KEYS``, its spec provides defaults for env_var/signup_url/hint.
    env_var, signup_url, hint, via_login, storage
        Overrides for ad-hoc specs. Use the pre-baked key when possible.
    chain
        If provided, a ``GATE`` step is appended describing the outcome.
        Values are never recorded — only the key name and resolution path.
    console
        Optional console override (useful for tests).
    prompt_fn, confirm_fn
        Injection points for test harnesses; default to ``typer.prompt`` /
        ``typer.confirm``.

    Returns
    -------
    str
        The credential value. Never empty — if the user aborts, raises
        ``typer.Abort``.
    """
    spec = KNOWN_KEYS.get(key_name)
    resolved_env_var = env_var or (spec.env_var if spec else key_name)
    resolved_signup = signup_url or (spec.signup_url if spec else "")
    resolved_hint = hint or (spec.hint if spec else "")
    resolved_login = via_login if via_login else (spec.via_login if spec else False)
    resolved_storage = storage or (spec.storage if spec else "config")
    c = console or get_console()
    prompt_fn = prompt_fn or _default_prompt
    confirm_fn = confirm_fn or _default_confirm

    started = monotonic()

    # 1. Environment
    if val := os.environ.get(resolved_env_var):
        _record(chain, key_name, resolved="env", success=True, started=started)
        return val

    # 2. Stored config
    if val := _lookup_stored(key_name, resolved_storage):
        _record(chain, key_name, resolved="config", success=True, started=started)
        return val

    # 3. Inline prompt — use the unified error_with_hint formatter so credential
    # prompts look structurally identical to tier/install errors across the CLI.
    c.blank()
    c.error_with_hint(
        f"{key_name} needed for this step.",
        detail=resolved_hint or None,
        hint=None,
        signup_url=resolved_signup or None,
        code=f"require:{key_name}",
    )

    if resolved_login:
        if confirm_fn("Open carl.camp login in browser now?", default=True):
            token = _run_login_and_return_key(key_name)
            if token:
                _record(chain, key_name, resolved="login", success=True, started=started)
                return token

    if resolved_signup and confirm_fn("Open signup page in browser?", default=False):
        try:
            webbrowser.open(resolved_signup)
        except Exception:  # pragma: no cover - best effort
            pass

    try:
        pasted = prompt_fn(
            f"Paste {key_name} (or Enter to abort)",
            hide_input=True,
            default="",
            show_default=False,
        )
    except (typer.Abort, KeyboardInterrupt):
        _record(chain, key_name, resolved="aborted", success=False, started=started)
        raise typer.Abort()

    if not pasted:
        _record(chain, key_name, resolved="aborted", success=False, started=started)
        raise typer.Abort()

    _save(key_name, pasted, resolved_storage)
    _record(chain, key_name, resolved="prompt", success=True, started=started)
    return pasted


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lookup_stored(key_name: str, storage: str) -> str | None:
    """Look up a persisted credential in LocalDB or settings."""
    # Settings first — covers canonical fields like anthropic_api_key etc.
    try:
        from carl_studio.settings import CARLSettings

        settings = CARLSettings()
        attr = key_name.lower()
        val = getattr(settings, attr, None)
        if isinstance(val, str) and val:
            return val
    except Exception:  # pragma: no cover - defensive
        pass

    # LocalDB fallback
    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
        if storage == "auth":
            return db.get_auth(key_name)
        return db.get_config(key_name)
    except Exception:  # pragma: no cover - db may not exist in test env
        return None


def _save(key_name: str, value: str, storage: str) -> None:
    """Persist a freshly entered credential."""
    if storage == "none":
        return
    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
        if storage == "auth":
            db.set_auth(key_name, value)
        else:
            db.set_config(key_name, value)
    except Exception:  # pragma: no cover - best effort
        pass


def _run_login_and_return_key(key_name: str) -> str | None:
    """Kick off the carl.camp browser login and return the resulting JWT."""
    try:
        from carl_studio.cli.platform import login as login_cmd
    except ImportError:  # pragma: no cover - optional CLI surface
        return None
    try:
        login_cmd()
    except typer.Exit:
        pass
    except Exception:  # pragma: no cover - defensive
        return None
    return _lookup_stored(key_name, storage="auth")


def _record(
    chain: InteractionChain | None,
    key_name: str,
    *,
    resolved: str,
    success: bool,
    started: float,
) -> None:
    if chain is None:
        return
    duration_ms = (monotonic() - started) * 1000
    chain.record(
        ActionType.GATE,
        name=f"require:{key_name}",
        input={"resolved_via": resolved},
        output={"success": success},
        success=success,
        duration_ms=duration_ms,
    )
