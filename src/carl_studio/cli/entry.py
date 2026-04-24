"""Unified entry-point router (v0.18 Track A).

This module owns the single decision point that picks between the three
entry modes of ``carl``:

1. Bare ``carl`` (empty argv) → interactive REPL via :func:`chat_cmd`.
2. ``carl "<prompt>"`` (a single positional that is not a registered
   subcommand) → REPL with the first user message pre-submitted.
3. ``carl -p "<prompt>" / --print "<prompt>"`` → one-shot non-interactive
   reply via :func:`ask_cmd` (the SDK-shaped path).
4. ``carl <verb> ...`` → fall through to Typer's normal subcommand dispatch.

Plus two cross-cutting concerns:

* when ``~/.carl/.initialized`` is missing and stdin is a TTY, the router runs
  ``init_cmd`` first so the user gets a guided setup instead of a bare prompt;
* when entering a detected project via bare interactive routes, a trust
  pre-check runs once per project unless disabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Protocol

import typer


class _ChatHandler(Protocol):
    def __call__(self, *, initial_message: str | None = None) -> None: ...


class _AskHandler(Protocol):
    def __call__(self, prompt: str) -> None: ...


class _InitHandler(Protocol):
    def __call__(self) -> None: ...


class _HelpHandler(Protocol):
    def __call__(self) -> None: ...


class _TrustPrecheckHandler(Protocol):
    def __call__(self) -> bool: ...


REGISTERED_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "admin",
        "agent",
        "align",
        "ask",
        "backends",
        "bench",
        "browse",
        "camp",
        "carlito",
        "chat",
        "checkpoint",
        "commit",
        "compute",
        "config",
        "contract",
        "credits",
        "curriculum",
        "data",
        "db",
        "dev",
        "doctor",
        "env",
        "eval",
        "flow",
        "frame",
        "golf",
        "hypothesize",
        "infer",
        "init",
        "lab",
        "learn",
        "login",
        "logout",
        "marketplace",
        "mcp",
        "metrics",
        "paper",
        "project",
        "publish",
        "push",
        "queue",
        "research",
        "resonant",
        "run",
        "session",
        "setup",
        "skill",
        "start",
        "sync",
        "trust",
        "update",
    }
)


def _default_first_run_marker() -> Path:
    try:
        from carl_studio.cli.init import FIRST_RUN_MARKER

        return Path(FIRST_RUN_MARKER)
    except Exception:
        return Path.home() / ".carl" / ".initialized"


def _default_stdin_is_tty() -> bool:
    stdin = getattr(sys, "stdin", None)
    if stdin is None:
        return False
    isatty = getattr(stdin, "isatty", None)
    if isatty is None:
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def _default_show_help() -> None:
    from carl_studio.cli.apps import app

    ctx = typer.Context(typer.main.get_command(app))
    typer.echo(ctx.get_help())
    typer.echo(
        "\nUse `carl chat` for an interactive session or "
        '`carl ask "<prompt>"` for one-shot.'
    )


def _default_trust_precheck() -> bool:
    from carl_studio.cli import ui
    from carl_studio.project_context import current
    from carl_studio.trust import get_trust_registry

    registry = get_trust_registry()
    if not registry.is_enabled():
        return True

    context = current(Path.cwd())
    if context is None:
        return True
    project_root = context.root
    if registry.is_trusted_root(project_root):
        return True

    choice = ui.select(
        "Trust this project for bare `carl` entry?",
        [
            ui.Choice(
                value="trust_once",
                label="Trust this project",
                hint=str(project_root),
                badge="recommended",
            ),
            ui.Choice(
                value="skip",
                label="Stop here",
                hint="keep routing deterministic until explicitly trusted",
            ),
            ui.Choice(
                value="disable",
                label="Disable trust pre-check globally",
                hint="future bare entry will not prompt",
            ),
        ],
        default=0,
        help="Arrow keys to choose; Enter to continue.",
    )
    if choice == "trust_once":
        registry.trust_root(project_root)
        return True
    if choice == "disable":
        registry.set_enabled(False)
        return True
    return False


def _resolve_default_handlers() -> tuple[
    _ChatHandler, _AskHandler, _InitHandler, _HelpHandler, _TrustPrecheckHandler
]:
    from carl_studio.cli.chat import ask_cmd, chat_cmd
    from carl_studio.cli.init import init_cmd

    def _chat(*, initial_message: str | None = None) -> None:
        chat_cmd(initial_message=initial_message)

    def _ask(prompt: str) -> None:
        ask_cmd(prompt=prompt)

    def _init() -> None:
        init_cmd(
            skip_extras=False,
            skip_project=False,
            force=False,
            json_output=False,
        )

    return _chat, _ask, _init, _default_show_help, _default_trust_precheck


def _looks_like_prompt(token: str) -> bool:
    if not token:
        return False
    if token.startswith("-"):
        return False
    return token[0].isalpha()


def _extract_print_prompt(argv: Iterable[str]) -> str | None:
    items = list(argv)
    for idx, token in enumerate(items):
        if token == "-p" or token == "--print":
            if idx + 1 < len(items):
                return items[idx + 1]
            return ""
        if token.startswith("--print="):
            return token.split("=", 1)[1]
        if token.startswith("-p="):
            return token.split("=", 1)[1]
    return None


def _has_help_flag(argv: Iterable[str]) -> bool:
    return any(tok in ("--help", "-h") for tok in argv)


def _has_version_flag(argv: Iterable[str]) -> bool:
    return any(tok == "--version" for tok in argv)


def route(
    argv: list[str] | None = None,
    *,
    chat: _ChatHandler | None = None,
    ask: _AskHandler | None = None,
    init: _InitHandler | None = None,
    show_help: _HelpHandler | None = None,
    trust_precheck: _TrustPrecheckHandler | None = None,
    first_run_marker: Path | None = None,
    stdin_is_tty: Callable[[], bool] | None = None,
) -> bool:
    if argv is None:
        argv = list(sys.argv[1:])

    default_chat = chat
    default_ask = ask
    default_init = init
    default_help = show_help
    default_trust_precheck = trust_precheck
    if (
        default_chat is None
        or default_ask is None
        or default_init is None
        or default_help is None
        or default_trust_precheck is None
    ):
        _chat, _ask, _init, _help, _trust = _resolve_default_handlers()
        if default_chat is None:
            default_chat = _chat
        if default_ask is None:
            default_ask = _ask
        if default_init is None:
            default_init = _init
        if default_help is None:
            default_help = _help
        if default_trust_precheck is None:
            default_trust_precheck = _trust

    marker_path = first_run_marker if first_run_marker is not None else _default_first_run_marker()
    stdin_tty_fn = stdin_is_tty if stdin_is_tty is not None else _default_stdin_is_tty

    if _has_help_flag(argv) or _has_version_flag(argv):
        return False

    stdin_tty = bool(stdin_tty_fn())

    if not marker_path.is_file() and stdin_tty:
        default_init()
        return True

    bare_prompt_entry = (
        stdin_tty
        and (
            not argv
            or (
                len(argv) == 1
                and argv[0] not in REGISTERED_SUBCOMMANDS
                and _looks_like_prompt(argv[0])
            )
        )
    )
    if bare_prompt_entry:
        if not default_trust_precheck():
            raise typer.Exit(1)

    print_prompt = _extract_print_prompt(argv)
    if print_prompt is not None:
        if not print_prompt.strip():
            return False
        default_ask(print_prompt)
        return True

    if not argv:
        if stdin_tty:
            default_chat(initial_message=None)
        else:
            default_help()
        return True

    if len(argv) == 1:
        token = argv[0]
        if token in REGISTERED_SUBCOMMANDS:
            return False
        if _looks_like_prompt(token):
            if not stdin_tty:
                return False
            default_chat(initial_message=token)
            return True

    return False


def route_from_sys_argv() -> bool:
    return route(list(sys.argv[1:]))


__all__ = ["REGISTERED_SUBCOMMANDS", "route", "route_from_sys_argv"]


if os.environ.get("CARL_ROUTE_DEBUG") == "1":  # pragma: no cover
    try:
        _ = sys.argv
    except Exception as exc:
        raise RuntimeError("sys.argv unavailable for carl router") from exc
