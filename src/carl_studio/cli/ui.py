"""Single prompt surface for all CARL CLI flows (v0.17.1).

Wraps ``questionary`` when available (arrow-key single-select, masked
password, validated path) and falls back to ``typer.prompt`` /
``typer.confirm`` when ``questionary`` is missing OR stdin/stdout is not
a TTY. Every menu in carl-studio should call through here — there should
be no direct ``typer.prompt`` / ``typer.confirm`` calls in new code.

Design decisions pinned in docs/v17_cli_ux_and_dep_probe_plan.md:

- **First-is-default.** ``default=0`` is the convention; the first
  option is also the recommended one.
- **No auto-advance on numeric keypress.** Arrow keys navigate; Enter
  commits. Typing a digit in questionary jumps focus but does not
  commit, which eliminates the "pressed 1 meant 2" mistype class.
- **Non-TTY fallback.** ``echo "" | carl init`` must not hang. The
  fallback path accepts default-on-empty + exits cleanly on EOF.
- **Cancel = typer.Abort.** Pressing Ctrl-C raises ``typer.Abort``
  so callers' existing ``except (typer.Abort, EOFError, OSError)``
  handlers keep working.

``questionary`` pulls ``prompt_toolkit`` — that's why it's behind the
``[cli]`` extra rather than a hard dep.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer


@dataclass(frozen=True)
class Choice:
    """A single option in a select menu.

    Attributes:
        value: What's returned to the caller. Always a string.
        label: Displayed text. Defaults to ``value`` when empty.
        hint: Right-side dim explanation (e.g. "fastest" or file path).
        badge: Bracketed tag shown after the label (e.g. "recommended").
        disabled: ``True``/reason-string disables the item.
    """

    value: str
    label: str | None = None
    hint: str | None = None
    badge: str | None = None
    disabled: bool | str = False


# ---------------------------------------------------------------------------
# Lazy questionary discovery
# ---------------------------------------------------------------------------


def _questionary() -> Any | None:
    """Lazy-import questionary; return None on ImportError."""
    try:
        import questionary

        return questionary
    except ImportError:
        return None


def _isatty() -> bool:
    """True when both stdin and stdout are real terminals."""
    try:
        return bool(sys.stdin.isatty()) and bool(sys.stdout.isatty())
    except (AttributeError, OSError):
        return False


def _use_modern_ui() -> bool:
    """Whether to use the arrow-key UI this call.

    Gates: TTY + questionary available. Exposed as a helper so tests
    can patch a single decision point.
    """
    return _isatty() and _questionary() is not None


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def select(
    prompt: str,
    choices: list[Choice | str],
    *,
    default: int = 0,
    help: str | None = None,
) -> str:
    """Single-select from a list.

    Arrow keys navigate, Enter commits. Numeric keys jump focus (in
    questionary mode). Returns the selected ``Choice.value`` — or the
    string itself when the caller passed plain strings.

    Raises ``typer.Abort`` on Ctrl-C.
    """
    normalized: list[Choice] = [
        c if isinstance(c, Choice) else Choice(value=c) for c in choices
    ]
    if not normalized:
        return ""
    default_idx = max(0, min(default, len(normalized) - 1))

    if _use_modern_ui():
        return _select_questionary(prompt, normalized, default_idx, help)
    return _select_fallback(prompt, normalized, default_idx)


def _select_questionary(
    prompt: str,
    choices: list[Choice],
    default_idx: int,
    help_text: str | None,
) -> str:
    q = _questionary()
    assert q is not None  # _use_modern_ui guarded this

    items: list[Any] = []
    for c in choices:
        label = c.label or c.value
        display = label
        if c.badge:
            display = f"{display}  [{c.badge}]"
        if c.hint:
            display = f"{display}  — {c.hint}"
        disabled_value: bool | str | None
        if c.disabled is False:
            disabled_value = None
        elif c.disabled is True:
            disabled_value = "disabled"
        else:
            disabled_value = c.disabled
        items.append(q.Choice(title=display, value=c.value, disabled=disabled_value))

    kwargs: dict[str, Any] = {"default": items[default_idx]}
    if help_text:
        kwargs["instruction"] = help_text

    try:
        answer = q.select(prompt, choices=items, **kwargs).ask()
    except KeyboardInterrupt as exc:
        raise typer.Abort() from exc
    if answer is None:
        raise typer.Abort()
    return str(answer)


def _select_fallback(
    prompt: str,
    choices: list[Choice],
    default_idx: int,
) -> str:
    typer.echo("")
    typer.secho(prompt, fg="cyan", bold=True)
    for i, c in enumerate(choices, start=1):
        marker = " *" if (i - 1) == default_idx else "  "
        label = c.label or c.value
        suffix = ""
        if c.badge:
            suffix += f"  [{c.badge}]"
        if c.hint:
            suffix += f"  — {c.hint}"
        typer.echo(f"  {marker} [{i}] {label}{suffix}")

    while True:
        try:
            raw = typer.prompt("  →", default=str(default_idx + 1))
        except (typer.Abort, EOFError):
            raise typer.Abort() from None
        try:
            pick = int(raw.strip())
        except (ValueError, TypeError):
            typer.secho("  please enter a number", fg="yellow")
            continue
        if 1 <= pick <= len(choices):
            return choices[pick - 1].value
        typer.secho(f"  out of range (1-{len(choices)})", fg="yellow")


# ---------------------------------------------------------------------------
# confirm
# ---------------------------------------------------------------------------


def confirm(prompt: str, *, default: bool = True) -> bool:
    """Yes/no prompt. Enter accepts the default."""
    if _use_modern_ui():
        q = _questionary()
        assert q is not None
        try:
            answer = q.confirm(prompt, default=default).ask()
        except KeyboardInterrupt as exc:
            raise typer.Abort() from exc
        if answer is None:
            raise typer.Abort()
        return bool(answer)
    return typer.confirm(prompt, default=default)


# ---------------------------------------------------------------------------
# text / secret
# ---------------------------------------------------------------------------


def text(
    prompt: str,
    *,
    default: str = "",
    secret: bool = False,
    validate: Callable[[str], bool | str] | None = None,
) -> str:
    """Free-form text input. Masked when ``secret=True``.

    ``validate`` returns ``True`` to accept, or a string message to re-prompt.
    """
    if _use_modern_ui():
        q = _questionary()
        assert q is not None
        try:
            if secret:
                answer = q.password(prompt, validate=validate).ask()
            else:
                kwargs: dict[str, Any] = {"default": default}
                if validate is not None:
                    kwargs["validate"] = validate
                answer = q.text(prompt, **kwargs).ask()
        except KeyboardInterrupt as exc:
            raise typer.Abort() from exc
        if answer is None:
            raise typer.Abort()
        return str(answer)

    # Fallback with manual validation loop
    while True:
        try:
            value = typer.prompt(
                prompt,
                default=default,
                hide_input=secret,
                show_default=not secret,
            )
        except (typer.Abort, EOFError):
            raise typer.Abort() from None
        if validate is None:
            return str(value)
        v = validate(str(value))
        if v is True:
            return str(value)
        typer.secho(f"  {v}", fg="yellow")


# ---------------------------------------------------------------------------
# path
# ---------------------------------------------------------------------------


def path(prompt: str, *, must_exist: bool = False, default: str = "") -> str:
    """Path input; validates existence when ``must_exist=True``."""

    def _validate(p: str) -> bool | str:
        if not p.strip():
            return True if not must_exist else "path required"
        if must_exist and not Path(p).expanduser().exists():
            return f"path does not exist: {p}"
        return True

    if _use_modern_ui():
        q = _questionary()
        assert q is not None
        try:
            answer = q.path(prompt, default=default, validate=_validate).ask()
        except KeyboardInterrupt as exc:
            raise typer.Abort() from exc
        if answer is None:
            raise typer.Abort()
        return str(answer)

    while True:
        try:
            value = typer.prompt(prompt, default=default, show_default=False).strip()
        except (typer.Abort, EOFError):
            raise typer.Abort() from None
        v = _validate(value)
        if v is True:
            return value
        typer.secho(f"  {v}", fg="yellow")


__all__ = ["Choice", "select", "confirm", "text", "path"]
