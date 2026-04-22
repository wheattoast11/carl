---
last_updated: 2026-04-22
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.17.1-planned
classification: internal — CLI UX doctrine for all future menu/prompt work
---

# CLI UX doctrine (v0.17.1+)

The rules below govern every interactive prompt in carl-studio. They
replace the old pattern of direct ``typer.prompt`` / ``typer.confirm``
sprinkled across individual CLI files with a single ``cli/ui.py``
facade that wraps ``questionary`` (arrow-key UX) and falls back to
``typer`` on non-TTY or when the ``[cli]`` extra is absent.

## TL;DR — follow these five rules

1. **First option is the default.** ``default=0`` in every ``ui.select``.
2. **No auto-advance on numeric keypress.** Typing a digit may jump
   focus (questionary supports it) but must NEVER commit without Enter.
3. **Arrow-key everywhere for ≤8-choice menus.** Never numbered
   ``[1] [2] [3]`` lists in new code.
4. **Non-TTY must not hang.** Every flow falls through to a fallback
   prompt OR exits with an install hint when piped.
5. **No direct ``typer.prompt`` / ``typer.confirm`` in new code.**
   All prompts route through ``cli/ui.py``.

## Why arrow-key over numeric

Numeric selection ("Pick 1-4") has two failure modes:

- **Mistype.** Pressing "2" when "1" was meant. With numeric+Enter the
  user can backspace; with auto-advance they cannot. Arrow-key
  navigation eliminates the class — movement is relative (↑/↓),
  commitment is separate (Enter), and the current selection is always
  visibly highlighted.
- **Cognitive tax.** Numeric requires read-options → pick-number →
  type-number → verify-match → Enter. Arrow-key is just read-options →
  arrow-to-highlight → Enter.

Industry default in 2026 is arrow-key: `gh`, `vite`, `claude-code`,
`openai-codex`, `create-*` scaffolders. Poetry + gcloud still use
numeric; poetry issue #8023 is the canonical user-ask-for-upgrade
case.

## The ``cli/ui.py`` facade

Single public surface:

```python
def select(prompt, choices, *, default=0, help=None) -> str
def confirm(prompt, *, default=True) -> bool
def text(prompt, *, default="", secret=False, validate=None) -> str
def path(prompt, *, must_exist=False, default="") -> str
```

``Choice`` dataclass for rich menus:

```python
@dataclass(frozen=True)
class Choice:
    value: str              # returned to caller
    label: str | None       # displayed text; defaults to value
    hint: str | None        # dim right-side explanation
    badge: str | None       # bracketed tag, e.g. "recommended"
    disabled: bool | str    # True / reason-string disables the item
```

## Conventions by prompt type

| Type | Use | Notes |
|---|---|---|
| 2–8 enumerated choices | ``ui.select`` + ``Choice`` list | First-is-default; badge "recommended" on the preferred option. |
| Yes / No | ``ui.confirm`` | Default True for safe confirmations (proceed?); False for destructive (delete?). |
| Free-text paste (API key, URL) | ``ui.text(..., secret=True)`` when masked | Rejects empty by default; pass ``validate=`` for custom checks. |
| Path / file | ``ui.path(must_exist=True)`` | Expands ``~`` and validates existence when ``must_exist``. |
| > 8 choices | Arrow-key ``ui.select`` with scroll (questionary handles automatically) | Consider splitting the question into sub-menus if >15. |

## Non-TTY behavior contract

Every ``ui.*`` call must behave sensibly when stdin or stdout isn't a
real terminal. The implementation routes to ``typer.prompt`` / typer's
fallback, which:

- ``select``: prints a numbered list + prompts for index. Empty input
  → default. Out-of-range → re-prompt.
- ``confirm``: standard ``[Y/n]`` behavior.
- ``text``: ``typer.prompt`` with ``hide_input=secret``.
- ``path``: ``typer.prompt`` + manual validation loop.

Test coverage for non-TTY paths is non-negotiable — see
``tests/test_cli_ui.py``.

## Cancel semantics

- **Ctrl-C** in the modern path → ``typer.Abort`` (consistent with
  legacy behavior).
- **Empty input** where the prompt has a default → returns the default.
- **Empty input** where no default and ``must_exist`` → re-prompts.

Callers' existing ``except (typer.Abort, EOFError, OSError):`` blocks
keep working without modification.

## gh-style browser auth

For OAuth / managed-tier sign-in (``carl camp login``, ``carl init``
step 1):

1. **Single arrow-key select** with [sign in with browser / create
   account / skip]. Default is "sign in".
2. On sign-in: open browser (``webbrowser.open``) + spin a local
   listener on ``127.0.0.1:<random>`` to receive the OAuth callback.
3. The listener's ``do_GET`` saves the token + renders a themed
   "you can close this tab" HTML response.
4. Main process polls / blocks until token received or timeout.
5. Persist token to LocalDB; confirm to user.

This is already implemented in ``cli/platform.py::login_cmd``. New
flows should invoke it rather than rolling their own.

## What NOT to do

- **No dedicated tab-form UI** in v0.17.x. Questionary doesn't do
  multi-field forms; that's Textual territory. If a flow has > 5
  fields with branching, it should stay sequential with arrow-key
  prompts. Review-and-edit screens are v0.18+.
- **No auto-advance on numeric keys.** Even when the underlying
  library supports it, we disable. Users must press Enter to commit.
- **No re-invention of confirmation patterns.** ``ui.confirm`` is the
  single yes/no primitive; don't roll a custom Y/N loop.
- **No emoji in prompts** unless the user requests it globally.

## When to deviate

Three situations justify skipping ``cli/ui.py``:

1. **REPL loops** (``carl chat``, ``carl lab repl``) that need raw
   ``input()`` for streaming text. The REPL is not a menu; use
   ``input()`` directly with a Rich-styled prompt prefix.
2. **Machine-consumed output modes** (``--json``, ``--auto``). These
   must not prompt at all — they require their inputs via flags or
   resumable state files.
3. **Third-party-owned flows** (``gh auth login`` shell-out, HF CLI
   shell-out). Don't wrap — invoke directly.

## Migration policy

When touching a file with existing ``typer.prompt`` / ``typer.confirm``
calls:

- **Replace them** with the ``ui.*`` equivalents in the same PR.
- **Keep** any ``prompt_fn`` / ``confirm_fn`` injection seams used by
  tests — defaults route through ``ui.*`` via adapter functions
  (see ``cli/prompt.py::_default_prompt``).
- **Don't** rewrite adjacent code that wasn't in scope. The migration
  is opportunistic per file — each touch site is its own commit.

## References

- Plan doc: ``docs/v17_cli_ux_and_dep_probe_plan.md``
- Implementation: ``src/carl_studio/cli/ui.py``
- Tests: ``tests/test_cli_ui.py``
- Companion doc: ``docs/v17_dep_probe_doctrine.md``
