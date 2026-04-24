"""Project context — the ``.carl/`` directory is the world-root.

v0.18 Track B primitive. A *project* is any directory containing a
``.carl/`` subdirectory (alongside its ``carl.yaml``). Commands that mint
artifacts (``train``, ``eval``, ``run``, ``publish``, ``agent publish``,
``resonant publish``) require a project context; commands that are
project-agnostic (``doctor``, ``update``, ``camp login``, ``config``,
``init``, bare REPL) do not.

The module exposes three small operations:

* :func:`current` — walk up from CWD looking for ``.carl/``; return
  ``None`` if none is found.
* :func:`require` — :func:`current` + raise ``typer.Exit(2)`` with a
  clear, actionable message when not in a project.
* :func:`scaffold` — create a fresh ``.carl/`` + ``sessions/`` skeleton
  alongside an existing ``carl.yaml``. Called by ``carl init`` after the
  project yaml has been written.

Design invariants:

* Immutable: :class:`ProjectContext` is a frozen dataclass.
* Deterministic: ``color`` is a hex string derived from
  ``sha256(name)`` — the same project always renders the same hue.
* Cheap: :func:`current` is O(depth) filesystem stats; no network.
* Pure-observe for sessions: this module only READS
  ``.carl/sessions/current.txt`` (Track D owns the write side).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import typer

__all__ = [
    "ProjectContext",
    "current",
    "require",
    "scaffold",
    "project_color",
    "default_theme",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_MARKER = ".carl"
PROJECT_YAML = "carl.yaml"
SESSIONS_DIRNAME = "sessions"
CURRENT_SESSION_FILE = "current.txt"
THEME_FILE = "theme.json"

_VALID_THEMES = frozenset({"carl", "carli"})
_DEFAULT_THEME = "carl"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectContext:
    """A snapshot of the current project's identity.

    Fields:

    * ``root`` — absolute path to the directory holding ``.carl/``.
    * ``name`` — project name (read from ``carl.yaml``; falls back to
      the root directory's basename when yaml is absent/malformed).
    * ``color`` — deterministic hex color (``#RRGGBB``) derived from
      ``sha256(name)``. Used by the REPL prompt, status lines, and any
      kubectl-context-style surface.
    * ``theme`` — ``"carl"`` or ``"carli"`` (persisted in
      ``.carl/theme.json``; defaults to ``"carl"``).
    * ``session_id`` — the most-recently active session id, read from
      ``.carl/sessions/current.txt`` if present. ``None`` otherwise.
    """

    root: Path
    name: str
    color: str
    theme: str
    session_id: str | None


# ---------------------------------------------------------------------------
# Colour derivation
# ---------------------------------------------------------------------------


def project_color(name: str) -> str:
    """Deterministic ``#RRGGBB`` hue from a project name.

    The mapping is stable across Python versions, platforms, and
    terminals — a given name always renders the same colour. We bias
    saturation + value upward so the prompt remains readable on dark
    terminals (Ghostty, iTerm2 dark mode, WezTerm — the 4 terminals
    used in the terminals-tech design-system target).

    Implementation: hash the name with SHA-256, fold the first 4 bytes
    into a hue H in [0, 360), then convert (H, 0.55, 0.85) back to
    sRGB. The constant S/V pair keeps the palette coherent.
    """
    if not name:  # pragma: no cover — defensive
        name = "carl-project"

    digest = hashlib.sha256(name.encode("utf-8")).digest()
    # Use the first 2 bytes as a 0..65535 seed; map to hue.
    hue_seed = (digest[0] << 8) | digest[1]
    hue = (hue_seed / 65535.0) * 360.0

    # Bias the palette for dark terminals. Saturation 0.55, value 0.85.
    r, g, b = _hsv_to_rgb(hue, 0.55, 0.85)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """HSV → RGB, both floats in [0, 1] except h in [0, 360).

    Stdlib's ``colorsys.hsv_to_rgb`` would work equivalently — we inline
    it to keep the project_context module dependency-light (no stdlib
    imports beyond ``hashlib`` + ``os`` + ``pathlib`` + ``dataclasses``
    + ``json``). This keeps import-time cheap for the CLI hot path
    (bare ``carl`` wants zero avoidable work).
    """
    if s <= 0.0:
        return v, v, v
    h = h % 360.0
    sector = h / 60.0
    i = int(sector)
    f = sector - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


# ---------------------------------------------------------------------------
# Theme resolution
# ---------------------------------------------------------------------------


def default_theme() -> str:
    """The fallback theme label (``"carl"``) when ``.carl/theme.json`` is absent."""
    return _DEFAULT_THEME


def _load_theme(project_root: Path) -> str:
    """Read ``.carl/theme.json`` and return the ``theme`` field.

    Malformed / missing files fall back to :func:`default_theme`. Unknown
    theme values are clamped to the default. Never raises.
    """
    theme_file = project_root / PROJECT_MARKER / THEME_FILE
    if not theme_file.is_file():
        return _DEFAULT_THEME
    try:
        raw = theme_file.read_text()
    except OSError:
        return _DEFAULT_THEME
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        return _DEFAULT_THEME
    if not isinstance(parsed, dict):
        return _DEFAULT_THEME
    value = cast(dict[str, Any], parsed).get("theme")
    if isinstance(value, str) and value in _VALID_THEMES:
        return value
    return _DEFAULT_THEME


# ---------------------------------------------------------------------------
# Project name resolution
# ---------------------------------------------------------------------------


def _load_project_name(project_root: Path) -> str:
    """Resolve the project's display name.

    Priority order:
    1. ``carl.yaml::name`` if the file is readable + the key is a non-empty str.
    2. The directory basename (``project_root.name``) as a stable fallback.

    YAML parsing is guarded — import failures (``pyyaml`` optional on
    headless runtimes) and malformed documents both fall through to the
    basename. Never raises.
    """
    yaml_path = project_root / PROJECT_YAML
    if yaml_path.is_file():
        try:
            import yaml as _yaml
        except ImportError:  # pragma: no cover — pyyaml is in the default extras
            return project_root.name or "carl-project"
        try:
            parsed: Any = _yaml.safe_load(yaml_path.read_text())
        except Exception:
            return project_root.name or "carl-project"
        if isinstance(parsed, dict):
            value = cast(dict[str, Any], parsed).get("name")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return project_root.name or "carl-project"


# ---------------------------------------------------------------------------
# Session resolution (read-only; Track D owns the writes)
# ---------------------------------------------------------------------------


def _load_current_session_id(project_root: Path) -> str | None:
    """Read ``.carl/sessions/current.txt`` if present — Track D writes this.

    Returns a trimmed id string or ``None``. Empty files, missing
    files, and OSErrors all map to ``None``. We never interpret the
    file's contents — Track D is the sole authority for session
    lifecycle; this module is a read-only consumer for display purposes.
    """
    target = project_root / PROJECT_MARKER / SESSIONS_DIRNAME / CURRENT_SESSION_FILE
    if not target.is_file():
        return None
    try:
        raw = target.read_text().strip()
    except OSError:
        return None
    return raw or None


# ---------------------------------------------------------------------------
# Walk-up discovery
# ---------------------------------------------------------------------------


def _walk_up_for_project(start: Path) -> Path | None:
    """Walk up from ``start`` looking for a directory containing ``.carl/``.

    Returns the first ancestor (inclusive of ``start``) that has BOTH
    a child ``.carl/`` directory AND a ``carl.yaml`` file — the two
    signals together disambiguate a project root from the user's
    global state directory ``~/.carl/`` (which typically has no
    sibling ``carl.yaml``).

    The user's home directory is explicitly skipped even if both
    markers are present there — ``~/`` is a legitimate project-less
    boundary, and treating it as a project would cause every command
    run outside any subdirectory to claim project-context.

    Returns ``None`` when the walk hits the filesystem root without
    finding a valid project.
    """
    try:
        resolved = start.resolve()
    except OSError:
        return None

    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        home = None

    current_path: Path = resolved
    while True:
        marker = current_path / PROJECT_MARKER
        yaml_file = current_path / PROJECT_YAML
        if marker.is_dir() and yaml_file.is_file():
            # Guard: never claim the user's home directory as a project.
            # ``~/.carl/`` is the global state dir — a stray ``carl.yaml``
            # in ``~/`` does not make it a project root.
            if home is None or current_path != home:
                return current_path
        parent = current_path.parent
        if parent == current_path:
            # Reached filesystem root.
            return None
        current_path = parent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolved_start(cwd: Path | None) -> Path:
    """Honour ``CARL_PROJECT_ROOT`` for explicit override; else return ``cwd``.

    Keeps the override check in one place so tests can monkeypatch the
    env var without worrying about helper-vs-public-API cross-talk.
    """
    env_override = os.environ.get("CARL_PROJECT_ROOT")
    if env_override:
        override_path = Path(env_override).expanduser()
        if override_path.is_dir():
            return override_path
    return cwd if cwd is not None else Path.cwd()


def current(cwd: Path | None = None) -> ProjectContext | None:
    """Return the enclosing :class:`ProjectContext`, or ``None``.

    Walks up from ``cwd`` (default: ``Path.cwd()``) looking for a
    directory containing ``.carl/``. When found, materializes the
    context by reading the project yaml (for ``name``), theme file
    (for ``theme``), and session pointer (for ``session_id``).

    The ``CARL_PROJECT_ROOT`` environment variable overrides the
    walk-up start point — power-users and test harnesses can pin an
    explicit project root without having to ``cd`` into it.

    This function never raises. Malformed metadata falls back to safe
    defaults so the REPL prompt can still render something sensible
    when the user's project has partially-written state (e.g. mid-init).
    """
    start = _resolved_start(cwd)
    try:
        root = _walk_up_for_project(start)
    except Exception:  # pragma: no cover — never raise from the hot path
        return None
    if root is None:
        return None

    name = _load_project_name(root)
    color = project_color(name)
    theme = _load_theme(root)
    session_id = _load_current_session_id(root)
    return ProjectContext(
        root=root,
        name=name,
        color=color,
        theme=theme,
        session_id=session_id,
    )


def require(cmd_name: str) -> ProjectContext:
    """Return the current :class:`ProjectContext` or abort with exit 2.

    Used by commands that mint artifacts (train / eval / run / publish
    / agent publish / resonant publish). The error message is single-
    line, actionable, and routes the user to the two fix paths:
    ``cd`` into an existing project OR bootstrap a new one with
    ``carl init``.

    Raises ``typer.Exit(2)`` — exit code 2 is reserved for "user
    environment is wrong," matching ``--dry-run`` + validation gates
    elsewhere in the CLI.
    """
    ctx = current()
    if ctx is not None:
        return ctx

    # typer.echo with err=True so the message lands on stderr, matching
    # the rest of the CLI's error-path conventions. We do NOT use Rich's
    # get_console() here — this path is called from inside command
    # bodies before the console is guaranteed to be initialized.
    typer.echo(
        f"carl {cmd_name}: not in a carl project. "
        "cd into one or run `carl init` to bootstrap a new project here.",
        err=True,
    )
    raise typer.Exit(2)


def scaffold(cwd: Path) -> Path:
    """Create the ``.carl/`` skeleton alongside an existing ``carl.yaml``.

    Called by ``carl init`` after the project yaml has been written. The
    scaffold layout:

    .. code-block:: text

        <cwd>/
        ├── carl.yaml          (exists; written by init_cmd)
        └── .carl/
            ├── theme.json     (minimal: {"theme": "carl"})
            └── sessions/      (empty dir; Track D populates)

    Returns the absolute path to the ``.carl/`` directory. Idempotent —
    re-running on an already-scaffolded project leaves existing files
    untouched. Only missing files / dirs are created.

    Raises:
        OSError: if the filesystem refuses the directory/file writes.
    """
    root = cwd.resolve()
    marker = root / PROJECT_MARKER
    marker.mkdir(parents=True, exist_ok=True)

    sessions_dir = marker / SESSIONS_DIRNAME
    sessions_dir.mkdir(parents=True, exist_ok=True)

    theme_file = marker / THEME_FILE
    if not theme_file.is_file():
        payload: dict[str, Any] = {"theme": _DEFAULT_THEME}
        theme_file.write_text(json.dumps(payload, indent=2) + "\n")

    return marker
