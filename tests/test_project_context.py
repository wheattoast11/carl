"""Tests for ``carl_studio.project_context`` (v0.18 Track B).

Covers:

* :func:`current` walk-up discovery (nested CWDs resolve to the
  enclosing project root).
* :func:`current` returning ``None`` when no ``.carl/`` ancestor exists.
* :func:`require` raising ``typer.Exit(2)`` with a clear message.
* :func:`project_color` determinism + format + palette stability.
* :func:`scaffold` idempotency + expected directory layout.
* ``theme.json`` parsing (valid, invalid, unknown, missing).
* ``current.txt`` session-id read (present, empty, missing).
* ``CARL_PROJECT_ROOT`` env-var override.
* Name fallback when ``carl.yaml`` is absent or malformed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from carl_studio import project_context as pc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Return a scaffolded project root under ``tmp_path``."""
    (tmp_path / "carl.yaml").write_text("name: sample-project\n")
    pc.scaffold(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# current() — walk-up discovery
# ---------------------------------------------------------------------------


class TestCurrentWalkUp:
    def test_finds_project_from_root(
        self, project_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``current()`` returns a ProjectContext when CWD == root."""
        monkeypatch.delenv("CARL_PROJECT_ROOT", raising=False)
        monkeypatch.chdir(project_root)
        ctx = pc.current()
        assert ctx is not None
        assert ctx.root == project_root.resolve()
        assert ctx.name == "sample-project"

    def test_finds_project_from_nested_cwd(
        self, project_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Walk-up resolves a deeply nested CWD to the enclosing root."""
        monkeypatch.delenv("CARL_PROJECT_ROOT", raising=False)
        nested = project_root / "a" / "b" / "c"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)
        ctx = pc.current()
        assert ctx is not None
        assert ctx.root == project_root.resolve()

    def test_explicit_cwd_argument_is_respected(
        self, project_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Passing ``cwd=`` overrides ``Path.cwd()``."""
        monkeypatch.delenv("CARL_PROJECT_ROOT", raising=False)
        outside = tmp_path.parent / "outside-home"
        outside.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(outside)
        ctx = pc.current(cwd=project_root)
        assert ctx is not None
        assert ctx.root == project_root.resolve()


class TestCurrentNoProject:
    def test_returns_none_when_no_ancestor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``current()`` returns None for a directory with no .carl/ above."""
        monkeypatch.delenv("CARL_PROJECT_ROOT", raising=False)
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.chdir(empty)
        assert pc.current() is None

    def test_returns_none_when_marker_is_a_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file named ``.carl`` does NOT qualify — must be a directory."""
        monkeypatch.delenv("CARL_PROJECT_ROOT", raising=False)
        fake_marker = tmp_path / ".carl"
        fake_marker.write_text("not a directory")
        monkeypatch.chdir(tmp_path)
        assert pc.current() is None


# ---------------------------------------------------------------------------
# require()
# ---------------------------------------------------------------------------


class TestRequire:
    def test_returns_context_when_in_project(
        self, project_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CARL_PROJECT_ROOT", raising=False)
        monkeypatch.chdir(project_root)
        ctx = pc.require("train")
        assert ctx.name == "sample-project"

    def test_raises_exit_2_when_not_in_project(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.delenv("CARL_PROJECT_ROOT", raising=False)
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.chdir(empty)
        with pytest.raises(typer.Exit) as excinfo:
            pc.require("train")
        assert excinfo.value.exit_code == 2
        captured = capsys.readouterr()
        assert "train" in captured.err
        assert "carl init" in captured.err
        assert "not in a carl project" in captured.err

    def test_message_includes_cmd_name(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.delenv("CARL_PROJECT_ROOT", raising=False)
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.chdir(empty)
        with pytest.raises(typer.Exit):
            pc.require("resonant publish")
        captured = capsys.readouterr()
        assert "resonant publish" in captured.err


# ---------------------------------------------------------------------------
# project_color() — deterministic hash → hex
# ---------------------------------------------------------------------------


class TestProjectColor:
    def test_format_is_hex(self) -> None:
        color = pc.project_color("carl-project")
        assert color.startswith("#")
        assert len(color) == 7  # '#' + 6 hex chars
        int(color[1:], 16)  # would raise if malformed

    def test_is_deterministic(self) -> None:
        """Same name → same color, always."""
        a = pc.project_color("my-project")
        b = pc.project_color("my-project")
        assert a == b

    def test_different_names_different_colors(self) -> None:
        """Different names should (almost always) yield different colors."""
        a = pc.project_color("project-alpha")
        b = pc.project_color("project-beta")
        # SHA-256 collision probability across 2 inputs: effectively 0.
        assert a != b

    def test_empty_name_does_not_raise(self) -> None:
        """Empty names fall back to a default label, still return valid hex."""
        color = pc.project_color("")
        assert color.startswith("#")
        assert len(color) == 7


# ---------------------------------------------------------------------------
# scaffold()
# ---------------------------------------------------------------------------


class TestScaffold:
    def test_creates_expected_layout(self, tmp_path: Path) -> None:
        (tmp_path / "carl.yaml").write_text("name: test\n")
        marker = pc.scaffold(tmp_path)
        assert marker == (tmp_path / ".carl").resolve()
        assert (tmp_path / ".carl").is_dir()
        assert (tmp_path / ".carl" / "sessions").is_dir()
        assert (tmp_path / ".carl" / "theme.json").is_file()

    def test_theme_json_is_valid_default(self, tmp_path: Path) -> None:
        (tmp_path / "carl.yaml").write_text("name: test\n")
        pc.scaffold(tmp_path)
        data = json.loads((tmp_path / ".carl" / "theme.json").read_text())
        assert data == {"theme": "carl"}

    def test_is_idempotent(self, tmp_path: Path) -> None:
        """Re-running scaffold must not clobber an existing theme.json."""
        (tmp_path / "carl.yaml").write_text("name: test\n")
        pc.scaffold(tmp_path)
        custom_theme = {"theme": "carli"}
        (tmp_path / ".carl" / "theme.json").write_text(json.dumps(custom_theme))
        pc.scaffold(tmp_path)
        data = json.loads((tmp_path / ".carl" / "theme.json").read_text())
        assert data == custom_theme

    def test_creates_parents_if_missing(self, tmp_path: Path) -> None:
        """``.carl/sessions/`` emerges even when only ``.carl/`` exists."""
        (tmp_path / "carl.yaml").write_text("name: test\n")
        (tmp_path / ".carl").mkdir()  # only the marker, no sessions yet
        pc.scaffold(tmp_path)
        assert (tmp_path / ".carl" / "sessions").is_dir()


# ---------------------------------------------------------------------------
# Theme parsing
# ---------------------------------------------------------------------------


class TestThemeParsing:
    def test_default_theme_when_file_missing(self, project_root: Path) -> None:
        # scaffold writes theme.json by default; remove to simulate missing.
        theme = project_root / ".carl" / "theme.json"
        theme.unlink()
        ctx = pc.current(cwd=project_root)
        assert ctx is not None
        assert ctx.theme == "carl"

    def test_valid_carli_theme_loaded(self, project_root: Path) -> None:
        theme = project_root / ".carl" / "theme.json"
        theme.write_text(json.dumps({"theme": "carli"}))
        ctx = pc.current(cwd=project_root)
        assert ctx is not None
        assert ctx.theme == "carli"

    def test_unknown_theme_falls_back(self, project_root: Path) -> None:
        theme = project_root / ".carl" / "theme.json"
        theme.write_text(json.dumps({"theme": "disco"}))
        ctx = pc.current(cwd=project_root)
        assert ctx is not None
        assert ctx.theme == "carl"

    def test_malformed_json_falls_back(self, project_root: Path) -> None:
        theme = project_root / ".carl" / "theme.json"
        theme.write_text("{not json")
        ctx = pc.current(cwd=project_root)
        assert ctx is not None
        assert ctx.theme == "carl"


# ---------------------------------------------------------------------------
# Session-id read
# ---------------------------------------------------------------------------


class TestSessionIdRead:
    def test_session_id_none_when_missing(self, project_root: Path) -> None:
        ctx = pc.current(cwd=project_root)
        assert ctx is not None
        assert ctx.session_id is None

    def test_session_id_read_when_present(self, project_root: Path) -> None:
        current_file = project_root / ".carl" / "sessions" / "current.txt"
        current_file.write_text("sess-abc-123\n")
        ctx = pc.current(cwd=project_root)
        assert ctx is not None
        assert ctx.session_id == "sess-abc-123"

    def test_empty_current_file_is_none(self, project_root: Path) -> None:
        current_file = project_root / ".carl" / "sessions" / "current.txt"
        current_file.write_text("")
        ctx = pc.current(cwd=project_root)
        assert ctx is not None
        assert ctx.session_id is None


# ---------------------------------------------------------------------------
# Name fallback
# ---------------------------------------------------------------------------


class TestNameFallback:
    def test_name_from_yaml(self, tmp_path: Path) -> None:
        pinned = tmp_path / "my-app"
        pinned.mkdir()
        (pinned / "carl.yaml").write_text("name: pinned-name\n")
        pc.scaffold(pinned)
        ctx = pc.current(cwd=pinned)
        assert ctx is not None
        assert ctx.name == "pinned-name"

    def test_name_falls_back_to_basename_when_yaml_missing_name_key(
        self, tmp_path: Path
    ) -> None:
        """A carl.yaml without a ``name:`` key falls back to the dir basename."""
        dir_basename = tmp_path / "my-app"
        dir_basename.mkdir()
        # carl.yaml exists but has no name key
        (dir_basename / "carl.yaml").write_text("other_field: value\n")
        pc.scaffold(dir_basename)
        ctx = pc.current(cwd=dir_basename)
        assert ctx is not None
        assert ctx.name == "my-app"

    def test_walk_up_requires_carl_yaml(self, tmp_path: Path) -> None:
        """A ``.carl/`` directory without ``carl.yaml`` is NOT a project.

        This guards against ``~/.carl/`` (the global state dir) being
        treated as a project by the walk-up.
        """
        (tmp_path / ".carl").mkdir()  # marker only
        # no carl.yaml
        ctx = pc.current(cwd=tmp_path)
        assert ctx is None

    def test_name_falls_back_on_malformed_yaml(self, tmp_path: Path) -> None:
        dir_x = tmp_path / "dir-x"
        dir_x.mkdir()
        (dir_x / "carl.yaml").write_text("::: not yaml :::")
        pc.scaffold(dir_x)
        ctx = pc.current(cwd=dir_x)
        assert ctx is not None
        assert ctx.name == "dir-x"


# ---------------------------------------------------------------------------
# CARL_PROJECT_ROOT env override
# ---------------------------------------------------------------------------


class TestEnvOverride:
    def test_env_var_pins_root(
        self,
        project_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``CARL_PROJECT_ROOT`` lets callers skip the walk-up."""
        outside = tmp_path.parent / "outside-env"
        outside.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(outside)
        monkeypatch.setenv("CARL_PROJECT_ROOT", str(project_root))
        ctx = pc.current()
        assert ctx is not None
        assert ctx.root == project_root.resolve()

    def test_env_var_ignored_when_not_a_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bad CARL_PROJECT_ROOT falls through to walk-up (no crash)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(
            "CARL_PROJECT_ROOT", str(tmp_path / "does-not-exist")
        )
        assert pc.current() is None


# ---------------------------------------------------------------------------
# ProjectContext frozen-ness
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_context_is_frozen(self, project_root: Path) -> None:
        ctx = pc.current(cwd=project_root)
        assert ctx is not None
        with pytest.raises(Exception):
            ctx.name = "new-name"  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# Color propagates into context
# ---------------------------------------------------------------------------


class TestColorPropagation:
    def test_color_on_context_matches_function(self, project_root: Path) -> None:
        ctx = pc.current(cwd=project_root)
        assert ctx is not None
        assert ctx.color == pc.project_color(ctx.name)
