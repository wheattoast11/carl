"""v0.18 user-journey coverage tests.

This module closes the 4 gaps identified in ``tests/journeys/JOURNEYS.md``:

* **J-C #2** — ``-p`` bypasses trust precheck (documented intent, no direct test).
* **J-F #4** — session ``_resolve_project_root`` walks up through ``project_context``.
* **J-H #2** — ``init --json`` payload exposes all 7 probe keys.
* **J-H #4** — ``init --json`` never prompts on non-TTY stdin.

These are retrofit journey-coverage tests on shipped v0.18.1 behavior, not
TDD-discovery tests. They assert end-to-end transition chains that the
unit-level tests leave implicit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import typer

from carl_studio.cli import entry as entry_mod
from carl_studio.cli import init as init_mod
from carl_studio.cli import session_cmd as session_mod


class _Recorder:
    def __init__(self) -> None:
        self.chat: list[dict[str, Any]] = []
        self.ask: list[str] = []
        self.init_calls = 0
        self.help_calls = 0
        self.trust_calls = 0

    def _chat(self, *, initial_message: str | None = None) -> None:
        self.chat.append({"initial_message": initial_message})

    def _ask(self, prompt: str) -> None:
        self.ask.append(prompt)

    def _init(self) -> None:
        self.init_calls += 1

    def _help(self) -> None:
        self.help_calls += 1

    def _trust(self) -> bool:
        self.trust_calls += 1
        return True

    def kwargs(self) -> dict[str, Any]:
        return {
            "chat": self._chat,
            "ask": self._ask,
            "init": self._init,
            "show_help": self._help,
            "trust_precheck": self._trust,
        }


@pytest.fixture
def rec() -> _Recorder:
    return _Recorder()


@pytest.fixture
def marker(tmp_path: Path) -> Path:
    m = tmp_path / ".initialized"
    m.touch()
    return m


# ---------------------------------------------------------------------------
# J-C #2 — `-p` bypasses trust precheck (even on untrusted TTY entry)
# ---------------------------------------------------------------------------


class TestJournyC_PrintBypassesTrust:
    """`carl -p "q"` is a one-shot path; trust precheck must not fire.

    The docstring in ``cli/entry.py`` states this intent explicitly.
    The test asserts both forms (`-p`, `--print`) and the `=` variant.
    """

    def test_print_short_flag_bypasses_trust(self, rec: _Recorder, marker: Path) -> None:
        handled = entry_mod.route(
            ["-p", "what is 2+2?"],
            **rec.kwargs(),
            first_run_marker=marker,
            stdin_is_tty=lambda: True,
        )
        assert handled is True
        assert rec.trust_calls == 0, "trust precheck must NOT fire on -p path"
        assert rec.ask == ["what is 2+2?"]
        assert rec.chat == []

    def test_print_long_flag_bypasses_trust(self, rec: _Recorder, marker: Path) -> None:
        handled = entry_mod.route(
            ["--print", "hello"],
            **rec.kwargs(),
            first_run_marker=marker,
            stdin_is_tty=lambda: True,
        )
        assert handled is True
        assert rec.trust_calls == 0
        assert rec.ask == ["hello"]

    def test_print_equals_form_bypasses_trust(self, rec: _Recorder, marker: Path) -> None:
        handled = entry_mod.route(
            ["--print=hi"],
            **rec.kwargs(),
            first_run_marker=marker,
            stdin_is_tty=lambda: True,
        )
        assert handled is True
        assert rec.trust_calls == 0
        assert rec.ask == ["hi"]

    def test_print_short_equals_form_bypasses_trust(
        self, rec: _Recorder, marker: Path
    ) -> None:
        handled = entry_mod.route(
            ["-p=hi"],
            **rec.kwargs(),
            first_run_marker=marker,
            stdin_is_tty=lambda: True,
        )
        assert handled is True
        assert rec.trust_calls == 0
        assert rec.ask == ["hi"]


# ---------------------------------------------------------------------------
# J-F #4 — session `_resolve_project_root` walks up via project_context
# ---------------------------------------------------------------------------


class TestJournyF_SessionProjectWalkUp:
    """`carl session list` from a nested subdir finds the project root.

    Pre-fd862b0 the session CLI used `Path.cwd()` directly, which broke
    when invoked from a subdir. The fix routes through
    `project_context.current` which walks up looking for the `.carl/`
    anchor.
    """

    def test_walks_up_from_nested_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Use home-guard-safe layout: HOME=tmp_path, project at tmp_path/proj
        monkeypatch.setenv("HOME", str(tmp_path))
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / ".carl").mkdir()
        (proj / "carl.yaml").write_text("name: test\n")
        nested = proj / "src" / "deep" / "nested"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)

        resolved = session_mod._resolve_project_root()  # pyright: ignore[reportPrivateUsage]
        assert resolved == proj, (
            f"session CLI must walk up to project root; got {resolved} "
            f"instead of {proj}"
        )

    def test_falls_back_to_cwd_when_no_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        not_a_project = tmp_path / "stray"
        not_a_project.mkdir()
        monkeypatch.chdir(not_a_project)

        resolved = session_mod._resolve_project_root()  # pyright: ignore[reportPrivateUsage]
        assert resolved == not_a_project


# ---------------------------------------------------------------------------
# J-H #2 — init --json payload has all 7 probe keys
# ---------------------------------------------------------------------------


class TestJournyH_InitJsonProbeContract:
    """The probe contract must remain stable for ``carl init --json`` consumers.

    If any of the 7 keys is removed or renamed, downstream callers
    (carl.camp dispatch, CI scripts, agent-card bootstrap) break silently.
    This test locks the contract.
    """

    EXPECTED_KEYS: frozenset[str] = frozenset(
        {
            "first_run_complete",
            "camp_session",
            "llm_provider_detected",
            "training_extras_healthy",
            "project_config_present",
            "consent_set",
            "context_present",
        }
    )

    def test_payload_has_all_seven_probe_keys(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        marker = tmp_path / ".initialized"
        monkeypatch.setattr(init_mod, "FIRST_RUN_MARKER", marker)
        monkeypatch.setattr(init_mod, "CARL_HOME", tmp_path)
        for var in (
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "CARL_CAMP_JWT",
        ):
            monkeypatch.delenv(var, raising=False)

        with (
            patch.object(init_mod, "_has_camp_session", return_value=False),
            patch.object(init_mod, "_detect_any_provider", return_value=None),
            patch.object(init_mod, "_training_extras_installed", return_value=False),
            patch.object(init_mod, "_has_project_config", return_value=False),
            patch.object(init_mod, "_consent_set", return_value=False),
        ):
            with pytest.raises(typer.Exit) as excinfo:
                init_mod.init_cmd(
                    skip_extras=True,
                    skip_project=True,
                    force=False,
                    json_output=True,
                )
            assert excinfo.value.exit_code == 0

        payload_text = capsys.readouterr().out
        parsed = json.loads(payload_text)
        assert parsed["status"] == "probed"
        # Probe payload lives inside the recorded chain step's `output`.
        steps = parsed["chain"]["steps"]
        probe_steps = [s for s in steps if s.get("name") == "probe_state"]
        assert probe_steps, "init --json must record a probe_state step"
        probe = probe_steps[0]["output"]
        missing = self.EXPECTED_KEYS - set(probe.keys())
        assert not missing, f"probe payload missing keys: {sorted(missing)}"
        extra = set(probe.keys()) - self.EXPECTED_KEYS
        assert not extra, (
            f"probe payload has unexpected keys (contract drift): "
            f"{sorted(extra)}"
        )


# ---------------------------------------------------------------------------
# J-H #4 — init --json never prompts on non-TTY stdin
# ---------------------------------------------------------------------------


class TestJournyH_InitJsonNoPromptOnNonTTY:
    """`init --json` must complete without interactive prompts.

    v0.18.1 moved `--json` to a probe-only fast-path. A regression where
    `--json` re-entered the prompt wizard would hang CI / piped callers.
    """

    def test_completes_with_closed_stdin(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        marker = tmp_path / ".initialized"
        monkeypatch.setattr(init_mod, "FIRST_RUN_MARKER", marker)
        monkeypatch.setattr(init_mod, "CARL_HOME", tmp_path)

        # Sentinel: if any prompt helper is called, fail loudly.
        def _explode(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError(
                "init --json must not call interactive prompts "
                "(select/confirm/text/path)"
            )

        # Patch every prompt entry point on the UI facade.
        import carl_studio.cli.ui as ui_mod

        for name in ("select", "confirm", "text", "path"):
            if hasattr(ui_mod, name):
                monkeypatch.setattr(ui_mod, name, _explode, raising=False)

        with (
            patch.object(init_mod, "_has_camp_session", return_value=True),
            patch.object(init_mod, "_detect_any_provider", return_value="Anthropic"),
            patch.object(init_mod, "_training_extras_installed", return_value=True),
            patch.object(init_mod, "_has_project_config", return_value=True),
            patch.object(init_mod, "_consent_set", return_value=True),
        ):
            with pytest.raises(typer.Exit) as excinfo:
                init_mod.init_cmd(
                    skip_extras=False,  # <-- deliberately False; probe must
                    skip_project=False,  # <-- still skip prompting
                    force=False,
                    json_output=True,
                )
            assert excinfo.value.exit_code == 0

        out = capsys.readouterr().out
        assert '"status": "probed"' in out
