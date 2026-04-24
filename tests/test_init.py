"""Tests for carl init — the onboarding wizard."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from carl_studio.cli import init as init_mod


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point first-run marker at tmp_path and clear any env bleed."""
    marker = tmp_path / ".initialized"
    monkeypatch.setattr(init_mod, "FIRST_RUN_MARKER", marker)
    monkeypatch.setattr(init_mod, "CARL_HOME", tmp_path)
    # Clear all provider env vars
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "CARL_CAMP_JWT"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


class TestFirstRunMarker:
    def test_absent_initially(self, isolated_home: Path) -> None:
        assert not init_mod._first_run_complete()

    def test_mark_and_check(self, isolated_home: Path) -> None:
        init_mod._mark_first_run_complete()
        assert init_mod._first_run_complete()

    def test_clear(self, isolated_home: Path) -> None:
        init_mod._mark_first_run_complete()
        init_mod._clear_first_run_marker()
        assert not init_mod._first_run_complete()


class TestFullInitFlow:
    def test_completes_end_to_end(self, isolated_home: Path) -> None:
        # Simulate: camp session active, provider key set, no extras, no project, consent set
        with (
            patch.object(init_mod, "_has_camp_session", return_value=True),
            patch.object(init_mod, "_detect_any_provider", return_value="Anthropic"),
            patch.object(init_mod, "_training_extras_installed", return_value=True),
            patch.object(init_mod, "_has_project_config", return_value=True),
            patch.object(init_mod, "_consent_set", return_value=True),
            patch.object(init_mod, "_baseline_freshness", return_value=None),
        ):
            try:
                init_mod.init_cmd(skip_extras=True, skip_project=True, force=False, json_output=False)
            except typer.Exit as exc:
                # Exit 0 is acceptable — means already done or success
                assert exc.exit_code == 0

        assert init_mod._first_run_complete()

    def test_already_initialized_short_circuits(self, isolated_home: Path) -> None:
        with patch.object(init_mod, "_first_run_complete", return_value=True):
            with pytest.raises(typer.Exit) as excinfo:
                init_mod.init_cmd(
                    skip_extras=False, skip_project=False, force=False, json_output=False
                )
        assert excinfo.value.exit_code == 0

    def test_force_re_runs(self, isolated_home: Path) -> None:
        init_mod._mark_first_run_complete()
        with (
            patch.object(init_mod, "_has_camp_session", return_value=True),
            patch.object(init_mod, "_detect_any_provider", return_value="Anthropic"),
            patch.object(init_mod, "_consent_set", return_value=True),
            patch.object(init_mod, "_baseline_freshness", return_value=None),
        ):
            # Should not raise Exit(0) from "already initialized" short-circuit
            init_mod.init_cmd(skip_extras=True, skip_project=True, force=True, json_output=False)


class TestJsonOutput:
    def test_emits_structured_summary(
        self, isolated_home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--json`` is non-interactive: probe state, emit summary, exit 0.

        Per v0.18.1, ``--json`` no longer drives the prompt-based wizard
        (which would hang or abort on piped stdin). Instead it probes
        every onboarding signal and reports them in a structured payload
        so callers can decide whether a fresh ``carl init`` is needed.
        """
        with (
            patch.object(init_mod, "_has_camp_session", return_value=True),
            patch.object(init_mod, "_detect_any_provider", return_value="Anthropic"),
            patch.object(init_mod, "_training_extras_installed", return_value=True),
            patch.object(init_mod, "_has_project_config", return_value=True),
            patch.object(init_mod, "_consent_set", return_value=True),
        ):
            with pytest.raises(typer.Exit) as excinfo:
                init_mod.init_cmd(
                    skip_extras=True, skip_project=True, force=False, json_output=True
                )
            assert excinfo.value.exit_code == 0

        out = capsys.readouterr().out
        assert '"status": "probed"' in out
        assert '"camp_session": true' in out
        assert '"llm_provider_detected": "Anthropic"' in out
        assert '"project_config_present": true' in out

    def test_already_initialized_json(
        self, isolated_home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.object(init_mod, "_first_run_complete", return_value=True):
            with pytest.raises(typer.Exit):
                init_mod.init_cmd(
                    skip_extras=False, skip_project=False, force=False, json_output=True
                )
        out = capsys.readouterr().out
        assert '"status": "already_initialized"' in out


class TestProviderDetection:
    def test_detects_anthropic_env(self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk")
        assert init_mod._detect_any_provider() == "Anthropic"

    def test_detects_openrouter(self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "or")
        assert init_mod._detect_any_provider() == "OpenRouter"

    def test_detects_openai(self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "oai")
        assert init_mod._detect_any_provider() == "OpenAI"

    def test_none_when_missing(self, isolated_home: Path) -> None:
        class FakeSettings:
            anthropic_api_key = None
            openrouter_api_key = None
            openai_api_key = None

        with patch("carl_studio.settings.CARLSettings", return_value=FakeSettings()):
            assert init_mod._detect_any_provider() is None
