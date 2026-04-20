"""First-run UX polish tests (JRN-004/005/006/008/009/010 + SIMP-003).

Covers:
- ``default_chat_model`` persistence after provider selection.
- sample-project scaffold (create + idempotency).
- celebration panel presence.
- doctor Next-steps gating on marker age.
- session_theme routing from intro move key.
- ``_pump_events`` handles all event kinds (one-shot + streaming flags).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carl_studio.cli import chat as chat_mod
from carl_studio.cli import init as init_mod
from carl_studio.cli import startup as startup_mod


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point first-run marker, global config, and context file at tmp_path."""
    home = tmp_path
    monkeypatch.setattr(init_mod, "CARL_HOME", home)
    monkeypatch.setattr(init_mod, "FIRST_RUN_MARKER", home / ".initialized")
    monkeypatch.setattr(init_mod, "CONTEXT_FILE", home / "context.json")
    # Clear env bleed
    for var in (
        "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
        "CARL_CAMP_JWT", "HF_TOKEN", "CARL_DEFAULT_CHAT_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    return home


# ---------------------------------------------------------------------------
# JRN-004 — default_chat_model persistence
# ---------------------------------------------------------------------------

class TestInitPersistsDefaultChatModel:
    def _fake_chain(self) -> Any:
        from carl_core.interaction import InteractionChain

        return InteractionChain()

    def test_init_persists_default_chat_model(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After provider selection, default_chat_model must land in config.yaml."""
        from carl_studio import settings as settings_mod

        cfg_path = isolated_home / "config.yaml"
        monkeypatch.setattr(settings_mod, "GLOBAL_CONFIG", cfg_path)
        monkeypatch.setattr(settings_mod, "CARL_HOME", isolated_home)
        monkeypatch.setattr(init_mod, "GLOBAL_CONFIG", cfg_path)

        chain = self._fake_chain()
        wrote = init_mod._persist_default_chat_model("Anthropic", chain)
        assert wrote is True
        assert cfg_path.is_file()
        # Reload settings and assert the value stuck.
        loaded = settings_mod.CARLSettings.load()
        assert loaded.default_chat_model == "claude-sonnet-4-6"

    def test_persist_skipped_on_missing_provider(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio import settings as settings_mod

        monkeypatch.setattr(settings_mod, "GLOBAL_CONFIG", isolated_home / "config.yaml")
        assert init_mod._persist_default_chat_model(None, self._fake_chain()) is False

    def test_persist_skipped_when_already_set(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio import settings as settings_mod

        cfg_path = isolated_home / "config.yaml"
        monkeypatch.setattr(settings_mod, "GLOBAL_CONFIG", cfg_path)
        monkeypatch.setattr(settings_mod, "CARL_HOME", isolated_home)
        # init.py imports GLOBAL_CONFIG at top of file; patch there too.
        monkeypatch.setattr(init_mod, "GLOBAL_CONFIG", cfg_path)

        # Pre-seed a config that already has the same value we'd write.
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text("default_chat_model: claude-sonnet-4-6\n")

        assert init_mod._persist_default_chat_model("Anthropic", self._fake_chain()) is False

    def test_persist_maps_openrouter_to_expected_model(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio import settings as settings_mod

        cfg_path = isolated_home / "config.yaml"
        monkeypatch.setattr(settings_mod, "GLOBAL_CONFIG", cfg_path)
        monkeypatch.setattr(settings_mod, "CARL_HOME", isolated_home)
        monkeypatch.setattr(init_mod, "GLOBAL_CONFIG", cfg_path)

        assert init_mod._persist_default_chat_model("OpenRouter", self._fake_chain())
        loaded = settings_mod.CARLSettings.load()
        assert loaded.default_chat_model.startswith("openrouter/")


# ---------------------------------------------------------------------------
# JRN-006 — sample-project scaffold
# ---------------------------------------------------------------------------

class TestSampleProject:
    def test_init_sample_project_scaffold_creates_files(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        workdir = tmp_path / "proj"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setattr("typer.confirm", lambda *_a, **_k: True)

        from carl_core.interaction import InteractionChain

        created = init_mod._offer_sample_project(InteractionChain())
        assert created is True
        assert (workdir / "carl.yaml").is_file()
        assert (workdir / "data").is_dir()
        assert (workdir / "outputs").is_dir()

        import yaml as _yaml

        body = _yaml.safe_load((workdir / "carl.yaml").read_text())
        assert body["base_model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert body["dataset_repo"] == "wikitext"
        assert body["compute_target"] == "local"

    def test_init_sample_project_respects_existing_config(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        workdir = tmp_path / "proj"
        workdir.mkdir()
        (workdir / "carl.yaml").write_text("name: pre-existing\n")
        monkeypatch.chdir(workdir)

        # Track whether the prompt fires — it must NOT with a pre-existing yaml.
        called: dict[str, int] = {"confirm": 0}

        def _confirm(*_a: Any, **_k: Any) -> bool:
            called["confirm"] += 1
            return True

        monkeypatch.setattr("typer.confirm", _confirm)
        from carl_core.interaction import InteractionChain

        result = init_mod._offer_sample_project(InteractionChain())
        assert result is False
        assert called["confirm"] == 0
        # The pre-existing content survives — scaffold does not clobber.
        assert "pre-existing" in (workdir / "carl.yaml").read_text()

    def test_init_sample_project_decline_skips(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        workdir = tmp_path / "decline"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setattr("typer.confirm", lambda *_a, **_k: False)

        from carl_core.interaction import InteractionChain

        assert init_mod._offer_sample_project(InteractionChain()) is False
        assert not (workdir / "carl.yaml").exists()

    # ------------------------------------------------------------------
    # F4 — context-aware scaffold overrides
    # ------------------------------------------------------------------

    def test_sample_project_uses_hf_model_from_context(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When ~/.carl/context.json has hf_model, the scaffold must pin it."""
        init_mod._save_context({"hf_model": "mistralai/Mistral-7B-v0.1"})

        workdir = tmp_path / "ctx-model"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setattr("typer.confirm", lambda *_a, **_k: True)

        from carl_core.interaction import InteractionChain

        assert init_mod._offer_sample_project(InteractionChain()) is True

        import yaml as _yaml

        body = _yaml.safe_load((workdir / "carl.yaml").read_text())
        assert body["base_model"] == "mistralai/Mistral-7B-v0.1"
        # GitHub repo absent -> dataset stays on the baseline.
        assert body["dataset_repo"] == "wikitext"

    def test_sample_project_uses_github_repo_from_context(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        init_mod._save_context({"github_repo": "acme/superproject"})

        workdir = tmp_path / "ctx-repo"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setattr("typer.confirm", lambda *_a, **_k: True)

        from carl_core.interaction import InteractionChain

        assert init_mod._offer_sample_project(InteractionChain()) is True

        import yaml as _yaml

        body = _yaml.safe_load((workdir / "carl.yaml").read_text())
        assert body["dataset_repo"] == "acme/superproject"
        # HF model absent -> base_model falls back to TinyLlama.
        assert body["base_model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_sample_project_without_context_falls_back_to_defaults(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No context.json -> TinyLlama + wikitext baseline."""
        workdir = tmp_path / "no-ctx"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setattr("typer.confirm", lambda *_a, **_k: True)

        from carl_core.interaction import InteractionChain

        assert init_mod._offer_sample_project(InteractionChain()) is True

        import yaml as _yaml

        body = _yaml.safe_load((workdir / "carl.yaml").read_text())
        assert body["base_model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert body["dataset_repo"] == "wikitext"

    def test_render_sample_project_yaml_overrides_roundtrip(
        self, isolated_home: Path
    ) -> None:
        """Unit-level check for the pure renderer: both overrides land."""
        init_mod._save_context({
            "hf_model": "google/gemma-2-9b",
            "github_repo": "tej/superrepo",
        })
        body, overrides = init_mod._render_sample_project_yaml()
        assert overrides["base_model"] == "google/gemma-2-9b"
        assert overrides["dataset_repo"] == "tej/superrepo"
        assert "google/gemma-2-9b" in body
        assert "tej/superrepo" in body


# ---------------------------------------------------------------------------
# JRN-008 — celebration + guidance block
# ---------------------------------------------------------------------------

class _RecordingConsole:
    """Minimal console stub that records calls in order."""

    def __init__(self) -> None:
        self.prints: list[tuple[Any, ...]] = []
        self.info_calls: list[str] = []
        self.kv_calls: list[tuple[str, Any]] = []
        self.ok_calls: list[str] = []
        self.warn_calls: list[str] = []
        self.error_calls: list[str] = []
        self.blank_count = 0

    def print(self, *args: Any, **_kwargs: Any) -> None:
        self.prints.append(args)

    def info(self, message: str) -> None:
        self.info_calls.append(message)

    def ok(self, message: str) -> None:
        self.ok_calls.append(message)

    def warn(self, message: str) -> None:
        self.warn_calls.append(message)

    def error(self, message: str) -> None:
        self.error_calls.append(message)

    def kv(self, key: str, value: Any, key_width: int = 12) -> None:
        self.kv_calls.append((key, value))

    def blank(self) -> None:
        self.blank_count += 1


class TestCelebration:
    def test_init_celebration_prints_next_steps(self, isolated_home: Path) -> None:
        c = _RecordingConsole()
        init_mod._celebrate_and_guide(c)

        # kv block should include all four suggested paths.
        keys = [k for k, _ in c.kv_calls]
        assert any("carl \"explore my repo\"" in k for k in keys)
        assert any("carl train" in k for k in keys)
        assert any("carl doctor" in k for k in keys)
        assert any("carl queue add" in k for k in keys)
        # No blocking prompts (celebration must be informative only).
        assert c.blank_count >= 1


# ---------------------------------------------------------------------------
# JRN-005 — doctor next-steps gating
# ---------------------------------------------------------------------------

class TestDoctorNextStepsGating:
    def test_doctor_next_steps_gated_by_marker_age(
        self, isolated_home: Path
    ) -> None:
        """Fresh marker -> show; stale marker -> hide; verbose overrides stale."""
        marker = isolated_home / ".initialized"

        # No marker: do not show (user hasn't initialized).
        assert (
            startup_mod._next_steps_should_show(verbose=False, marker_path=marker)
            is False
        )

        # Fresh marker: show.
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        assert (
            startup_mod._next_steps_should_show(verbose=False, marker_path=marker)
            is True
        )

        # Stale marker (simulate 8 days old): hide.
        import os

        eight_days_ago = marker.stat().st_mtime - (8 * 24 * 60 * 60)
        os.utime(marker, (eight_days_ago, eight_days_ago))
        assert (
            startup_mod._next_steps_should_show(verbose=False, marker_path=marker)
            is False
        )

        # --verbose always shows, even with a stale marker.
        assert (
            startup_mod._next_steps_should_show(verbose=True, marker_path=marker)
            is True
        )

    def test_doctor_next_steps_clock_skew_handled(
        self, isolated_home: Path
    ) -> None:
        """A marker with a future mtime (clock skew) must not crash the gate."""
        marker = isolated_home / ".initialized"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        # Fake `now` earlier than the mtime — negative age must be rejected.
        age_future = marker.stat().st_mtime + 10_000

        def _fake_now() -> float:
            # Return a time BEFORE the marker's mtime to force age < 0.
            return marker.stat().st_mtime - 10

        assert (
            startup_mod._next_steps_should_show(
                verbose=False, marker_path=marker, now_fn=_fake_now,
            )
            is False
        )
        # sanity: ensure the future value we crafted is indeed in the future
        assert age_future > marker.stat().st_mtime


# ---------------------------------------------------------------------------
# JRN-009 — session theme
# ---------------------------------------------------------------------------

class TestSessionTheme:
    def test_session_theme_set_from_move_key(self) -> None:
        """Intro-move keys map to the canonical ``move:*`` labels; unknown -> free-form."""
        assert chat_mod._session_theme_for_move("e") == "move:explore"
        assert chat_mod._session_theme_for_move("t") == "move:train"
        assert chat_mod._session_theme_for_move("v") == "move:eval"
        assert chat_mod._session_theme_for_move("s") == "move:sticky"
        assert chat_mod._session_theme_for_move(None) == "free-form"
        assert chat_mod._session_theme_for_move("") == "free-form"
        assert chat_mod._session_theme_for_move("z") == "free-form"

    def test_agent_set_session_theme_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Agent stores the theme and round-trips it through save/load."""
        from carl_studio.chat_agent import CARLAgent, SessionStore

        class _FakeClient:
            pass

        agent = CARLAgent(_client=_FakeClient(), model="claude-sonnet-4-6")
        agent.set_session_theme("move:train")
        assert agent.session_theme == "move:train"

        # Stuff one message in so save_session has content to persist.
        agent._messages = [{"role": "user", "content": "hello"}]
        store = SessionStore(sessions_dir=tmp_path)
        sid = agent.save_session(title="test", store=store)

        loaded = CARLAgent(_client=_FakeClient(), model="claude-sonnet-4-6")
        assert loaded.load_session(sid, store=store) is True
        assert loaded.session_theme == "move:train"

    def test_agent_system_prompt_includes_move_theme(self) -> None:
        """A ``move:*`` theme should surface in the system prompt."""
        from carl_studio.chat_agent import CARLAgent

        class _FakeClient:
            pass

        agent = CARLAgent(_client=_FakeClient(), model="claude-sonnet-4-6")
        agent.set_session_theme("move:explore")
        prompt = agent._build_system_prompt()
        assert "SESSION THEME: move:explore" in prompt

    def test_agent_free_form_theme_hidden_from_prompt(self) -> None:
        """``free-form`` and unset themes should not clutter the prompt."""
        from carl_studio.chat_agent import CARLAgent

        class _FakeClient:
            pass

        agent = CARLAgent(_client=_FakeClient(), model="claude-sonnet-4-6")
        agent.set_session_theme("free-form")
        prompt = agent._build_system_prompt()
        assert "SESSION THEME" not in prompt


# ---------------------------------------------------------------------------
# SIMP-003 — shared event-pump helper
# ---------------------------------------------------------------------------

def _event(kind: str, **kwargs: Any) -> Any:
    """Build a tiny duck-typed event matching ``chat_agent.AgentEvent``."""
    base: dict[str, Any] = {
        "kind": kind,
        "content": "",
        "tool_name": "",
        "tool_args": {},
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestPumpEvents:
    def test_event_pump_handles_all_event_kinds(self) -> None:
        """One helper must cover every event kind without raising.

        Exercises both paths by calling the helper with the enum-keyed
        render mode expected by the one-shot path AND the interactive path.
        """
        events = [
            _event("text_delta", content="hello"),
            _event("text_delta", content=" world"),
            _event("tool_call", tool_name="ingest_source", tool_args={"path": "a"}),
            _event("tool_blocked", tool_name="ingest_source", content="denied"),
            _event("tool_result", content="x" * 250),  # triggers truncation
            _event("text", content="done."),
            _event("done"),
            _event("error", content="oops"),
        ]

        # One-shot path: no tool_blocked surfacing, no trailing blanks.
        c_oneshot = _RecordingConsole()
        chat_mod._pump_events(
            iter(events), c_oneshot, mode=chat_mod.RenderMode.ONE_SHOT
        )
        # At least one info call for tool_call + tool_result, no warn for blocked.
        assert any("ingest_source" in m for m in c_oneshot.info_calls)
        # tool_result preview must be truncated with '...'
        assert any(m.endswith("...") for m in c_oneshot.info_calls)
        # One-shot must NOT surface tool_blocked as a warn.
        assert not any("denied" in m for m in c_oneshot.warn_calls)

        # Interactive path: surface_blocked + blank_after_done + carl_prefix_newline.
        c_interactive = _RecordingConsole()
        chat_mod._pump_events(
            iter(events), c_interactive, mode=chat_mod.RenderMode.INTERACTIVE
        )
        # The interactive path emits a warn for tool_blocked
        # (one-shot path does not).
        assert any("denied" in m for m in c_interactive.warn_calls)
        # Minimal: the blank_count on the interactive run must exceed the
        # one-shot run (extra blanks after done / text).
        assert c_interactive.blank_count > c_oneshot.blank_count

    def test_event_pump_defaults_to_one_shot_mode(self) -> None:
        """No mode argument -> the quiet one-shot surface."""
        events = [_event("tool_blocked", tool_name="x", content="denied")]
        c = _RecordingConsole()
        chat_mod._pump_events(iter(events), c)
        # tool_blocked is silenced by default.
        assert c.warn_calls == []

    def test_event_pump_tolerates_exception_mid_stream(self) -> None:
        """A stream that raises mid-iteration must route to ``c.error`` and exit."""

        def _bad_stream() -> Any:
            yield _event("text_delta", content="hi")
            raise RuntimeError("boom")

        c = _RecordingConsole()
        chat_mod._pump_events(_bad_stream(), c)
        assert any("boom" in m for m in c.error_calls)

    def test_render_mode_is_stable_enum(self) -> None:
        """Each surface must bind to a distinct value."""
        assert chat_mod.RenderMode.ONE_SHOT is not chat_mod.RenderMode.INTERACTIVE
        # Stable string value is part of the contract (e.g. tracing/logging).
        assert chat_mod.RenderMode.ONE_SHOT.value == "one_shot"
        assert chat_mod.RenderMode.INTERACTIVE.value == "interactive"


# ---------------------------------------------------------------------------
# JRN-010 — context gathering (additional spot-check)
# ---------------------------------------------------------------------------

class TestContextGathering:
    def test_context_save_and_load_roundtrip(self, isolated_home: Path) -> None:
        path = init_mod._save_context({"github_repo": "acme/x", "hf_model": "acme/y"})
        assert path.is_file()
        loaded = init_mod._load_context()
        assert loaded["github_repo"] == "acme/x"
        assert loaded["hf_model"] == "acme/y"

    def test_context_load_returns_empty_on_bad_json(
        self, isolated_home: Path
    ) -> None:
        init_mod.CONTEXT_FILE.write_text("not valid json")
        assert init_mod._load_context() == {}

    def test_context_gathering_empty_answers_skip_persistence(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both answers blank => no file written."""
        monkeypatch.setattr("typer.confirm", lambda *_a, **_k: True)
        # typer.prompt returns empty strings for both
        monkeypatch.setattr("typer.prompt", lambda *_a, **_k: "")
        from carl_core.interaction import InteractionChain

        ok = init_mod._offer_context_gathering(InteractionChain())
        assert ok is False
        assert not init_mod.CONTEXT_FILE.is_file()

    def test_context_gathering_keep_existing(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Idempotency: existing context triggers 'keep current context?' (default yes)."""
        init_mod._save_context({"github_repo": "acme/x"})
        # First confirm fires for keep-current (default True -> keep).
        calls: list[tuple[Any, ...]] = []

        def _confirm(prompt: str, *a: Any, **kw: Any) -> bool:
            calls.append((prompt, kw))
            # "Keep current context?" -> True
            return True

        monkeypatch.setattr("typer.confirm", _confirm)
        from carl_core.interaction import InteractionChain

        assert init_mod._offer_context_gathering(InteractionChain()) is False
        # File still holds the prior value.
        assert init_mod._load_context()["github_repo"] == "acme/x"
