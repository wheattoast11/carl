"""Tests for the research-cycle CLI verbs: hypothesize, infer --propose/--from-eval, commit.

Mocks CARLAgent to avoid any real Anthropic API calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
import typer
import yaml
from typer.testing import CliRunner


# ---------------------------------------------------------------------------
# Test harness -- build a minimal Typer app registered with just the commands
# we're exercising so we don't pull the full carl CLI wiring.
# ---------------------------------------------------------------------------


def _make_event(kind: str, content: str = "") -> MagicMock:
    ev = MagicMock()
    ev.kind = kind
    ev.content = content
    return ev


class _FakeAgent:
    """Lightweight stand-in for CARLAgent that yields a canned text stream.

    Each call to chat() returns a generator over the stored events.
    """

    def __init__(self, events_factory: Any = None) -> None:
        self.events_factory = events_factory or (lambda _prompt: [])
        self._tools: list[Any] = []

    def chat(self, prompt: str) -> Iterator[Any]:
        for ev in self.events_factory(prompt):
            yield ev


def _agent_factory(events: list[Any]) -> Any:
    """Return a callable that builds a _FakeAgent emitting ``events`` every chat call."""

    def _factory(*_args: Any, **_kwargs: Any) -> _FakeAgent:
        return _FakeAgent(events_factory=lambda _p: list(events))

    return _factory


@pytest.fixture()
def app() -> typer.Typer:
    """Build a Typer app with just the three research-cycle commands."""
    from carl_studio.cli.commit import commit_cmd
    from carl_studio.cli.hypothesize import hypothesize_cmd
    from carl_studio.cli.infer import infer_cmd

    a = typer.Typer()
    a.command("hypothesize")(hypothesize_cmd)
    a.command("infer")(infer_cmd)
    a.command("commit")(commit_cmd)
    return a


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# carl hypothesize
# ---------------------------------------------------------------------------


_VALID_YAML = """```yaml
name: coh-vs-ent-gsm8k
hypothesis: coherence rewards beat entropy on GSM8K
prediction: task_completion_rate > 0.75
model:
  base: Qwen/Qwen2.5-0.5B
  method: grpo
dataset:
  id: openai/gsm8k
  split: train
rewards:
  - name: coherence
    weight: 1.0
eval:
  phase: 2
  threshold: 0.5
```"""


class TestHypothesize:
    def test_writes_carl_yaml_with_required_keys(
        self, app: typer.Typer, runner: CliRunner, tmp_path: Path
    ) -> None:
        events = [_make_event("text", _VALID_YAML)]
        out = tmp_path / "carl.yaml"
        with patch(
            "carl_studio.chat_agent.CARLAgent",
            side_effect=_agent_factory(events),
        ):
            result = runner.invoke(
                app,
                [
                    "hypothesize",
                    "coherence beats entropy on GSM8K",
                    "-o",
                    str(out),
                ],
            )
        assert result.exit_code == 0, result.output
        assert out.exists()
        parsed = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict)
        for key in ("name", "hypothesis", "prediction", "model", "dataset", "rewards", "eval"):
            assert key in parsed, f"missing key {key}"

    def test_dry_run_does_not_write(
        self, app: typer.Typer, runner: CliRunner, tmp_path: Path
    ) -> None:
        events = [_make_event("text", _VALID_YAML)]
        out = tmp_path / "carl.yaml"
        with patch(
            "carl_studio.chat_agent.CARLAgent",
            side_effect=_agent_factory(events),
        ):
            result = runner.invoke(
                app,
                ["hypothesize", "x beats y on z", "-o", str(out), "--dry-run"],
            )
        assert result.exit_code == 0, result.output
        assert not out.exists()

    def test_refuses_to_overwrite_without_force(
        self, app: typer.Typer, runner: CliRunner, tmp_path: Path
    ) -> None:
        out = tmp_path / "carl.yaml"
        out.write_text("pre-existing: true\n", encoding="utf-8")
        result = runner.invoke(
            app, ["hypothesize", "some hypothesis", "-o", str(out)]
        )
        assert result.exit_code == 2, result.output
        assert "already exists" in result.output
        assert "carl.hypothesize.exists" in result.output
        assert out.read_text(encoding="utf-8") == "pre-existing: true\n"

    def test_malformed_yaml_emits_bad_output_code(
        self, app: typer.Typer, runner: CliRunner, tmp_path: Path
    ) -> None:
        # Model emits gibberish that won't parse as a dict
        events = [_make_event("text", "not: valid: yaml: at: all: [")]
        out = tmp_path / "carl.yaml"
        with patch(
            "carl_studio.chat_agent.CARLAgent",
            side_effect=_agent_factory(events),
        ):
            result = runner.invoke(
                app, ["hypothesize", "x beats y", "-o", str(out)]
            )
        assert result.exit_code == 1, result.output
        assert "carl.hypothesize.bad_output" in result.output
        assert not out.exists()

    def test_strips_yaml_fences(
        self, app: typer.Typer, runner: CliRunner, tmp_path: Path
    ) -> None:
        # Use ```yml fence variant to confirm stripping works
        blob = "```yml\nname: test\nhypothesis: H\nmodel:\n  base: a\n  method: sft\n```"
        events = [_make_event("text", blob)]
        out = tmp_path / "carl.yaml"
        with patch(
            "carl_studio.chat_agent.CARLAgent",
            side_effect=_agent_factory(events),
        ):
            result = runner.invoke(
                app, ["hypothesize", "test", "-o", str(out), "--force"]
            )
        assert result.exit_code == 0, result.output
        written = out.read_text(encoding="utf-8")
        assert "```" not in written
        parsed = yaml.safe_load(written)
        assert isinstance(parsed, dict)
        assert parsed["name"] == "test"

    def test_force_overwrites_existing(
        self, app: typer.Typer, runner: CliRunner, tmp_path: Path
    ) -> None:
        out = tmp_path / "carl.yaml"
        out.write_text("old: true\n", encoding="utf-8")
        events = [_make_event("text", _VALID_YAML)]
        with patch(
            "carl_studio.chat_agent.CARLAgent",
            side_effect=_agent_factory(events),
        ):
            result = runner.invoke(
                app,
                [
                    "hypothesize",
                    "coherence beats entropy on GSM8K",
                    "-o",
                    str(out),
                    "--force",
                ],
            )
        assert result.exit_code == 0, result.output
        parsed = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert parsed is not None
        assert "old" not in parsed


# ---------------------------------------------------------------------------
# carl commit
# ---------------------------------------------------------------------------


class TestCommit:
    def test_commits_rule_to_constitution(
        self,
        app: typer.Typer,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        overlay = tmp_path / "constitution.yaml"
        monkeypatch.setattr(
            "carl_studio.constitution.USER_OVERLAY", overlay, raising=False
        )
        result = runner.invoke(
            app,
            [
                "commit",
                "always verify tool input before dispatch",
                "--tags",
                "research,training",
                "--priority",
                "70",
            ],
        )
        assert result.exit_code == 0, result.output
        assert overlay.exists()
        data = yaml.safe_load(overlay.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        rules = data.get("rules") or []
        assert rules, "expected at least one rule written"
        entry = rules[0]
        assert entry["text"] == "always verify tool input before dispatch"
        assert entry["priority"] == 70
        assert set(entry["tags"]) == {"research", "training"}

    def test_dry_run_does_not_write(
        self,
        app: typer.Typer,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        overlay = tmp_path / "constitution.yaml"
        monkeypatch.setattr(
            "carl_studio.constitution.USER_OVERLAY", overlay, raising=False
        )
        result = runner.invoke(
            app, ["commit", "never trust transient state", "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        assert "would commit" in result.output
        assert not overlay.exists()

    def test_empty_commit_exits_with_code(
        self, app: typer.Typer, runner: CliRunner
    ) -> None:
        result = runner.invoke(app, ["commit"])
        assert result.exit_code == 2, result.output
        assert "carl.commit.empty" in result.output

    def test_ambiguous_learning_and_session_exits(
        self, app: typer.Typer, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            app, ["commit", "some learning", "--from-session", "abc"]
        )
        assert result.exit_code == 2, result.output
        assert "carl.commit.ambiguous" in result.output

    def test_from_session_nonexistent_raises(
        self, app: typer.Typer, runner: CliRunner, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        (fake_home / ".carl" / "sessions").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        result = runner.invoke(
            app, ["commit", "--from-session", "does-not-exist"]
        )
        # typer.BadParameter -> non-zero exit
        assert result.exit_code != 0
        assert "Session not found" in result.output or "does-not-exist" in result.output

    def test_from_session_extracts_and_writes_rules(
        self,
        app: typer.Typer,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Set up a fake session file
        fake_home = tmp_path / "home"
        sessions_dir = fake_home / ".carl" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        session_id = "abc123"
        (sessions_dir / f"{session_id}.json").write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "always cache expensive calls"},
                        {"role": "assistant", "content": "agreed"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        overlay = tmp_path / "constitution.yaml"
        monkeypatch.setattr(
            "carl_studio.constitution.USER_OVERLAY", overlay, raising=False
        )

        # Model returns a JSON array the commit code will parse
        agent_json = (
            "Here are the lessons:\n"
            '[{"id": "cache.expensive_calls", "text": "cache expensive calls", '
            '"priority": 60, "tags": ["caching"]}]'
        )
        events = [_make_event("text", agent_json)]
        with patch(
            "carl_studio.chat_agent.CARLAgent",
            side_effect=_agent_factory(events),
        ):
            result = runner.invoke(
                app, ["commit", "--from-session", session_id]
            )
        assert result.exit_code == 0, result.output
        assert overlay.exists()
        data = yaml.safe_load(overlay.read_text(encoding="utf-8"))
        rules = data.get("rules") or []
        assert any(r["id"] == "cache.expensive_calls" for r in rules)


# ---------------------------------------------------------------------------
# carl infer --from-eval and --propose-hypothesis
# ---------------------------------------------------------------------------


def _write_eval_report(path: Path, passed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "primary_metric": "task_completion_rate",
                "primary_value": 0.78,
                "threshold": 0.50,
                "metrics": {
                    "task_completion_rate": 0.78,
                    "failure_rate": 0.22,
                },
                "coherence": {"phi": 0.82, "kappa": 21.3},
            }
        ),
        encoding="utf-8",
    )


class TestInferResearchFlags:
    def test_from_eval_reads_report_without_model(
        self, app: typer.Typer, runner: CliRunner, tmp_path: Path
    ) -> None:
        report_path = tmp_path / "runs" / "demo" / "eval_report.json"
        _write_eval_report(report_path, passed=True)

        # Assert no model load -- patch the loader to raise if called
        def _boom(*a: Any, **kw: Any) -> Any:
            raise AssertionError("model loader should not be called with --from-eval")

        with patch("carl_studio.cli.infer._load_model_for_inference", side_effect=_boom):
            result = runner.invoke(
                app, ["infer", "--from-eval", str(report_path)]
            )
        assert result.exit_code == 0, result.output
        assert "task_completion_rate" in result.output

    def test_from_eval_with_propose_invokes_agent(
        self, app: typer.Typer, runner: CliRunner, tmp_path: Path
    ) -> None:
        report_path = tmp_path / "eval_report.json"
        _write_eval_report(report_path, passed=False)
        events = [_make_event("text", "Try GRPO with coherence weight 0.5 on GSM8K.")]
        with patch(
            "carl_studio.chat_agent.CARLAgent",
            side_effect=_agent_factory(events),
        ):
            result = runner.invoke(
                app,
                [
                    "infer",
                    "--from-eval",
                    str(report_path),
                    "--propose-hypothesis",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Try GRPO with coherence weight" in result.output
        assert "Suggested next hypothesis" in result.output

    def test_from_eval_missing_file_errors(
        self, app: typer.Typer, runner: CliRunner, tmp_path: Path
    ) -> None:
        missing = tmp_path / "nope.json"
        result = runner.invoke(
            app, ["infer", "--from-eval", str(missing)]
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_propose_without_any_report_warns(
        self,
        app: typer.Typer,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

        # Patch loader to a stub so the inference code path doesn't actually load
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with patch(
            "carl_studio.cli.infer._load_model_for_inference",
            return_value=(mock_model, mock_tokenizer),
        ), patch(
            "carl_studio.cli.infer._generate", return_value="hello world"
        ):
            result = runner.invoke(
                app,
                [
                    "infer",
                    "--model",
                    "x/y",
                    "--propose-hypothesis",
                    "hello",
                ],
            )
        assert result.exit_code == 0, result.output
        assert (
            "No eval_report.json" in result.output
            or "no eval_report" in result.output.lower()
        )
