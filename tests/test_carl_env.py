"""v0.12 — `carl env` wizard tests.

Covers:
- EnvState serialization (JSON round-trip; resume)
- Question selection: next_question picks in order, skips when answered
- Question handlers: choice resolution (index / value / shortcut)
- Rendering: valid YAML structure, TODO markers for missing fields,
  GRPO-specific block appears only for grpo/cascade
- CLI: --auto + --json flag for non-interactive verification
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from carl_studio.env_setup.questions import (
    Choice,
    QUESTION_REGISTRY,
    next_question,
)
from carl_studio.env_setup.render import (
    render_summary_dict,
    render_training_config_yaml,
)
from carl_studio.env_setup.state import EnvState


# ---------------------------------------------------------------------------
# EnvState
# ---------------------------------------------------------------------------


class TestEnvState:
    def test_empty_state_not_complete(self) -> None:
        assert EnvState().is_complete is False

    def test_infer_mode_alone_is_complete(self) -> None:
        s = EnvState(mode="infer")
        assert s.is_complete is True

    def test_train_needs_full_fields(self) -> None:
        s = EnvState(mode="train", method="sft")
        assert s.is_complete is False
        s2 = s.model_copy(update={"dataset": "x", "compute": "local"})
        assert s2.is_complete is True

    def test_json_round_trip(self, tmp_path: Path) -> None:
        s = EnvState(mode="train", method="grpo", dataset="hf/foo", compute="local")
        path = tmp_path / "state.json"
        s.to_json_path(path)
        restored = EnvState.from_json_path(path)
        assert restored.mode == "train"
        assert restored.method == "grpo"
        assert restored.dataset == "hf/foo"

    def test_missing_file_returns_fresh_state(self, tmp_path: Path) -> None:
        s = EnvState.from_json_path(tmp_path / "missing.json")
        # Compare semantically — started_at is a factory default so two
        # fresh states are structurally equivalent but not `==` as pydantic
        # models (different timestamps).
        assert s.mode is None
        assert s.method is None
        assert s.is_complete is False


# ---------------------------------------------------------------------------
# next_question
# ---------------------------------------------------------------------------


class TestNextQuestion:
    def test_first_question_is_mode(self) -> None:
        q = next_question(EnvState())
        assert q is not None
        assert q.id == "mode"

    def test_second_is_method_after_mode(self) -> None:
        q = next_question(EnvState(mode="train"))
        assert q is not None
        assert q.id == "method"

    def test_infer_mode_skips_all(self) -> None:
        q = next_question(EnvState(mode="infer"))
        # Mode set + infer skips method/dataset/compute → all done
        assert q is None

    def test_progresses_through_registry(self) -> None:
        """MVP 4 core questions answered → is_complete + flow continues
        through the v0.14 expanded questions (reward skipped for sft,
        eval_gate still prompts)."""
        s = EnvState(mode="train")
        q = next_question(s)
        assert q is not None
        s = q.handle(s, "sft")
        q2 = next_question(s)
        assert q2 is not None and q2.id == "dataset"
        s = q2.handle(s, "trl-lib/foo")
        q3 = next_question(s)
        assert q3 is not None and q3.id == "compute"
        s = q3.handle(s, "local")
        # Core fields complete
        assert s.is_complete
        # But expanded questions still advance (eval_gate for sft)
        q4 = next_question(s)
        assert q4 is not None and q4.id == "eval_gate"


# ---------------------------------------------------------------------------
# Choice resolution
# ---------------------------------------------------------------------------


class TestChoiceResolution:
    def test_accept_choice_index(self) -> None:
        q = next_question(EnvState())
        assert q is not None
        state = q.handle(EnvState(), "1")
        assert state.mode == "train"

    def test_accept_literal_value(self) -> None:
        q = next_question(EnvState())
        assert q is not None
        state = q.handle(EnvState(), "infer")
        assert state.mode == "infer"

    def test_reject_unrecognized(self) -> None:
        q = next_question(EnvState())
        assert q is not None
        with pytest.raises(ValueError):
            q.handle(EnvState(), "not-a-choice")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_yaml_has_required_keys(self) -> None:
        s = EnvState(
            mode="train", method="sft", dataset="hf/x", compute="local"
        )
        yaml_text = render_training_config_yaml(s)
        assert "mode: train" in yaml_text
        assert "method: sft" in yaml_text
        assert "dataset: hf/x" in yaml_text
        assert "compute: local" in yaml_text

    def test_missing_fields_marked_todo(self) -> None:
        s = EnvState(mode="train")
        yaml_text = render_training_config_yaml(s)
        assert "TODO" in yaml_text  # method / dataset / compute unset

    def test_grpo_block_only_for_grpo_or_cascade(self) -> None:
        sft = EnvState(mode="train", method="sft", dataset="x", compute="local")
        grpo = EnvState(mode="train", method="grpo", dataset="x", compute="local")
        assert "num_generations" not in render_training_config_yaml(sft)
        assert "num_generations" in render_training_config_yaml(grpo)

    def test_summary_dict_round_trip(self) -> None:
        s = EnvState(mode="infer")
        summary = render_summary_dict(s)
        assert summary["mode"] == "infer"
        assert summary["complete"] is True


# ---------------------------------------------------------------------------
# CLI — non-interactive paths only
# ---------------------------------------------------------------------------


class TestEnvCli:
    def test_auto_without_state_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--auto with no resumable state should fail (nothing to write)."""
        from carl_studio.cli.env import env_cmd
        import typer

        # Point the state path at a missing file
        monkeypatch.setattr("carl_studio.cli.env._STATE_PATH", tmp_path / "no.json")
        app = typer.Typer()
        app.command()(env_cmd)
        runner = CliRunner()
        result = runner.invoke(app, ["--auto"])
        assert result.exit_code == 1

    def test_auto_with_complete_state_writes_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio.cli.env import env_cmd
        import typer

        state_path = tmp_path / "state.json"
        output_path = tmp_path / "carl.yaml"
        EnvState(mode="infer").to_json_path(state_path)

        monkeypatch.setattr("carl_studio.cli.env._STATE_PATH", state_path)
        app = typer.Typer()
        app.command()(env_cmd)
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--auto", "--resume", "--output", str(output_path)],
        )
        assert result.exit_code == 0, result.output
        assert output_path.is_file()
        assert "mode: infer" in output_path.read_text()

    def test_json_flag_emits_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio.cli.env import env_cmd
        import typer

        state_path = tmp_path / "state.json"
        EnvState(mode="infer").to_json_path(state_path)
        monkeypatch.setattr("carl_studio.cli.env._STATE_PATH", state_path)

        app = typer.Typer()
        app.command()(env_cmd)
        runner = CliRunner()
        result = runner.invoke(app, ["--auto", "--resume", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["mode"] == "infer"
        assert payload["complete"] is True


# ---------------------------------------------------------------------------
# v0.14 expanded question set — reward / cascade / eval
# ---------------------------------------------------------------------------


class TestExpandedQuestions:
    def test_reward_question_appears_for_grpo(self) -> None:
        s = EnvState(
            mode="train", method="grpo", dataset="hf/x", compute="local"
        )
        # is_complete is True (4-Q MVP fields), but next_question should
        # still surface reward before the flow ends
        q = next_question(s)
        assert q is not None
        assert q.id == "reward"

    def test_reward_question_skipped_for_sft(self) -> None:
        s = EnvState(
            mode="train", method="sft", dataset="hf/x", compute="local"
        )
        # SFT doesn't use reward shaping — should skip directly to eval
        q = next_question(s)
        assert q is not None
        assert q.id != "reward"
        assert q.id == "eval_gate"

    def test_cascade_question_only_for_cascade_method(self) -> None:
        s = EnvState(
            mode="train",
            method="cascade",
            dataset="hf/x",
            compute="local",
            reward="static",
        )
        q = next_question(s)
        assert q is not None
        assert q.id == "cascade_stages"

    def test_cascade_stages_stored_as_int(self) -> None:
        s = EnvState(
            mode="train",
            method="cascade",
            dataset="hf/x",
            compute="local",
            reward="static",
        )
        q = next_question(s)
        assert q is not None
        s = q.handle(s, "2")  # choice index "1" maps to value "2"
        assert s.cascade_stages == 2

    def test_eval_gate_choices(self) -> None:
        s = EnvState(
            mode="train", method="sft", dataset="hf/x", compute="local"
        )
        q = next_question(s)
        assert q is not None and q.id == "eval_gate"
        s = q.handle(s, "crystallization")
        assert s.eval_gate == "crystallization"

    def test_full_train_flow_walks_all_questions(self) -> None:
        """From empty → all 7 questions answered for a cascade run."""
        s = EnvState()
        answers = [
            ("mode", "train"),
            ("method", "cascade"),
            ("dataset", "hf/tool-use"),
            ("compute", "local"),
            ("reward", "phase_adaptive"),
            ("cascade_stages", "3"),  # choice index "2" maps to value "3"
            ("eval_gate", "crystallization"),
        ]
        for expected_id, answer in answers:
            q = next_question(s)
            assert q is not None
            assert q.id == expected_id
            s = q.handle(s, answer)
        # Now the flow is exhausted
        assert next_question(s) is None
        assert s.is_complete
        assert s.reward == "phase_adaptive"
        assert s.cascade_stages == 3
        assert s.eval_gate == "crystallization"


class TestExpandedRender:
    def test_phase_adaptive_reward_renders(self) -> None:
        s = EnvState(
            mode="train",
            method="grpo",
            dataset="hf/x",
            compute="local",
            reward="phase_adaptive",
        )
        yaml_text = render_training_config_yaml(s)
        assert "phase_adaptive" in yaml_text

    def test_cascade_stages_render_in_yaml(self) -> None:
        s = EnvState(
            mode="train",
            method="cascade",
            dataset="hf/x",
            compute="local",
            reward="static",
            cascade_stages=3,
        )
        yaml_text = render_training_config_yaml(s)
        assert "cascade_stages: 3" in yaml_text

    def test_eval_gate_renders_when_not_none(self) -> None:
        s = EnvState(
            mode="train",
            method="sft",
            dataset="hf/x",
            compute="local",
            eval_gate="crystallization",
        )
        yaml_text = render_training_config_yaml(s)
        assert "eval_gate: crystallization" in yaml_text

    def test_eval_gate_omitted_when_none(self) -> None:
        s = EnvState(
            mode="train",
            method="sft",
            dataset="hf/x",
            compute="local",
            eval_gate="none",
        )
        yaml_text = render_training_config_yaml(s)
        assert "eval_gate" not in yaml_text
