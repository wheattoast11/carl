"""Tests for carl flow — slash-prefixed op chain execution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from typer.testing import CliRunner

from carl_core.interaction import ActionType, InteractionChain
from carl_studio.cli import flow as flow_mod
from carl_studio.cli import operations as ops_mod
from carl_studio.cli.apps import app


def _ensure_wired() -> None:
    import carl_studio.cli.wiring  # noqa: F401


class TestParseChain:
    def test_single_op_no_args(self) -> None:
        assert flow_mod.parse_chain("/doctor") == [("doctor", [])]

    def test_single_op_with_args(self) -> None:
        assert flow_mod.parse_chain("/echo hello world") == [("echo", ["hello", "world"])]

    def test_multiple_ops(self) -> None:
        result = flow_mod.parse_chain("/echo hi /echo bye")
        assert result == [("echo", ["hi"]), ("echo", ["bye"])]

    def test_quoted_argument(self) -> None:
        result = flow_mod.parse_chain('/ask "train a small model"')
        assert result == [("ask", ["train a small model"])]

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            flow_mod.parse_chain("")

    def test_arg_before_any_op_raises(self) -> None:
        with pytest.raises(ValueError):
            flow_mod.parse_chain("bareword /echo x")

    def test_lone_slash_is_treated_as_arg(self) -> None:
        # A single "/" is not a valid op name (len <= 1)
        with pytest.raises(ValueError):
            flow_mod.parse_chain("/ hello")


class TestOperationsRegistry:
    def test_has_core_ops(self) -> None:
        ops = ops_mod.list_operations()
        assert "doctor" in ops
        assert "echo" in ops
        assert "freshness" in ops

    def test_get_unknown_returns_none(self) -> None:
        assert ops_mod.get_operation("does-not-exist") is None

    def test_echo_records_step(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["echo"](chain, ["hello", "world"])
        assert len(chain) == 1
        step = chain.last()
        assert step is not None
        assert step.action == ActionType.CLI_CMD
        assert step.name == "echo"
        assert step.input == {"args": ["hello", "world"]}
        assert step.output == {"message": "hello world"}


class TestFlowCommand:
    def test_dry_run_renders_plan(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/echo foo /echo bar", "--dry-run"])
        assert result.exit_code == 0
        assert "/echo foo" in result.output
        assert "/echo bar" in result.output
        # No trace file written on dry-run
        assert list(tmp_path.glob("*.jsonl")) == []

    def test_list_ops(self) -> None:
        _ensure_wired()
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "--list"])
        assert result.exit_code == 0
        for name in ops_mod.list_operations():
            assert f"/{name}" in result.output

    def test_flow_list_shows_descriptions(self) -> None:
        """JRN-007: ``carl flow --list`` prints ``/name  — description``.

        The console wraps long lines, so we normalise whitespace before the
        substring check and look for the opening fragment of each description.
        """
        _ensure_wired()
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "--list"])
        assert result.exit_code == 0
        # Collapse wrapped lines so we can assert on descriptions regardless
        # of the rendered terminal width.
        flat = " ".join(result.output.split())
        for name in ops_mod.list_operations():
            desc = ops_mod.get_description(name)
            assert desc, f"op /{name} must have a description"
            opening = " ".join(desc.split())[:40]
            assert opening in flat, (
                f"description for /{name} missing from --list output (opening={opening!r})"
            )
        # At least one op produces a line with the em-dash separator.
        assert "—" in result.output

    def test_missing_chain_errors(self) -> None:
        _ensure_wired()
        runner = CliRunner()
        result = runner.invoke(app, ["flow"])
        assert result.exit_code != 0

    def test_execute_persists_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/echo ping"])
        assert result.exit_code == 0
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text()
        assert "ping" in content
        assert "echo" in content

    def test_no_persist_skips_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/echo x", "--no-persist"])
        assert result.exit_code == 0
        assert list(tmp_path.glob("*.jsonl")) == []

    def test_unknown_op_continues_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown op is a failed step; remaining ops still run; exit code 1."""
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/nope /echo survived"])
        # WS-D2: any failed step => exit code 1.
        assert result.exit_code == 1
        # Trace records the unknown op as failed AND the echo as success.
        files = list(tmp_path.glob("*.jsonl"))
        content = files[0].read_text()
        assert "unknown:/nope" in content
        assert "survived" in content


# ---------------------------------------------------------------------------
# WS-D1 — expanded operation registry
# ---------------------------------------------------------------------------


class TestExpandedOpRegistry:
    """WS-D1: new ops are registered and dispatchable."""

    @pytest.mark.parametrize(
        "op_name",
        [
            "bench",
            "align",
            "learn",
            "eval",
            "infer",
            "publish",
            "push",
            "diagnose",
            "project_status",
            "doctor_freshness",
        ],
    )
    def test_new_op_is_registered(self, op_name: str) -> None:
        assert op_name in ops_mod.list_operations()
        assert ops_mod.get_operation(op_name) is not None

    def test_bench_op_dispatches_underlying_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """bench op wraps carl lab bench with its first arg as model_id."""
        calls: dict[str, Any] = {}

        def fake_bench(
            ctx: Any = None,
            model_id: str = "",
            suite: str = "all",
            compare: str = "",
        ) -> None:
            calls["model_id"] = model_id
            calls["suite"] = suite

        monkeypatch.setattr("carl_studio.cli.lab.bench_cmd", fake_bench)
        chain = InteractionChain()
        ops_mod.OPERATIONS["bench"](chain, ["org/model-a"])
        assert calls["model_id"] == "org/model-a"
        step = chain.last()
        assert step is not None
        assert step.success is True
        assert step.name == "bench"

    def test_bench_op_without_model_id_records_failure(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["bench"](chain, [])
        step = chain.last()
        assert step is not None
        assert step.success is False

    def test_eval_op_dispatches_underlying_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        def fake_eval(**kwargs: Any) -> None:
            calls.update(kwargs)

        monkeypatch.setattr("carl_studio.cli.training.eval_cmd", fake_eval)
        chain = InteractionChain()
        ops_mod.OPERATIONS["eval"](
            chain, ["--adapter", "org/ckpt", "--phase", "2prime"]
        )
        assert calls["adapter"] == "org/ckpt"
        assert calls["phase"] == "2prime"
        step = chain.last()
        assert step is not None
        assert step.success is True

    def test_eval_op_without_adapter_records_failure(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["eval"](chain, [])
        step = chain.last()
        assert step is not None
        assert step.success is False

    def test_infer_op_dispatches_underlying_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        def fake_infer(**kwargs: Any) -> None:
            calls.update(kwargs)

        monkeypatch.setattr("carl_studio.cli.infer.infer_cmd", fake_infer)
        chain = InteractionChain()
        ops_mod.OPERATIONS["infer"](
            chain,
            ["--model", "org/base", "--adapter", "org/lora", "say hi"],
        )
        assert calls["model"] == "org/base"
        assert calls["adapter"] == "org/lora"
        assert calls["prompt"] == "say hi"

    def test_publish_op_dispatches_underlying_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        def fake_publish(**kwargs: Any) -> None:
            calls.update(kwargs)

        monkeypatch.setattr("carl_studio.cli.marketplace.publish_cmd", fake_publish)
        chain = InteractionChain()
        ops_mod.OPERATIONS["publish"](chain, ["org/my-model", "--type", "adapter"])
        assert calls["hub_id"] == "org/my-model"
        assert calls["item_type"] == "adapter"

    def test_push_op_dispatches_sync_push(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        def fake_push(types: str = "runs") -> None:
            calls["types"] = types

        monkeypatch.setattr("carl_studio.cli.platform.sync_push_cmd", fake_push)
        chain = InteractionChain()
        ops_mod.OPERATIONS["push"](chain, ["--types", "runs,metrics"])
        assert calls["types"] == "runs,metrics"

    def test_diagnose_op_dispatches_observe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        def fake_observe(**kwargs: Any) -> None:
            calls.update(kwargs)

        monkeypatch.setattr("carl_studio.cli.observe.observe", fake_observe)
        chain = InteractionChain()
        ops_mod.OPERATIONS["diagnose"](
            chain, ["--file", "train.jsonl", "--run", "demo"]
        )
        assert calls["diagnose"] is True
        assert calls["file"] == "train.jsonl"
        assert calls["run_name"] == "demo"

    def test_align_op_without_mode_records_failure(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["align"](chain, [])
        step = chain.last()
        assert step is not None
        assert step.success is False

    def test_learn_op_without_source_records_failure(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["learn"](chain, [])
        step = chain.last()
        assert step is not None
        assert step.success is False

    def test_project_status_dispatches_project_show(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        def fake_show(config: str = "") -> None:
            calls["config"] = config

        monkeypatch.setattr("carl_studio.cli.project_data.project_show", fake_show)
        chain = InteractionChain()
        ops_mod.OPERATIONS["project_status"](chain, ["--config", "other.yaml"])
        assert calls["config"] == "other.yaml"

    def test_doctor_freshness_calls_doctor_with_check_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        def fake_doctor(check_freshness: bool = False, json_output: bool = False) -> None:
            calls["check_freshness"] = check_freshness
            calls["json_output"] = json_output

        monkeypatch.setattr("carl_studio.cli.startup.doctor", fake_doctor)
        chain = InteractionChain()
        ops_mod.OPERATIONS["doctor_freshness"](chain, [])
        assert calls["check_freshness"] is True
        assert calls["json_output"] is False


# ---------------------------------------------------------------------------
# WS-D2 — flow --json + exit-code recording
# ---------------------------------------------------------------------------


def _register_fake_op(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    handler: Callable[[InteractionChain, list[str]], InteractionChain],
) -> None:
    """Temporarily register a fake op under ``name`` for the duration of a test.

    ``monkeypatch`` handles cleanup: when the test finishes, the original
    mapping is restored (or the entry is deleted if it did not exist).
    """
    monkeypatch.setitem(ops_mod.OPERATIONS, name, handler)


class TestFlowJsonMode:
    def test_json_list_emits_json(self) -> None:
        _ensure_wired()
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "--list", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "ops" in payload
        assert "echo" in payload["ops"]
        # JRN-007: descriptions are part of the JSON payload too.
        assert "descriptions" in payload
        assert payload["descriptions"]["echo"] == ops_mod.get_description("echo")
        for name in payload["ops"]:
            assert payload["descriptions"][name], f"/{name} missing description in JSON"

    def test_json_dry_run_emits_plan(self) -> None:
        _ensure_wired()
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/echo foo /echo bar", "--dry-run", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["plan"] == [
            {"op": "echo", "args": ["foo"]},
            {"op": "echo", "args": ["bar"]},
        ]

    def test_json_execution_produces_valid_chain_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/echo hello", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "steps" in payload
        assert "context" in payload
        assert "summary" in payload
        assert payload["summary"]["total"] == 1
        assert payload["summary"]["failed"] == 0
        assert payload["summary"]["exit_code"] == 0
        assert payload["context"] == {"flow": "/echo hello"}
        assert len(payload["steps"]) == 1

    def test_json_failure_sets_exit_code_and_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)

        def failing_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
            raise SystemExit(1)

        _register_fake_op(monkeypatch, "flow_test_failing", failing_op)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/flow_test_failing", "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["summary"]["failed"] >= 1
        assert payload["summary"]["exit_code"] == 1

    def test_json_missing_chain_emits_error_json(self) -> None:
        _ensure_wired()
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["error"] == "missing chain"


class TestFlowExitCodes:
    def test_failing_op_exits_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single failing op makes the flow command exit 1."""
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)

        def failing_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
            raise SystemExit(1)

        _register_fake_op(monkeypatch, "flow_test_failing1", failing_op)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/flow_test_failing1"])
        assert result.exit_code == 1

    def test_system_exit_nonzero_records_success_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SystemExit(2) from an op records success=False with exit_code=2."""
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)

        def failing_two_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
            # Let _run classify the exit code.
            return ops_mod._run(chain, "failing_two", args, lambda: (_ for _ in ()).throw(SystemExit(2)))

        _register_fake_op(monkeypatch, "flow_test_failing2", failing_two_op)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/flow_test_failing2", "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        step = payload["steps"][0]
        assert step["success"] is False
        assert step["output"]["exit_code"] == 2

    def test_system_exit_zero_records_success_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SystemExit(0) is treated as success — matches clean typer.Exit(0)."""
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)

        def clean_exit_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
            return ops_mod._run(chain, "clean_exit", args, lambda: (_ for _ in ()).throw(SystemExit(0)))

        _register_fake_op(monkeypatch, "flow_test_cleanexit", clean_exit_op)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/flow_test_cleanexit", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        step = payload["steps"][0]
        assert step["success"] is True
        assert step["output"]["exit_code"] == 0

    def test_unhandled_exception_is_captured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An op raising a plain Exception is captured as a failed step."""
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)

        def boom_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
            return ops_mod._run(chain, "boom", args, lambda: (_ for _ in ()).throw(RuntimeError("kaboom")))

        _register_fake_op(monkeypatch, "flow_test_boom", boom_op)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/flow_test_boom /echo after", "--json"])
        # Failed op + successful echo => exit code 1, but the echo still ran.
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert len(payload["steps"]) == 2
        assert payload["steps"][0]["success"] is False
        assert "kaboom" in payload["steps"][0]["output"]["error"]
        assert payload["steps"][1]["success"] is True


class TestFlowNoContinueOnFailure:
    def test_aborts_after_first_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With --no-continue-on-failure, the chain stops after the first failure."""
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["flow", "/nope /echo never-reached", "--no-continue-on-failure", "--json"],
        )
        assert result.exit_code == 1
        payload = json.loads(result.output)
        # Only the unknown op is recorded — echo never ran.
        assert len(payload["steps"]) == 1
        assert payload["steps"][0]["name"] == "unknown:/nope"
        assert payload["summary"]["aborted"] is True

    def test_continues_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without --no-continue-on-failure, subsequent ops still run."""
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/nope /echo ran-anyway", "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert len(payload["steps"]) == 2
        assert payload["summary"]["aborted"] is False
