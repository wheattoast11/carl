"""UAT (User Acceptance Testing) for carl-studio.

Validates full user journeys end-to-end:
  1. Observe: mock Trackio data -> FileSource -> ObserveFrame
  2. Config:  settings create -> preset apply -> YAML persistence
  3. Tier:    FREE gates -> PRO auto-elevation -> ENTERPRISE gating
  4. Eval:    EvalConfig -> EvalGate -> EvalReport -> pass/fail
  5. Environment: CodingSandboxEnv -> tool calls -> reward
  6. Naming:  naming_prefix validation, settings field types

All tests are self-contained: no GPU, no network, no API keys.
"""

from __future__ import annotations

import json

import pytest


# =========================================================================
# 1. Observe Journey
# =========================================================================


class TestObserveJourney:
    """User creates training data, observes it through FileSource, gets ObserveFrames."""

    def test_file_source_reads_jsonl(self, tmp_path):
        """Write JSONL metrics -> FileSource.poll() -> ObserveFrame with correct fields."""
        from carl_studio.observe.data_source import FileSource, ObserveFrame

        log_file = tmp_path / "training_log.jsonl"
        frames_data = [
            {"step": 1, "phi": 0.1, "loss": 2.5, "reward_mean": 0.0},
            {"step": 2, "phi": 0.3, "loss": 2.1, "reward_mean": 0.1},
            {"step": 3, "phi": 0.6, "loss": 1.8, "reward_mean": 0.3, "phase_transition": True},
        ]
        log_file.write_text("\n".join(json.dumps(f) for f in frames_data) + "\n")

        source = FileSource(str(log_file))
        frames = source.poll()

        assert len(frames) == 3
        assert all(isinstance(f, ObserveFrame) for f in frames)
        assert frames[0].step == 1
        assert frames[0].phi == 0.1
        assert frames[0].loss == 2.5
        assert frames[2].phase_transition is True

    def test_file_source_incremental_poll(self, tmp_path):
        """Second poll only returns new lines (tailing behavior)."""
        from carl_studio.observe.data_source import FileSource

        log_file = tmp_path / "train.jsonl"
        log_file.write_text(json.dumps({"step": 1, "phi": 0.1}) + "\n")

        source = FileSource(str(log_file))
        first = source.poll()
        assert len(first) == 1

        # Append more data
        with open(log_file, "a") as f:
            f.write(json.dumps({"step": 2, "phi": 0.2}) + "\n")
            f.write(json.dumps({"step": 3, "phi": 0.3}) + "\n")

        second = source.poll()
        assert len(second) == 2
        assert second[0].step == 2

    def test_file_source_nonexistent_returns_empty(self, tmp_path):
        """Polling a nonexistent file returns empty list (no crash)."""
        from carl_studio.observe.data_source import FileSource

        source = FileSource(str(tmp_path / "does_not_exist.jsonl"))
        assert source.poll() == []

    def test_file_source_handles_malformed_lines(self, tmp_path):
        """Malformed JSON lines are skipped without error."""
        from carl_studio.observe.data_source import FileSource

        log_file = tmp_path / "train.jsonl"
        log_file.write_text(
            json.dumps({"step": 1, "phi": 0.1})
            + "\n"
            + "NOT VALID JSON\n"
            + json.dumps({"step": 2, "phi": 0.2})
            + "\n"
        )

        source = FileSource(str(log_file))
        frames = source.poll()
        assert len(frames) == 2

    def test_observe_frame_defaults(self):
        """ObserveFrame has sensible defaults for all fields."""
        from carl_studio.observe.data_source import ObserveFrame

        frame = ObserveFrame()
        assert frame.step == 0
        assert frame.phi == 0.0
        assert frame.loss == 0.0
        assert frame.reward_mean == 0.0
        assert frame.rewards == {}
        assert frame.completion_sample == ""
        assert frame.phase_transition is False

    def test_observe_frame_reward_extraction(self, tmp_path):
        """Reward sub-fields (reward_*) are extracted into the rewards dict."""
        from carl_studio.observe.data_source import FileSource

        log_file = tmp_path / "train.jsonl"
        log_file.write_text(
            json.dumps(
                {
                    "step": 5,
                    "reward_task": 0.8,
                    "reward_format": 1.0,
                    "reward_carl": 0.6,
                    "not_a_reward": 42,
                }
            )
            + "\n"
        )

        source = FileSource(str(log_file))
        frames = source.poll()
        assert len(frames) == 1
        assert frames[0].rewards == {
            "reward_task": 0.8,
            "reward_format": 1.0,
            "reward_carl": 0.6,
        }


# =========================================================================
# 2. Config Journey
# =========================================================================


class TestConfigJourney:
    """User creates settings, applies preset, persists to YAML, loads back."""

    def test_default_settings(self):
        """Fresh CARLSettings has correct defaults."""
        from carl_studio.settings import CARLSettings
        from carl_studio.tier import Tier
        from carl_studio.types.config import ComputeTarget

        s = CARLSettings()
        assert s.tier == Tier.FREE
        assert s.default_model == "Tesslate/OmniCoder-9B"
        assert s.default_compute == ComputeTarget.L40SX1
        assert s.log_level == "info"

    def test_preset_applies_defaults(self):
        """Applying RESEARCH preset changes log_level and compute."""
        from carl_studio.settings import CARLSettings, Preset
        from carl_studio.types.config import ComputeTarget

        s = CARLSettings(preset=Preset.RESEARCH)
        assert s.log_level == "debug"
        assert s.default_compute == ComputeTarget.A100_LARGE

    def test_explicit_overrides_preset(self):
        """Explicit values override preset defaults."""
        from carl_studio.settings import CARLSettings, Preset

        s = CARLSettings(preset=Preset.RESEARCH, log_level="error")
        assert s.log_level == "error"

    def test_yaml_roundtrip(self, tmp_path):
        """Settings survive save -> load YAML cycle."""
        from carl_studio.settings import CARLSettings
        from carl_studio.tier import Tier

        config_path = tmp_path / "config.yaml"
        original = CARLSettings(
            tier=Tier.PRO,
            default_model="test/model-9B",
            log_level="debug",
            naming_prefix="test-prefix",
        )
        original.save(config_path)

        loaded = CARLSettings.from_yaml(config_path)
        assert loaded.tier == Tier.PRO
        assert loaded.default_model == "test/model-9B"
        assert loaded.log_level == "debug"

    def test_secrets_excluded_from_yaml(self, tmp_path):
        """hf_token and anthropic_api_key are never persisted to disk."""
        import yaml
        from carl_studio.settings import CARLSettings

        config_path = tmp_path / "config.yaml"
        s = CARLSettings(hf_token="hf_secret_token", anthropic_api_key="sk-secret")
        s.save(config_path)

        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert "hf_token" not in data
        assert "anthropic_api_key" not in data

    def test_set_field_nested(self):
        """set_field handles dotted keys for nested ObserveDefaults."""
        from carl_studio.settings import CARLSettings, set_field

        s = CARLSettings()
        s = set_field(s, "observe_defaults.show_entropy", "false")
        assert s.observe_defaults.show_entropy is False

    def test_set_field_invalid_key_raises(self):
        """set_field raises ValueError for unknown keys."""
        from carl_studio.settings import CARLSettings, set_field

        s = CARLSettings()
        with pytest.raises(ValueError, match="Unknown setting"):
            set_field(s, "nonexistent_key", "value")

    def test_display_dict_masks_secrets(self):
        """display_dict hides token values by default."""
        from carl_studio.settings import CARLSettings

        s = CARLSettings(hf_token="hf_test_1234567890", anthropic_api_key=None)
        d = s.display_dict(mask_secrets=True)
        assert "****" in d["hf_token"]
        assert d["anthropic_api_key"] == "(not set)"


# =========================================================================
# 3. Tier Journey
# =========================================================================


class TestTierJourney:
    """Verify FREE/PAID gating for the full feature matrix."""

    def test_free_allows_core_features(self, monkeypatch):
        """FREE tier allows observe, train, eval -- the core CARL loop."""
        import carl_studio.tier as tier_mod
        from carl_studio.tier import Tier, tier_allows

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: None)

        for feature in ("observe", "train", "eval", "config", "bench"):
            assert tier_allows(Tier.FREE, feature), f"FREE should allow {feature}"

    def test_pro_gates_diagnose(self, monkeypatch):
        """observe.diagnose is available on FREE and PAID tiers."""
        from carl_studio.tier import Tier, tier_allows

        assert tier_allows(Tier.FREE, "observe.diagnose")
        assert tier_allows(Tier.PAID, "observe.diagnose")

    def test_pro_gates_send_it(self, monkeypatch):
        """train.send_it requires PRO tier."""
        from carl_studio.tier import Tier, tier_allows

        assert not tier_allows(Tier.FREE, "train.send_it")
        assert tier_allows(Tier.PRO, "train.send_it")

    def test_anthropic_key_elevates_to_pro(self, monkeypatch):
        """Having ANTHROPIC_API_KEY auto-elevates FREE to PRO."""
        import carl_studio.tier as tier_mod
        from carl_studio.tier import Tier, detect_effective_tier

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: None)

        assert detect_effective_tier(Tier.FREE) == Tier.PRO

    def test_hf_token_elevates_to_pro(self, monkeypatch):
        """Having HF_TOKEN auto-elevates FREE to PRO."""
        import carl_studio.tier as tier_mod
        from carl_studio.tier import Tier, detect_effective_tier

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: "hf_test_token")

        assert detect_effective_tier(Tier.FREE) == Tier.PRO

    def test_enterprise_needs_both_keys_and_flag(self, monkeypatch):
        """Credential-based elevation resolves to PAID in the FREE/PAID model."""
        import carl_studio.tier as tier_mod
        from carl_studio.tier import Tier, detect_effective_tier

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("CARL_ENTERPRISE", "1")
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: "hf_test")

        assert detect_effective_tier(Tier.FREE) == Tier.PAID

    def test_tier_gate_decorator_blocks(self, monkeypatch):
        """@tier_gate(Tier.PRO) blocks a FREE user."""
        import carl_studio.settings as settings_mod
        import carl_studio.tier as tier_mod
        from carl_studio.tier import Tier, TierGateError, tier_gate

        fake_settings = type(
            "FakeSettings",
            (),
            {
                "load": staticmethod(lambda: type("S", (), {"tier": Tier.FREE})()),
            },
        )
        monkeypatch.setattr(settings_mod, "CARLSettings", fake_settings)
        monkeypatch.setattr(tier_mod, "detect_effective_tier", lambda t: t)

        @tier_gate(Tier.PRO, feature="train.send_it")
        def send_it():
            return "launched"

        with pytest.raises(TierGateError):
            send_it()

    def test_tier_gate_decorator_allows(self, monkeypatch):
        """@tier_gate(Tier.PRO) allows a PRO user."""
        import carl_studio.settings as settings_mod
        import carl_studio.tier as tier_mod
        from carl_studio.tier import Tier, tier_gate

        fake_settings = type(
            "FakeSettings",
            (),
            {
                "load": staticmethod(lambda: type("S", (), {"tier": Tier.PRO})()),
            },
        )
        monkeypatch.setattr(settings_mod, "CARLSettings", fake_settings)
        monkeypatch.setattr(tier_mod, "detect_effective_tier", lambda t: t)

        @tier_gate(Tier.PRO, feature="train.send_it")
        def send_it():
            return "launched"

        assert send_it() == "launched"

    def test_never_downgrades(self, monkeypatch):
        """Auto-elevation never downgrades a configured tier."""
        import carl_studio.tier as tier_mod
        from carl_studio.tier import Tier, detect_effective_tier

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: None)

        assert detect_effective_tier(Tier.PRO) == Tier.PRO

    def test_enterprise_gates_mcp(self):
        """MCP features require PAID."""
        from carl_studio.tier import Tier, tier_allows

        assert not tier_allows(Tier.FREE, "mcp")
        assert tier_allows(Tier.PAID, "mcp")


# =========================================================================
# 4. Eval Journey
# =========================================================================


class TestEvalJourney:
    """Create eval config, build report, apply gate."""

    def test_eval_config_defaults(self):
        """EvalConfig has correct defaults."""
        from carl_studio.eval.runner import EvalConfig

        config = EvalConfig(checkpoint="wheattoast11/OmniCoder-9B-Zero-Phase2")
        assert config.dataset == "wheattoast11/zero-rl-tool-calling-data"
        assert config.phase == "auto"
        assert config.threshold == 0.3
        assert config.max_new_tokens == 2048

    def test_eval_config_phase_validation(self):
        """EvalConfig rejects invalid phase values."""
        from carl_studio.eval.runner import EvalConfig

        with pytest.raises(Exception):
            EvalConfig(checkpoint="test", phase="invalid")

    def test_eval_report_structure(self):
        """EvalReport holds structured results."""
        from carl_studio.eval.runner import EvalReport

        report = EvalReport(
            checkpoint="test/model",
            phase="1",
            n_samples=100,
            metrics={"format_accuracy": 0.95, "selection_accuracy": 0.88},
            primary_metric="format_accuracy",
            primary_value=0.95,
            threshold=0.5,
            passed=True,
        )
        assert report.passed is True
        assert report.primary_value == 0.95
        assert report.n_samples == 100

    def test_eval_gate_pass(self):
        """EvalGate.check() returns True when primary_value >= threshold."""
        from carl_studio.eval.runner import EvalGate, EvalReport

        gate = EvalGate(threshold=0.5, phase="1")
        report = EvalReport(
            checkpoint="test",
            phase="1",
            n_samples=10,
            primary_metric="acc",
            primary_value=0.75,
            threshold=0.5,
            passed=True,
        )
        assert gate.check(report) is True

    def test_eval_gate_fail(self):
        """EvalGate.check() returns False when primary_value < threshold."""
        from carl_studio.eval.runner import EvalGate, EvalReport

        gate = EvalGate(threshold=0.5, phase="1")
        report = EvalReport(
            checkpoint="test",
            phase="1",
            n_samples=10,
            primary_metric="acc",
            primary_value=0.3,
            threshold=0.5,
            passed=False,
        )
        assert gate.check(report) is False

    def test_eval_gate_phase_defaults(self):
        """Each phase has a default threshold."""
        from carl_studio.eval.runner import EvalGate

        gate_1 = EvalGate(phase="1")
        gate_2prime = EvalGate(phase="2prime")
        assert gate_1.threshold == 0.5
        assert gate_2prime.threshold == 0.30

    def test_eval_gate_invalid_threshold(self):
        """EvalGate rejects threshold outside [0, 1]."""
        from carl_studio.eval.runner import EvalGate

        with pytest.raises(ValueError):
            EvalGate(threshold=1.5)

    def test_parse_tool_calls_qwen_format(self):
        """parse_tool_calls handles Qwen3.5 native format."""
        from carl_studio.eval.runner import parse_tool_calls

        text = (
            "<tool_call>"
            "<function=write_file>"
            "<parameter=path>test.py</parameter>"
            "<parameter=content>print('hello')</parameter>"
            "</function>"
            "</tool_call>"
        )
        calls = parse_tool_calls(text)
        assert len(calls) >= 1
        assert calls[0]["name"] == "write_file"
        assert calls[0]["arguments"]["path"] == "test.py"

    def test_parse_tool_calls_json_format(self):
        """parse_tool_calls handles JSON-in-tags format."""
        from carl_studio.eval.runner import parse_tool_calls

        text = '<tool_call>{"name": "execute_code", "arguments": {"code": "print(1)"}}</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "execute_code"

    def test_parse_tool_calls_empty(self):
        """parse_tool_calls returns empty list for no-tool text."""
        from carl_studio.eval.runner import parse_tool_calls

        assert parse_tool_calls("Just a regular response, no tools.") == []


# =========================================================================
# 5. Environment Journey
# =========================================================================


class TestEnvironmentJourney:
    """Full user journey: create env -> reset -> tool calls -> reward."""

    def test_code_sandbox_full_cycle(self):
        """Write file -> execute code -> check reward."""
        from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv

        env = CodingSandboxEnv()
        obs = env.reset(task_description="Write hello world")
        assert obs == "Write hello world"
        assert env.reward == 0.0

        env.write_file("main.py", "print('Hello, World!')")
        result = env.execute_code("print('Hello, World!')")
        assert "Hello, World!" in result
        assert env.reward == 1.0
        assert env.turn_count == 2  # write + execute
        assert env._execution_succeeded is True

    def test_code_sandbox_failure_recovery(self):
        """Failed execution -> fix -> success shows reward recovery."""
        from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv

        env = CodingSandboxEnv()
        env.reset(task_description="Fix and run")

        # First attempt fails
        env.execute_code("raise RuntimeError('bug')")
        assert env.reward == 0.0

        # Second attempt succeeds
        env.execute_code("print('fixed')")
        assert env.reward == 1.0

    def test_code_sandbox_security(self):
        """Path traversal is blocked."""
        from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv

        env = CodingSandboxEnv()
        env.reset(task_description="Security test")

        result = env.write_file("../../etc/evil", "payload")
        assert "Error" in result

        result = env.read_file("../../../etc/passwd")
        assert "Error" in result

    def test_code_sandbox_history_tracking(self):
        """History records all tool calls with metadata."""
        from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv

        env = CodingSandboxEnv()
        env.reset(task_description="History test")

        env.write_file("a.py", "x = 1")
        env.read_file("a.py")
        env.execute_code("print(1)")

        history = env.history
        assert len(history) == 3
        assert history[0]["tool"] == "write_file"
        assert history[1]["tool"] == "read_file"
        assert history[2]["tool"] == "execute_code"
        assert all("turn" in h for h in history)
        assert all("reward_at_turn" in h for h in history)

    def test_sql_sandbox_full_cycle(self):
        """Create schema -> query -> score against expected."""
        from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv

        env = SQLSandboxEnv()
        obs = env.reset(
            task_description="How many orders are there?",
            schema_ddl=(
                "CREATE TABLE orders (id INTEGER, customer TEXT, amount REAL);"
                "INSERT INTO orders VALUES (1, 'Alice', 99.99);"
                "INSERT INTO orders VALUES (2, 'Bob', 49.50);"
                "INSERT INTO orders VALUES (3, 'Charlie', 150.00);"
            ),
            expected_result="3",
        )
        assert obs == "How many orders are there?"

        # Inspect schema
        tables = env.list_tables()
        assert "orders" in tables
        assert "3 rows" in tables

        # Query
        result = env.execute_query("SELECT COUNT(*) as total FROM orders")
        assert "3" in result
        assert env.reward == 1.0

    def test_sql_sandbox_read_only_enforcement(self):
        """Non-SELECT queries are blocked."""
        from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv

        env = SQLSandboxEnv()
        env.reset(
            task_description="Test",
            schema_ddl="CREATE TABLE t (x INT);",
        )

        for dangerous in [
            "DROP TABLE t",
            "DELETE FROM t",
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET x = 1",
        ]:
            result = env.execute_query(dangerous)
            assert "Error" in result, f"Should block: {dangerous}"

    def test_sql_sandbox_writable_tables(self):
        """insert_data works only on designated writable tables."""
        from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv

        env = SQLSandboxEnv()
        env.reset(
            task_description="Insert test",
            schema_ddl="CREATE TABLE log (msg TEXT); CREATE TABLE config (k TEXT);",
            writable_tables=["log"],
        )

        # Writable table accepts insert
        result = env.insert_data("log", "msg", "'test message'")
        assert "Inserted" in result

        # Non-writable table is blocked
        result = env.insert_data("config", "k", "'bad'")
        assert "Error" in result
        assert "read-only" in result.lower() or "writable" in result.lower()

    def test_environment_validation(self):
        """Both builtin environments pass TRL validation."""
        from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv
        from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv
        from carl_studio.environments.validation import validate_environment

        assert validate_environment(CodingSandboxEnv) == []
        assert validate_environment(SQLSandboxEnv) == []


# =========================================================================
# 6. Naming Journey
# =========================================================================


class TestNamingJourney:
    """Naming prefix validation and settings field contract."""

    def test_naming_prefix_default(self):
        """Default naming prefix follows convention."""
        from carl_studio.settings import CARLSettings

        s = CARLSettings()
        assert s.naming_prefix == "il-terminals-carl"

    def test_naming_prefix_settable(self):
        """naming_prefix can be set via set_field."""
        from carl_studio.settings import CARLSettings, set_field

        s = CARLSettings()
        s = set_field(s, "naming_prefix", "my-org-project")
        assert s.naming_prefix == "my-org-project"

    def test_naming_prefix_persists(self, tmp_path):
        """naming_prefix survives YAML roundtrip."""
        import yaml
        from carl_studio.settings import CARLSettings

        config_path = tmp_path / "config.yaml"
        s = CARLSettings(naming_prefix="custom-prefix-v2")
        s.save(config_path)

        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert data["naming_prefix"] == "custom-prefix-v2"

    def test_settable_fields_registry(self):
        """All SETTABLE_FIELDS keys map to valid CARLSettings paths."""
        from carl_studio.settings import CARLSettings, SETTABLE_FIELDS

        s = CARLSettings()
        for key in SETTABLE_FIELDS:
            parts = key.split(".")
            obj = s
            for part in parts:
                assert hasattr(obj, part), f"Settings has no attribute path: {key}"
                obj = getattr(obj, part)

    def test_compute_target_enum_coverage(self):
        """ComputeTarget enum covers all expected hardware flavors."""
        from carl_studio.types.config import ComputeTarget

        expected = {"l4x1", "l4x4", "a10g-large", "a100-large", "l40sx1", "local"}
        actual = {ct.value for ct in ComputeTarget}
        assert expected.issubset(actual), f"Missing: {expected - actual}"

    def test_log_level_validation(self):
        """Invalid log_level is rejected."""
        from carl_studio.settings import CARLSettings

        with pytest.raises(Exception):
            CARLSettings(log_level="INVALID")

    def test_log_level_case_insensitive(self):
        """Log level accepts any case."""
        from carl_studio.settings import CARLSettings

        s = CARLSettings(log_level="DEBUG")
        assert s.log_level == "debug"

    def test_preset_enum_values(self):
        """All preset values are valid."""
        from carl_studio.settings import Preset

        assert Preset.RESEARCH.value == "research"
        assert Preset.PRODUCTION.value == "production"
        assert Preset.QUICK.value == "quick"
        assert Preset.CUSTOM.value == "custom"

    def test_tier_enum_ordering(self):
        """Tier ordering: FREE < PAID; aliases map to PAID."""
        from carl_studio.tier import Tier

        assert Tier.FREE < Tier.PAID
        assert Tier.PRO == Tier.PAID
        assert Tier.ENTERPRISE == Tier.PAID
        assert not Tier.FREE > Tier.PAID


# =========================================================================
# Integration: Cross-journey validation
# =========================================================================


class TestCrossJourney:
    """Tests that span multiple subsystems."""

    def test_observe_to_eval_pipeline(self, tmp_path):
        """Observe data -> check if training is ready -> eval gate decision."""
        from carl_studio.eval.runner import EvalGate, EvalReport
        from carl_studio.observe.data_source import FileSource

        # Simulate a training run producing metrics
        log_file = tmp_path / "train.jsonl"
        log_file.write_text(
            "\n".join(
                json.dumps(
                    {
                        "step": i,
                        "phi": 0.1 * i,
                        "loss": 3.0 - 0.2 * i,
                        "reward_mean": 0.1 * i,
                    }
                )
                for i in range(1, 11)
            )
            + "\n"
        )

        source = FileSource(str(log_file))
        frames = source.poll()
        assert len(frames) == 10
        final_phi = frames[-1].phi
        assert final_phi > 0.5  # Training converged

        # Create eval report based on "model output"
        report = EvalReport(
            checkpoint="test/model",
            phase="1",
            n_samples=50,
            metrics={"format_accuracy": 0.92},
            primary_metric="format_accuracy",
            primary_value=0.92,
            threshold=0.5,
            passed=True,
        )

        gate = EvalGate(threshold=0.5, phase="1")
        assert gate.check(report) is True

    def test_settings_tier_environment_integration(self, monkeypatch):
        """Settings tier affects which environments are accessible."""
        import carl_studio.tier as tier_mod
        from carl_studio.settings import CARLSettings
        from carl_studio.tier import Tier

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: None)

        # FREE tier can use environments (they're part of train)
        s = CARLSettings(tier=Tier.FREE)
        assert s.tier_allows("train") is True

        # But can't use send-it pipeline
        assert s.tier_allows("train.send_it") is False

    def test_full_sandbox_with_eval_report(self):
        """Run code in sandbox -> build eval report from results."""
        from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv
        from carl_studio.eval.runner import EvalReport

        env = CodingSandboxEnv()
        env.reset(task_description="Add two numbers")
        env.execute_code("print(2 + 3)")

        # Build eval report from sandbox result
        report = EvalReport(
            checkpoint="test/model",
            phase="2prime",
            n_samples=1,
            metrics={
                "task_completion": env.reward,
                "tool_calls": env.turn_count,
            },
            primary_metric="task_completion",
            primary_value=env.reward,
            threshold=0.3,
            passed=env.reward >= 0.3,
        )

        assert report.passed is True
        assert report.metrics["task_completion"] == 1.0
