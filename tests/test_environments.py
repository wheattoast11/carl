"""Tests for CARL environment protocol, registry, validation, and builtins.

Converted from script-style to pytest to work with conftest.py stubs
(avoids carl_studio.__init__ triggering transformers import).
"""

import pytest

from carl_studio.environments.protocol import BaseEnvironment, EnvironmentLane, EnvironmentSpec
from carl_studio.environments.registry import (
    register_environment, get_environment, list_environments, clear_registry,
)
from carl_studio.environments.validation import validate_environment


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the environment registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


class _TestEnv(BaseEnvironment):
    """Minimal valid environment for testing."""

    spec = EnvironmentSpec(
        lane=EnvironmentLane.CODE, name="test-reg", tools=("do_thing",),
    )

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return None

    def do_thing(self, x: str) -> str:
        """Do a thing.

        Args:
            x: The thing.
        """
        return x


# =========================================================================
# EnvironmentLane
# =========================================================================


class TestEnvironmentLane:
    def test_code_value(self):
        assert EnvironmentLane.CODE == "code"

    def test_query_value(self):
        assert EnvironmentLane.QUERY == "query"

    def test_retrieval_value(self):
        assert EnvironmentLane.RETRIEVAL == "retrieval"

    def test_routing_value(self):
        assert EnvironmentLane.ROUTING == "routing"

    def test_infra_value(self):
        assert EnvironmentLane.INFRA == "infra"

    def test_visual_value(self):
        assert EnvironmentLane.VISUAL == "visual"

    def test_mece_count(self):
        assert len(EnvironmentLane) == 6


# =========================================================================
# EnvironmentSpec
# =========================================================================


class TestEnvironmentSpec:
    def test_basic_fields(self):
        spec = EnvironmentSpec(
            lane=EnvironmentLane.CODE, name="test-env",
            tools=("read", "write"), max_turns=5,
        )
        assert spec.lane == EnvironmentLane.CODE
        assert spec.name == "test-env"
        assert spec.tools == ("read", "write")
        assert spec.max_turns == 5

    def test_defaults(self):
        spec = EnvironmentSpec(
            lane=EnvironmentLane.CODE, name="t", tools=("x",),
        )
        assert spec.reward_type == "binary"
        assert spec.multimodal is False

    def test_frozen(self):
        spec = EnvironmentSpec(
            lane=EnvironmentLane.CODE, name="t", tools=("x",),
        )
        with pytest.raises(AttributeError):
            spec.name = "changed"


# =========================================================================
# Registry
# =========================================================================


class TestRegistry:
    def test_empty_after_clear(self):
        assert list_environments() == []

    def test_register_and_lookup(self):
        register_environment(_TestEnv)
        assert len(list_environments()) == 1
        assert get_environment("test-reg") is _TestEnv

    def test_duplicate_raises(self):
        register_environment(_TestEnv)
        with pytest.raises(ValueError):
            register_environment(_TestEnv)

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_environment("nonexistent")

    def test_filter_by_lane(self):
        register_environment(_TestEnv)
        assert len(list_environments(lane=EnvironmentLane.CODE)) == 1
        assert len(list_environments(lane=EnvironmentLane.QUERY)) == 0


# =========================================================================
# Validation
# =========================================================================


class TestValidation:
    def test_valid_env(self):
        assert validate_environment(_TestEnv) == []

    def test_missing_spec(self):
        class _NoSpec(BaseEnvironment):
            def reset(self, **kwargs):
                return None
            def tool(self, x: str) -> str:
                """T. Args: x: thing."""
                return x

        errors = validate_environment(_NoSpec)
        assert any("spec" in e.lower() for e in errors)

    def test_required_init_arg(self):
        class _BadInit(BaseEnvironment):
            spec = EnvironmentSpec(
                lane=EnvironmentLane.CODE, name="bad-init", tools=("t",),
            )
            def __init__(self, required_arg):
                super().__init__()
            def reset(self, **kwargs):
                return None
            def t(self, x: str) -> str:
                """T. Args: x: thing."""
                return x

        errors = validate_environment(_BadInit)
        assert any("init" in e.lower() or "__init__" in e for e in errors)

    def test_no_tools(self):
        class _NoTools(BaseEnvironment):
            spec = EnvironmentSpec(
                lane=EnvironmentLane.CODE, name="no-tools", tools=(),
            )
            def reset(self, **kwargs):
                return None

        errors = validate_environment(_NoTools)
        assert any("tool" in e.lower() for e in errors)

    def test_missing_docstring(self):
        class _NoDoc(BaseEnvironment):
            spec = EnvironmentSpec(
                lane=EnvironmentLane.CODE, name="no-doc", tools=("bad_tool",),
            )
            def reset(self, **kwargs):
                return None
            def bad_tool(self, x: str) -> str:
                return x

        errors = validate_environment(_NoDoc)
        assert any("docstring" in e.lower() for e in errors)

    def test_missing_type_hint(self):
        class _NoHint(BaseEnvironment):
            spec = EnvironmentSpec(
                lane=EnvironmentLane.CODE, name="no-hint", tools=("untyped",),
            )
            def reset(self, **kwargs):
                return None
            def untyped(self, x) -> str:
                """Do thing. Args: x: thing."""
                return str(x)

        errors = validate_environment(_NoHint)
        assert any("type hint" in e.lower() for e in errors)

    def test_spec_method_mismatch(self):
        class _Mismatch(BaseEnvironment):
            spec = EnvironmentSpec(
                lane=EnvironmentLane.CODE, name="mismatch",
                tools=("declared_but_missing",),
            )
            def reset(self, **kwargs):
                return None
            def actual_method(self, x: str) -> str:
                """Do thing. Args: x: thing."""
                return x

        errors = validate_environment(_Mismatch)
        assert any("not found" in e.lower() or "not declared" in e.lower() for e in errors)


# =========================================================================
# CodingSandboxEnv
# =========================================================================


class TestCodingSandboxEnv:
    @pytest.fixture(autouse=True)
    def setup(self):
        from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv
        self.EnvClass = CodingSandboxEnv
        self.env = CodingSandboxEnv()
        self.env.reset(task_description="Test task")
        yield
        del self.env

    def test_validation(self):
        assert validate_environment(self.EnvClass) == []

    def test_spec(self):
        assert self.EnvClass.spec.lane == EnvironmentLane.CODE
        assert self.EnvClass.spec.name == "python-sandbox"
        assert len(self.EnvClass.spec.tools) == 4

    def test_reset(self):
        env = self.EnvClass()
        obs = env.reset(task_description="Test task")
        assert obs == "Test task"
        assert env.reward == 0.0
        assert env.turn_count == 0

    def test_write_increments_turns(self):
        self.env.write_file("test.py", "print('hello')")
        assert self.env.turn_count == 1

    def test_execute_success(self):
        result = self.env.execute_code("print('hello')")
        assert "hello" in result
        assert self.env.reward == 1.0
        assert self.env._execution_succeeded is True

    def test_execute_failure(self):
        self.env.execute_code("raise ValueError('oops')")
        assert self.env.reward == 0.0

    def test_read_write_file(self):
        self.env.write_file("data.txt", "content here")
        result = self.env.read_file("data.txt")
        assert result == "content here"

    def test_read_nonexistent(self):
        result = self.env.read_file("nope.txt")
        assert "Error" in result

    def test_path_escape_blocked(self):
        result = self.env.read_file("../../etc/passwd")
        assert "Error" in result

    def test_history_recorded(self):
        self.env.write_file("t.py", "x = 1")
        assert len(self.env.history) > 0


# =========================================================================
# SQLSandboxEnv
# =========================================================================


class TestSQLSandboxEnv:
    @pytest.fixture(autouse=True)
    def setup(self):
        from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv
        self.EnvClass = SQLSandboxEnv
        yield

    def test_validation(self):
        assert validate_environment(self.EnvClass) == []

    def test_spec(self):
        assert self.EnvClass.spec.lane == EnvironmentLane.QUERY
        assert self.EnvClass.spec.name == "sqlite-sandbox"

    def test_reset_returns_task(self):
        env = self.EnvClass()
        obs = env.reset(
            task_description="Count users",
            schema_ddl=(
                "CREATE TABLE users (id INTEGER, name TEXT); "
                "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');"
            ),
            expected_result="3",
        )
        assert obs == "Count users"

    def test_list_tables(self):
        env = self.EnvClass()
        env.reset(
            task_description="Count users",
            schema_ddl=(
                "CREATE TABLE users (id INTEGER, name TEXT); "
                "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');"
            ),
        )
        tables = env.list_tables()
        assert "users" in tables
        assert "3 rows" in tables

    def test_describe_table(self):
        env = self.EnvClass()
        env.reset(
            task_description="Inspect",
            schema_ddl="CREATE TABLE t (x INTEGER, y TEXT);",
        )
        desc = env.describe_table("t")
        assert "t" in desc

    def test_execute_query(self):
        env = self.EnvClass()
        env.reset(
            task_description="Count users",
            schema_ddl=(
                "CREATE TABLE users (id INTEGER, name TEXT); "
                "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');"
            ),
            expected_result="3",
        )
        result = env.execute_query("SELECT COUNT(*) as cnt FROM users")
        assert "3" in result

    def test_reject_non_select(self):
        env = self.EnvClass()
        env.reset(
            task_description="Test",
            schema_ddl="CREATE TABLE t (x INT);",
        )
        result = env.execute_query("DROP TABLE t")
        assert "Error" in result

    def test_correct_score(self):
        env = self.EnvClass()
        env.reset(
            task_description="Count users",
            schema_ddl=(
                "CREATE TABLE users (id INTEGER, name TEXT); "
                "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');"
            ),
            expected_result="3",
        )
        env.execute_query("SELECT COUNT(*) as cnt FROM users")
        assert env.reward == 1.0

    def test_wrong_score(self):
        env = self.EnvClass()
        env.reset(
            task_description="Count",
            schema_ddl="CREATE TABLE t (x INT);",
            expected_result="99",
        )
        env.execute_query("SELECT COUNT(*) FROM t")
        assert env.reward == 0.0

    def test_partial_match_score(self):
        env = self.EnvClass()
        env.reset(
            task_description="Find name",
            schema_ddl="CREATE TABLE t (name TEXT); INSERT INTO t VALUES ('Alice'), ('Bob');",
            expected_result="Alice and Bob",
        )
        env.execute_query("SELECT name FROM t WHERE name = 'Alice'")
        assert env.reward == 0.5

    def test_turn_count(self):
        env = self.EnvClass()
        env.reset(
            task_description="Test",
            schema_ddl="CREATE TABLE t (x INT);",
        )
        env.list_tables()
        env.describe_table("t")
        assert env.turn_count == 2


# =========================================================================
# Cross-lane registry
# =========================================================================


class TestCrossLaneRegistry:
    def test_both_envs_registered(self):
        from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv
        from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv

        register_environment(CodingSandboxEnv)
        register_environment(SQLSandboxEnv)
        all_envs = list_environments()
        assert len(all_envs) == 2

        code = list_environments(lane=EnvironmentLane.CODE)
        query = list_environments(lane=EnvironmentLane.QUERY)
        assert len(code) == 1
        assert len(query) == 1
