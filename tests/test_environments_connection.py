"""Tests for EnvironmentConnection — the carl_core.connection bridge.

Covers:
  * FSM lifecycle (INIT -> CONNECTING -> READY -> TRANSACTING -> READY -> CLOSED)
  * InteractionChain event recording (connection.open/reset/turn/close)
  * TRL zero-arg __init__ preservation for auto-discovery
  * EnvironmentSpec.lane metadata preservation on migrated builtins
  * connection_spec declaration on each builtin
"""

from __future__ import annotations

import pytest

from carl_core.connection import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionState,
    ConnectionTransport,
    ConnectionTrust,
    reset_registry,
)
from carl_core.interaction import InteractionChain

# Import from specific submodules rather than the package __init__ — the
# test conftest stubs the package namespace and pre-registers light
# submodules; new modules (like connection) need to come straight from the
# file to bypass the stub.
from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv
from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv
from carl_studio.environments.connection import (
    DEFAULT_ENVIRONMENT_CONNECTION_SPEC,
    EnvironmentConnection,
)
from carl_studio.environments.protocol import EnvironmentLane, EnvironmentSpec
from carl_studio.environments.validation import validate_environment


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def _reset_conn_registry():  # pyright: ignore[reportUnusedFunction]
    """Each test starts with a fresh connection registry."""
    reset_registry()
    yield
    reset_registry()


# =========================================================================
# Default ConnectionSpec
# =========================================================================


class TestDefaultConnectionSpec:
    def test_default_is_in_process(self):
        assert DEFAULT_ENVIRONMENT_CONNECTION_SPEC.transport == ConnectionTransport.IN_PROCESS

    def test_default_is_one_p(self):
        assert DEFAULT_ENVIRONMENT_CONNECTION_SPEC.scope == ConnectionScope.ONE_P

    def test_default_is_environment_kind(self):
        assert DEFAULT_ENVIRONMENT_CONNECTION_SPEC.kind == ConnectionKind.ENVIRONMENT

    def test_default_is_bidirectional(self):
        assert DEFAULT_ENVIRONMENT_CONNECTION_SPEC.direction == ConnectionDirection.BIDIRECTIONAL

    def test_default_is_public(self):
        assert DEFAULT_ENVIRONMENT_CONNECTION_SPEC.trust == ConnectionTrust.PUBLIC


# =========================================================================
# TRL zero-arg __init__ preservation
# =========================================================================


class TestTRLZeroArgConstruction:
    """TRL's environment_factory auto-discovers envs and calls cls()."""

    def test_environment_connection_base_zero_arg(self):
        # EnvironmentConnection itself is zero-arg constructible — the
        # reset() override in the base class is concrete so subclasses
        # that don't override it still satisfy TRL's discovery contract.
        e = EnvironmentConnection()
        assert e.state == ConnectionState.INIT
        # It ships with the default connection spec.
        assert e.connection_spec is DEFAULT_ENVIRONMENT_CONNECTION_SPEC

    def test_coding_sandbox_zero_arg(self):
        e = CodingSandboxEnv()
        assert e.spec.name == "python-sandbox"

    def test_sql_sandbox_zero_arg(self):
        e = SQLSandboxEnv()
        assert e.spec.name == "sqlite-sandbox"

    def test_init_signature_has_no_required_params(self):
        import inspect

        for cls in (CodingSandboxEnv, SQLSandboxEnv):
            sig = inspect.signature(cls.__init__)
            required = [
                p for p in sig.parameters.values()
                if p.name != "self"
                and p.default is inspect.Parameter.empty
                and p.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            assert required == [], f"{cls.__name__} requires params beyond self: {required}"


# =========================================================================
# EnvironmentSpec.lane metadata preservation
# =========================================================================


class TestLaneMetadataPreserved:
    def test_coding_sandbox_lane(self):
        e = CodingSandboxEnv()
        assert isinstance(e.spec, EnvironmentSpec)
        assert e.spec.lane == EnvironmentLane.CODE

    def test_sql_sandbox_lane(self):
        e = SQLSandboxEnv()
        assert isinstance(e.spec, EnvironmentSpec)
        assert e.spec.lane == EnvironmentLane.QUERY

    def test_coding_spec_still_env_spec_type(self):
        # Registry and validator both rely on isinstance(cls.spec, EnvironmentSpec).
        assert isinstance(CodingSandboxEnv.spec, EnvironmentSpec)

    def test_sql_spec_still_env_spec_type(self):
        assert isinstance(SQLSandboxEnv.spec, EnvironmentSpec)


# =========================================================================
# connection_spec declaration
# =========================================================================


class TestConnectionSpecDeclaration:
    def test_coding_connection_spec_is_connection_spec(self):
        assert isinstance(CodingSandboxEnv.connection_spec, ConnectionSpec)

    def test_coding_connection_spec_name(self):
        assert CodingSandboxEnv.connection_spec.name == "carl.env.code"

    def test_sql_connection_spec_is_connection_spec(self):
        assert isinstance(SQLSandboxEnv.connection_spec, ConnectionSpec)

    def test_sql_connection_spec_name(self):
        assert SQLSandboxEnv.connection_spec.name == "carl.env.sql"

    def test_coding_connection_spec_kind(self):
        assert CodingSandboxEnv.connection_spec.kind == ConnectionKind.ENVIRONMENT

    def test_coding_connection_spec_in_process(self):
        assert CodingSandboxEnv.connection_spec.transport == ConnectionTransport.IN_PROCESS


# =========================================================================
# FSM lifecycle
# =========================================================================


class TestLifecycle:
    def test_init_state(self):
        e = CodingSandboxEnv()
        assert e.state == ConnectionState.INIT

    def test_open_transitions_to_ready(self):
        e = CodingSandboxEnv()
        e.open()
        try:
            assert e.state == ConnectionState.READY
        finally:
            e.close()

    def test_close_transitions_to_closed(self):
        e = CodingSandboxEnv()
        e.open()
        e.close()
        assert e.state == ConnectionState.CLOSED

    def test_close_is_idempotent(self):
        e = CodingSandboxEnv()
        e.open()
        e.close()
        e.close()  # should not raise
        assert e.state == ConnectionState.CLOSED

    def test_context_manager(self):
        e = CodingSandboxEnv()
        with e:
            assert e.state == ConnectionState.READY
        assert e.state == ConnectionState.CLOSED

    def test_transact_cycle_preserves_ready(self):
        e = CodingSandboxEnv()
        try:
            e.open()
            e.reset(task_description="noop")
            with e.transact("execute_code"):
                e.execute_code("pass")
            assert e.state == ConnectionState.READY
        finally:
            e.close()

    def test_transact_moves_to_degraded_on_error(self):
        e = CodingSandboxEnv()
        try:
            e.open()
            e.reset(task_description="noop")
            with pytest.raises(RuntimeError):
                with e.transact("execute_code"):
                    raise RuntimeError("boom")
            assert e.state == ConnectionState.DEGRADED
        finally:
            e.close()

    def test_transact_requires_ready(self):
        e = CodingSandboxEnv()
        # Not opened — must raise on transact.
        from carl_core.connection import ConnectionClosedError
        with pytest.raises(ConnectionClosedError):
            with e.transact("execute_code"):
                pass

    def test_stats_counts_opens_and_closes(self):
        e = CodingSandboxEnv()
        e.open()
        e.close()
        stats = e.stats
        assert stats.opens == 1
        assert stats.closes == 1


# =========================================================================
# InteractionChain event recording
# =========================================================================


class TestChainRecording:
    def test_events_emitted_when_chain_attached(self):
        chain = InteractionChain()
        e = CodingSandboxEnv()
        e.attach_chain(chain)
        e.open()
        e.reset(task_description="hello")
        with e.transact("execute_code"):
            e.execute_code("print('x')")
        e.close()

        names = [s.name for s in chain.steps]
        assert "connection.open" in names
        assert "connection.reset" in names
        assert "connection.turn" in names
        assert "connection.execute_code" in names
        assert "connection.close" in names

    def test_no_events_when_no_chain(self):
        # Default construction = no chain attached = no leaks.
        e = CodingSandboxEnv()
        e.open()
        e.reset(task_description="hello")
        with e.transact("execute_code"):
            e.execute_code("print('x')")
        e.close()
        # No chain means no error — just verify terminal state.
        assert e.state == ConnectionState.CLOSED

    def test_attach_chain_after_construction(self):
        chain = InteractionChain()
        e = CodingSandboxEnv()
        # Attach BEFORE open to capture the full trace.
        e.attach_chain(chain)
        e.open()
        e.close()
        names = [s.name for s in chain.steps]
        assert "connection.open" in names
        assert "connection.close" in names

    def test_chain_step_success_flag_on_transact_error(self):
        chain = InteractionChain()
        e = CodingSandboxEnv()
        e.attach_chain(chain)
        e.open()
        e.reset(task_description="boom")
        with pytest.raises(RuntimeError):
            with e.transact("oops"):
                raise RuntimeError("x")
        # Close the failed connection (from DEGRADED).
        e.close()

        transact_steps = [s for s in chain.steps if s.name == "connection.oops"]
        assert len(transact_steps) == 1
        assert transact_steps[0].success is False

    def test_chain_records_connection_id_in_input(self):
        chain = InteractionChain()
        e = CodingSandboxEnv()
        e.attach_chain(chain)
        e.open()
        e.close()
        opens = [s for s in chain.steps if s.name == "connection.open"]
        assert len(opens) == 1
        assert opens[0].input["connection_id"] == e.connection_id


# =========================================================================
# End-to-end: open -> reset -> transact -> close with CodingSandboxEnv
# =========================================================================


class TestCodingSandboxEndToEnd:
    def test_full_lifecycle_with_code_execution(self):
        chain = InteractionChain()
        e = CodingSandboxEnv()
        e.attach_chain(chain)
        e.open()
        try:
            obs = e.reset(task_description="print 42")
            assert obs == "print 42"
            with e.transact("execute_code"):
                result = e.execute_code("print(42)")
            assert "42" in result
            assert e.reward == 1.0
        finally:
            e.close()

        # Chain recorded the full sequence.
        names = [s.name for s in chain.steps]
        assert names[0] == "connection.open"
        assert names[-1] == "connection.close"
        # connection.turn records per _record_turn call (execute_code).
        assert any(n == "connection.turn" for n in names)

    def test_to_dict_includes_env_spec_and_conn_state(self):
        e = CodingSandboxEnv()
        e.open()
        try:
            payload = e.to_dict()
            assert payload["state"] == "ready"
            assert payload["spec"]["name"] == "carl.env.code"
            assert payload["environment_spec"]["lane"] == "code"
            assert payload["environment_spec"]["name"] == "python-sandbox"
            assert "read_file" in payload["environment_spec"]["tools"]
        finally:
            e.close()


# =========================================================================
# Validation still passes after migration
# =========================================================================


class TestMigratedEnvsValidate:
    def test_coding_sandbox_validates_empty(self):
        errs = validate_environment(CodingSandboxEnv)
        assert errs == []

    def test_sql_sandbox_validates_empty(self):
        errs = validate_environment(SQLSandboxEnv)
        assert errs == []
