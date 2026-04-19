"""Tests for the OpenEnv bridge — server + client round-trip."""

from __future__ import annotations

import json
from typing import Any

import pytest

from carl_core.connection import ConnectionState, reset_registry
from carl_core.interaction import InteractionChain

from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv
from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv
from carl_studio.environments.openenv import (
    OPENENV_CLIENT_CONNECTION_SPEC,
    OpenEnvAction,
    OpenEnvClient,
    OpenEnvObservation,
    OpenEnvServer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_conn_registry():  # pyright: ignore[reportUnusedFunction]
    reset_registry()
    yield
    reset_registry()


# ---------------------------------------------------------------------------
# OpenEnvAction / OpenEnvObservation
# ---------------------------------------------------------------------------


class TestOpenEnvAction:
    def test_roundtrip(self):
        a = OpenEnvAction(tool="execute_code", args={"code": "print(1)"})
        d = a.to_dict()
        assert d == {"tool": "execute_code", "args": {"code": "print(1)"}, "reasoning": None}
        b = OpenEnvAction.from_dict(d)
        assert b.tool == a.tool
        assert b.args == a.args

    def test_reasoning_optional(self):
        a = OpenEnvAction(tool="run_shell", args={"command": "ls"}, reasoning="listing")
        assert a.reasoning == "listing"
        b = OpenEnvAction.from_dict(a.to_dict())
        assert b.reasoning == "listing"

    def test_from_dict_rejects_empty_tool(self):
        with pytest.raises(ValueError):
            OpenEnvAction.from_dict({"tool": ""})

    def test_from_dict_rejects_non_dict_args(self):
        with pytest.raises(ValueError):
            OpenEnvAction.from_dict({"tool": "x", "args": "not a dict"})

    def test_from_dict_rejects_non_string_reasoning(self):
        with pytest.raises(ValueError):
            OpenEnvAction.from_dict({"tool": "x", "args": {}, "reasoning": 5})


class TestOpenEnvObservation:
    def test_defaults(self):
        obs = OpenEnvObservation()
        assert obs.text is None
        assert obs.reward == 0.0
        assert obs.done is False
        assert obs.turn == 0

    def test_to_dict_is_json_safe(self):
        obs = OpenEnvObservation(
            text="ok",
            tool_result=None,
            reward=0.5,
            done=True,
            turn=3,
            meta={"tool": "execute_code"},
        )
        payload = json.dumps(obs.to_dict())
        assert "execute_code" in payload

    def test_from_dict_coerces_bad_reward(self):
        obs = OpenEnvObservation.from_dict({"reward": "not a float"})
        assert obs.reward == 0.0

    def test_from_dict_requires_dict(self):
        with pytest.raises(ValueError):
            OpenEnvObservation.from_dict([])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OpenEnvServer wrapping CodingSandboxEnv
# ---------------------------------------------------------------------------


class TestOpenEnvServerWithCodingSandbox:
    def test_requires_environment_connection(self):
        with pytest.raises(TypeError):
            OpenEnvServer("not an env")  # type: ignore[arg-type]

    def test_reset_returns_observation(self):
        env = CodingSandboxEnv()
        env.open()
        try:
            server = OpenEnvServer(env)
            obs = server.reset(task_description="print 42")
            assert isinstance(obs, OpenEnvObservation)
            assert obs.text == "print 42"
            assert obs.turn == 0
            assert obs.done is False
            assert obs.meta["op"] == "reset"
        finally:
            env.close()

    def test_reset_auto_opens_if_init(self):
        env = CodingSandboxEnv()
        # Do NOT open — server should open it on first reset.
        try:
            server = OpenEnvServer(env)
            server.reset(task_description="x")
            assert env.state == ConnectionState.READY
        finally:
            env.close()

    def test_step_dispatches_to_tool(self):
        env = CodingSandboxEnv()
        env.open()
        try:
            server = OpenEnvServer(env)
            server.reset(task_description="eval")
            obs = server.step(OpenEnvAction(tool="execute_code", args={"code": "print(42)"}))
            assert isinstance(obs, OpenEnvObservation)
            assert "42" in (obs.text or "")
            assert obs.turn == 1
            assert obs.reward == 1.0
            assert obs.meta["tool"] == "execute_code"
        finally:
            env.close()

    def test_step_accepts_dict(self):
        env = CodingSandboxEnv()
        env.open()
        try:
            server = OpenEnvServer(env)
            server.reset(task_description="x")
            obs = server.step({"tool": "execute_code", "args": {"code": "pass"}})
            assert obs.turn == 1
        finally:
            env.close()

    def test_step_rejects_non_action_non_dict(self):
        env = CodingSandboxEnv()
        env.open()
        try:
            server = OpenEnvServer(env)
            server.reset(task_description="x")
            with pytest.raises(TypeError):
                server.step("not-an-action")  # type: ignore[arg-type]
        finally:
            env.close()

    def test_step_unknown_tool_raises(self):
        env = CodingSandboxEnv()
        env.open()
        try:
            server = OpenEnvServer(env)
            server.reset(task_description="x")
            with pytest.raises(ValueError, match="not declared"):
                server.step(OpenEnvAction(tool="definitely_not_a_tool"))
        finally:
            env.close()

    def test_state_shape(self):
        env = CodingSandboxEnv()
        env.open()
        try:
            server = OpenEnvServer(env)
            server.reset(task_description="x")
            server.step(OpenEnvAction(tool="execute_code", args={"code": "pass"}))
            st = server.state()
            assert st["connection_id"] == env.connection_id
            assert st["turn_count"] == 1
            assert st["connection_state"] == "ready"
            assert st["spec"]["name"] == "python-sandbox"
            assert st["spec"]["lane"] == "code"
            assert "execute_code" in st["spec"]["tools"]
            assert isinstance(st["history"], list)
        finally:
            env.close()

    def test_serialize_is_deterministic_json(self):
        env = CodingSandboxEnv()
        env.open()
        try:
            server = OpenEnvServer(env)
            server.reset(task_description="x")
            blob = server.serialize()
            parsed = json.loads(blob)
            assert parsed["spec"]["name"] == "python-sandbox"
            # Sorted keys invariant.
            assert blob == json.dumps(server.state(), sort_keys=True)
        finally:
            env.close()

    def test_chain_records_step_events(self):
        chain = InteractionChain()
        env = CodingSandboxEnv()
        env.open()
        try:
            server = OpenEnvServer(env, chain=chain)
            server.reset(task_description="x")
            server.step(OpenEnvAction(tool="execute_code", args={"code": "pass"}))
            names = [s.name for s in chain.steps]
            assert "openenv.reset" in names
            assert "openenv.step" in names
        finally:
            env.close()


class TestOpenEnvServerWithSQLSandbox:
    def test_lane_preserved_in_state(self):
        env = SQLSandboxEnv()
        env.open()
        try:
            server = OpenEnvServer(env)
            server.reset(task_description="q", schema_ddl="CREATE TABLE t (x INT);")
            assert server.state()["spec"]["lane"] == "query"
        finally:
            env.close()


# ---------------------------------------------------------------------------
# OpenEnvClient against a mocked httpx peer
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self._status = status

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")


class _FakeHTTPClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []
        self._state_payload: dict[str, Any] = {
            "connection_id": "conn-fake",
            "connection_state": "ready",
            "spec": {
                "lane": "code",
                "name": "python-sandbox-remote",
                "tools": ["execute_code", "read_file"],
                "max_turns": 7,
                "reward_type": "binary",
                "multimodal": False,
                "dataset_columns": ["task_description"],
            },
            "turn_count": 0,
            "history": [],
            "reward": 0.0,
            "done": False,
        }
        self._reset_payload: dict[str, Any] = {
            "text": "remote task",
            "reward": 0.0,
            "done": False,
            "turn": 0,
            "meta": {"op": "reset"},
        }
        self._step_payload: dict[str, Any] = {
            "text": "42",
            "tool_result": "42",
            "reward": 1.0,
            "done": True,
            "turn": 1,
            "meta": {"op": "step", "tool": "execute_code"},
        }
        self.closed = False

    def get(self, path: str) -> _FakeResponse:
        self.gets.append(path)
        if path == "/state":
            return _FakeResponse(self._state_payload)
        return _FakeResponse({}, status=404)

    def post(self, path: str, json: dict[str, Any]) -> _FakeResponse:
        self.posts.append((path, json))
        if path == "/reset":
            return _FakeResponse(self._reset_payload)
        if path == "/step":
            return _FakeResponse(self._step_payload)
        return _FakeResponse({}, status=404)

    def close(self) -> None:
        self.closed = True


class TestOpenEnvClient:
    def test_connection_spec_is_http_authenticated(self):
        assert OPENENV_CLIENT_CONNECTION_SPEC.transport.value == "http"
        assert OPENENV_CLIENT_CONNECTION_SPEC.trust.value == "authenticated"

    def test_zero_arg_construction_allowed_for_trl(self):
        # TRL's env_factory calls cls() — this must not raise.
        client = OpenEnvClient()
        assert client.state == ConnectionState.INIT

    def test_open_without_base_url_raises(self):
        client = OpenEnvClient()
        # State becomes ERROR after a failed open(); exception propagates.
        from carl_core.connection import ConnectionUnavailableError
        with pytest.raises(ConnectionUnavailableError):
            client.open()

    def test_reset_before_open_uses_local_path(self):
        # Without a httpx session, reset should fall through to BaseEnvironment
        # behaviour instead of trying to talk to a remote.
        client = OpenEnvClient()
        result = client.reset()
        assert result is None
        assert client.turn_count == 0

    def test_reset_round_trip_through_mock(self, monkeypatch: pytest.MonkeyPatch):
        client = OpenEnvClient(base_url="http://fake.local")
        fake = _FakeHTTPClient()

        # Patch the transport without going through real httpx.
        def _fake_connect() -> None:
            client._httpx_client = fake  # pyright: ignore[reportPrivateUsage]
            client._load_remote_spec()  # pyright: ignore[reportPrivateUsage]

        monkeypatch.setattr(client, "_env_connect", _fake_connect)
        client.open()
        try:
            text = client.reset(task_description="remote task")
            assert text == "remote task"
            # The remote-authoritative reward/done should persist past super().reset().
            assert client.reward == 0.0
            assert client.done is False
            # /state was called on open to load the spec.
            assert "/state" in fake.gets
            assert fake.posts[0][0] == "/reset"
        finally:
            client.close()

    def test_step_round_trip_through_mock(self, monkeypatch: pytest.MonkeyPatch):
        client = OpenEnvClient(base_url="http://fake.local")
        fake = _FakeHTTPClient()

        def _fake_connect() -> None:
            client._httpx_client = fake  # pyright: ignore[reportPrivateUsage]
            client._load_remote_spec()  # pyright: ignore[reportPrivateUsage]

        monkeypatch.setattr(client, "_env_connect", _fake_connect)
        client.open()
        try:
            client.reset(task_description="x")
            obs = client.step(OpenEnvAction(tool="execute_code", args={"code": "print(42)"}))
            assert isinstance(obs, OpenEnvObservation)
            assert obs.reward == 1.0
            assert obs.done is True
            assert obs.turn == 1
            # Client mirrors remote-authoritative values locally.
            assert client.turn_count == 1
            assert client.reward == 1.0
            assert client.done is True
        finally:
            client.close()

    def test_remote_spec_replaces_placeholder(self, monkeypatch: pytest.MonkeyPatch):
        client = OpenEnvClient(base_url="http://fake.local")
        fake = _FakeHTTPClient()

        def _fake_connect() -> None:
            client._httpx_client = fake  # pyright: ignore[reportPrivateUsage]
            client._load_remote_spec()  # pyright: ignore[reportPrivateUsage]

        monkeypatch.setattr(client, "_env_connect", _fake_connect)
        client.open()
        try:
            assert client.spec.name == "python-sandbox-remote"
            assert "execute_code" in client.spec.tools
            assert client.spec.max_turns == 7
        finally:
            client.close()

    def test_step_before_open_raises(self):
        client = OpenEnvClient(base_url="http://fake.local")
        from carl_core.connection import ConnectionUnavailableError
        with pytest.raises(ConnectionUnavailableError):
            client.step(OpenEnvAction(tool="execute_code"))

    def test_close_is_idempotent(self, monkeypatch: pytest.MonkeyPatch):
        client = OpenEnvClient(base_url="http://fake.local")
        fake = _FakeHTTPClient()

        def _fake_connect() -> None:
            client._httpx_client = fake  # pyright: ignore[reportPrivateUsage]

        monkeypatch.setattr(client, "_env_connect", _fake_connect)
        client.open()
        client.close()
        client.close()
        assert fake.closed is True
