"""Robustness tests for carl_studio.chat_agent — WS-R1..R6.

Covers the six production-hardening passes:

* WS-R1 — Streaming error resilience: APIError, ConnectionError, KeyboardInterrupt
* WS-R2 — Session corruption handling: bad JSON, schema mismatch, quarantine
* WS-R3 — Budget pre-check + hard cap
* WS-R4 — Tool timeout + cancellation, hook isolation
* WS-R5 — Tool arg schema validation
* WS-R6 — dispatch_cli agent tool
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator
from unittest.mock import MagicMock

import pytest

from carl_studio.chat_agent import (
    TOOLS,
    CARLAgent,
    SessionStore,
    ToolPermission,
    _SESSION_SCHEMA_VERSION,
    _estimate_turn_upper_bound_cost,
    _validate_tool_args,
)


# =========================================================================
# SDK stream stubs
# =========================================================================


class _Delta:
    def __init__(self, text: str) -> None:
        self.type = "text_delta"
        self.text = text


class _Event:
    def __init__(self, text: str) -> None:
        self.type = "content_block_delta"
        self.delta = _Delta(text)


class _Usage:
    def __init__(
        self,
        input_tokens: int = 100,
        output_tokens: int = 20,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self, name: str, args: dict[str, Any], block_id: str = "tu_1") -> None:
        self.type = "tool_use"
        self.name = name
        self.input = args
        self.id = block_id


class _Response:
    def __init__(
        self,
        content: Iterable[Any],
        *,
        stop_reason: str = "end_turn",
        usage: _Usage | None = None,
    ) -> None:
        self.content = list(content)
        self.stop_reason = stop_reason
        self.usage = usage or _Usage()


class _StubStream:
    """Mock of the Anthropic streaming context manager.

    - ``events`` is the iterable yielded during streaming.
    - ``final`` is returned by ``get_final_message()``.
    - ``raise_during`` replaces stream iteration with a raise.
    """

    def __init__(
        self,
        events: Iterable[Any] | None = None,
        final: _Response | None = None,
        raise_during: BaseException | None = None,
        raise_on_final: BaseException | None = None,
    ) -> None:
        self._events = list(events or [])
        self._final = final
        self._raise_during = raise_during
        self._raise_on_final = raise_on_final

    def __enter__(self) -> _StubStream:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False  # propagate

    def __iter__(self) -> Iterator[Any]:
        for e in self._events:
            yield e
        # After yielding all scripted events, simulate the failure mid-stream.
        if self._raise_during is not None:
            raise self._raise_during

    def get_final_message(self) -> _Response:
        if self._raise_on_final is not None:
            raise self._raise_on_final
        if self._final is None:
            # Reasonable default so callers can ignore.
            return _Response([_TextBlock("")], usage=_Usage())
        return self._final


def _make_client(stream: _StubStream) -> MagicMock:
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=stream)
    return client


# =========================================================================
# WS-R1 — Streaming error resilience
# =========================================================================


class TestStreamingResilience:
    def test_api_error_mid_stream_yields_error_event(self) -> None:
        """An APIError-like exception during streaming emits a code=carl.stream_error event."""

        class FakeAPIError(RuntimeError):
            pass

        stream = _StubStream(
            events=[_Event("hello ")],
            raise_during=FakeAPIError("upstream 500"),
            final=_Response([_TextBlock("hello ")]),
        )
        agent = CARLAgent(model="test-model", _client=_make_client(stream))

        events = list(agent.chat("hi"))
        assert any(e.kind == "text_delta" and e.content == "hello " for e in events)
        errors = [e for e in events if e.kind == "error"]
        assert errors, "expected an error event"
        assert errors[0].code in ("carl.stream_error", "carl.interrupted")

    def test_connection_error_preserves_partial_text_in_messages(self) -> None:
        stream = _StubStream(
            events=[_Event("hello ")],
            raise_during=ConnectionError("link dropped"),
            final=_Response([_TextBlock("hello ")]),
        )
        agent = CARLAgent(model="test-model", _client=_make_client(stream))

        list(agent.chat("hi"))
        # The assistant side of the turn should have something persisted.
        assistants = [m for m in agent._messages if m.get("role") == "assistant"]
        assert assistants, "expected assistant turn to be persisted"
        # content is list of dicts — text parts with "hello "
        content = assistants[-1]["content"]
        assert isinstance(content, list)
        text_blobs = [b.get("text", "") for b in content if isinstance(b, dict)]
        assert any("hello" in t for t in text_blobs)

    def test_keyboard_interrupt_mid_stream_completes_generator(self) -> None:
        stream = _StubStream(
            events=[_Event("pa")],
            raise_during=KeyboardInterrupt(),
            final=_Response([_TextBlock("pa")]),
        )
        agent = CARLAgent(model="test-model", _client=_make_client(stream))

        events = list(agent.chat("hi"))  # generator must complete
        errors = [e for e in events if e.kind == "error"]
        assert errors
        assert errors[0].code == "carl.interrupted"

    def test_generator_never_propagates_exception(self) -> None:
        """Even when the stream blows up entering the CM, .chat() does not raise."""

        bad_client = MagicMock()
        bad_client.messages.stream.side_effect = RuntimeError("cannot open stream")
        agent = CARLAgent(model="test-model", _client=bad_client)

        events = list(agent.chat("hi"))  # must not raise
        errors = [e for e in events if e.kind == "error"]
        assert errors
        # code should be carl.api_error or stream_error
        assert errors[0].code in ("carl.api_error", "carl.stream_error")

    def test_partial_cost_attributed_on_stream_failure(self) -> None:
        usage = _Usage(input_tokens=200, output_tokens=50)
        stream = _StubStream(
            events=[_Event("hi ")],
            raise_during=RuntimeError("boom"),
            final=_Response([_TextBlock("hi ")], usage=usage),
        )
        agent = CARLAgent(model="claude-sonnet-4-6", _client=_make_client(stream))
        list(agent.chat("start"))
        # 200 * 3/1M + 50 * 15/1M
        assert agent._total_cost_usd > 0
        assert agent._total_input_tokens >= 200


# =========================================================================
# WS-R2 — Session corruption handling
# =========================================================================


class TestSessionCorruption:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> SessionStore:
        return SessionStore(sessions_dir=tmp_path / "sessions")

    def test_corrupted_json_returns_none_and_quarantines(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        sid = "broken"
        path = (tmp_path / "sessions") / f"{sid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not: valid json}")

        assert store.load(sid) is None
        # Original file should no longer exist; a quarantine copy should.
        assert not path.is_file()
        quarantine_dir = tmp_path / "sessions" / ".quarantine"
        assert quarantine_dir.is_dir()
        quarantined = list(quarantine_dir.glob(f"{sid}-*.json"))
        assert quarantined, "expected quarantined file"

    def test_schema_mismatch_quarantines(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        sid = "mismatch"
        path = (tmp_path / "sessions") / f"{sid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": sid, "schema_version": "99"}))

        assert store.load(sid) is None
        quarantine_dir = tmp_path / "sessions" / ".quarantine"
        assert list(quarantine_dir.glob(f"{sid}-*.json"))

    def test_legacy_sessions_without_schema_load(
        self, store: SessionStore
    ) -> None:
        store.save("legacy", {"id": "legacy", "messages": []})
        # Manually strip schema_version to simulate older file.
        path = store._dir / "legacy.json"
        data = json.loads(path.read_text())
        data.pop("schema_version", None)
        path.write_text(json.dumps(data))

        loaded = store.load("legacy")
        # Legacy (unstamped) sessions are accepted.
        assert loaded is not None

    def test_save_stamps_schema_version(self, store: SessionStore) -> None:
        store.save("v", {"id": "v"})
        raw = json.loads((store._dir / "v.json").read_text())
        assert raw["schema_version"] == _SESSION_SCHEMA_VERSION

    def test_list_excludes_quarantined_and_survives_bad_entries(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        # 1 good session, 1 corrupt session at the top level, 1 quarantined.
        store.save("good", {"id": "good", "title": "keep"})

        bad_path = (tmp_path / "sessions") / "bad.json"
        bad_path.write_text("garbage")

        quarantine_dir = tmp_path / "sessions" / ".quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        (quarantine_dir / "something-20260101T000000Z.json").write_text("{}")

        listing = store.list_sessions(limit=20)
        ids = {s["id"] for s in listing}
        assert "good" in ids
        assert "something-20260101T000000Z" not in ids  # not reached by glob
        # bad.json gets skipped silently during list_sessions, not quarantined.
        # Either it's missing (quarantined later) or simply absent from listing.

    def test_load_nonexistent_returns_none(self, store: SessionStore) -> None:
        assert store.load("nope") is None


# =========================================================================
# WS-R3 — Budget pre-check + hard cap
# =========================================================================


class TestBudgetPreCheck:
    def test_first_turn_rejected_when_budget_cents(self) -> None:
        """A sub-penny budget triggers the pre-check before any API call."""
        client = MagicMock()
        agent = CARLAgent(
            model="claude-opus-4-6",
            max_budget_usd=0.001,
            _client=client,
        )
        events = list(agent.chat("start"))
        errs = [e for e in events if e.kind == "error"]
        assert errs
        assert errs[0].code == "carl.budget"
        assert client.messages.stream.call_count == 0

    def test_budget_upper_bound_estimator_is_nonzero(self) -> None:
        est = _estimate_turn_upper_bound_cost("claude-sonnet-4-6", 64000)
        assert est > 0

    def test_mid_loop_abort_when_accumulated_cost_exceeds_cap(self) -> None:
        """If total cost already >= cap (hard floor), no new turn is issued."""
        client = MagicMock()
        agent = CARLAgent(
            model="claude-haiku-4-5",
            max_budget_usd=5.0,
            _client=client,
        )
        # Pretend we already spent more than the cap.
        agent._total_cost_usd = 5.5
        events = list(agent.chat("go"))
        errs = [e for e in events if e.kind == "error"]
        assert errs
        assert errs[0].code == "carl.budget"
        assert client.messages.stream.call_count == 0

    def test_generous_budget_lets_turn_happen(self) -> None:
        """A $100 budget doesn't trip the pre-check even with Opus pricing."""
        stream = _StubStream(
            events=[_Event("ok")],
            final=_Response([_TextBlock("ok")]),
        )
        agent = CARLAgent(
            model="claude-haiku-4-5",
            max_budget_usd=100.0,
            _client=_make_client(stream),
        )
        events = list(agent.chat("hi"))
        # Turn should complete.
        assert any(e.kind == "done" for e in events)


# =========================================================================
# WS-R4 — Tool timeout + cancellation, hook isolation
# =========================================================================


class TestToolTimeout:
    def test_run_analysis_timeout_returns_is_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If subprocess.run raises TimeoutExpired, dispatch surfaces a tool_result error."""
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )

        import subprocess as sp

        def _raise_timeout(*args: Any, **kwargs: Any) -> Any:
            raise sp.TimeoutExpired(cmd="python", timeout=60)

        monkeypatch.setattr("subprocess.run", _raise_timeout)

        result, is_error = agent._dispatch_tool_safe(
            "run_analysis", {"code": "while True: pass"}
        )
        assert is_error
        assert "timed out" in result.lower() or "timeout" in result.lower()

    def test_tool_timeout_wrapper_kills_hang(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hanging pure-Python tool returns an is_error result via ThreadPool timeout."""
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )

        # Force the default timeout down to 0.2s so the test is quick.
        monkeypatch.setattr(
            "carl_studio.chat_agent._TOOL_TIMEOUTS_S",
            {"query_knowledge": 0.2},
        )

        def _hang(_q: str) -> str:
            import time

            time.sleep(5)
            return "never"

        agent._tool_query = _hang  # type: ignore[assignment]
        result, is_error = agent._dispatch_tool_safe(
            "query_knowledge", {"question": "hang please"}
        )
        assert is_error
        assert "timed out" in result.lower()

    def test_ingest_timeout_surfaces_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )
        monkeypatch.setattr(
            "carl_studio.chat_agent._TOOL_TIMEOUTS_S",
            {"ingest_source": 0.2},
        )

        def _hang(_p: str) -> str:
            import time
            time.sleep(5)
            return "never"

        agent._tool_ingest = _hang  # type: ignore[assignment]
        result, is_error = agent._dispatch_tool_safe("ingest_source", {"path": "."})
        assert is_error
        assert "timed out" in result.lower()

    def test_pre_hook_exception_does_not_kill_loop(self) -> None:
        """A raising pre_tool_use yields a carl.hook_failed event, agent continues."""

        def bad_pre(name: str, args: dict[str, Any]) -> ToolPermission:
            raise RuntimeError("pre-hook crashed")

        stream = _StubStream(
            events=[],
            final=_Response(
                [
                    _TextBlock("using tool now"),
                    _ToolUseBlock("query_knowledge", {"question": "x"}, "tu_1"),
                ],
                stop_reason="end_turn",
            ),
        )
        agent = CARLAgent(
            model="test",
            pre_tool_use=bad_pre,
            _client=_make_client(stream),
        )

        events = list(agent.chat("hello"))
        errors = [e for e in events if e.kind == "error" and e.code == "carl.hook_failed"]
        # Generator must complete AND yield the hook_failed error event.
        assert errors, "expected hook_failed error"
        assert any(e.kind == "done" for e in events)

    def test_post_hook_exception_does_not_kill_loop(self) -> None:
        def bad_post(name: str, args: dict[str, Any], result: str) -> None:
            raise RuntimeError("post-hook crashed")

        stream = _StubStream(
            events=[],
            final=_Response(
                [
                    _ToolUseBlock("query_knowledge", {"question": "x"}, "tu_1"),
                ],
                stop_reason="end_turn",
            ),
        )
        agent = CARLAgent(
            model="test",
            post_tool_use=bad_post,
            _client=_make_client(stream),
        )

        events = list(agent.chat("hello"))
        errors = [e for e in events if e.kind == "error" and e.code == "carl.hook_failed"]
        assert errors


# =========================================================================
# WS-R5 — Tool arg schema validation
# =========================================================================


class TestToolArgValidation:
    def test_missing_required_is_rejected(self) -> None:
        # create_file requires path + content
        err = _validate_tool_args("create_file", {"path": "x.txt"})
        assert err is not None
        assert "content" in err

    def test_type_mismatch_rejected(self) -> None:
        # run_analysis code must be a string
        err = _validate_tool_args("run_analysis", {"code": 42})
        assert err is not None
        assert "code" in err

    def test_well_formed_args_pass(self) -> None:
        err = _validate_tool_args(
            "create_file", {"path": "x.txt", "content": "hi"}
        )
        assert err is None

    def test_unknown_tool_falls_through_to_dispatch(self) -> None:
        err = _validate_tool_args("not_a_tool", {})
        assert err is None  # downstream dispatch returns "Unknown tool"

    def test_missing_required_args_surfaces_tool_result_error(self) -> None:
        """End-to-end: a tool_use block with bad args produces is_error tool_result."""
        stream = _StubStream(
            events=[],
            final=_Response(
                [
                    _ToolUseBlock("create_file", {"path": "x.txt"}, "tu_1"),
                ],
                stop_reason="end_turn",
            ),
        )
        agent = CARLAgent(model="test", _client=_make_client(stream))

        events = list(agent.chat("please"))
        tool_results = [e for e in events if e.kind == "tool_result"]
        assert tool_results
        assert "content" in tool_results[0].content.lower()
        # is_error is encoded in the API-bound message history
        last_user = agent._messages[-1]
        assert last_user["role"] == "user"
        api_results = last_user["content"]
        assert api_results[0].get("is_error") is True


# =========================================================================
# WS-R6 — dispatch_cli agent tool
# =========================================================================


class TestDispatchCLI:
    @pytest.fixture()
    def agent(self, tmp_path: Path) -> CARLAgent:
        return CARLAgent(model="test", workdir=str(tmp_path), _client=MagicMock())

    def test_allowed_op_dispatches(
        self, agent: CARLAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A registered op builds ['carl', cmd, *args] and surfaces stdout/exit."""
        captured: dict[str, Any] = {}

        class FakeProc:
            def __init__(self) -> None:
                self.stdout = "OK\n"
                self.stderr = ""
                self.returncode = 0

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeProc:
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return FakeProc()

        monkeypatch.setattr("subprocess.run", fake_run)

        payload, is_error = agent._dispatch_tool_safe(
            "dispatch_cli",
            {"command": "doctor", "args": ["--json"]},
        )
        assert not is_error
        parsed = json.loads(payload)
        assert parsed["exit_code"] == 0
        assert "OK" in parsed["stdout"]
        assert captured["cmd"] == ["carl", "doctor", "--json"]
        # subprocess.run must NOT be invoked with shell=True.
        assert captured["kwargs"].get("shell") is not True

    def test_unknown_op_returns_error(self, agent: CARLAgent) -> None:
        payload, is_error = agent._dispatch_tool_safe(
            "dispatch_cli",
            {"command": "definitely-not-a-command", "args": []},
        )
        assert is_error
        parsed = json.loads(payload)
        assert parsed.get("code") == "carl.validation"
        assert "unknown" in parsed["error"].lower()

    @pytest.mark.parametrize(
        "bad_arg",
        ["foo; rm -rf /", "a | b", "a && b", "a > b", "a < b", "`whoami`", "$(id)"],
    )
    def test_shell_metacharacter_in_args_rejected(
        self, agent: CARLAgent, bad_arg: str
    ) -> None:
        payload, is_error = agent._dispatch_tool_safe(
            "dispatch_cli",
            {"command": "doctor", "args": [bad_arg]},
        )
        assert is_error
        parsed = json.loads(payload)
        assert "metacharacter" in parsed["error"].lower() or "forbidden" in parsed["error"].lower()

    def test_timeout_returns_error(
        self, agent: CARLAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as sp

        def _raise_timeout(*args: Any, **kwargs: Any) -> Any:
            raise sp.TimeoutExpired(cmd="carl doctor", timeout=1)

        monkeypatch.setattr("subprocess.run", _raise_timeout)

        payload, is_error = agent._dispatch_tool_safe(
            "dispatch_cli",
            {"command": "doctor", "args": [], "timeout_s": 1},
        )
        assert is_error
        parsed = json.loads(payload)
        assert "timed out" in parsed["error"].lower() or "timeout" in parsed["error"].lower()

    def test_exit_code_propagated(
        self, agent: CARLAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProc:
            def __init__(self) -> None:
                self.stdout = "fail"
                self.stderr = "bad"
                self.returncode = 2

        monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeProc())

        payload, is_error = agent._dispatch_tool_safe(
            "dispatch_cli",
            {"command": "doctor", "args": []},
        )
        assert is_error  # non-zero exit is treated as an error for the model
        parsed = json.loads(payload)
        assert parsed["exit_code"] == 2

    def test_args_type_validation(self, agent: CARLAgent) -> None:
        """Non-string elements in args are rejected."""
        # jsonschema validation runs *before* dispatch — go through the
        # direct dispatcher to test our own type check.
        payload, is_error = agent._tool_dispatch_cli({
            "command": "doctor",
            "args": ["ok", 123],  # second element is int
        })
        assert is_error
        parsed = json.loads(payload)
        assert "args[1]" in parsed["error"]

    def test_dispatch_cli_missing_binary(
        self, agent: CARLAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_fnf(*args: Any, **kwargs: Any) -> Any:
            raise FileNotFoundError("no carl")

        monkeypatch.setattr("subprocess.run", _raise_fnf)
        payload, is_error = agent._dispatch_tool_safe(
            "dispatch_cli",
            {"command": "doctor", "args": []},
        )
        assert is_error
        parsed = json.loads(payload)
        assert "not found" in parsed["error"].lower()

    def test_dispatch_cli_chain_step(
        self, agent: CARLAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successful dispatch appends a CLI_CMD step to the interaction chain."""
        class FakeProc:
            def __init__(self) -> None:
                self.stdout = "ok"
                self.stderr = ""
                self.returncode = 0

        monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeProc())

        agent._dispatch_tool_safe(
            "dispatch_cli",
            {"command": "doctor", "args": []},
        )
        chain = agent._get_chain()
        assert chain is not None
        assert len(chain.steps) >= 1
        from carl_core.interaction import ActionType

        assert any(s.action == ActionType.CLI_CMD for s in chain.steps)

    def test_dispatch_cli_schema_in_tools_registry(self) -> None:
        """The dispatch_cli tool is exported in TOOLS with the right schema."""
        names = {t["name"] for t in TOOLS}
        assert "dispatch_cli" in names
        tool = next(t for t in TOOLS if t["name"] == "dispatch_cli")
        props = tool["input_schema"]["properties"]
        assert "command" in props and props["command"]["type"] == "string"
        assert "args" in props and props["args"]["type"] == "array"
        assert "timeout_s" in props and props["timeout_s"]["type"] == "integer"
        assert "command" in tool["input_schema"]["required"]


# =========================================================================
# Smoke: public API survives
# =========================================================================


class TestPublicAPIPreserved:
    def test_agent_constructs_with_existing_signature(self) -> None:
        agent = CARLAgent(
            api_key="",
            model="test",
            frame=None,
            workdir=".",
            max_budget_usd=0.0,
            pre_tool_use=None,
            post_tool_use=None,
            session_id="",
            _client=MagicMock(),
        )
        assert agent.provider_info["model"] == "test"
        assert agent.cost_summary["turn_count"] == 0

    def test_save_load_roundtrip_with_schema(self, tmp_path: Path) -> None:
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        agent = CARLAgent(model="test", _client=MagicMock())
        agent._messages = [{"role": "user", "content": "hi"}]
        sid = agent.save_session(title="t", store=store)

        loaded = store.load(sid)
        assert loaded is not None
        assert loaded["schema_version"] == _SESSION_SCHEMA_VERSION
        assert loaded["messages"][0]["content"] == "hi"
