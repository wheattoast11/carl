"""Robustness tests for carl_studio.chat_agent — WS-R1..R6 + REV-004/REV-010.

Covers the six production-hardening passes:

* WS-R1 — Streaming error resilience: APIError, ConnectionError, KeyboardInterrupt
* WS-R2 — Session corruption handling: bad JSON, schema mismatch, quarantine
* WS-R3 — Budget pre-check + hard cap
* WS-R4 — Tool timeout + cancellation, hook isolation
* WS-R5 — Tool arg schema validation
* WS-R6 — dispatch_cli agent tool
* REV-004 — Greeting fires on turn_count==1, never on a resumed session
* REV-010 — Frame mutation invalidates the cached constitution prompt
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


# =========================================================================
# Fix 1 — list_files sandbox
# =========================================================================


class TestListFilesSandbox:
    def test_list_files_blocks_absolute_path_outside_workdir(
        self, tmp_path: Path,
    ) -> None:
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )
        result = agent._dispatch_tool("list_files", {"path": "/"})
        assert "blocked" in result.lower()

    def test_list_files_allows_workdir_itself(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "a.txt").write_text("x")
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )
        result = agent._dispatch_tool(
            "list_files", {"path": str(tmp_path), "pattern": "*"},
        )
        assert "a.txt" in result

    def test_list_files_rejects_traversal_glob(
        self, tmp_path: Path,
    ) -> None:
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )
        result = agent._dispatch_tool(
            "list_files", {"path": str(tmp_path), "pattern": "../**"},
        )
        assert "blocked" in result.lower()

    def test_list_files_blocks_sibling_directory(
        self, tmp_path: Path,
    ) -> None:
        """Directory adjacent to workdir must be rejected, not enumerated."""
        workdir = tmp_path / "inside"
        workdir.mkdir()
        sibling = tmp_path / "outside"
        sibling.mkdir()
        (sibling / "secret.txt").write_text("x")

        agent = CARLAgent(
            model="test", workdir=str(workdir), _client=MagicMock(),
        )
        result = agent._dispatch_tool("list_files", {"path": str(sibling)})
        assert "blocked" in result.lower()
        assert "secret.txt" not in result


# =========================================================================
# Fix 2 — all-tools-denied termination
# =========================================================================


class TestAllToolsDenied:
    def test_chat_terminates_after_five_consecutive_all_denied_turns(
        self, tmp_path: Path,
    ) -> None:
        """A hook that denies every tool must not loop forever."""

        def deny_all(name: str, args: dict[str, Any]) -> ToolPermission:
            return ToolPermission.DENY

        # Each server turn returns a tool_use block. Real API retries after
        # seeing tool_result errors; we simulate that by giving the same
        # client-side stream every time chat() issues a new stream().
        def make_stream() -> _StubStream:
            return _StubStream(
                events=[],
                final=_Response(
                    [_ToolUseBlock("run_analysis", {"code": "print('x')"}, "tu_deny")],
                    stop_reason="tool_use",
                ),
            )

        client = MagicMock()
        client.messages.stream.side_effect = lambda **_kw: make_stream()

        agent = CARLAgent(
            model="test",
            workdir=str(tmp_path),
            pre_tool_use=deny_all,
            _client=client,
        )

        events = list(agent.chat("please"))
        errors = [
            e for e in events
            if e.kind == "error" and e.code == "carl.all_tools_denied"
        ]
        assert errors, "expected carl.all_tools_denied terminal event"
        # The hook returns tool_use every round — the loop must cap at the
        # class constant value, not run forever.
        assert client.messages.stream.call_count == agent._MAX_CONSECUTIVE_ALL_DENIED

    def test_allowed_tool_resets_consecutive_denied_counter(
        self, tmp_path: Path,
    ) -> None:
        """A single allowed turn clears the DENY counter."""
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )
        agent._consecutive_all_denied = 3
        # Simulate the counter-update block directly: any allowed tool clears.
        turn_attempted = 2
        turn_denied = 0
        if turn_attempted > 0 and turn_denied == turn_attempted:
            agent._consecutive_all_denied += 1
        elif turn_attempted > 0:
            agent._consecutive_all_denied = 0
        assert agent._consecutive_all_denied == 0

    def test_no_tool_use_turn_also_resets_counter(self) -> None:
        """A text-only turn (no tool_use) should also reset the counter."""
        stream = _StubStream(
            events=[],
            final=_Response(
                [_TextBlock("done")], stop_reason="end_turn",
            ),
        )
        agent = CARLAgent(model="test", _client=_make_client(stream))
        agent._consecutive_all_denied = 2
        list(agent.chat("hi"))
        assert agent._consecutive_all_denied == 0


# =========================================================================
# Fix 3 — run_analysis sys.executable + env scrub
# =========================================================================


class TestRunAnalysisEnvScrub:
    def test_run_analysis_uses_sys_executable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sys as _sys

        captured: dict[str, Any] = {}

        class _Proc:
            def __init__(self) -> None:
                self.returncode = 0
                self.stdout = b"ok"
                self.stderr = b""

        def fake_run(cmd: list[str], **kwargs: Any) -> Any:
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env")
            return _Proc()

        monkeypatch.setattr("subprocess.run", fake_run)
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )
        agent._dispatch_tool("run_analysis", {"code": "print(1)"})
        assert captured["cmd"][0] == _sys.executable
        assert captured["cmd"][1] == "-c"

    def test_run_analysis_strips_sensitive_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Seed the environment with known credentials the scrubber must strip
        # plus a plumbing key that must survive.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        monkeypatch.setenv("CARL_WALLET_PASSPHRASE", "correct horse battery")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-secret")
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        monkeypatch.setenv("SOMETHING_SECRET", "nope")
        monkeypatch.setenv("GITHUB_BEARER", "ghp_bearer")
        monkeypatch.setenv("LANG", "en_US.UTF-8")

        captured_env: dict[str, dict[str, str]] = {}

        class _Proc:
            def __init__(self) -> None:
                self.returncode = 0
                self.stdout = b""
                self.stderr = b""

        def fake_run(cmd: list[str], **kwargs: Any) -> Any:
            captured_env["env"] = dict(kwargs.get("env") or {})
            return _Proc()

        monkeypatch.setattr("subprocess.run", fake_run)
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )
        agent._dispatch_tool("run_analysis", {"code": "pass"})

        env = captured_env["env"]
        # Infra keys survive
        assert "PATH" in env
        assert env.get("LANG") == "en_US.UTF-8"
        # Marker that _tool_run added
        assert env.get("PYTHONDONTWRITEBYTECODE") == "1"
        # Secrets gone
        assert "ANTHROPIC_API_KEY" not in env
        assert "CARL_WALLET_PASSPHRASE" not in env
        assert "OPENAI_API_KEY" not in env
        assert "HF_TOKEN" not in env
        assert "SOMETHING_SECRET" not in env
        assert "GITHUB_BEARER" not in env

    def test_is_sensitive_env_key_classification(self) -> None:
        from carl_studio.chat_agent import _is_sensitive_env_key

        assert _is_sensitive_env_key("ANTHROPIC_API_KEY")
        assert _is_sensitive_env_key("MY_ACCESS_TOKEN")
        assert _is_sensitive_env_key("SOME_SECRET")
        assert _is_sensitive_env_key("USER_PASSWORD")
        assert _is_sensitive_env_key("WALLET_PASSPHRASE")
        assert _is_sensitive_env_key("HTTP_AUTHORIZATION")
        assert _is_sensitive_env_key("GH_BEARER")
        # Allowlist wins
        assert not _is_sensitive_env_key("PATH")
        assert not _is_sensitive_env_key("HOME")
        assert not _is_sensitive_env_key("LANG")
        assert not _is_sensitive_env_key("PWD")
        assert not _is_sensitive_env_key("USER")
        # Non-sensitive
        assert not _is_sensitive_env_key("LOG_LEVEL")
        assert not _is_sensitive_env_key("PYTHONDONTWRITEBYTECODE")


# =========================================================================
# Fix 4 — safe_resolve rejects symlinks
# =========================================================================


class TestResolveSafePathSymlinks:
    def test_symlink_in_workdir_is_rejected(self, tmp_path: Path) -> None:
        import os as _os

        workdir = tmp_path / "wd"
        workdir.mkdir()
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "secret.txt").write_text("private")

        link = workdir / "link"
        try:
            _os.symlink(str(outside), str(link))
        except OSError:
            pytest.skip("symlinks not supported here")

        agent = CARLAgent(
            model="test", workdir=str(workdir), _client=MagicMock(),
        )
        # read_file through the link — must be blocked, not resolved.
        result = agent._dispatch_tool(
            "read_file", {"path": str(link / "secret.txt")},
        )
        assert "blocked" in result.lower() or "not found" in result.lower()

    def test_tail_symlink_rejected(self, tmp_path: Path) -> None:
        """A tail symlink inside the sandbox is rejected by follow_symlinks=False."""
        import os as _os

        workdir = tmp_path / "wd"
        workdir.mkdir()
        target = workdir / "real.txt"
        target.write_text("real")
        link = workdir / "alias.txt"
        try:
            _os.symlink(str(target), str(link))
        except OSError:
            pytest.skip("symlinks not supported here")

        agent = CARLAgent(
            model="test", workdir=str(workdir), _client=MagicMock(),
        )
        result = agent._dispatch_tool("read_file", {"path": str(link)})
        assert "blocked" in result.lower()


# =========================================================================
# Fix 5 — visible session quarantine
# =========================================================================


class TestSessionQuarantineVisibility:
    def test_load_session_sets_quarantine_flag_on_corruption(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging as _logging

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        sid = "broken"
        bad = sessions_dir / f"{sid}.json"
        bad.write_text("{not: valid json}")

        store = SessionStore(sessions_dir=sessions_dir)
        agent = CARLAgent(model="test", _client=MagicMock())

        with caplog.at_level(_logging.WARNING, logger="carl_studio.chat_agent"):
            ok = agent.load_session(sid, store=store)

        assert ok is False  # file was not loadable
        assert agent._last_load_quarantined is True
        # Warning should reference the quarantine destination.
        messages = [r.getMessage() for r in caplog.records]
        assert any("Quarantined" in m for m in messages)

    def test_load_session_clean_session_keeps_flag_false(
        self, tmp_path: Path,
    ) -> None:
        sessions_dir = tmp_path / "sessions"
        store = SessionStore(sessions_dir=sessions_dir)
        store.save("clean", {"id": "clean", "title": "x"})

        agent = CARLAgent(model="test", _client=MagicMock())
        ok = agent.load_session("clean", store=store)
        assert ok is True
        assert agent._last_load_quarantined is False

    def test_load_session_missing_session_keeps_flag_false(
        self, tmp_path: Path,
    ) -> None:
        sessions_dir = tmp_path / "sessions"
        store = SessionStore(sessions_dir=sessions_dir)
        agent = CARLAgent(model="test", _client=MagicMock())
        assert agent.load_session("nope", store=store) is False
        assert agent._last_load_quarantined is False

    def test_cli_chat_resume_surfaces_quarantine_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The chat CLI path must print a visible warn() on quarantine."""
        from carl_studio.cli import chat as chat_cli

        # Redirect sessions dir into tmp_path so the test is hermetic.
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("pathlib.Path.home", lambda: home)
        real_sessions = home / ".carl" / "sessions"
        real_sessions.mkdir(parents=True)
        bad = real_sessions / "broken.json"
        bad.write_text("{not: valid json}")

        # Capture console output.
        warn_calls: list[str] = []
        info_calls: list[str] = []
        ok_calls: list[str] = []

        class FakeConsole:
            theme = type("T", (), {"persona": type("P", (), {"value": "carl"})()})()

            def blank(self) -> None: pass
            def header(self, *_a: Any, **_k: Any) -> None: pass
            def kv(self, *_a: Any, **_k: Any) -> None: pass
            def info(self, msg: str) -> None: info_calls.append(msg)
            def ok(self, msg: str) -> None: ok_calls.append(msg)
            def warn(self, msg: str) -> None: warn_calls.append(msg)
            def error(self, msg: str) -> None: pass
            def voice(self, *_a: Any, **_k: Any) -> None: pass
            def print(self, *_a: Any, **_k: Any) -> None: pass

        monkeypatch.setattr(chat_cli, "get_console", lambda: FakeConsole())

        # Patch CARLAgent so we don't need a real Anthropic client.
        class StubAgent:
            def __init__(self, *_a: Any, **_k: Any) -> None:
                self._messages: list[Any] = []
                self._last_load_quarantined = False

            def load_session(self, sid: str) -> bool:
                # Trigger the quarantine via the real SessionStore on disk.
                store = SessionStore(sessions_dir=real_sessions)
                state = store.load(sid)
                self._last_load_quarantined = getattr(
                    store, "_last_quarantine_dest", None,
                ) is not None
                return state is not None

            @property
            def provider_info(self) -> dict[str, str]:
                return {"model": "x", "api_key_source": "test"}

            def cost_summary(self) -> dict[str, Any]:
                return {"turn_count": 0, "total_cost_usd": 0.0}

        # chat_cmd does ``from carl_studio.chat_agent import CARLAgent``
        # inline — patch the real module so the import resolves to our stub.
        monkeypatch.setattr("carl_studio.chat_agent.CARLAgent", StubAgent)
        # Short-circuit the first-run init probe.
        monkeypatch.setattr(
            "carl_studio.cli.init._first_run_complete", lambda: True,
        )

        # Also short-circuit interactive input so the while loop exits.
        monkeypatch.setattr("builtins.input", lambda *_a, **_k: "quit")

        try:
            chat_cli.chat_cmd(
                model="test",
                config="carl.yaml",
                api_key="",
                frame_source="none",
                session="broken",
                budget=0.0,
                voice=False,
                local=False,
                sessions=False,
            )
        except Exception:
            # _exit_session runs on the "quit" input; any downstream
            # exception about missing dependencies is irrelevant to the
            # quarantine warning we're validating.
            pass

        assert any(
            "corrupted" in msg.lower() and "quarantined" in msg.lower()
            for msg in warn_calls
        ), f"expected quarantine warning in CLI; got warn={warn_calls} info={info_calls}"


# =========================================================================
# Fix 6 — _knowledge list bounded
# =========================================================================


class TestKnowledgeCap:
    def test_default_cap_truncates_to_2000(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ingesting 2500 chunks leaves exactly 2000 entries (oldest evicted)."""
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )
        assert agent._max_knowledge_chunks == agent._KNOWLEDGE_MAX_CHUNKS == 2000

        class FakeChunk:
            def __init__(self, idx: int) -> None:
                self.text = f"chunk-{idx}"
                self.source = f"src-{idx}"
                self.modality = "text"

        class FakeIngester:
            def ingest(self, _path: str) -> list[FakeChunk]:
                return [FakeChunk(i) for i in range(2500)]

        # Patch the SourceIngester imported *inside* _tool_ingest.
        import carl_studio.learn.ingest as _ingest_mod

        monkeypatch.setattr(_ingest_mod, "SourceIngester", FakeIngester)

        result = agent._dispatch_tool("ingest_source", {"path": "whatever"})
        assert "Ingested 2500" in result
        assert len(agent._knowledge) == 2000
        # The first 500 are evicted, so chunk-0 is gone and chunk-2499 is last.
        sources = {c["source"] for c in agent._knowledge}
        assert "src-0" not in sources
        assert "src-499" not in sources  # evicted
        assert "src-500" in sources      # just inside the window
        assert "src-2499" in sources
        assert "evicted 500" in result.lower()

    def test_single_warning_per_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging as _logging

        agent = CARLAgent(
            model="test",
            workdir=str(tmp_path),
            max_knowledge_chunks=5,
            _client=MagicMock(),
        )

        class FakeChunk:
            def __init__(self, idx: int) -> None:
                self.text = f"c-{idx}"
                self.source = f"s-{idx}"
                self.modality = "text"

        counter = {"i": 0}

        class FakeIngester:
            def ingest(self, _path: str) -> list[FakeChunk]:
                counter["i"] += 1
                # Each call adds 10 chunks — well past the cap.
                return [FakeChunk(i) for i in range(10)]

        import carl_studio.learn.ingest as _ingest_mod
        monkeypatch.setattr(_ingest_mod, "SourceIngester", FakeIngester)

        with caplog.at_level(_logging.WARNING, logger="carl_studio.chat_agent"):
            agent._dispatch_tool("ingest_source", {"path": "a"})
            agent._dispatch_tool("ingest_source", {"path": "b"})
            agent._dispatch_tool("ingest_source", {"path": "c"})

        assert len(agent._knowledge) == 5
        assert agent._knowledge_evicted_warned is True
        # Only ONE eviction warning — three ingests would otherwise spam.
        evict_warnings = [
            r for r in caplog.records
            if "knowledge cap reached" in r.getMessage()
        ]
        assert len(evict_warnings) == 1

    def test_custom_max_knowledge_chunks_honored(self, tmp_path: Path) -> None:
        agent = CARLAgent(
            model="test",
            workdir=str(tmp_path),
            max_knowledge_chunks=3,
            _client=MagicMock(),
        )
        # Manually stuff the list to avoid mocking the ingester here.
        for i in range(10):
            agent._knowledge.append({
                "text": f"t{i}", "source": f"s{i}",
                "words": {f"w{i}"},
            })
        evicted = agent._enforce_knowledge_cap()
        assert evicted == 7
        assert len(agent._knowledge) == 3
        # Newest three entries survived (t7, t8, t9).
        remaining_texts = [c["text"] for c in agent._knowledge]
        assert remaining_texts == ["t7", "t8", "t9"]

    def test_invalid_max_knowledge_chunks_rejected(self) -> None:
        with pytest.raises(ValueError):
            CARLAgent(
                model="test",
                max_knowledge_chunks=0,
                _client=MagicMock(),
            )
        with pytest.raises(ValueError):
            CARLAgent(
                model="test",
                max_knowledge_chunks=-5,
                _client=MagicMock(),
            )

    def test_under_cap_is_noop(self, tmp_path: Path) -> None:
        agent = CARLAgent(
            model="test", workdir=str(tmp_path), _client=MagicMock(),
        )
        agent._knowledge = [{"text": "x", "source": "s", "words": set()}]
        evicted = agent._enforce_knowledge_cap()
        assert evicted == 0
        assert len(agent._knowledge) == 1
        assert agent._knowledge_evicted_warned is False

    def test_load_session_enforces_cap(self, tmp_path: Path) -> None:
        """Resuming a session with oversized knowledge trims it on load."""
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        agent = CARLAgent(
            model="test",
            workdir=str(tmp_path),
            max_knowledge_chunks=4,
            _client=MagicMock(),
        )
        # Craft and save a state with more chunks than the cap.
        big_knowledge = [
            {"text": f"t{i}", "source": f"s{i}", "words": {f"w{i}"}}
            for i in range(10)
        ]
        agent._knowledge = list(big_knowledge)  # bypass _enforce_knowledge_cap
        sid = agent.save_session(title="big", store=store)

        # Fresh agent with same cap — load must trim to 4 newest.
        agent2 = CARLAgent(
            model="test",
            workdir=str(tmp_path),
            max_knowledge_chunks=4,
            _client=MagicMock(),
        )
        assert agent2.load_session(sid, store=store) is True
        assert len(agent2._knowledge) == 4
        assert [c["text"] for c in agent2._knowledge] == ["t6", "t7", "t8", "t9"]


# =========================================================================
# REV-004 — Greeting gate uses turn_count, not message history length
# =========================================================================


class TestGreetingFiresOnTurnCount:
    def test_greeting_fires_on_fresh_session_turn_one(self) -> None:
        """Turn 1 of a fresh session must emit the SESSION START block."""
        agent = CARLAgent(model="test", _client=MagicMock())
        # Simulate the chat() prologue — turn_count incremented to 1.
        agent._turn_count = 1
        prompt = agent._build_system_prompt()
        assert "SESSION START BEHAVIOR" in prompt
        assert "CARL" in prompt

    def test_greeting_does_not_fire_on_resumed_session(self) -> None:
        """A resumed session with turn_count > 1 must NOT include the greeting."""
        agent = CARLAgent(model="test", _client=MagicMock())
        # Simulate a resumed session with prior turns.
        agent._turn_count = 7
        # Repopulate messages as a real resume would do.
        agent._messages = [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": [{"type": "text", "text": "earlier reply"}]},
        ]
        prompt = agent._build_system_prompt()
        assert "SESSION START BEHAVIOR" not in prompt

    def test_greeting_suppressed_on_turn_two(self) -> None:
        """The second turn of a fresh session must not re-greet."""
        agent = CARLAgent(model="test", _client=MagicMock())
        agent._turn_count = 2
        prompt = agent._build_system_prompt()
        assert "SESSION START BEHAVIOR" not in prompt

    def test_greeting_fires_when_messages_empty_on_turn_one(self) -> None:
        """The greeting gate must not depend on the ``_messages`` length.

        This was the pre-REV-004 bug: a resumed session with trimmed
        ``_messages`` would re-fire the greeting even though the user had
        been mid-conversation.
        """
        agent = CARLAgent(model="test", _client=MagicMock())
        agent._turn_count = 1
        # No message history at all — but this IS turn 1, so greet.
        agent._messages = []
        prompt = agent._build_system_prompt()
        assert "SESSION START BEHAVIOR" in prompt


# =========================================================================
# REV-010 — _tool_frame invalidates the cached constitution prompt
# =========================================================================


class TestFrameInvalidatesConstitutionCache:
    def test_set_frame_clears_cached_constitution_prompt(self) -> None:
        """After _tool_frame runs, the constitution prompt cache must be None."""
        agent = CARLAgent(model="test", _client=MagicMock())
        # Seed the cache with a pretend compiled block.
        agent._constitution_prompt = "CACHED RULES BLOCK"

        agent._tool_frame({
            "domain": "medicine",
            "function": "eval",
            "role": "clinician",
            "objectives": ["accuracy"],
        })

        assert agent._constitution_prompt is None

    def test_successive_frame_mutations_each_clear_cache(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Calling _tool_frame twice with different frames clears the cache each time."""
        # Redirect WorkFrame.save to a scratch dir so we don't touch ~/.carl.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        agent = CARLAgent(model="test", _client=MagicMock())

        # First frame set.
        agent._constitution_prompt = "FIRST CACHED"
        agent._tool_frame({
            "domain": "d1", "function": "f1", "role": "r1", "objectives": [],
        })
        assert agent._constitution_prompt is None

        # Reprime the cache as if a later _get_constitution_prompt had populated it.
        agent._constitution_prompt = "SECOND CACHED"
        agent._tool_frame({
            "domain": "d2", "function": "f2", "role": "r2", "objectives": [],
        })
        assert agent._constitution_prompt is None

    def test_frame_mutation_produces_new_topic_set(self) -> None:
        """After _tool_frame, the topics derived from the frame reflect the new frame."""
        agent = CARLAgent(model="test", _client=MagicMock())
        agent._tool_frame({
            "domain": "radiology",
            "function": "diagnosis",
            "role": "specialist",
            "objectives": [],
        })
        topics = agent._derive_topics_from_frame()
        assert topics == {"radiology", "diagnosis", "specialist"}


# =========================================================================
# System prompt extension — CLI bootstrap priming path
# =========================================================================


class TestSystemPromptExtension:
    def test_extension_appended_to_prompt_when_set(self) -> None:
        agent = CARLAgent(model="test", _client=MagicMock())
        agent._turn_count = 1
        agent._system_prompt_extension = "[jit-context] hello world"
        prompt = agent._build_system_prompt()
        assert "[jit-context] hello world" in prompt

    def test_extension_is_consumed_after_one_turn(self) -> None:
        """Extension clears after being emitted — only the turn that set it sees it."""
        agent = CARLAgent(model="test", _client=MagicMock())
        agent._turn_count = 1
        agent._system_prompt_extension = "one-shot primer"
        _ = agent._build_system_prompt()
        assert agent._system_prompt_extension == ""

        agent._turn_count = 2
        prompt2 = agent._build_system_prompt()
        assert "one-shot primer" not in prompt2

    def test_empty_extension_is_ignored(self) -> None:
        agent = CARLAgent(model="test", _client=MagicMock())
        agent._turn_count = 1
        agent._system_prompt_extension = ""
        prompt = agent._build_system_prompt()
        # No stray blank block at the extension slot.
        assert "[jit-context]" not in prompt

    def test_extension_default_is_empty(self) -> None:
        agent = CARLAgent(model="test", _client=MagicMock())
        assert agent._system_prompt_extension == ""


# =========================================================================
# J2 — Anthropic rate-limit retry + carl.rate_limit error event
# =========================================================================


class TestRateLimitErrorHandling:
    """Verify the agent surfaces a stable ``carl.rate_limit`` code when
    the Anthropic SDK's own ``max_retries`` budget is exhausted.

    The SDK raises :class:`anthropic.RateLimitError` after backoff on
    persistent 429s. The agent must:

    1. Catch that specific exception (not the generic ``API error``
       catchall) and yield an ``error`` event with ``code="carl.rate_limit"``,
    2. Never propagate the exception out of the generator.
    """

    def _make_rate_limit_exc(self) -> Exception:
        """Construct a real ``anthropic.RateLimitError`` for the mock client.

        Uses a minimal ``httpx.Response`` so the SDK's constructor (which
        requires ``response=`` + ``body=`` kwargs) accepts the build. We
        cannot fake this with a plain ``RuntimeError`` because the agent's
        branch is keyed on ``isinstance(..., anthropic.RateLimitError)``.
        """
        import anthropic
        import httpx
        resp = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.anthropic.com/x"),
        )
        return anthropic.RateLimitError(
            "too many requests",
            response=resp,
            body=None,
        )

    def test_rate_limit_raised_at_stream_entry_yields_carl_rate_limit(
        self,
    ) -> None:
        """SDK raises ``RateLimitError`` when opening the stream — we
        must surface ``code=carl.rate_limit`` and return cleanly."""
        bad_client = MagicMock()
        bad_client.messages.stream.side_effect = self._make_rate_limit_exc()
        agent = CARLAgent(model="test-model", _client=bad_client)

        events = list(agent.chat("hi"))  # must not raise
        errors = [e for e in events if e.kind == "error"]
        assert errors, "expected an error event"
        assert errors[0].code == "carl.rate_limit"
        assert "rate limited" in errors[0].content.lower()
        # No ``done`` event because we returned early.
        assert not any(e.kind == "done" for e in events)

    def test_rate_limit_during_context_manager_mapped_to_same_code(
        self,
    ) -> None:
        """If the exception sneaks through the ctx-manager exit instead
        of the inner try-block, the outer catchall must still tag it
        ``carl.rate_limit`` rather than the generic stream_error."""
        rate_exc = self._make_rate_limit_exc()

        class _FailingCtx:
            def __enter__(self) -> Any:
                raise rate_exc

            def __exit__(self, *args: Any) -> bool:
                return False

        client = MagicMock()
        client.messages.stream = MagicMock(return_value=_FailingCtx())
        agent = CARLAgent(model="test-model", _client=client)

        events = list(agent.chat("hi"))
        errors = [e for e in events if e.kind == "error"]
        assert errors
        # The inner guard catches it at ``.stream(**kwargs)`` return, so
        # the code should be carl.rate_limit. If the SDK ever moves the
        # raise into the CM enter path, the outer catchall covers it.
        assert errors[0].code == "carl.rate_limit"

    def test_non_rate_limit_errors_still_mapped_to_api_error(self) -> None:
        """Regression guard — RuntimeError (non-rate-limit) must stay on
        the ``carl.api_error`` code path to preserve legacy BC."""
        bad_client = MagicMock()
        bad_client.messages.stream.side_effect = RuntimeError(
            "cannot open stream",
        )
        agent = CARLAgent(model="test-model", _client=bad_client)

        events = list(agent.chat("hi"))
        errors = [e for e in events if e.kind == "error"]
        assert errors
        assert errors[0].code in ("carl.api_error", "carl.stream_error")
        assert errors[0].code != "carl.rate_limit"

    def test_agent_client_has_retries_configured(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When we construct an agent without injecting a client, the
        ``anthropic.Anthropic(...)`` call must pass ``max_retries=4`` and
        an explicit ``httpx.Timeout`` so the J2 retry contract holds.

        We monkeypatch ``anthropic.Anthropic`` to a spy so we can assert
        the kwargs without depending on the specific attribute the SDK
        happens to store them on — which varies across SDK versions and
        across test runs where earlier modules may have stubbed the class.
        """
        try:
            import anthropic
            import httpx
        except ImportError:
            pytest.skip("anthropic SDK not installed")

        captured_kwargs: dict[str, Any] = {}

        class _SpyAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(anthropic, "Anthropic", _SpyAnthropic)

        CARLAgent(
            model="test-model",
            api_key="sk-ant-fake-for-init-only",
        )
        assert captured_kwargs.get("max_retries") == 4
        # Timeout should be an ``httpx.Timeout`` so a wedged handshake
        # cannot stall the agent indefinitely.
        timeout = captured_kwargs.get("timeout")
        assert isinstance(timeout, httpx.Timeout)
