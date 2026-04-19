"""Tests for the MCP 2025-11-25 elicitation primitive.

We avoid spinning up a real FastMCP server; instead we inject a stub
session onto the connection's ``_session_override`` attribute (the
fallback branch in :func:`elicit._extract_session`). This lets us unit-test
the request/response plumbing deterministically without a live client.
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from typing import Any, cast

import pytest

from carl_core.errors import CARLError, CARLTimeoutError
from carl_studio.mcp.connection import MCPServerConnection
from carl_studio.mcp.elicitation import (
    ElicitationRequest,
    ElicitationResponse,
    arg_is_missing,
    elicit,
    elicits_on_missing,
    marshal_sdk_result,
)


# ---------------------------------------------------------------------------
# Stubs that mimic mcp.types.ElicitResult and ServerSession.elicit
# ---------------------------------------------------------------------------


@dataclass
class FakeElicitResult:
    action: str
    content: dict[str, Any] | None = None


class StubSession:
    """Captures the (message, requestedSchema) pair and returns a canned result."""

    def __init__(
        self,
        *,
        result: FakeElicitResult | None = None,
        delay_s: float = 0.0,
        raise_exc: Exception | None = None,
    ) -> None:
        self.result = result or FakeElicitResult(action="accept", content={})
        self.delay_s = delay_s
        self.raise_exc = raise_exc
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def elicit(
        self,
        *,
        message: str,
        requestedSchema: dict[str, Any],
    ) -> FakeElicitResult:
        self.calls.append((message, dict(requestedSchema)))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.result


class FakeConnection:
    """Minimal connection shape that satisfies _resolve/_extract."""

    def __init__(self, session: Any = None, chain: Any = None) -> None:
        self.connection_id = "fake-conn"
        self.fastmcp = None
        self._session_override = session
        self.chain = chain


# ---------------------------------------------------------------------------
# Dataclass + helpers — pure logic, no SDK
# ---------------------------------------------------------------------------


class TestElicitationDataclasses(unittest.TestCase):
    def test_request_rejects_empty_prompt(self) -> None:
        with pytest.raises(ValueError):
            ElicitationRequest(prompt="", schema={"type": "string"})

    def test_request_rejects_non_dict_schema(self) -> None:
        with pytest.raises(TypeError):
            ElicitationRequest(prompt="ask", schema="not-a-dict")  # type: ignore[arg-type]

    def test_response_defaults(self) -> None:
        resp = ElicitationResponse()
        assert resp.values == {}
        assert resp.declined is False
        assert resp.reason is None

    def test_marshal_accept(self) -> None:
        raw = FakeElicitResult(action="accept", content={"name": "carl"})
        out = marshal_sdk_result(raw)
        assert out.declined is False
        assert out.values == {"name": "carl"}

    def test_marshal_decline(self) -> None:
        raw = FakeElicitResult(action="decline")
        out = marshal_sdk_result(raw)
        assert out.declined is True
        assert out.reason == "declined"

    def test_marshal_cancel(self) -> None:
        raw = FakeElicitResult(action="cancel")
        out = marshal_sdk_result(raw)
        assert out.declined is True
        assert out.reason == "cancel"

    def testarg_is_missing(self) -> None:
        assert arg_is_missing(None) is True
        assert arg_is_missing("") is True
        assert arg_is_missing([]) is True
        assert arg_is_missing({}) is True
        assert arg_is_missing("ok") is False
        assert arg_is_missing(0) is False
        assert arg_is_missing(False) is False


# ---------------------------------------------------------------------------
# elicit() — happy / decline / timeout / no-session
# ---------------------------------------------------------------------------


class TestElicit(unittest.IsolatedAsyncioTestCase):
    async def test_sends_frame_and_returns_values(self) -> None:
        session = StubSession(
            result=FakeElicitResult(action="accept", content={"k": "v"}),
        )
        conn = FakeConnection(session=session)
        req = ElicitationRequest(prompt="Pick one", schema={"type": "object"})
        resp = await elicit(req, connection=cast(MCPServerConnection, conn), timeout_s=5.0)
        assert resp.declined is False
        assert resp.values == {"k": "v"}
        assert len(session.calls) == 1
        msg, schema = session.calls[0]
        assert msg == "Pick one"
        assert schema == {"type": "object"}

    async def test_decline_path(self) -> None:
        session = StubSession(result=FakeElicitResult(action="decline"))
        conn = FakeConnection(session=session)
        req = ElicitationRequest(prompt="Pick", schema={"type": "string"})
        resp = await elicit(req, connection=cast(MCPServerConnection, conn))
        assert resp.declined is True
        assert resp.reason == "declined"

    async def test_timeout_raises_carl_timeout_error(self) -> None:
        session = StubSession(delay_s=0.3)
        conn = FakeConnection(session=session)
        req = ElicitationRequest(prompt="Wait", schema={"type": "string"})
        with pytest.raises(CARLTimeoutError) as excinfo:
            await elicit(req, connection=cast(MCPServerConnection, conn), timeout_s=0.05)
        assert excinfo.value.code == "carl.timeout.elicitation"

    async def test_no_session_raises_carl_error(self) -> None:
        conn = FakeConnection(session=None)
        req = ElicitationRequest(prompt="Hi", schema={"type": "string"})
        with pytest.raises(CARLError) as excinfo:
            await elicit(req, connection=cast(MCPServerConnection, conn))
        assert "no_session" in (excinfo.value.code or "")

    async def test_underlying_exception_wrapped(self) -> None:
        session = StubSession(raise_exc=RuntimeError("wire glitch"))
        conn = FakeConnection(session=session)
        req = ElicitationRequest(prompt="Hi", schema={"type": "string"})
        with pytest.raises(CARLError) as excinfo:
            await elicit(req, connection=cast(MCPServerConnection, conn))
        assert excinfo.value.code == "carl.mcp.elicit.send_failed"

    async def test_rejects_non_positive_timeout(self) -> None:
        conn = FakeConnection(session=StubSession())
        req = ElicitationRequest(prompt="Hi", schema={"type": "string"})
        with pytest.raises(ValueError):
            await elicit(req, connection=cast(MCPServerConnection, conn), timeout_s=0.0)


# ---------------------------------------------------------------------------
# @elicits_on_missing decorator
# ---------------------------------------------------------------------------


class TestElicitsOnMissing(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Bind a stub connection so the decorator's elicit() lookup succeeds.
        import carl_studio.mcp.server as srv

        self._previous = srv.get_bound_connection()

    async def asyncTearDown(self) -> None:
        import carl_studio.mcp.server as srv

        srv.bind_connection(self._previous)

    async def test_decorator_skips_elicitation_when_arg_present(self) -> None:
        @elicits_on_missing(
            "name",
            prompt="Who?",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        async def greet(*, name: str | None = None) -> str:
            return f"hi {name}"

        out = await greet(name="carl")
        assert out == "hi carl"

    async def test_decorator_elicits_when_missing(self) -> None:
        import carl_studio.mcp.server as srv

        session = StubSession(
            result=FakeElicitResult(action="accept", content={"name": "tej"}),
        )
        conn = FakeConnection(session=session)
        srv.bind_connection(cast(MCPServerConnection, conn))

        @elicits_on_missing(
            "name",
            prompt="Who?",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        async def greet(*, name: str | None = None) -> str:
            return f"hi {name}"

        out = await greet()
        assert out == "hi tej"
        assert len(session.calls) == 1

    async def test_decorator_declined_raises_carl_error(self) -> None:
        import carl_studio.mcp.server as srv

        session = StubSession(result=FakeElicitResult(action="decline"))
        conn = FakeConnection(session=session)
        srv.bind_connection(cast(MCPServerConnection, conn))

        @elicits_on_missing(
            "name",
            prompt="Who?",
            schema={"type": "object"},
        )
        async def greet(*, name: str | None = None) -> str:
            return f"hi {name}"

        with pytest.raises(CARLError) as excinfo:
            await greet()
        assert excinfo.value.code == "carl.mcp.elicit.declined"

    async def test_decorator_falls_back_to_single_value(self) -> None:
        import carl_studio.mcp.server as srv

        # Response content has a single field with a DIFFERENT key — we
        # should still bind it to the missing argument.
        session = StubSession(
            result=FakeElicitResult(
                action="accept",
                content={"answer": "singleton"},
            ),
        )
        conn = FakeConnection(session=session)
        srv.bind_connection(cast(MCPServerConnection, conn))

        @elicits_on_missing(
            "target",
            prompt="?",
            schema={"type": "object"},
        )
        async def fn(*, target: str | None = None) -> str:
            return f"target={target}"

        out = await fn()
        assert out == "target=singleton"

    async def test_decorator_rejects_empty_arg_name(self) -> None:
        with pytest.raises(ValueError):
            elicits_on_missing("", prompt="x", schema={})


if __name__ == "__main__":
    unittest.main()
