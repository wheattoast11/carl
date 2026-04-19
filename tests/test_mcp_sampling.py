"""Tests for the MCP 2025-11-25 sampling primitive.

As with elicitation, we inject a stub ServerSession via the connection's
``_session_override`` hook so the plumbing is testable without a live
client.
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from typing import Any, cast

import pytest

from carl_core.errors import CARLError, CARLTimeoutError
from carl_core.interaction import InteractionChain
from carl_studio.mcp.connection import MCPServerConnection
from carl_studio.mcp.sampling import (
    SamplingCostEvent,
    SamplingRequest,
    SamplingResponse,
    estimate_cost,
    marshal_sdk_response,
    sample,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class FakeContent:
    type: str = "text"
    text: str = "hello"


@dataclass
class FakeCreateMessageResult:
    content: FakeContent
    model: str = "claude-sonnet-4-6"
    stopReason: str | None = "endTurn"
    usage: dict[str, int] | None = None
    meta: dict[str, Any] | None = None


class StubSession:
    def __init__(
        self,
        *,
        result: FakeCreateMessageResult | None = None,
        delay_s: float = 0.0,
        raise_exc: Exception | None = None,
    ) -> None:
        self.result = result or FakeCreateMessageResult(
            content=FakeContent(text="greetings"),
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        self.delay_s = delay_s
        self.raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    async def create_message(
        self,
        *,
        messages: Any,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        model_preferences: Any = None,
        **_extra: Any,
    ) -> FakeCreateMessageResult:
        self.calls.append(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "stop_sequences": stop_sequences,
                "model_preferences": model_preferences,
            }
        )
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.result


class FakeConnection:
    def __init__(self, session: Any = None, chain: Any = None) -> None:
        self.connection_id = "fake-conn"
        self.fastmcp = None
        self._session_override = session
        self.chain = chain


# ---------------------------------------------------------------------------
# Dataclass validation
# ---------------------------------------------------------------------------


class TestSamplingDataclasses(unittest.TestCase):
    def test_request_requires_non_empty_messages(self) -> None:
        with pytest.raises(ValueError):
            SamplingRequest(messages=[])

    def test_request_rejects_bad_max_tokens(self) -> None:
        with pytest.raises(ValueError):
            SamplingRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=0)

    def test_request_rejects_bad_temperature(self) -> None:
        with pytest.raises(ValueError):
            SamplingRequest(
                messages=[{"role": "user", "content": "hi"}], temperature=3.0
            )

    def test_response_defaults(self) -> None:
        resp = SamplingResponse(content="", model="", stop_reason="endTurn")
        assert resp.usage == {}


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestEstimateCost(unittest.TestCase):
    def test_known_model_returns_float(self) -> None:
        cost = estimate_cost(
            "claude-sonnet-4-6", {"input_tokens": 1000, "output_tokens": 500}
        )
        assert cost is not None
        # 1000/1000 * 0.003 + 500/1000 * 0.015 = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 1e-9

    def test_unknown_model_returns_none(self) -> None:
        assert estimate_cost("no-such-model", {"input_tokens": 10}) is None

    def test_empty_usage_returns_zero(self) -> None:
        cost = estimate_cost("claude-haiku-4-5", {})
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Response marshalling
# ---------------------------------------------------------------------------


class TestMarshalSdkResponse(unittest.TestCase):
    def test_extracts_text_and_metadata(self) -> None:
        raw = FakeCreateMessageResult(
            content=FakeContent(text="answer"),
            model="claude-opus-4-6",
            stopReason="endTurn",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        resp = marshal_sdk_response(raw)
        assert resp.content == "answer"
        assert resp.model == "claude-opus-4-6"
        assert resp.stop_reason == "endTurn"
        assert resp.usage == {"input_tokens": 100, "output_tokens": 50}

    def test_missing_stop_reason_defaults_to_end_turn(self) -> None:
        raw = FakeCreateMessageResult(
            content=FakeContent(text=""),
            stopReason=None,
        )
        resp = marshal_sdk_response(raw)
        assert resp.stop_reason == "endTurn"


# ---------------------------------------------------------------------------
# sample() — happy / timeout / send-fail / no-session
# ---------------------------------------------------------------------------


class TestSample(unittest.IsolatedAsyncioTestCase):
    async def test_happy_path_returns_response(self) -> None:
        pytest.importorskip("mcp")
        session = StubSession()
        conn = FakeConnection(session=session)
        req = SamplingRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=128,
            temperature=0.5,
            system_prompt="be brief",
        )
        resp = await sample(req, connection=cast(MCPServerConnection, conn), timeout_s=5.0)
        assert resp.content == "greetings"
        assert resp.model == "claude-sonnet-4-6"
        assert len(session.calls) == 1
        call = session.calls[0]
        assert call["max_tokens"] == 128
        assert call["temperature"] == 0.5
        assert call["system_prompt"] == "be brief"

    async def test_cost_event_recorded_on_chain(self) -> None:
        pytest.importorskip("mcp")
        chain = InteractionChain()
        session = StubSession()
        conn = FakeConnection(session=session, chain=chain)
        req = SamplingRequest(messages=[{"role": "user", "content": "hi"}])
        await sample(req, connection=cast(MCPServerConnection, conn), timeout_s=5.0)

        cost_steps = [
            s for s in chain.steps
            if s.name == "connection.mcp.sampling.cost"
        ]
        assert len(cost_steps) == 1
        output = cost_steps[0].output
        assert output is not None
        assert output.get("prompt_tokens") == 10
        assert output.get("completion_tokens") == 5

    async def test_timeout_raises_carl_timeout_error(self) -> None:
        pytest.importorskip("mcp")
        session = StubSession(delay_s=0.3)
        conn = FakeConnection(session=session)
        req = SamplingRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(CARLTimeoutError) as excinfo:
            await sample(req, connection=cast(MCPServerConnection, conn), timeout_s=0.05)
        assert excinfo.value.code == "carl.timeout.sampling"

    async def test_no_session_raises_carl_error(self) -> None:
        conn = FakeConnection(session=None)
        req = SamplingRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(CARLError) as excinfo:
            await sample(req, connection=cast(MCPServerConnection, conn))
        assert "no_session" in (excinfo.value.code or "")

    async def test_underlying_failure_wrapped(self) -> None:
        pytest.importorskip("mcp")
        session = StubSession(raise_exc=RuntimeError("wire err"))
        conn = FakeConnection(session=session)
        req = SamplingRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(CARLError) as excinfo:
            await sample(req, connection=cast(MCPServerConnection, conn))
        assert excinfo.value.code == "carl.mcp.sampling.send_failed"

    async def test_rejects_non_positive_timeout(self) -> None:
        conn = FakeConnection(session=StubSession())
        req = SamplingRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(ValueError):
            await sample(req, connection=cast(MCPServerConnection, conn), timeout_s=0.0)

    async def test_cost_event_dataclass_shape(self) -> None:
        event = SamplingCostEvent(
            model="claude-haiku-4-5",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.00003,
        )
        assert event.model == "claude-haiku-4-5"
        assert event.cost_usd == 0.00003


if __name__ == "__main__":
    unittest.main()
