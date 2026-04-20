"""Server → client sampling (``sampling/createMessage``).

MCP 2025-11-25 lets a server borrow the *client's* LLM for sub-tasks
instead of calling its own model. We expose :func:`sample`, a thin
async helper that maps our internal dataclasses onto the SDK's
``ServerSession.create_message`` and records the call on the
:class:`InteractionChain` so cost accounting is consistent with every
other LLM-shaped interaction in CARL Studio.

Cost accounting
---------------
When the caller's agent module supports a ``SamplingCostEvent`` hook,
we emit one after the response arrives. The minimum fields are model,
prompt/completion tokens, and ``cost_usd`` (``None`` when the model is
unknown to the pricing table). The pricing table is intentionally
minimal — we only care about the frontier models CARL Studio's chat
agent already uses; unknown models fall through to ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio

from carl_core.errors import CARLError, CARLTimeoutError
from carl_core.interaction import ActionType

from carl_studio.mcp.session import extract_session

if TYPE_CHECKING:  # pragma: no cover - type-only
    from carl_studio.mcp.connection import MCPServerConnection


# ---------------------------------------------------------------------------
# Pricing table (USD per 1k tokens) — minimal and conservative.
# ---------------------------------------------------------------------------
#
# Numbers reflect the rates the carl_studio.chat_agent module assumes for
# its internal Anthropic calls. Unknown models return ``None`` so we never
# emit a fabricated cost.

_PRICING_USD_PER_1K: dict[str, tuple[float, float]] = {
    # (input, output)
    "claude-opus-4-6": (0.015, 0.075),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-haiku-4-5": (0.0008, 0.004),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SamplingRequest:
    """Payload for ``sampling/createMessage``.

    ``messages`` mirrors the JSON-RPC shape: each item is
    ``{"role": "user" | "assistant", "content": [...]}`` where the content
    blocks are dicts compatible with ``mcp.types.SamplingMessage``.
    """

    messages: list[dict[str, Any]]
    max_tokens: int = 1024
    model_preferences: dict[str, Any] | None = None
    temperature: float = 0.7
    stop_sequences: list[str] = field(default_factory=lambda: [])
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.messages, list) or not self.messages:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("SamplingRequest.messages must be a non-empty list")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be in [0, 2], got {self.temperature}"
            )


@dataclass
class SamplingResponse:
    content: str
    model: str
    stop_reason: str
    usage: dict[str, int] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class SamplingCostEvent:
    """Structured event emitted after a sampling round-trip succeeds."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None


# ---------------------------------------------------------------------------
# Connection / session helpers (mirror elicitation.py)
# ---------------------------------------------------------------------------


def _resolve_connection(
    connection: "MCPServerConnection | None",
) -> "MCPServerConnection":
    if connection is not None:
        return connection
    from carl_studio.mcp.server import get_bound_connection

    resolved = get_bound_connection()
    if resolved is None:
        raise CARLError(
            "MCP sampling requires a bound MCPServerConnection; "
            "call server.bind_connection(conn) after opening the connection.",
            code="carl.mcp.no_connection",
            context={"operation": "sample"},
        )
    return resolved


# Backwards-compat alias — canonical helper is
# :func:`carl_studio.mcp.session.extract_session`.
_extract_session = extract_session


def _to_sampling_message(msg: dict[str, Any]) -> Any:
    """Convert our plain-dict message to an ``mcp.types.SamplingMessage``.

    Kept as a runtime import so :mod:`carl_studio.mcp` stays importable
    when ``mcp`` is not installed — callers who reach sampling always
    have the dep by construction, but we don't force eager import at
    module load.
    """
    from mcp.types import SamplingMessage, TextContent

    role = msg.get("role", "user")
    content_raw = msg.get("content", "")
    if isinstance(content_raw, str):
        content: Any = TextContent(type="text", text=content_raw)
    elif isinstance(content_raw, dict):
        # Already SDK-shaped — trust the caller. Cast the dict values to
        # object to keep pyright from complaining about the **unpack.
        typed: dict[str, Any] = {str(k): v for k, v in content_raw.items()}  # type: ignore[misc]
        content = TextContent(**typed)  # type: ignore[arg-type]
    else:
        content = TextContent(type="text", text=str(content_raw))
    return SamplingMessage(role=role, content=content)


def estimate_cost(model: str, usage: dict[str, int]) -> float | None:
    if model not in _PRICING_USD_PER_1K:
        return None
    in_rate, out_rate = _PRICING_USD_PER_1K[model]
    prompt = usage.get("input_tokens", 0)
    completion = usage.get("output_tokens", 0)
    return round((prompt / 1000.0) * in_rate + (completion / 1000.0) * out_rate, 6)


def _emit_cost_event(conn: "MCPServerConnection", event: SamplingCostEvent) -> None:
    """Forward the cost event to chat_agent's hook when importable.

    Failures are swallowed: cost emission is advisory and must not break
    the sampling path.
    """
    chain = getattr(conn, "chain", None)
    if chain is not None:
        chain.record(
            ActionType.EXTERNAL,
            "connection.mcp.sampling.cost",
            input={"model": event.model},
            output={
                "prompt_tokens": event.prompt_tokens,
                "completion_tokens": event.completion_tokens,
                "cost_usd": event.cost_usd,
            },
            success=True,
        )
    try:
        from carl_studio.chat_agent import record_sampling_cost  # type: ignore[attr-defined]

        record_sampling_cost(event)
    except (ImportError, AttributeError):
        # The chat_agent hook is optional; CARL Studio still records the
        # event on the chain above for downstream consumers.
        return
    except Exception:
        # Never let an advisory hook break the real sampling path.
        return


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def sample(
    request: SamplingRequest,
    *,
    connection: "MCPServerConnection | None" = None,
    timeout_s: float = 60.0,
) -> SamplingResponse:
    """Ask the connected client to sample a response from its LLM.

    Args:
        request: The sampling payload (messages + knobs).
        connection: Explicit connection; defaults to the one bound on
            :mod:`carl_studio.mcp.server`.
        timeout_s: Seconds to wait for the client's reply.

    Returns:
        A :class:`SamplingResponse` with the text content, model id, stop
        reason, and token usage dict.

    Raises:
        CARLTimeoutError: Client did not reply before the deadline.
        CARLError: SDK surface missing, framing failed, or no session.
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")

    conn = _resolve_connection(connection)
    chain = getattr(conn, "chain", None)
    step_id: str | None = None
    if chain is not None:
        step = chain.record(
            ActionType.EXTERNAL,
            "connection.mcp.sampling.request",
            input={
                "message_count": len(request.messages),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "has_system_prompt": request.system_prompt is not None,
            },
            success=True,
        )
        step_id = step.step_id

    session = extract_session(conn)
    if session is None:
        err = CARLError(
            "MCP sampling requires an active ServerSession; "
            "ensure the server is inside a live request context.",
            code="carl.mcp.sampling.no_session",
            context={"connection_id": conn.connection_id},
        )
        if chain is not None and step_id is not None:
            chain.record(
                ActionType.EXTERNAL,
                "connection.mcp.sampling.response",
                input={"model_hint": None},
                output={"error": str(err)},
                success=False,
                parent_id=step_id,
            )
        raise err

    sdk_messages = [_to_sampling_message(m) for m in request.messages]
    # Translate model_preferences to the SDK's typed form when present.
    model_preferences = _marshal_model_prefs(request.model_preferences)

    try:
        with anyio.fail_after(timeout_s):
            raw = await session.create_message(
                messages=sdk_messages,
                max_tokens=request.max_tokens,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                stop_sequences=list(request.stop_sequences) if request.stop_sequences else None,
                model_preferences=model_preferences,
            )
    except TimeoutError as exc:
        timeout_err = CARLTimeoutError(
            f"Sampling timed out after {timeout_s}s",
            code="carl.timeout.sampling",
            context={
                "timeout_s": timeout_s,
                "connection_id": conn.connection_id,
            },
            cause=exc,
        )
        if chain is not None and step_id is not None:
            chain.record(
                ActionType.EXTERNAL,
                "connection.mcp.sampling.response",
                input={"model_hint": None},
                output={"error": str(timeout_err)},
                success=False,
                parent_id=step_id,
            )
        raise timeout_err from exc
    except CARLError:
        raise
    except Exception as exc:  # noqa: BLE001
        wrapped = CARLError(
            f"Sampling send failed: {exc}",
            code="carl.mcp.sampling.send_failed",
            context={"connection_id": conn.connection_id},
            cause=exc,
        )
        if chain is not None and step_id is not None:
            chain.record(
                ActionType.EXTERNAL,
                "connection.mcp.sampling.response",
                input={"model_hint": None},
                output={"error": str(wrapped)},
                success=False,
                parent_id=step_id,
            )
        raise wrapped from exc

    response = marshal_sdk_response(raw)
    if chain is not None and step_id is not None:
        chain.record(
            ActionType.EXTERNAL,
            "connection.mcp.sampling.response",
            input={"model_hint": response.model},
            output={
                "model": response.model,
                "stop_reason": response.stop_reason,
                "content_length": len(response.content),
                "usage": response.usage,
            },
            success=True,
            parent_id=step_id,
        )

    cost_event = SamplingCostEvent(
        model=response.model,
        prompt_tokens=int(response.usage.get("input_tokens", 0)),
        completion_tokens=int(response.usage.get("output_tokens", 0)),
        cost_usd=estimate_cost(response.model, response.usage),
    )
    _emit_cost_event(conn, cost_event)
    return response


def _marshal_model_prefs(prefs: dict[str, Any] | None) -> Any:
    """Convert a plain-dict preferences payload to ``mcp.types.ModelPreferences``.

    Returns ``None`` when the caller did not pass preferences, so the SDK
    can leave the field absent on the wire.
    """
    if prefs is None:
        return None
    try:
        from mcp.types import ModelPreferences
    except ImportError:
        return None
    try:
        return ModelPreferences(**prefs)
    except Exception:
        # Accept dicts that aren't strictly-shaped — the SDK will reject
        # them with a clearer error; we don't want to swallow in that
        # case, but we also don't want to crash tests that pass ad-hoc
        # shapes. Re-raise as CARLError so the caller has context.
        raise CARLError(
            "invalid model_preferences payload",
            code="carl.mcp.sampling.bad_prefs",
            context={"prefs": prefs},
        )


def marshal_sdk_response(raw: Any) -> SamplingResponse:
    """Translate ``mcp.types.CreateMessageResult`` to our dataclass."""
    model = str(getattr(raw, "model", ""))
    stop_reason_raw = getattr(raw, "stopReason", None) or getattr(raw, "stop_reason", None)
    stop_reason = str(stop_reason_raw or "endTurn")

    content_obj = getattr(raw, "content", None)
    if content_obj is None:
        content_text = ""
    elif hasattr(content_obj, "text"):
        content_text = str(content_obj.text)
    elif isinstance(content_obj, str):
        content_text = content_obj
    else:
        content_text = str(content_obj)

    meta_obj: Any = getattr(raw, "meta", None) or {}
    usage_raw: Any = getattr(raw, "usage", None)
    if usage_raw is None and isinstance(meta_obj, dict):
        usage_raw = meta_obj.get("usage")  # type: ignore[assignment]
    if usage_raw is None:
        usage_raw = {}
    usage: dict[str, int] = {}
    if isinstance(usage_raw, dict):
        typed_usage: dict[str, Any] = usage_raw  # type: ignore[assignment]
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            val: Any = typed_usage.get(key)
            if val is not None:
                try:
                    usage[key] = int(val)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue

    return SamplingResponse(
        content=content_text,
        model=model,
        stop_reason=stop_reason,
        usage=usage,
    )


__all__ = [
    "SamplingCostEvent",
    "SamplingRequest",
    "SamplingResponse",
    "estimate_cost",
    "marshal_sdk_response",
    "sample",
]
