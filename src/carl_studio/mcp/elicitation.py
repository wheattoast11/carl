"""Server → client elicitation for missing tool args.

MCP 2025-11-25 lets a server ask the caller for missing information mid-tool
via ``elicit``. We expose two primitives:

* :func:`elicit` — send an elicitation request to the bound client and
  ``await`` the typed response. Wraps ``mcp.server.session.ServerSession.elicit``
  when the SDK is available; otherwise raises a clear
  :class:`ConnectionUnavailableError`.
* :func:`elicits_on_missing` — decorator that turns "required arg missing"
  into an automatic elicitation round-trip. The wrapped tool is re-called
  with the elicited value once the client responds.

Integration points
------------------
All request/response boundaries record steps on the connection's
``InteractionChain`` so downstream observers (Coherence, billing, audit
log) see elicitation as a first-class external interaction.

Timeouts bubble up as :class:`CARLTimeoutError` with
``code="carl.timeout.elicitation"``. Client ``decline`` or ``cancel``
returns an :class:`ElicitationResponse` with ``declined=True`` — callers
decide whether to raise or degrade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

import anyio

from carl_core.errors import CARLError, CARLTimeoutError
from carl_core.interaction import ActionType

from carl_studio.mcp.session import extract_session

if TYPE_CHECKING:  # pragma: no cover - type-only
    from carl_studio.mcp.connection import MCPServerConnection


_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Request / response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ElicitationRequest:
    """A request for the client to supply structured input.

    ``schema`` is a plain JSON Schema dict. The MCP spec constrains it to
    primitive fields (string / integer / number / boolean / enum); we
    don't re-validate here — the client is the final arbiter of what
    shapes it will accept.
    """

    prompt: str
    schema: dict[str, Any]
    context: dict[str, Any] = field(default_factory=lambda: {})
    required: bool = True

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("ElicitationRequest.prompt must be non-empty")
        if not isinstance(self.schema, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"ElicitationRequest.schema must be a dict, got {type(self.schema).__name__}"
            )


@dataclass
class ElicitationResponse:
    """The client's response to an elicitation.

    ``declined`` is True when the client returned ``action='decline'`` or
    ``action='cancel'``. ``reason`` is populated for decline/cancel or
    when the response is malformed; ``values`` is always present (empty
    dict on decline) so consumers can unpack without branching.
    """

    values: dict[str, Any] = field(default_factory=lambda: {})
    declined: bool = False
    reason: str | None = None


# ---------------------------------------------------------------------------
# Core send helper
# ---------------------------------------------------------------------------


def _resolve_connection(
    connection: "MCPServerConnection | None",
) -> "MCPServerConnection":
    if connection is not None:
        return connection
    # Late import: ``server.py`` imports this module indirectly during the
    # @mcp.tool() registration pass, so we defer to call-time.
    from carl_studio.mcp.server import get_bound_connection

    resolved = get_bound_connection()
    if resolved is None:
        raise CARLError(
            "MCP elicitation requires a bound MCPServerConnection; "
            "call server.bind_connection(conn) after opening the connection.",
            code="carl.mcp.no_connection",
            context={"operation": "elicit"},
        )
    return resolved


async def _send_via_sdk(
    session: Any,
    prompt: str,
    schema: dict[str, Any],
) -> Any:
    """Delegate to ``ServerSession.elicit(message, requestedSchema)``.

    The return is an ``mcp.types.ElicitResult`` — we marshal it to our
    local :class:`ElicitationResponse` in the caller.
    """
    # Keyword-order matches the MCP SDK 1.10+ surface we verified at
    # implementation time. ``requestedSchema`` is the camelCase parameter
    # name used by the JSON-RPC wire format and the Python binding.
    return await session.elicit(message=prompt, requestedSchema=schema)


def marshal_sdk_result(raw: Any) -> ElicitationResponse:
    """Translate an ``mcp.types.ElicitResult``-like object to our dataclass."""
    action: str = str(getattr(raw, "action", "accept"))
    content = getattr(raw, "content", None)
    if action == "accept":
        if isinstance(content, dict):
            values: dict[str, Any] = {str(k): v for k, v in content.items()}  # type: ignore[misc]
        else:
            values = {}
        return ElicitationResponse(values=values, declined=False, reason=None)
    reason = "declined" if action == "decline" else action
    return ElicitationResponse(values={}, declined=True, reason=reason)


async def elicit(
    request: ElicitationRequest,
    *,
    connection: "MCPServerConnection | None" = None,
    timeout_s: float = 30.0,
) -> ElicitationResponse:
    """Send an elicitation request to the bound client and await the response.

    Args:
        request: The typed elicitation payload.
        connection: Explicit connection; defaults to the one bound on the
            server module via :func:`bind_connection`.
        timeout_s: Seconds to wait before raising :class:`CARLTimeoutError`.

    Returns:
        An :class:`ElicitationResponse`. Use ``.declined`` to branch.

    Raises:
        CARLTimeoutError: If the client does not reply before the deadline.
        CARLError: If the MCP SDK elicitation surface is missing or the
            framing fails.
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")

    conn = _resolve_connection(connection)
    chain = getattr(conn, "chain", None)

    # Record the request on the chain (input only; output is logged after reply).
    step_id: str | None = None
    if chain is not None:
        step = chain.record(
            ActionType.EXTERNAL,
            "connection.mcp.elicit.request",
            input={"prompt": request.prompt, "schema": request.schema},
            success=True,
        )
        step_id = step.step_id

    # Resolve the ServerSession we send on. :func:`extract_session` is the
    # canonical helper shared with :mod:`carl_studio.mcp.sampling` — it
    # honours the ``_session_override`` test hook and otherwise falls back
    # to ``fastmcp.get_context().session``.
    session = extract_session(conn)
    if session is None:
        if chain is not None and step_id is not None:
            chain.record(
                ActionType.EXTERNAL,
                "connection.mcp.elicit.response",
                input={"prompt": request.prompt},
                output={"error": "no_session"},
                success=False,
                parent_id=step_id,
            )
        raise CARLError(
            "MCP elicitation requires an active ServerSession; "
            "ensure the server is inside a live request context.",
            code="carl.mcp.elicit.no_session",
            context={"connection_id": conn.connection_id},
        )

    try:
        with anyio.fail_after(timeout_s):
            raw = await _send_via_sdk(session, request.prompt, request.schema)
    except TimeoutError as exc:
        timeout_err = CARLTimeoutError(
            f"Elicitation timed out after {timeout_s}s",
            code="carl.timeout.elicitation",
            context={
                "prompt": request.prompt,
                "timeout_s": timeout_s,
                "connection_id": conn.connection_id,
            },
            cause=exc,
        )
        if chain is not None and step_id is not None:
            chain.record(
                ActionType.EXTERNAL,
                "connection.mcp.elicit.response",
                input={"prompt": request.prompt},
                output={"error": str(timeout_err)},
                success=False,
                parent_id=step_id,
            )
        raise timeout_err from exc
    except CARLError:
        raise
    except Exception as exc:  # noqa: BLE001 — wrap for the caller
        wrapped = CARLError(
            f"Elicitation send failed: {exc}",
            code="carl.mcp.elicit.send_failed",
            context={"connection_id": conn.connection_id},
            cause=exc,
        )
        if chain is not None and step_id is not None:
            chain.record(
                ActionType.EXTERNAL,
                "connection.mcp.elicit.response",
                input={"prompt": request.prompt},
                output={"error": str(wrapped)},
                success=False,
                parent_id=step_id,
            )
        raise wrapped from exc

    response = marshal_sdk_result(raw)
    if chain is not None and step_id is not None:
        chain.record(
            ActionType.EXTERNAL,
            "connection.mcp.elicit.response",
            input={"prompt": request.prompt},
            output={
                "declined": response.declined,
                "reason": response.reason,
                "value_keys": sorted(response.values.keys()),
            },
            success=not response.declined,
            parent_id=step_id,
        )
    return response


# Backwards-compat alias for in-tree tests that still reference the
# private helper by name. The canonical entry point is
# :func:`carl_studio.mcp.session.extract_session`.
_extract_session = extract_session


# ---------------------------------------------------------------------------
# Decorator — elicits on missing arg
# ---------------------------------------------------------------------------


def elicits_on_missing(
    arg_name: str,
    *,
    prompt: str,
    schema: dict[str, Any],
    timeout_s: float = 30.0,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Decorator: auto-elicit ``arg_name`` from the client when missing.

    The wrapped coroutine is inspected after normal arg binding; if
    ``arg_name`` is missing from the call (or bound to ``None`` / empty
    string / empty collection), we issue an elicitation request and
    re-invoke the body with the returned value.

    If the client declines, the caller sees the usual ``None`` — they
    decide whether to return a tier-style JSON error or raise.

    Args:
        arg_name: Name of the argument that may be missing.
        prompt: Human-readable question shown to the client.
        schema: JSON Schema for the expected argument (single-field).
        timeout_s: Elicitation timeout seconds.
    """
    if not arg_name:
        raise ValueError("arg_name must be non-empty")

    def _decorator(
        body: Callable[..., Awaitable[_T]],
    ) -> Callable[..., Awaitable[_T]]:
        async def _wrapper(*args: Any, **kwargs: Any) -> _T:
            missing = arg_is_missing(kwargs.get(arg_name))
            if not missing:
                return await body(*args, **kwargs)

            req = ElicitationRequest(
                prompt=prompt,
                schema=schema,
                context={"tool": body.__name__, "arg": arg_name},
                required=True,
            )
            response = await elicit(req, timeout_s=timeout_s)
            if response.declined:
                # Re-raise as a typed error so the tool body (or its wrapper)
                # can surface a stable JSON error to the client.
                raise CARLError(
                    f"Client declined elicitation for '{arg_name}'",
                    code="carl.mcp.elicit.declined",
                    context={"arg": arg_name, "reason": response.reason},
                )
            # Prefer exact key, else the single provided value, else fail.
            if arg_name in response.values:
                kwargs[arg_name] = response.values[arg_name]
            elif len(response.values) == 1:
                kwargs[arg_name] = next(iter(response.values.values()))
            else:
                raise CARLError(
                    f"Elicitation response did not supply '{arg_name}'",
                    code="carl.mcp.elicit.malformed",
                    context={
                        "arg": arg_name,
                        "keys": sorted(response.values.keys()),
                    },
                )
            return await body(*args, **kwargs)

        _wrapper.__name__ = getattr(body, "__name__", "elicited_tool")
        _wrapper.__doc__ = body.__doc__
        return _wrapper

    return _decorator


def arg_is_missing(value: Any) -> bool:
    """Return True for values that should trigger an elicitation round-trip.

    ``None`` and empty strings / collections count as missing. Zero,
    ``False``, and ``0.0`` do not — callers that need to treat those as
    missing should use a richer schema default.
    """
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0  # pyright: ignore[reportUnknownArgumentType]
    return False


__all__ = [
    "ElicitationRequest",
    "ElicitationResponse",
    "arg_is_missing",
    "elicit",
    "elicits_on_missing",
    "marshal_sdk_result",
]
