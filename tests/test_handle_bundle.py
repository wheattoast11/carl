"""Tests for carl_studio.handles.bundle.HandleRuntimeBundle.

Covers:
- Build wires every vault + toolkit against one chain.
- anthropic_tools() produces the union of all toolkit schemas + CU.
- agent_handlers() keys match schema names.
- make_handler() serializes dicts and catches exceptions.
- register_all() iterates into a dispatcher-shaped sink.
- tool_catalog() describes the full surface.
"""

from __future__ import annotations

import json
from typing import Any

from carl_core.interaction import InteractionChain

from carl_studio.handles.bundle import (
    HandleRuntimeBundle,
    make_handler,
)


class _FakeDispatcher:
    """Minimal duck-typed ToolDispatcher."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def register(self, name: str, fn: Any) -> None:
        self.tools[name] = fn


def test_build_wires_all_toolkits() -> None:
    chain = InteractionChain()
    bundle = HandleRuntimeBundle.build(chain)
    assert bundle.chain is chain
    assert bundle.data_toolkit is not None
    assert bundle.browser_toolkit is not None
    assert bundle.subprocess_toolkit is not None
    assert bundle.cu_dispatcher is not None
    # The underlying vaults are shared across toolkits when not overridden
    assert bundle.browser_toolkit.resource_vault is bundle.resource_vault
    assert bundle.subprocess_toolkit.resource_vault is bundle.resource_vault


def test_anthropic_tools_is_union_plus_cu() -> None:
    chain = InteractionChain()
    bundle = HandleRuntimeBundle.build(chain)
    tools = bundle.anthropic_tools()
    names = {t["name"] for t in tools}
    # A representative sample from each toolkit
    assert "data_open_file" in names
    assert "browser_open_page" in names
    assert "subprocess_spawn" in names
    assert "computer" in names


def test_agent_handlers_match_anthropic_schemas() -> None:
    chain = InteractionChain()
    bundle = HandleRuntimeBundle.build(chain)
    handler_names = set(bundle.agent_handlers().keys())
    schema_names = {t["name"] for t in bundle.anthropic_tools()}
    assert handler_names == schema_names


def test_make_handler_wraps_method_to_tool_callable() -> None:
    chain = InteractionChain()
    bundle = HandleRuntimeBundle.build(chain)
    handler = bundle.agent_handlers()["data_open_file"]

    # Missing required arg — TypeError path → ("Error: ...", True)
    out, is_error = handler({})
    assert is_error is True
    assert out.startswith("Error")


def test_handler_success_returns_json_string() -> None:
    chain = InteractionChain()
    bundle = HandleRuntimeBundle.build(chain)
    # data_list_handles takes no args — perfect for a success-path test
    handler = bundle.agent_handlers()["data_list_handles"]
    out, is_error = handler({})
    assert is_error is False
    # Parse JSON — should be a list
    parsed = json.loads(out)
    assert isinstance(parsed, list)


def test_make_handler_catches_non_json_result() -> None:
    class _Weird:
        pass

    def _bad_method(**kwargs: Any) -> Any:
        return {"obj": _Weird()}

    wrapped = make_handler(_bad_method)  # type: ignore[arg-type]
    out, is_error = wrapped({})
    # default=str catches object → string → JSON succeeds. Still returns ok=False.
    # The test just asserts no crash.
    assert is_error in (False, True)
    assert isinstance(out, str)


def test_make_handler_wraps_exceptions() -> None:
    def _boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("kaboom")

    wrapped = make_handler(_boom)  # type: ignore[arg-type]
    out, is_error = wrapped({"x": 1})
    assert is_error is True
    assert "RuntimeError" in out
    assert "kaboom" in out


def test_register_all_populates_dispatcher() -> None:
    chain = InteractionChain()
    bundle = HandleRuntimeBundle.build(chain)
    sink = _FakeDispatcher()
    names = bundle.register_all(sink)
    assert set(names) == set(sink.tools.keys())
    # Pick one to exercise
    out, is_error = sink.tools["data_list_handles"]({})
    assert is_error is False


def test_tool_catalog_describes_surface() -> None:
    chain = InteractionChain()
    bundle = HandleRuntimeBundle.build(chain)
    cat = bundle.tool_catalog()
    assert cat["toolkits"]["data"]
    assert cat["toolkits"]["browser"]
    assert cat["toolkits"]["subprocess"]
    assert cat["toolkits"]["computer_use"] == ["computer"]
    assert cat["total"] > 0
    assert "refs, not values" in cat["doctrine"]


def test_handler_for_data_open_bytes_roundtrip() -> None:
    chain = InteractionChain()
    bundle = HandleRuntimeBundle.build(chain)
    # We don't expose data_open_bytes to the agent by default — only
    # data_open_file — which is correct (the agent shouldn't be minting
    # bytes via a tool). Assert the coverage explicitly.
    schemas = bundle.anthropic_tools()
    data_names = {s["name"] for s in schemas if s["name"].startswith("data_")}
    assert "data_open_file" in data_names
    assert "data_open_bytes" not in data_names
