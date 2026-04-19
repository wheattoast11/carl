"""Tests for the MCP 2025-11-25 tool output-schema registry.

Covers:
    * All 16 legacy tools have an entry in :data:`OUTPUT_SCHEMAS`.
    * Each schema round-trips through :func:`json.dumps`.
    * :func:`register_output_schemas` attaches the schema onto the
      FastMCP tool objects (when FastMCP is available).
"""

from __future__ import annotations

import json
import unittest

import pytest

from carl_studio.mcp.output_schemas import (
    LEGACY_TOOL_NAMES,
    OUTPUT_SCHEMAS,
    register_output_schemas,
)


class TestOutputSchemaRegistry(unittest.TestCase):
    def test_all_legacy_tools_have_entries(self) -> None:
        missing = [name for name in LEGACY_TOOL_NAMES if name not in OUTPUT_SCHEMAS]
        assert not missing, f"missing schemas for: {missing}"

    def test_legacy_count_is_sixteen_plus_authenticate(self) -> None:
        # Nine "original" tools + seven new-surface tools = 16 (excluding
        # authenticate which brings the count to 17; task tools are
        # separate). This guards against accidentally dropping an entry.
        # The 16-legacy requirement from the spec matches the following:
        legacy_set = set(LEGACY_TOOL_NAMES)
        assert len(legacy_set) >= 16
        # All of them must have schemas.
        for name in legacy_set:
            assert name in OUTPUT_SCHEMAS

    def test_new_2025_11_tools_have_entries(self) -> None:
        for name in ("submit_async_training", "tasks_get", "tasks_cancel"):
            assert name in OUTPUT_SCHEMAS, f"missing {name!r} schema"

    def test_every_schema_json_round_trips(self) -> None:
        for name, schema in OUTPUT_SCHEMAS.items():
            blob = json.dumps(schema)
            loaded = json.loads(blob)
            assert loaded == schema, f"schema for {name} failed round-trip"

    def test_every_schema_has_type_or_one_of(self) -> None:
        for name, schema in OUTPUT_SCHEMAS.items():
            assert isinstance(schema, dict), f"{name} schema not a dict"
            assert "type" in schema or "oneOf" in schema, (
                f"{name} schema has neither 'type' nor 'oneOf'"
            )

    def test_start_training_schema_shape(self) -> None:
        s = OUTPUT_SCHEMAS["start_training"]
        assert s["type"] == "object"
        assert "run_id" in s["properties"]
        assert "phase" in s["properties"]

    def test_tasks_get_schema_enum_covers_all_states(self) -> None:
        schema = OUTPUT_SCHEMAS["tasks_get"]
        # oneOf[0] is the happy-path with the status enum
        happy = schema["oneOf"][0]
        states = set(happy["properties"]["status"]["enum"])
        assert states == {"pending", "running", "completed", "failed", "cancelled"}

    def test_tasks_cancel_schema_shape(self) -> None:
        s = OUTPUT_SCHEMAS["tasks_cancel"]
        assert s["type"] == "object"
        assert "cancelled" in s["properties"]
        assert s["properties"]["cancelled"]["type"] == "boolean"

    def test_authenticate_schema_one_of_covers_success_and_error(self) -> None:
        s = OUTPUT_SCHEMAS["authenticate"]
        assert "oneOf" in s
        assert len(s["oneOf"]) == 2
        authenticated_true = next(
            branch for branch in s["oneOf"]
            if branch["properties"]["authenticated"].get("const") is True
        )
        assert "tier" in authenticated_true["properties"]

    def test_generate_bundle_returns_raw_string(self) -> None:
        assert OUTPUT_SCHEMAS["generate_bundle"] == {"type": "string"}


class TestRegisterOutputSchemas(unittest.TestCase):
    def test_attaches_schemas_onto_fastmcp_tools(self) -> None:
        pytest.importorskip("mcp")
        from mcp.server.fastmcp import FastMCP

        m = FastMCP("test-schemas")

        @m.tool()
        async def start_training(config_yaml: str) -> str:  # noqa: ARG001 # pyright: ignore[reportUnusedFunction]
            return "{}"

        @m.tool()
        async def generate_bundle(config_yaml: str) -> str:  # noqa: ARG001 # pyright: ignore[reportUnusedFunction]
            return "print('hi')"

        attached = register_output_schemas(m)
        # Only the tools actually registered on `m` should be attached.
        assert "start_training" in attached
        assert "generate_bundle" in attached
        # Instance-level mirror for introspection.
        mirror = getattr(m, "_carl_output_schemas", None)
        assert isinstance(mirror, dict)
        assert "start_training" in mirror

    def test_ignores_unknown_tools_silently(self) -> None:
        pytest.importorskip("mcp")
        from mcp.server.fastmcp import FastMCP

        m = FastMCP("empty-schemas")
        # No tools registered — should succeed and attach nothing.
        attached = register_output_schemas(m)
        assert attached == {}

    def test_real_server_has_schemas_attached(self) -> None:
        """The live carl-studio server registers all schemas at import."""
        pytest.importorskip("mcp")
        from carl_studio.mcp import mcp as live_mcp

        # Import-time side-effect: schemas attached to the server's tools.
        attached = getattr(live_mcp, "_carl_output_schemas", {})
        assert isinstance(attached, dict)
        # Every legacy tool on the live server should have a schema.
        for name in LEGACY_TOOL_NAMES:
            assert name in attached, f"{name} not schema-registered on live server"
        # And the new 2025-11 tools too.
        assert "submit_async_training" in attached
        assert "tasks_get" in attached
        assert "tasks_cancel" in attached


if __name__ == "__main__":
    unittest.main()
