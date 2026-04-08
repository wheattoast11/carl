"""Test MCP server tools are importable."""


def test_mcp_server_importable():
    from carl_studio.mcp import mcp
    assert mcp is not None
    assert mcp.name == "carl-studio"
