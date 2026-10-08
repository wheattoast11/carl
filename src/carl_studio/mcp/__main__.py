"""Protocol-only entrypoint shared by CARL CLI and local plugins."""

from __future__ import annotations

import asyncio
import sys


async def serve(transport: str = "stdio", *, host: str = "127.0.0.1", port: int = 8100) -> None:
    """Serve through the connection owner and release its resources on EOF."""
    from carl_studio.mcp.connection import MCPServerConnection
    from carl_studio.mcp.server import bind_connection

    async with MCPServerConnection(transport=transport, host=host, port=port) as connection:
        bind_connection(connection)
        try:
            await connection.run()
        finally:
            from carl_studio.harness.runtime import close_runtime
            await close_runtime()
            bind_connection(None)


def main() -> None:
    """Start stdio without CLI banners or first-run interaction."""
    from carl_studio.logging_config import configure_logging

    configure_logging()
    try:
        asyncio.run(serve())
    except ImportError:
        print("Install CARL MCP support: pip install 'carl-studio[mcp]'", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
