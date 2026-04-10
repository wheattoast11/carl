"""External tool availability checker."""
from __future__ import annotations
import shutil

__all__ = ["check_tools", "ensure_tools"]

TOOLS = {
    "ast-grep": "cargo install ast-grep",
    "rg": "brew install ripgrep",
    "python3": "(should be available)",
}


def check_tools() -> dict[str, bool]:
    return {name: shutil.which(name) is not None for name in TOOLS}


def ensure_tools(console) -> None:
    status = check_tools()
    for name, available in status.items():
        if available:
            console.ok(f"{name} available")
        else:
            console.warn(f"{name} not found — install: {TOOLS[name]}")
