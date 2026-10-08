"""Generate native compatibility carriers from the canonical CARL plugin."""

from __future__ import annotations

import json
from pathlib import Path

from carl_studio.plugin import project


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "plugin.json").read_text())
    identity = {key: manifest[key] for key in ("name", "version", "description", "author")}
    carriers = {
        ".claude-plugin/plugin.json": {**identity, "mcpServers": "./.mcp.json"},
        ".codex-plugin/plugin.json": {
            **identity,
            "skills": "./skills/",
            "mcpServers": "./.mcp.json",
            "interface": manifest["extensions"]["com.openai"]["interface"],
        },
        ".mcp.json": {"mcpServers": {"carl": {"command": "carl-mcp", "args": []}}},
        ".claude-plugin/marketplace.json": {
            "name": "carl-local",
            "owner": manifest["author"],
            "metadata": {"description": manifest["description"]},
            "plugins": [{"name": "carl", "source": "./"}],
        },
    }
    for name, value in carriers.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")
    alias = root / ".claude/skills/terminals/carl.md"
    alias.write_text(
        (root / "skills/carl/SKILL.md")
        .read_text()
        .replace("references/workflows.md", "../../../skills/carl/references/workflows.md")
    )
    project(root, root / "build/carl-marketplace")


if __name__ == "__main__":
    main()
