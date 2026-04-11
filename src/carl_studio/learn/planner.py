"""Natural language interpretation for carl learn.

When users type `carl learn "review MCP servers and recommend training"`,
this module interprets the intent and produces a structured LearningPlan.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from carl_studio.data.kits import BUILTIN_KITS
from carl_studio.llm import LLMProvider, parse_llm_json

__all__ = ["LearningPlan", "interpret_natural"]


@dataclass
class LearningPlan:
    """Structured result of NL interpretation."""
    sources: list[str]
    kit: str
    action: str        # "gather" | "synthesize" | "train"
    count: int
    explanation: str


NL_SYSTEM = """You interpret user requests for the CARL training CLI.

CARL can:
- Gather from: local directories, skill:// files, mcp:// servers, claw:// datasets, hf:// datasets, GitHub URLs
- Synthesize: generate graded training samples from gathered code

Available Kits: {kits}

The user's MCP servers (from config): {mcp_servers}

Respond with ONLY this JSON:
{{
  "sources": ["list of source paths/URIs to gather from"],
  "kit": "kit ID from the list above, or empty string if unclear",
  "action": "gather or synthesize",
  "count": 10,
  "explanation": "one-line explanation of what you'll do"
}}"""


def _read_mcp_servers() -> list[str]:
    """Read MCP server names from user config."""
    for config_path in [Path.home() / ".claude.json", Path(".mcp.json")]:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                servers = config.get("mcpServers", config.get("servers", {}))
                return list(servers.keys())
            except Exception:
                pass
    return []


def interpret_natural(source: str) -> LearningPlan:
    """Interpret natural language input into a structured plan."""
    llm = LLMProvider.auto()

    kits = ", ".join(BUILTIN_KITS.keys())
    mcp_servers = ", ".join(_read_mcp_servers()) or "(none found)"

    system = NL_SYSTEM.format(kits=kits, mcp_servers=mcp_servers)

    response = llm.complete([
        {"role": "system", "content": system},
        {"role": "user", "content": source},
    ], max_tokens=500)

    # Parse JSON
    data = parse_llm_json(response)
    if data is None:
        return LearningPlan(
            sources=[],
            kit="coding-agent",
            action="gather",
            count=10,
            explanation=f"Could not interpret: {source[:100]}",
        )

    return LearningPlan(
        sources=data.get("sources", []),
        kit=data.get("kit", "coding-agent"),
        action=data.get("action", "gather"),
        count=data.get("count", 10),
        explanation=data.get("explanation", ""),
    )
