"""Shared helpers for reward functions.

extract_text  -- handles string or conversational format completions
extract_json  -- extract tool-call JSON with fenced + bare detection, brace-depth counting
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_text(completion: str | list[dict[str, Any]]) -> str:
    """Extract plain text from a completion (string or conversational format)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("content"):
                return str(msg["content"])
        return ""
    return str(completion)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract a tool-call dict from model output text.

    Tries fenced ```json blocks first, then bare JSON objects.
    Uses brace-depth counting for nested JSON.
    Returns a dict with at least a ``"name"`` key, or None.
    """
    # --- Try fenced json block first ---
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        parsed = _try_parse_tool_json(fenced.group(1).strip())
        if parsed is not None:
            return parsed

    # --- Try bare JSON: find first { and its matching } via brace-depth ---
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None

    parsed = _try_parse_tool_json(text[start : end + 1])
    return parsed


def _try_parse_tool_json(raw: str) -> dict[str, Any] | None:
    """Parse a JSON string into a tool-call dict. Returns None on failure."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    if "name" not in obj:
        return None
    # Normalize: if "arguments" is a string, parse it to dict
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            obj["arguments"] = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            pass  # leave as string -- caller handles
    return obj
