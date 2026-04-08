"""Task reward functions R1-R5 for tool-calling RL training.

R1: tool_call_format_reward  -- 5-level structural correctness (w=2.0)
R2: tool_selection_reward    -- correct tool with 0.5 partial credit (w=1.5)
R3: chain_completion_reward  -- multi-step chain progress (w=2.0)
R4: neuralese_v2_reward      -- action density ratio (w=0.5)
R5: conciseness_reward       -- linear length penalty (w=0.5)

All rewards follow the TRL GRPO signature:
    def reward_fn(completions, **kwargs) -> list[float]
"""

from __future__ import annotations

import json
import re
from typing import Any

from carl_studio.training.rewards.base import extract_text, extract_json

# ---------------------------------------------------------------------------
# Minimal Tool Registry (8 tools matching Agent Zero's ToolRegistry)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict[str, list[str]]] = {
    "read_file": {"required": ["path"]},
    "write_file": {"required": ["path", "content"]},
    "list_directory": {"required": ["path"]},
    "web_search": {"required": ["query"]},
    "execute_code": {"required": ["code"]},
    "search_codebase": {"required": ["pattern"]},
    "run_shell": {"required": ["command"]},
    "git_operation": {"required": ["operation"]},
}
TOOL_NAMES: set[str] = set(TOOL_REGISTRY.keys())


def _get_required_args(name: str) -> list[str]:
    """Return required argument names for a tool, or empty list."""
    t = TOOL_REGISTRY.get(name)
    return t["required"] if t else []


# ---------------------------------------------------------------------------
# Fenced code block pattern (for neuralese)
# ---------------------------------------------------------------------------

_FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```")


# ---------------------------------------------------------------------------
# R1: tool_call_format_reward  (weight 2.0)
# ---------------------------------------------------------------------------

def tool_call_format_reward(completions: list, **kwargs: Any) -> list[float]:
    """5-level format scoring for tool call structure.

    0.0 -- No JSON-like content at all
    0.3 -- Contains JSON object but no "name" key
    0.6 -- Valid JSON with "name" field, no "arguments"
    0.8 -- Valid JSON with "name" + "arguments" fields
    1.0 -- Arguments keys match the tool's required JSON Schema
    """
    scores: list[float] = []
    for completion in completions:
        text = extract_text(completion)
        scores.append(_score_format(text))
    return scores


def _score_format(text: str) -> float:
    """Score a single completion's tool-call format quality."""
    if "{" not in text:
        return 0.0

    # Try to extract a tool call (requires "name" key)
    tool_call = extract_json(text)

    if tool_call is None:
        # Maybe there's a JSON object without "name" -- check via brace-depth
        start = text.find("{")
        if start != -1:
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
            if end != -1:
                try:
                    json.loads(text[start : end + 1])
                    return 0.3  # valid JSON, no "name"
                except (json.JSONDecodeError, ValueError):
                    pass
        return 0.0

    # Has "name" -- check for "arguments"
    name = tool_call.get("name", "")
    args = tool_call.get("arguments")

    if args is None or not isinstance(args, dict):
        return 0.6  # name but no/invalid arguments

    # Has name + arguments dict -- check if keys match required schema
    if name in TOOL_NAMES:
        required = _get_required_args(name)
        if required and all(k in args for k in required):
            return 1.0

    return 0.8  # name + arguments, but keys don't match or tool unknown


# ---------------------------------------------------------------------------
# R2: tool_selection_reward  (weight 1.5)
# ---------------------------------------------------------------------------

def tool_selection_reward(
    completions: list,
    expected_tools: list[list[str]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Score whether the model selected the correct tool.

    0.0 -- No tool call found OR tool name not in registry
    0.5 -- Valid tool name in registry but not in expected_tools list
    1.0 -- Tool name is in expected_tools
    """
    if expected_tools is None:
        return [0.0] * len(completions)

    scores: list[float] = []
    for i, completion in enumerate(completions):
        text = extract_text(completion)
        tool_call = extract_json(text)

        if tool_call is None:
            scores.append(0.0)
            continue

        name = tool_call.get("name", "")

        if name not in TOOL_NAMES:
            scores.append(0.0)
            continue

        expected = expected_tools[i] if i < len(expected_tools) else []
        exp_list = expected if isinstance(expected, list) else [expected]
        if name in exp_list:
            scores.append(1.0)
        else:
            scores.append(0.5)  # Valid tool but not expected

    return scores


# ---------------------------------------------------------------------------
# R3: chain_completion_reward  (weight 2.0)
# ---------------------------------------------------------------------------

def chain_completion_reward(
    completions: list,
    expected_tools: list[list[str]] | None = None,
    chain_length: list[int] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Score tool chain progress.

    chain_length=0 (no tool expected): 1.0 if no tool, 0.0 if tool called
    chain_length>0, no tool call: 0.0
    chain_length>0, wrong tool: 0.0 (0.3 if mentions expected tool name)
    chain_length=1, correct first tool: 1.0
    chain_length>1, correct first tool: 0.6
    """
    if expected_tools is None or chain_length is None:
        return [0.0] * len(completions)

    scores: list[float] = []
    for i, completion in enumerate(completions):
        text = extract_text(completion)
        cl = chain_length[i] if i < len(chain_length) else 0
        expected = expected_tools[i] if i < len(expected_tools) else []
        exp_list = expected if isinstance(expected, list) else [expected]
        tool_call = extract_json(text)

        if cl == 0:
            # No tool expected
            scores.append(1.0 if tool_call is None else 0.0)
            continue

        if tool_call is None:
            scores.append(0.0)
            continue

        name = tool_call.get("name", "")

        if name in exp_list:
            # Correct tool
            if cl == 1:
                scores.append(1.0)
            else:
                scores.append(0.6)
        else:
            # Wrong tool -- partial credit if mentions expected name
            if any(exp_name in text for exp_name in exp_list):
                scores.append(0.3)
            else:
                scores.append(0.0)

    return scores


# ---------------------------------------------------------------------------
# R4: neuralese_v2_reward  (weight 0.5)
# ---------------------------------------------------------------------------

def neuralese_v2_reward(completions: list, **kwargs: Any) -> list[float]:
    """Action density reward: structured_chars / total_chars * 2.0, capped at 1.0.

    Structured chars = JSON objects + fenced code blocks.
    Short completions (<30 chars) count all as structured.
    Empty = 0.0.
    """
    scores: list[float] = []
    for completion in completions:
        text = extract_text(completion)
        if not text:
            scores.append(0.0)
            continue
        total = len(text)
        if total < 30:
            scores.append(1.0)
            continue

        structured = 0

        # Count fenced code blocks
        for m in _FENCED_BLOCK_RE.finditer(text):
            structured += len(m.group(0))

        # Count JSON objects (top-level braces) with brace-depth counting
        # Remove already-counted fenced blocks to avoid double counting
        remaining = _FENCED_BLOCK_RE.sub("", text)
        i = 0
        while i < len(remaining):
            if remaining[i] == "{":
                depth = 0
                start = i
                for j in range(i, len(remaining)):
                    if remaining[j] == "{":
                        depth += 1
                    elif remaining[j] == "}":
                        depth -= 1
                        if depth == 0:
                            structured += j - start + 1
                            i = j + 1
                            break
                else:
                    # Unmatched brace -- skip
                    i += 1
            else:
                i += 1

        density = (structured / total) * 2.0
        scores.append(min(1.0, round(density, 4)))
    return scores


# ---------------------------------------------------------------------------
# R5: conciseness_reward  (weight 0.5)
# ---------------------------------------------------------------------------

def conciseness_reward(
    completions: list,
    max_completion_length: int = 512,
    **kwargs: Any,
) -> list[float]:
    """Linear conciseness: score = max(0.0, 1.0 - len(text)/max_completion_length)."""
    scores: list[float] = []
    for completion in completions:
        text = extract_text(completion)
        score = max(0.0, 1.0 - (len(text) / max_completion_length))
        scores.append(round(score, 4))
    return scores
