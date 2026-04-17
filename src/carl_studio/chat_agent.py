"""CARL agent loop — agentic chat with tool use and proactive guidance.

Wires existing infrastructure (SourceIngester, WorkFrame, Python execution)
as callable tools via the Claude API tool use protocol.  The agent drives
the workflow: it recommends next steps, ingests files on mention, runs
analyses, and creates deliverables proactively.

Features:
  - Session persistence: save/resume conversations across invocations
  - Permission hooks: pre/post tool-use callbacks for consent gating
  - Cost tracking: per-turn cost accumulation with budget enforcement
  - Context compaction: automatic summarization at token threshold
  - Streaming: real-time token display via text_delta events
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard context bounds
# ---------------------------------------------------------------------------
_COMPACT_THRESHOLD = 140_000
_TOOL_RESULT_MAX = 8_000
_KEEP_RECENT = 10

# ---------------------------------------------------------------------------
# Model pricing ($/M tokens: input, output)
# ---------------------------------------------------------------------------
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
}
_DEFAULT_PRICING = (5.00, 25.00)  # assume Opus pricing if unknown


def _compute_turn_cost(usage: Any, model: str) -> float:
    """Compute USD cost for a single API turn from response.usage."""
    input_rate, output_rate = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (
        (input_tokens * input_rate / 1_000_000)
        + (output_tokens * output_rate / 1_000_000)
        + (cache_read * input_rate * 0.1 / 1_000_000)
        + (cache_create * input_rate * 1.25 / 1_000_000)
    )


# ---------------------------------------------------------------------------
# Permission hooks
# ---------------------------------------------------------------------------


class ToolPermission(str, Enum):
    """Result from a pre-tool-use hook."""

    ALLOW = "allow"
    DENY = "deny"


# Callback types
PreToolHook = Callable[[str, dict[str, Any]], ToolPermission]
PostToolHook = Callable[[str, dict[str, Any], str], None]


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

_SESSIONS_DIR = Path.home() / ".carl" / "sessions"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SessionStore:
    """Save and resume CARLAgent sessions at ~/.carl/sessions/."""

    def __init__(self, sessions_dir: Path | str | None = None) -> None:
        self._dir = Path(sessions_dir) if sessions_dir else _SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, state: dict[str, Any]) -> Path:
        """Persist session state to JSON."""
        state["updated_at"] = _now_iso()
        if "created_at" not in state:
            state["created_at"] = state["updated_at"]
        path = self._dir / f"{session_id}.json"
        # Serialize sets in knowledge entries
        knowledge = state.get("knowledge", [])
        serializable_knowledge = []
        for entry in knowledge:
            entry_copy = dict(entry)
            if "words" in entry_copy and isinstance(entry_copy["words"], set):
                entry_copy["words"] = sorted(entry_copy["words"])
            serializable_knowledge.append(entry_copy)
        state["knowledge"] = serializable_knowledge
        path.write_text(json.dumps(state, indent=2, default=str))
        return path

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load session state from JSON. Returns None if not found."""
        path = self._dir / f"{session_id}.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        # Deserialize word lists back to sets
        for entry in data.get("knowledge", []):
            if "words" in entry and isinstance(entry["words"], list):
                entry["words"] = set(entry["words"])
        return data

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List sessions, most recent first."""
        sessions = []
        for p in sorted(self._dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            if len(sessions) >= limit:
                break
            try:
                data = json.loads(p.read_text())
                sessions.append({
                    "id": p.stem,
                    "title": data.get("title", ""),
                    "model": data.get("model", ""),
                    "turn_count": data.get("turn_count", 0),
                    "total_cost_usd": data.get("total_cost_usd", 0.0),
                    "updated_at": data.get("updated_at", ""),
                })
            except Exception:
                continue
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if found."""
        path = self._dir / f"{session_id}.json"
        if path.is_file():
            path.unlink()
            return True
        return False

# ---------------------------------------------------------------------------
# Tool schemas for Claude API
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "ingest_source",
        "description": "Ingest files from a path, URL, or directory into the knowledge base for later querying.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, directory, URL, or HF dataset ID"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "query_knowledge",
        "description": "Search the ingested knowledge base. Returns the most relevant chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "What to search for"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "run_analysis",
        "description": "Execute Python code for data analysis. Use pandas, csv, json, math — standard lib + common packages. Print results to stdout.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "create_file",
        "description": "Create or overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to create"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file and return its contents (up to 8K chars).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "set_frame",
        "description": "Set the analytical WorkFrame lens that shapes how you approach the problem. Call this when the user describes their domain, function, or goals.",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Subject matter (e.g. saas_sales, pharma)"},
                "function": {"type": "string", "description": "Analytical function (e.g. territory_planning)"},
                "role": {"type": "string", "description": "User role (e.g. analyst, manager)"},
                "objectives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Measurable goals",
                },
            },
            "required": ["domain"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory. Returns file names and sizes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current dir)"},
                "pattern": {"type": "string", "description": "Glob pattern (default: *)"},
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


class AgentEvent(BaseModel):
    """Event emitted by the agent loop for the caller to render."""

    kind: str  # "text", "tool_call", "tool_result", "done", "error"
    content: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CARLAgent:
    """Agentic chat loop with tool use, frame awareness, and proactive guidance.

    Features beyond raw API relay:
      - **Session persistence**: save/resume via SessionStore
      - **Permission hooks**: pre_tool_use callback can deny tool calls
      - **Cost tracking**: per-turn cost from response.usage, max_budget_usd cap
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        frame: Any | None = None,
        workdir: str = ".",
        *,
        max_budget_usd: float = 0.0,
        pre_tool_use: PreToolHook | None = None,
        post_tool_use: PostToolHook | None = None,
        session_id: str = "",
        _client: Any | None = None,
    ) -> None:
        # -- Resolve model from settings when not explicitly passed ----------
        if not model:
            try:
                from carl_studio.settings import CARLSettings

                settings = CARLSettings.load()
                model = settings.default_chat_model or "claude-sonnet-4-6"
            except Exception:
                model = "claude-sonnet-4-6"

        # -- Resolve api_key from settings when not explicitly passed --------
        api_key_source = "default"
        if api_key:
            api_key_source = "explicit"
        elif _client is None:
            try:
                from carl_studio.settings import CARLSettings

                settings = CARLSettings.load()
                resolved_key = settings.anthropic_api_key or ""
                if resolved_key:
                    api_key = resolved_key
                    api_key_source = "settings"
            except Exception:
                pass

        if _client is not None:
            self._client = _client
            api_key_source = "injected"
        else:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "Anthropic SDK required: pip install carl-studio[observe]"
                ) from exc
            self._client = anthropic.Anthropic(api_key=api_key or None)
        self._model = model
        self._api_key_source = api_key_source
        self._frame = frame
        self._workdir = str(Path(workdir).resolve())
        self._messages: list[dict[str, Any]] = []
        self._knowledge: list[dict[str, Any]] = []  # {text, source, words}
        self._token_count = 0

        # Cost tracking
        self._total_cost_usd = 0.0
        self._max_budget_usd = max_budget_usd  # 0 = unlimited
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._turn_count = 0

        # Permission hooks
        self._pre_tool_use = pre_tool_use
        self._post_tool_use = post_tool_use

        # Session
        self._session_id = session_id

        # Project context (cached at construction)
        self._project_line = ""
        try:
            from carl_studio.project import load_project

            proj = load_project("carl.yaml")
            self._project_line = f"PROJECT: {proj.name} | Model: {proj.base_model} | Method: {proj.method}"
        except Exception:
            pass

    @property
    def provider_info(self) -> dict[str, str]:
        """Return provider resolution info for display."""
        return {
            "model": self._model,
            "provider": "anthropic",
            "api_key_source": self._api_key_source,
        }

    def chat(self, user_input: str) -> Iterator[AgentEvent]:
        """Send message, handle tool calls, yield events.

        Uses streaming to prevent HTTP timeouts on long responses.
        Yields text_delta events for real-time token display.
        Enforces max_budget_usd if set. Runs pre/post tool hooks.
        """
        self._messages.append({"role": "user", "content": user_input})
        self._turn_count += 1

        while True:
            # Budget check before API call
            if self._max_budget_usd > 0 and self._total_cost_usd >= self._max_budget_usd:
                yield AgentEvent(
                    kind="error",
                    content=f"Budget exceeded: ${self._total_cost_usd:.4f} >= ${self._max_budget_usd:.2f}",
                )
                return

            system = self._build_system_prompt()

            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "max_tokens": 64000,
                    "system": system,
                    "messages": self._messages,
                    "tools": TOOLS,
                    "cache_control": {"type": "ephemeral"},
                }
                if any(fam in self._model for fam in ("opus-4-6", "sonnet-4-6")):
                    kwargs["thinking"] = {"type": "adaptive"}

                with self._client.messages.stream(**kwargs) as stream:
                    for event in stream:
                        if event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                yield AgentEvent(
                                    kind="text_delta", content=event.delta.text
                                )
                    response = stream.get_final_message()
            except Exception as exc:
                yield AgentEvent(kind="error", content=f"API error: {exc}")
                return

            # Track cost
            usage = getattr(response, "usage", None)
            if usage is not None:
                turn_cost = _compute_turn_cost(usage, self._model)
                self._total_cost_usd += turn_cost
                self._total_input_tokens += getattr(usage, "input_tokens", 0) or 0
                self._total_output_tokens += getattr(usage, "output_tokens", 0) or 0

            self._token_count = getattr(usage, "input_tokens", self._token_count) if usage else self._token_count

            # Context pressure
            if self._token_count > _COMPACT_THRESHOLD:
                self._compact()

            # pause_turn: server-side tool iteration limit
            if response.stop_reason == "pause_turn":
                self._messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": self._content_to_dicts(response.content)},
                ]
                continue

            # Collect blocks
            assistant_content: list[dict[str, Any]] = []
            tool_use_blocks: list[Any] = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "thinking":
                    pass
                elif block.type == "tool_use":
                    tool_input: dict[str, Any] = block.input if isinstance(block.input, dict) else {}
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                    tool_use_blocks.append(block)
                    yield AgentEvent(
                        kind="tool_call",
                        tool_name=block.name,
                        tool_args=tool_input,
                    )

            if not tool_use_blocks:
                self._messages.append({"role": "assistant", "content": assistant_content})
                yield AgentEvent(kind="done", content=f"${self._total_cost_usd:.4f}")
                return

            # Execute ALL tool calls with permission hooks
            self._messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in tool_use_blocks:
                tool_input = block.input if isinstance(block.input, dict) else {}

                # Pre-tool hook
                if self._pre_tool_use is not None:
                    permission = self._pre_tool_use(block.name, tool_input)
                    if permission == ToolPermission.DENY:
                        result = f"Blocked by permission policy: {block.name}"
                        yield AgentEvent(kind="tool_blocked", tool_name=block.name, content=result)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                            "is_error": True,
                        })
                        continue

                result = self._dispatch_tool(block.name, tool_input)
                yield AgentEvent(kind="tool_result", tool_name=block.name, content=result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

                # Post-tool hook
                if self._post_tool_use is not None:
                    self._post_tool_use(block.name, tool_input, result)

            self._messages.append({"role": "user", "content": tool_results})

            if response.stop_reason == "end_turn":
                yield AgentEvent(kind="done", content=f"${self._total_cost_usd:.4f}")
                return

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    _GREETING_INSTRUCTIONS = (
        "On the FIRST message of a new session (when you have no prior conversation history), begin by:\n"
        "1. Briefly introduce yourself as CARL — a coherence-aware training assistant.\n"
        "2. Ask 2-3 focused questions to understand what the user is working on:\n"
        "   - What model are they training or want to train?\n"
        "   - What is their training objective (task-specific, general, alignment)?\n"
        "   - Do they have a dataset ready, or need help generating one?\n"
        "3. If a WorkFrame is active, reference it to show you already have context.\n"
        "\n"
        "After each response, suggest 1-2 concrete next steps the user could take.\n"
        "Keep responses concise — no walls of text."
    )

    def _build_system_prompt(self) -> str:
        parts: list[str] = [
            "You are CARL, a coherence-aware training assistant from terminals.tech (Intuition Labs LLC).",
            "",
        ]

        # Frame context
        if self._frame and getattr(self._frame, "active", False):
            parts.append("ACTIVE FRAME:")
            parts.append(self._frame.attention_query())
            parts.append("")
        else:
            parts.append("No frame set. When the user describes their goal, call set_frame immediately.")
            parts.append("")

        # Project context (cached at construction time)
        if self._project_line:
            parts.append(self._project_line)
            parts.append("")

        # Knowledge base
        if self._knowledge:
            sources = len({k["source"] for k in self._knowledge})
            parts.append(f"KNOWLEDGE BASE: {len(self._knowledge)} chunks from {sources} sources.")
            parts.append("")

        # Greeting / proactive agency for new sessions
        is_new_session = len(self._messages) <= 1
        if is_new_session:
            parts.append("SESSION START BEHAVIOR:")
            parts.append(self._GREETING_INSTRUCTIONS)
            parts.append("")

        # Behavioral instructions
        parts.extend([
            "YOUR APPROACH:",
            "1. DRIVE the workflow. Recommend the next step proactively.",
            "2. When the user describes their goal, call set_frame immediately.",
            "3. When files or directories are mentioned, call ingest_source immediately.",
            "4. For analytical tasks, write and run Python code via run_analysis. Show results.",
            "5. Create deliverable files (CSV, reports, models) via create_file proactively.",
            "6. Keep responses concise. Lead with action, not explanation.",
            "7. When asked about ingested data, use query_knowledge first.",
        ])

        return "\n".join(parts)

    @staticmethod
    def _content_to_dicts(content: Any) -> list[dict[str, Any]]:
        """Convert SDK content blocks to dicts for message history."""
        result: list[dict[str, Any]] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                result.append({"type": "text", "text": block.text})
            elif getattr(block, "type", None) == "tool_use":
                result.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return result

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "ingest_source":
                return self._tool_ingest(args.get("path", ""))
            if name == "query_knowledge":
                return self._tool_query(args.get("question", ""))
            if name == "run_analysis":
                return self._tool_run(args.get("code", ""))
            if name == "create_file":
                return self._tool_create(args.get("path", ""), args.get("content", ""))
            if name == "read_file":
                return self._tool_read(args.get("path", ""))
            if name == "set_frame":
                return self._tool_frame(args)
            if name == "list_files":
                return self._tool_list(args.get("path", "."), args.get("pattern", "*"))
            return f"Unknown tool: {name}"
        except Exception as exc:
            return f"Error: {exc}"

    def _tool_ingest(self, path: str) -> str:
        from carl_studio.learn.ingest import SourceIngester

        ingester = SourceIngester()
        chunks = ingester.ingest(path)
        modalities: dict[str, int] = {}
        for c in chunks:
            self._knowledge.append({
                "text": c.text,
                "source": c.source,
                "words": set(c.text.lower().split()),
            })
            mod = getattr(c, "modality", "text")
            modalities[mod] = modalities.get(mod, 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(modalities.items()))
        return f"Ingested {len(chunks)} chunks from {path} ({summary})"

    def _tool_query(self, question: str) -> str:
        if not self._knowledge:
            return "Knowledge base is empty. Ingest files first with ingest_source."
        terms = set(question.lower().split())
        scored: list[tuple[int, dict[str, Any]]] = []
        for chunk in self._knowledge:
            overlap = len(terms & chunk.get("words", set()))
            if overlap > 0:
                scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = scored[:5]
        if not results:
            return "No relevant chunks found for that query."
        parts: list[str] = []
        for score, chunk in results:
            text = str(chunk["text"])[:1500]
            parts.append(f"[{chunk['source']}] (relevance: {score})\n{text}")
        return "\n\n---\n\n".join(parts)

    def _tool_run(self, code: str) -> str:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            timeout=30,
            cwd=self._workdir,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        stdout = result.stdout.decode(errors="replace")[:6000]
        stderr = result.stderr.decode(errors="replace")[:2000]
        if result.returncode == 0:
            return stdout or "(no output)"
        return f"Exit code {result.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"

    def _resolve_safe_path(self, path: str) -> Path | None:
        """Resolve path against workdir and reject traversal attempts."""
        resolved = Path(path).resolve()
        workdir = Path(self._workdir).resolve()
        if resolved == workdir or str(resolved).startswith(str(workdir) + os.sep):
            return resolved
        return None

    def _tool_create(self, path: str, content: str) -> str:
        p = self._resolve_safe_path(path)
        if p is None:
            return f"Blocked: path '{path}' is outside the working directory."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Created {p} ({len(content)} chars)"

    def _tool_read(self, path: str) -> str:
        p = self._resolve_safe_path(path)
        if p is None:
            return f"Blocked: path '{path}' is outside the working directory."
        if not p.is_file():
            return f"File not found: {path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > _TOOL_RESULT_MAX:
            return text[:_TOOL_RESULT_MAX] + f"\n... [truncated, {len(text) - _TOOL_RESULT_MAX} chars omitted]"
        return text

    def _tool_frame(self, args: dict[str, Any]) -> str:
        from carl_studio.frame import WorkFrame

        frame = WorkFrame(
            domain=args.get("domain", ""),
            function=args.get("function", ""),
            role=args.get("role", ""),
            objectives=args.get("objectives", []),
        )
        frame.save()
        self._frame = frame
        return f"Frame set: {frame.domain}/{frame.function}/{frame.role}"

    def _tool_list(self, path: str, pattern: str) -> str:
        p = Path(path)
        if not p.is_dir():
            return f"Not a directory: {path}"
        files = sorted(p.glob(pattern))[:50]
        if not files:
            return f"No files matching '{pattern}' in {path}"
        lines = []
        for f in files:
            if f.is_file():
                size = f.stat().st_size
                if size > 1_000_000:
                    size_str = f"{size / 1_000_000:.1f}MB"
                elif size > 1000:
                    size_str = f"{size / 1000:.1f}KB"
                else:
                    size_str = f"{size}B"
                lines.append(f"  {f.name}  ({size_str})")
            else:
                lines.append(f"  {f.name}/")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context compaction
    # ------------------------------------------------------------------

    def _compact(self) -> None:
        """Summarize older messages to stay within context bounds."""
        if len(self._messages) <= _KEEP_RECENT:
            return

        old = self._messages[:-_KEEP_RECENT]
        recent = self._messages[-_KEEP_RECENT:]

        summary_parts: list[str] = []
        for msg in old:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_bits = [
                    b.get("text", b.get("content", ""))
                    for b in content
                    if isinstance(b, dict)
                ]
                content = " ".join(str(t) for t in text_bits)
            if isinstance(content, str) and content.strip():
                summary_parts.append(f"{role}: {content[:200]}")

        summary = "Session summary (compacted):\n" + "\n".join(summary_parts[-20:])

        self._messages = [
            {"role": "user", "content": summary},
            {"role": "assistant", "content": "Understood. Continuing from the session summary."},
            *recent,
        ]
        self._token_count = 0
        logger.info("Compacted %d messages into summary + %d recent", len(old), len(recent))

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def save_session(
        self, title: str = "", store: SessionStore | None = None
    ) -> str:
        """Save current state to a session file. Returns session ID."""
        from uuid import uuid4

        if not self._session_id:
            self._session_id = str(uuid4())[:8]

        state: dict[str, Any] = {
            "id": self._session_id,
            "title": title,
            "model": self._model,
            "workdir": self._workdir,
            "messages": self._messages,
            "knowledge": self._knowledge,
            "frame": self._frame.model_dump() if self._frame and hasattr(self._frame, "model_dump") else None,
            "total_cost_usd": self._total_cost_usd,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "turn_count": self._turn_count,
        }

        s = store or SessionStore()
        s.save(self._session_id, state)
        return self._session_id

    def load_session(
        self, session_id: str, store: SessionStore | None = None
    ) -> bool:
        """Load a saved session. Returns True if found."""
        s = store or SessionStore()
        state = s.load(session_id)
        if state is None:
            return False

        self._session_id = state.get("id", session_id)
        self._messages = state.get("messages", [])
        self._knowledge = state.get("knowledge", [])
        self._total_cost_usd = state.get("total_cost_usd", 0.0)
        self._total_input_tokens = state.get("total_input_tokens", 0)
        self._total_output_tokens = state.get("total_output_tokens", 0)
        self._turn_count = state.get("turn_count", 0)

        frame_data = state.get("frame")
        if frame_data:
            from carl_studio.frame import WorkFrame

            self._frame = WorkFrame.model_validate(frame_data)

        return True

    @property
    def cost_summary(self) -> dict[str, Any]:
        """Current cost and token usage."""
        return {
            "total_cost_usd": round(self._total_cost_usd, 6),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "turn_count": self._turn_count,
            "model": self._model,
            "budget_remaining_usd": round(self._max_budget_usd - self._total_cost_usd, 6) if self._max_budget_usd > 0 else None,
        }
