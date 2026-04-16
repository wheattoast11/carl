"""CARL agent loop — agentic chat with tool use and proactive guidance.

Wires existing infrastructure (SourceIngester, WorkFrame, Python execution)
as callable tools via the Claude API tool use protocol.  The agent drives
the workflow: it recommends next steps, ingests files on mention, runs
analyses, and creates deliverables proactively.

Context window management:
  System prompt:  ~6K tokens  (frame + project + tools)
  Working context: up to 140K (messages + tool results)
  Buffer:          54K        (never touched — overflow guard)

When working context exceeds _COMPACT_THRESHOLD, older messages are
summarized and compacted.  Tool results are truncated to _TOOL_RESULT_MAX.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard context bounds
# ---------------------------------------------------------------------------
_COMPACT_THRESHOLD = 140_000
_TOOL_RESULT_MAX = 8_000
_KEEP_RECENT = 10

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
    """Agentic chat loop with tool use, frame awareness, and proactive guidance."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        frame: Any | None = None,
        workdir: str = ".",
        *,
        _client: Any | None = None,
    ) -> None:
        if _client is not None:
            self._client = _client
        else:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "Anthropic SDK required: pip install carl-studio[observe]"
                ) from exc
            self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._frame = frame
        self._workdir = str(Path(workdir).resolve())
        self._messages: list[dict[str, Any]] = []
        self._knowledge: list[dict[str, Any]] = []  # {text, source, words}
        self._token_count = 0
        self._project_line = ""
        try:
            from carl_studio.project import load_project

            proj = load_project("carl.yaml")
            self._project_line = f"PROJECT: {proj.name} | Model: {proj.base_model} | Method: {proj.method}"
        except Exception:
            pass

    def chat(self, user_input: str) -> Iterator[AgentEvent]:
        """Send message, handle tool calls, yield events."""
        self._messages.append({"role": "user", "content": user_input})

        while True:
            system = self._build_system_prompt()

            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=self._messages,
                tools=TOOLS,
            )

            self._token_count = getattr(
                getattr(response, "usage", None), "input_tokens", self._token_count
            )

            # Check context pressure
            if self._token_count > _COMPACT_THRESHOLD:
                self._compact()

            # Process response blocks
            assistant_content: list[dict[str, Any]] = []
            has_tool_use = False

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                    yield AgentEvent(kind="text", content=block.text)
                elif block.type == "tool_use":
                    has_tool_use = True
                    tool_input: dict[str, Any] = block.input if isinstance(block.input, dict) else {}
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                    yield AgentEvent(
                        kind="tool_call",
                        tool_name=block.name,
                        tool_args=tool_input,
                    )

                    # Execute tool
                    result = self._dispatch_tool(block.name, tool_input)
                    yield AgentEvent(kind="tool_result", tool_name=block.name, content=result)

                    # Append assistant message + tool result for continuation
                    self._messages.append({"role": "assistant", "content": assistant_content})
                    self._messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }],
                    })
                    assistant_content = []
                    break  # re-enter loop for next API call

            if not has_tool_use:
                # Final response — no more tool calls
                self._messages.append({"role": "assistant", "content": assistant_content})
                yield AgentEvent(kind="done")
                return

            if response.stop_reason == "end_turn":
                yield AgentEvent(kind="done")
                return

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

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
