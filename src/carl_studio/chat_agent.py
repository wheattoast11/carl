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
  - Corruption resilience: quarantine bad sessions on load
  - Budget pre-check: refuse turns that would exceed the budget cap
  - Tool timeouts: per-tool deadlines, hook isolation
  - Arg schema validation: typed errors surfaced back to the model
  - dispatch_cli: agent can invoke registered ``carl <cmd>`` ops
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

if TYPE_CHECKING:  # pragma: no cover — typing only
    from carl_core.memory import MemoryItem, MemoryLayer, MemoryStore
    from carl_studio.constitution import Constitution, ConstitutionalRule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory recall defaults
# ---------------------------------------------------------------------------
_MEMORY_RECALL_TOP_K = 3
_MEMORY_RECALL_MIN_SCORE = 0.15
_MEMORY_CONTENT_PREVIEW = 200
_MEMORY_ITEM_OUTPUT_CAP = 500

# Signal phrases that trigger an auto-remember of the user's utterance.
_AUTO_REMEMBER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bremember\b", re.IGNORECASE),
    re.compile(r"^\s*note\s*:", re.IGNORECASE),
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
)

# ---------------------------------------------------------------------------
# Hard context bounds
# ---------------------------------------------------------------------------
_COMPACT_THRESHOLD = 140_000
_TOOL_RESULT_MAX = 8_000
_KEEP_RECENT = 10

# ---------------------------------------------------------------------------
# Session persistence schema version
# ---------------------------------------------------------------------------
_SESSION_SCHEMA_VERSION = "1"

# ---------------------------------------------------------------------------
# Tool execution bounds
# ---------------------------------------------------------------------------
_DEFAULT_TOOL_TIMEOUT_S = 30.0
_TOOL_TIMEOUTS_S: dict[str, float] = {
    "run_analysis": 60.0,
    "ingest_source": 120.0,
    "dispatch_cli": 60.0,
}

# Shell metacharacters forbidden in dispatch_cli args — we never use shell=True
# but we reject them defensively so operators cannot social-engineer the model
# into requesting an expansion that carl itself would interpret.
_FORBIDDEN_SHELL_METAS: tuple[str, ...] = (
    ";", "&", "|", ">", "<", "`", "$(", "\n", "\r",
)

# Conservative max output tokens assumed for the budget pre-check.
_BUDGET_PRECHECK_MAX_OUTPUT_TOKENS = 64000

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


def _estimate_turn_upper_bound_cost(model: str, max_output_tokens: int) -> float:
    """Conservative USD cost ceiling for a turn using `max_output_tokens`.

    This assumes the model emits the maximum possible output. Input tokens are
    excluded from the pre-check because they are already counted once the
    turn lands, and overestimating too aggressively would make any non-trivial
    budget reject every turn on the first try. The output cap is the dominant
    runaway vector; that is what we gate on.
    """
    _input_rate, output_rate = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return max(0, max_output_tokens) * output_rate / 1_000_000


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
_QUARANTINE_SUBDIR = ".quarantine"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _quarantine_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class SessionStore:
    """Save and resume CARLAgent sessions at ``~/.carl/sessions/``.

    Corrupted session files (bad JSON, schema mismatch, or validation error)
    are moved to ``~/.carl/sessions/.quarantine/<id>-<timestamp>.json`` and
    :meth:`load` returns ``None`` so callers fall through to a fresh session.
    """

    def __init__(self, sessions_dir: Path | str | None = None) -> None:
        self._dir = Path(sessions_dir) if sessions_dir else _SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._quarantine_dir = self._dir / _QUARANTINE_SUBDIR

    # -- helpers -----------------------------------------------------------

    def _quarantine_path(self, session_id: str, path: Path) -> Path:
        """Move ``path`` into the quarantine directory. Returns the destination.

        Never raises — quarantine is best-effort. On OS failure we log a
        warning and the caller still sees a None-load result.
        """
        try:
            self._quarantine_dir.mkdir(parents=True, exist_ok=True)
            dest = self._quarantine_dir / f"{session_id}-{_quarantine_stamp()}.json"
            shutil.move(str(path), str(dest))
            logger.warning("Quarantined corrupted session %s -> %s", session_id, dest)
            return dest
        except OSError as exc:
            logger.warning("Failed to quarantine %s: %s", path, exc)
            return path

    # -- public API --------------------------------------------------------

    def save(self, session_id: str, state: dict[str, Any]) -> Path:
        """Persist session state to JSON."""
        state["updated_at"] = _now_iso()
        if "created_at" not in state:
            state["created_at"] = state["updated_at"]
        state["schema_version"] = _SESSION_SCHEMA_VERSION
        path = self._dir / f"{session_id}.json"
        # Serialize sets in knowledge entries
        knowledge = state.get("knowledge", [])
        serializable_knowledge: list[dict[str, Any]] = []
        for entry in knowledge:
            entry_copy = dict(entry)
            if "words" in entry_copy and isinstance(entry_copy["words"], set):
                entry_copy["words"] = sorted(entry_copy["words"])
            serializable_knowledge.append(entry_copy)
        state["knowledge"] = serializable_knowledge
        path.write_text(json.dumps(state, indent=2, default=str))
        return path

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load session state from JSON.

        Returns ``None`` when the file is missing, malformed, or of an
        unsupported schema version. Corrupted files are moved to the
        quarantine subdirectory.
        """
        path = self._dir / f"{session_id}.json"
        if not path.is_file():
            return None

        try:
            raw = path.read_text()
        except OSError as exc:
            logger.warning("Could not read session %s: %s", session_id, exc)
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Session %s has invalid JSON: %s", session_id, exc)
            self._quarantine_path(session_id, path)
            return None
        except Exception as exc:
            logger.warning("Session %s load failed: %s", session_id, exc)
            self._quarantine_path(session_id, path)
            return None

        if not isinstance(data, dict):
            logger.warning("Session %s is not a JSON object", session_id)
            self._quarantine_path(session_id, path)
            return None

        schema_version = data.get("schema_version")
        # Unstamped (legacy) sessions default to schema 1 for forward-compat.
        if schema_version is not None and schema_version != _SESSION_SCHEMA_VERSION:
            logger.warning(
                "Session %s has schema_version=%s; expected %s — quarantining",
                session_id, schema_version, _SESSION_SCHEMA_VERSION,
            )
            self._quarantine_path(session_id, path)
            return None

        try:
            # Deserialize word lists back to sets
            for entry in data.get("knowledge", []):
                if isinstance(entry, dict) and "words" in entry and isinstance(entry["words"], list):
                    entry["words"] = set(entry["words"])
        except (TypeError, PydanticValidationError) as exc:
            logger.warning("Session %s knowledge deserialize failed: %s", session_id, exc)
            self._quarantine_path(session_id, path)
            return None

        return data

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List sessions, most recent first. Skips quarantined / corrupt files."""
        sessions: list[dict[str, Any]] = []
        try:
            candidates = sorted(
                self._dir.glob("*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
        except OSError as exc:
            logger.warning("Could not enumerate sessions: %s", exc)
            return []

        for p in candidates:
            # Skip entries inside the quarantine directory. Path.glob("*.json")
            # on the top-level session dir never recurses, but guard anyway.
            try:
                if _QUARANTINE_SUBDIR in p.parts:
                    continue
            except Exception:
                continue
            if len(sessions) >= limit:
                break
            try:
                data = json.loads(p.read_text())
                if not isinstance(data, dict):
                    continue
                sessions.append({
                    "id": p.stem,
                    "title": data.get("title", ""),
                    "model": data.get("model", ""),
                    "turn_count": data.get("turn_count", 0),
                    "total_cost_usd": data.get("total_cost_usd", 0.0),
                    "updated_at": data.get("updated_at", ""),
                })
            except (json.JSONDecodeError, OSError):
                # Don't fail the whole list because one file is bad.
                continue
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
    {
        "name": "dispatch_cli",
        "description": (
            "Run a carl CLI subcommand (e.g. 'doctor', 'train', 'eval'). "
            "Only commands registered in the ops registry are allowed. "
            "Returns stdout/stderr + exit code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Top-level ops name (doctor, train, eval, ...)",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "timeout_s": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 600,
                    "default": 60,
                },
            },
            "required": ["command"],
        },
    },
]


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


class AgentEvent(BaseModel):
    """Event emitted by the agent loop for the caller to render.

    ``code`` carries a stable programmatic tag (e.g. ``carl.stream_error``,
    ``carl.budget``) for error events and empty for everything else.
    """

    kind: str  # "text", "tool_call", "tool_result", "done", "error"
    content: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    code: str = ""


# ---------------------------------------------------------------------------
# Arg schema validation
# ---------------------------------------------------------------------------


def _validate_tool_args(
    tool_name: str,
    tool_args: dict[str, Any],
) -> str | None:
    """Return None if args look valid; else a human-readable error string.

    Uses jsonschema when available for a thorough check. Falls back to a
    manual required-fields-and-type check when jsonschema is not installed.
    """
    schema = None
    for tool in TOOLS:
        if tool.get("name") == tool_name:
            schema = tool.get("input_schema")
            break
    if schema is None:
        return None  # unknown tool falls through to the dispatch fallback

    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("jsonschema not available; using manual validation for %s", tool_name)
        return _manual_schema_check(tool_name, schema, tool_args)

    try:
        jsonschema.validate(instance=tool_args, schema=schema)
    except jsonschema.ValidationError as exc:
        # Path points to the offending field; keep the message model-friendly.
        field_path = ".".join(str(p) for p in exc.absolute_path) or "<root>"
        return (
            f"Invalid arguments for '{tool_name}': at '{field_path}': {exc.message}. "
            f"Re-send with corrected arguments."
        )
    except Exception as exc:  # pragma: no cover — defensive
        return f"Schema validation error for '{tool_name}': {exc}"
    return None


def _manual_schema_check(
    tool_name: str,
    schema: dict[str, Any],
    tool_args: dict[str, Any],
) -> str | None:
    """Minimal required-field + type validation when jsonschema is unavailable."""
    if not isinstance(tool_args, dict):
        return f"Invalid arguments for '{tool_name}': expected object, got {type(tool_args).__name__}."

    required = schema.get("required", []) or []
    missing = [f for f in required if f not in tool_args]
    if missing:
        return (
            f"Invalid arguments for '{tool_name}': missing required fields "
            f"{missing}. Re-send with all required fields."
        )

    properties: dict[str, Any] = schema.get("properties") or {}
    type_map: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list, tuple),
        "object": (dict,),
    }
    bad: list[str] = []
    for fname, spec in properties.items():
        if fname not in tool_args:
            continue
        expected = spec.get("type") if isinstance(spec, dict) else None
        if expected is None:
            continue
        allowed = type_map.get(expected)
        if allowed is None:
            continue
        value = tool_args[fname]
        # bool is a subclass of int — keep the distinction.
        if expected == "integer" and isinstance(value, bool):
            bad.append(f"{fname}: expected integer, got boolean")
        elif expected == "boolean" and not isinstance(value, bool):
            bad.append(f"{fname}: expected boolean, got {type(value).__name__}")
        elif not isinstance(value, allowed):
            bad.append(f"{fname}: expected {expected}, got {type(value).__name__}")
    if bad:
        return (
            f"Invalid arguments for '{tool_name}': "
            + "; ".join(bad)
            + ". Re-send with corrected types."
        )
    return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CARLAgent:
    """Agentic chat loop with tool use, frame awareness, and proactive guidance.

    Features beyond raw API relay:
      - **Session persistence**: save/resume via SessionStore
      - **Permission hooks**: pre_tool_use callback can deny tool calls
      - **Cost tracking**: per-turn cost from response.usage, max_budget_usd cap
      - **Streaming resilience**: partial state preserved on mid-stream errors
      - **Tool timeouts**: per-tool deadlines; hook exceptions never kill loop
      - **Arg validation**: malformed tool args become typed error results
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
        memory_store: MemoryStore | None = None,
        constitution: Constitution | None = None,
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

        # Interaction chain — lazily constructed by dispatch_cli.
        self._chain: Any | None = None

        # Tier — resolved lazily from settings; FREE by default for safety.
        # Use a distinct sentinel so a failed resolution doesn't re-run each call.
        self._tier: Any | None = None
        self._tier_resolved: bool = False

        # Project context (cached at construction)
        self._project_line = ""
        try:
            from carl_studio.project import load_project

            proj = load_project("carl.yaml")
            self._project_line = f"PROJECT: {proj.name} | Model: {proj.base_model} | Method: {proj.method}"
        except Exception:
            pass

        # -- Constitution (CRYSTAL layer) --------------------------------
        # Attempt to resolve a Constitution instance. If the caller passed
        # one, trust it verbatim; otherwise try the default loader. Failures
        # must never break agent construction — the agent degrades to the
        # legacy behaviour (no constitutional rules injected).
        self._constitution: Constitution | None = constitution
        if self._constitution is None:
            try:
                from carl_studio.constitution import Constitution as _Constitution

                self._constitution = _Constitution.load()
            except Exception as exc:
                logger.debug("Constitution.load() failed; continuing without: %s", exc)
                self._constitution = None

        # Compiled system-prompt block — cached per-instance so a noisy
        # large ruleset doesn't re-serialize on every turn.
        self._constitution_prompt: str | None = None

        # -- Memory store (WORKING + LONG + SHORT/CRYSTAL) ---------------
        self._memory: MemoryStore | None = memory_store
        if self._memory is None:
            try:
                from carl_core.memory import MemoryStore as _MemoryStore

                self._memory = _MemoryStore(Path.home() / ".carl" / "memory")
            except Exception as exc:
                logger.debug("MemoryStore bootstrap failed; continuing without: %s", exc)
                self._memory = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_chain(self) -> Any:
        """Return the InteractionChain for this agent session, creating if needed."""
        if self._chain is not None:
            return self._chain
        try:
            from carl_core.interaction import InteractionChain

            self._chain = InteractionChain()
        except Exception:  # pragma: no cover — optional
            self._chain = None
        return self._chain

    def _resolve_tier(self) -> Any:
        """Resolve effective tier once, cached on the instance."""
        if self._tier_resolved:
            return self._tier
        try:
            from carl_core.tier import Tier

            try:
                from carl_studio.tier import detect_effective_tier
                from carl_studio.settings import CARLSettings

                settings = CARLSettings.load()
                configured = getattr(settings, "tier", None)
                if isinstance(configured, Tier):
                    self._tier = detect_effective_tier(configured)
                elif isinstance(configured, str):
                    try:
                        self._tier = detect_effective_tier(Tier(configured))
                    except Exception:
                        self._tier = Tier.FREE
                else:
                    self._tier = Tier.FREE
            except Exception:
                self._tier = Tier.FREE
        except Exception:
            self._tier = None
        self._tier_resolved = True
        return self._tier

    @property
    def provider_info(self) -> dict[str, str]:
        """Return provider resolution info for display."""
        return {
            "model": self._model,
            "provider": "anthropic",
            "api_key_source": self._api_key_source,
        }

    # ------------------------------------------------------------------
    # Chat loop
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> Iterator[AgentEvent]:
        """Send message, handle tool calls, yield events.

        Uses streaming to prevent HTTP timeouts on long responses.
        Yields text_delta events for real-time token display.

        Resilience contract:
          * Streaming exceptions never propagate out — the generator always
            completes. Mid-stream errors yield an ``error`` event with a
            stable ``code`` attribute (e.g. ``carl.stream_error``), attribute
            partial cost for consumed tokens, and persist emitted text so
            session save captures what the user saw.
          * A budget pre-check runs before every API call. When the upper-bound
            estimate would push total cost above ``max_budget_usd``, a
            ``BudgetError`` event is yielded and the API is not called.
          * Hook exceptions (pre/post tool use) yield a ``carl.hook_failed``
            error event and the loop continues.
        """
        # Memory recall — prepend relevant past learnings to the user turn,
        # and emit MEMORY_READ steps into the InteractionChain. Recall is
        # best-effort: any failure leaves the original prompt unchanged.
        recalled_items = self._recall_memories(user_input)
        augmented_input = self._augment_prompt_with_recall(user_input, recalled_items)

        # Auto-remember strong signals ("always", "never", "remember", "note:")
        # BEFORE the API call so the new SHORT memory is visible to the next
        # recall pass — this is the crystallize-in-place pattern.
        self._maybe_auto_remember(user_input)

        self._messages.append({"role": "user", "content": augmented_input})
        self._turn_count += 1

        while True:
            # ------- Budget pre-check (upper-bound ceiling) -----------------
            if self._max_budget_usd > 0:
                est = _estimate_turn_upper_bound_cost(
                    self._model, _BUDGET_PRECHECK_MAX_OUTPUT_TOKENS,
                )
                projected = self._total_cost_usd + est
                if projected > self._max_budget_usd:
                    msg = (
                        f"Budget pre-check blocked turn: projected "
                        f"${projected:.4f} (current ${self._total_cost_usd:.4f} + "
                        f"estimated ${est:.4f}) exceeds cap ${self._max_budget_usd:.4f}"
                    )
                    code = "carl.budget"
                    # Best-effort carl_core.errors.BudgetError marker in logs.
                    try:
                        from carl_core.errors import BudgetError

                        logger.warning(
                            "BudgetError: %s",
                            BudgetError(msg, context={
                                "current": self._total_cost_usd,
                                "estimated": est,
                                "cap": self._max_budget_usd,
                            }),
                        )
                    except Exception:  # pragma: no cover — carl_core optional
                        pass
                    yield AgentEvent(kind="error", content=msg, code=code)
                    return
                if self._total_cost_usd >= self._max_budget_usd:
                    # Hard floor — we already spent the whole budget.
                    yield AgentEvent(
                        kind="error",
                        content=(
                            f"Budget exceeded: ${self._total_cost_usd:.4f} "
                            f">= ${self._max_budget_usd:.2f}"
                        ),
                        code="carl.budget",
                    )
                    return

            system = self._build_system_prompt()

            response: Any = None
            partial_text_parts: list[str] = []
            stream: Any = None
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "max_tokens": _BUDGET_PRECHECK_MAX_OUTPUT_TOKENS,
                    "system": system,
                    "messages": self._messages,
                    "tools": TOOLS,
                    "cache_control": {"type": "ephemeral"},
                }
                if any(fam in self._model for fam in ("opus-4-6", "sonnet-4-6")):
                    kwargs["thinking"] = {"type": "adaptive"}

                try:
                    stream_cm = self._client.messages.stream(**kwargs)
                except Exception as exc:
                    # API call could not even be issued.
                    yield AgentEvent(
                        kind="error",
                        content=f"API error: {exc}",
                        code="carl.api_error",
                    )
                    return

                with stream_cm as stream:
                    try:
                        for event in stream:
                            if event.type == "content_block_delta":
                                if event.delta.type == "text_delta":
                                    partial_text_parts.append(event.delta.text)
                                    yield AgentEvent(
                                        kind="text_delta",
                                        content=event.delta.text,
                                    )
                    except (KeyboardInterrupt, BaseException) as exc:
                        # Baseexception catches KeyboardInterrupt/SystemExit.
                        # We still want to best-effort finalize + persist state.
                        code = (
                            "carl.interrupted"
                            if isinstance(exc, KeyboardInterrupt)
                            else "carl.stream_error"
                        )
                        response = self._safe_finalize_stream(stream)
                        self._attribute_partial_cost(response, partial_text_parts)
                        self._persist_partial_turn(response, partial_text_parts)
                        yield AgentEvent(
                            kind="error",
                            content=f"Stream interrupted: {exc}",
                            code=code,
                        )
                        return

                    # Clean exit — collect the final message.
                    try:
                        response = stream.get_final_message()
                    except Exception as exc:
                        self._attribute_partial_cost(None, partial_text_parts)
                        self._persist_partial_turn(None, partial_text_parts)
                        yield AgentEvent(
                            kind="error",
                            content=f"Failed to finalize stream: {exc}",
                            code="carl.stream_error",
                        )
                        return
            except BaseException as exc:
                # Catchall for exceptions while entering/exiting the ctx-mgr.
                try:
                    response = self._safe_finalize_stream(stream)
                except Exception:
                    response = None
                self._attribute_partial_cost(response, partial_text_parts)
                self._persist_partial_turn(response, partial_text_parts)
                code = (
                    "carl.interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "carl.stream_error"
                )
                yield AgentEvent(
                    kind="error",
                    content=f"Stream error: {exc}",
                    code=code,
                )
                return

            # Track cost (full accounting path on clean turns).
            usage = getattr(response, "usage", None)
            if usage is not None:
                turn_cost = _compute_turn_cost(usage, self._model)
                self._total_cost_usd += turn_cost
                self._total_input_tokens += getattr(usage, "input_tokens", 0) or 0
                self._total_output_tokens += getattr(usage, "output_tokens", 0) or 0

            self._token_count = (
                getattr(usage, "input_tokens", self._token_count)
                if usage else self._token_count
            )

            # Context pressure
            if self._token_count > _COMPACT_THRESHOLD:
                self._compact()

            # pause_turn: server-side tool iteration limit
            if getattr(response, "stop_reason", None) == "pause_turn":
                self._messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": self._content_to_dicts(response.content)},
                ]
                continue

            # Collect blocks
            assistant_content: list[dict[str, Any]] = []
            tool_use_blocks: list[Any] = []

            for block in getattr(response, "content", []) or []:
                btype = getattr(block, "type", None)
                if btype == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif btype == "thinking":
                    pass
                elif btype == "tool_use":
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

                # Pre-tool hook — never let an exception kill the loop.
                permission: ToolPermission = ToolPermission.ALLOW
                if self._pre_tool_use is not None:
                    try:
                        permission = self._pre_tool_use(block.name, tool_input)
                    except Exception as exc:
                        logger.warning(
                            "pre_tool_use hook failed for %s: %s", block.name, exc,
                        )
                        yield AgentEvent(
                            kind="error",
                            content=f"pre_tool_use hook failed for {block.name}: {exc}",
                            code="carl.hook_failed",
                        )
                        permission = ToolPermission.ALLOW

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

                # Validate args against the tool schema.
                schema_err = _validate_tool_args(block.name, tool_input)
                if schema_err is not None:
                    yield AgentEvent(
                        kind="tool_result",
                        tool_name=block.name,
                        content=schema_err,
                        code="carl.tool_schema",
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": schema_err,
                        "is_error": True,
                    })
                    continue

                # Dispatch — dispatch_tool already contains the timeout wrapper
                # and its own failure -> is_error policy.
                result, is_error = self._dispatch_tool_safe(block.name, tool_input)
                yield AgentEvent(kind="tool_result", tool_name=block.name, content=result)
                tool_result_entry: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
                if is_error:
                    tool_result_entry["is_error"] = True
                tool_results.append(tool_result_entry)

                # Post-tool hook — isolated from loop lifetime.
                if self._post_tool_use is not None:
                    try:
                        self._post_tool_use(block.name, tool_input, result)
                    except Exception as exc:
                        logger.warning(
                            "post_tool_use hook failed for %s: %s", block.name, exc,
                        )
                        yield AgentEvent(
                            kind="error",
                            content=(
                                f"post_tool_use hook failed for {block.name}: {exc}"
                            ),
                            code="carl.hook_failed",
                        )

            self._messages.append({"role": "user", "content": tool_results})

            if getattr(response, "stop_reason", None) == "end_turn":
                yield AgentEvent(kind="done", content=f"${self._total_cost_usd:.4f}")
                return

    # ------------------------------------------------------------------
    # Memory: recall, remember, auto-remember, suggest_learnings
    # ------------------------------------------------------------------

    def _recall_memories(self, prompt: str) -> list[MemoryItem]:
        """Return recalled WORKING+LONG memory items or []; never raises.

        Emits a MEMORY_READ step into the InteractionChain for each item
        surfaced. Items below ``_MEMORY_RECALL_MIN_SCORE`` are dropped by the
        MemoryStore before they reach us.
        """
        if self._memory is None:
            return []
        try:
            from carl_core.memory import MemoryLayer as _MemoryLayer

            recalled = self._memory.recall(
                prompt,
                layers={_MemoryLayer.WORKING, _MemoryLayer.LONG},
                top_k=_MEMORY_RECALL_TOP_K,
                min_score=_MEMORY_RECALL_MIN_SCORE,
            )
        except Exception as exc:
            logger.debug("memory.recall failed: %s", exc)
            return []

        if not recalled:
            return []

        # Emit MEMORY_READ steps — failure to record must not kill the turn.
        chain = self._get_chain()
        if chain is not None:
            try:
                from carl_core.interaction import ActionType

                for item in recalled:
                    chain.record(
                        ActionType.MEMORY_READ,
                        name=item.id,
                        input=prompt,
                        output=item.content[:_MEMORY_ITEM_OUTPUT_CAP],
                    )
            except Exception as exc:
                logger.debug("chain.record MEMORY_READ failed: %s", exc)
        return recalled

    @staticmethod
    def _augment_prompt_with_recall(
        prompt: str, recalled: list[MemoryItem],
    ) -> str:
        """Prepend a visible recall block to the user's turn, or passthrough."""
        if not recalled:
            return prompt
        note_lines = ["# Recalled context (from past sessions):"]
        for item in recalled:
            preview = item.content[:_MEMORY_CONTENT_PREVIEW]
            note_lines.append(f"- [{item.id}] {preview}")
        note = "\n".join(note_lines)
        return f"{note}\n\n---\n\n{prompt}"

    def remember(
        self,
        content: str,
        *,
        layer: MemoryLayer | None = None,
        tags: set[str] | None = None,
    ) -> MemoryItem | None:
        """Write ``content`` to memory and record a MEMORY_WRITE step.

        ``layer`` defaults to ``MemoryLayer.SHORT`` when None. Returns the
        freshly written ``MemoryItem`` or ``None`` when the store is absent.
        """
        if self._memory is None:
            return None
        try:
            from carl_core.memory import MemoryLayer as _MemoryLayer
        except Exception as exc:
            logger.debug("memory import failed in remember(): %s", exc)
            return None

        target_layer = layer if layer is not None else _MemoryLayer.SHORT
        try:
            item = self._memory.write(
                content, layer=target_layer, tags=tags or set(),
            )
        except Exception as exc:
            logger.warning("memory.write failed: %s", exc)
            return None

        chain = self._get_chain()
        if chain is not None:
            try:
                from carl_core.interaction import ActionType

                chain.record(
                    ActionType.MEMORY_WRITE,
                    name=item.id,
                    input={"layer": target_layer.name, "tags": sorted(tags or set())},
                    output=content[:_MEMORY_ITEM_OUTPUT_CAP],
                )
            except Exception as exc:
                logger.debug("chain.record MEMORY_WRITE failed: %s", exc)
        return item

    def _maybe_auto_remember(self, user_input: str) -> None:
        """If the user's utterance contains a learn-signal, persist it.

        Looks for: ``remember``, ``note:``, ``always``, ``never``. Tags come
        from the active frame's domain/function/role so recall surfaces them
        under relevant topics.
        """
        if self._memory is None:
            return
        if not any(pat.search(user_input) for pat in _AUTO_REMEMBER_PATTERNS):
            return
        tags = self._derive_topics_from_frame()
        tags.add("auto_remember")
        self.remember(user_input, tags=tags)

    def suggest_learnings(self) -> list[ConstitutionalRule]:
        """Ask the agent to propose 1-3 durable rules from recent turns.

        Zero-tool sub-invocation. Reuses the same model with a reduced
        output budget. Returns [] on malformed output so callers can always
        treat the result as a list.
        """
        try:
            from carl_studio.constitution import ConstitutionalRule
        except Exception as exc:
            logger.debug("ConstitutionalRule import failed: %s", exc)
            return []

        # Gather the last few user+assistant exchanges as a digest. Tool
        # results and auto-injected recall blocks are de-emphasized.
        digest = self._recent_turns_digest(max_turns=6)
        if not digest:
            return []

        suggest_prompt = (
            "Review the recent conversation turns below. Propose 1-3 durable "
            "rules to codify as learnings for future sessions. Respond with "
            "ONLY a JSON array. Each element must be an object with keys: "
            "id (dot-separated string), text (natural-language rule), "
            "priority (integer 0-100), tags (array of strings).\n\n"
            f"Recent turns:\n{digest}"
        )

        raw_json = self._one_shot_text(
            suggest_prompt, max_tokens=1024,
        )
        if not raw_json:
            return []

        return self._parse_learnings_json(raw_json, ConstitutionalRule)

    def _recent_turns_digest(self, *, max_turns: int) -> str:
        """Compact textual digest of the last ``max_turns`` messages."""
        if not self._messages:
            return ""
        window = self._messages[-max_turns:]
        lines: list[str] = []
        for msg in window:
            role = str(msg.get("role", "?"))
            content = msg.get("content", "")
            if isinstance(content, list):
                bits: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        bits.append(str(block.get("text", "")))
                    elif block.get("type") == "tool_use":
                        bits.append(f"[tool_use:{block.get('name','?')}]")
                text = " ".join(b for b in bits if b).strip()
            else:
                text = str(content).strip()
            if text:
                lines.append(f"{role}: {text[:500]}")
        return "\n".join(lines)

    def _one_shot_text(self, prompt: str, *, max_tokens: int) -> str:
        """Zero-tool synchronous call — returns the assistant text or ""."""
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": max(1, int(max_tokens)),
                "messages": [{"role": "user", "content": prompt}],
            }
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            logger.debug("one-shot suggest_learnings call failed: %s", exc)
            return ""

        # Best-effort text extraction across SDK shapes.
        content = getattr(response, "content", None) or []
        texts: list[str] = []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_val = getattr(block, "text", "") or ""
                if text_val:
                    texts.append(str(text_val))
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
        return "".join(texts).strip()

    @staticmethod
    def _parse_learnings_json(
        raw: str,
        rule_cls: type[ConstitutionalRule],
    ) -> list[ConstitutionalRule]:
        """Parse a JSON array of learnings, skipping malformed entries."""
        text = raw.strip()
        # Try to extract the first JSON array if there's surrounding prose.
        if not text.startswith("["):
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]
            else:
                return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []

        out: list[ConstitutionalRule] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            # ConstitutionalRule requires id + text; tolerate soft fields.
            try:
                rule = rule_cls(
                    id=str(entry.get("id", "")).strip(),
                    text=str(entry.get("text", "")).strip(),
                    priority=int(entry.get("priority", 50)),
                    tags=[str(t) for t in (entry.get("tags") or []) if t],
                    source="user",
                )
            except Exception:
                continue
            if not rule.id or not rule.text:
                continue
            out.append(rule)
        return out

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def _safe_finalize_stream(self, stream: Any) -> Any:
        """Best-effort ``get_final_message()`` on a stream that raised mid-iter."""
        if stream is None:
            return None
        try:
            return stream.get_final_message()
        except Exception as exc:  # pragma: no cover — rare SDK path
            logger.debug("get_final_message failed: %s", exc)
            return None

    def _attribute_partial_cost(self, response: Any, partial_text_parts: list[str]) -> None:
        """Attribute whatever cost we can for the partial/failed turn."""
        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                cost = _compute_turn_cost(usage, self._model)
                self._total_cost_usd += cost
                self._total_input_tokens += getattr(usage, "input_tokens", 0) or 0
                self._total_output_tokens += getattr(usage, "output_tokens", 0) or 0
            except Exception as exc:
                logger.debug("cost attribution failed: %s", exc)
            return

        # No usage from the SDK — charge for what we emitted to keep budget honest.
        if partial_text_parts:
            output_tokens_est = max(1, sum(len(p) for p in partial_text_parts) // 4)
            _input_rate, output_rate = _MODEL_PRICING.get(self._model, _DEFAULT_PRICING)
            self._total_cost_usd += output_tokens_est * output_rate / 1_000_000
            self._total_output_tokens += output_tokens_est

    def _persist_partial_turn(self, response: Any, partial_text_parts: list[str]) -> None:
        """Append whatever the assistant emitted so session.save captures it."""
        # Prefer the full SDK response if available.
        if response is not None and getattr(response, "content", None):
            try:
                dicts = self._content_to_dicts(response.content)
                if dicts:
                    self._messages.append({"role": "assistant", "content": dicts})
                    return
            except Exception as exc:
                logger.debug("persist partial: content_to_dicts failed: %s", exc)

        # Fall back to reconstructed text from the delta stream.
        text = "".join(partial_text_parts).strip()
        if text:
            self._messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            })

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

        base = "\n".join(parts)

        # Constitutional rules (CRYSTAL layer) — injected at most once per
        # agent instance. A failure here must never block the turn: log and
        # return the base prompt unchanged.
        rules_block = self._get_constitution_prompt()
        if rules_block:
            return base + "\n\n" + rules_block
        return base

    def _get_constitution_prompt(self) -> str:
        """Return the cached constitutional block, compiling on first use."""
        if self._constitution is None:
            return ""
        if self._constitution_prompt is not None:
            return self._constitution_prompt
        try:
            topics = self._derive_topics_from_frame() or None
            compiled = self._constitution.compile_system_prompt(
                topics=topics, max_rules=20,
            )
        except Exception as exc:
            logger.debug("Constitution.compile_system_prompt failed: %s", exc)
            compiled = ""
        self._constitution_prompt = compiled
        return compiled

    def _derive_topics_from_frame(self) -> set[str]:
        """Extract topic tags from the active WorkFrame for rule selection.

        Returns a set built from domain/function/role. Empty set when no
        frame is set (callers treat an empty set as "all topics").
        """
        frame = self._frame
        if frame is None:
            return set()
        topics: set[str] = set()
        for attr in ("domain", "function", "role"):
            value = getattr(frame, attr, "")
            if isinstance(value, str) and value:
                topics.add(value.lower())
        return topics

    @staticmethod
    def _content_to_dicts(content: Any) -> list[dict[str, Any]]:
        """Convert SDK content blocks to dicts for message history."""
        result: list[dict[str, Any]] = []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "text":
                result.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
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
        """Public dispatch (back-compat).

        Retained for tests and callers that don't need the ``is_error`` flag.
        The return value is the raw tool output string; when a tool errors,
        the string is prefixed with ``"Error:"`` or the tool's own error text.
        """
        result, _is_error = self._dispatch_tool_safe(name, args)
        return result

    def _dispatch_tool_safe(
        self, name: str, args: dict[str, Any]
    ) -> tuple[str, bool]:
        """Run a tool with per-tool timeout + error handling.

        Returns ``(content, is_error)`` so the caller can flag the API
        ``tool_result`` as erroring so the model can self-correct.
        """
        timeout_s = _TOOL_TIMEOUTS_S.get(name, _DEFAULT_TOOL_TIMEOUT_S)

        def _run() -> tuple[str, bool]:
            try:
                if name == "ingest_source":
                    return self._tool_ingest(args.get("path", "")), False
                if name == "query_knowledge":
                    return self._tool_query(args.get("question", "")), False
                if name == "run_analysis":
                    return self._tool_run(args.get("code", "")), False
                if name == "create_file":
                    return self._tool_create(args.get("path", ""), args.get("content", "")), False
                if name == "read_file":
                    return self._tool_read(args.get("path", "")), False
                if name == "set_frame":
                    return self._tool_frame(args), False
                if name == "list_files":
                    return self._tool_list(args.get("path", "."), args.get("pattern", "*")), False
                if name == "dispatch_cli":
                    return self._tool_dispatch_cli(args)
                return f"Unknown tool: {name}", True
            except subprocess.TimeoutExpired:
                return f"Tool {name} timed out after {timeout_s:.0f}s", True
            except TimeoutError:
                return f"Tool {name} timed out after {timeout_s:.0f}s", True
            except Exception as exc:
                return f"Error: {exc}", True

        # Wrap synchronous tool invocation in a bounded worker so we can enforce
        # a wall-clock timeout even for CPU-bound tools. subprocess-based tools
        # (run_analysis) already have their own subprocess timeout, but this
        # guards everything else uniformly.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                return future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                # Cancel best-effort; the worker will continue running until
                # it naturally returns, but its result is dropped.
                future.cancel()
                return f"Tool {name} timed out after {timeout_s:.0f}s", True
            except Exception as exc:
                return f"Error: {exc}", True

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

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
        # TimeoutExpired here bubbles up to _dispatch_tool_safe which maps it
        # to a model-recoverable is_error tool_result. Do not swallow here.
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            timeout=_TOOL_TIMEOUTS_S["run_analysis"],
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
        try:
            from carl_core.safepath import PathEscape, safe_resolve

            try:
                return safe_resolve(path, self._workdir, follow_symlinks=True)
            except PathEscape:
                return None
        except Exception:
            # Fallback if carl_core is unavailable — keep legacy behavior.
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

    def _tool_dispatch_cli(self, args: dict[str, Any]) -> tuple[str, bool]:
        """Run a registered ``carl <command>`` subprocess.

        Returns ``(serialized_output, is_error)``. Records a CLI_CMD step in
        the agent's InteractionChain with the (redacted) args + exit code.
        """
        command = args.get("command", "")
        subcommand_args = args.get("args", []) or []
        timeout_s = int(args.get("timeout_s", 60) or 60)
        if not isinstance(command, str) or not command.strip():
            return self._cli_error("missing 'command' argument"), True
        if not isinstance(subcommand_args, list):
            return self._cli_error("'args' must be a list"), True
        timeout_s = max(1, min(timeout_s, 600))

        # Whitelist lookup.
        try:
            from carl_studio.cli.operations import OPERATIONS
        except Exception as exc:
            return self._cli_error(f"Ops registry unavailable: {exc}"), True

        if command not in OPERATIONS:
            allowed = ", ".join(sorted(OPERATIONS.keys()))
            return self._cli_error(
                f"Unknown carl command '{command}'. Allowed: {allowed}"
            ), True

        # Shell-metachar scan (pure string values only).
        stringified: list[str] = []
        for idx, value in enumerate(subcommand_args):
            if not isinstance(value, str):
                return self._cli_error(
                    f"args[{idx}] is not a string (got {type(value).__name__})"
                ), True
            for meta in _FORBIDDEN_SHELL_METAS:
                if meta in value:
                    return self._cli_error(
                        f"args[{idx}] contains forbidden metacharacter "
                        f"{meta!r}"
                    ), True
            stringified.append(value)

        # Tier gate — check whether the current tier permits this feature.
        tier = self._resolve_tier()
        if tier is not None:
            try:
                from carl_core.tier import feature_tier, tier_allows

                required = feature_tier(command)
                if not tier_allows(tier, command):
                    msg = (
                        f"Command '{command}' requires tier "
                        f"{required.value!r}; current tier is {tier.value!r}."
                    )
                    chain = self._get_chain()
                    if chain is not None:
                        try:
                            from carl_core.interaction import ActionType

                            chain.record(
                                ActionType.CLI_CMD,
                                f"dispatch_cli:{command}",
                                input={
                                    "command": command,
                                    "args": stringified,
                                },
                                output={"tier_block": tier.value, "required": required.value},
                                success=False,
                            )
                        except Exception:
                            pass
                    return self._cli_error(msg, code="carl.permission"), True
            except Exception:
                # Tier module not available — do not gate.
                pass

        cmd = ["carl", command, *stringified]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=self._workdir,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            chain = self._get_chain()
            if chain is not None:
                try:
                    from carl_core.interaction import ActionType

                    chain.record(
                        ActionType.CLI_CMD,
                        f"dispatch_cli:{command}",
                        input={"command": command, "args": stringified},
                        output={"error": "timeout", "timeout_s": timeout_s},
                        success=False,
                    )
                except Exception:
                    pass
            return self._cli_error(
                f"carl {command} timed out after {timeout_s}s"
            ), True
        except FileNotFoundError:
            return self._cli_error(
                "'carl' binary not found on PATH — is carl-studio installed?"
            ), True
        except Exception as exc:
            return self._cli_error(f"subprocess failed: {exc}"), True

        stdout = (proc.stdout or "")[:6000]
        stderr = (proc.stderr or "")[:2000]
        exit_code = int(proc.returncode)
        payload = {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}

        chain = self._get_chain()
        if chain is not None:
            try:
                from carl_core.interaction import ActionType

                chain.record(
                    ActionType.CLI_CMD,
                    f"dispatch_cli:{command}",
                    input={"command": command, "args": stringified, "timeout_s": timeout_s},
                    output={"exit_code": exit_code, "stdout_len": len(stdout), "stderr_len": len(stderr)},
                    success=(exit_code == 0),
                )
            except Exception:
                pass

        return json.dumps(payload), exit_code != 0

    def _cli_error(self, message: str, *, code: str = "carl.validation") -> str:
        """Serialize a dispatch_cli error as JSON the model can parse."""
        return json.dumps({"error": message, "code": code})

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
            try:
                from carl_studio.frame import WorkFrame

                self._frame = WorkFrame.model_validate(frame_data)
            except Exception as exc:
                logger.warning("Could not rebuild WorkFrame from session: %s", exc)

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
