"""Tool dispatch mechanics for :class:`carl_studio.chat_agent.CARLAgent`.

Extracted from ``chat_agent.py`` in the v0.7 Phase-2 god-class decomposition
(ticket G1 Phase-2 / F-PYTH-002). This module owns:

* the tool registry (name -> callable),
* per-tool timeouts with a bounded worker,
* the ``(content, is_error)`` dispatch contract — so the caller can flag
  the API ``tool_result`` as erroring and let the model self-correct,
* the "all-denied consecutive turn" counter + terminal limit that stops a
  misbehaving model+hook combination from looping forever,
* (v0.14) :meth:`ToolDispatcher.execute_block` — full per-block lifecycle
  (pre-hook → validate → dispatch → post-hook → outcome). Replaces the
  ad-hoc inline loop body in ``chat_agent.py`` with a single delegatable
  call so the agent's tool-use loop shrinks to its structural minimum.

The agent still owns domain-specific tool implementations (``_tool_run``,
``_tool_ingest``, ``_tool_frame``, ...). The dispatcher receives them as
partials at construction time and provides ONLY the mechanics: lookup,
timeout, exception mapping, denial counting.

Backward-compat contract
------------------------

Legacy tests (``tests/test_chat_agent_robustness.py``, ``tests/test_agent.py``)
reach into these private attrs on the agent:

* ``agent._dispatch_tool(name, args) -> str``
* ``agent._dispatch_tool_safe(name, args) -> (str, bool)``
* ``agent._consecutive_all_denied`` — read AND write
* ``agent._MAX_CONSECUTIVE_ALL_DENIED`` — read as a class constant

:class:`CARLAgent` exposes ``@property`` shims that delegate to the
dispatcher so every assertion in those tests still passes without edits.
"""

from __future__ import annotations

import concurrent.futures
import enum
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# v0.14: ToolOutcome + execute_block
# ---------------------------------------------------------------------------


class ToolPermission(str, enum.Enum):
    """Pre-hook verdict on whether a tool call proceeds."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class ToolOutcome:
    """Structured outcome of one tool-block execution.

    Consumed by the agent's main loop to build the API ``tool_result``
    block, update counters, and record an :class:`InteractionChain` step.
    """

    tool_use_id: str
    name: str
    input: dict[str, Any]
    result: str
    is_error: bool
    outcome: str  # "ok" | "denied" | "schema_error" | "error"
    duration_ms: float


@dataclass(frozen=True)
class ToolEvent:
    """One agent-visible event produced during execute_block.

    The agent's generator loop maps these to ``AgentEvent`` yields so the
    streaming contract (tool_blocked / tool_result / error) is preserved.
    """

    kind: str  # "tool_blocked" | "tool_result" | "error"
    name: str
    content: str
    code: str | None = None


PreToolHookT = Callable[[str, dict[str, Any]], ToolPermission]
PostToolHookT = Callable[[str, dict[str, Any], str], None]
ValidatorT = Callable[[str, dict[str, Any]], str | None]

# Default wall-clock timeout applied to any tool without a dedicated entry.
_DEFAULT_TOOL_TIMEOUT_S: float = 30.0

# Class-level default the agent mirrors as ``_MAX_CONSECUTIVE_ALL_DENIED``.
_MAX_CONSECUTIVE_ALL_DENIED: int = 5

# Callable signature a registered tool must satisfy:
#   dict[str, Any] -> tuple[str, bool]
# The bool is the is_error flag. Tools that never error can return
# ``(output, False)``; tools that can error (e.g. dispatch_cli) return the
# pre-computed flag. Legacy tools that returned plain strings get wrapped
# by :meth:`ToolDispatcher.register_simple`.
ToolCallable = Callable[[dict[str, Any]], tuple[str, bool]]


TimeoutResolver = Callable[[str], float]


@dataclass
class ToolDispatcher:
    """Registry + dispatch loop for ``CARLAgent`` tools.

    Parameters
    ----------
    tools:
        Map of tool name to ``ToolCallable``. Populated via
        :meth:`register` / :meth:`register_simple` during agent construction
        so each callable can close over ``self``.
    timeouts:
        Per-tool wall-clock limit in seconds. Missing entries fall back to
        :data:`_DEFAULT_TOOL_TIMEOUT_S`. Ignored when ``timeout_resolver``
        is set — the resolver is the single source of truth for timeouts.
    timeout_resolver:
        Optional callable ``(name) -> seconds``. Lets callers swap the
        underlying timeout source at runtime (e.g. the agent points this
        at the module-level ``_TOOL_TIMEOUTS_S`` constant so
        ``monkeypatch.setattr`` on that constant flows through to the
        live dispatcher without rebuilding it).
    default_timeout_s:
        Timeout applied when ``timeouts`` has no entry for a tool name.
    max_consecutive_all_denied:
        Terminal bound on the number of consecutive turns where every
        attempted tool call was ``DENY``-ed by the permission hook. Hitting
        this triggers a ``carl.all_tools_denied`` terminal — the agent
        checks this via :meth:`all_denied_limit_hit`.
    consecutive_all_denied:
        The running counter; the agent's ``_consecutive_all_denied``
        property delegates here.
    """

    tools: dict[str, ToolCallable] = field(
        default_factory=lambda: {},  # type: dict[str, ToolCallable]
    )
    timeouts: dict[str, float] = field(
        default_factory=lambda: {},  # type: dict[str, float]
    )
    timeout_resolver: TimeoutResolver | None = None
    default_timeout_s: float = _DEFAULT_TOOL_TIMEOUT_S
    max_consecutive_all_denied: int = _MAX_CONSECUTIVE_ALL_DENIED
    consecutive_all_denied: int = 0

    def _resolve_timeout(self, name: str) -> float:
        """Return the effective timeout for ``name``.

        Priority: ``timeout_resolver`` (if set) > ``timeouts`` map >
        ``default_timeout_s``. The resolver is consulted per-dispatch so
        test harnesses can swap the underlying timeout table between calls.
        """
        if self.timeout_resolver is not None:
            try:
                return float(self.timeout_resolver(name))
            except Exception:  # pragma: no cover — defensive
                pass
        return self.timeouts.get(name, self.default_timeout_s)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, fn: ToolCallable) -> None:
        """Register a tool that returns ``(content, is_error)`` directly."""
        if not name or not callable(fn):
            raise ValueError(
                f"ToolDispatcher.register: name={name!r}, fn callable? "
                f"{callable(fn)}",
            )
        self.tools[name] = fn

    def register_simple(
        self, name: str, fn: Callable[[dict[str, Any]], str],
    ) -> None:
        """Register a tool that returns a plain string; auto-wraps as success.

        Most of ``CARLAgent``'s tools (``_tool_read``, ``_tool_create``,
        ``_tool_frame``, ``_tool_list``, ``_tool_query``, ``_tool_ingest``,
        ``_tool_run``) return a string and rely on an exception for the
        error path. This shim rescues the legacy contract without
        forcing every implementation to return the tuple directly.
        """
        def _wrapped(args: dict[str, Any]) -> tuple[str, bool]:
            return fn(args), False
        self.register(name, _wrapped)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Back-compat shim: dispatch and return just the content string.

        Retained for tests that don't need the ``is_error`` flag.
        """
        content, _ = self.dispatch_safe(name, args)
        return content

    def dispatch_safe(
        self, name: str, args: dict[str, Any],
    ) -> tuple[str, bool]:
        """Dispatch a tool call with timeout + error mapping.

        Returns ``(content, is_error)`` so the caller can flag the API
        ``tool_result`` block as erroring so the model can self-correct.
        Exceptions raised by the tool become ``("Error: ...", True)``.
        A ``subprocess.TimeoutExpired`` or ``TimeoutError`` becomes the
        canonical ``"Tool {name} timed out after {N}s"`` message so tests
        can assert against a stable string.
        """
        timeout_s = self._resolve_timeout(name)

        def _run() -> tuple[str, bool]:
            fn = self.tools.get(name)
            if fn is None:
                return f"Unknown tool: {name}", True
            try:
                return fn(args)
            except subprocess.TimeoutExpired:
                return f"Tool {name} timed out after {timeout_s:.0f}s", True
            except TimeoutError:
                return f"Tool {name} timed out after {timeout_s:.0f}s", True
            except Exception as exc:
                return f"Error: {exc}", True

        # Wrap synchronous tool invocation in a bounded worker so we can
        # enforce a wall-clock timeout even for CPU-bound tools. Tools that
        # spawn subprocesses (run_analysis, dispatch_cli) have their own
        # subprocess timeout; this guards everything else uniformly.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                return future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                # Cancel best-effort; the worker will continue running
                # until it naturally returns, but its result is dropped.
                future.cancel()
                return f"Tool {name} timed out after {timeout_s:.0f}s", True
            except Exception as exc:
                return f"Error: {exc}", True

    # ------------------------------------------------------------------
    # All-denied terminal guard
    # ------------------------------------------------------------------

    def record_turn_denials(self, attempted: int, denied: int) -> None:
        """Update the consecutive-all-denied counter for one turn.

        ``attempted`` is the count of tool_use blocks the model emitted
        in this turn; ``denied`` is how many the permission hook rejected.
        When the hook rejected everything, bump the counter; when at
        least one tool ran, reset it.
        """
        if attempted <= 0:
            # No tools attempted this turn — doesn't clear the counter,
            # the previous state stands. (The agent's "no tool_use" branch
            # explicitly resets the counter, which is a stronger signal.)
            return
        if denied == attempted:
            self.consecutive_all_denied += 1
        else:
            self.consecutive_all_denied = 0

    def reset_denial_counter(self) -> None:
        """Reset the counter to zero. Called on a clean no-tool-use turn."""
        self.consecutive_all_denied = 0

    def all_denied_limit_hit(self) -> bool:
        """True once the counter reaches :pyattr:`max_consecutive_all_denied`."""
        return self.consecutive_all_denied >= self.max_consecutive_all_denied

    def describe_limit(self) -> str:
        """Return the human-readable terminal message the agent emits."""
        return (
            f"Terminating: permission policy denied every tool call "
            f"for {self.consecutive_all_denied} consecutive turns. "
            f"The model cannot make progress under the current hook."
        )

    # ------------------------------------------------------------------
    # v0.14: full per-block lifecycle
    # ------------------------------------------------------------------

    def execute_block(
        self,
        *,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str,
        pre_hook: PreToolHookT | None = None,
        post_hook: PostToolHookT | None = None,
        validator: ValidatorT | None = None,
    ) -> tuple[ToolOutcome, list[ToolEvent]]:
        """Run one tool-use block end-to-end and return its outcome + events.

        Replaces the ad-hoc inline per-block body in ``chat_agent.py``.
        Handles all four outcome branches:

        * ``"ok"`` — pre-hook allowed, schema valid, dispatch returned
          ``is_error=False``.
        * ``"denied"`` — pre-hook returned ``DENY``.
        * ``"schema_error"`` — validator rejected the args.
        * ``"error"`` — dispatch returned ``is_error=True``.

        Pre/post hook exceptions are swallowed (observability-only) and
        surface as ``ToolEvent(kind="error", code="carl.hook_failed")``.
        """

        started_at = time.perf_counter()
        events: list[ToolEvent] = []

        # --- pre-hook -------------------------------------------------
        permission = ToolPermission.ALLOW
        if pre_hook is not None:
            try:
                permission = pre_hook(tool_name, tool_input)
            except Exception as exc:
                logger.warning(
                    "pre_tool_use hook failed for %s: %s", tool_name, exc,
                )
                events.append(
                    ToolEvent(
                        kind="error",
                        name=tool_name,
                        content=f"pre_tool_use hook failed for {tool_name}: {exc}",
                        code="carl.hook_failed",
                    )
                )
                permission = ToolPermission.ALLOW

        if permission == ToolPermission.DENY:
            result = f"Blocked by permission policy: {tool_name}"
            events.append(
                ToolEvent(kind="tool_blocked", name=tool_name, content=result)
            )
            return (
                ToolOutcome(
                    tool_use_id=tool_use_id,
                    name=tool_name,
                    input=tool_input,
                    result=result,
                    is_error=True,
                    outcome="denied",
                    duration_ms=(time.perf_counter() - started_at) * 1000.0,
                ),
                events,
            )

        # --- schema validation ---------------------------------------
        if validator is not None:
            schema_err = validator(tool_name, tool_input)
            if schema_err is not None:
                events.append(
                    ToolEvent(
                        kind="tool_result",
                        name=tool_name,
                        content=schema_err,
                        code="carl.tool_schema",
                    )
                )
                return (
                    ToolOutcome(
                        tool_use_id=tool_use_id,
                        name=tool_name,
                        input=tool_input,
                        result=schema_err,
                        is_error=True,
                        outcome="schema_error",
                        duration_ms=(time.perf_counter() - started_at) * 1000.0,
                    ),
                    events,
                )

        # --- dispatch -------------------------------------------------
        result, is_error = self.dispatch_safe(tool_name, tool_input)
        events.append(
            ToolEvent(kind="tool_result", name=tool_name, content=result)
        )

        # --- post-hook ------------------------------------------------
        if post_hook is not None:
            try:
                post_hook(tool_name, tool_input, result)
            except Exception as exc:
                logger.warning(
                    "post_tool_use hook failed for %s: %s", tool_name, exc,
                )
                events.append(
                    ToolEvent(
                        kind="error",
                        name=tool_name,
                        content=(
                            f"post_tool_use hook failed for {tool_name}: {exc}"
                        ),
                        code="carl.hook_failed",
                    )
                )

        return (
            ToolOutcome(
                tool_use_id=tool_use_id,
                name=tool_name,
                input=tool_input,
                result=result,
                is_error=is_error,
                outcome="error" if is_error else "ok",
                duration_ms=(time.perf_counter() - started_at) * 1000.0,
            ),
            events,
        )
