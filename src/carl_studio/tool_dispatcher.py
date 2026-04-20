"""Tool dispatch mechanics for :class:`carl_studio.chat_agent.CARLAgent`.

Extracted from ``chat_agent.py`` in the v0.7 Phase-2 god-class decomposition
(ticket G1 Phase-2 / F-PYTH-002). This module owns:

* the tool registry (name -> callable),
* per-tool timeouts with a bounded worker,
* the ``(content, is_error)`` dispatch contract — so the caller can flag
  the API ``tool_result`` as erroring and let the model self-correct,
* the "all-denied consecutive turn" counter + terminal limit that stops a
  misbehaving model+hook combination from looping forever.

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
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

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
