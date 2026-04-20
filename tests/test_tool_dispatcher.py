"""Unit tests for :class:`carl_studio.tool_dispatcher.ToolDispatcher`.

Locks in the BC contract the v0.7 Phase-2 extraction promised:

* ``dispatch_safe(name, args) -> (content, is_error)`` returns the error
  flag so the agent can surface ``tool_result`` as erroring,
* ``dispatch(name, args) -> str`` is the back-compat shim,
* per-tool timeouts kick in via a bounded worker,
* ``subprocess.TimeoutExpired`` / ``TimeoutError`` become a stable
  ``"Tool ... timed out"`` string,
* the all-denied counter increments + resets correctly and the terminal
  limit kicks in at :data:`_MAX_CONSECUTIVE_ALL_DENIED`.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

import pytest

from carl_studio.tool_dispatcher import (
    ToolDispatcher,
    _DEFAULT_TOOL_TIMEOUT_S,
    _MAX_CONSECUTIVE_ALL_DENIED,
)


class TestRegistration:
    def test_register_empty_name_rejected(self) -> None:
        d = ToolDispatcher()
        with pytest.raises(ValueError):
            d.register("", lambda args: ("", False))

    def test_register_non_callable_rejected(self) -> None:
        d = ToolDispatcher()
        with pytest.raises(ValueError):
            d.register("x", "not-callable")  # type: ignore[arg-type]

    def test_register_simple_wraps_as_success(self) -> None:
        d = ToolDispatcher()
        d.register_simple("echo", lambda a: str(a.get("v", "")))
        content, is_error = d.dispatch_safe("echo", {"v": "hi"})
        assert content == "hi"
        assert is_error is False


class TestDispatchBasics:
    def test_unknown_tool_returns_error_flag(self) -> None:
        d = ToolDispatcher()
        content, is_error = d.dispatch_safe("nope", {})
        assert is_error is True
        assert "Unknown tool" in content

    def test_dispatch_simple_shim_returns_content_only(self) -> None:
        d = ToolDispatcher()
        d.register_simple("echo", lambda a: str(a.get("v", "")))
        assert d.dispatch("echo", {"v": "hi"}) == "hi"

    def test_tool_exception_mapped_to_error_string(self) -> None:
        d = ToolDispatcher()

        def _boom(_args: dict[str, Any]) -> tuple[str, bool]:
            raise RuntimeError("kaboom")

        d.register("fail", _boom)
        content, is_error = d.dispatch_safe("fail", {})
        assert is_error is True
        assert "kaboom" in content
        assert content.startswith("Error:")

    def test_subprocess_timeout_expired_mapped_to_string(self) -> None:
        d = ToolDispatcher()

        def _slow(_args: dict[str, Any]) -> tuple[str, bool]:
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

        d.register("slow", _slow)
        d.timeouts["slow"] = 30.0
        content, is_error = d.dispatch_safe("slow", {})
        assert is_error is True
        assert "timed out" in content
        assert "slow" in content

    def test_timeout_error_mapped_to_string(self) -> None:
        d = ToolDispatcher()

        def _slow(_args: dict[str, Any]) -> tuple[str, bool]:
            raise TimeoutError("nope")

        d.register("slow", _slow)
        content, is_error = d.dispatch_safe("slow", {})
        assert is_error is True
        assert "timed out" in content

    def test_is_error_flag_propagated_from_tool(self) -> None:
        d = ToolDispatcher()
        d.register("cli", lambda a: ("exit=1", True))
        content, is_error = d.dispatch_safe("cli", {})
        assert content == "exit=1"
        assert is_error is True


class TestTimeoutWrapper:
    def test_hang_triggers_wall_clock_timeout(self) -> None:
        d = ToolDispatcher(default_timeout_s=0.2)

        def _hang(_args: dict[str, Any]) -> tuple[str, bool]:
            time.sleep(5.0)
            return "never", False

        d.register("hang", _hang)
        content, is_error = d.dispatch_safe("hang", {})
        assert is_error is True
        assert "timed out" in content
        assert "hang" in content

    def test_per_tool_timeout_override(self) -> None:
        d = ToolDispatcher(default_timeout_s=30.0)

        def _hang(_args: dict[str, Any]) -> tuple[str, bool]:
            time.sleep(2.0)
            return "never", False

        d.register("hang", _hang)
        d.timeouts["hang"] = 0.2
        content, is_error = d.dispatch_safe("hang", {})
        assert is_error is True
        assert "timed out after 0s" in content

    def test_fast_tool_completes_under_timeout(self) -> None:
        d = ToolDispatcher(default_timeout_s=2.0)
        d.register_simple("ok", lambda _a: "fine")
        content, is_error = d.dispatch_safe("ok", {})
        assert content == "fine"
        assert is_error is False


class TestAllDeniedCounter:
    def test_initial_counter_zero(self) -> None:
        d = ToolDispatcher()
        assert d.consecutive_all_denied == 0
        assert d.all_denied_limit_hit() is False

    def test_all_denied_increments(self) -> None:
        d = ToolDispatcher()
        d.record_turn_denials(attempted=2, denied=2)
        d.record_turn_denials(attempted=3, denied=3)
        assert d.consecutive_all_denied == 2

    def test_partial_denial_resets(self) -> None:
        d = ToolDispatcher()
        d.record_turn_denials(attempted=2, denied=2)
        d.record_turn_denials(attempted=2, denied=1)
        assert d.consecutive_all_denied == 0

    def test_zero_attempted_does_not_touch_counter(self) -> None:
        d = ToolDispatcher()
        d.consecutive_all_denied = 3
        d.record_turn_denials(attempted=0, denied=0)
        assert d.consecutive_all_denied == 3

    def test_reset_denial_counter(self) -> None:
        d = ToolDispatcher()
        d.consecutive_all_denied = 7
        d.reset_denial_counter()
        assert d.consecutive_all_denied == 0

    def test_limit_hit_at_max(self) -> None:
        d = ToolDispatcher()
        for _ in range(_MAX_CONSECUTIVE_ALL_DENIED):
            d.record_turn_denials(attempted=1, denied=1)
        assert d.all_denied_limit_hit() is True

    def test_limit_not_hit_below_max(self) -> None:
        d = ToolDispatcher()
        for _ in range(_MAX_CONSECUTIVE_ALL_DENIED - 1):
            d.record_turn_denials(attempted=1, denied=1)
        assert d.all_denied_limit_hit() is False

    def test_describe_limit_includes_counter(self) -> None:
        d = ToolDispatcher()
        d.consecutive_all_denied = 5
        msg = d.describe_limit()
        assert "5 consecutive turns" in msg
        assert "permission policy" in msg

    def test_custom_limit_override(self) -> None:
        d = ToolDispatcher(max_consecutive_all_denied=2)
        d.record_turn_denials(attempted=1, denied=1)
        assert d.all_denied_limit_hit() is False
        d.record_turn_denials(attempted=1, denied=1)
        assert d.all_denied_limit_hit() is True


class TestDefaults:
    def test_default_timeout_constant(self) -> None:
        assert _DEFAULT_TOOL_TIMEOUT_S == 30.0

    def test_max_consecutive_all_denied_constant(self) -> None:
        assert _MAX_CONSECUTIVE_ALL_DENIED == 5

    def test_default_factory_instance(self) -> None:
        d = ToolDispatcher()
        assert d.tools == {}
        assert d.timeouts == {}
        assert d.default_timeout_s == _DEFAULT_TOOL_TIMEOUT_S
        assert d.max_consecutive_all_denied == _MAX_CONSECUTIVE_ALL_DENIED
