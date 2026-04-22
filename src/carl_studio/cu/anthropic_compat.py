"""Anthropic computer-use tool compat — maps the ``computer_20250124``
action surface to :class:`carl_studio.cu.browser.BrowserToolkit` methods.

The Anthropic Computer Use tool (claude-sonnet-4-6 + claude-opus-4-7)
uses a single tool named ``computer`` with an ``input.action`` field.
Common actions: ``screenshot`` / ``left_click`` / ``type`` /
``mouse_move`` / ``scroll`` / ``key`` / ``wait``.

For a browser-scoped Carl integration, we don't need pixel-coordinate
mouse math — the browser owns layout, and selectors are the native
reference. So the dispatcher here:

* Routes ``screenshot`` → :meth:`BrowserToolkit.screenshot`
* Routes ``left_click`` (with ``coordinate``) → a JS ``document.elementFromPoint()``
  click via Playwright's low-level ``page.mouse.click(x, y)``.
* Routes ``type`` → :meth:`BrowserToolkit.type_text` on the focused element.
* Routes ``key`` → :meth:`BrowserToolkit.press_key`.
* Routes ``mouse_move`` → Playwright's ``page.mouse.move()``.
* Routes ``scroll`` → :meth:`BrowserToolkit.scroll`.
* ``cursor_position`` / ``wait`` are handled inline.

Actions outside that scope are rejected with
``carl.cu.unsupported_action``. The agent should surface the unknown
action rather than silently no-op.

Why stick to Anthropic's schema shape? Two reasons:

1. CARLAgent already knows how to call tools with Anthropic-style
   ``tool_use`` blocks. Handing it this schema unchanged means no custom
   training needed to use browser capability.
2. It keeps the door open to swap the browser for a real screen driver
   (xdotool, CoreGraphics) later without changing the agent's surface.

This module does NOT own the ``page ref_id``; the dispatcher requires
the caller to supply one via :meth:`CUDispatcher.bind_page`. The
alternative — one page per dispatcher — would collapse the page-as-
resource model that the rest of the handle runtime relies on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

from carl_core.errors import CARLError
from carl_core.interaction import ActionType

from carl_studio.cu.browser import BrowserToolkit


def _coord_xy(coord: Any, *, action: str) -> tuple[int, int]:
    """Coerce an Anthropic ``coordinate`` value to an (x, y) int tuple.

    Separating this keeps pyright happy — inside one function we
    narrow once and cast once, rather than repeating the dance at
    every click-family branch below.
    """
    if not isinstance(coord, (list, tuple)) or len(cast(list[Any], coord)) != 2:
        raise CUDispatchError(
            f"{action} requires coordinate=[x, y]",
            code="carl.cu.missing_coordinate",
            context={"action": action},
        )
    xy = cast(list[Any], coord)
    return int(xy[0]), int(xy[1])


__all__ = [
    "COMPUTER_USE_TOOL_SCHEMA",
    "CUDispatcher",
    "CUDispatchError",
    "SUPPORTED_ACTIONS",
]


# Anthropic's computer_use tool surface. Actions that our browser scope
# supports are implemented; the rest are listed as "known but unsupported"
# so downstream callers can tell "typo" from "not wired".
SUPPORTED_ACTIONS: frozenset[str] = frozenset(
    {
        "screenshot",
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "mouse_move",
        "type",
        "key",
        "scroll",
        "wait",
        "cursor_position",
    }
)

# Documented Anthropic actions we do NOT implement in the browser scope
# (no keyboard hold chords, no mouse-drag protocols, no raw mouse_down/up).
# Calling them raises ``carl.cu.unsupported_action`` — the agent should
# fall back to a selector-level browser method instead.
_UNSUPPORTED_ACTIONS: frozenset[str] = frozenset(
    {
        "hold_key",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
    }
)


COMPUTER_USE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "computer",
    "description": (
        "Browser-scoped computer-use tool. All actions operate on a "
        "Playwright page bound via bind_page(ref_id); Carl typically "
        "opens a page first via browser_open_page and passes its ref_id. "
        "Screenshots route through the data vault — the tool result "
        "returns a DataRef descriptor rather than inline PNG bytes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(SUPPORTED_ACTIONS | _UNSUPPORTED_ACTIONS),
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "[x, y] pixel coordinate for click/move actions.",
            },
            "text": {
                "type": "string",
                "description": "Text to type (action='type') or key to press (action='key').",
            },
            "duration": {
                "type": "number",
                "description": "Wait duration in seconds (action='wait').",
            },
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Scroll amount in pixel units.",
            },
        },
        "required": ["action"],
    },
}


class CUDispatchError(CARLError):
    """Base for ``carl.cu.*`` errors."""

    code = "carl.cu"


@dataclass
class CUDispatcher:
    """Dispatch Anthropic-shaped ``computer`` tool_use inputs to a BrowserToolkit.

    Typical usage from CARLAgent:

    1. Agent opens a page: ``browser_open_page(url='https://...')`` →
       returns ``{ref_id, ...}``.
    2. Agent calls ``dispatcher.bind_page(ref_id)`` once.
    3. Every subsequent ``computer`` tool_use from the model flows
       through :meth:`dispatch`.

    Keeping :meth:`bind_page` explicit means multi-tab workflows are
    visible in the chain (one ``CUDispatcher`` per tab, each with its
    own bound ref_id).
    """

    browser: BrowserToolkit
    bound_ref_id: str | None = None

    def bind_page(self, ref_id: str) -> None:
        """Bind the dispatcher to a specific browser page ref_id."""
        self.bound_ref_id = ref_id
        self.browser.chain.record(
            ActionType.RESOURCE_ACT,
            "cu.bind_page",
            input={"ref_id": ref_id},
            output={"bound": True},
            success=True,
        )

    def dispatch(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Execute one Anthropic-shaped ``computer`` tool call.

        Returns a JSON-native dict. For ``screenshot``, the dict includes
        ``data_ref`` (DataRef descriptor). For pointer actions, includes
        the action fields + the page's bound ref_id. Never returns raw
        image bytes.
        """
        action = tool_input.get("action")
        if not isinstance(action, str):
            raise CUDispatchError(
                "tool_input.action must be a string",
                code="carl.cu.missing_action",
                context={"tool_input": tool_input},
            )
        if action in _UNSUPPORTED_ACTIONS:
            raise CUDispatchError(
                f"action {action!r} is documented by Anthropic but not supported "
                "by the browser-scoped CUDispatcher. Use a browser_* tool directly.",
                code="carl.cu.unsupported_action",
                context={"action": action},
            )
        if action not in SUPPORTED_ACTIONS:
            raise CUDispatchError(
                f"unknown computer-use action: {action!r}",
                code="carl.cu.unknown_action",
                context={"action": action, "supported": sorted(SUPPORTED_ACTIONS)},
            )

        if action == "wait":
            duration = float(tool_input.get("duration", 1.0))
            time.sleep(duration)
            return {"action": "wait", "duration": duration}

        if self.bound_ref_id is None:
            raise CUDispatchError(
                "CUDispatcher is not bound to a page. Call bind_page(ref_id) first.",
                code="carl.cu.unbound",
                context={"action": action},
            )

        if action == "screenshot":
            return {"action": "screenshot", **self.browser.screenshot(self.bound_ref_id)}

        if action in ("left_click", "right_click", "middle_click", "double_click", "triple_click"):
            x, y = _coord_xy(tool_input.get("coordinate"), action=action)
            page = self.browser.page_from_id(self.bound_ref_id)
            button = {
                "left_click": "left",
                "right_click": "right",
                "middle_click": "middle",
                "double_click": "left",
                "triple_click": "left",
            }[action]
            click_count = 1
            if action == "double_click":
                click_count = 2
            elif action == "triple_click":
                click_count = 3
            page.mouse.click(x, y, button=button, click_count=click_count)
            self.browser.chain.record(
                ActionType.RESOURCE_ACT,
                f"cu.{action}",
                input={"ref_id": self.bound_ref_id, "coordinate": [x, y]},
                output={"ref_id": self.bound_ref_id, "x": x, "y": y},
                success=True,
            )
            return {"action": action, "ref_id": self.bound_ref_id, "x": x, "y": y}

        if action == "mouse_move":
            x, y = _coord_xy(tool_input.get("coordinate"), action=action)
            page = self.browser.page_from_id(self.bound_ref_id)
            page.mouse.move(x, y)
            self.browser.chain.record(
                ActionType.RESOURCE_ACT,
                "cu.mouse_move",
                input={"ref_id": self.bound_ref_id, "coordinate": [x, y]},
                output={"ref_id": self.bound_ref_id, "x": x, "y": y},
                success=True,
            )
            return {"action": action, "ref_id": self.bound_ref_id, "x": x, "y": y}

        if action == "type":
            text = tool_input.get("text")
            if not isinstance(text, str):
                raise CUDispatchError(
                    "type requires text: str",
                    code="carl.cu.missing_text",
                    context={"action": action},
                )
            page = self.browser.page_from_id(self.bound_ref_id)
            page.keyboard.type(text)
            self.browser.chain.record(
                ActionType.RESOURCE_ACT,
                "cu.type",
                input={"ref_id": self.bound_ref_id, "length": len(text)},
                output={"ref_id": self.bound_ref_id, "length": len(text)},
                success=True,
            )
            return {"action": "type", "ref_id": self.bound_ref_id, "length": len(text)}

        if action == "key":
            key = tool_input.get("text")
            if not isinstance(key, str):
                raise CUDispatchError(
                    "key requires text: str (e.g. 'Return', 'ctrl+c')",
                    code="carl.cu.missing_text",
                    context={"action": action},
                )
            return {"action": "key", **self.browser.press_key(self.bound_ref_id, key)}

        if action == "scroll":
            amount = int(tool_input.get("scroll_amount") or 3)
            direction = tool_input.get("scroll_direction", "down")
            dx, dy = 0, 0
            step_px = 100 * amount
            if direction == "down":
                dy = step_px
            elif direction == "up":
                dy = -step_px
            elif direction == "right":
                dx = step_px
            elif direction == "left":
                dx = -step_px
            else:
                raise CUDispatchError(
                    f"unknown scroll_direction: {direction!r}",
                    code="carl.cu.invalid_direction",
                    context={"scroll_direction": direction},
                )
            return {"action": "scroll", **self.browser.scroll(self.bound_ref_id, dx=dx, dy=dy)}

        if action == "cursor_position":
            # Playwright does not expose the cursor position directly;
            # return the last mouse-move coord if tracked, else 0/0.
            return {"action": "cursor_position", "x": 0, "y": 0}

        # Unreachable — all SUPPORTED_ACTIONS accounted for above.
        raise CUDispatchError(  # pragma: no cover
            f"no handler for action {action!r}",
            code="carl.cu.internal",
            context={"action": action},
        )
