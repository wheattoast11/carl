"""Tests for carl_studio.cu.anthropic_compat.CUDispatcher.

Uses the shared fake-Playwright plumbing from ``tests/_playwright_stub.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from carl_core.interaction import ActionType, InteractionChain
from carl_core.resource_handles import ResourceVault
from carl_core.secrets import SecretVault

from carl_studio.handles.data import DataToolkit

from _playwright_stub import (
    install_fake_playwright,
    uninstall_fake_playwright,
)


@pytest.fixture
def fake_playwright() -> Any:  # pyright: ignore[reportUnusedFunction]
    install_fake_playwright()
    yield
    uninstall_fake_playwright()


def _make_dispatcher(chain: InteractionChain) -> Any:
    from carl_studio.cu.anthropic_compat import CUDispatcher
    from carl_studio.cu.browser import BrowserToolkit

    tk = BrowserToolkit.build(
        chain,
        data_toolkit=DataToolkit.build(chain),
        secret_vault=SecretVault(),
        resource_vault=ResourceVault(),
    )
    desc = tk.open_page()
    d = CUDispatcher(browser=tk)
    d.bind_page(desc["ref_id"])
    return d, tk


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_schema_surface_is_stable() -> None:
    from carl_studio.cu.anthropic_compat import (
        COMPUTER_USE_TOOL_SCHEMA,
        SUPPORTED_ACTIONS,
    )

    assert COMPUTER_USE_TOOL_SCHEMA["name"] == "computer"
    assert "screenshot" in SUPPORTED_ACTIONS
    assert "left_click" in SUPPORTED_ACTIONS
    # Action enum in schema is the union of supported + unsupported
    enum = COMPUTER_USE_TOOL_SCHEMA["input_schema"]["properties"]["action"]["enum"]
    assert "screenshot" in enum
    assert "left_click_drag" in enum  # documented but rejected


def test_dispatch_screenshot_returns_data_ref(fake_playwright: Any) -> None:
    chain = InteractionChain()
    d, _ = _make_dispatcher(chain)
    result = d.dispatch({"action": "screenshot"})
    assert result["action"] == "screenshot"
    assert "data_ref" in result
    assert result["data_ref"]["content_type"] == "image/png"


def test_dispatch_left_click(fake_playwright: Any) -> None:
    chain = InteractionChain()
    d, _ = _make_dispatcher(chain)
    result = d.dispatch({"action": "left_click", "coordinate": [100, 200]})
    assert result["action"] == "left_click"
    assert result["x"] == 100 and result["y"] == 200


def test_dispatch_type(fake_playwright: Any) -> None:
    chain = InteractionChain()
    d, _ = _make_dispatcher(chain)
    result = d.dispatch({"action": "type", "text": "hello"})
    assert result["length"] == 5


def test_dispatch_key(fake_playwright: Any) -> None:
    chain = InteractionChain()
    d, _ = _make_dispatcher(chain)
    result = d.dispatch({"action": "key", "text": "Return"})
    assert result["action"] == "key"


def test_dispatch_scroll_down(fake_playwright: Any) -> None:
    chain = InteractionChain()
    d, _ = _make_dispatcher(chain)
    result = d.dispatch(
        {"action": "scroll", "scroll_direction": "down", "scroll_amount": 2}
    )
    assert result["dy"] == 200
    assert result["dx"] == 0


def test_dispatch_wait_sleeps() -> None:
    # No page needed for wait; dispatcher just sleeps.
    from carl_studio.cu.anthropic_compat import CUDispatcher

    d = CUDispatcher(browser=None)  # type: ignore[arg-type]
    result = d.dispatch({"action": "wait", "duration": 0.01})
    assert result["action"] == "wait"


def test_dispatch_requires_bind_before_pointer_actions(fake_playwright: Any) -> None:
    from carl_studio.cu.anthropic_compat import CUDispatchError, CUDispatcher
    from carl_studio.cu.browser import BrowserToolkit

    chain = InteractionChain()
    tk = BrowserToolkit.build(
        chain,
        data_toolkit=DataToolkit.build(chain),
        secret_vault=SecretVault(),
        resource_vault=ResourceVault(),
    )
    d = CUDispatcher(browser=tk)  # not bound
    with pytest.raises(CUDispatchError) as exc:
        d.dispatch({"action": "left_click", "coordinate": [1, 2]})
    assert exc.value.code == "carl.cu.unbound"


def test_dispatch_rejects_drag(fake_playwright: Any) -> None:
    from carl_studio.cu.anthropic_compat import CUDispatchError

    chain = InteractionChain()
    d, _ = _make_dispatcher(chain)
    with pytest.raises(CUDispatchError) as exc:
        d.dispatch({"action": "left_click_drag", "coordinate": [0, 0]})
    assert exc.value.code == "carl.cu.unsupported_action"


def test_dispatch_rejects_unknown_action(fake_playwright: Any) -> None:
    from carl_studio.cu.anthropic_compat import CUDispatchError

    chain = InteractionChain()
    d, _ = _make_dispatcher(chain)
    with pytest.raises(CUDispatchError) as exc:
        d.dispatch({"action": "fly"})
    assert exc.value.code == "carl.cu.unknown_action"


def test_dispatch_rejects_missing_coordinate(fake_playwright: Any) -> None:
    from carl_studio.cu.anthropic_compat import CUDispatchError

    chain = InteractionChain()
    d, _ = _make_dispatcher(chain)
    with pytest.raises(CUDispatchError) as exc:
        d.dispatch({"action": "left_click"})
    assert exc.value.code == "carl.cu.missing_coordinate"


def test_dispatch_audit_emits_resource_act_steps(fake_playwright: Any) -> None:
    chain = InteractionChain()
    d, _ = _make_dispatcher(chain)
    d.dispatch({"action": "left_click", "coordinate": [10, 10]})
    d.dispatch({"action": "mouse_move", "coordinate": [20, 20]})
    d.dispatch({"action": "type", "text": "abc"})
    d.dispatch({"action": "key", "text": "Return"})
    acts = chain.by_action(ActionType.RESOURCE_ACT)
    # bind_page + open_page's navigate-free open = 2 setup steps + 4 actions
    assert len(acts) >= 4
