"""Tests for carl_studio.cu.browser.BrowserToolkit with mocked Playwright.

Fake Playwright plumbing lives in ``tests/_playwright_stub.py`` so this
file and ``test_cu_dispatcher.py`` share one stub.
"""

from __future__ import annotations

from typing import Any, cast

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


def _build_toolkit(chain: InteractionChain) -> Any:
    from carl_studio.cu.browser import BrowserToolkit

    data = DataToolkit.build(chain)
    secret_vault = SecretVault()
    return BrowserToolkit.build(
        chain,
        data_toolkit=data,
        secret_vault=secret_vault,
        resource_vault=ResourceVault(),
    )


def test_available_false_without_playwright() -> None:
    # Skip when the real playwright is installed in the dev env — the
    # "without playwright" premise only holds on a fresh install.
    uninstall_fake_playwright()
    import importlib.util

    if importlib.util.find_spec("playwright") is not None:
        pytest.skip("real playwright installed; absence can't be asserted here")
    chain = InteractionChain()
    tk = _build_toolkit(chain)
    assert tk.available() is False
    assert tk.missing_dependencies() == ["playwright"]


def test_available_true_with_fake(fake_playwright: Any) -> None:
    chain = InteractionChain()
    tk = _build_toolkit(chain)
    assert tk.available() is True


def test_open_page_emits_resource_open(fake_playwright: Any) -> None:
    chain = InteractionChain()
    tk = _build_toolkit(chain)
    desc = tk.open_page(url="https://example.com")
    assert desc["kind"] == "browser_page"
    assert desc["provider"] == "playwright"
    assert desc["uri"] == "https://example.com"
    opens = chain.by_action(ActionType.RESOURCE_OPEN)
    assert len(opens) == 1


def test_navigate_click_and_type(fake_playwright: Any) -> None:
    chain = InteractionChain()
    tk = _build_toolkit(chain)
    desc = tk.open_page()
    rid = desc["ref_id"]
    tk.navigate(rid, "https://x.test")
    tk.click(rid, "#submit")
    tk.type_text(rid, "#name", "Tej")
    acts = chain.by_action(ActionType.RESOURCE_ACT)
    assert len(acts) == 3
    assert acts[0].name == "browser.navigate"
    assert acts[1].name == "browser.click"
    assert acts[2].name == "browser.type_text"
    # typed value appears in the chain input only as length
    raw = cast(dict[str, Any], acts[2].input)
    assert raw["length"] == 3


def test_type_from_secret_does_not_leak_value(fake_playwright: Any) -> None:
    chain = InteractionChain()
    tk = _build_toolkit(chain)
    # Mint a secret in the vault the toolkit uses
    secret_ref = tk.secret_vault.put("hunter2", kind="mint")
    desc = tk.open_page()
    result = tk.type_from_secret(desc["ref_id"], "#password", str(secret_ref.ref_id))
    assert "secret_fingerprint" in result
    # Serialized chain must not contain the value
    serialized = chain.to_jsonl()
    assert "hunter2" not in serialized


def test_screenshot_routes_png_to_datavault(fake_playwright: Any) -> None:
    chain = InteractionChain()
    tk = _build_toolkit(chain)
    desc = tk.open_page()
    cap = tk.screenshot(desc["ref_id"])
    assert "data_ref" in cap
    data_ref = cap["data_ref"]
    assert data_ref["kind"] == "bytes"
    assert data_ref["content_type"] == "image/png"
    # Only one DATA_OPEN step (from the data toolkit)
    data_opens = chain.by_action(ActionType.DATA_OPEN)
    assert len(data_opens) == 1


def test_extract_text_routes_to_datavault(fake_playwright: Any) -> None:
    chain = InteractionChain()
    tk = _build_toolkit(chain)
    desc = tk.open_page()
    result = tk.extract_text(desc["ref_id"])
    assert "data_ref" in result
    data_ref = result["data_ref"]
    assert data_ref["kind"] == "bytes"
    assert data_ref["content_type"].startswith("text/plain")


def test_close_page_releases_ref(fake_playwright: Any) -> None:
    chain = InteractionChain()
    tk = _build_toolkit(chain)
    desc = tk.open_page()
    assert tk.close_page(desc["ref_id"]) is True
    assert tk.list_pages() == []
    closes = chain.by_action(ActionType.RESOURCE_CLOSE)
    assert len(closes) == 1


def test_tool_schemas_cover_expected_methods(fake_playwright: Any) -> None:
    chain = InteractionChain()
    tk = _build_toolkit(chain)
    names = {s["name"] for s in tk.tool_schemas()}
    for expected in (
        "browser_open_page",
        "browser_navigate",
        "browser_click",
        "browser_type_text",
        "browser_type_from_secret",
        "browser_screenshot",
        "browser_extract_text",
        "browser_close_page",
    ):
        assert expected in names


def test_teardown_is_idempotent(fake_playwright: Any) -> None:
    chain = InteractionChain()
    tk = _build_toolkit(chain)
    desc = tk.open_page()
    # Touch a page so _browser is cached
    tk.navigate(desc["ref_id"], "about:blank")
    tk.teardown()
    tk.teardown()  # second call is a no-op
