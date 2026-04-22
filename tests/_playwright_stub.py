"""Shared fake-Playwright plumbing for browser + CU dispatcher tests.

Pure Python stand-ins for Playwright's sync_api surface. Installed via
``sys.modules`` patches so the real import path runs; the toolkit under
test calls ``.goto()`` / ``.click()`` / ``.screenshot()`` on our fakes
and we verify the routing without booting a real browser.

One consolidated fake instead of two (was duplicated across
``test_browser_toolkit.py`` and ``test_cu_dispatcher.py``) so the stub
doesn't drift across test files.
"""

from __future__ import annotations

import sys
import types


class FakeKeyboard:
    def __init__(self) -> None:
        self.presses: list[str] = []
        self.typed: list[str] = []

    def press(self, key: str) -> None:
        self.presses.append(key)

    def type(self, text: str) -> None:
        self.typed.append(text)


class FakeMouse:
    def __init__(self) -> None:
        self.clicks: list[tuple[int, int, str, int]] = []
        self.moves: list[tuple[int, int]] = []
        self.wheels: list[tuple[int, int]] = []

    def click(
        self,
        x: int,
        y: int,
        *,
        button: str = "left",
        click_count: int = 1,
    ) -> None:
        self.clicks.append((x, y, button, click_count))

    def move(self, x: int, y: int) -> None:
        self.moves.append((x, y))

    def wheel(self, dx: int, dy: int) -> None:
        self.wheels.append((dx, dy))


class FakeLocator:
    def __init__(self, text: str) -> None:
        self._text = text

    def inner_text(self) -> str:
        return self._text


class FakePage:
    def __init__(self, url: str = "about:blank") -> None:
        self.url = url
        self.filled: dict[str, str] = {}
        self.clicked: list[str] = []
        self.navigated: list[str] = []
        self.keyboard = FakeKeyboard()
        self.mouse = FakeMouse()
        self.closed = False
        self.screenshots_taken = 0

    def goto(self, url: str) -> None:
        self.url = url
        self.navigated.append(url)

    def title(self) -> str:
        return f"title[{self.url}]"

    def fill(self, selector: str, value: str) -> None:
        self.filled[selector] = value

    def click(self, selector: str) -> None:
        self.clicked.append(selector)

    def screenshot(self, *, full_page: bool = False) -> bytes:
        self.screenshots_taken += 1
        tag = b"FULL" if full_page else b"VIEW"
        return b"\x89PNG" + tag + b"\x00" * 16

    def evaluate(self, _js: str) -> str:
        return f"[body innerText for {self.url}]"

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(f"[text in {selector} for {self.url}]")

    def close(self) -> None:
        self.closed = True


class FakeBrowser:
    def __init__(self) -> None:
        self.pages: list[FakePage] = []
        self.closed = False

    def new_page(self) -> FakePage:
        p = FakePage()
        self.pages.append(p)
        return p

    def close(self) -> None:
        self.closed = True


class FakeChromium:
    launched_headless: bool = True

    def launch(self, *, headless: bool = True) -> FakeBrowser:
        self.launched_headless = headless
        return FakeBrowser()


class FakePlaywright:
    def __init__(self) -> None:
        self.chromium = FakeChromium()
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class FakePlaywrightContextMgr:
    """Mimics the ``sync_playwright()`` handle that exposes ``.start()``."""

    def start(self) -> FakePlaywright:
        return FakePlaywright()


def install_fake_playwright() -> None:
    """Insert ``playwright`` + ``playwright.sync_api`` into ``sys.modules``."""
    pkg = types.ModuleType("playwright")
    sync_api = types.ModuleType("playwright.sync_api")

    def sync_playwright() -> FakePlaywrightContextMgr:
        return FakePlaywrightContextMgr()

    sync_api.sync_playwright = sync_playwright  # type: ignore[attr-defined]
    pkg.sync_api = sync_api  # type: ignore[attr-defined]
    sys.modules["playwright"] = pkg
    sys.modules["playwright.sync_api"] = sync_api


def uninstall_fake_playwright() -> None:
    """Undo :func:`install_fake_playwright`. Idempotent."""
    for name in ("playwright.sync_api", "playwright"):
        sys.modules.pop(name, None)


__all__ = [
    "FakeBrowser",
    "FakeChromium",
    "FakeKeyboard",
    "FakeLocator",
    "FakeMouse",
    "FakePage",
    "FakePlaywright",
    "FakePlaywrightContextMgr",
    "install_fake_playwright",
    "uninstall_fake_playwright",
]
