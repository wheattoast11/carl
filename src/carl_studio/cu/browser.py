"""BrowserToolkit — capability-constrained Playwright automation.

Shape follows :class:`carl_studio.handles.data.DataToolkit` and
:class:`carl_studio.secrets.toolkit.SecretsToolkit`: each action is
agent-callable, emits an audit step, and returns a JSON-native dict.

Key design invariants:

* Pages live in a :class:`~carl_core.resource_handles.ResourceVault`.
  Agent gets ref_ids; Playwright ``Page`` objects never leave the toolkit.
* ``type_from_secret()`` resolves a :class:`~carl_core.secrets.SecretRef`
  inside the toolkit (privileged=True) and types the bytes into a DOM
  input. Secret value never crosses an agent tool boundary.
* ``screenshot()`` and ``extract_text()`` route output through the
  caller-supplied :class:`~carl_studio.handles.data.DataToolkit`. Agent
  sees a :class:`DataRef` descriptor (``{ref_id, kind, size, sha256}``);
  PNG or page text lives in the data vault, out of agent context.
* Every action emits one of ``RESOURCE_OPEN`` / ``RESOURCE_ACT`` /
  ``RESOURCE_CLOSE`` + optionally ``DATA_OPEN`` (for screenshot/text).

Optional dep: ``playwright>=1.45``. Lazy-imported. ``available()`` returns
False if missing. Install: ``pip install playwright && playwright install``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from carl_core.errors import CARLError
from carl_core.interaction import ActionType, InteractionChain
from carl_core.resource_handles import (
    ResourceError,
    ResourceRef,
    ResourceVault,
)
from carl_core.secrets import SecretVault

from carl_studio.handles.data import DataToolkit

__all__ = ["BrowserToolkit", "BrowserToolkitError"]


class BrowserToolkitError(CARLError):
    """Base for ``carl.browser.*`` errors."""

    code = "carl.browser"


@dataclass
class BrowserToolkit:
    """Agent-callable browser automation with vault-mediated pages."""

    resource_vault: ResourceVault
    data_toolkit: DataToolkit
    secret_vault: SecretVault
    chain: InteractionChain
    headless: bool = True
    _playwright: Any = None
    _browser: Any = None

    @classmethod
    def build(
        cls,
        chain: InteractionChain,
        *,
        data_toolkit: DataToolkit,
        secret_vault: SecretVault,
        resource_vault: ResourceVault | None = None,
        headless: bool = True,
    ) -> BrowserToolkit:
        return cls(
            resource_vault=(
                resource_vault if resource_vault is not None else ResourceVault()
            ),
            data_toolkit=data_toolkit,
            secret_vault=secret_vault,
            chain=chain,
            headless=headless,
        )

    # -- availability -----------------------------------------------------

    def available(self) -> bool:
        """True when the sync_api subset of Playwright is importable.

        Uses ``__import__`` so ``sys.modules`` patches in tests also register
        — ``importlib.util.find_spec`` rejects spec-less fakes.
        """
        try:
            __import__("playwright.sync_api")
        except ImportError:
            return False
        return True

    def missing_dependencies(self) -> list[str]:
        return [] if self.available() else ["playwright"]

    def _ensure_playwright(self) -> Any:
        """Lazy-import + launch Playwright. Cached on the instance."""
        if self._browser is not None:
            return self._browser
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserToolkitError(
                "playwright is required. Install with: "
                "pip install playwright && playwright install chromium",
                code="carl.browser.backend_unavailable",
                context={"missing": "playwright"},
                cause=exc,
            ) from exc
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        return self._browser

    # -- page lifecycle ---------------------------------------------------

    def open_page(
        self,
        *,
        url: str | None = None,
        labels: dict[str, str] | None = None,
        ttl_s: int | None = None,
    ) -> dict[str, Any]:
        """Open a new browser page. Navigates to ``url`` if given.

        Returns the :class:`ResourceRef` descriptor — ref_id is the handle
        the agent uses for subsequent actions.
        """
        browser = self._ensure_playwright()
        page = browser.new_page()
        if url is not None:
            page.goto(url)
        ref = self.resource_vault.put(
            backend=page,
            kind="browser_page",
            provider="playwright",
            uri=url or "about:blank",
            labels=labels or {},
            ttl_s=ttl_s,
            closer=_close_page,
        )
        desc = ref.describe()
        self.chain.record(
            ActionType.RESOURCE_OPEN,
            "browser.open_page",
            input={"url": url, "labels": labels, "ttl_s": ttl_s},
            output=desc,
            success=True,
        )
        return desc

    def close_page(self, ref_id: str) -> bool:
        ref = self._ref_from_id(ref_id)
        closed = self.resource_vault.revoke(ref)
        self.chain.record(
            ActionType.RESOURCE_CLOSE,
            "browser.close_page",
            input={"ref_id": ref_id},
            output={"closed": closed},
            success=closed,
        )
        return closed

    def list_pages(self) -> list[dict[str, Any]]:
        return [r.describe() for r in self.resource_vault.list_refs()]

    # -- navigation + actions --------------------------------------------

    def navigate(self, ref_id: str, url: str) -> dict[str, Any]:
        page = self._page_from_id(ref_id)
        page.goto(url)
        result = {"ref_id": ref_id, "url": url, "title": page.title()}
        self.chain.record(
            ActionType.RESOURCE_ACT,
            "browser.navigate",
            input={"ref_id": ref_id, "url": url},
            output=result,
            success=True,
        )
        return result

    def click(self, ref_id: str, selector: str) -> dict[str, Any]:
        page = self._page_from_id(ref_id)
        page.click(selector)
        result = {"ref_id": ref_id, "selector": selector}
        self.chain.record(
            ActionType.RESOURCE_ACT,
            "browser.click",
            input={"ref_id": ref_id, "selector": selector},
            output=result,
            success=True,
        )
        return result

    def type_text(
        self,
        ref_id: str,
        selector: str,
        text: str,
        *,
        delay_ms: int = 0,
    ) -> dict[str, Any]:
        """Type literal text into a DOM input.

        Use this for non-sensitive values. Secrets flow through
        :meth:`type_from_secret`.
        """
        page = self._page_from_id(ref_id)
        page.fill(selector, text)
        result = {
            "ref_id": ref_id,
            "selector": selector,
            "length": len(text),
            "delay_ms": delay_ms,
        }
        self.chain.record(
            ActionType.RESOURCE_ACT,
            "browser.type_text",
            input={
                "ref_id": ref_id,
                "selector": selector,
                "length": len(text),
                "delay_ms": delay_ms,
            },
            output=result,
            success=True,
        )
        return result

    def type_from_secret(
        self,
        ref_id: str,
        selector: str,
        secret_ref_id: str,
    ) -> dict[str, Any]:
        """Type the value behind a :class:`SecretRef` into the page.

        The raw bytes are resolved *inside the toolkit* (privileged=True)
        and handed directly to Playwright. The agent's call site records
        only the ``(ref_id, selector, secret_ref_id, fingerprint)`` tuple.
        """
        page = self._page_from_id(ref_id)
        secret_ref = self._secret_ref_from_id(secret_ref_id)
        value = self.secret_vault.resolve(secret_ref, privileged=True)
        fingerprint_hex = self.secret_vault.fingerprint_of(secret_ref)
        try:
            page.fill(selector, value.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise BrowserToolkitError(
                "secret value is not utf-8 decodable; type_from_secret "
                "requires text-shaped secrets. Use a different minter.",
                code="carl.browser.secret_not_text",
                context={"secret_ref_id": secret_ref_id},
                cause=exc,
            ) from exc
        result = {
            "ref_id": ref_id,
            "selector": selector,
            "secret_ref_id": secret_ref_id,
            "secret_fingerprint": fingerprint_hex,
        }
        self.chain.record(
            ActionType.RESOURCE_ACT,
            "browser.type_from_secret",
            input={
                "ref_id": ref_id,
                "selector": selector,
                "secret_ref_id": secret_ref_id,
                "secret_fingerprint": fingerprint_hex,
            },
            output=result,
            success=True,
        )
        return result

    def press_key(self, ref_id: str, key: str) -> dict[str, Any]:
        page = self._page_from_id(ref_id)
        page.keyboard.press(key)
        result = {"ref_id": ref_id, "key": key}
        self.chain.record(
            ActionType.RESOURCE_ACT,
            "browser.press_key",
            input={"ref_id": ref_id, "key": key},
            output=result,
            success=True,
        )
        return result

    def scroll(
        self,
        ref_id: str,
        *,
        dx: int = 0,
        dy: int = 0,
    ) -> dict[str, Any]:
        page = self._page_from_id(ref_id)
        page.mouse.wheel(dx, dy)
        result = {"ref_id": ref_id, "dx": dx, "dy": dy}
        self.chain.record(
            ActionType.RESOURCE_ACT,
            "browser.scroll",
            input={"ref_id": ref_id, "dx": dx, "dy": dy},
            output=result,
            success=True,
        )
        return result

    # -- capture → DataRef ----------------------------------------------

    def screenshot(
        self,
        ref_id: str,
        *,
        full_page: bool = False,
    ) -> dict[str, Any]:
        """Capture a screenshot. PNG bytes route through ``data_toolkit``;
        the agent sees only the :class:`DataRef` descriptor."""
        page = self._page_from_id(ref_id)
        png_bytes = page.screenshot(full_page=full_page)
        data_desc = self.data_toolkit.open_bytes(
            png_bytes,
            content_type="image/png",
            uri=f"carl-data://screenshot/{ref_id}",
        )
        result = {
            "ref_id": ref_id,
            "full_page": full_page,
            "data_ref": data_desc,
        }
        self.chain.record(
            ActionType.RESOURCE_ACT,
            "browser.screenshot",
            input={"ref_id": ref_id, "full_page": full_page},
            output={"ref_id": ref_id, "data_ref_id": data_desc["ref_id"]},
            success=True,
        )
        return result

    def extract_text(
        self,
        ref_id: str,
        *,
        selector: str | None = None,
    ) -> dict[str, Any]:
        """Extract page text into a :class:`DataRef`.

        ``selector=None`` captures the full ``document.body.innerText``;
        a selector narrows to a subtree. Text is pushed to the data vault
        so large pages don't flood the chain.
        """
        page = self._page_from_id(ref_id)
        if selector is None:
            text = page.evaluate("document.body ? document.body.innerText : ''")
        else:
            element = page.locator(selector)
            text = element.inner_text()
        text_str: str = text if isinstance(text, str) else str(text)
        data_desc = self.data_toolkit.open_bytes(
            text_str.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
            uri=f"carl-data://page-text/{ref_id}",
        )
        result = {
            "ref_id": ref_id,
            "selector": selector,
            "length": len(text_str),
            "data_ref": data_desc,
        }
        self.chain.record(
            ActionType.RESOURCE_ACT,
            "browser.extract_text",
            input={"ref_id": ref_id, "selector": selector},
            output={
                "ref_id": ref_id,
                "length": len(text_str),
                "data_ref_id": data_desc["ref_id"],
            },
            success=True,
        )
        return result

    # -- schema helpers --------------------------------------------------

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "browser_open_page",
                "description": "Open a browser page and optionally navigate to a URL.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": ["string", "null"]},
                        "ttl_s": {"type": ["integer", "null"]},
                    },
                },
            },
            {
                "name": "browser_navigate",
                "description": "Navigate a page to a URL.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["ref_id", "url"],
                },
            },
            {
                "name": "browser_click",
                "description": "Click a DOM selector on a page.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "selector": {"type": "string"},
                    },
                    "required": ["ref_id", "selector"],
                },
            },
            {
                "name": "browser_type_text",
                "description": "Type literal text into a DOM input.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "selector": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["ref_id", "selector", "text"],
                },
            },
            {
                "name": "browser_type_from_secret",
                "description": "Type a SecretRef's value into a DOM input without the agent seeing the value.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "selector": {"type": "string"},
                        "secret_ref_id": {"type": "string"},
                    },
                    "required": ["ref_id", "selector", "secret_ref_id"],
                },
            },
            {
                "name": "browser_press_key",
                "description": "Press a keyboard key (e.g. Enter, ArrowDown).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "key": {"type": "string"},
                    },
                    "required": ["ref_id", "key"],
                },
            },
            {
                "name": "browser_scroll",
                "description": "Scroll a page by (dx, dy) pixels.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "dx": {"type": "integer", "default": 0},
                        "dy": {"type": "integer", "default": 0},
                    },
                    "required": ["ref_id"],
                },
            },
            {
                "name": "browser_screenshot",
                "description": "Take a screenshot. PNG bytes route into DataVault; agent receives the DataRef.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "full_page": {"type": "boolean", "default": False},
                    },
                    "required": ["ref_id"],
                },
            },
            {
                "name": "browser_extract_text",
                "description": "Extract page text (whole body or selector subtree) into a DataRef.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "selector": {"type": ["string", "null"]},
                    },
                    "required": ["ref_id"],
                },
            },
            {
                "name": "browser_close_page",
                "description": "Close a browser page ref.",
                "input_schema": {
                    "type": "object",
                    "properties": {"ref_id": {"type": "string"}},
                    "required": ["ref_id"],
                },
            },
            {
                "name": "browser_list_pages",
                "description": "List current non-revoked browser page refs.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def teardown(self) -> None:
        """Close all pages + shut down Playwright. Safe to call multiple times."""
        for ref in self.resource_vault.list_refs():
            self.resource_vault.revoke(ref)
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:  # pragma: no cover
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:  # pragma: no cover
                pass
            self._playwright = None

    # -- public helpers for collaborating toolkits (e.g. CUDispatcher) ----

    def page_from_id(self, ref_id: str) -> Any:
        """Return the underlying Playwright ``Page`` for ``ref_id``.

        Privileged: only toolkit-layer collaborators (CUDispatcher,
        :class:`~carl_studio.cu.anthropic_compat.CUDispatcher`) should call
        this. Not agent-facing.
        """
        return self._page_from_id(ref_id)

    def ref_from_id(self, ref_id: str) -> ResourceRef:
        """Return the :class:`ResourceRef` for ``ref_id``. Non-sensitive."""
        return self._ref_from_id(ref_id)

    # -- internals --------------------------------------------------------

    def _ref_from_id(self, ref_id: str) -> ResourceRef:
        try:
            parsed = uuid.UUID(ref_id)
        except (TypeError, ValueError) as exc:
            raise BrowserToolkitError(
                f"ref_id is not a valid UUID: {ref_id!r}",
                code="carl.browser.invalid_ref_id",
                context={"ref_id": ref_id},
                cause=exc,
            ) from exc
        for ref in self.resource_vault.list_refs():
            if ref.ref_id == parsed:
                return ref
        raise ResourceError(
            f"unknown or closed browser page: {ref_id}",
            code="carl.resource.not_found",
            context={"ref_id": ref_id},
        )

    def _page_from_id(self, ref_id: str) -> Any:
        ref = self._ref_from_id(ref_id)
        return self.resource_vault.resolve(ref, privileged=True)

    def _secret_ref_from_id(self, secret_ref_id: str) -> Any:
        try:
            parsed = uuid.UUID(secret_ref_id)
        except (TypeError, ValueError) as exc:
            raise BrowserToolkitError(
                f"secret_ref_id is not a valid UUID: {secret_ref_id!r}",
                code="carl.browser.invalid_ref_id",
                context={"secret_ref_id": secret_ref_id},
                cause=exc,
            ) from exc
        for ref in self.secret_vault.list_refs():
            if ref.ref_id == parsed:
                return ref
        raise BrowserToolkitError(
            f"unknown or revoked secret handle: {secret_ref_id}",
            code="carl.browser.secret_not_found",
            context={"secret_ref_id": secret_ref_id},
        )


def _close_page(page: Any) -> None:
    """Closer callback stored on the ResourceRef entry."""
    try:
        page.close()
    except Exception:  # pragma: no cover
        pass
