"""HandleRuntimeBundle — one-call construction of the full handle runtime.

Wiring the v0.16.1 handle runtime into :class:`~carl_studio.chat_agent.CARLAgent`
without this bundle looks like:

* Build a ``SecretVault`` + ``ResourceVault``.
* Build ``DataToolkit`` + ``SecretsToolkit`` + ``BrowserToolkit`` +
  ``SubprocessToolkit`` separately.
* Register every method on every toolkit into
  :class:`~carl_studio.tool_dispatcher.ToolDispatcher` with a
  hand-rolled ``dict → (str, bool)`` adapter for each.
* Collect every ``tool_schemas()`` list and flatten.

That's ~40 lines of boilerplate that future Claude sessions *will* get
slightly wrong. The bundle standardizes it to four lines:

    bundle = HandleRuntimeBundle.build(chain=chain)
    bundle.register_all(agent.tool_dispatcher)
    tools_for_anthropic = bundle.anthropic_tools()
    # ... pass `tools=tools_for_anthropic` to the Anthropic API

:class:`ToolCallable` (from :mod:`carl_studio.tool_dispatcher`) takes one
positional ``dict[str, Any]`` and returns ``(content_str, is_error_bool)``.
Toolkit methods, by contrast, take kwargs and return structured dicts.
:func:`make_handler` bridges the two.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from carl_core.interaction import InteractionChain
from carl_core.resource_handles import ResourceVault
from carl_core.secrets import SecretVault

from carl_studio.cu.anthropic_compat import (
    COMPUTER_USE_TOOL_SCHEMA,
    CUDispatcher,
)
from carl_studio.cu.browser import BrowserToolkit
from carl_studio.handles.data import DataToolkit
from carl_studio.handles.subprocess import SubprocessToolkit


__all__ = [
    "HandleRuntimeBundle",
    "make_handler",
]


ToolHandler = Callable[[dict[str, Any]], tuple[str, bool]]


def make_handler(
    method: Callable[..., Any],
) -> ToolHandler:
    """Wrap a toolkit method (kwargs → dict) as a :class:`ToolCallable`.

    The returned handler:

    * takes one positional ``dict[str, Any]`` of arguments,
    * calls the method via ``method(**args)``,
    * JSON-encodes the result (so CARLAgent's ``tool_result`` block
      receives a stable string payload),
    * catches exceptions and returns ``(f"Error: ...", True)``.
    """

    def _handler(args: dict[str, Any]) -> tuple[str, bool]:
        try:
            result = method(**args)
        except TypeError as exc:
            # Missing / extra keyword — surface as error string without
            # raising so CARLAgent's dispatcher records it cleanly.
            return f"Error: {exc}", True
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}", True
        try:
            return json.dumps(result, default=str), False
        except (TypeError, ValueError) as exc:
            # Should be rare — toolkit methods return JSON-native dicts
            # by contract. Still, don't crash the agent.
            return f"Error: non-JSON tool result: {exc}", True

    _handler.__name__ = getattr(method, "__name__", "tool_handler")
    return _handler


@dataclass
class HandleRuntimeBundle:
    """All handle-runtime toolkits constructed + wired against one chain."""

    chain: InteractionChain
    secret_vault: SecretVault
    resource_vault: ResourceVault
    data_toolkit: DataToolkit
    browser_toolkit: BrowserToolkit
    subprocess_toolkit: SubprocessToolkit
    cu_dispatcher: CUDispatcher
    # Populated lazily by :meth:`register_all`.
    registered_tool_names: list[str] = field(
        default_factory=lambda: [],  # type: list[str]
    )

    @classmethod
    def build(
        cls,
        chain: InteractionChain,
        *,
        secret_vault: SecretVault | None = None,
        resource_vault: ResourceVault | None = None,
        headless_browser: bool = True,
        data_preview_bytes: int = 65536,
    ) -> HandleRuntimeBundle:
        """Construct every vault + toolkit against the supplied chain.

        ``secret_vault`` / ``resource_vault`` fall back to fresh instances
        when explicitly ``None``. The ``is not None`` dance avoids the
        empty-vault falsy trap (vaults define ``__len__``; ``X or Vault()``
        silently replaces an empty-but-live vault with a new one).
        """
        sv = secret_vault if secret_vault is not None else SecretVault()
        rv = resource_vault if resource_vault is not None else ResourceVault()
        data = DataToolkit.build(chain, preview_bytes=data_preview_bytes)
        browser = BrowserToolkit.build(
            chain,
            data_toolkit=data,
            secret_vault=sv,
            resource_vault=rv,
            headless=headless_browser,
        )
        subprocess_tk = SubprocessToolkit.build(
            chain,
            data_toolkit=data,
            resource_vault=rv,
        )
        cu = CUDispatcher(browser=browser)
        return cls(
            chain=chain,
            secret_vault=sv,
            resource_vault=rv,
            data_toolkit=data,
            browser_toolkit=browser,
            subprocess_toolkit=subprocess_tk,
            cu_dispatcher=cu,
        )

    # -- agent-facing surface -------------------------------------------

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Flat list of tool schemas for the Anthropic ``tools=`` API param.

        The union of all toolkit surfaces. Safe to pass directly.
        """
        schemas: list[dict[str, Any]] = []
        schemas.extend(self.data_toolkit.tool_schemas())
        schemas.extend(self.browser_toolkit.tool_schemas())
        schemas.extend(self.subprocess_toolkit.tool_schemas())
        schemas.append(COMPUTER_USE_TOOL_SCHEMA)
        return schemas

    def agent_handlers(self) -> dict[str, ToolHandler]:
        """Map of tool name → wrapped ``ToolCallable`` handler.

        Names match the schemas from :meth:`anthropic_tools`.
        """
        handlers: dict[str, ToolHandler] = {}

        # --- DataToolkit ---
        handlers["data_open_file"] = make_handler(self.data_toolkit.open_file)
        handlers["data_read_text"] = make_handler(self.data_toolkit.read_text)
        handlers["data_read_json"] = make_handler(self.data_toolkit.read_json)
        handlers["data_transform"] = make_handler(self.data_toolkit.transform)
        handlers["data_publish_to_file"] = make_handler(
            self.data_toolkit.publish_to_file
        )
        handlers["data_list_handles"] = make_handler(
            _adapt_zero_arg(self.data_toolkit.list_handles)
        )

        # --- BrowserToolkit ---
        b = self.browser_toolkit
        handlers["browser_open_page"] = make_handler(b.open_page)
        handlers["browser_navigate"] = make_handler(b.navigate)
        handlers["browser_click"] = make_handler(b.click)
        handlers["browser_type_text"] = make_handler(b.type_text)
        handlers["browser_type_from_secret"] = make_handler(b.type_from_secret)
        handlers["browser_press_key"] = make_handler(b.press_key)
        handlers["browser_scroll"] = make_handler(b.scroll)
        handlers["browser_screenshot"] = make_handler(b.screenshot)
        handlers["browser_extract_text"] = make_handler(b.extract_text)
        handlers["browser_close_page"] = make_handler(b.close_page)
        handlers["browser_list_pages"] = make_handler(_adapt_zero_arg(b.list_pages))

        # --- SubprocessToolkit ---
        s = self.subprocess_toolkit
        handlers["subprocess_spawn"] = make_handler(s.spawn)
        handlers["subprocess_poll"] = make_handler(s.poll)
        handlers["subprocess_wait"] = make_handler(s.wait)
        handlers["subprocess_terminate"] = make_handler(s.terminate)
        handlers["subprocess_read_stdout"] = make_handler(s.read_stdout)
        handlers["subprocess_read_stderr"] = make_handler(s.read_stderr)
        handlers["subprocess_list"] = make_handler(
            _adapt_zero_arg(s.list_processes)
        )

        # --- CUDispatcher ---
        handlers["computer"] = make_handler(self.cu_dispatcher.dispatch)

        return handlers

    def register_all(self, dispatcher: Any) -> list[str]:
        """Register every handler on a :class:`ToolDispatcher`-shaped object.

        Accepts ``Any`` to avoid a hard dependency on
        ``carl_studio.tool_dispatcher.ToolDispatcher`` here — the only
        requirement is a ``.register(name: str, fn: ToolCallable)`` method.
        Returns the list of registered names (also stored on
        :attr:`registered_tool_names`).
        """
        handlers = self.agent_handlers()
        for name, handler in handlers.items():
            dispatcher.register(name, handler)
        self.registered_tool_names = sorted(handlers.keys())
        return self.registered_tool_names

    def tool_catalog(self) -> dict[str, Any]:
        """Catalog suitable for a Carl "what can you do?" meta-tool."""
        return {
            "toolkits": {
                "data": [s["name"] for s in self.data_toolkit.tool_schemas()],
                "browser": [s["name"] for s in self.browser_toolkit.tool_schemas()],
                "subprocess": [
                    s["name"] for s in self.subprocess_toolkit.tool_schemas()
                ],
                "computer_use": [COMPUTER_USE_TOOL_SCHEMA["name"]],
            },
            "total": len(self.anthropic_tools()),
            "doctrine": (
                "Carl moves refs, not values. Every tool returns handles "
                "with fingerprints; bytes live in the vaults."
            ),
        }


def _adapt_zero_arg(
    fn: Callable[[], Any],
) -> Callable[..., Any]:
    """Wrap a no-arg method so ``method(**{})`` works without TypeError."""

    def _wrapped(**_kwargs: Any) -> Any:
        if _kwargs:
            # The agent might pass empty-object args; tolerate but warn
            # in the result so debugging is easier.
            pass
        return fn()

    _wrapped.__name__ = getattr(fn, "__name__", "zero_arg_fn")
    return _wrapped
