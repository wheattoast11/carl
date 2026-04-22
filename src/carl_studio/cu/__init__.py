"""Computer-use (CU) toolkits — browser automation + Anthropic CU compat.

Stage C-2 of the handle runtime. Two tightly-coupled modules:

* :mod:`carl_studio.cu.browser` — :class:`BrowserToolkit` wraps Playwright
  behind a :class:`~carl_core.resource_handles.ResourceVault`. Pages, clicks,
  typing, screenshots, and text extraction all go through the vault; agent
  context sees handles + descriptors, never raw DOM or bitmap bytes.
* :mod:`carl_studio.cu.anthropic_compat` — the Anthropic ``computer_use``
  tool schema mapping + dispatcher. Lets CARLAgent (or any Anthropic
  client) issue CU actions that route through :class:`BrowserToolkit`'s
  capability-constrained surface.
* :mod:`carl_studio.cu.privacy` — screen / text redaction helpers
  (PII scrubbing before capture emits a DataRef). Scaffold today;
  openadapt-privacy inspiration.

Optional deps: ``playwright`` (browser). Lazy-imported; toolkit
``available()`` reports honestly.
"""

from __future__ import annotations

from carl_studio.cu.anthropic_compat import (
    COMPUTER_USE_TOOL_SCHEMA,
    CUDispatcher,
    CUDispatchError,
)
from carl_studio.cu.browser import BrowserToolkit, BrowserToolkitError

__all__ = [
    "BrowserToolkit",
    "BrowserToolkitError",
    "COMPUTER_USE_TOOL_SCHEMA",
    "CUDispatcher",
    "CUDispatchError",
]
