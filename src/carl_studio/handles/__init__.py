"""Capability-constrained handle runtime — non-secret toolkit surfaces.

The ``carl_studio.secrets`` package is the specialized handle runtime for
credentials (zero-knowledge by default, privileged resolve only). This
package is the *generalization*: the same ``*Ref → *Vault → *Toolkit``
shape applied to data payloads, long-lived resources, queries, streams,
and anything else the agent should reason about by reference rather than
by value.

Stage C shipping order:

* Stage C-1 (here): :mod:`carl_studio.handles.data` — ``DataToolkit``
  wrapping :class:`carl_core.data_handles.DataVault` with audit steps,
  chunked reads, transforms, publish sinks.
* Stage C-2 (follow-up): ``carl_studio.handles.resource`` — browser /
  subprocess / Playwright resources, Anthropic CU 28-action compat.

Top-level re-exports kept minimal on purpose — the agent-facing contract
is the ``*Toolkit`` classes themselves, not every internal helper.
"""

from __future__ import annotations

from carl_studio.handles.bundle import (
    HandleRuntimeBundle,
    make_handler,
)
from carl_studio.handles.data import DataToolkit, DataToolkitError
from carl_studio.handles.subprocess import (
    SubprocessToolkit,
    SubprocessToolkitError,
)

__all__ = [
    "DataToolkit",
    "DataToolkitError",
    "SubprocessToolkit",
    "SubprocessToolkitError",
    "HandleRuntimeBundle",
    "make_handler",
]
