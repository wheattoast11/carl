"""Session primitive + TwinCheckpoint — v0.17 top-level handle-runtime entry.

Session is the canonical user-facing handle-runtime API. It bundles a chain,
three vaults (secrets / data / resource), four toolkits (data / browser /
subprocess / computer-use), and a content-addressed snapshot/restore pair
into one context-manageable object.

Additive, not replacing:

- Existing ``InteractionChain()`` constructs still work everywhere — Session
  wraps a chain, it doesn't supplant it.
- Existing ``HandleRuntimeBundle.build(chain=...)`` still works — Session is
  the richer superset (adds TwinCheckpoint + the user-friendly context-manager
  lifecycle).

Typical usage::

    import carl

    with carl.Session("tej") as s:
        s.register_tools_to(agent.tool_dispatcher)
        # agent can now call data_* / browser_* / subprocess_* / computer tools
        # and s.chain captures every step for audit / replay.

Digital-twin checkpoint::

    ckpt = s.snapshot()          # content-addressed TwinCheckpoint (JSON-safe)
    json.dumps(ckpt.model_dump()) # persistable / wire-safe
    restored = carl.Session.restore(ckpt)  # new session; resolvers re-attach lazily
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from carl_core.data_handles import DataVault
from carl_core.interaction import InteractionChain
from carl_core.resource_handles import ResourceVault
from carl_core.secrets import SecretVault

from carl_studio.cu.anthropic_compat import CUDispatcher
from carl_studio.cu.browser import BrowserToolkit
from carl_studio.handles.data import DataToolkit
from carl_studio.handles.subprocess import SubprocessToolkit


__all__ = [
    "Session",
    "TwinCheckpoint",
]


# ---------------------------------------------------------------------------
# TwinCheckpoint — content-addressed snapshot
# ---------------------------------------------------------------------------


class TwinCheckpoint(BaseModel):
    """Digital-twin snapshot. JSON-safe, content-addressed.

    Stores chain metadata + vault refs (NEVER values — resolvers handle the
    actual bytes on restore). Content-hash computed over canonical JSON of all
    fields except the hash itself, so re-serializing the same checkpoint
    produces the same content_hash. Future Terminals-OS kernels can verify
    chain integrity + twin continuity without rolling their own format.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1)
    created_at: datetime = Field(...)
    user: str | None = Field(default=None)
    chain: dict[str, Any] = Field(..., description="InteractionChain.to_dict() output")
    refs: dict[str, list[dict[str, Any]]] = Field(
        ...,
        description="{'secret': [...], 'data': [...], 'resource': [...]} — "
        "each list contains ref.describe() dicts (metadata only; never values).",
    )
    content_hash: str = Field(
        ...,
        description="sha256 over canonical JSON of all fields except this one "
        "(64 lowercase hex chars). Stable under re-serialization.",
    )

    @classmethod
    def build(
        cls,
        *,
        user: str | None,
        chain: InteractionChain,
        secret_refs: list[dict[str, Any]],
        data_refs: list[dict[str, Any]],
        resource_refs: list[dict[str, Any]],
        schema_version: int = 1,
        created_at: datetime | None = None,
    ) -> TwinCheckpoint:
        """Construct a checkpoint + compute content_hash."""
        stamp = created_at or datetime.now(timezone.utc)
        chain_dict: dict[str, Any] = chain.to_dict()
        refs: dict[str, list[dict[str, Any]]] = {
            "secret": secret_refs,
            "data": data_refs,
            "resource": resource_refs,
        }
        payload: dict[str, Any] = {
            "schema_version": schema_version,
            "created_at": stamp.isoformat(),
            "user": user,
            "chain": chain_dict,
            "refs": refs,
        }
        canonical = json.dumps(payload, sort_keys=True, default=str)
        h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(
            schema_version=schema_version,
            created_at=stamp,
            user=user,
            chain=chain_dict,
            refs=refs,
            content_hash=h,
        )

    def verify(self) -> bool:
        """Recompute content_hash and compare. True iff snapshot is untampered."""
        payload = {
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat(),
            "user": self.user,
            "chain": self.chain,
            "refs": self.refs,
        }
        canonical = json.dumps(payload, sort_keys=True, default=str)
        h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return h == self.content_hash


# ---------------------------------------------------------------------------
# Session — bundled chain + vaults + toolkits
# ---------------------------------------------------------------------------


def _fresh_secret_vault() -> SecretVault:
    return SecretVault()


def _fresh_resource_vault() -> ResourceVault:
    return ResourceVault()


def _fresh_data_vault() -> DataVault:
    return DataVault()


def _fresh_chain() -> InteractionChain:
    return InteractionChain()


@dataclass
class Session:
    """Top-level handle-runtime entry point.

    Constructs + wires: chain, three vaults, four toolkits. Context-manageable
    (``__enter__`` / ``__exit__``) — the exit path tears down long-lived
    browser + subprocess resources. Tests that want raw primitives keep
    constructing ``InteractionChain()`` / ``SecretVault()`` directly; Session
    is for app-level code that benefits from the full bundle.
    """

    user: str | None = None
    chain: InteractionChain = field(default_factory=_fresh_chain)
    secret_vault: SecretVault = field(default_factory=_fresh_secret_vault)
    resource_vault: ResourceVault = field(default_factory=_fresh_resource_vault)
    data_vault: DataVault = field(default_factory=_fresh_data_vault)
    headless_browser: bool = True
    # Downstream toolkits — lazy-constructed in __post_init__ so they share the
    # session's chain + vaults rather than each creating their own.
    data_toolkit: DataToolkit = field(init=False)
    browser_toolkit: BrowserToolkit = field(init=False)
    subprocess_toolkit: SubprocessToolkit = field(init=False)
    cu_dispatcher: CUDispatcher = field(init=False)

    def __post_init__(self) -> None:
        self.data_toolkit = DataToolkit(vault=self.data_vault, chain=self.chain)
        self.browser_toolkit = BrowserToolkit(
            resource_vault=self.resource_vault,
            data_toolkit=self.data_toolkit,
            secret_vault=self.secret_vault,
            chain=self.chain,
            headless=self.headless_browser,
        )
        self.subprocess_toolkit = SubprocessToolkit(
            resource_vault=self.resource_vault,
            data_toolkit=self.data_toolkit,
            chain=self.chain,
        )
        self.cu_dispatcher = CUDispatcher(browser=self.browser_toolkit)

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> Session:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        """Tear down long-lived resources. Does not suppress exceptions."""
        self.teardown()
        return False

    def teardown(self) -> None:
        """Explicitly release browser + subprocess + other long-lived resources.

        Called from ``__exit__``. Safe to call multiple times (the toolkits
        themselves are idempotent on re-teardown).
        """
        try:
            self.browser_toolkit.teardown()
        except Exception:  # pragma: no cover — teardown is best-effort
            pass
        # SubprocessToolkit does not own playwright-style handles; the resource
        # vault's revoke path runs each subprocess's closer. Revoke everything.
        for ref in self.resource_vault.list_refs():
            try:
                self.resource_vault.revoke(ref)
            except Exception:  # pragma: no cover
                pass

    # -- tool registration --------------------------------------------------

    def register_tools_to(self, dispatcher: Any) -> list[str]:
        """Register every handle-runtime tool onto ``dispatcher``.

        Accepts any object with a ``.register(name: str, fn: ToolCallable)``
        method (duck-typed against ``carl_studio.tool_dispatcher.ToolDispatcher``).
        Returns the sorted list of registered names.
        """
        from carl_studio.handles.bundle import HandleRuntimeBundle

        # Build a transient bundle that shares our vaults/toolkits; use its
        # agent_handlers surface to register the 25-tool handle-runtime API.
        bundle = HandleRuntimeBundle(
            chain=self.chain,
            secret_vault=self.secret_vault,
            resource_vault=self.resource_vault,
            data_toolkit=self.data_toolkit,
            browser_toolkit=self.browser_toolkit,
            subprocess_toolkit=self.subprocess_toolkit,
            cu_dispatcher=self.cu_dispatcher,
        )
        return bundle.register_all(dispatcher)

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Flat list of tool schemas for the Anthropic ``tools=`` API param."""
        from carl_studio.handles.bundle import HandleRuntimeBundle

        bundle = HandleRuntimeBundle(
            chain=self.chain,
            secret_vault=self.secret_vault,
            resource_vault=self.resource_vault,
            data_toolkit=self.data_toolkit,
            browser_toolkit=self.browser_toolkit,
            subprocess_toolkit=self.subprocess_toolkit,
            cu_dispatcher=self.cu_dispatcher,
        )
        return bundle.anthropic_tools()

    # -- snapshot / restore -------------------------------------------------

    def snapshot(self) -> TwinCheckpoint:
        """Capture a content-addressed twin checkpoint.

        Serializes chain + vault ref metadata (NOT values). The resolver chain
        re-attaches values on restore — external-backed refs (env, keyring,
        fernet-file, 1password) reconnect via their registered resolvers;
        inline-bytes refs need explicit re-injection via put (they're the
        "values held only in this process memory" case).
        """
        return TwinCheckpoint.build(
            user=self.user,
            chain=self.chain,
            secret_refs=[r.describe() for r in self.secret_vault.list_refs()],
            data_refs=[r.describe() for r in self.data_vault.list_refs()],
            resource_refs=[r.describe() for r in self.resource_vault.list_refs()],
        )

    @classmethod
    def restore(cls, checkpoint: TwinCheckpoint) -> Session:
        """Restore a session from a :class:`TwinCheckpoint`.

        Chain state is rehydrated from checkpoint.chain. Vault refs are
        *metadata-only* in the snapshot — callers typically re-register
        resolvers on the new session's vaults so external-backed refs
        reconnect to their backends on next resolve.
        """
        if not checkpoint.verify():
            from carl_core.errors import CARLError

            raise CARLError(
                "TwinCheckpoint content_hash mismatch — snapshot corrupted",
                code="carl.session.checkpoint_tampered",
                context={"stored_hash": checkpoint.content_hash},
            )
        chain = InteractionChain.from_dict(checkpoint.chain)
        return cls(user=checkpoint.user, chain=chain)
