from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from carl_studio.db import LocalDB


TRUST_NAMESPACE = "carl.trust"
TRUST_KEY = "carl.trust.state"


class TrustState(BaseModel):
    """Persisted trust preferences for the unified CLI entry router."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(
        default=True,
        description="Whether the trust pre-check should run before bare-entry routing.",
    )
    acknowledged_project_root: str | None = Field(
        default=None,
        description="Absolute project root path the user has already trusted.",
    )


class TrustRegistry:
    """Typed LocalDB-backed trust state for the current CARL install."""

    def __init__(self, db: LocalDB | None = None) -> None:
        self._db = db if db is not None else LocalDB()
        self._registry = self._db.config_registry(
            TrustState,
            namespace=TRUST_NAMESPACE,
            key=TRUST_KEY,
        )

    @property
    def key(self) -> str:
        return self._registry.key

    def get(self) -> TrustState:
        return self._registry.with_defaults(TrustState)

    def set(self, state: TrustState) -> None:
        self._registry.set(state)

    def reset(self) -> TrustState:
        state = TrustState()
        self.set(state)
        return state

    def is_enabled(self) -> bool:
        return self.get().enabled

    def set_enabled(self, enabled: bool) -> TrustState:
        current = self.get()
        next_state = current.model_copy(update={"enabled": bool(enabled)})
        self.set(next_state)
        return next_state

    def current_acknowledged_root(self) -> Path | None:
        raw = self.get().acknowledged_project_root
        if not raw:
            return None
        return Path(raw).expanduser().resolve()

    def is_trusted_root(self, root: Path) -> bool:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            return False
        acknowledged = self.current_acknowledged_root()
        return acknowledged is not None and acknowledged == resolved

    def trust_root(self, root: Path) -> TrustState:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            resolved = root.expanduser().absolute()
        next_state = self.get().model_copy(
            update={"acknowledged_project_root": str(resolved)}
        )
        self.set(next_state)
        return next_state


def get_trust_registry(db: LocalDB | None = None) -> TrustRegistry:
    """Return a LocalDB-backed trust registry.

    Kept as a tiny seam so router/CLI tests can inject a fake registry factory
    without patching LocalDB itself.
    """

    return TrustRegistry(db=db)


__all__ = [
    "TRUST_KEY",
    "TRUST_NAMESPACE",
    "TrustRegistry",
    "TrustState",
    "get_trust_registry",
]
