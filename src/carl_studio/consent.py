"""Privacy-first consent state machine.

All flags default to off. The user must explicitly opt in.
Local state is authoritative — the server cannot silently enable tracking.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

_CONSENT_KEY = "consent_state"

CONSENT_KEYS = frozenset({
    "observability",
    "telemetry",
    "usage_analytics",
    "contract_witnessing",
})


class ConsentError(Exception):
    """Raised when consent operations fail."""


class ConsentFlag(BaseModel):
    """A single consent preference with change tracking."""

    enabled: bool = False
    changed_at: str | None = None


class ConsentState(BaseModel):
    """Complete consent state — all flags off by default."""

    observability: ConsentFlag = Field(default_factory=ConsentFlag)
    telemetry: ConsentFlag = Field(default_factory=ConsentFlag)
    usage_analytics: ConsentFlag = Field(default_factory=ConsentFlag)
    contract_witnessing: ConsentFlag = Field(default_factory=ConsentFlag)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ConsentManager:
    """Load, update, and sync consent state with LocalDB."""

    def __init__(self, db: Any | None = None) -> None:
        self._db = db

    def _get_db(self) -> Any:
        if self._db is not None:
            return self._db
        from carl_studio.db import LocalDB

        self._db = LocalDB()
        return self._db

    def load(self) -> ConsentState:
        """Load consent state from LocalDB config table."""
        raw = self._get_db().get_config(_CONSENT_KEY)
        if not raw:
            return ConsentState()
        try:
            return ConsentState.model_validate_json(raw)
        except Exception:
            return ConsentState()

    def save(self, state: ConsentState) -> None:
        """Persist consent state to LocalDB config table."""
        self._get_db().set_config(_CONSENT_KEY, state.model_dump_json())

    def update(self, key: str, enabled: bool) -> ConsentState:
        """Update a single consent flag with timestamp.

        Raises ConsentError if *key* is not a valid consent flag.
        """
        if key not in CONSENT_KEYS:
            raise ConsentError(
                f"Unknown consent key '{key}'. Valid: {', '.join(sorted(CONSENT_KEYS))}"
            )
        state = self.load()
        flag = ConsentFlag(enabled=enabled, changed_at=_now_iso())
        state = state.model_copy(update={key: flag})
        self.save(state)
        return state

    def sync_with_profile(self, profile: Any) -> ConsentState:
        """Sync local consent with remote CampProfile flags.

        Local state wins — remote is informational only. We read remote
        flags but do not overwrite locally granted or revoked consent.
        """
        state = self.load()
        # If user has never set a flag (changed_at is None), adopt remote value
        _remote_map = {
            "observability": getattr(profile, "observability_opt_in", False),
            "telemetry": getattr(profile, "telemetry_opt_in", False),
            "usage_analytics": getattr(profile, "usage_tracking_enabled", False),
            "contract_witnessing": getattr(profile, "contract_witnessing", False),
        }
        updates: dict[str, Any] = {}
        for key, remote_val in _remote_map.items():
            local_flag: ConsentFlag = getattr(state, key)
            if local_flag.changed_at is None and remote_val:
                updates[key] = ConsentFlag(enabled=True, changed_at=_now_iso())
        if updates:
            state = state.model_copy(update=updates)
            self.save(state)
        return state

    def all_off(self) -> ConsentState:
        """Reset all consents to off (privacy-first defaults)."""
        ts = _now_iso()
        state = ConsentState(
            observability=ConsentFlag(enabled=False, changed_at=ts),
            telemetry=ConsentFlag(enabled=False, changed_at=ts),
            usage_analytics=ConsentFlag(enabled=False, changed_at=ts),
            contract_witnessing=ConsentFlag(enabled=False, changed_at=ts),
        )
        self.save(state)
        return state

    def present_first_run(self) -> ConsentState:
        """Interactive first-run consent prompt.

        Defaults all-off in non-interactive (CI) environments.
        """
        if not sys.stdin.isatty():
            return self.all_off()

        import typer

        ts = _now_iso()

        typer.echo("\n  Privacy consent (all off by default, change any time):\n")

        obs = typer.confirm(
            "  Enable observability (coherence probes sent to carl.camp)?", default=False
        )
        tel = typer.confirm(
            "  Enable anonymous telemetry (CLI usage counts)?", default=False
        )
        ana = typer.confirm(
            "  Enable usage analytics (feature analytics)?", default=False
        )
        cw = typer.confirm(
            "  Enable contract witnessing (hash-sign service terms)?", default=False
        )

        state = ConsentState(
            observability=ConsentFlag(enabled=obs, changed_at=ts),
            telemetry=ConsentFlag(enabled=tel, changed_at=ts),
            usage_analytics=ConsentFlag(enabled=ana, changed_at=ts),
            contract_witnessing=ConsentFlag(enabled=cw, changed_at=ts),
        )
        self.save(state)
        return state


def consent_state_from_profile(profile: Any) -> ConsentState:
    """Project a CampProfile onto a ConsentState (informational)."""
    return ConsentState(
        observability=ConsentFlag(enabled=getattr(profile, "observability_opt_in", False)),
        telemetry=ConsentFlag(enabled=getattr(profile, "telemetry_opt_in", False)),
        usage_analytics=ConsentFlag(enabled=getattr(profile, "usage_tracking_enabled", False)),
        contract_witnessing=ConsentFlag(enabled=getattr(profile, "contract_witnessing", False)),
    )
