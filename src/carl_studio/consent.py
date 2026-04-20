"""Privacy-first consent state machine.

All flags default to off. The user must explicitly opt in.
Local state is authoritative — the server cannot silently enable tracking.

Enforcement
-----------
Runtime network-egress paths MUST call :func:`consent_gate` at their
boundary with the relevant consent flag. Gates currently wired:

* ``"telemetry"`` — :func:`carl_studio.sync.push` / :func:`~carl_studio.sync.pull`,
  MCP ``authenticate`` tool.
* ``"contract_witnessing"`` — :class:`carl_studio.x402.X402Client.execute` and
  :meth:`carl_studio.x402_connection.PaymentConnection.get` (payments create
  service-contract witnesses).
* ``"observability"`` — reserved for outbound coherence-probe publish paths.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any, Literal

from carl_core import now_iso
from carl_core.errors import CARLError
from pydantic import BaseModel, Field

_CONSENT_KEY = "consent_state"

CONSENT_KEYS = frozenset({
    "observability",
    "telemetry",
    "usage_analytics",
    "contract_witnessing",
})

#: Literal alias for consent-flag keys. Enables static checking at call
#: sites (``consent_gate("telemetry")``) without introducing a parallel
#: enum. Runtime validation still goes through :data:`CONSENT_KEYS`.
ConsentKey = Literal[
    "observability",
    "telemetry",
    "usage_analytics",
    "contract_witnessing",
]


class ConsentError(CARLError):
    """Raised when consent operations fail."""

    code = "carl.consent"


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

    def is_granted(self, key: ConsentKey | str) -> bool:
        """Return ``True`` iff the consent flag *key* is currently enabled.

        Unknown keys return ``False`` (fail-closed — a typo or removed flag
        must never leak into granted-by-mistake behavior).
        """
        flag_str = str(key)
        if flag_str not in CONSENT_KEYS:
            return False
        state = self.load()
        flag: ConsentFlag = getattr(state, flag_str)
        return bool(flag.enabled)

    def update(self, key: str, enabled: bool) -> ConsentState:
        """Update a single consent flag with timestamp.

        Raises ConsentError if *key* is not a valid consent flag.
        """
        if key not in CONSENT_KEYS:
            raise ConsentError(
                f"Unknown consent key '{key}'. Valid: {', '.join(sorted(CONSENT_KEYS))}"
            )
        state = self.load()
        flag = ConsentFlag(enabled=enabled, changed_at=now_iso())
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
                updates[key] = ConsentFlag(enabled=True, changed_at=now_iso())
        if updates:
            state = state.model_copy(update=updates)
            self.save(state)
        return state

    def all_off(self) -> ConsentState:
        """Reset all consents to off (privacy-first defaults)."""
        ts = now_iso()
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

        ts = now_iso()

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


def consent_gate(
    flag: ConsentKey | str,
    *,
    manager: ConsentManager | None = None,
) -> None:
    """Raise :class:`ConsentError` if the given flag is not granted.

    Call at every network-egress boundary that the flag protects. When
    *manager* is ``None`` the gate reads from a freshly constructed
    :class:`ConsentManager` (which lazy-opens the process-default
    :class:`~carl_studio.db.LocalDB`).

    Parameters
    ----------
    flag
        One of the :data:`ConsentKey` literals (``"observability"``,
        ``"telemetry"``, ``"usage_analytics"``, ``"contract_witnessing"``).
        Accepts a bare ``str`` for runtime flexibility; unknown values raise.
    manager
        Optional :class:`ConsentManager` to use. Tests typically inject a
        manager bound to a ``FakeDB`` to exercise the gate deterministically.

    Raises
    ------
    ConsentError
        When the flag is unknown or not currently granted. The raised error
        carries ``code="carl.consent.denied"`` and ``context={"flag": ...}``
        so operators can distinguish a denied gate from other consent
        failures in logs.
    """
    flag_str = str(flag)
    if flag_str not in CONSENT_KEYS:
        raise ConsentError(
            f"Unknown consent flag '{flag_str}'. "
            f"Valid: {', '.join(sorted(CONSENT_KEYS))}",
            code="carl.consent.unknown_flag",
            context={"flag": flag_str},
        )
    mgr = manager or ConsentManager()
    if not mgr.is_granted(flag_str):
        raise ConsentError(
            f"consent flag '{flag_str}' not granted — "
            f"run `carl camp consent update {flag_str} --enable` to allow",
            code="carl.consent.denied",
            context={"flag": flag_str},
        )


class _ConsentFlagKeyShim:
    """Namespace matching the retired ``ConsentFlagKey`` enum values.

    Each attribute exposes the underlying string so ``.value`` / equality
    checks against strings continue to work. Emits a ``DeprecationWarning``
    on first access (the module-level ``__getattr__`` does the warn, this
    class only carries the constants).
    """

    OBSERVABILITY = "observability"
    TELEMETRY = "telemetry"
    USAGE_ANALYTICS = "usage_analytics"
    CONTRACT_WITNESSING = "contract_witnessing"


def __getattr__(name: str) -> Any:
    if name == "ConsentFlagKey":
        warnings.warn(
            "ConsentFlagKey is deprecated; use the ConsentKey Literal alias "
            "or CONSENT_KEYS strings directly. Removal in v0.8.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ConsentFlagKeyShim
    raise AttributeError(f"module 'carl_studio.consent' has no attribute {name!r}")


__all__ = [
    "CONSENT_KEYS",
    "ConsentError",
    "ConsentFlag",
    "ConsentKey",
    "ConsentManager",
    "ConsentState",
    "consent_gate",
    "consent_state_from_profile",
]
