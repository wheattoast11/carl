"""x402 HTTP payment rail client for machine-to-machine micropayments.

The facilitator handles all on-chain work. This client sends HTTP requests
with x402 headers. No web3.py dependency — stdlib urllib only.

Consent gate
------------
Every payment execution implicitly witnesses a service contract (the
merchant's x402 terms), so :meth:`X402Client.execute` is gated by the
``"contract_witnessing"`` consent flag. The check-only path
(``check_x402``) and negotiation are permitted — users can discover cost
without paying — but the actual settlement is blocked when
contract-witnessing consent has not been granted.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

from pydantic import BaseModel, Field

from carl_core.errors import BudgetError, CARLError, NetworkError
from carl_core.interaction import ActionType, InteractionChain
from carl_core.resilience import BreakAndRetryStrategy
from carl_core.retry import CircuitBreaker, RetryPolicy

from carl_studio.consent import consent_gate

if TYPE_CHECKING:
    from carl_studio.config_registry import ConfigRegistry
    from carl_studio.db import LocalDB


#: Signature for an x402 payment-confirmation callback. Called with
#: keyword arguments ``url`` and ``amount``; returns ``True`` to approve,
#: ``False`` to deny. A denial is surfaced as
#: ``BudgetError(code="carl.budget.confirm_denied")`` — other exceptions
#: propagate unchanged so UI-layer aborts are never swallowed.
ConfirmPaymentCallback: TypeAlias = Callable[..., bool]

# Module-level plug-point registry for named ``confirm_payment_cb``
# hooks. Private-runtime integrations persist only a string name in
# ``carl.yaml`` (or `~/.carl/config.yaml`) and resolve the live callable
# at execute time through this registry. Intentionally not persisted —
# the registry is a per-process wiring surface, not durable state.
_CALLBACK_REGISTRY: dict[str, ConfirmPaymentCallback] = {}


def register_confirm_callback(name: str, cb: ConfirmPaymentCallback) -> None:
    """Register a named ``confirm_payment_cb`` for string-based resolution.

    Parameters
    ----------
    name:
        The identifier that appears in ``X402Config.confirm_payment_cb``
        when stored as a string (non-empty, stripped). Re-registering the
        same name overwrites the prior entry — callers wanting a
        one-shot install should call :func:`get_confirm_callback` first.
    cb:
        A callable matching :data:`ConfirmPaymentCallback` —
        ``(*, url: str, amount: float) -> bool``.

    Raises
    ------
    ValueError
        If ``name`` is empty / whitespace-only or ``cb`` is not callable.
    """
    if not name or not name.strip():
        raise ValueError("confirm_payment_cb name must be a non-empty string")
    if not callable(cb):
        raise ValueError("confirm_payment_cb must be callable")
    _CALLBACK_REGISTRY[name] = cb


def get_confirm_callback(name: str) -> ConfirmPaymentCallback | None:
    """Return the registered callback for ``name`` or ``None`` if absent."""
    return _CALLBACK_REGISTRY.get(name)


def unregister_confirm_callback(name: str) -> None:
    """Remove ``name`` from the registry. No-op when the name is absent.

    Primarily for test teardown and for private-runtime code that needs
    to swap implementations at runtime without a process restart.
    """
    _CALLBACK_REGISTRY.pop(name, None)

_X402_CONFIG_KEY = "x402_config"

# Legacy SpendTracker persistence keys — kept as module-level constants
# so the one-shot migration path can find and delete them. New state is
# stored as a single JSON blob under ``ConfigRegistry[SpendState]``.
_LEGACY_SPEND_DAILY_KEY = "carl.x402.spend_today"
_LEGACY_SPEND_DAILY_RESET_KEY = "carl.x402.daily_reset_at"

# Env-var escape hatch for the zero-touch rollback — when set to
# ``"skip"`` the migration is a no-op and legacy keys are ignored.
_MIGRATE_ENV = "CARL_CONFIG_MIGRATE"

# Module-level breaker for facilitator calls. Only infrastructure failures
# count against the threshold — programming bugs in our own code (attribute
# errors, type errors, ...) propagate unchanged so callers see the real
# traceback instead of a misleading "circuit_open" error.
_FACILITATOR_BREAKER = CircuitBreaker(
    failure_threshold=5,
    reset_s=60.0,
    tracked_exceptions=(NetworkError, ConnectionError, TimeoutError, IOError),
)

# Conservative retry policy for facilitator HTTP calls. Payments are not
# idempotent — we retry only on transport faults (network / timeout /
# connection errors), never on application-level HTTP errors which would
# risk double-charging. ``urllib.error.HTTPError`` is a subclass of
# ``URLError`` / ``OSError``, so it is intentionally omitted from the
# retryable tuple to keep HTTP 4xx / 5xx responses from being silently
# re-driven by the strategy.
_FACILITATOR_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    backoff_base=0.5,
    max_delay=2.0,
    jitter=True,
    retryable=(NetworkError, ConnectionError, TimeoutError),
)

# Composed strategy: consult the breaker first (fail fast if OPEN), then
# retry transient faults under the conservative policy above. Existing
# call sites that still use ``_FACILITATOR_BREAKER`` directly are
# unchanged — the strategy is additive so new code paths can opt in
# without forcing a migration.
_FACILITATOR_STRATEGY = BreakAndRetryStrategy(
    retry_policy=_FACILITATOR_RETRY_POLICY,
    breaker=_FACILITATOR_BREAKER,
)


class X402Error(CARLError):
    """Raised when x402 payment operations fail."""

    code = "carl.x402"


class X402Config(BaseModel):
    """x402 payment rail configuration."""

    wallet_address: str = ""
    chain: str = "base"
    facilitator_url: str = ""
    payment_token: str = "USDC"
    auto_approve_below: float = 0.0
    enabled: bool = False
    # H1: spend caps + confirmation hook
    daily_spend_cap: float | None = None  # USD; None = unlimited (legacy behavior)
    session_spend_cap: float | None = None  # USD; None = unlimited
    # Either a registered hook name (string; resolved via
    # :func:`get_confirm_callback` at ``X402Client.execute`` time) or a
    # direct :data:`ConfirmPaymentCallback`. ``None`` auto-approves
    # within spend caps. Strings allow persistence through
    # ``carl.yaml``/``~/.carl/config.yaml`` where only JSON-serializable
    # values survive a write/read cycle; callables must be wired
    # in-process via :func:`register_confirm_callback`.
    confirm_payment_cb: str | ConfirmPaymentCallback | None = Field(
        default=None,
    )

    # Pydantic v2 model config — allow the non-JSON-serializable
    # Callable variant without forcing ``arbitrary_types_allowed`` on
    # every other field. Pydantic already accepts ``str``/``None``.
    model_config = {"arbitrary_types_allowed": True}


class PaymentRequirement(BaseModel):
    """Parsed from a 402 response's x-payment header."""

    amount: str = "0"
    token: str = ""
    chain: str = ""
    recipient: str = ""
    facilitator: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


def _parse_x_payment_header(header: str) -> PaymentRequirement:
    """Parse an x-payment response header into a PaymentRequirement."""
    try:
        data = json.loads(header)
        return PaymentRequirement.model_validate(data)
    except (json.JSONDecodeError, Exception):
        # Fallback: key=value pairs separated by semicolons
        parts: dict[str, str] = {}
        for segment in header.split(";"):
            segment = segment.strip()
            if "=" in segment:
                k, v = segment.split("=", 1)
                parts[k.strip()] = v.strip()
        return PaymentRequirement(
            amount=parts.get("amount", "0"),
            token=parts.get("token", ""),
            chain=parts.get("chain", ""),
            recipient=parts.get("recipient", ""),
            facilitator=parts.get("facilitator", ""),
        )


class SpendState(BaseModel):
    """Persisted daily-rolling spend state for :class:`SpendTracker`.

    Stored as a single JSON blob under
    ``carl.x402.spendstate`` via :class:`ConfigRegistry`. Only the
    daily rolling window persists; session totals are intentionally
    in-memory so every new shell starts fresh.
    """

    #: Running total for the current UTC day (USD).
    spend_today: float = 0.0
    #: Timestamp of the last midnight reset (UTC ISO-8601 string).
    daily_reset_at: str = ""


class SpendTracker:
    """Rolling-window spend accounting for x402 payments.

    Persists the daily total and a daily-reset timestamp through a
    typed :class:`ConfigRegistry[SpendState]` over the shared
    :class:`~carl_studio.db.LocalDB`. Session total is held in-process
    only (no cross-process leakage between shells). Cap enforcement is
    synchronous and happens before any network call so a breach raises
    immediately without partial state.

    Back-compat
    -----------
    On first use against a DB that still has the pre-v0.8 flat keys
    (``carl.x402.spend_today`` + ``carl.x402.daily_reset_at``) the
    tracker migrates them into the new :class:`SpendState` blob and
    deletes the legacy rows. Set ``CARL_CONFIG_MIGRATE=skip`` in the
    environment to disable the migration (zero-touch rollback).

    Parameters
    ----------
    daily_cap:
        Maximum USD a single UTC day may accumulate. ``None`` disables
        the daily check (legacy unlimited behavior).
    session_cap:
        Maximum USD this :class:`SpendTracker` instance may accumulate
        across its lifetime. ``None`` disables the session check.
    db:
        Optional :class:`~carl_studio.db.LocalDB` for persisting the
        daily rolling window. When ``None`` the tracker runs in-memory
        only — useful for short-lived tools and tests.
    now:
        Clock override for testing (returns a timezone-aware
        :class:`~datetime.datetime`). Defaults to ``datetime.now(UTC)``.
    """

    def __init__(
        self,
        *,
        daily_cap: float | None = None,
        session_cap: float | None = None,
        db: LocalDB | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._daily_cap = daily_cap
        self._session_cap = session_cap
        self._db = db
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._session_total = 0.0
        self._registry: ConfigRegistry[SpendState] | None = None
        self._migrated = False
        if db is not None:
            self._registry = db.config_registry(
                SpendState, namespace="carl.x402"
            )
            self._maybe_migrate_legacy()

    def check_and_record(self, amount: float) -> None:
        """Gate a payment of ``amount`` USD.

        Raises
        ------
        ValueError
            If ``amount`` is negative.
        BudgetError
            With ``code='carl.budget.daily_cap_exceeded'`` or
            ``code='carl.budget.session_cap_exceeded'`` when a cap would
            be breached. ``context`` carries the attempted amount,
            current running total, and the cap.
        """
        if amount < 0:
            raise ValueError(f"amount must be >= 0, got {amount}")
        daily_total = self._read_daily_total_or_reset()
        projected_daily = daily_total + amount
        projected_session = self._session_total + amount
        if self._daily_cap is not None and projected_daily > self._daily_cap:
            raise BudgetError(
                f"daily spend cap exceeded: "
                f"{projected_daily:.4f} > {self._daily_cap:.4f} USD",
                code="carl.budget.daily_cap_exceeded",
                context={
                    "amount": amount,
                    "daily_total": daily_total,
                    "cap": self._daily_cap,
                },
            )
        if self._session_cap is not None and projected_session > self._session_cap:
            raise BudgetError(
                f"session spend cap exceeded: "
                f"{projected_session:.4f} > {self._session_cap:.4f} USD",
                code="carl.budget.session_cap_exceeded",
                context={
                    "amount": amount,
                    "session_total": self._session_total,
                    "cap": self._session_cap,
                },
            )
        # Commit — both caps passed (or were unset).
        self._session_total = projected_session
        self._write_daily_total(projected_daily)

    def _maybe_migrate_legacy(self) -> None:
        """One-shot migration from flat legacy keys to :class:`SpendState`.

        Idempotent by construction — once a blob exists under the new
        key or the legacy keys are absent, subsequent calls short-circuit.
        The ``CARL_CONFIG_MIGRATE=skip`` env var is an explicit opt-out
        for users rolling back to pre-v0.8 behavior.
        """
        if self._migrated or self._db is None or self._registry is None:
            return
        self._migrated = True
        if os.environ.get(_MIGRATE_ENV, "").strip().lower() == "skip":
            return
        # If new state already exists, leave it alone — a prior run
        # already migrated (or the user started fresh on v0.8).
        try:
            existing = self._registry.get()
        except CARLError:
            # Schema mismatch — leave legacy keys for a human to inspect.
            return
        if existing is not None:
            return
        legacy_total = self._db.get_config(_LEGACY_SPEND_DAILY_KEY)
        legacy_reset = self._db.get_config(_LEGACY_SPEND_DAILY_RESET_KEY)
        if legacy_total is None and legacy_reset is None:
            return
        try:
            total = float(legacy_total) if legacy_total is not None else 0.0
        except ValueError:
            total = 0.0
        reset_at = legacy_reset or ""
        self._registry.set(
            SpendState(spend_today=total, daily_reset_at=reset_at)
        )
        # Remove legacy rows so the migration is observable and doesn't
        # fire again — this is safe because the new blob is the single
        # source of truth from here on.
        self._db.delete_config(_LEGACY_SPEND_DAILY_KEY)
        self._db.delete_config(_LEGACY_SPEND_DAILY_RESET_KEY)

    def _load_state(self) -> SpendState:
        if self._registry is None:
            return SpendState()
        try:
            stored = self._registry.get()
        except CARLError:
            # Corrupt/mismatched blob — fall back to an empty state so
            # the tracker keeps working. Next write will overwrite.
            return SpendState()
        return stored if stored is not None else SpendState()

    def _save_state(self, state: SpendState) -> None:
        if self._registry is None:
            return
        self._registry.set(state)

    def _read_daily_total_or_reset(self) -> float:
        if self._registry is None:
            return 0.0
        now = self._now()
        state = self._load_state()
        reset_at: datetime | None = None
        if state.daily_reset_at:
            try:
                parsed = datetime.fromisoformat(state.daily_reset_at)
                # Ensure tz-aware for cross-DST safety.
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                reset_at = parsed
            except ValueError:
                reset_at = None
        if reset_at is None or self._past_midnight(reset_at, now):
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self._save_state(
                SpendState(
                    spend_today=0.0,
                    daily_reset_at=midnight.isoformat(),
                )
            )
            return 0.0
        return state.spend_today

    def _write_daily_total(self, total: float) -> None:
        if self._registry is None:
            return
        state = self._load_state()
        state.spend_today = total
        if not state.daily_reset_at:
            # Populate the reset marker lazily if something cleared it.
            now = self._now()
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            state.daily_reset_at = midnight.isoformat()
        self._save_state(state)

    @staticmethod
    def _past_midnight(reset_at: datetime, now: datetime) -> bool:
        """True iff ``now`` is on a later UTC date than ``reset_at``."""
        return reset_at.date() < now.date()

    @property
    def session_total(self) -> float:
        """USD spent on this tracker instance so far (in-memory)."""
        return self._session_total

    @property
    def daily_cap(self) -> float | None:
        return self._daily_cap

    @property
    def session_cap(self) -> float | None:
        return self._session_cap


class X402Client:
    """Lightweight x402 payment rail client using stdlib urllib.

    Pass an ``InteractionChain`` to ``chain`` to have every check / negotiate
    / execute call recorded as a ``PAYMENT`` step with the facilitator URL,
    amount, token, and success/failure state. When ``chain`` is ``None`` the
    client behaves exactly as before and records nothing.
    """

    def __init__(
        self,
        config: X402Config,
        *,
        chain: InteractionChain | None = None,
        spend_tracker: SpendTracker | None = None,
        confirm_payment_cb: ConfirmPaymentCallback | None = None,
    ) -> None:
        self._config = config
        self._chain = chain
        self._spend_tracker = spend_tracker
        # Constructor-provided callable wins over any config-level value;
        # the config-level value (string name or inline callable) is the
        # fallback resolution path used by private-runtime integrations
        # that only see a persisted config.
        self._confirm_payment_cb = confirm_payment_cb

    def _resolve_confirm_callback(self) -> ConfirmPaymentCallback | None:
        """Resolve the active confirm-payment callback or ``None``.

        Resolution order:

        1. The Callable passed directly to :class:`X402Client`'s
           constructor. Back-compat for call sites that wire hooks in
           code.
        2. ``X402Config.confirm_payment_cb`` — either a string name
           looked up via :func:`get_confirm_callback`, or a callable
           assigned directly on the config.

        A string name that has no registered callback raises
        :class:`~carl_core.errors.CARLError` with
        ``code="carl.x402.callback_unregistered"`` so misconfiguration
        fails loudly at payment time rather than silently auto-approving.
        """
        if self._confirm_payment_cb is not None:
            return self._confirm_payment_cb
        configured = self._config.confirm_payment_cb
        if configured is None:
            return None
        if callable(configured):
            return configured
        # ``configured`` is a string name — resolve through the registry.
        name = configured.strip()
        if not name:
            return None
        resolved = get_confirm_callback(name)
        if resolved is None:
            raise CARLError(
                f"confirm_payment_cb '{name}' is not registered; "
                f"call register_confirm_callback({name!r}, ...) before "
                f"executing payments",
                code="carl.x402.callback_unregistered",
                context={"name": name},
            )
        return resolved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        output: Any = None,
        success: bool = True,
        duration_ms: float | None = None,
    ) -> None:
        if self._chain is None:
            return
        try:
            self._chain.record(
                ActionType.PAYMENT,
                name,
                input=input,
                output=output,
                success=success,
                duration_ms=duration_ms,
            )
        except Exception:
            # Never let trace recording propagate into the payment flow.
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_x402(self, url: str, timeout: int = 10) -> PaymentRequirement | None:
        """HEAD request; parse x-payment header from 402 response.

        Returns None if the URL does not support x402.
        """
        req = urllib.request.Request(url, method="HEAD")
        started = time.monotonic()
        try:
            urllib.request.urlopen(req, timeout=timeout)
            self._record(
                "x402.check:no_payment",
                input={"url": url},
                output={"status": 200},
                success=True,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            return None  # 200 OK — no payment required
        except urllib.error.HTTPError as exc:
            if exc.code != 402:
                self._record(
                    "x402.check:unsupported",
                    input={"url": url},
                    output={"status": exc.code},
                    success=False,
                    duration_ms=(time.monotonic() - started) * 1000,
                )
                return None
            header = exc.headers.get("x-payment", "")
            if not header:
                self._record(
                    "x402.check:402_no_header",
                    input={"url": url},
                    output={"status": 402},
                    success=False,
                    duration_ms=(time.monotonic() - started) * 1000,
                )
                return None
            requirement = _parse_x_payment_header(header)
            self._record(
                "x402.check:402_requirement",
                input={"url": url},
                output={
                    "amount": requirement.amount,
                    "token": requirement.token,
                    "chain": requirement.chain,
                },
                success=True,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            return requirement
        except urllib.error.URLError as exc:
            self._record(
                "x402.check:network_error",
                input={"url": url},
                output={"error": str(exc)},
                success=False,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            return None

    def negotiate(
        self, requirement: PaymentRequirement, timeout: int = 10
    ) -> dict[str, Any]:
        """POST to facilitator to get a payment authorization token."""
        facilitator = requirement.facilitator or self._config.facilitator_url
        if not facilitator:
            self._record(
                "x402.negotiate:missing_facilitator",
                input={
                    "amount": requirement.amount,
                    "token": requirement.token,
                },
                success=False,
            )
            raise X402Error("No facilitator URL configured or in payment requirement.")

        body = json.dumps({
            "amount": requirement.amount,
            "token": requirement.token or self._config.payment_token,
            "chain": requirement.chain or self._config.chain,
            "recipient": requirement.recipient,
            "payer": self._config.wallet_address,
        }).encode()

        req = urllib.request.Request(
            f"{facilitator}/negotiate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            result = json.loads(raw)
            self._record(
                "x402.negotiate",
                input={
                    "facilitator": facilitator,
                    "amount": requirement.amount,
                    "token": requirement.token,
                },
                output={"token_issued": bool(result)},
                success=True,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            return result
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode(errors="replace") if exc.fp else str(exc)
            self._record(
                "x402.negotiate:failed",
                input={"facilitator": facilitator},
                output={"status": exc.code, "message": msg[:200]},
                success=False,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise X402Error(f"Negotiation failed ({exc.code}): {msg}") from exc
        except urllib.error.URLError as exc:
            self._record(
                "x402.negotiate:network_error",
                input={"facilitator": facilitator},
                output={"error": str(exc.reason)},
                success=False,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise X402Error(f"Network error: {exc.reason}") from exc

    def execute(
        self,
        url: str,
        payment_token: str,
        amount: float = 0.0,
        timeout: int = 10,
    ) -> bytes:
        """Replay the original request with x-payment-token header.

        Parameters
        ----------
        url:
            The resource URL that required payment.
        payment_token:
            The facilitator-issued ``X-Payment-Token`` header value.
        amount:
            USD cost of this call. Defaults to ``0.0`` — with
            ``amount == 0`` both the :class:`SpendTracker` check and
            the ``confirm_payment_cb`` hook are skipped, preserving
            legacy call-site behavior.
        timeout:
            Socket timeout (seconds) for the HTTP replay.

        Raises
        ------
        carl_studio.consent.ConsentError
            When the ``CONTRACT_WITNESSING`` consent flag is not
            granted — payment creates a service-contract witness and
            therefore inherits the same consent surface as
            :class:`carl_studio.contract.ContractWitness`.
        carl_core.errors.BudgetError
            When a spend-cap is breached
            (``carl.budget.daily_cap_exceeded`` /
            ``carl.budget.session_cap_exceeded``) or when the confirm-
            payment hook denies the call
            (``carl.budget.confirm_denied``). All three fire *before*
            the consent gate so no witness is recorded on budget denial.
        """
        # Budget check — must fire BEFORE consent_gate so a cap breach
        # never records a contract witness.
        if self._spend_tracker is not None and amount > 0:
            try:
                self._spend_tracker.check_and_record(amount)
            except BudgetError as exc:
                self._record(
                    "x402.budget_exceeded",
                    input={"url": url, "amount": amount},
                    output={"code": exc.code, "message": str(exc)},
                    success=False,
                )
                raise
        # Optional interactive confirmation — after budget check, still
        # before the consent gate. A hook that raises propagates as-is.
        # Resolution considers both the constructor-provided callable
        # and the config-level ``confirm_payment_cb`` (string name or
        # inline callable) so private runtimes that persist only a name
        # get the same user-gate as in-process wiring.
        confirm_cb = self._resolve_confirm_callback() if amount > 0 else None
        if confirm_cb is not None:
            approved = confirm_cb(url=url, amount=amount)
            if not approved:
                self._record(
                    "x402.confirm_denied",
                    input={"url": url, "amount": amount},
                    output={"approved": False},
                    success=False,
                )
                raise BudgetError(
                    f"payment confirmation denied by hook: amount={amount:.4f}",
                    code="carl.budget.confirm_denied",
                    context={"url": url, "amount": amount},
                )
        consent_gate("contract_witnessing")
        req = urllib.request.Request(
            url,
            headers={"X-Payment-Token": payment_token},
            method="GET",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            self._record(
                "x402.execute",
                input={"url": url},
                output={"bytes": len(body)},
                success=True,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            return body
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode(errors="replace") if exc.fp else str(exc)
            self._record(
                "x402.execute:failed",
                input={"url": url},
                output={"status": exc.code, "message": msg[:200]},
                success=False,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise X402Error(f"Payment execution failed ({exc.code}): {msg}") from exc
        except urllib.error.URLError as exc:
            self._record(
                "x402.execute:network_error",
                input={"url": url},
                output={"error": str(exc.reason)},
                success=False,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise X402Error(f"Network error: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# CampProfile projection and persistence
# ---------------------------------------------------------------------------


def x402_config_from_profile(profile: Any) -> X402Config:
    """Project x402 flags from a CampProfile."""
    return X402Config(
        enabled=getattr(profile, "x402_enabled", False),
        wallet_address=getattr(profile, "metadata", {}).get("wallet_address", ""),
        chain=getattr(profile, "metadata", {}).get("x402_chain", "base"),
        facilitator_url=getattr(profile, "metadata", {}).get("x402_facilitator", ""),
    )


def load_x402_config(db: Any | None = None) -> X402Config:
    """Load x402 config from LocalDB."""
    if db is None:
        from carl_studio.db import LocalDB

        db = LocalDB()
    raw = db.get_config(_X402_CONFIG_KEY)
    if not raw:
        return X402Config()
    try:
        return X402Config.model_validate_json(raw)
    except Exception:
        return X402Config()


def save_x402_config(config: X402Config, db: Any | None = None) -> None:
    """Persist x402 config to LocalDB."""
    if db is None:
        from carl_studio.db import LocalDB

        db = LocalDB()
    db.set_config(_X402_CONFIG_KEY, config.model_dump_json())
