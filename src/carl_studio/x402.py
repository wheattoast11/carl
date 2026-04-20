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
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from carl_core.errors import BudgetError, CARLError, NetworkError
from carl_core.interaction import ActionType, InteractionChain
from carl_core.retry import CircuitBreaker

from carl_studio.consent import consent_gate

if TYPE_CHECKING:
    from carl_studio.db import LocalDB

_X402_CONFIG_KEY = "x402_config"

# SpendTracker persistence keys — stored in LocalDB.config
_SPEND_DAILY_KEY = "carl.x402.spend_today"
_SPEND_DAILY_RESET_KEY = "carl.x402.daily_reset_at"

# Module-level breaker for facilitator calls. Only infrastructure failures
# count against the threshold — programming bugs in our own code (attribute
# errors, type errors, ...) propagate unchanged so callers see the real
# traceback instead of a misleading "circuit_open" error.
_FACILITATOR_BREAKER = CircuitBreaker(
    failure_threshold=5,
    reset_s=60.0,
    tracked_exceptions=(NetworkError, ConnectionError, TimeoutError, IOError),
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
    confirm_payment_cb: str | None = None  # registered hook name; None = auto-approve within caps


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


class SpendTracker:
    """Rolling-window spend accounting for x402 payments.

    Persists the daily total and a daily-reset timestamp to
    :class:`~carl_studio.db.LocalDB` config storage. Session total is
    held in-process only (no cross-process leakage between shells).
    Cap enforcement is synchronous and happens before any network call
    so a breach raises immediately without partial state.

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

    def _read_daily_total_or_reset(self) -> float:
        if self._db is None:
            return 0.0
        now = self._now()
        raw_reset = self._db.get_config(_SPEND_DAILY_RESET_KEY)
        reset_at: datetime | None = None
        if raw_reset:
            try:
                parsed = datetime.fromisoformat(raw_reset)
                # Ensure tz-aware for cross-DST safety.
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                reset_at = parsed
            except ValueError:
                reset_at = None
        if reset_at is None or self._past_midnight(reset_at, now):
            self._db.set_config(_SPEND_DAILY_KEY, "0.0")
            midnight = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            self._db.set_config(_SPEND_DAILY_RESET_KEY, midnight.isoformat())
            return 0.0
        raw = self._db.get_config(_SPEND_DAILY_KEY) or "0.0"
        try:
            return float(raw)
        except ValueError:
            return 0.0

    def _write_daily_total(self, total: float) -> None:
        if self._db is None:
            return
        self._db.set_config(_SPEND_DAILY_KEY, str(total))

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
        confirm_payment_cb: Callable[..., bool] | None = None,
    ) -> None:
        self._config = config
        self._chain = chain
        self._spend_tracker = spend_tracker
        self._confirm_payment_cb = confirm_payment_cb

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
        if self._confirm_payment_cb is not None and amount > 0:
            approved = self._confirm_payment_cb(url=url, amount=amount)
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
