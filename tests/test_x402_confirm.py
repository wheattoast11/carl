"""H1 — ``confirm_payment`` hook semantics for x402 execute.

The hook runs *after* the spend-cap check passes and *before* the
consent gate, so a denial never produces a contract witness and a
pre-authorized call still honors interactive user control.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from carl_core.errors import BudgetError
from carl_core.interaction import InteractionChain
from carl_studio.db import LocalDB
from carl_studio.x402 import SpendTracker, X402Client, X402Config


@pytest.fixture
def tmp_db(tmp_path: Path) -> LocalDB:
    return LocalDB(db_path=tmp_path / "carl.db")


@pytest.fixture
def frozen_clock():
    class Clock:
        value = datetime(2026, 4, 19, 10, 0, tzinfo=timezone.utc)

        def __call__(self) -> datetime:
            return self.value

    return Clock()


def _mock_urlopen_200(body: bytes = b"ok"):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return body

    return _Resp()


class TestConfirmHook:
    def test_confirm_payment_hook_called_with_url_and_amount(
        self, tmp_db: LocalDB
    ) -> None:
        calls: list[dict] = []

        def hook(*, url: str, amount: float) -> bool:
            calls.append({"url": url, "amount": amount})
            return True

        client = X402Client(X402Config(), confirm_payment_cb=hook)
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_200()):
            client.execute("https://example.com/x", "tok-1", amount=0.25)
        assert len(calls) == 1
        assert calls[0]["url"] == "https://example.com/x"
        assert calls[0]["amount"] == 0.25

    def test_confirm_payment_hook_returns_false_raises(
        self, tmp_db: LocalDB
    ) -> None:
        def hook(*, url: str, amount: float) -> bool:
            return False

        client = X402Client(X402Config(), confirm_payment_cb=hook)
        with patch("urllib.request.urlopen") as urlopen:
            with pytest.raises(BudgetError) as exc_info:
                client.execute("https://example.com/x", "tok-1", amount=0.5)
            assert urlopen.call_count == 0
        assert exc_info.value.code == "carl.budget.confirm_denied"

    def test_confirm_payment_hook_raises_propagates(self) -> None:
        class HookError(RuntimeError):
            pass

        def hook(*, url: str, amount: float) -> bool:
            raise HookError("user aborted")

        client = X402Client(X402Config(), confirm_payment_cb=hook)
        with patch("urllib.request.urlopen"):
            with pytest.raises(HookError):
                client.execute("https://example.com/x", "tok-1", amount=0.5)

    def test_confirm_payment_hook_none_with_caps_auto_approves(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        tracker = SpendTracker(daily_cap=5.0, db=tmp_db, now=frozen_clock)
        client = X402Client(X402Config(), spend_tracker=tracker)
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_200()):
            client.execute("https://example.com/x", "tok-1", amount=0.5)
        assert tracker.session_total == pytest.approx(0.5)

    def test_confirm_payment_hook_none_with_no_caps_auto_approves(
        self, tmp_db: LocalDB
    ) -> None:
        client = X402Client(X402Config())
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_200()):
            body = client.execute("https://example.com/x", "tok-1", amount=5.0)
        assert body == b"ok"

    def test_confirm_payment_fires_after_budget_check_passes(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        """Budget breach must beat the hook — the hook never runs."""
        calls: list[dict] = []

        def hook(*, url: str, amount: float) -> bool:
            calls.append({"url": url, "amount": amount})
            return True

        tracker = SpendTracker(daily_cap=0.01, db=tmp_db, now=frozen_clock)
        client = X402Client(
            X402Config(), spend_tracker=tracker, confirm_payment_cb=hook
        )
        with patch("urllib.request.urlopen"):
            with pytest.raises(BudgetError) as exc_info:
                client.execute("https://example.com/x", "tok-1", amount=0.10)
        assert exc_info.value.code == "carl.budget.daily_cap_exceeded"
        assert calls == []  # hook did NOT run

    def test_confirm_payment_denial_records_to_chain(
        self, tmp_db: LocalDB
    ) -> None:
        chain = InteractionChain()

        def hook(*, url: str, amount: float) -> bool:
            return False

        client = X402Client(
            X402Config(), chain=chain, confirm_payment_cb=hook
        )
        with patch("urllib.request.urlopen"):
            with pytest.raises(BudgetError):
                client.execute("https://example.com/x", "tok-1", amount=0.5)
        names = [step.name for step in chain.steps]
        assert "x402.confirm_denied" in names

    def test_confirm_payment_does_not_fire_when_amount_zero(
        self, tmp_db: LocalDB
    ) -> None:
        calls: list[dict] = []

        def hook(*, url: str, amount: float) -> bool:
            calls.append({"amount": amount})
            return True

        client = X402Client(X402Config(), confirm_payment_cb=hook)
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_200()):
            client.execute("https://example.com/x", "tok-1")  # default 0.0
        assert calls == []
