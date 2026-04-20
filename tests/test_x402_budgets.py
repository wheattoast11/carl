"""H1 — spend-cap enforcement for x402.

Covers :class:`carl_studio.x402.SpendTracker` in isolation and its
integration path through :meth:`X402Client.execute`. A frozen-clock
fixture pins UTC so the "reset at midnight" check is deterministic.
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path) -> LocalDB:
    return LocalDB(db_path=tmp_path / "carl.db")


@pytest.fixture
def frozen_clock():
    """Mutable clock — tests advance ``clock.value`` directly."""

    class Clock:
        value: datetime = datetime(2026, 4, 19, 10, 0, tzinfo=timezone.utc)

        def __call__(self) -> datetime:
            return self.value

    return Clock()


# ---------------------------------------------------------------------------
# SpendTracker unit tests
# ---------------------------------------------------------------------------


class TestSpendTracker:
    def test_spend_tracker_no_caps_allows_unlimited(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        tracker = SpendTracker(db=tmp_db, now=frozen_clock)
        # No caps — should never raise no matter how much we spend.
        for _ in range(100):
            tracker.check_and_record(10.0)
        assert tracker.session_total == pytest.approx(1000.0)

    def test_spend_tracker_daily_cap_blocks_at_boundary(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        tracker = SpendTracker(daily_cap=1.0, db=tmp_db, now=frozen_clock)
        tracker.check_and_record(0.99)  # fine
        with pytest.raises(BudgetError) as exc_info:
            tracker.check_and_record(0.02)  # pushes to 1.01 > 1.0
        assert exc_info.value.code == "carl.budget.daily_cap_exceeded"

    def test_spend_tracker_session_cap_blocks_at_boundary(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        tracker = SpendTracker(session_cap=0.5, db=tmp_db, now=frozen_clock)
        tracker.check_and_record(0.49)
        with pytest.raises(BudgetError) as exc_info:
            tracker.check_and_record(0.02)
        assert exc_info.value.code == "carl.budget.session_cap_exceeded"

    def test_spend_tracker_resets_at_utc_midnight(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        tracker = SpendTracker(daily_cap=1.0, db=tmp_db, now=frozen_clock)
        tracker.check_and_record(0.95)
        # Advance clock past midnight — next day.
        frozen_clock.value = datetime(2026, 4, 20, 0, 5, tzinfo=timezone.utc)
        # Fresh tracker to drop in-memory session state but reuse DB.
        tracker2 = SpendTracker(daily_cap=1.0, db=tmp_db, now=frozen_clock)
        # Should reset — 0.95 of old day is gone.
        tracker2.check_and_record(0.95)
        assert tracker2.session_total == pytest.approx(0.95)

    def test_spend_tracker_persists_across_instances(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        t1 = SpendTracker(daily_cap=10.0, db=tmp_db, now=frozen_clock)
        t1.check_and_record(3.0)
        # New instance — daily total must be read from DB.
        t2 = SpendTracker(daily_cap=10.0, db=tmp_db, now=frozen_clock)
        # 7.01 would succeed after 3.0 prior => projected_daily == 10.01 > 10.
        with pytest.raises(BudgetError) as exc_info:
            t2.check_and_record(7.01)
        assert exc_info.value.code == "carl.budget.daily_cap_exceeded"
        assert exc_info.value.context["daily_total"] == pytest.approx(3.0)

    def test_spend_tracker_rejects_negative_amount(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        tracker = SpendTracker(db=tmp_db, now=frozen_clock)
        with pytest.raises(ValueError):
            tracker.check_and_record(-0.01)

    def test_spend_tracker_budget_error_context_has_totals(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        tracker = SpendTracker(
            daily_cap=2.0, session_cap=10.0, db=tmp_db, now=frozen_clock
        )
        tracker.check_and_record(1.5)
        with pytest.raises(BudgetError) as exc_info:
            tracker.check_and_record(1.0)
        ctx = exc_info.value.context
        assert ctx["amount"] == 1.0
        assert ctx["daily_total"] == pytest.approx(1.5)
        assert ctx["cap"] == 2.0

    def test_spend_tracker_daily_reset_at_stored_at_utc_midnight(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        from carl_studio.x402 import SpendState

        tracker = SpendTracker(daily_cap=5.0, db=tmp_db, now=frozen_clock)
        tracker.check_and_record(1.0)
        # v0.8: state is wrapped in ConfigRegistry[SpendState].
        reg = tmp_db.config_registry(SpendState, namespace="carl.x402")
        state = reg.get()
        assert state is not None
        parsed = datetime.fromisoformat(state.daily_reset_at)
        assert parsed.hour == 0
        assert parsed.minute == 0
        assert parsed.second == 0
        assert parsed.date() == frozen_clock.value.date()

    def test_spend_tracker_in_memory_only_when_no_db(self, frozen_clock) -> None:
        # Without a DB the daily total is ephemeral (always resets to 0).
        # Session cap still enforces against the in-process running total.
        tracker = SpendTracker(session_cap=1.0, now=frozen_clock)
        tracker.check_and_record(0.5)
        tracker.check_and_record(0.5)  # total 1.0 exactly — still fine
        with pytest.raises(BudgetError):
            tracker.check_and_record(0.01)

    def test_spend_tracker_zero_amount_does_not_increment(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        tracker = SpendTracker(daily_cap=0.5, db=tmp_db, now=frozen_clock)
        tracker.check_and_record(0.0)
        tracker.check_and_record(0.5)
        assert tracker.session_total == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# X402Client integration
# ---------------------------------------------------------------------------


def _mock_urlopen_200(body: bytes = b"ok"):
    """Return an object compatible with ``urlopen(...).__enter__`` usage."""

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return body

    return _Resp()


class TestExecuteIntegration:
    def test_execute_without_caps_is_legacy_passthrough(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        """No tracker + no hook — execute() behaves exactly as pre-H1."""
        config = X402Config()
        client = X402Client(config)
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_200(b"data")):
            body = client.execute("https://example.com/r", "tok-1")
        assert body == b"data"

    def test_execute_raises_on_daily_cap_breach_before_consent(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        tracker = SpendTracker(daily_cap=0.01, db=tmp_db, now=frozen_clock)
        client = X402Client(X402Config(), spend_tracker=tracker)
        # urlopen should NEVER be called — budget fires first.
        with patch("urllib.request.urlopen") as urlopen:
            with pytest.raises(BudgetError) as exc_info:
                client.execute(
                    "https://example.com/r", "tok-1", amount=0.10
                )
            assert urlopen.call_count == 0
        assert exc_info.value.code == "carl.budget.daily_cap_exceeded"

    def test_execute_records_budget_exceeded_in_interaction_chain(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        chain = InteractionChain()
        tracker = SpendTracker(daily_cap=0.01, db=tmp_db, now=frozen_clock)
        client = X402Client(
            X402Config(), chain=chain, spend_tracker=tracker
        )
        with patch("urllib.request.urlopen"):
            with pytest.raises(BudgetError):
                client.execute(
                    "https://example.com/r", "tok-1", amount=0.10
                )
        # At least one PAYMENT step with name 'x402.budget_exceeded'.
        names = [step.name for step in chain.steps]
        assert "x402.budget_exceeded" in names

    def test_execute_amount_zero_skips_budget(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        # Even with a 0.0 cap, amount=0 passes the guard (default).
        tracker = SpendTracker(daily_cap=0.0, db=tmp_db, now=frozen_clock)
        client = X402Client(X402Config(), spend_tracker=tracker)
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_200()):
            body = client.execute("https://example.com/r", "tok-1")
        assert body == b"ok"

    def test_execute_success_path_records_to_tracker(
        self, tmp_db: LocalDB, frozen_clock
    ) -> None:
        from carl_studio.x402 import SpendState

        tracker = SpendTracker(daily_cap=10.0, db=tmp_db, now=frozen_clock)
        client = X402Client(X402Config(), spend_tracker=tracker)
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_200()):
            client.execute("https://example.com/r", "tok-1", amount=1.25)
        assert tracker.session_total == pytest.approx(1.25)
        # v0.8: persisted state is a SpendState blob, not a bare float.
        reg = tmp_db.config_registry(SpendState, namespace="carl.x402")
        state = reg.get()
        assert state is not None
        assert state.spend_today == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# v0.8 migration — legacy flat keys → SpendState blob
# ---------------------------------------------------------------------------


class TestSpendStateMigration:
    def test_migration_from_legacy_spend_keys(
        self,
        tmp_db: LocalDB,
        frozen_clock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Legacy keys present → SpendTracker rehydrates, rewrites, deletes."""
        from carl_studio.x402 import SpendState

        monkeypatch.delenv("CARL_CONFIG_MIGRATE", raising=False)
        # Seed the DB with the pre-v0.8 flat keys.
        legacy_reset = datetime(
            2026, 4, 19, 0, 0, tzinfo=timezone.utc
        ).isoformat()
        tmp_db.set_config("carl.x402.spend_today", "4.25")
        tmp_db.set_config("carl.x402.daily_reset_at", legacy_reset)

        # Instantiating the tracker triggers migration.
        tracker = SpendTracker(daily_cap=10.0, db=tmp_db, now=frozen_clock)

        # The legacy keys must be gone.
        assert tmp_db.get_config("carl.x402.spend_today") is None
        assert tmp_db.get_config("carl.x402.daily_reset_at") is None

        # And the new SpendState blob must carry the old values.
        reg = tmp_db.config_registry(SpendState, namespace="carl.x402")
        state = reg.get()
        assert state is not None
        assert state.spend_today == pytest.approx(4.25)
        assert state.daily_reset_at == legacy_reset

        # The tracker should *see* the migrated total — adding 5.0 succeeds
        # (4.25 + 5.0 = 9.25 ≤ cap of 10) but adding 6.0 must fail.
        tracker.check_and_record(5.0)
        with pytest.raises(BudgetError):
            tracker.check_and_record(1.0)  # projected 10.25 > 10.0

    def test_migration_skipped_when_env_says_skip(
        self,
        tmp_db: LocalDB,
        frozen_clock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CARL_CONFIG_MIGRATE=skip leaves legacy keys alone and starts empty."""
        from carl_studio.x402 import SpendState

        monkeypatch.setenv("CARL_CONFIG_MIGRATE", "skip")
        tmp_db.set_config("carl.x402.spend_today", "9.99")
        tmp_db.set_config(
            "carl.x402.daily_reset_at",
            datetime(2026, 4, 19, 0, 0, tzinfo=timezone.utc).isoformat(),
        )

        SpendTracker(daily_cap=10.0, db=tmp_db, now=frozen_clock)

        # Legacy keys must still be there — we didn't touch them.
        assert tmp_db.get_config("carl.x402.spend_today") == "9.99"
        # And no new blob was written.
        reg = tmp_db.config_registry(SpendState, namespace="carl.x402")
        assert reg.get() is None

    def test_migration_is_idempotent(
        self,
        tmp_db: LocalDB,
        frozen_clock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-instantiating SpendTracker after migration is a no-op."""
        from carl_studio.x402 import SpendState

        monkeypatch.delenv("CARL_CONFIG_MIGRATE", raising=False)
        tmp_db.set_config("carl.x402.spend_today", "2.50")
        tmp_db.set_config(
            "carl.x402.daily_reset_at",
            datetime(2026, 4, 19, 0, 0, tzinfo=timezone.utc).isoformat(),
        )

        SpendTracker(db=tmp_db, now=frozen_clock)
        reg = tmp_db.config_registry(SpendState, namespace="carl.x402")
        state_after_first = reg.get()
        assert state_after_first is not None

        # A second tracker should not overwrite or re-migrate.
        SpendTracker(db=tmp_db, now=frozen_clock)
        state_after_second = reg.get()
        assert state_after_second is not None
        assert state_after_second.spend_today == state_after_first.spend_today
        assert (
            state_after_second.daily_reset_at == state_after_first.daily_reset_at
        )
