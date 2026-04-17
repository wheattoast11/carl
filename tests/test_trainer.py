"""Tests for CARLTrainer -- no GPU required.

Covers:
  * Original baseline: init, timeout parsing, compute target dispatch, uniqueness.
  * WS-T1 (checkpoint save/resume): _save_carl_checkpoint payload shape,
    save-failure containment, existing-checkpoint detection.
  * WS-T2 (.watch() retry + backoff + max failures): retry-then-success,
    retry exhaustion -> CARLError(code="carl.watch_exhausted"), terminal
    states (completed/failed/canceled), timeout path.
  * WS-P1 (credits synchronous deduction): credit-module missing + no
    --skip-credits raises CARLError; --skip-credits bypasses; post-submission
    failure triggers refund.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from carl_studio.training.trainer import CARL_CHECKPOINT_FILE, CARLTrainer
from carl_studio.types.config import TrainingConfig
from carl_studio.types.run import RunPhase


def _make_config(**overrides: Any) -> TrainingConfig:
    defaults: dict[str, Any] = {
        "run_name": "test-run",
        "base_model": "Qwen/Qwen3-8B",
        "output_repo": "test/test",
        "method": "sft",
        "dataset_repo": "trl-lib/Capybara",
        "compute_target": "local",
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# Baseline (preserved from original suite)
# ---------------------------------------------------------------------------


def test_trainer_init() -> None:
    config = _make_config()
    trainer = CARLTrainer(config)
    assert trainer.run.phase == RunPhase.INITIALIZING
    assert trainer.run.id  # UUID generated


def test_trainer_parse_timeout() -> None:
    assert CARLTrainer._parse_timeout("3h") == 10800
    assert CARLTrainer._parse_timeout("90m") == 5400
    assert CARLTrainer._parse_timeout("14400") == 14400


def test_trainer_parse_timeout_fractional() -> None:
    assert CARLTrainer._parse_timeout("1.5h") == 5400
    assert CARLTrainer._parse_timeout("2.5m") == 150


def test_trainer_parse_timeout_compound() -> None:
    assert CARLTrainer._parse_timeout("2h30m") == 9000


def test_trainer_parse_timeout_empty_returns_default() -> None:
    assert CARLTrainer._parse_timeout("") == 10800  # default 3h


def test_trainer_remote_dispatch() -> None:
    """Test that non-LOCAL targets dispatch to remote mode."""
    config = _make_config(compute_target="l4x1")
    trainer = CARLTrainer(config)
    assert trainer.config.compute_target.value == "l4x1"
    assert trainer.is_remote is True


def test_trainer_local_dispatch() -> None:
    """Test that LOCAL target is not remote."""
    config = _make_config(compute_target="local")
    trainer = CARLTrainer(config)
    assert trainer.is_remote is False


def test_trainer_run_id_unique() -> None:
    c1 = CARLTrainer(_make_config())
    c2 = CARLTrainer(_make_config())
    assert c1.run.id != c2.run.id


def test_trainer_run_has_config() -> None:
    config = _make_config(run_name="my-special-run")
    trainer = CARLTrainer(config)
    assert trainer.run.config.run_name == "my-special-run"


def test_trainer_all_compute_targets() -> None:
    """Every non-LOCAL compute target should be remote."""
    for target in ["l4x1", "l4x4", "a10g-large", "a10g-largex2", "a100-large"]:
        config = _make_config(compute_target=target)
        trainer = CARLTrainer(config)
        assert trainer.is_remote is True, f"{target} should be remote"


def test_trainer_initial_step_zero() -> None:
    trainer = CARLTrainer(_make_config())
    assert trainer.run.current_step == 0
    assert trainer.run.total_steps == 0


# ---------------------------------------------------------------------------
# WS-T1 -- Checkpoint save on crash + resume
# ---------------------------------------------------------------------------


class _FakeTrainerState:
    """Stand-in for HF Trainer.state; enough fields for snapshot."""

    def __init__(self, step: int = 42, history: list[dict[str, Any]] | None = None) -> None:
        self.global_step = step
        self.log_history = history or [{"loss": 1.0}, {"loss": 0.9}]


class _FakeOptim:
    def __init__(self) -> None:
        self._sd = {"param_groups": [{"lr": 1e-5}]}

    def state_dict(self) -> dict[str, Any]:
        return self._sd


class _FakeScheduler:
    def state_dict(self) -> dict[str, Any]:
        return {"last_epoch": 3}


class _FakeHFTrainer:
    """Mimic a HuggingFace Trainer surface area for checkpoint snapshots."""

    def __init__(self, step: int = 42) -> None:
        self.state = _FakeTrainerState(step=step)
        self.optimizer = _FakeOptim()
        self.lr_scheduler = _FakeScheduler()


def test_save_carl_checkpoint_writes_expected_keys(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    fake_trainer = _FakeHFTrainer(step=17)
    exc = RuntimeError("kaboom")

    CARLTrainer._save_carl_checkpoint(fake_trainer, str(tmp_path), exc)

    ckpt_path = tmp_path / CARL_CHECKPOINT_FILE
    assert ckpt_path.exists(), "crash-checkpoint file should be written"

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert payload["step"] == 17
    assert payload["optimizer_state_dict"] == {"param_groups": [{"lr": 1e-5}]}
    assert payload["scheduler_state_dict"] == {"last_epoch": 3}
    assert "rng_state" in payload and isinstance(payload["rng_state"], dict)
    assert "torch" in payload["rng_state"]
    assert payload["metrics"] == [{"loss": 1.0}, {"loss": 0.9}]
    assert payload["exception_type"] == "RuntimeError"
    assert payload["exception_message"] == "kaboom"
    assert "T" in payload["saved_at"]


def test_save_carl_checkpoint_write_failure_does_not_mask_original(tmp_path: Path) -> None:
    """If torch.save explodes inside checkpoint save, original exception propagates unchanged."""
    pytest.importorskip("torch")
    fake_trainer = _FakeHFTrainer()
    original = ValueError("original training failure")

    with patch("torch.save", side_effect=OSError("disk full")):
        CARLTrainer._save_carl_checkpoint(fake_trainer, str(tmp_path), original)

    # No file written, but no crash either -- containment verified.
    assert not (tmp_path / CARL_CHECKPOINT_FILE).exists()


def test_save_carl_checkpoint_handles_missing_optimizer(tmp_path: Path) -> None:
    """Partial trainer state (no optimizer/scheduler) should still persist."""
    torch = pytest.importorskip("torch")

    minimal = types.SimpleNamespace(
        state=_FakeTrainerState(step=5),
        optimizer=None,
        lr_scheduler=None,
    )
    CARLTrainer._save_carl_checkpoint(minimal, str(tmp_path), RuntimeError("boom"))
    payload = torch.load(str(tmp_path / CARL_CHECKPOINT_FILE), weights_only=False)
    assert payload["step"] == 5
    assert payload["optimizer_state_dict"] is None
    assert payload["scheduler_state_dict"] is None


def test_announce_existing_checkpoint_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """If a prior .carl_checkpoint.pt exists, log a warning with step + exception."""
    torch = pytest.importorskip("torch")
    ckpt_path = tmp_path / CARL_CHECKPOINT_FILE
    torch.save(
        {
            "step": 99,
            "exception_type": "KeyboardInterrupt",
            "exception_message": "user cancelled",
            "saved_at": "2026-04-17T00:00:00+00:00",
        },
        str(ckpt_path),
    )

    trainer = CARLTrainer(_make_config())
    with caplog.at_level("WARNING"):
        trainer._announce_existing_checkpoint(str(tmp_path))
    combined = " ".join(r.message for r in caplog.records)
    assert "Prior crash-checkpoint" in combined
    assert "step=99" in combined
    assert "KeyboardInterrupt" in combined


def test_announce_no_checkpoint_is_silent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    trainer = CARLTrainer(_make_config())
    with caplog.at_level("WARNING"):
        trainer._announce_existing_checkpoint(str(tmp_path))
    assert not caplog.records


def test_trainer_accepts_resume_flag() -> None:
    trainer = CARLTrainer(_make_config(), resume_from_checkpoint=True)
    assert trainer.resume_from_checkpoint is True
    assert trainer._resolve_resume_arg("out") is True

    trainer2 = CARLTrainer(_make_config(), resume_from_checkpoint="out/checkpoint-50")
    assert trainer2._resolve_resume_arg("out") == "out/checkpoint-50"

    trainer3 = CARLTrainer(_make_config())  # default
    assert trainer3._resolve_resume_arg("out") is None


# ---------------------------------------------------------------------------
# WS-T2 -- .watch() retry + backoff + max failures
# ---------------------------------------------------------------------------


def test_watch_without_job_id_raises() -> None:
    trainer = CARLTrainer(_make_config(compute_target="l4x1"))
    with pytest.raises(ValueError, match="No job to watch"):
        asyncio.run(trainer.watch())


def test_watch_rejects_invalid_args() -> None:
    trainer = CARLTrainer(_make_config(compute_target="l4x1"))
    trainer.run.hub_job_id = "job-123"
    with pytest.raises(ValueError, match="poll interval must be > 0"):
        asyncio.run(trainer.watch(poll_interval=0))
    with pytest.raises(ValueError, match="max_consecutive_failures"):
        asyncio.run(trainer.watch(max_consecutive_failures=0))


def _patch_get_backend(backend: Any) -> Any:
    return patch("carl_studio.compute.get_backend", return_value=backend)


def test_watch_completes_on_terminal_status() -> None:
    """Status=completed -> phase=COMPLETE, returns cleanly."""
    trainer = CARLTrainer(_make_config(compute_target="l4x1"))
    trainer.run.hub_job_id = "job-ok"

    backend = MagicMock()
    backend.status = AsyncMock(return_value="completed")

    with _patch_get_backend(backend):
        run = asyncio.run(
            trainer.watch(poll_interval_s=0.01, timeout_s=5.0, max_consecutive_failures=5)
        )

    assert run.phase == RunPhase.COMPLETE
    backend.status.assert_awaited()


def test_watch_handles_failed_status() -> None:
    trainer = CARLTrainer(_make_config(compute_target="l4x1"))
    trainer.run.hub_job_id = "job-bad"

    backend = MagicMock()
    backend.status = AsyncMock(return_value="failed")

    with _patch_get_backend(backend):
        run = asyncio.run(
            trainer.watch(poll_interval_s=0.01, timeout_s=5.0, max_consecutive_failures=5)
        )

    assert run.phase == RunPhase.FAILED
    assert run.error_message and "Remote job failed" in run.error_message


def test_watch_handles_canceled_status() -> None:
    trainer = CARLTrainer(_make_config(compute_target="l4x1"))
    trainer.run.hub_job_id = "job-cancel"

    backend = MagicMock()
    backend.status = AsyncMock(return_value="canceled")

    with _patch_get_backend(backend):
        run = asyncio.run(
            trainer.watch(poll_interval_s=0.01, timeout_s=5.0, max_consecutive_failures=5)
        )

    assert run.phase == RunPhase.FAILED
    assert run.error_message == "Job canceled"


def test_watch_retries_then_succeeds() -> None:
    """ConnectionError x2, then completed -> inner retry recovers."""
    trainer = CARLTrainer(_make_config(compute_target="l4x1"))
    trainer.run.hub_job_id = "job-retry"

    call_log: list[str] = []

    async def status_impl(job_id: str) -> str:
        call_log.append(job_id)
        if len(call_log) <= 2:
            raise ConnectionError(f"transient-{len(call_log)}")
        return "completed"

    backend = MagicMock()
    backend.status = AsyncMock(side_effect=status_impl)

    from carl_core.retry import RetryPolicy as RealPolicy

    def _make_policy(*_a: Any, **_kw: Any) -> RealPolicy:
        return RealPolicy(
            max_attempts=3,
            backoff_base=0.0,
            max_delay=0.0,
            jitter=False,
            retryable=(ConnectionError, IOError, TimeoutError),
        )

    with _patch_get_backend(backend), patch(
        "carl_studio.training.trainer.RetryPolicy", side_effect=_make_policy
    ):
        run = asyncio.run(
            trainer.watch(poll_interval_s=0.01, timeout_s=10.0, max_consecutive_failures=5)
        )

    assert run.phase == RunPhase.COMPLETE
    assert len(call_log) == 3, f"expected 3 status calls, got {len(call_log)}"


def test_watch_aborts_after_consecutive_failures() -> None:
    """5 consecutive exhausted retries -> CARLError(code='carl.watch_exhausted')."""
    from carl_core.errors import CARLError

    trainer = CARLTrainer(_make_config(compute_target="l4x1"))
    trainer.run.hub_job_id = "job-dead"

    backend = MagicMock()
    backend.status = AsyncMock(side_effect=ConnectionError("always-fails"))

    from carl_core.retry import RetryPolicy as RealPolicy

    def _make_policy(*_a: Any, **_kw: Any) -> RealPolicy:
        return RealPolicy(
            max_attempts=2,
            backoff_base=0.0,
            max_delay=0.0,
            jitter=False,
            retryable=(ConnectionError, IOError, TimeoutError),
        )

    with _patch_get_backend(backend), patch(
        "carl_studio.training.trainer.RetryPolicy", side_effect=_make_policy
    ):
        with pytest.raises(CARLError) as exc_info:
            asyncio.run(
                trainer.watch(
                    poll_interval_s=0.001,
                    timeout_s=10.0,
                    max_consecutive_failures=3,
                )
            )

    assert exc_info.value.code == "carl.watch_exhausted"
    ctx = exc_info.value.context
    assert ctx["job_id"] == "job-dead"
    assert ctx["consecutive_failures"] >= 3
    assert ctx["last_error_type"] == "ConnectionError"


def test_watch_timeout_path() -> None:
    """Status always 'running' + short timeout -> CARLTimeoutError."""
    from carl_core.errors import CARLTimeoutError

    trainer = CARLTrainer(_make_config(compute_target="l4x1"))
    trainer.run.hub_job_id = "job-slow"

    backend = MagicMock()
    backend.status = AsyncMock(return_value="running")

    with _patch_get_backend(backend):
        with pytest.raises(CARLTimeoutError) as exc_info:
            asyncio.run(
                trainer.watch(
                    poll_interval_s=0.02,
                    timeout_s=0.05,
                    max_consecutive_failures=5,
                )
            )

    assert "watch timed out" in str(exc_info.value)
    assert exc_info.value.context.get("job_id") == "job-slow"


# ---------------------------------------------------------------------------
# WS-P1 -- Credits synchronous deduction
# ---------------------------------------------------------------------------


def test_prededuct_credits_raises_carlerror_when_module_missing() -> None:
    """carl_studio.camp missing + no --skip-credits -> CARLError(credits_failed)."""
    from carl_core.errors import CARLError

    trainer = CARLTrainer(_make_config(compute_target="l4x1"), skip_credits=False)

    with patch.dict(sys.modules, {"carl_studio.camp": None}):
        with pytest.raises(CARLError) as exc_info:
            trainer._prededuct_credits()

    assert exc_info.value.code == "carl.credits_failed"
    assert "credits module unavailable" in str(exc_info.value)


def test_prededuct_credits_skip_flag_bypasses_missing_module() -> None:
    """--skip-credits + missing camp -> (0, None, None), no raise."""
    trainer = CARLTrainer(_make_config(compute_target="l4x1"), skip_credits=True)

    with patch.dict(sys.modules, {"carl_studio.camp": None}):
        result = trainer._prededuct_credits()

    assert result == (0, None, None)


def test_prededuct_credits_no_jwt_returns_zeros() -> None:
    """Authenticated-but-BYOK case: no JWT -> (0, None, None)."""
    trainer = CARLTrainer(_make_config(compute_target="l4x1"), skip_credits=False)

    fake_session = types.SimpleNamespace(jwt=None, supabase_url=None)
    with patch("carl_studio.camp.load_camp_session", return_value=fake_session):
        result = trainer._prededuct_credits()
    assert result == (0, None, None)


def test_prededuct_credits_raises_when_deduct_fails() -> None:
    """deduct_credits() raising CreditError -> CARLError(credits_failed)."""
    from carl_core.errors import CARLError

    trainer = CARLTrainer(
        _make_config(compute_target="l4x1", max_steps=100),
        skip_credits=False,
    )

    fake_session = types.SimpleNamespace(
        jwt="test-jwt",
        supabase_url="https://example.supabase.co",
    )

    from carl_studio.credits.balance import CreditError

    with patch("carl_studio.camp.load_camp_session", return_value=fake_session), patch(
        "carl_studio.credits.balance.deduct_credits",
        side_effect=CreditError("simulated network failure"),
    ), patch("carl_studio.credits.estimate.estimate_job_cost") as mock_estimate:
        mock_estimate.return_value = types.SimpleNamespace(total_with_buffer=42)
        with pytest.raises(CARLError) as exc_info:
            trainer._prededuct_credits()

    assert exc_info.value.code == "carl.credits_failed"
    assert "credit pre-deduction failed" in str(exc_info.value)


def test_prededuct_credits_skip_flag_bypasses_deduct_failure() -> None:
    """--skip-credits + deduct fails -> warn & return zeros, no raise."""
    trainer = CARLTrainer(
        _make_config(compute_target="l4x1", max_steps=100),
        skip_credits=True,
    )

    fake_session = types.SimpleNamespace(
        jwt="test-jwt",
        supabase_url="https://example.supabase.co",
    )

    from carl_studio.credits.balance import CreditError

    with patch("carl_studio.camp.load_camp_session", return_value=fake_session), patch(
        "carl_studio.credits.balance.deduct_credits",
        side_effect=CreditError("boom"),
    ), patch("carl_studio.credits.estimate.estimate_job_cost") as mock_estimate:
        mock_estimate.return_value = types.SimpleNamespace(total_with_buffer=42)
        result = trainer._prededuct_credits()

    assert result == (0, None, None)


def test_prededuct_credits_success_returns_tuple() -> None:
    trainer = CARLTrainer(
        _make_config(compute_target="l4x1", max_steps=100),
        skip_credits=False,
    )
    fake_session = types.SimpleNamespace(
        jwt="test-jwt",
        supabase_url="https://example.supabase.co",
    )
    with patch("carl_studio.camp.load_camp_session", return_value=fake_session), patch(
        "carl_studio.credits.balance.deduct_credits", return_value=True
    ), patch("carl_studio.credits.estimate.estimate_job_cost") as mock_estimate:
        mock_estimate.return_value = types.SimpleNamespace(total_with_buffer=99)
        amount, jwt, url = trainer._prededuct_credits()

    assert amount == 99
    assert jwt == "test-jwt"
    assert url == "https://example.supabase.co"


def test_prededuct_credits_zero_estimate_no_op() -> None:
    trainer = CARLTrainer(_make_config(compute_target="l4x1", max_steps=100))
    fake_session = types.SimpleNamespace(jwt="t", supabase_url="u")
    with patch("carl_studio.camp.load_camp_session", return_value=fake_session), patch(
        "carl_studio.credits.estimate.estimate_job_cost"
    ) as mock_estimate:
        mock_estimate.return_value = types.SimpleNamespace(total_with_buffer=0)
        result = trainer._prededuct_credits()
    assert result == (0, None, None)


def test_refund_credits_no_op_with_zero() -> None:
    trainer = CARLTrainer(_make_config())
    with patch("carl_studio.credits.balance.refund_credits") as mock_refund:
        trainer._refund_credits("jwt", "url", 0)
    mock_refund.assert_not_called()


def test_refund_credits_swallows_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    trainer = CARLTrainer(_make_config())
    with patch(
        "carl_studio.credits.balance.refund_credits",
        side_effect=RuntimeError("edge fn down"),
    ):
        with caplog.at_level("WARNING"):
            trainer._refund_credits("jwt", "url", 10)
    assert any("Credit refund failed" in r.message for r in caplog.records)


def test_train_remote_refunds_on_post_submission_failure() -> None:
    """If execute() raises AFTER provision() succeeds, refund is still issued."""
    trainer = CARLTrainer(
        _make_config(compute_target="l4x1", max_steps=100),
        skip_credits=False,
    )

    with patch.object(
        trainer,
        "_prededuct_credits",
        return_value=(50, "jwt-x", "https://s.co"),
    ), patch.object(trainer, "_refund_credits") as mock_refund, patch(
        "carl_studio.bundler.Bundler"
    ) as mock_bundler_cls, patch(
        "carl_studio.compute.get_backend"
    ) as mock_get_backend:
        mock_bundler_cls.return_value.generate.return_value = "print('hi')"
        backend = MagicMock()
        backend.provision = AsyncMock(return_value="ok")
        backend.execute = AsyncMock(side_effect=RuntimeError("backend down"))
        mock_get_backend.return_value = backend

        run = asyncio.run(trainer.train())

    assert run.phase == RunPhase.FAILED
    assert "backend down" in (run.error_message or "")

    mock_refund.assert_called_once()
    args, _kwargs = mock_refund.call_args
    assert args[0] == "jwt-x"
    assert args[1] == "https://s.co"
    assert args[2] == 50


def test_train_remote_refunds_on_pre_submission_failure() -> None:
    """If bundler raises, refund is also issued (pre-submit path)."""
    trainer = CARLTrainer(
        _make_config(compute_target="l4x1", max_steps=100),
        skip_credits=False,
    )

    with patch.object(
        trainer,
        "_prededuct_credits",
        return_value=(25, "jwt-y", "https://s.co"),
    ), patch.object(trainer, "_refund_credits") as mock_refund, patch(
        "carl_studio.bundler.Bundler"
    ) as mock_bundler_cls:
        mock_bundler_cls.return_value.generate.side_effect = RuntimeError("bundle fail")

        run = asyncio.run(trainer.train())

    assert run.phase == RunPhase.FAILED
    mock_refund.assert_called_once()
    args, _kwargs = mock_refund.call_args
    assert args[0] == "jwt-y"
    assert args[2] == 25


def test_train_remote_no_refund_on_success() -> None:
    trainer = CARLTrainer(
        _make_config(compute_target="l4x1", max_steps=100),
        skip_credits=False,
    )

    with patch.object(
        trainer,
        "_prededuct_credits",
        return_value=(25, "jwt", "url"),
    ), patch.object(trainer, "_refund_credits") as mock_refund, patch(
        "carl_studio.bundler.Bundler"
    ) as mock_bundler_cls, patch(
        "carl_studio.compute.get_backend"
    ) as mock_get_backend:
        mock_bundler_cls.return_value.generate.return_value = "#!/usr/bin/env python"
        backend = MagicMock()
        backend.provision = AsyncMock(return_value="ok")
        backend.execute = AsyncMock(return_value="job-123")
        mock_get_backend.return_value = backend

        run = asyncio.run(trainer.train())

    assert run.phase == RunPhase.TRAINING
    assert run.hub_job_id == "job-123"
    mock_refund.assert_not_called()


# ---------------------------------------------------------------------------
# Back-compat aliases
# ---------------------------------------------------------------------------


def test_try_prededuct_credits_alias_still_works() -> None:
    trainer = CARLTrainer(_make_config(compute_target="l4x1", max_steps=100), skip_credits=True)
    fake_session = types.SimpleNamespace(jwt=None, supabase_url=None)
    with patch("carl_studio.camp.load_camp_session", return_value=fake_session):
        result = trainer._try_prededuct_credits()
    assert result == (0, None, None)


def test_try_refund_credits_alias_still_works() -> None:
    trainer = CARLTrainer(_make_config())
    with patch("carl_studio.credits.balance.refund_credits") as mock_refund:
        trainer._try_refund_credits("jwt", "url", 0)
    mock_refund.assert_not_called()
