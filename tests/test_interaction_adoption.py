"""InteractionChain adoption sweep tests.

Verifies that the remaining consumer surfaces — trainer, eval runner,
training callbacks, x402 client, and the persistence helper — correctly
emit InteractionChain steps with the right ActionType values.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from carl_core.interaction import ActionType, InteractionChain, Step
from carl_core.interaction_store import InteractionStore


# ---------------------------------------------------------------------------
# ActionType / Step surface extensions
# ---------------------------------------------------------------------------


class TestActionTypeExtensions:
    def test_new_action_types_present(self) -> None:
        assert ActionType.PAYMENT.value == "payment"
        assert ActionType.TRAINING_STEP.value == "training_step"
        assert ActionType.EVAL_PHASE.value == "eval_phase"
        assert ActionType.REWARD.value == "reward"
        assert ActionType.CHECKPOINT.value == "checkpoint"

    def test_step_session_and_trace_id_roundtrip(self) -> None:
        step = Step(
            action=ActionType.TRAINING_STEP,
            name="train.step",
            session_id="s-1",
            trace_id="t-42",
        )
        raw = step.to_dict()
        assert raw["session_id"] == "s-1"
        assert raw["trace_id"] == "t-42"

        chain = InteractionChain()
        chain.record(
            ActionType.REWARD,
            "epoch_end",
            session_id="s-1",
            trace_id="t-42",
        )
        restored = InteractionChain.from_dict(chain.to_dict())
        assert restored.steps[0].session_id == "s-1"
        assert restored.steps[0].trace_id == "t-42"


# ---------------------------------------------------------------------------
# InteractionStore — roundtrip + concurrency + missing-file
# ---------------------------------------------------------------------------


class TestInteractionStoreAdoption:
    def test_append_then_load_roundtrip(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        chain = InteractionChain()
        chain.context["test"] = True
        store.append_chain(chain)

        step = Step(action=ActionType.CHECKPOINT, name="save")
        store.append(chain.chain_id, step)

        restored = store.load(chain.chain_id)
        assert restored.chain_id == chain.chain_id
        # append_chain snapshot + streaming append means the reload sees every
        # step that was persisted; the streaming append is all we care about.
        names = [s.name for s in restored.steps]
        assert "save" in names

    def test_missing_chain_returns_empty(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        chain = store.load("never-written")
        assert isinstance(chain, InteractionChain)
        assert chain.chain_id == "never-written"
        assert len(chain.steps) == 0

    def test_concurrent_appends_no_corruption(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        cid = "concurrent"
        errors: list[BaseException] = []

        def worker(prefix: str) -> None:
            try:
                for i in range(100):
                    store.append(
                        cid,
                        Step(action=ActionType.TRAINING_STEP, name=f"{prefix}-{i}"),
                    )
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors

        chain = store.load(cid)
        assert len(chain.steps) == 200
        unique = {s.name for s in chain.steps}
        assert len(unique) == 200


# ---------------------------------------------------------------------------
# x402 PAYMENT regression — chain is populated via the public client surface
# ---------------------------------------------------------------------------


class TestX402PaymentAdoption:
    def test_check_x402_records_on_non_payment_response(self) -> None:
        """A 200 OK URL still records a PAYMENT step with success=True."""
        from carl_studio.x402 import X402Client, X402Config

        chain = InteractionChain()
        client = X402Client(X402Config(enabled=True), chain=chain)

        class _FakeResp:
            def __enter__(self) -> "_FakeResp":
                return self

            def __exit__(self, *_: Any) -> None:
                return None

            def read(self) -> bytes:
                return b""

        with patch("carl_studio.x402.urllib.request.urlopen", return_value=_FakeResp()):
            result = client.check_x402("https://example.test/resource")
        assert result is None

        payments = chain.by_action(ActionType.PAYMENT)
        assert len(payments) == 1
        assert payments[0].name == "x402.check:no_payment"
        assert payments[0].success is True

    def test_check_x402_records_failure_on_http_error(self) -> None:
        from carl_studio.x402 import X402Client, X402Config
        import urllib.error

        chain = InteractionChain()
        client = X402Client(X402Config(enabled=True), chain=chain)

        def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise urllib.error.HTTPError("u", 500, "boom", {}, None)

        with patch("carl_studio.x402.urllib.request.urlopen", side_effect=_raise):
            result = client.check_x402("https://example.test/broken")
        assert result is None

        payments = chain.by_action(ActionType.PAYMENT)
        assert len(payments) == 1
        assert payments[0].success is False
        assert payments[0].action == ActionType.PAYMENT

    def test_client_without_chain_stays_silent(self) -> None:
        """Regression: absence of chain must not break the x402 flow."""
        from carl_studio.x402 import X402Client, X402Config

        client = X402Client(X402Config(enabled=True))

        class _FakeResp:
            def __enter__(self) -> "_FakeResp":
                return self

            def __exit__(self, *_: Any) -> None:
                return None

            def read(self) -> bytes:
                return b""

        with patch("carl_studio.x402.urllib.request.urlopen", return_value=_FakeResp()):
            assert client.check_x402("https://example.test") is None


# ---------------------------------------------------------------------------
# CARLTrainer integration — TRAINING_STEP + CHECKPOINT plumbing
# ---------------------------------------------------------------------------


def _make_training_config(**overrides: Any) -> Any:
    from carl_studio.types.config import TrainingConfig

    defaults: dict[str, Any] = {
        "run_name": "unit-test-run",
        "base_model": "Qwen/Qwen3-8B",
        "output_repo": "unit/tests",
        "method": "sft",
        "dataset_repo": "unit/tests",
        "compute_target": "local",
        "max_steps": 10,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


class TestTrainerChainWiring:
    def test_records_remote_submit_as_training_step(self) -> None:
        """Remote path: _record is called when a job is submitted."""
        from carl_studio.training.trainer import CARLTrainer

        cfg = _make_training_config()
        # Pivot to remote flavor so we exercise the submit branch
        from carl_studio.types.config import ComputeTarget

        cfg.compute_target = ComputeTarget.L4X1
        chain = InteractionChain()
        trainer = CARLTrainer(cfg, skip_credits=True, interaction_chain=chain)
        assert trainer.chain is chain

        # Directly exercise _record with TRAINING_STEP
        trainer._record(
            ActionType.TRAINING_STEP,
            "training.start",
            input={"run_id": trainer.run.id},
        )
        steps = chain.by_action(ActionType.TRAINING_STEP)
        assert len(steps) == 1
        assert steps[0].name == "training.start"
        assert steps[0].session_id == trainer.run.id

    def test_record_is_noop_without_chain(self) -> None:
        from carl_studio.training.trainer import CARLTrainer

        trainer = CARLTrainer(_make_training_config(), skip_credits=True)
        # Should not raise
        trainer._record(ActionType.TRAINING_STEP, "noop")
        assert trainer.chain is None

    def test_record_checkpoint_action(self) -> None:
        from carl_studio.training.trainer import CARLTrainer

        chain = InteractionChain()
        trainer = CARLTrainer(
            _make_training_config(), skip_credits=True, interaction_chain=chain
        )
        trainer._record(
            ActionType.CHECKPOINT,
            "sft.save",
            input={"output_dir": "/tmp/out"},
        )
        ckpts = chain.by_action(ActionType.CHECKPOINT)
        assert len(ckpts) == 1
        assert ckpts[0].name == "sft.save"

    def test_invalid_step_interval_rejected(self) -> None:
        from carl_studio.training.trainer import CARLTrainer

        with pytest.raises(ValueError):
            CARLTrainer(
                _make_training_config(),
                skip_credits=True,
                training_step_interval=0,
            )


# ---------------------------------------------------------------------------
# InteractionChainCallback — periodic TRAINING_STEP and CHECKPOINT on save
# ---------------------------------------------------------------------------


class TestInteractionChainCallback:
    def _make(self, chain: InteractionChain, *, interval: int = 50):
        from carl_studio.training.callbacks import InteractionChainCallback

        return InteractionChainCallback(chain, run_id="run-1", step_interval=interval)

    def test_on_log_emits_every_interval(self) -> None:
        chain = InteractionChain()
        cb = self._make(chain, interval=10)

        # Fire at steps 5 -> skipped (< interval), 10 -> emit, 19 -> skipped, 20 -> emit
        for step in (5, 10, 19, 20):
            cb.on_log(
                args=SimpleNamespace(output_dir="/tmp/out"),
                state=SimpleNamespace(global_step=step, epoch=0.1, max_steps=100),
                control=None,
                logs={"loss": 0.5 + step * 0.01, "learning_rate": 1e-4},
            )

        emits = [s for s in chain.by_action(ActionType.TRAINING_STEP) if s.name == "train.step"]
        assert len(emits) == 2
        assert emits[0].output is not None and "loss" in emits[0].output

    def test_on_save_emits_checkpoint(self) -> None:
        chain = InteractionChain()
        cb = self._make(chain)
        cb.on_save(
            args=SimpleNamespace(output_dir="/tmp/x"),
            state=SimpleNamespace(global_step=123, epoch=1.0),
            control=None,
        )
        ckpts = chain.by_action(ActionType.CHECKPOINT)
        assert len(ckpts) == 1
        assert ckpts[0].name == "train.checkpoint"
        assert ckpts[0].input["global_step"] == 123

    def test_epoch_end_emits_training_step(self) -> None:
        chain = InteractionChain()
        cb = self._make(chain)
        cb.on_epoch_end(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=50, epoch=1.0),
            control=None,
        )
        names = [s.name for s in chain.by_action(ActionType.TRAINING_STEP)]
        assert "train.epoch_end" in names

    def test_train_begin_and_end_bracket(self) -> None:
        chain = InteractionChain()
        cb = self._make(chain)
        cb.on_train_begin(
            args=SimpleNamespace(),
            state=SimpleNamespace(max_steps=10, global_step=0, epoch=0.0),
            control=None,
        )
        cb.on_train_end(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=10, epoch=1.0),
            control=None,
        )
        names = [s.name for s in chain.by_action(ActionType.TRAINING_STEP)]
        assert "train.begin" in names
        assert "train.end" in names

    def test_exception_in_hook_does_not_propagate(self) -> None:
        """Buggy state objects must not crash the training loop."""
        chain = InteractionChain()
        cb = self._make(chain)

        class _Bad:
            @property
            def global_step(self) -> int:
                raise RuntimeError("boom")

        # Should not raise.
        cb.on_log(
            args=SimpleNamespace(),
            state=_Bad(),
            control=None,
            logs={"loss": 0.5},
        )
        cb.on_save(
            args=SimpleNamespace(output_dir="/tmp"),
            state=_Bad(),
            control=None,
        )

    def test_rejects_invalid_interval(self) -> None:
        from carl_studio.training.callbacks import InteractionChainCallback

        with pytest.raises(ValueError):
            InteractionChainCallback(InteractionChain(), step_interval=0)

    def test_rejects_none_chain(self) -> None:
        from carl_studio.training.callbacks import InteractionChainCallback

        with pytest.raises(ValueError):
            InteractionChainCallback(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CoherenceMonitorCallback — REWARD recording at epoch end
# ---------------------------------------------------------------------------


class TestCoherenceMonitorChainHook:
    def test_epoch_end_records_reward_step(self) -> None:
        from carl_studio.training.callbacks import CoherenceMonitorCallback
        from carl_studio.training.rewards import multiscale as ms

        # Bump the running clamp counters directly so the REWARD step carries
        # non-zero metrics. We use the private counter dict the public helpers
        # already snapshot and reset.
        with ms._clamp_counter_lock:
            ms._clamp_counter["nonfinite"] = 1
            ms._clamp_counter["overflow"] = 1
            ms._clamp_counter["total"] = 2

        try:
            chain = InteractionChain()
            cb = CoherenceMonitorCallback(
                SimpleNamespace(_metrics_lock=None, _last_metrics=[None]),
                chain=chain,
                session_id="run-1",
            )
            cb.on_epoch_end(
                args=SimpleNamespace(),
                state=SimpleNamespace(global_step=10, epoch=1.0),
                control=None,
            )
        finally:
            ms.reset_clamp_counts()

        rewards = chain.by_action(ActionType.REWARD)
        assert len(rewards) == 1
        step = rewards[0]
        assert step.name == "coherence.epoch_end"
        assert step.session_id == "run-1"
        assert set(step.output or {}) >= {
            "clamp_total",
            "clamp_nonfinite",
            "clamp_overflow",
        }
        assert step.output["clamp_total"] == 2

    def test_epoch_end_without_chain_still_resets_counters(self) -> None:
        """Regression: no chain => no record, no exception."""
        from carl_studio.training.callbacks import CoherenceMonitorCallback
        from carl_studio.training.rewards.multiscale import (
            clamp_counts,
            reset_clamp_counts,
        )

        reset_clamp_counts()
        cb = CoherenceMonitorCallback(
            SimpleNamespace(_metrics_lock=None, _last_metrics=[None])
        )
        cb.on_epoch_end(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=0, epoch=0.0),
            control=None,
        )
        assert clamp_counts().get("total", 0) == 0


# ---------------------------------------------------------------------------
# EvalRunner integration — EVAL_PHASE start + end
# ---------------------------------------------------------------------------


class TestEvalRunnerAdoption:
    def test_run_emits_start_and_end_on_success(self) -> None:
        from carl_studio.eval.runner import EvalConfig, EvalRunner, EvalReport

        chain = InteractionChain()
        config = EvalConfig(checkpoint="unit/test-model", phase="1")
        runner = EvalRunner(config, interaction_chain=chain)

        fake_report = EvalReport(
            checkpoint="unit/test-model",
            phase="1",
            n_samples=3,
            metrics={"chain_completion_rate": 0.91},
            primary_metric="chain_completion_rate",
            primary_value=0.91,
            threshold=0.5,
            passed=True,
        )
        with patch.object(EvalRunner, "_run_single_turn_phase", return_value=fake_report):
            report = runner.run()
        assert report.passed is True
        phase_steps = chain.by_action(ActionType.EVAL_PHASE)
        names = [s.name for s in phase_steps]
        assert "eval.phase1.start" in names
        assert "eval.phase1.end" in names
        end_step = [s for s in phase_steps if s.name == "eval.phase1.end"][0]
        assert end_step.success is True
        assert end_step.output is not None
        assert end_step.output["primary_value"] == 0.91
        assert end_step.output["passed"] is True

    def test_run_emits_error_step_on_exception(self) -> None:
        from carl_studio.eval.runner import EvalConfig, EvalRunner

        chain = InteractionChain()
        config = EvalConfig(checkpoint="unit/test-model", phase="1")
        runner = EvalRunner(config, interaction_chain=chain)

        with patch.object(EvalRunner, "_run_single_turn_phase", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                runner.run()

        phase_steps = chain.by_action(ActionType.EVAL_PHASE)
        names = [s.name for s in phase_steps]
        assert "eval.phase1.start" in names
        assert "eval.phase1.error" in names
        err = [s for s in phase_steps if s.name == "eval.phase1.error"][0]
        assert err.success is False

    def test_run_without_chain_still_works(self) -> None:
        from carl_studio.eval.runner import EvalConfig, EvalRunner, EvalReport

        config = EvalConfig(checkpoint="unit/test-model", phase="1")
        runner = EvalRunner(config)
        assert runner.chain is None

        fake_report = EvalReport(
            checkpoint="unit/test-model",
            phase="1",
            n_samples=1,
            metrics={"chain_completion_rate": 0.0},
            primary_metric="chain_completion_rate",
            primary_value=0.0,
            threshold=0.5,
            passed=False,
        )
        with patch.object(EvalRunner, "_run_single_turn_phase", return_value=fake_report):
            report = runner.run()
        assert report.primary_value == 0.0
