"""Tests for :mod:`carl_studio.training.slime_bridge`.

The bridge wires slime's rollout + training callbacks into CARL's
:class:`InteractionChain` and computes rewards via a duck-typed
:class:`RewardScorer`. These tests use fake scorers and a real
InteractionChain so they never touch Megatron / SGLang / torch.
"""

from __future__ import annotations

from typing import Any, cast

from carl_core.interaction import ActionType, InteractionChain
from carl_studio.training.slime_bridge import (
    CompletionTraceAdapter,
    RolloutCompletion,
    SlimeRolloutBridge,
    TrainingStep,
)


# ---------------------------------------------------------------------------
# Fake scorers
# ---------------------------------------------------------------------------


class _StaticScorer:
    """Returns a fixed reward and records the trace it was invoked with."""

    def __init__(self, value: float = 1.25) -> None:
        self.value = value
        self.calls: list[Any] = []

    def score_from_trace(self, trace: Any) -> tuple[float, dict[str, Any]]:
        self.calls.append(trace)
        return self.value, {"source": "static-scorer"}


class _ExplodingScorer:
    def score_from_trace(self, trace: Any) -> tuple[float, dict[str, Any]]:
        raise RuntimeError("reward kaboom")


# ---------------------------------------------------------------------------
# score_completion
# ---------------------------------------------------------------------------


def test_score_completion_records_step_and_invokes_scorer() -> None:
    chain = InteractionChain()
    scorer = _StaticScorer(value=0.75)
    bridge = SlimeRolloutBridge(chain, reward=scorer, run_name="t1")

    completion = RolloutCompletion(
        prompt="hello",
        text="world",
        logprobs=[-0.5, -1.0, -0.2],
        rollout_index=3,
    )

    reward = bridge.score_completion(completion)

    assert abs(reward - 0.75) < 1e-9
    assert len(chain) == 1
    step = chain.last()
    assert step is not None
    assert step.action == ActionType.LLM_REPLY
    assert step.name == "t1.rollout"
    assert step.success is True
    output_raw = step.output
    assert isinstance(output_raw, dict)
    output = cast(dict[str, Any], output_raw)
    reward_out = output["reward"]
    assert isinstance(reward_out, float)
    assert abs(reward_out - 0.75) < 1e-9
    assert output["has_logprobs"] is True
    assert bridge.rollouts_seen == 1
    # Scorer was invoked with a CompletionTraceAdapter.
    assert len(scorer.calls) == 1
    assert isinstance(scorer.calls[0], CompletionTraceAdapter)


def test_score_completion_without_reward_returns_zero_but_still_records() -> None:
    chain = InteractionChain()
    bridge = SlimeRolloutBridge(chain, reward=None)

    reward = bridge.score_completion(
        RolloutCompletion(prompt="p", text="c", logprobs=None)
    )

    assert reward == 0.0
    assert len(chain) == 1


def test_score_completion_handles_scorer_failure_gracefully() -> None:
    chain = InteractionChain()
    bridge = SlimeRolloutBridge(chain, reward=_ExplodingScorer())

    reward = bridge.score_completion(
        RolloutCompletion(prompt="p", text="c", logprobs=[-0.1])
    )

    assert reward == 0.0
    step = chain.last()
    assert step is not None
    assert step.success is False
    output_raw = step.output
    assert isinstance(output_raw, dict)
    output = cast(dict[str, Any], output_raw)
    reward_meta = cast(dict[str, Any], output["reward_meta"])
    assert reward_meta["error_type"] == "RuntimeError"
    assert "kaboom" in reward_meta["error"]


# ---------------------------------------------------------------------------
# record_training_step
# ---------------------------------------------------------------------------


def test_record_training_step_emits_training_step_action() -> None:
    chain = InteractionChain()
    bridge = SlimeRolloutBridge(chain, run_name="run-A")

    bridge.record_training_step(
        TrainingStep(step=42, loss=1.23, grad_norm=0.5, lr=1e-6, meta={"extra": "ok"})
    )

    assert len(chain) == 1
    step = chain.last()
    assert step is not None
    assert step.action == ActionType.TRAINING_STEP
    assert step.name == "run-A.step"
    assert step.input == {"step": 42}
    output_raw = step.output
    assert isinstance(output_raw, dict)
    output = cast(dict[str, Any], output_raw)
    loss_val = output["loss"]
    assert isinstance(loss_val, float)
    assert abs(loss_val - 1.23) < 1e-9
    assert output["extra"] == "ok"
    assert bridge.training_steps_seen == 1


# ---------------------------------------------------------------------------
# CompletionTraceAdapter
# ---------------------------------------------------------------------------


def test_completion_trace_adapter_multiscale_coherence_defaults() -> None:
    adapter = CompletionTraceAdapter(
        completion=RolloutCompletion(prompt="p", text="c", logprobs=None)
    )
    assert adapter.multiscale_coherence == 0.5
    assert adapter.phi_mean == 0.5
    assert adapter.cloud_quality == 0.5
    assert adapter.defect_score == 0.0


def test_completion_trace_adapter_multiscale_increases_with_better_logprobs() -> None:
    low = CompletionTraceAdapter(
        completion=RolloutCompletion(prompt="p", text="c", logprobs=[-8.0, -9.0])
    )
    high = CompletionTraceAdapter(
        completion=RolloutCompletion(prompt="p", text="c", logprobs=[-0.1, -0.1])
    )
    assert high.multiscale_coherence > low.multiscale_coherence
    # defect_score counts tokens strictly below -5.0. Both samples in `low`
    # qualify; none in `high` do.
    assert low.defect_score > high.defect_score
    assert high.defect_score == 0.0


# ---------------------------------------------------------------------------
# as_slime_reward — shim for slime's native reward hook
# ---------------------------------------------------------------------------


def test_as_slime_reward_adapts_positional_plus_kwargs() -> None:
    chain = InteractionChain()
    scorer = _StaticScorer(value=0.3)
    bridge = SlimeRolloutBridge(chain, reward=scorer)

    slime_cb = bridge.as_slime_reward()
    reward = slime_cb(
        "user prompt",
        "model output",
        logprobs=[-0.2, -0.4],
        token_ids=[10, 20],
        meta={"source_id": "dataset-1"},
        rollout_index=5,
    )

    assert abs(reward - 0.3) < 1e-9
    assert bridge.rollouts_seen == 1
    step = chain.last()
    assert step is not None
    step_input_raw = step.input
    assert isinstance(step_input_raw, dict)
    step_input = cast(dict[str, Any], step_input_raw)
    assert step_input["rollout_index"] == 5


def test_as_slime_reward_tolerates_missing_kwargs() -> None:
    chain = InteractionChain()
    bridge = SlimeRolloutBridge(chain, reward=_StaticScorer(value=0.1))

    reward = bridge.as_slime_reward()("p", "c")
    assert abs(reward - 0.1) < 1e-9
    assert bridge.rollouts_seen == 1


def test_as_slime_reward_coerces_bad_kwarg_types() -> None:
    chain = InteractionChain()
    bridge = SlimeRolloutBridge(chain, reward=_StaticScorer(value=0.0))

    # logprobs not a list → treated as None; meta not a dict → treated as {}.
    reward = bridge.as_slime_reward()(
        "p", "c", logprobs="garbage", meta="also-garbage", rollout_index="nan"
    )
    assert reward == 0.0
    assert bridge.rollouts_seen == 1
