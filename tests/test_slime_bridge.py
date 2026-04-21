"""Tests for :mod:`carl_studio.training.slime_bridge`.

The bridge wires slime's rollout + training callbacks into CARL's
:class:`InteractionChain` and computes rewards via a duck-typed
:class:`RewardScorer`. These tests use fake scorers and a real
InteractionChain so they never touch Megatron / SGLang / torch.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

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


# ---------------------------------------------------------------------------
# finalize_resonant — v0.16 artifact emission
# ---------------------------------------------------------------------------


def test_finalize_resonant_requires_reward_attached() -> None:
    chain = InteractionChain()
    bridge = SlimeRolloutBridge(chain, reward=None)
    with pytest.raises(ValueError, match="no reward was attached"):
        bridge.finalize_resonant()


def test_finalize_resonant_requires_reward_with_tree_attribute() -> None:
    chain = InteractionChain()
    bridge = SlimeRolloutBridge(chain, reward=_StaticScorer(value=1.0))
    with pytest.raises(ValueError, match="no '.tree' attribute"):
        bridge.finalize_resonant()


def test_finalize_resonant_emits_joint_mode_resonant() -> None:
    """End-to-end: EML-shaped reward → Resonant with the trained tree."""
    from carl_core.eml import EMLTree

    tree = EMLTree.identity(input_dim=3)

    class _EMLScorer:
        """Duck-typed EMLCompositeReward stand-in."""

        def __init__(self, tree: EMLTree) -> None:
            self.tree = tree

        def score_from_trace(self, _trace: Any) -> tuple[float, dict[str, Any]]:
            return 0.5, {"source": "eml-like"}

    chain = InteractionChain()
    bridge = SlimeRolloutBridge(
        chain,
        reward=_EMLScorer(tree),
        run_name="slime-run-A",
    )

    resonant = bridge.finalize_resonant(observation_dim=3)

    assert resonant.cognition_mode == "joint"
    assert resonant.action_dim == 1
    assert resonant.observation_dim == 3
    assert resonant.latent_dim == 3  # matches tree.input_dim
    assert len(resonant.identity) == 64

    # Metadata captures run provenance without leaking values.
    meta = resonant.metadata
    assert meta["run_name"] == "slime-run-A"
    assert meta["source"] == "slime-rollout-bridge"
    assert meta["reward_class"] == "_EMLScorer"
    assert meta["tree_input_dim"] == 3
    assert meta["bridge_rollouts_seen"] == 0
    assert meta["bridge_training_steps_seen"] == 0


def test_finalize_resonant_records_checkpoint_step() -> None:
    """The finalization must leave a CHECKPOINT step in the chain."""
    from carl_core.eml import EMLTree
    from carl_core.interaction import ActionType

    class _EMLScorer:
        def __init__(self, tree: EMLTree) -> None:
            self.tree = tree

        def score_from_trace(self, _t: Any) -> tuple[float, dict[str, Any]]:
            return 0.0, {}

    tree = EMLTree.identity(input_dim=3)
    chain = InteractionChain()
    bridge = SlimeRolloutBridge(chain, reward=_EMLScorer(tree), run_name="run-B")

    resonant = bridge.finalize_resonant()

    # Must have recorded exactly one step: the CHECKPOINT boundary.
    assert len(chain) == 1
    step = chain.last()
    assert step is not None
    assert step.action == ActionType.CHECKPOINT
    assert step.name == "run-B.finalize_resonant"

    output_raw = step.output
    assert isinstance(output_raw, dict)
    output = cast(dict[str, Any], output_raw)
    # Output records the identity fingerprint but NOT the tree bytes or matrices.
    assert output["identity"] == resonant.identity
    assert "tree_bytes" not in output
    assert "projection" not in output
    assert "readout" not in output


def test_finalize_resonant_counters_persist_into_metadata() -> None:
    from carl_core.eml import EMLTree

    class _EMLScorer:
        def __init__(self, tree: EMLTree) -> None:
            self.tree = tree

        def score_from_trace(self, _t: Any) -> tuple[float, dict[str, Any]]:
            return 0.1, {}

    tree = EMLTree.identity(input_dim=3)
    chain = InteractionChain()
    bridge = SlimeRolloutBridge(chain, reward=_EMLScorer(tree))

    # Exercise some rollouts + training steps so the counters bump.
    for i in range(4):
        bridge.score_completion(
            RolloutCompletion(prompt="p", text="c", rollout_index=i)
        )
    for s in range(2):
        bridge.record_training_step(TrainingStep(step=s))

    resonant = bridge.finalize_resonant()
    assert resonant.metadata["bridge_rollouts_seen"] == 4
    assert resonant.metadata["bridge_training_steps_seen"] == 2


def test_finalize_resonant_extra_metadata_merges() -> None:
    from carl_core.eml import EMLTree

    class _EMLScorer:
        def __init__(self, tree: EMLTree) -> None:
            self.tree = tree

        def score_from_trace(self, _t: Any) -> tuple[float, dict[str, Any]]:
            return 0.0, {}

    tree = EMLTree.identity(input_dim=3)
    chain = InteractionChain()
    bridge = SlimeRolloutBridge(chain, reward=_EMLScorer(tree))

    resonant = bridge.finalize_resonant(
        extra_metadata={"dataset": "yourorg/grpo-prompts", "gpu_hours": 12.5},
    )
    assert resonant.metadata["dataset"] == "yourorg/grpo-prompts"
    assert resonant.metadata["gpu_hours"] == 12.5
    # Standard fields are not clobbered.
    assert resonant.metadata["source"] == "slime-rollout-bridge"
