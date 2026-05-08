"""Slime ↔ InteractionChain bridge.

Wires THUDM/slime's rollout + training callbacks into CARL's
:class:`carl_core.interaction.InteractionChain` so every slime run
becomes a phi-witnessable, AXON-shape-compatible trace.

This module is the CARL-specific value-add on top of slime. The adapter
in :mod:`carl_studio.adapters.slime_adapter` only orchestrates subprocess
launch; this bridge is what makes the run *coherence-aware*. Users wire
it into slime's ``--custom-reward-fn`` hook (via a dotted path that
resolves to :meth:`SlimeRolloutBridge.reward_fn`) and slime's training
callbacks (PyTorch hooks or slime's ``on_step_end`` handler).

Philosophy:

  * The bridge does not know slime's data shapes. It accepts typed
    :class:`RolloutCompletion` / :class:`TrainingStep` DTOs and leaves
    slime-specific adaptation to the caller's glue function.
  * The reward plug is duck-typed to the same ``score_from_trace``
    signature used by :class:`CARLReward` / :class:`PhaseAdaptiveCARLReward`
    / :class:`EMLCompositeReward`. A tiny adapter
    :class:`CompletionTraceAdapter` shims a raw completion into a
    minimal trace-shaped object so the existing reward heads run
    unchanged.
  * Coherence probe registration is delegated to the chain
    (``chain.register_coherence_probe``) so this bridge stays passive.

Tier: FREE (``train.slime.rollout_bridge``). No autonomy gated here —
this is core capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from carl_core.interaction import ActionType, InteractionChain

if TYPE_CHECKING:
    from carl_core.eml import EMLTree
    from carl_core.resonant import Resonant


def _empty_meta() -> dict[str, Any]:
    return {}


# ---------------------------------------------------------------------------
# Public DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutCompletion:
    """A single rollout sample emitted by slime's rollout engine."""

    prompt: str
    text: str
    logprobs: list[float] | None = None
    token_ids: list[int] | None = None
    rollout_index: int = 0
    meta: dict[str, Any] = field(default_factory=_empty_meta)


@dataclass(frozen=True)
class TrainingStep:
    """A single training-step event emitted by slime's Megatron loop."""

    step: int
    loss: float | None = None
    grad_norm: float | None = None
    lr: float | None = None
    meta: dict[str, Any] = field(default_factory=_empty_meta)


# ---------------------------------------------------------------------------
# Reward plug protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RewardScorer(Protocol):
    """Duck-typed surface that :class:`CARLReward` / ``EMLCompositeReward``
    satisfy natively. A simple ``lambda trace: (float(reward), {})`` works
    equally well.
    """

    def score_from_trace(self, trace: Any) -> tuple[float, dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# Minimal trace shim — adapts a single completion for score_from_trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletionTraceAdapter:
    """Fake-but-useful trace shape that ``score_from_trace`` callers accept.

    ``CARLReward.score_from_trace`` and ``EMLCompositeReward.score_from_trace``
    read a handful of attributes off their trace argument
    (``multiscale_coherence``, ``phi_mean``, ``cloud_quality``, etc.). When
    slime's rollout hook only gives us raw text + (optionally) logprobs, we
    compute a degraded-but-meaningful trace here so the existing reward heads
    run without modification. Consumers that supply a real
    :class:`~carl_core.coherence_trace.CoherenceTrace` should do so directly
    and not pass through this adapter.
    """

    completion: RolloutCompletion

    @property
    def text(self) -> str:
        return self.completion.text

    @property
    def multiscale_coherence(self) -> float:
        """Fallback coherence proxy — mean logprob normalized to [0, 1].

        Real phi measurement requires a CARL forward pass on the model; the
        bridge does not load the model. When callers want true phi, they
        should register a coherence probe on the chain and let the record
        path populate it.
        """
        lp = self.completion.logprobs
        if not lp:
            return 0.5
        mean = sum(lp) / len(lp)
        # Logprobs are in (-inf, 0]; sigmoid-squash to [0, 1].
        import math

        return 1.0 / (1.0 + math.exp(-mean))

    @property
    def phi_mean(self) -> float:
        return self.multiscale_coherence

    @property
    def cloud_quality(self) -> float:
        """Proxy for token-distribution quality — stdev of logprobs,
        squashed to [0, 1]. Zero variance → 0.5 default.
        """
        lp = self.completion.logprobs
        if not lp or len(lp) < 2:
            return 0.5
        mean = sum(lp) / len(lp)
        var = sum((x - mean) ** 2 for x in lp) / len(lp)
        import math

        return 1.0 - math.tanh(math.sqrt(var))

    @property
    def defect_score(self) -> float:
        """Simple defect proxy: fraction of very-low-logprob tokens."""
        lp = self.completion.logprobs
        if not lp:
            return 0.0
        very_low = sum(1 for x in lp if x < -5.0)
        return very_low / len(lp)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class SlimeRolloutBridge:
    """Wire slime rollouts + training events into an InteractionChain.

    Construct once per slime run. The ``reward_fn`` returned by
    :meth:`as_slime_reward` is what you pass through slime's
    ``--custom-reward-fn`` dotted path (after registering the bridge in a
    module-level variable or an explicit registry).

    Typical wiring (user's Python glue):

        bridge = SlimeRolloutBridge(chain, reward=EMLCompositeReward())
        slime.register_reward(bridge.score_completion)  # or equivalent
        slime.register_on_step_end(bridge.record_training_step)
    """

    def __init__(
        self,
        chain: InteractionChain,
        *,
        reward: RewardScorer | None = None,
        run_name: str = "slime-run",
    ) -> None:
        self.chain = chain
        self.reward = reward
        self.run_name = run_name
        self._rollouts_seen = 0
        self._training_steps_seen = 0

    # -- rollout path ---------------------------------------------------

    def score_completion(self, completion: RolloutCompletion) -> float:
        """Record the completion as an LLM_REPLY step, score it, return reward.

        The reward is computed via ``self.reward.score_from_trace`` when a
        :class:`RewardScorer` was supplied; otherwise returns 0.0. Never
        raises — a reward-scoring failure is recorded as a failed step with
        ``success=False`` and a zero reward so slime's training loop keeps
        moving.
        """
        self._rollouts_seen += 1
        reward_value = 0.0
        reward_meta: dict[str, Any] = {}
        success = True
        if self.reward is not None:
            try:
                adapter = CompletionTraceAdapter(completion=completion)
                reward_value, reward_meta = self.reward.score_from_trace(adapter)
            except Exception as exc:  # noqa: BLE001
                success = False
                reward_meta = {"error": str(exc), "error_type": type(exc).__name__}

        self.chain.record(
            ActionType.LLM_REPLY,
            name=f"{self.run_name}.rollout",
            input={
                "prompt": completion.prompt,
                "rollout_index": completion.rollout_index,
            },
            output={
                "text_len": len(completion.text),
                "reward": reward_value,
                "reward_meta": reward_meta,
                "has_logprobs": completion.logprobs is not None,
            },
            success=success,
        )
        return float(reward_value)

    # -- training path --------------------------------------------------

    def record_training_step(self, step: TrainingStep) -> None:
        """Record a training-step event on the chain (no reward computation)."""
        self._training_steps_seen += 1
        self.chain.record(
            ActionType.TRAINING_STEP,
            name=f"{self.run_name}.step",
            input={"step": step.step},
            output={
                "loss": step.loss,
                "grad_norm": step.grad_norm,
                "lr": step.lr,
                **step.meta,
            },
        )

    # -- counters (useful in tests and diagnostics) ---------------------

    @property
    def rollouts_seen(self) -> int:
        return self._rollouts_seen

    @property
    def training_steps_seen(self) -> int:
        return self._training_steps_seen

    # -- artifact emission (v0.16) ---------------------------------------

    def finalize_resonant(
        self,
        *,
        observation_dim: int | None = None,
        extra_metadata: dict[str, Any] | None = None,
        slime_run_id: str | None = None,
    ) -> Resonant:
        """Snapshot the trained reward tree as a publishable Resonant.

        This is CARL's artifact emission point at the end of a slime run.
        When the bridge's reward is an ``EMLCompositeReward`` (or any
        duck-typed scorer exposing ``.tree: EMLTree``), its trained tree
        becomes the cognition step of a ``make_reward_resonant``-built
        :class:`Resonant`. The returned Resonant is ready to be passed to
        :func:`carl_studio.resonant_store.save_resonant` and published
        through ``carl resonant publish`` → carl.camp ``POST /api/resonants``.

        The Resonant uses ``cognition_mode="joint"``: the tree runs once over
        the full feature vector (not per-dim), preserving the reward's
        multi-input semantics.

        Args:
            observation_dim: Caller override for the observation dim. Default
                matches ``reward.tree.input_dim`` (3 for ``EMLCompositeReward``).
            extra_metadata: Optional dict merged into the Resonant's metadata
                after the standard slime-run fields. Useful for caller-specified
                tags (dataset id, experiment name, etc.).
            slime_run_id: Optional UUID for the managed-slime run that
                produced this reward. When set, recorded in the Resonant
                metadata under ``slime_run_id`` AND propagated as the
                ``X-Carl-Slime-Run-Id`` HTTP header on
                ``carl resonant publish`` so carl.camp's resonants route
                can populate ``slime_runs.resonant_id`` for the linkage.
                The route accepts binary octet-stream bodies, so the
                metadata channel is HTTP headers, not envelope JSON
                — see ``docs/eml_signing_protocol.md`` and the F-S3a
                contract correction.

        Raises:
            ValueError: when ``self.reward`` is ``None`` or the reward has no
                ``.tree`` attribute (non-EML reward classes).
        """
        from carl_core.resonant import make_reward_resonant

        if self.reward is None:
            raise ValueError(
                "SlimeRolloutBridge.finalize_resonant(): no reward was "
                "attached to the bridge — nothing to materialize."
            )
        tree = getattr(self.reward, "tree", None)
        if tree is None:
            raise ValueError(
                "SlimeRolloutBridge.finalize_resonant(): bridge reward has no "
                "'.tree' attribute. Only EML-class rewards "
                "(EMLCompositeReward et al) can be materialized as Resonants."
            )
        cast_tree: EMLTree = tree

        meta: dict[str, Any] = {
            "run_name": self.run_name,
            "source": "slime-rollout-bridge",
            "bridge_rollouts_seen": self._rollouts_seen,
            "bridge_training_steps_seen": self._training_steps_seen,
            "reward_class": type(self.reward).__name__,
            "tree_depth": int(cast_tree.depth()),
            "tree_input_dim": int(cast_tree.input_dim),
        }
        if slime_run_id:
            meta["slime_run_id"] = str(slime_run_id)
        if extra_metadata:
            meta.update(extra_metadata)

        resonant = make_reward_resonant(
            cast_tree,
            observation_dim=observation_dim,
            metadata=meta,
        )

        # Record the finalization as a Step in the chain so the trace carries
        # the artifact-emission boundary explicitly. Output is the identity
        # fingerprint only (no tree bytes, no matrix contents).
        step_output: dict[str, Any] = {
            "identity": resonant.identity,
            "tree_depth": meta["tree_depth"],
            "tree_input_dim": meta["tree_input_dim"],
            "rollouts_seen": self._rollouts_seen,
            "training_steps_seen": self._training_steps_seen,
        }
        if slime_run_id:
            step_output["slime_run_id"] = str(slime_run_id)
        self.chain.record(
            ActionType.CHECKPOINT,
            name=f"{self.run_name}.finalize_resonant",
            input={"reward_class": type(self.reward).__name__},
            output=step_output,
        )
        return resonant

    # -- slime-shaped callable adapters ---------------------------------

    def as_slime_reward(self):  # type: ignore[no-untyped-def]
        """Return a callable with the shape slime's ``--custom-reward-fn``
        expects: ``(prompt: str, completion: str, **kwargs) -> float``.

        Slime invokes the reward hook per sample with keyword kwargs
        (``logprobs``, ``meta``, etc.). This adapter builds a
        :class:`RolloutCompletion` from those kwargs and delegates to
        :meth:`score_completion`.
        """

        def _slime_reward(
            prompt: str, completion: str, /, **kwargs: Any
        ) -> float:
            raw_lp = kwargs.get("logprobs")
            logprobs: list[float] | None
            if isinstance(raw_lp, list):
                logprobs = [float(x) for x in raw_lp]  # type: ignore[arg-type]
            else:
                logprobs = None

            raw_ids = kwargs.get("token_ids")
            token_ids: list[int] | None
            if isinstance(raw_ids, list):
                token_ids = [int(x) for x in raw_ids]  # type: ignore[arg-type]
            else:
                token_ids = None

            raw_meta = kwargs.get("meta")
            if isinstance(raw_meta, dict):
                meta = cast(dict[str, Any], raw_meta)
            else:
                meta = {}

            raw_idx = kwargs.get("rollout_index", 0)
            try:
                idx = int(raw_idx)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                idx = 0

            c = RolloutCompletion(
                prompt=prompt,
                text=completion,
                logprobs=logprobs,
                token_ids=token_ids,
                rollout_index=idx,
                meta=meta,
            )
            return self.score_completion(c)

        return _slime_reward


__all__ = [
    "CompletionTraceAdapter",
    "RewardScorer",
    "RolloutCompletion",
    "SlimeRolloutBridge",
    "TrainingStep",
]
