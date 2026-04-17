"""Integration seam tests — the paths where things actually break.

Every test here corresponds to a bug that was found in production
and had ZERO test coverage. These are the tests that matter.
"""
import threading

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Seam 1: Package-level imports
# Every public symbol must be reachable through the package API.
# Bug found: rewards/__init__.py didn't export task functions.
# CARLTrainer._load_task_rewards() failed silently.
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_reward_task_functions_importable(self):
        """The 5 task reward functions must be importable from the rewards package."""
        from carl_studio.training.rewards import (
            tool_call_format_reward,
            tool_selection_reward,
            chain_completion_reward,
            neuralese_v2_reward,
            conciseness_reward,
        )
        # Each must be callable
        assert callable(tool_call_format_reward)
        assert callable(tool_selection_reward)
        assert callable(chain_completion_reward)
        assert callable(neuralese_v2_reward)
        assert callable(conciseness_reward)

    def test_make_carl_reward_importable(self):
        from carl_studio.training.rewards import make_carl_reward
        assert callable(make_carl_reward)

    def test_coherence_trace_importable(self):
        from carl_core import CoherenceTrace, select_traces
        assert CoherenceTrace is not None
        assert callable(select_traces)

    def test_trace_callback_importable(self):
        from carl_studio.training import CoherenceTraceCallback
        assert CoherenceTraceCallback is not None

    def test_cascade_importable(self):
        from carl_studio.training import CascadeRewardManager, CascadeCallback
        assert CascadeRewardManager is not None
        assert CascadeCallback is not None


# ---------------------------------------------------------------------------
# Seam 2: _apply_weight propagates ALL CARL attributes
# Bug found: _apply_weight copied _last_metrics but not _last_traces.
# CoherenceTraceCallback couldn't find traces on wrapped functions.
# ---------------------------------------------------------------------------

class TestApplyWeightPropagation:
    def _make_mock_carl_fn(self):
        """Create a mock CARL reward function with all expected attributes."""
        def carl_fn(completions, **kwargs):
            return [0.5] * len(completions)

        carl_fn._last_metrics = [None]
        carl_fn._last_traces = [None]
        carl_fn._last_components = [None]
        carl_fn._metrics_lock = threading.Lock()
        carl_fn._step = [0]
        return carl_fn

    def test_weight_1_returns_original(self):
        from carl_studio.training.trainer import _apply_weight
        fn = self._make_mock_carl_fn()
        wrapped = _apply_weight(fn, 1.0)
        assert wrapped is fn  # no wrapping needed

    def test_propagates_last_traces(self):
        from carl_studio.training.trainer import _apply_weight
        fn = self._make_mock_carl_fn()
        wrapped = _apply_weight(fn, 2.0)
        assert hasattr(wrapped, "_last_traces")
        assert wrapped._last_traces is fn._last_traces

    def test_propagates_last_components(self):
        from carl_studio.training.trainer import _apply_weight
        fn = self._make_mock_carl_fn()
        wrapped = _apply_weight(fn, 0.5)
        assert hasattr(wrapped, "_last_components")
        assert wrapped._last_components is fn._last_components

    def test_propagates_metrics_lock(self):
        from carl_studio.training.trainer import _apply_weight
        fn = self._make_mock_carl_fn()
        wrapped = _apply_weight(fn, 2.0)
        assert hasattr(wrapped, "_metrics_lock")
        assert wrapped._metrics_lock is fn._metrics_lock

    def test_propagates_step(self):
        from carl_studio.training.trainer import _apply_weight
        fn = self._make_mock_carl_fn()
        wrapped = _apply_weight(fn, 2.0)
        assert hasattr(wrapped, "_step")
        assert wrapped._step is fn._step

    def test_weighted_output_scaled(self):
        from carl_studio.training.trainer import _apply_weight
        fn = self._make_mock_carl_fn()
        wrapped = _apply_weight(fn, 3.0)
        result = wrapped(["a", "b"])
        assert result == [1.5, 1.5]  # 0.5 * 3.0

    def test_all_five_attributes_propagated(self):
        """The complete set that callbacks depend on."""
        from carl_studio.training.trainer import _apply_weight
        fn = self._make_mock_carl_fn()
        wrapped = _apply_weight(fn, 2.0)
        required = ["_last_metrics", "_last_traces", "_last_components", "_metrics_lock", "_step"]
        for attr in required:
            assert hasattr(wrapped, attr), f"Missing attribute: {attr}"
            assert getattr(wrapped, attr) is getattr(fn, attr), f"Not same reference: {attr}"


# ---------------------------------------------------------------------------
# Seam 3: CascadeRewardManager 2-stage API matches trainer usage
# Bug found: trainer called CascadeRewardManager(stage_b_start=...)
# but cascade only accepts carl_start. TypeError at runtime.
# ---------------------------------------------------------------------------

class TestCascadeTrainerContract:
    def test_2stage_api_carl_start(self):
        """Trainer uses carl_start — verify it works."""
        from carl_studio.training.cascade import CascadeRewardManager
        cascade = CascadeRewardManager(carl_start=50, warmup_steps=10)
        assert cascade.carl_start == 50
        cascade._step = 0
        assert cascade.get_stage() == "A"
        cascade._step = 50
        assert cascade.get_stage() == "B"

    def test_3stage_api_raises(self):
        """The old 3-stage API must not silently work — it should raise."""
        from carl_studio.training.cascade import CascadeRewardManager
        with pytest.raises(TypeError):
            CascadeRewardManager(stage_b_start=100, stage_c_start=200)

    def test_adaptive_gate_api(self):
        """Trainer may use adaptive gate — verify the API."""
        from carl_studio.training.cascade import CascadeRewardManager
        cascade = CascadeRewardManager(
            gate_metric="task_completion",
            gate_percentile=0.6,
        )
        assert cascade._mode == "adaptive"
        assert cascade.carl_start == float("inf")

    def test_wrap_reward_produces_callable(self):
        from carl_studio.training.cascade import CascadeRewardManager

        cascade = CascadeRewardManager(carl_start=10)

        def reward_fn(completions, **kwargs):
            return [1.0] * len(completions)

        wrapped = cascade.wrap_reward(reward_fn, active_in_stages={"B"})
        assert callable(wrapped)
        assert hasattr(wrapped, "__name__")

    def test_wrap_reward_stage_gating(self):
        from carl_studio.training.cascade import CascadeRewardManager

        cascade = CascadeRewardManager(carl_start=10, warmup_steps=5)

        def reward_fn(completions, **kwargs):
            return [1.0] * len(completions)

        wrapped = cascade.wrap_reward(reward_fn, active_in_stages={"B"})

        # Stage A: gated to zero
        cascade._step = 5
        assert wrapped(["x"]) == [0.0]

        # Stage B, mid-warmup
        cascade._step = 12
        result = wrapped(["x"])
        assert 0.0 < result[0] < 1.0

        # Stage B, post-warmup
        cascade._step = 20
        assert wrapped(["x"]) == [1.0]


# ---------------------------------------------------------------------------
# Seam 4: CoherenceTrace storage on make_carl_reward
# Bug found: _last_traces not stored → CoherenceTraceCallback got nothing.
# ---------------------------------------------------------------------------

class TestCarlRewardTraceStorage:
    def test_make_carl_reward_has_last_traces(self):
        """make_carl_reward closure must expose _last_traces for callbacks."""
        from carl_studio.training.rewards.composite import make_carl_reward

        # Use a trivial mock model
        class MockModel:
            training = False
            device = "cpu"
            def eval(self): pass
            def train(self): pass
            def __call__(self, **kwargs):
                class Out:
                    logits = __import__("torch").randn(1, 32, 100)
                return Out()

        class MockTokenizer:
            vocab_size = 100
            pad_token = "<pad>"
            eos_token = "</s>"
            def __call__(self, text, **kwargs):
                return {"input_ids": __import__("torch").randint(0, 100, (1, 32))}

        fn = make_carl_reward(MockModel(), MockTokenizer())
        assert hasattr(fn, "_last_traces")
        assert hasattr(fn, "_last_metrics")
        assert hasattr(fn, "_last_components")
        assert hasattr(fn, "_metrics_lock")
        assert hasattr(fn, "_step")


# ---------------------------------------------------------------------------
# Seam 5: CoherenceTraceCallback reads from wrapped CARL function
# Bug found: _apply_weight didn't propagate _last_traces, so callback
# couldn't find traces through the cascade-wrapped function.
# ---------------------------------------------------------------------------

class TestTraceCallbackChain:
    def _build_chain(self):
        """Build the full chain: carl_fn → cascade wrap → _apply_weight → callback reads."""
        from carl_core.coherence_trace import CoherenceTrace
        from carl_studio.training.cascade import CascadeRewardManager
        from carl_studio.training.trace_callback import CoherenceTraceCallback
        from carl_studio.training.trainer import _apply_weight

        # Mock CARL reward function with stored traces
        traces = [
            CoherenceTrace.from_logits(
                np.random.randn(64, 100).astype(np.float32),
                np.random.randint(0, 100, 64),
                step=5,
                sample_idx=i,
            )
            for i in range(4)
        ]

        def carl_fn(completions, **kwargs):
            return [0.5] * len(completions)

        carl_fn._last_metrics = [(5, [{"multiscale": 0.5, "cloud_quality": 0.3, "discontinuity": 0.4}] * 4)]
        carl_fn._last_traces = [traces]
        carl_fn._last_components = [None]
        carl_fn._metrics_lock = threading.Lock()
        carl_fn._step = [5]

        # Cascade wrap
        cascade = CascadeRewardManager(carl_start=0)
        cascade._step = 5
        cascade_wrapped = cascade.wrap_reward(carl_fn, active_in_stages={"B"})

        # Weight wrap
        weighted = _apply_weight(cascade_wrapped, 1.5)

        # Build callback
        callback = CoherenceTraceCallback(weighted)

        return callback, traces

    def test_callback_reads_traces_through_chain(self):
        callback, original_traces = self._build_chain()

        class MockState:
            global_step = 5

        logs = {}
        callback.on_log(args=None, state=MockState(), control=None, logs=logs)

        # Callback should have found traces and logged metrics
        assert "trace/phi_mean" in logs
        assert "trace/carl_reward" in logs
        assert logs["trace/phi_mean"] > 0

    def test_callback_logs_backward_compat_keys(self):
        callback, _ = self._build_chain()

        class MockState:
            global_step = 5

        logs = {}
        callback.on_log(args=None, state=MockState(), control=None, logs=logs)

        # Backward compat keys for existing dashboards
        assert "coherence/phi_mean" in logs
        assert "coherence/cloud_quality" in logs

    def test_callback_skips_stale_data(self):
        callback, _ = self._build_chain()

        class MockState:
            global_step = 100  # Far from step 5

        logs = {}
        callback.on_log(args=None, state=MockState(), control=None, logs=logs)

        # Should NOT have logged trace metrics (stale check)
        assert "trace/phi_mean" not in logs


# ---------------------------------------------------------------------------
# Seam 6: CoherenceTrace from_entropy matches from_logits
# This is the TRL-efficient path. If it diverges, the reward signal
# changes when we wire it in.
# ---------------------------------------------------------------------------

class TestFromEntropyFidelity:
    def test_carl_reward_identical(self):
        """The efficient path must produce identical CARL reward."""
        from carl_core.coherence_trace import CoherenceTrace
        import math

        rng = np.random.default_rng(42)
        logits = rng.standard_normal((128, 1000)).astype(np.float32)
        token_ids = logits.argmax(axis=-1)

        trace_full = CoherenceTrace.from_logits(logits, token_ids)

        # Simulate what TRL provides
        entropy = trace_full.entropy
        selected_logprobs = np.log(trace_full.selected_prob + 1e-10)

        trace_trl = CoherenceTrace.from_entropy(
            entropy=entropy,
            selected_logprobs=selected_logprobs,
            vocab_size=1000,
        )

        # CARL reward must match
        assert abs(trace_full.carl_reward() - trace_trl.carl_reward()) < 1e-5

        # Individual components must match
        assert abs(trace_full.multiscale_coherence - trace_trl.multiscale_coherence) < 1e-5
        assert abs(trace_full.cloud_quality - trace_trl.cloud_quality) < 1e-4
        assert abs(trace_full.discontinuity_score - trace_trl.discontinuity_score) < 1e-5

    def test_kuramoto_R_identical(self):
        """Kuramoto R must match between construction paths."""
        from carl_core.coherence_trace import CoherenceTrace

        rng = np.random.default_rng(99)
        logits = rng.standard_normal((64, 500)).astype(np.float32)
        token_ids = logits.argmax(axis=-1)

        trace_full = CoherenceTrace.from_logits(logits, token_ids)
        trace_trl = CoherenceTrace.from_entropy(
            entropy=trace_full.entropy,
            selected_logprobs=np.log(trace_full.selected_prob + 1e-10),
            vocab_size=500,
        )

        assert abs(trace_full.kuramoto_R() - trace_trl.kuramoto_R()) < 1e-6


# ---------------------------------------------------------------------------
# Seam 7: Task reward functions actually produce nonzero output
# Not just "callable" — they return meaningful scores for valid input.
# ---------------------------------------------------------------------------

class TestTaskRewardsProduceSignal:
    def _make_tool_call_completion(self):
        return [{'role': 'assistant', 'content': '```json\n{"name": "read_file", "arguments": {"path": "main.py"}}\n```'}]

    def _make_plain_text_completion(self):
        return [{'role': 'assistant', 'content': 'I would read the file main.py to check the code.'}]

    def test_format_reward_nonzero_for_tool_call(self):
        from carl_studio.training.rewards import tool_call_format_reward
        scores = tool_call_format_reward([self._make_tool_call_completion()])
        assert scores[0] > 0, "Format reward should be nonzero for valid tool call"

    def test_format_reward_zero_for_plain_text(self):
        from carl_studio.training.rewards import tool_call_format_reward
        scores = tool_call_format_reward([self._make_plain_text_completion()])
        assert scores[0] == 0, "Format reward should be zero for plain text"

    def test_conciseness_rewards_short_higher(self):
        from carl_studio.training.rewards import conciseness_reward
        short = [{'role': 'assistant', 'content': '{"name": "run", "arguments": {}}'}]
        long = [{'role': 'assistant', 'content': 'Let me think about this carefully. ' * 50}]
        score_short = conciseness_reward([short])[0]
        score_long = conciseness_reward([long])[0]
        assert score_short >= score_long, "Short completions should score >= long ones"


# ---------------------------------------------------------------------------
# Seam 8: Cascade + CARL reward + callback full assembly
# The closest we can get to the real training pipeline without a GPU.
# ---------------------------------------------------------------------------

class TestFullRewardAssembly:
    def test_reward_chain_shape(self):
        """Assemble the full reward chain and verify shapes."""
        from carl_studio.training.cascade import CascadeRewardManager
        from carl_studio.training.trainer import _apply_weight

        cascade = CascadeRewardManager(carl_start=10, warmup_steps=5)

        def task_reward(completions, **kwargs):
            return [1.0] * len(completions)

        def carl_reward(completions, **kwargs):
            return [0.5] * len(completions)

        # Assemble like trainer does
        task_wrapped = cascade.wrap_reward(task_reward, active_in_stages={"A", "B"})
        carl_wrapped = cascade.wrap_reward(carl_reward, active_in_stages={"B"})

        task_weighted = _apply_weight(task_wrapped, 3.0)
        carl_weighted = _apply_weight(carl_wrapped, 1.5)

        batch = ["completion1", "completion2", "completion3"]

        # Stage A: task active, carl gated
        cascade._step = 5
        assert all(r > 0 for r in task_weighted(batch))
        assert all(r == 0 for r in carl_weighted(batch))

        # Stage B post-warmup: both active
        cascade._step = 20
        task_scores = task_weighted(batch)
        carl_scores = carl_weighted(batch)
        assert all(r == 3.0 for r in task_scores)  # 1.0 * 3.0
        assert all(r == 0.75 for r in carl_scores)  # 0.5 * 1.5
