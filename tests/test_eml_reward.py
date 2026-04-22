"""Tests for EMLCompositeReward — the depth-3 EML tree reward head.

Covers:
  1. Drop-in initialization (output close to CARLReward on same trace)
     — requires the benchmark-tuned coefficients, so marked ``@pytest.mark.private``
     (skips cleanly when the resonance runtime is unavailable).
  2. Gradient sanity (loss decreases under fit_step) — structural, always-on.
  3. Serialization round-trip preserves reward output exactly — structural.
  4. Depth enforcement (max_depth > MAX_DEPTH raises) — structural.
  5. Factory dispatch via make_carl_reward(reward_class="eml") — structural.
  6. Integration shape (matches PhaseAdaptiveCARLReward signature) — structural.
  7. Determinism (same tree + same trace -> same reward) — structural.

Post-v0.17 moat extraction: the +0.972 benchmark correlation with
PhaseAdaptive + drop-in CARL approximation property belong to the private
runtime (``resonance.rewards.eml_weights``). Public random-init fallback
produces a working but un-tuned reward, so the drop-in assertions live
under ``@pytest.mark.private`` and exercise the ``admin_unlocked`` fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from carl_core.coherence_trace import CoherenceTrace
from carl_core.eml import MAX_DEPTH, EMLNode, EMLOp, EMLTree

from carl_studio.training.rewards import (
    CARLReward,
    PhaseAdaptiveCARLReward,
    make_eml_reward,
)
from carl_studio.training.rewards.eml import (
    EMLCompositeReward,
    eml_reward_from_trace,
)

from tests.conftest import skip_if_private_unavailable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trace(seed: int = 0, T: int = 16, V: int = 32) -> CoherenceTrace:
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((T, V))
    token_ids = rng.integers(0, V, size=T)
    return CoherenceTrace.from_logits(logits, token_ids)


@pytest.fixture
def trace() -> CoherenceTrace:
    return _make_trace(seed=42)


@pytest.fixture
def reward() -> EMLCompositeReward:
    return make_eml_reward()


# ---------------------------------------------------------------------------
# 1. Initialization — drop-in close to CARLReward
#
# Drop-in parity with CARLReward requires the benchmark-tuned calibration
# constants + tuned tree topology shipped in the private runtime
# (``resonance.rewards.eml_weights``). Tests that assert on the
# benchmark-accurate +0.972 correlation live under ``@pytest.mark.private``
# and exercise the ``admin_unlocked`` fixture. Structural tests
# (metadata shape, bounded output, finite output) stay always-on because
# the public random-init fallback preserves those invariants.
# ---------------------------------------------------------------------------


class TestDropInInit:
    @pytest.mark.private
    def test_output_close_to_carl_reward(
        self,
        admin_unlocked: bool,
        trace: CoherenceTrace,
    ) -> None:
        """On the same trace, EML reward must land within 0.3 abs of CARL.

        Benchmark-tuned property — requires the private runtime's
        calibration constants.
        """
        skip_if_private_unavailable(admin_unlocked)
        carl = CARLReward()
        eml = make_eml_reward()
        s_carl, _ = carl.score_from_trace(trace)
        s_eml, _meta = eml.score_from_trace(trace)
        assert abs(s_carl - s_eml) < 0.3, (
            f"EML reward {s_eml:.4f} too far from CARL {s_carl:.4f}; "
            f"diff={abs(s_carl - s_eml):.4f}"
        )

    @pytest.mark.private
    def test_output_close_across_many_traces(self, admin_unlocked: bool) -> None:
        """Drop-in property must hold on a distribution of traces, not one.

        Benchmark-tuned property — requires the private runtime's
        calibration constants.
        """
        skip_if_private_unavailable(admin_unlocked)
        carl = CARLReward()
        eml = make_eml_reward()
        diffs = []
        for seed in range(20):
            tr = _make_trace(seed=seed)
            s_carl, _ = carl.score_from_trace(tr)
            s_eml, _ = eml.score_from_trace(tr)
            diffs.append(abs(s_carl - s_eml))
        assert max(diffs) < 0.3, (
            f"drop-in failed on {sum(d >= 0.3 for d in diffs)} / {len(diffs)} "
            f"traces; max diff = {max(diffs):.4f}"
        )

    def test_metadata_shape(self, reward: EMLCompositeReward, trace: CoherenceTrace) -> None:
        _, meta = reward.score_from_trace(trace)
        assert set(meta.keys()) == {"ms", "cq", "defect", "tree_depth"}
        assert meta["tree_depth"] == 3.0
        assert 0.0 <= meta["ms"] <= 1.0
        assert 0.0 <= meta["cq"] <= 1.0
        assert 0.0 <= meta["defect"] <= 1.0

    def test_output_finite(self, reward: EMLCompositeReward, trace: CoherenceTrace) -> None:
        """No NaN/inf in the output ever."""
        s, _ = reward.score_from_trace(trace)
        assert np.isfinite(s)

    def test_output_clamped(self, reward: EMLCompositeReward) -> None:
        """Even for extreme feature inputs the reward stays in the clamp range."""
        # Feed adversarial inputs that would explode without guards.
        lo, hi = reward.clamp
        for feats in [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1e-10, 1e-10, 1e-10]),
            np.array([10.0, 10.0, 10.0]),
        ]:
            out = reward.forward(feats)
            assert lo <= out <= hi, f"{out} not in [{lo}, {hi}] for feats {feats}"


# ---------------------------------------------------------------------------
# 2. Depth-1 baseline
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_eml_reward_from_trace_returns_finite(self, trace: CoherenceTrace) -> None:
        r = eml_reward_from_trace(trace)
        assert np.isfinite(r)
        assert -5.0 <= r <= 5.0

    def test_eml_reward_from_trace_deterministic(self, trace: CoherenceTrace) -> None:
        assert eml_reward_from_trace(trace) == eml_reward_from_trace(trace)


# ---------------------------------------------------------------------------
# 3. Gradient sanity
# ---------------------------------------------------------------------------


class TestGradient:
    def test_fit_step_decreases_loss(self) -> None:
        """50 Adam steps on a fixed target must monotonically trend downward."""
        reward = make_eml_reward()
        x = np.array([0.8, 0.5, 0.6])
        target = 0.1
        losses: list[float] = []
        for _ in range(50):
            result = reward.fit_step(x, target, lr=5e-2)
            losses.append(result["loss"])
        # The last loss must be strictly less than the first (strong condition).
        assert losses[-1] < losses[0], (
            f"fit_step failed to decrease loss: start={losses[0]:.4f} end={losses[-1]:.4f}"
        )
        # Median of last 10 should be much lower than median of first 10.
        assert float(np.median(losses[-10:])) < float(np.median(losses[:10]))

    def test_fit_step_returns_expected_keys(self) -> None:
        reward = make_eml_reward()
        result = reward.fit_step(np.array([0.5, 0.5, 0.5]), 0.5)
        assert set(result.keys()) == {"loss", "grad_norm"}
        assert np.isfinite(result["loss"])
        assert np.isfinite(result["grad_norm"])
        assert result["grad_norm"] >= 0.0

    def test_fit_step_zero_params_noop(self) -> None:
        """Tree with no CONST leaves (only VAR_X) still returns sensible dict."""
        root = EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.EML,
                         left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
                         right=EMLNode(op=EMLOp.VAR_X, var_idx=1)),
            right=EMLNode(op=EMLOp.VAR_X, var_idx=2),
        )
        tree = EMLTree(root=root, input_dim=3)
        reward = EMLCompositeReward(tree=tree, max_depth=3)
        result = reward.fit_step(np.array([0.5, 0.5, 0.5]), 0.5)
        assert result["grad_norm"] == 0.0


# ---------------------------------------------------------------------------
# 4. Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_from_dict_preserves_reward(self, trace: CoherenceTrace) -> None:
        """Exact reward output must be preserved across to_dict/from_dict."""
        reward = make_eml_reward()
        # Do some training to perturb leaf params.
        for _ in range(20):
            reward.fit_step(np.array([0.7, 0.4, 0.5]), 0.3, lr=1e-2)
        s_before, meta_before = reward.score_from_trace(trace)
        d = reward.to_dict()
        revived = EMLCompositeReward.from_dict(d)
        s_after, meta_after = revived.score_from_trace(trace)
        assert s_before == pytest.approx(s_after, abs=1e-10), (
            f"reward drifted on round-trip: {s_before} vs {s_after}"
        )
        assert meta_before == meta_after

    def test_to_dict_contains_required_keys(self) -> None:
        reward = make_eml_reward()
        d = reward.to_dict()
        for key in ("version", "max_depth", "clamp", "root", "input_dim",
                    "leaf_params", "adam_m", "adam_v", "adam_t"):
            assert key in d

    def test_from_dict_preserves_adam_state(self) -> None:
        """Adam moments and step counter survive a serialization cycle."""
        reward = make_eml_reward()
        for _ in range(10):
            reward.fit_step(np.array([0.5, 0.5, 0.5]), 0.2, lr=1e-2)
        d = reward.to_dict()
        revived = EMLCompositeReward.from_dict(d)
        np.testing.assert_array_equal(revived._adam_m, reward._adam_m)
        np.testing.assert_array_equal(revived._adam_v, reward._adam_v)
        assert revived._adam_t == reward._adam_t


# ---------------------------------------------------------------------------
# 5. Depth enforcement
# ---------------------------------------------------------------------------


class TestDepthEnforcement:
    def test_max_depth_above_ceiling_raises(self) -> None:
        with pytest.raises(ValueError, match="MAX_DEPTH"):
            EMLCompositeReward(max_depth=MAX_DEPTH + 1)

    def test_tree_deeper_than_max_depth_raises(self) -> None:
        """Passing a tree deeper than max_depth must be rejected."""
        # Build a depth-2 tree, then try to wrap with max_depth=1.
        root = EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.EML,
                         left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
                         right=EMLNode(op=EMLOp.VAR_X, var_idx=1)),
            right=EMLNode(op=EMLOp.VAR_X, var_idx=2),
        )
        tree = EMLTree(root=root, input_dim=3)
        with pytest.raises(ValueError, match="depth"):
            EMLCompositeReward(tree=tree, max_depth=1)

    def test_default_tree_within_bounds(self) -> None:
        reward = make_eml_reward()
        assert reward.tree.depth() <= MAX_DEPTH
        assert reward.tree.depth() <= reward.max_depth


# ---------------------------------------------------------------------------
# 6. Factory dispatch
# ---------------------------------------------------------------------------


class TestFactory:
    def test_make_carl_reward_eml_returns_eml_composite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """make_carl_reward(reward_class="eml") must construct an EMLCompositeReward."""
        from carl_studio.training.rewards import composite as comp

        # Patch torch to avoid a hard torch import in the CI environment.
        import sys
        if "torch" not in sys.modules:
            pytest.skip("torch not installed — factory uses torch closure")

        # We don't need a real model/tokenizer for the construction test —
        # make_carl_reward builds the reward object before returning the
        # closure, so we can probe the branch via exception on bad reward_class.
        class _FakeModel:
            device = "cpu"
            training = False
            def eval(self): return self
            def train(self): return self

        class _FakeTok:
            def __call__(self, *a, **k):
                raise RuntimeError("tokenizer should not be called for construction")

        closure = comp.make_carl_reward(
            model=_FakeModel(),
            tokenizer=_FakeTok(),
            reward_class="eml",
        )
        # The factory returns a closure; the underlying reward is in its closure
        # cell. We verify no ValueError was raised — the "eml" branch was hit.
        assert callable(closure)

    def test_make_carl_reward_unknown_class_raises(self) -> None:
        from carl_studio.training.rewards import composite as comp

        class _FakeModel:
            device = "cpu"
            training = False
            def eval(self): return self
            def train(self): return self

        import sys
        if "torch" not in sys.modules:
            pytest.skip("torch not installed — factory uses torch closure")

        with pytest.raises(ValueError, match="reward_class"):
            comp.make_carl_reward(
                model=_FakeModel(),
                tokenizer=None,
                reward_class="not-a-real-class",
            )


# ---------------------------------------------------------------------------
# 7. Integration shape (plug-compat with existing rewards)
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_signature_matches_phase_adaptive(self, trace: CoherenceTrace) -> None:
        """EML and PhaseAdaptive must return the same tuple shape."""
        eml = make_eml_reward()
        pa = PhaseAdaptiveCARLReward()
        s_eml, meta_eml = eml.score_from_trace(trace)
        s_pa, meta_pa = pa.score_from_trace(trace)
        assert isinstance(s_eml, float)
        assert isinstance(s_pa, float)
        assert isinstance(meta_eml, dict)
        assert isinstance(meta_pa, dict)

    def test_callable_via_dunder_call(self, trace: CoherenceTrace) -> None:
        reward = make_eml_reward()
        s1, m1 = reward(trace)
        s2, m2 = reward.score_from_trace(trace)
        assert s1 == s2
        assert m1 == m2


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_tree_same_trace_same_reward_100x(self, trace: CoherenceTrace) -> None:
        """100 back-to-back calls with no state mutation must return the same float."""
        reward = make_eml_reward()
        first, _ = reward.score_from_trace(trace)
        for _ in range(100):
            s, _ = reward.score_from_trace(trace)
            assert s == first

    def test_fresh_instances_agree(self, trace: CoherenceTrace) -> None:
        """Two freshly-constructed default rewards must produce the same output."""
        r1 = make_eml_reward()
        r2 = make_eml_reward()
        s1, _ = r1.score_from_trace(trace)
        s2, _ = r2.score_from_trace(trace)
        assert s1 == s2
