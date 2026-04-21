"""T3 — EML-smoothed CoherenceGatePredicate tests.

Pins down the two-mode contract on :class:`CoherenceGatePredicate`:

1. ``use_eml_smoothing=False`` (default) preserves the pre-v0.8.1 hard
   cliff byte-for-byte — every allow/deny decision is identical to the
   prior implementation.
2. ``use_eml_smoothing=True`` swaps in the EML kernel
   ``exp(R) - ln(min_R)`` with threshold ``tau``; the allow/deny
   boundary becomes ``score > tau`` and the margin is smooth.

The Adam-trainable story requires that a sample just below ``min_R``
produce a score *near* ``tau`` (not a cliff), a sample well above
``min_R`` produce a large positive margin, and a sample well below
``min_R`` produce a large negative margin. The tests below pin each
regime.
"""
from __future__ import annotations

import math

import pytest

from carl_core.interaction import ActionType, InteractionChain
from carl_studio.gating import CoherenceGatePredicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_with_R(values: list[float]) -> InteractionChain:
    chain = InteractionChain()
    for v in values:
        chain.record(
            ActionType.LLM_REPLY,
            "eml.fixture",
            input={"seq": len(chain.steps)},
            output={"r": v},
            success=True,
            kuramoto_r=v,
        )
    return chain


def _eml(x: float, y: float) -> float:
    return math.exp(x) - math.log(max(y, 1e-12))


# ---------------------------------------------------------------------------
# 1. Default path unchanged — smoothing off is byte-for-byte equal to the
#    pre-existing hard-cliff implementation.
# ---------------------------------------------------------------------------


class TestDefaultPathUnchanged:
    def test_default_use_eml_smoothing_is_false(self) -> None:
        pred = CoherenceGatePredicate(min_R=0.5, chain=None)
        assert pred.use_eml_smoothing is False
        assert pred.tau is None

    def test_no_data_still_allows(self) -> None:
        pred = CoherenceGatePredicate(min_R=0.5, chain=None)
        allowed, reason = pred.check()
        assert allowed is True
        assert "no coherence data" in reason

    def test_above_threshold_allows_hard_cliff(self) -> None:
        chain = _chain_with_R([0.8, 0.85, 0.9])
        pred = CoherenceGatePredicate(min_R=0.5, chain=chain)
        allowed, reason = pred.check()
        assert allowed is True
        # Old reason string shape is preserved.
        assert "R=" in reason
        assert ">=" in reason

    def test_below_threshold_denies_hard_cliff(self) -> None:
        chain = _chain_with_R([0.1, 0.2, 0.3])
        pred = CoherenceGatePredicate(min_R=0.5, chain=chain)
        allowed, reason = pred.check()
        assert allowed is False
        assert "below required" in reason

    def test_at_threshold_allows_hard_cliff(self) -> None:
        # R >= min_R is allow (equality included) — same as pre-existing.
        chain = _chain_with_R([0.5])
        pred = CoherenceGatePredicate(min_R=0.5, chain=chain)
        allowed, _ = pred.check()
        assert allowed is True


# ---------------------------------------------------------------------------
# 2. Smooth path toggled — use_eml_smoothing=True takes the EML branch.
# ---------------------------------------------------------------------------


class TestSmoothPathToggled:
    def test_smooth_flag_stored(self) -> None:
        pred = CoherenceGatePredicate(
            min_R=0.5, chain=None, use_eml_smoothing=True
        )
        assert pred.use_eml_smoothing is True

    def test_smooth_name_flags_eml(self) -> None:
        pred = CoherenceGatePredicate(
            min_R=0.5, chain=None, use_eml_smoothing=True
        )
        assert "eml" in pred.name

    def test_no_data_still_allows_smooth_mode(self) -> None:
        pred = CoherenceGatePredicate(
            min_R=0.5, chain=None, use_eml_smoothing=True
        )
        allowed, reason = pred.check()
        assert allowed is True
        assert "no coherence data" in reason


# ---------------------------------------------------------------------------
# 3. Near-threshold regime — the smooth path produces a result close to
#    tau, not a cliff.
# ---------------------------------------------------------------------------


class TestNearThresholdRegime:
    def test_just_below_min_R_is_near_threshold(self) -> None:
        # With min_R=0.5 and R=0.499, score is ~0.001 below tau — the
        # distance is small and continuous.
        chain = _chain_with_R([0.499])
        pred = CoherenceGatePredicate(
            min_R=0.5, chain=chain, use_eml_smoothing=True
        )
        allowed, reason = pred.check()
        # Below tau → deny — but the margin is tiny, not a cliff.
        tau = math.exp(0.5) - math.log(0.5)
        score = _eml(0.499, 0.5)
        assert allowed is False
        assert abs(score - tau) < 0.01
        assert "eml" in reason

    def test_at_min_R_is_neutral(self) -> None:
        # R == min_R → score == tau (default tau). Strict ``>`` → deny at
        # equality, just like ``>`` would. The score is, however, neutral.
        chain = _chain_with_R([0.5])
        pred = CoherenceGatePredicate(
            min_R=0.5, chain=chain, use_eml_smoothing=True
        )
        allowed, reason = pred.check()
        tau = math.exp(0.5) - math.log(0.5)
        score = _eml(0.5, 0.5)
        assert score == pytest.approx(tau)
        # Strict > tau boundary → deny at equality (smoothness budget)
        assert allowed is False
        assert "eml" in reason


# ---------------------------------------------------------------------------
# 4. Far-from-threshold regimes — large positive/negative margins.
# ---------------------------------------------------------------------------


class TestFarFromThresholdRegimes:
    def test_well_above_min_R_produces_large_positive_margin(self) -> None:
        chain = _chain_with_R([0.95])
        pred = CoherenceGatePredicate(
            min_R=0.5, chain=chain, use_eml_smoothing=True
        )
        allowed, reason = pred.check()
        tau = math.exp(0.5) - math.log(0.5)
        score = _eml(0.95, 0.5)
        assert allowed is True
        assert score - tau > 0.5
        assert "eml" in reason

    def test_well_below_min_R_denies_with_negative_margin(self) -> None:
        chain = _chain_with_R([0.05])
        pred = CoherenceGatePredicate(
            min_R=0.5, chain=chain, use_eml_smoothing=True
        )
        allowed, reason = pred.check()
        tau = math.exp(0.5) - math.log(0.5)
        score = _eml(0.05, 0.5)
        assert allowed is False
        assert score - tau < -0.5
        assert "eml" in reason


# ---------------------------------------------------------------------------
# 5. tau default is sensible — at R = min_R the score is neutral.
# ---------------------------------------------------------------------------


class TestTauDefault:
    def test_default_tau_neutral_at_min_R(self) -> None:
        chain = _chain_with_R([0.3])
        pred = CoherenceGatePredicate(
            min_R=0.3, chain=chain, use_eml_smoothing=True
        )
        pred.check()
        tau = math.exp(0.3) - math.log(0.3)
        score = _eml(0.3, 0.3)
        assert score == pytest.approx(tau)

    def test_explicit_tau_shifts_policy_stricter(self) -> None:
        # Raising tau above the neutral default should deny a sample
        # that the default would allow at the margin.
        chain = _chain_with_R([0.6])
        default_tau = math.exp(0.5) - math.log(0.5)
        strict = CoherenceGatePredicate(
            min_R=0.5,
            chain=chain,
            use_eml_smoothing=True,
            tau=default_tau + 100.0,  # absurdly strict
        )
        allowed, _ = strict.check()
        assert allowed is False

    def test_explicit_tau_shifts_policy_looser(self) -> None:
        # Lowering tau below the neutral default should allow a sample
        # that the default would deny at the margin.
        chain = _chain_with_R([0.4])
        loose = CoherenceGatePredicate(
            min_R=0.5,
            chain=chain,
            use_eml_smoothing=True,
            tau=-100.0,  # absurdly loose
        )
        allowed, _ = loose.check()
        assert allowed is True


# ---------------------------------------------------------------------------
# 6. Reason string is informative in both modes.
# ---------------------------------------------------------------------------


class TestReasonStrings:
    def test_reason_informative_hard_cliff_allow(self) -> None:
        chain = _chain_with_R([0.9])
        pred = CoherenceGatePredicate(min_R=0.5, chain=chain)
        _, reason = pred.check()
        assert "0.900" in reason
        assert "0.500" in reason

    def test_reason_informative_hard_cliff_deny(self) -> None:
        chain = _chain_with_R([0.1])
        pred = CoherenceGatePredicate(min_R=0.5, chain=chain)
        _, reason = pred.check()
        assert "0.100" in reason
        assert "0.500" in reason
        assert "window=" in reason

    def test_reason_informative_smooth_allow(self) -> None:
        chain = _chain_with_R([0.9])
        pred = CoherenceGatePredicate(
            min_R=0.5, chain=chain, use_eml_smoothing=True
        )
        _, reason = pred.check()
        assert "eml(R=0.900" in reason
        assert "min_R=0.500" in reason
        assert "tau=" in reason
        assert ">" in reason

    def test_reason_informative_smooth_deny(self) -> None:
        chain = _chain_with_R([0.1])
        pred = CoherenceGatePredicate(
            min_R=0.5, chain=chain, use_eml_smoothing=True
        )
        _, reason = pred.check()
        assert "eml(R=0.100" in reason
        assert "min_R=0.500" in reason
        assert "tau=" in reason
        assert "<=" in reason


# ---------------------------------------------------------------------------
# 7. Backwards-compat — smoothing off == legacy path. Cross-check against
#    a brute-force reimplementation.
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    @pytest.mark.parametrize("R", [0.0, 0.1, 0.3, 0.49, 0.5, 0.51, 0.7, 0.9, 1.0])
    def test_hard_cliff_matches_legacy_predicate(self, R: float) -> None:
        chain = _chain_with_R([R])
        pred = CoherenceGatePredicate(min_R=0.5, chain=chain)
        allowed, _ = pred.check()
        # Legacy rule: allowed iff R >= min_R.
        assert allowed is (R >= 0.5)

    def test_last_snapshot_populated_in_both_modes(self) -> None:
        chain = _chain_with_R([0.7])
        hard = CoherenceGatePredicate(min_R=0.5, chain=chain)
        hard.check()
        assert hard.last_snapshot is not None
        assert hard.last_snapshot.R == pytest.approx(0.7)

        smooth = CoherenceGatePredicate(
            min_R=0.5, chain=chain, use_eml_smoothing=True
        )
        smooth.check()
        assert smooth.last_snapshot is not None
        assert smooth.last_snapshot.R == pytest.approx(0.7)
