"""Tests for ``carl_core.constants``.

Validates the conservation-law constants (KAPPA, SIGMA, T_STAR) and the
empirical Kuramoto-R phase-boundary + weight-profile constants used by
``PhaseAdaptiveCARLReward``. These constants are cited in methods sections,
so the test file exists to catch silent drift.
"""
from __future__ import annotations

from carl_core.constants import (
    DEFECT_THRESHOLD,
    KAPPA,
    KURAMOTO_R_GASEOUS_MAX,
    KURAMOTO_R_LIQUID_MAX,
    PHASE_WEIGHTS_CRYSTALLINE,
    PHASE_WEIGHTS_GASEOUS,
    PHASE_WEIGHTS_LIQUID,
    SIGMA,
    T_STAR,
)


# ---------------------------------------------------------------------------
# Conservation-law constants
# ---------------------------------------------------------------------------


def test_kappa_is_64_over_3():
    """KAPPA = 64/3 per the conservation law paper (Zenodo 18906944)."""
    assert KAPPA == 64 / 3


def test_sigma_is_3_over_16():
    """SIGMA = 3/16 per the conservation law paper (Zenodo 18906944)."""
    assert SIGMA == 3 / 16


def test_defect_threshold_is_finite_positive():
    assert 0 < DEFECT_THRESHOLD < 1


def test_t_star_is_integer_and_scales_linearly():
    assert T_STAR(1) == int(KAPPA)
    assert T_STAR(2) == int(KAPPA * 2)
    assert T_STAR(10) == int(KAPPA * 10)
    assert isinstance(T_STAR(5), int)


# ---------------------------------------------------------------------------
# Kuramoto-R phase boundaries
# ---------------------------------------------------------------------------


def test_phase_boundaries_ordered():
    """Boundaries must be in [0, 1] and strictly ordered gaseous < liquid."""
    assert 0 <= KURAMOTO_R_GASEOUS_MAX < KURAMOTO_R_LIQUID_MAX <= 1


def test_phase_boundary_values_match_doi_methods():
    """Pin the published thresholds so any change is loud (methods-section drift)."""
    assert KURAMOTO_R_GASEOUS_MAX == 0.30
    assert KURAMOTO_R_LIQUID_MAX == 0.70


# ---------------------------------------------------------------------------
# Phase-weight profiles
# ---------------------------------------------------------------------------


def test_phase_weight_profiles_sum_to_one():
    """Each profile must partition reward-weight mass (sum == 1.0)."""
    for profile in (
        PHASE_WEIGHTS_GASEOUS,
        PHASE_WEIGHTS_LIQUID,
        PHASE_WEIGHTS_CRYSTALLINE,
    ):
        assert abs(sum(profile) - 1.0) < 1e-9


def test_phase_weight_profile_shape():
    """Each profile is a 3-tuple of floats in [0, 1]."""
    for profile in (
        PHASE_WEIGHTS_GASEOUS,
        PHASE_WEIGHTS_LIQUID,
        PHASE_WEIGHTS_CRYSTALLINE,
    ):
        assert len(profile) == 3
        assert all(0 <= w <= 1 for w in profile)


def test_phase_weight_profile_values_pinned():
    """Pin the published weight tuples so methods-section citations stay valid."""
    assert PHASE_WEIGHTS_GASEOUS == (0.20, 0.30, 0.50)
    assert PHASE_WEIGHTS_LIQUID == (0.40, 0.30, 0.30)
    assert PHASE_WEIGHTS_CRYSTALLINE == (0.60, 0.30, 0.10)


def test_phase_weight_profiles_encode_intended_ordering():
    """w_multiscale monotonically increases gaseous -> liquid -> crystalline;
    w_discontinuity monotonically decreases. Cloud weight is constant.

    This encodes the published rationale (reward commitment when gaseous,
    reward stability when crystalline).
    """
    w_mc = (PHASE_WEIGHTS_GASEOUS[0], PHASE_WEIGHTS_LIQUID[0], PHASE_WEIGHTS_CRYSTALLINE[0])
    w_disc = (PHASE_WEIGHTS_GASEOUS[2], PHASE_WEIGHTS_LIQUID[2], PHASE_WEIGHTS_CRYSTALLINE[2])
    w_cloud = (PHASE_WEIGHTS_GASEOUS[1], PHASE_WEIGHTS_LIQUID[1], PHASE_WEIGHTS_CRYSTALLINE[1])

    assert w_mc[0] < w_mc[1] < w_mc[2]
    assert w_disc[0] > w_disc[1] > w_disc[2]
    assert w_cloud[0] == w_cloud[1] == w_cloud[2]
