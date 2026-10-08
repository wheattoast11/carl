"""Compare a trained candidate with its bound baseline and declared goal."""

from __future__ import annotations

from carl_studio.types.preparation import (
    EvaluationMeasurement,
    TrainingAcceptance,
    TrainingGoal,
)


def goal_progress(
    goal: TrainingGoal, baseline_value: float, candidate_value: float
) -> tuple[float, bool, bool]:
    """Return directional progress, target satisfaction and improvement."""
    delta = candidate_value - baseline_value
    progress = delta if goal.direction == "higher" else -delta
    target = (
        candidate_value >= goal.threshold
        if goal.direction == "higher"
        else candidate_value <= goal.threshold
    )
    return progress, target, progress > goal.min_delta


def compare_candidate(
    goal: TrainingGoal,
    baseline: EvaluationMeasurement,
    candidate: EvaluationMeasurement,
) -> TrainingAcceptance:
    """Accept only comparable goal progress, policies and measured coherence."""
    reasons: list[str] = []
    comparable = (
        bool(baseline.sample_ids)
        and len(set(baseline.sample_ids)) == len(baseline.sample_ids)
        and baseline.sample_ids == candidate.sample_ids
        and baseline.dataset_sha256 == candidate.dataset_sha256
        and baseline.generation == candidate.generation
        and bool(baseline.generation)
        and baseline.primary_metric == candidate.primary_metric == goal.primary_metric
    )
    if not comparable:
        reasons.append("Evaluation populations or settings differ")
    delta = candidate.primary_value - baseline.primary_value
    _, target, improved = goal_progress(goal, baseline.primary_value, candidate.primary_value)
    if not target:
        reasons.append("The declared task target failed")
    if not improved:
        reasons.append("The candidate did not improve beyond the declared margin")
    missing_policies = [
        policy.id for policy in goal.policies if policy.id not in candidate.policy_results
    ]
    policy_passed = not missing_policies and all(
        candidate.policy_results[policy.id] >= policy.threshold for policy in goal.policies
    )
    if not policy_passed:
        reasons.append("A required policy check failed or is unavailable")
    coherence = candidate.coherence or {}
    has_coherence = candidate.coherence_source == "full_logits" and all(
        metric in coherence for metric in ("phi_mean", "discontinuity_score")
    )
    coherence_passed = has_coherence and (
        coherence["phi_mean"] >= goal.coherence_phi_floor
        and goal.discontinuity_min <= coherence["discontinuity_score"] <= goal.discontinuity_max
    )
    if not coherence_passed:
        reasons.append("The required full-logit coherence check failed or is unavailable")
    if not comparable or not has_coherence or missing_policies:
        status = "inconclusive"
    elif target and improved and policy_passed and coherence_passed:
        status = "accepted"
    else:
        status = "rejected"
    return TrainingAcceptance(
        status=status,
        reasons=reasons or ["Goal progress, policy checks and coherence passed"],
        baseline=baseline,
        candidate=candidate,
        goal_delta=delta,
        policy_passed=policy_passed,
        coherence_passed=coherence_passed,
    )
