"""Experiment types — the formal structure of the discovery loop.

Design principles (from mempalace + realizability):
  - Store raw, structure later. Artifacts are verbatim. Judgments are derived.
  - Tiered access: hypothesis (always loaded) → config (on demand) → artifacts (deep search)
  - 2D index: experiment phase (hypothesis/running/witnessing/judged) × domain (training/eval/theory)
  - Witnesses are the convergence mechanism. No witness = no claim.
  - The loop is the same at every scale. An Experiment can contain sub-experiments.

The realizability chain (from Semantic Realizability, DOI: 10.5281/zenodo.18992031):
  Hypothesis → Prediction → Witness → Judgment

  A hypothesis is REALIZED when all required predictions have witnesses.
  A hypothesis is REFUTED when any required prediction has a counter-witness.
  A hypothesis is PARTIAL when some predictions are witnessed and none refuted.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Prediction — a testable claim within a hypothesis
# ---------------------------------------------------------------------------

class PredictionComparator(str, Enum):
    """How to compare observed value against expected."""
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    IN_RANGE = "in_range"
    CORRELATION = "correlation"


class Prediction(BaseModel):
    """A single testable prediction.

    Each prediction specifies what to measure, what threshold constitutes
    support, and what would refute it. The witness is the measurement.
    """
    id: str = Field(description="P{N} within the hypothesis")
    claim: str = Field(description="What we predict will happen")
    metric: str = Field(description="What to measure (e.g. 'task_completion_mean')")
    comparator: PredictionComparator
    threshold: float = Field(description="The boundary value")
    threshold_upper: float | None = Field(default=None, description="Upper bound for in_range")
    required: bool = Field(default=True, description="Must this be witnessed for realization?")
    null_hypothesis: str = Field(default="", description="What the null model predicts")


# ---------------------------------------------------------------------------
# Witness — evidence for or against a prediction
# ---------------------------------------------------------------------------

class Witness(BaseModel):
    """Observed evidence attached to a prediction.

    A witness is the measurement. It doesn't interpret — it records.
    The judgment interprets the collection of witnesses.
    """
    prediction_id: str
    observed_value: float
    observed_at: str = Field(description="Step, timestamp, or run ID where observed")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    supports: bool = Field(description="Does this witness support the prediction?")
    detail: dict[str, Any] = Field(default_factory=dict, description="Raw measurement context")


# ---------------------------------------------------------------------------
# Hypothesis — a collection of predictions about an experiment
# ---------------------------------------------------------------------------

class Hypothesis(BaseModel):
    """A pre-registered hypothesis with testable predictions.

    Pre-registration means: write the predictions BEFORE running the experiment.
    This prevents post-hoc rationalization. The predictions are the contract.
    """
    id: str = Field(description="H{N}_{slug}")
    title: str
    observation: str = Field(description="What was observed that motivated this hypothesis")
    statement: str = Field(description="The hypothesis itself")
    predictions: list[Prediction]
    domain: str = Field(default="training", description="training, eval, theory, product")
    pre_registered_at: datetime = Field(default_factory=datetime.utcnow)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Artifact — any file or data produced by the experiment
# ---------------------------------------------------------------------------

class Artifact(BaseModel):
    """A verifiable output from an experiment step."""
    name: str
    path: str = Field(description="File path or URL")
    artifact_type: str = Field(description="checkpoint, log, plot, report, config, dataset")
    produced_at: str = Field(description="Step or phase that produced this")
    checksum: str | None = Field(default=None, description="SHA-256 for reproducibility")


# ---------------------------------------------------------------------------
# Judgment — the verdict on a hypothesis
# ---------------------------------------------------------------------------

class JudgmentVerdict(str, Enum):
    REALIZED = "realized"       # All required predictions witnessed
    PARTIAL = "partial"         # Some witnessed, none refuted
    REFUTED = "refuted"         # At least one required prediction counter-witnessed
    INCONCLUSIVE = "inconclusive"  # Insufficient evidence


class Judgment(BaseModel):
    """The verdict on a hypothesis given accumulated witnesses."""
    verdict: JudgmentVerdict
    witnesses: list[Witness]
    residual_obligations: list[str] = Field(
        default_factory=list,
        description="What remains unwitnessed",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    judged_at: datetime = Field(default_factory=datetime.utcnow)
    notes: str = Field(default="")

    @classmethod
    def from_hypothesis(cls, hypothesis: Hypothesis, witnesses: list[Witness]) -> Judgment:
        """Compute judgment from hypothesis predictions and collected witnesses."""
        witnessed_ids = {w.prediction_id for w in witnesses if w.supports}
        refuted_ids = {w.prediction_id for w in witnesses if not w.supports}
        required_ids = {p.id for p in hypothesis.predictions if p.required}

        residual = [
            p.id for p in hypothesis.predictions
            if p.required and p.id not in witnessed_ids and p.id not in refuted_ids
        ]

        if refuted_ids & required_ids:
            verdict = JudgmentVerdict.REFUTED
        elif required_ids <= witnessed_ids:
            verdict = JudgmentVerdict.REALIZED
        elif witnessed_ids:
            verdict = JudgmentVerdict.PARTIAL
        else:
            verdict = JudgmentVerdict.INCONCLUSIVE

        # Confidence = geometric mean of witness confidences
        confidences = [w.confidence for w in witnesses if w.supports]
        if confidences:
            from math import prod
            confidence = prod(confidences) ** (1.0 / len(confidences))
        else:
            confidence = 0.0

        return cls(
            verdict=verdict,
            witnesses=witnesses,
            residual_obligations=residual,
            confidence=confidence,
            notes=f"{len(witnessed_ids)} witnessed, {len(refuted_ids)} refuted, {len(residual)} remaining",
        )


# ---------------------------------------------------------------------------
# Experiment — the full loop
# ---------------------------------------------------------------------------

class ExperimentStatus(str, Enum):
    PRE_REGISTERED = "pre_registered"   # Hypothesis written, not yet run
    CONFIGURED = "configured"           # Config attached, ready to submit
    RUNNING = "running"                 # Job submitted, collecting data
    WITNESSING = "witnessing"           # Run complete, analyzing artifacts
    JUDGED = "judged"                   # Judgment rendered


class Experiment(BaseModel):
    """The atomic unit of scientific work in CARL.

    An experiment IS the discovery loop:
      pre_registered → configured → running → witnessing → judged

    Each experiment links a hypothesis to a concrete run (training or eval),
    collects artifacts, gathers witnesses, and renders a judgment.

    The accumulation of judged experiments IS the evidence base.
    The collection of realized experiments IS the proof.
    """
    id: str = Field(description="E{N}_{slug}")
    hypothesis: Hypothesis
    status: ExperimentStatus = ExperimentStatus.PRE_REGISTERED

    # Configuration (attached when status → configured)
    config: dict[str, Any] = Field(default_factory=dict, description="TrainingConfig or EvalConfig as dict")
    run_id: str | None = Field(default=None, description="HF Job ID or local run ID")

    # Artifacts (accumulated during running → witnessing)
    artifacts: list[Artifact] = Field(default_factory=list)

    # Witnesses (collected during witnessing)
    witnesses: list[Witness] = Field(default_factory=list)

    # Judgment (rendered when status → judged)
    judgment: Judgment | None = None

    # Provenance
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_experiment: str | None = Field(default=None, description="E{N} this refines or follows")
    tags: list[str] = Field(default_factory=list)

    # Sub-experiments (for nested loops)
    sub_experiments: list[str] = Field(default_factory=list, description="E{N} IDs of child experiments")

    def judge(self) -> Judgment:
        """Render judgment from accumulated witnesses."""
        self.judgment = Judgment.from_hypothesis(self.hypothesis, self.witnesses)
        self.status = ExperimentStatus.JUDGED
        return self.judgment
