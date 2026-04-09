"""CARL Experiment primitive — the canonical discovery-to-proof loop.

An Experiment is the atomic unit of scientific work in CARL:
  hypothesis → config → run → observe → witness → judge → revise

The loop is fractally self-similar:
  - The agent running carl-studio uses this loop to develop carl-studio
  - The model trained by carl-studio uses this loop to solve tasks
  - The researcher using carl-studio uses this loop to validate theory

Every experiment produces verifiable artifacts. The accumulation of
witnessed experiments IS the convergence toward proof.
"""

from carl_studio.experiment.types import (
    Artifact,
    Experiment,
    ExperimentStatus,
    Hypothesis,
    Judgment,
    JudgmentVerdict,
    Prediction,
    Witness,
)

__all__ = [
    "Artifact",
    "Experiment",
    "ExperimentStatus",
    "Hypothesis",
    "Judgment",
    "JudgmentVerdict",
    "Prediction",
    "Witness",
]
