"""Experimental design for CARL research papers.

Formalizes the 4 carl-bench probes as reproducible experiments with
structured result output for paper figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from carl_core.constants import KAPPA, SIGMA


class ExperimentResult(BaseModel):
    """Structured result from a single experiment run."""

    experiment: str
    model: str
    timestamp: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    series: dict[str, list[float]] = Field(default_factory=dict)
    verdict: str = ""
    detail: str = ""

    def save(self, path: Path | str) -> None:
        """Save result as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path | str) -> ExperimentResult:
        """Load result from JSON."""
        return cls.model_validate_json(Path(path).read_text())


class ExperimentSuite:
    """Runs the 4 CARL experiments for paper publication.

    E1: Phase Transition Speed — steps to crystallize a capability
    E2: Phi Stability — order parameter robustness across domain shift
    E3: Coherence Under Pressure — Phi vs task complexity profile
    E4: Adaptation Rate — TTT iterations to accuracy threshold

    Each experiment is designed to be reproducible with fixed seeds.
    """

    def __init__(self, model: str = "", output_dir: str = "experiments/results"):
        self.model = model
        self.output_dir = Path(output_dir)

    def run_all(self, seed: int = 42) -> list[ExperimentResult]:
        """Run all 4 experiments and save results."""
        results = []
        for name, runner in [
            ("E1_phase_transition", self._run_phase_transition),
            ("E2_phi_stability", self._run_phi_stability),
            ("E3_coherence_pressure", self._run_coherence_pressure),
            ("E4_adaptation_rate", self._run_adaptation_rate),
        ]:
            result = runner(seed=seed)
            result.save(self.output_dir / f"{name}.json")
            results.append(result)
        return results

    def _run_phase_transition(self, seed: int = 42) -> ExperimentResult:
        """E1: Measure steps to crystallize a new capability.

        Protocol:
        1. Compute baseline Phi on reference prompts
        2. Fine-tune for N steps on capability dataset
        3. Measure Phi at each step
        4. Find step where Phi crosses 0.5 threshold
        5. Grade: A(<30), B(30-50), C(50-100), D(100-200), F(>200)
        """
        # Without GPU, return design-only result
        return ExperimentResult(
            experiment="E1_phase_transition",
            model=self.model,
            params={
                "seed": seed,
                "protocol": "Fine-tune N steps, track Phi, find crystallization point",
                "threshold": 0.5,
                "grading": {"A": "<30 steps", "B": "30-50", "C": "50-100", "D": "100-200", "F": ">200"},
                "constants": {"kappa": KAPPA, "sigma": SIGMA},
            },
            metrics={},
            series={},
            verdict="pending",
            detail="Requires GPU. Run with: carl bench <model> --suite transition",
        )

    def _run_phi_stability(self, seed: int = 42) -> ExperimentResult:
        """E2: Measure Phi robustness across domain shift.

        Protocol:
        1. Compute Phi on domain A prompts (e.g., code)
        2. Compute Phi on domain B prompts (e.g., natural language)
        3. Measure |delta_phi| between domains
        4. Grade: A(<0.05), B(0.05-0.1), C(0.1-0.2), D(0.2-0.5), F(>0.5)
        """
        return ExperimentResult(
            experiment="E2_phi_stability",
            model=self.model,
            params={
                "seed": seed,
                "protocol": "Compute Phi on two domains, measure delta",
                "domains": ["code", "natural_language"],
                "grading": {"A": "<0.05", "B": "0.05-0.1", "C": "0.1-0.2", "D": "0.2-0.5", "F": ">0.5"},
            },
            metrics={},
            series={},
            verdict="pending",
            detail="Requires GPU. Run with: carl bench <model> --suite stability",
        )

    def _run_coherence_pressure(self, seed: int = 42) -> ExperimentResult:
        """E3: Phi vs task complexity profile.

        Protocol:
        1. Generate prompts at 5 complexity levels
        2. Compute Phi at each level
        3. Classify profile: plateau/graceful/cliff/chaotic/collapse
        4. Grade by profile shape
        """
        return ExperimentResult(
            experiment="E3_coherence_pressure",
            model=self.model,
            params={
                "seed": seed,
                "protocol": "Compute Phi at 5 complexity levels, classify profile",
                "complexity_levels": 5,
                "profiles": ["plateau", "graceful", "cliff", "chaotic", "collapse"],
                "grading": {"A": "plateau", "B": "graceful", "C": "cliff", "D": "chaotic", "F": "collapse"},
            },
            metrics={},
            series={},
            verdict="pending",
            detail="Requires GPU. Run with: carl bench <model> --suite pressure",
        )

    def _run_adaptation_rate(self, seed: int = 42) -> ExperimentResult:
        """E4: TTT iterations to 80% success.

        Protocol:
        1. Present model with novel task
        2. Apply TTT (SLOT/LoRA) iterations
        3. Count iterations to reach 80% accuracy
        4. Grade: A(<5), B(5-10), C(10-20), D(20-50), F(>50)
        """
        return ExperimentResult(
            experiment="E4_adaptation_rate",
            model=self.model,
            params={
                "seed": seed,
                "protocol": "Apply TTT iterations, count to 80% accuracy",
                "target_accuracy": 0.8,
                "ttt_methods": ["slot", "lora"],
                "grading": {"A": "<5", "B": "5-10", "C": "10-20", "D": "20-50", "F": ">50"},
            },
            metrics={},
            series={},
            verdict="pending",
            detail="Requires GPU. Run with: carl bench <model> --suite adaptation",
        )
