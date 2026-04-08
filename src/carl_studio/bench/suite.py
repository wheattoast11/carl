"""
BenchSuite -- orchestrates probe execution and computes the CTI.

The CARL Trainability Index (CTI) is a weighted composite:
    0.30 * transition + 0.25 * stability + 0.25 * pressure + 0.20 * adaptation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from carl_studio.bench.probes import (
    AdaptationRateProbe,
    CoherenceUnderPressureProbe,
    PhaseTransitionProbe,
    PhiStabilityProbe,
    ProbeResult,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_VALID_SUITES = {"all", "transition", "stability", "pressure", "adaptation"}


class BenchConfig(BaseModel):
    """Configuration for a bench run."""

    model: str
    suite: str = "all"
    compare_model: Optional[str] = None
    device: str = "auto"

    @field_validator("suite")
    @classmethod
    def _validate_suite(cls, v: str) -> str:
        if v not in _VALID_SUITES:
            raise ValueError(
                f"suite must be one of {sorted(_VALID_SUITES)}, got {v!r}"
            )
        return v

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model must be a non-empty string")
        return v.strip()


# ---------------------------------------------------------------------------
# CTI
# ---------------------------------------------------------------------------

# Weights for the composite score
_WEIGHTS = {
    "transition": 0.30,
    "stability": 0.25,
    "pressure": 0.25,
    "adaptation": 0.20,
}


def _null_probe(name: str) -> ProbeResult:
    """Return a skipped probe result."""
    return ProbeResult(
        probe_name=name,
        grade="F",
        score=0.0,
        value=0.0,
        detail="Probe skipped (not in selected suite).",
    )


def compute_verdict(score: float) -> str:
    """Map composite score to human-readable verdict."""
    if score >= 0.85:
        return "excellent"
    if score >= 0.65:
        return "good"
    if score >= 0.45:
        return "marginal"
    return "poor"


class CTI(BaseModel):
    """CARL Trainability Index -- composite of 4 probe results."""

    transition: ProbeResult
    stability: ProbeResult
    pressure: ProbeResult
    adaptation: ProbeResult
    score: float
    verdict: str  # "excellent", "good", "marginal", "poor"

    def summary(self) -> str:
        """One-line summary for CLI output."""
        probes = [self.transition, self.stability, self.pressure, self.adaptation]
        grades = " ".join(f"{p.probe_name}={p.grade}" for p in probes)
        return f"CTI={self.score:.2f} ({self.verdict}) | {grades}"


# ---------------------------------------------------------------------------
# BenchReport (optional comparison output)
# ---------------------------------------------------------------------------


class BenchReport(BaseModel):
    """Full bench report, optionally including a comparison model."""

    primary: CTI
    comparison: Optional[CTI] = None
    delta: Optional[float] = None  # primary.score - comparison.score

    def summary(self) -> str:
        lines = [f"Primary:    {self.primary.summary()}"]
        if self.comparison is not None:
            lines.append(f"Comparison: {self.comparison.summary()}")
            lines.append(f"Delta:      {self.delta:+.2f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BenchSuite
# ---------------------------------------------------------------------------


class BenchSuite:
    """
    Orchestrates probe execution.

    Loads the model lazily on first run, runs selected probes, computes CTI.
    """

    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None

    def _load_model(self, model_name: str) -> tuple:
        """
        Load model and tokenizer from HuggingFace.

        Returns (model, tokenizer). Raises ImportError if transformers
        is not available, RuntimeError on load failure.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.config.device
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            model = model.float()
        model.eval()

        return model, tokenizer

    def _run_probes(
        self, model: Any, tokenizer: Any
    ) -> Dict[str, ProbeResult]:
        """Run selected probes and return results keyed by probe name."""
        suite = self.config.suite
        device = self.config.device
        results: Dict[str, ProbeResult] = {}

        if suite in ("all", "transition"):
            results["transition"] = PhaseTransitionProbe().run(
                model, tokenizer, device=device
            )
        else:
            results["transition"] = _null_probe("phase_transition")

        if suite in ("all", "stability"):
            results["stability"] = PhiStabilityProbe().run(
                model, tokenizer, device=device
            )
        else:
            results["stability"] = _null_probe("phi_stability")

        if suite in ("all", "pressure"):
            results["pressure"] = CoherenceUnderPressureProbe().run(
                model, tokenizer, device=device
            )
        else:
            results["pressure"] = _null_probe("coherence_pressure")

        if suite in ("all", "adaptation"):
            results["adaptation"] = AdaptationRateProbe().run(
                model, tokenizer, device=device
            )
        else:
            results["adaptation"] = _null_probe("adaptation_rate")

        return results

    def _build_cti(self, results: Dict[str, ProbeResult]) -> CTI:
        """Compute the composite CTI from probe results."""
        score = sum(
            _WEIGHTS[key] * results[key].score for key in _WEIGHTS
        )
        return CTI(
            transition=results["transition"],
            stability=results["stability"],
            pressure=results["pressure"],
            adaptation=results["adaptation"],
            score=round(score, 4),
            verdict=compute_verdict(score),
        )

    def run(self) -> CTI:
        """
        Run the bench suite and return the CTI.

        Loads model, runs probes, computes composite.
        If model loading fails (no GPU, missing deps), all probes get F.
        """
        try:
            model, tokenizer = self._load_model(self.config.model)
        except Exception as e:
            # Graceful degradation: return F grades with explanation
            error_detail = f"Model load failed: {e}"
            fail = lambda name: ProbeResult(
                probe_name=name,
                grade="F",
                score=0.0,
                value=0.0,
                detail=error_detail,
            )
            return CTI(
                transition=fail("phase_transition"),
                stability=fail("phi_stability"),
                pressure=fail("coherence_pressure"),
                adaptation=fail("adaptation_rate"),
                score=0.0,
                verdict="poor",
            )

        results = self._run_probes(model, tokenizer)
        return self._build_cti(results)

    def run_comparison(self) -> BenchReport:
        """
        Run bench on primary model and optionally a comparison model.

        Returns a BenchReport with both CTIs and their delta.
        """
        primary_cti = self.run()

        if self.config.compare_model is None:
            return BenchReport(primary=primary_cti)

        # Run comparison model
        try:
            comp_model, comp_tokenizer = self._load_model(self.config.compare_model)
            comp_results = self._run_probes(comp_model, comp_tokenizer)
            comp_cti = self._build_cti(comp_results)
        except Exception as e:
            error_detail = f"Comparison model load failed: {e}"
            fail = lambda name: ProbeResult(
                probe_name=name,
                grade="F",
                score=0.0,
                value=0.0,
                detail=error_detail,
            )
            comp_cti = CTI(
                transition=fail("phase_transition"),
                stability=fail("phi_stability"),
                pressure=fail("coherence_pressure"),
                adaptation=fail("adaptation_rate"),
                score=0.0,
                verdict="poor",
            )

        delta = round(primary_cti.score - comp_cti.score, 4)
        return BenchReport(primary=primary_cti, comparison=comp_cti, delta=delta)
