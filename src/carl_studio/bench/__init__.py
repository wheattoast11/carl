"""
carl_studio.bench -- CARL Trainability Index (CTI).

Measures how trainable a model is by running 4 coherence probes:

1. PhaseTransitionProbe   -- steps to crystallize a new capability
2. PhiStabilityProbe      -- Phi robustness across domain shift
3. CoherenceUnderPressureProbe -- Phi vs task complexity profile
4. AdaptationRateProbe    -- TTT iterations to 80% success

Usage::

    from carl_studio.bench import BenchConfig, BenchSuite

    config = BenchConfig(model="your-org/your-model")
    suite = BenchSuite(config)
    cti = suite.run()
    print(cti.score, cti.verdict)
"""

from carl_studio.bench.probes import (
    ProbeResult,
    PhaseTransitionProbe,
    PhiStabilityProbe,
    CoherenceUnderPressureProbe,
    AdaptationRateProbe,
)
from carl_studio.bench.suite import BenchConfig, BenchSuite, BenchReport, CTI

__all__ = [
    "BenchConfig",
    "BenchSuite",
    "BenchReport",
    "CTI",
    "ProbeResult",
    "PhaseTransitionProbe",
    "PhiStabilityProbe",
    "CoherenceUnderPressureProbe",
    "AdaptationRateProbe",
]
