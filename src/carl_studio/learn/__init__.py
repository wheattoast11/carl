"""carl_studio.learn -- Knowledge ingestion pipeline.

Reads source material, generates QA pairs, validates quality,
and optionally trains.  No LLM required for generation or validation --
templates + deterministic checks only.

Usage::

    from carl_studio.learn import LearnConfig, LearnPipeline

    config = LearnConfig(source="./docs/", depth="deep")
    result = LearnPipeline(config).run()
"""

from carl_studio.learn.ingest import SourceType, SourceChunk, SourceIngester
from carl_studio.learn.qa_gen import QATier, QAPair, QAGenerator
from carl_studio.learn.quality import QualityResult, QualityGate
from carl_studio.learn.pipeline import LearnConfig, LearnPipeline
from carl_studio.learn.synthesize import SynthesizeConfig, SynthesizePipeline, SynthesizeResult

__all__ = [
    "SourceType",
    "SourceChunk",
    "SourceIngester",
    "QATier",
    "QAPair",
    "QAGenerator",
    "QualityResult",
    "QualityGate",
    "LearnConfig",
    "LearnPipeline",
    "SynthesizeConfig",
    "SynthesizePipeline",
    "SynthesizeResult",
]
