"""carl_studio.align -- Targeted model realignment for three kinds of drift.

Three modes:
  PATTERNS  -- API migration, library changes (contrastive SFT)
  TEMPORAL  -- World knowledge refresh (calibration SFT)
  FORMAT    -- Output format correction (contrastive GRPO)

Usage::

    from carl_studio.align import AlignConfig, AlignMode, AlignPipeline

    config = AlignConfig(
        mode=AlignMode.PATTERNS,
        model="wheattoast11/OmniCoder-9B-Zero-Phase2",
        source="./migration-notes.md",
    )
    result = AlignPipeline(config).run()
"""

from carl_studio.align.pipeline import (
    AlignConfig,
    AlignMode,
    AlignPipeline,
    DataGenerator,
    DataPair,
)

__all__ = [
    "AlignConfig",
    "AlignMode",
    "AlignPipeline",
    "DataGenerator",
    "DataPair",
]
